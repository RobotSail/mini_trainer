"""
Optimizer utilities for mini_trainer.

Supports:
- AdamW (default, from torch.optim)
- Muon (requires PyTorch >= 2.9 or standalone muon package)
- Muon with FSDP2 support (requires muon-fsdp2 package)

This module provides optimizer factory functions that handle the complexity of
setting up Muon with proper parameter groups (2D+ hidden weights for Muon,
embeddings/heads/1D params for AdamW).
"""

import torch
from torch.optim import AdamW
from typing import Literal
from transformers import PreTrainedModel

from mini_trainer.utils import log_rank_0

OptimizerType = Literal["adamw", "muon"]


def get_muon_param_groups(
    model: torch.nn.Module,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
) -> tuple[list, list]:
    """
    Create parameter groups for Muon optimizer.

    Muon applies to 2D+ hidden layer weights, while AdamW handles:
    - Embedding layers
    - Output/classifier heads
    - 1D parameters (biases, layernorms)

    Args:
        model: The model to create parameter groups for
        muon_lr: Learning rate for Muon parameters
        adamw_lr: Learning rate for AdamW parameters
        weight_decay: Weight decay for all parameters

    Returns:
        Tuple of (muon_params, adamw_params) - lists of parameters
    """
    muon_params = []
    adamw_params = []

    # Get embedding and lm_head parameter names
    embed_and_head_names = set()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_and_head_names.add("model.embed_tokens")
    if hasattr(model, "lm_head"):
        embed_and_head_names.add("lm_head")
    # Also check for common embedding names across different architectures
    embed_and_head_names.update(
        ["embed_tokens", "wte", "wpe", "lm_head", "embed_out", "embed_positions"]
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is an embedding or output head
        is_embed_or_head = any(n in name for n in embed_and_head_names)

        # Muon for 2D+ hidden weights, AdamW for everything else
        if param.ndim >= 2 and not is_embed_or_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def create_fsdp2_muon_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-5,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    rms_scale: bool = True,
) -> torch.optim.Optimizer:
    """
    Create Muon optimizer compatible with FSDP2 using muon-fsdp2 package.

    This optimizer handles DTensor parameters correctly with gather/scatter operations
    for the Newton-Schulz orthogonalization.

    Args:
        model: The FSDP2-wrapped model to optimize
        muon_lr: Learning rate for Muon parameters (2D+ hidden weights)
        adamw_lr: Learning rate for Adam parameters (embeddings, heads, 1D params)
        beta1, beta2: Adam betas for non-Muon parameters
        eps: Epsilon for numerical stability in Adam
        weight_decay: Weight decay
        momentum: Muon momentum
        nesterov: Use Nesterov momentum
        ns_steps: Newton-Schulz iteration steps
        rms_scale: Scale gradients by RMS (Moonlight paper style)

    Returns:
        muon_fsdp2.Muon optimizer

    Raises:
        ImportError: If muon-fsdp2 package is not installed
    """
    try:
        from muon_fsdp2 import Muon
    except ImportError:
        raise ImportError(
            "FSDP2 Muon requires the 'muon-fsdp2' package. "
            "Install with: pip install muon-fsdp2"
        )

    muon_params, adamw_params = get_muon_param_groups(
        model, muon_lr, adamw_lr, weight_decay
    )

    log_rank_0(
        f"📊 Muon optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params"
    )

    param_groups = []

    if muon_params:
        param_groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scale=rms_scale,
            )
        )

    if adamw_params:
        param_groups.append(
            dict(
                params=adamw_params,
                use_muon=False,
                lr=adamw_lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )
        )

    return Muon(param_groups)


class CombinedOptimizer:
    """
    Wrapper that combines two optimizers (e.g., Muon + AdamW).
    Allows using Muon for hidden layers and AdamW for embeddings/heads.
    """

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    muon_lr: float | None = None,
) -> torch.optim.Optimizer:
    """
    Create optimizer based on type.

    Args:
        model: The model to optimize
        optimizer_type: "adamw" or "muon"
        lr: Learning rate (used for AdamW params, and as default for Muon if muon_lr not specified)
        beta1, beta2: Adam betas
        eps: Epsilon for numerical stability
        weight_decay: Weight decay
        muon_lr: Learning rate for Muon parameters (default: same as lr)

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer_type is not recognized
        ImportError: If Muon is requested but not available
    """
    optimizer_type = optimizer_type.lower()

    # Default muon_lr to lr if not specified
    if muon_lr is None:
        muon_lr = lr

    # Filter to only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adamw":
        log_rank_0(f"📊 Creating AdamW optimizer with lr={lr}")
        return AdamW(
            trainable_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "muon":
        # For FSDP2-wrapped models, use the muon-fsdp2 package
        # which handles DTensor parameters correctly
        try:
            return create_fsdp2_muon_optimizer(
                model=model,
                muon_lr=muon_lr,
                adamw_lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
            )
        except ImportError:
            # Fall back to trying native PyTorch Muon or standalone muon package
            pass

        # Try PyTorch 2.9+ native Muon
        try:
            from torch.optim import Muon

            has_native_muon = True
        except ImportError:
            has_native_muon = False

        if has_native_muon:
            muon_params, adamw_params = get_muon_param_groups(
                model, muon_lr, lr, weight_decay
            )

            log_rank_0(
                f"📊 Native Muon optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params"
            )

            if muon_params and adamw_params:
                # Use Muon for hidden weights, AdamW for embeddings/heads
                param_groups = [
                    {
                        "params": muon_params,
                        "lr": muon_lr,
                        "weight_decay": weight_decay,
                    },
                ]
                muon_opt = Muon(
                    param_groups,
                    lr=muon_lr,
                    momentum=0.95,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
                adamw_opt = AdamW(
                    adamw_params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay,
                )
                return CombinedOptimizer(muon_opt, adamw_opt)
            elif muon_params:
                return Muon(
                    [{"params": muon_params, "lr": muon_lr}],
                    momentum=0.95,
                    adjust_lr_fn="match_rms_adamw",
                    weight_decay=weight_decay,
                )
            else:
                # No Muon-eligible params, fall back to AdamW
                log_rank_0("⚠️ No Muon-eligible parameters found, using AdamW")
                return AdamW(
                    trainable_params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay,
                )

        else:
            # Fallback: try standalone muon package
            try:
                from muon import MuonWithAuxAdam
            except ImportError:
                raise ImportError(
                    "Muon optimizer requires one of:\n"
                    "  1. muon-fsdp2 package (for FSDP2 support): pip install muon-fsdp2\n"
                    "  2. PyTorch >= 2.9 (native Muon)\n"
                    "  3. muon package (standalone): pip install muon"
                )

            muon_params, adamw_params = get_muon_param_groups(
                model, muon_lr, lr, weight_decay
            )

            log_rank_0(
                f"📊 Standalone Muon optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params"
            )

            param_groups = []
            if muon_params:
                param_groups.append(
                    dict(
                        params=muon_params,
                        use_muon=True,
                        lr=muon_lr,
                        weight_decay=weight_decay,
                    )
                )
            if adamw_params:
                param_groups.append(
                    dict(
                        params=adamw_params,
                        use_muon=False,
                        lr=lr,
                        betas=(beta1, beta2),
                        eps=eps,
                        weight_decay=weight_decay,
                    )
                )

            return MuonWithAuxAdam(param_groups)

    else:
        raise ValueError(
            f"Unknown optimizer type: '{optimizer_type}'. Choose 'adamw' or 'muon'."
        )
