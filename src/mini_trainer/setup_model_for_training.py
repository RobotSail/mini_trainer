import math
import os
import gc
import inspect
from typing import Optional, Dict, Any
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from transformers import AutoTokenizer, AutoConfig, Mxfp4Config
from mini_trainer.utils import (
    get_model_class_from_config,
    log_rank_0,
    patch_target_module,
)
from mini_trainer.osft_utils import (
    OSFTModel,
    _build_osft_kwargs,
    create_osft_model_class,
    _set_osft_dtypes,
)
from mini_trainer.gpt_oss_utils import freeze_router_params, is_gpt_oss_model
import mini_trainer.osft_utils as osft_utils
from mini_trainer.fsdp2_lazy_init import (
    FSDP2_LAZY_INIT_SFT,
    FSDP2_LAZY_INIT_OSFT,
    get_fsdp2_lazy_init_mode,
    set_fsdp2_lazy_init_mode,
)


def _distributed_initialized() -> bool:
    """
    Returns True when torch.distributed is both available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


def _require_distributed_initialized(action: str) -> None:
    """
    Raises a RuntimeError with a helpful message when an action requires the
    torch.distributed process group but it has not been initialized.
    """
    if not _distributed_initialized():
        raise RuntimeError(
            f"{action} requires torch.distributed to be initialized. "
            "Call torch.distributed.init_process_group() first."
        )


def _apply_liger_kernels_if_requested(use_liger_kernels, model_config, base_model_args):
    """
    Mirror AutoLiger’s monkey-patching behavior before OSFT creates its wrapper class.
    Mutates base_model_args by removing kwargs consumed by the Liger patcher.
    """
    if not use_liger_kernels:
        return

    try:
        from liger_kernel.transformers.monkey_patch import (
            MODEL_TYPE_TO_APPLY_LIGER_FN,
            _apply_liger_kernel,
        )
    except ImportError as e:
        raise ImportError(
            "Tried to use liger kernels for OSFT, but they are not installed. "
            "Please install the CUDA dependencies or disable Liger kernels."
        ) from e

    model_type = getattr(model_config, "model_type", None)
    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if apply_fn is None:
        raise ValueError(f"Liger kernels do not support model type '{model_type}'.")

    apply_signature = inspect.signature(apply_fn)
    liger_kwargs = {}
    for key in list(base_model_args.keys()):
        if key in apply_signature.parameters:
            liger_kwargs[key] = base_model_args.pop(key)

    _apply_liger_kernel(model_type, **liger_kwargs)


@dataclass
class ModelInitializationContext:
    """
    Context object that holds state for model initialization across the three-phase pipeline.

    Attributes:
        is_sft: Whether this is an SFT model
        is_osft: Whether this is an OSFT model
        state_dict: State dict from rank 0 (None on other ranks)
        train_dtype: Training dtype for SFT models
    """

    is_sft: bool = False
    is_osft: bool = False
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    train_dtype: Optional[torch.dtype] = None


def _sanitize_meta_attribute_aliases(model: torch.nn.Module) -> int:
    """Repairs non-param/buffer tensor attributes generically.

    Rules (simple and model-agnostic):
    - If an attribute tensor is on meta and not OSFT-owned, clone the ONLY module-local
      param/buffer with identical shape and dtype. If there is not exactly one match, skip.
    - If an attribute tensor is on CPU and the module has a non-CPU param/buffer device,
      move the attribute to that device. Otherwise keep as CPU.

    Returns the number of attributes repaired or moved.
    """
    repaired = 0

    # best-effort local target device per rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    default_device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def _is_osft_owned_attribute(module: torch.nn.Module, name: str) -> bool:
        if name.startswith("osft_") or name in {"U_low", "S_low", "V_low", "rank_high"}:
            return True
        return hasattr(module, "osft_params") and name in {
            "U_low",
            "S_low",
            "V_low",
            "rank_high",
        }

    for module in model.modules():
        # collect available candidates from this module
        buf_map = dict(module._buffers) if hasattr(module, "_buffers") else {}
        param_map = dict(module._parameters) if hasattr(module, "_parameters") else {}

        # helper to find a unique same-shape/dtype candidate among module-local param/buffer tensors
        def _unique_match_by_shape_dtype(target: torch.Tensor) -> torch.Tensor | None:
            matches = [
                t
                for t in list(buf_map.values()) + list(param_map.values())
                if isinstance(t, torch.Tensor)
                and t.device.type != "meta"
                and t.shape == target.shape
                and t.dtype == target.dtype
            ]
            if len(matches) == 1:
                return matches[0]
            return None

        # derive a reasonable target device from module-local params/buffers
        target_device = None
        for t in list(param_map.values()) + list(buf_map.values()):
            if isinstance(t, torch.Tensor) and t.device.type != "meta":
                target_device = t.device
                break
        if target_device is None:
            target_device = default_device

        # iterate module attributes (avoid dir(); use __dict__ to skip methods)
        for attr_name, value in list(getattr(module, "__dict__", {}).items()):
            if not isinstance(value, torch.Tensor):
                continue
            # skip real params/buffers
            if attr_name in buf_map or attr_name in param_map:
                continue
            # skip OSFT-owned attributes
            if _is_osft_owned_attribute(module, attr_name):
                continue

            # meta → clone from a unique module-local shape/dtype match
            if value.device.type == "meta":
                candidate = _unique_match_by_shape_dtype(value)

                if candidate is None:
                    # no safe materialization path; leave untouched
                    continue

                try:
                    # check if model has expected dtype (e.g., from OSFT)
                    expected_dtype = None
                    if hasattr(model, "output_dtype"):
                        expected_dtype = model.output_dtype
                    elif hasattr(model, "dtype"):
                        expected_dtype = model.dtype

                    fixed = candidate.detach().clone()
                    if expected_dtype and fixed.dtype != expected_dtype:
                        fixed = fixed.to(dtype=expected_dtype)

                    module.__dict__[attr_name] = fixed
                    repaired += 1
                except Exception:
                    pass
                continue

            # CPU → move to module-local device (only for non-param/buffer attributes)
            if value.device.type == "cpu" and target_device.type != "cpu":
                try:
                    module.__dict__[attr_name] = value.to(
                        device=target_device, dtype=value.dtype
                    )
                    repaired += 1
                except Exception:
                    # leave as-is if movement is unsafe
                    pass

    return repaired


def _get_module_by_name(
    model: torch.nn.Module, name: str
) -> tuple[Optional[torch.nn.Module], Optional[str]]:
    """
    Helper to traverse and retrieve a module and its attribute by name.

    Args:
        model: Root model to search from
        name: Dotted name of the buffer/parameter (e.g., "model.layers.0.self_attn.rotary_emb.inv_freq")

    Returns:
        Tuple of (module, attribute_name) or (None, None) if not found
    """
    parts = name.split(".")
    attr = parts[-1]
    mod = model
    for p in parts[:-1]:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        elif p.isdigit():
            mod = mod[int(p)]
        else:
            return None, None
    return mod, attr


def _materialize_meta_buffers(
    model: torch.nn.Module,
    buffer_dict: Dict[str, torch.Tensor],
    expected_dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Materialize buffers from CPU/other device to the model, replacing meta device buffers.

    This function is shared by both SFT and OSFT initialization paths. It handles the case
    where some models (like Phi3) have buffers that are not registered but stored as direct
    attributes.

    Args:
        model: Model with meta device buffers to materialize
        buffer_dict: Dictionary mapping buffer names to their data
        expected_dtype: Optional dtype to convert buffers to (if None, uses buffer's existing dtype)

    Returns:
        Number of buffers materialized
    """
    if not buffer_dict:
        return 0

    log_rank_0(f"🔧 Materializing {len(buffer_dict)} buffers before FSDP2 wrapping")
    materialized = 0

    for buf_name, buf_data in buffer_dict.items():
        mod, attr = _get_module_by_name(model, buf_name)
        if mod is not None:
            # Verify current buffer is on meta device
            curr_buff = getattr(mod, attr, None)
            if curr_buff is not None and curr_buff.device.type == "meta":
                # Determine target dtype
                target_dtype = (
                    expected_dtype if expected_dtype is not None else curr_buff.dtype
                )

                # Convert dtype if needed
                if buf_data.dtype != target_dtype:
                    buf_data = buf_data.to(dtype=target_dtype)

                # Clone the buffer data and register it
                new_data = buf_data.detach().clone()
                mod.register_buffer(attr, new_data, persistent=True)
                materialized += 1

    log_rank_0(f"✅ Materialized {materialized} buffers successfully")
    return materialized


# ==============================================================================
# Generic distributed model loading abstractions for SFT/OSFT integration
# ==============================================================================


def _synchronize_state_dict_fsdp2(
    model,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
):
    """
    Generic state dict synchronization for FSDP2-wrapped models.

    Broadcasts state dict from rank 0 to all other ranks after FSDP2 sharding.

    Args:
        model: FSDP2-wrapped model
        state_dict: Full state dict (only populated on rank 0, None/empty on others)
        strict: Whether to enforce strict state dict loading
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "_synchronize_state_dict_fsdp2 requires torch.distributed to be initialized"
        )

    # prepare state dict for rank 0
    final_state_dict = {}
    if dist.get_rank() == 0:
        if state_dict is None or len(state_dict) == 0:
            raise ValueError("Rank 0 must provide a non-empty state dict")
        final_state_dict = state_dict

    # broadcast to all ranks
    log_rank_0("📤 Broadcasting state dict to all ranks...")
    set_model_state_dict(
        model=model,
        model_state_dict=final_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=strict,
        ),
    )
    log_rank_0("✅ State dict synchronized across all ranks")


def prepare_model_for_fsdp2(model: torch.nn.Module) -> ModelInitializationContext:
    """
    Phase 1: Prepare model for FSDP2 wrapping by handling lazy initialization.

    This function:
    - Detects whether the model is using SFT or OSFT lazy initialization
    - Extracts state dicts from the model (rank 0 only)
    - Materializes buffers from meta device
    - Returns context for later phases

    Args:
        model: Model to prepare (may have lazy init flags set)

    Returns:
        ModelInitializationContext with state dict and metadata for later phases
    """
    context = ModelInitializationContext()

    init_mode = get_fsdp2_lazy_init_mode(model)

    if init_mode == FSDP2_LAZY_INIT_SFT:
        _require_distributed_initialized("SFT lazy initialization")
        context.is_sft = True
        log_rank_0("🔧 [Phase 1] Detected SFT lazy initialization")

        # Extract state dict from rank 0
        context.state_dict = getattr(model, "_fsdp2_pending_state_dict", None)
        if dist.get_rank() == 0:
            if context.state_dict is None or len(context.state_dict) == 0:
                raise RuntimeError(
                    "Rank 0 must have a non-empty state dict for SFT lazy init"
                )
            log_rank_0("📦 [SFT] Rank 0 has state dict for lazy init")
        else:
            if context.state_dict is not None and len(context.state_dict) > 0:
                raise RuntimeError(
                    "Non-rank 0 should not have state dict for SFT lazy init"
                )

        # Extract training dtype and buffer dict
        context.train_dtype = getattr(model, "_fsdp2_train_dtype", None)
        buffer_dict = getattr(model, "_fsdp2_pending_buffers", None)

        # Materialize buffers before FSDP2 wrapping
        if buffer_dict:
            _materialize_meta_buffers(model, buffer_dict, expected_dtype=None)

        # Clean up temporary attributes
        model._fsdp2_pending_state_dict = None
        model._fsdp2_pending_buffers = None

        log_rank_0("✅ [Phase 1] SFT preparation complete")
        return context

    if init_mode == FSDP2_LAZY_INIT_OSFT:
        _require_distributed_initialized("OSFT lazy initialization")
        context.is_osft = True
        log_rank_0("🔧 [Phase 1] Detected OSFT lazy initialization")

        # Eject the original state dict (only rank 0 has it populated)
        context.state_dict = model.eject_og_state_dict()
        if dist.get_rank() == 0:
            if context.state_dict is None or len(context.state_dict) == 0:
                raise RuntimeError(
                    "Rank 0 must have a non-empty state dict for OSFT lazy init"
                )
            log_rank_0("📦 [OSFT] Rank 0 has original state dict")
        else:
            if context.state_dict is not None and len(context.state_dict) > 0:
                raise RuntimeError(
                    "Non-rank 0 should not have state dict for OSFT lazy init"
                )

        log_rank_0("✅ [Phase 1] OSFT preparation complete")
        return context

    # No lazy initialization detected
    log_rank_0(
        "ℹ️  [Phase 1] No lazy initialization detected, proceeding with standard path"
    )
    return context


def wrap_fsdp2(model: torch.nn.Module) -> torch.nn.Module:
    """
    Phase 2: Pure FSDP2 wrapping with activation checkpointing.

    This function only handles FSDP2 wrapping and does NOT handle state dict
    distribution or lazy initialization. Those are handled in Phase 1 (prepare_model_for_fsdp2)
    and Phase 3 (finalize_model_initialization).

    This mirrors TorchTitan's approach: checkpoint each block, then shard each block and the full model.

    Args:
        model: Model to wrap with FSDP2 (should already have buffers materialized)

    Returns:
        FSDP2-wrapped model
    """
    log_rank_0("🔄 [Phase 2] Starting FSDP2 wrapping")

    # Configure mixed precision policy (bfloat16 for Flash Attention compatibility)
    # TODO: make these settings configurable so that non-FA users can leverage FP32 param storage
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Disable HuggingFace cache if present
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception as e:
            print(
                f"WARNING: Failed to disable HuggingFace cache for model {model.__class__.__name__}: {e}"
            )

    # Find the transformer block container
    # Support common architectures: Llama (model.layers), GPT-2 (transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(
            "Cannot find transformer block container on model. "
            "This likely means we need to update the code to support this model."
        )

    # Apply activation checkpointing to each block
    log_rank_0(
        f"🔄 [Phase 2] Applying activation checkpointing to {len(layers)} blocks"
    )
    for idx, block in enumerate(layers):
        # preserve_rng_state needs to be true so that the backward pass can be accurate
        layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=True)

    # Build 1D device mesh over all ranks
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", [world_size], mesh_dim_names=["fsdp"])

    # FSDP2 wrap each transformer block
    log_rank_0("🔄 [Phase 2] Wrapping transformer blocks with FSDP2")
    for idx, block in enumerate(layers):
        reshard = idx < len(layers) - 1
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard,
        )
    log_rank_0(f"   ✓ Wrapped {len(layers)} blocks with FSDP2")

    # FSDP2 wrap the full model
    log_rank_0("🔄 [Phase 2] Wrapping full model with FSDP2")
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    log_rank_0("   ✓ Full model wrapped with FSDP2")

    log_rank_0("✅ [Phase 2] FSDP2 wrapping complete")
    return model


def finalize_model_initialization(
    model: torch.nn.Module, context: ModelInitializationContext
) -> torch.nn.Module:
    """
    Phase 3: Finalize model initialization by distributing weights.

    This function handles:
    - SFT: Distribute state dict from rank 0 to all ranks
    - OSFT: Distribute non-OSFT params, compute distributed SVD, distribute OSFT params
    - Both: Sanitize meta tensor attributes

    Args:
        model: FSDP2-wrapped model
        context: Context from prepare_model_for_fsdp2() containing state dicts

    Returns:
        Fully initialized model ready for training
    """
    # Handle SFT finalization
    if context.is_sft:
        _require_distributed_initialized("SFT finalization")
        log_rank_0("🔄 [Phase 3] Finalizing SFT initialization")

        # Convert dtypes on rank 0 before broadcasting
        # (set_model_state_dict doesn't auto-cast like load_state_dict)
        if dist.get_rank() == 0 and context.state_dict:
            expected_dtype = (
                context.train_dtype if context.train_dtype else model.config.torch_dtype
            )
            converted_state_dict = {}
            conversions = 0

            for key, value in context.state_dict.items():
                if isinstance(value, torch.Tensor) and value.dtype != expected_dtype:
                    converted_state_dict[key] = value.to(dtype=expected_dtype)
                    conversions += 1
                else:
                    converted_state_dict[key] = value

            if conversions > 0:
                log_rank_0(
                    f"🔧 [SFT] Converted {conversions} parameters to {expected_dtype}"
                )
            context.state_dict = converted_state_dict

        # Broadcast state dict to all ranks
        _synchronize_state_dict_fsdp2(
            model=model,
            state_dict=context.state_dict if dist.get_rank() == 0 else {},
            strict=False,  # Use strict=False since buffers are handled separately
        )

        # Next, we need to delete the state dict
        sd = context.state_dict
        del sd
        context.state_dict = None
        # Ensures no residual data is still being allocated by Pytorch
        torch.cuda.empty_cache()
        gc.collect()

        log_rank_0("✅ [SFT] State dict distributed successfully")
        log_rank_0("✅ [Phase 3] SFT finalization complete")

    # Handle OSFT finalization
    elif context.is_osft:
        _require_distributed_initialized("OSFT finalization")
        log_rank_0("🔄 [Phase 3] Finalizing OSFT initialization")

        # Step 1: Synchronize non-OSFT parameters across ranks
        log_rank_0("🔄 [OSFT] Distributing non-OSFT parameters")
        model.post_fsdp2_wrap_synchronize_state_dict_across_procs(
            model, context.state_dict
        )
        log_rank_0("   ✓ Non-OSFT parameters distributed")

        # Step 2: Compute distributed SVD and distribute OSFT parameters
        log_rank_0("🔄 [OSFT] Computing distributed SVD for OSFT parameters")
        model.compute_distributed_svd(model, context.state_dict)
        log_rank_0("   ✓ OSFT parameters computed and distributed")

        # Mark OSFT initialization as complete
        model.mark_fsdp2_initialized()
        log_rank_0("✅ [Phase 3] OSFT finalization complete")
    else:
        if _distributed_initialized():
            raise ValueError("invalid model type, expected SFT or OSFT model")
        log_rank_0(
            "ℹ️  [Phase 3] Non-distributed initialization detected, skipping distributed finalization logic"
        )

    if context.is_sft:
        set_fsdp2_lazy_init_mode(model, None)

    # Sanitize meta tensor attributes (common to all paths)
    fixed_generic = _sanitize_meta_attribute_aliases(model)
    if fixed_generic:
        path_type = (
            "SFT" if context.is_sft else ("OSFT" if context.is_osft else "Standard")
        )
        log_rank_0(f"🧩 [{path_type}] Sanitized {fixed_generic} meta tensor attributes")

    return model


def align_model_and_tokenizer(model, tokenizer):
    """
    Aligns the model's vocabulary and special tokens with the tokenizer.
    """
    if len(tokenizer) > model.config.vocab_size:
        print(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Step 1: Ensure tokenizer has a pad_token_id (required for training)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            log_rank_0(
                "\033[38;5;226m"
                f"⚠️  Tokenizer missing pad_token_id, setting to eos_token_id ({tokenizer.eos_token_id})"
                "\033[0m"
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id. "
                "Cannot proceed with training - please configure the tokenizer properly."
            )

    # Step 2: Sync all special tokens from tokenizer to model.config
    # This ensures model.config always reflects tokenizer's special tokens
    special_tokens = {
        "pad": ("pad_token_id", "Syncing model pad token id"),
        "bos": ("bos_token_id", "Syncing model bos token id"),
        "eos": ("eos_token_id", "Syncing model eos token id"),
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)

        # Always sync tokenizer -> model.config when tokenizer has a valid value
        if tokenizer_token is not None and model_token != tokenizer_token:
            log_rank_0(f"{message}: {model_token} -> {tokenizer_token}")
            setattr(model.config, token_attr, tokenizer_token)

    return model


def get_model_save_dtype(
    save_dtype: str | torch.dtype | None, model_name_or_path: str
) -> torch.dtype:
    """
    Given an HF model reference and an optional user-provided save_dtype, returns the PyTorch data type that it should
    be saved in.

    If the user does not provide a save_dtype, we will use the model's original dtype.
    However; if the data-type is not in the supported list, we will raise an error.

    If both the model `torch_dtype` and user-provided `save_dtype` are missing,
    we default to saving in BF16.

    Args:
        save_dtype (str | None): The dtype we should be saving the model as.
        model_name_or_path (str): The name or path of the model to load.
    Returns:
        The PyTorch data type that the model should be saved in.

    """
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    default_dtype = torch.bfloat16

    # FSDP2 requires us to load the model in FP32 to begin with for the
    # correct mixed-precision settings. So to circumvent this, we load the
    # original model's config separately
    original_config = AutoConfig.from_pretrained(model_name_or_path)
    original_dtype = getattr(original_config, "torch_dtype", None)

    # HF models return a torch.dtype from this field, but docs mark it as an optional string
    if original_dtype is not None and isinstance(original_dtype, str):
        original_dtype = dtype_map[original_dtype]

    # this handles the case when save_dtype > original_dtype > bf16
    if not original_dtype and not save_dtype:
        log_rank_0(
            f"⚠️ Model does not have a setting for `torch_dtype` and not `save_dtype` was provided, falling back to '{default_dtype}'"
        )
        return default_dtype

    # handles the case save_dtype > original_dtype
    if not save_dtype:
        return original_dtype

    # by now we know that we are going to use a custom data type, so we just validate
    if not isinstance(save_dtype, (str, torch.dtype)):
        raise ValueError(
            f"error: could not recognize '{save_dtype}' as a supported dtype for saving model checkpoints"
        )

    # convert dtype to a str
    if isinstance(save_dtype, str):
        if save_dtype not in dtype_map:
            raise ValueError(
                f"error: could not recognize '{save_dtype}' as a supported dtype for saving model checkpoints"
            )
        save_dtype = dtype_map[save_dtype]

    # alert the user when the dtype differs
    if original_dtype and original_dtype != save_dtype:
        log_rank_0(
            f"⚠️ Model's original dtype is '{original_dtype}', but new checkpoints will be saved as '{save_dtype}'. ⚠️"
        )
    return save_dtype


def setup_osft_model_distributed(
    model_name_or_path: str,
    base_model_args: dict,
    tokenizer,
    rank: int,
    osft_rank_ratio=None,
    osft_target_patterns=None,
    osft_upcast_dtype=torch.float32,
    osft_output_dtype=None,
):
    """
    Initialize an OSFT model for distributed training with memory-efficient loading.

    This function uses the FSDP2 lazy initialization path where:
    - Rank 0 loads the full model to CPU
    - All other ranks create meta device models
    - State dict is broadcast after FSDP2 sharding

    This requires torch.distributed to be initialized.

    Args:
        model_name_or_path: HuggingFace model name or path
        base_model_args: Base arguments for model loading
        tokenizer: Tokenizer for model alignment
        rank: Current process rank
        osft_rank_ratio: Ratio for OSFT rank selection
        osft_target_patterns: Patterns for selecting OSFT target parameters
        osft_upcast_dtype: Dtype for OSFT computations
        osft_output_dtype: Dtype for OSFT outputs

    Returns:
        OSFT model ready for FSDP2 wrapping
    """
    log_rank_0("setting up OSFT model using the distributed loading strategy")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "setup_osft_model_distributed requires torch.distributed to be available and initialized. "
            "For non-distributed training, use the model's from_pretrained method directly with fsdp2_lazy_init=False."
        )

    osft_kwargs = _build_osft_kwargs(osft_rank_ratio, osft_target_patterns)

    # Determine the actual model class and config
    actual_model_class = get_model_class_from_config(model_name_or_path)

    # Create OSFT model class and load model
    osft_cls = create_osft_model_class(actual_model_class)
    model_load_args = {
        **base_model_args,
        "initialize_osft": True,
        "fsdp2_lazy_init": True,
        **osft_kwargs,
    }

    # Provide an alignment hook so rank 0 can adjust embeddings before broadcasting
    def _lazy_align(model):
        return align_model_and_tokenizer(model, tokenizer)

    model_load_args["lazy_init_tokenizer_align_fn"] = _lazy_align

    log_rank_0("loading OSFT model")
    model: OSFTModel = osft_cls.from_pretrained(
        **model_load_args,
    )

    # only global rank 0 should have this state dict
    if dist.get_rank() == 0:
        if not (model._lazy_init_pending and model._lazy_init_og_state_dict):
            raise RuntimeError(
                "Rank 0: Expected model._lazy_init_pending=True and model._lazy_init_og_state_dict to be set"
            )
    else:
        if not (model._lazy_init_pending and not model._lazy_init_og_state_dict):
            raise RuntimeError(
                f"Rank {dist.get_rank()}: Expected model._lazy_init_pending=True and model._lazy_init_og_state_dict to be None"
            )

    # wait for all ranks to reach this point -- if an exception occurs
    # then it will be easy to trace
    dist.barrier()

    # Handle initialization based on memory_efficient_init flag
    return model


def setup_sft_model_distributed(
    model_name_or_path: str,
    base_model_args: dict,
    tokenizer,
    ModelClass: type,
    train_dtype: torch.dtype,
):
    """
    Initialize an SFT model for distributed training with memory-efficient loading.

    Minimal implementation:
    - Rank 0: Load model to CPU, extract config and state dict
    - All ranks: Create model on meta device
    - After FSDP2: Broadcast state dict via set_model_state_dict

    This requires torch.distributed to be initialized.

    Args:
        model_name_or_path: HuggingFace model name or path
        base_model_args: Base arguments for model loading
        tokenizer: Tokenizer for model alignment
        ModelClass: Model class to use for loading (e.g., AutoModelForCausalLM)
        train_dtype: Training dtype for model parameters

    Returns:
        SFT model on meta device, ready for FSDP2 wrapping
    """
    log_rank_0("setting up SFT model using the distributed loading strategy")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "setup_sft_model_distributed requires torch.distributed to be available and initialized. "
            "For non-distributed training, use the model's from_pretrained method directly."
        )

    # Rank 0: Load model to CPU and extract config + state dict + buffers
    config = None
    state_dict = None
    buffer_dict = None

    if dist.get_rank() == 0:
        log_rank_0("rank 0: loading model to CPU")
        try:
            with torch.no_grad():
                # Default load targets CPU when no device_map or accelerate is present
                cpu_model = ModelClass.from_pretrained(**base_model_args)
                cpu_model = align_model_and_tokenizer(cpu_model, tokenizer)
                config = cpu_model.config
                state_dict = cpu_model.state_dict()
                buffer_dict = dict(cpu_model.named_buffers())  # Extract all buffers
        finally:
            # Clean up immediately to free memory
            if "cpu_model" in locals():
                del cpu_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_rank_0("state dict and buffers extracted, model deleted")

    # Broadcast config and buffer_dict to all ranks
    dist.barrier()
    mailbox = [config, buffer_dict]
    dist.broadcast_object_list(mailbox, src=0)
    if dist.get_rank() != 0:
        config, buffer_dict = mailbox
    log_rank_0("config and buffers broadcast to all ranks")

    # All ranks: Create model on meta device
    log_rank_0("creating model on meta device")
    with torch.device("meta"):
        model = ModelClass.from_config(config)

    # Align model with tokenizer
    model = align_model_and_tokenizer(model, tokenizer)

    # Store state dict and buffers for post-FSDP loading
    model._fsdp2_pending_state_dict = state_dict if dist.get_rank() == 0 else None
    model._fsdp2_pending_buffers = buffer_dict  # All ranks have buffer_dict
    model._fsdp2_train_dtype = train_dtype  # Store train_dtype for dtype conversion
    set_fsdp2_lazy_init_mode(model, FSDP2_LAZY_INIT_SFT)

    log_rank_0("meta model created, ready for FSDP2 wrapping")
    return model


def setup_model(
    model_name_or_path: str,
    osft: bool = False,
    local_rank: int = 0,
    save_dtype: str | torch.dtype | None = None,
    train_dtype: torch.dtype = torch.float32,
    osft_upcast_dtype: torch.dtype = torch.float32,
    osft_output_dtype: torch.dtype | None = None,
    osft_rank_ratio: float | None = None,
    osft_target_patterns: list[str] | None = None,
    use_liger_kernels: bool = False,
) -> torch.nn.Module | OSFTModel:
    base_model_args = {
        "pretrained_model_name_or_path": model_name_or_path,
        "torch_dtype": train_dtype,  # Ensure models are loaded in the training dtype
    }

    # Get model config to check for GPT-OSS and set appropriate configurations
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    is_gpt_oss = is_gpt_oss_model(model_config)

    # Set up quantization config for GPT-OSS models
    if is_gpt_oss:
        try:
            # Try to specify the target dtype for dequantization
            quantization_config = Mxfp4Config(dequantize=True)
            # If the config supports dtype specification, use it
            if hasattr(quantization_config, "torch_dtype"):
                quantization_config.torch_dtype = train_dtype
            # Pass quantization_config to from_pretrained
            base_model_args["quantization_config"] = quantization_config
            log_rank_0(
                "🎯 Detected GPT-OSS model - applying dequantization for training"
            )
        except ImportError:
            log_rank_0(
                "⚠️ GPT-OSS model detected but Mxfp4Config not available - using default config"
            )

    # Check if flash_attn is available and set appropriate attention implementation
    try:
        import flash_attn as _  # noqa: F401

        if is_gpt_oss:
            base_model_args["attn_implementation"] = (
                "kernels-community/vllm-flash-attn3"
            )
            log_rank_0("Set attention implementation to vllm-flash-attn3 for GPT-OSS")
        else:
            base_model_args["attn_implementation"] = "flash_attention_2"

    except ImportError as e:
        if os.environ.get("TESTING", "false").lower() == "true":
            base_model_args["attn_implementation"] = "sdpa"
        else:
            raise e

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # patch both loss functions, since models will use the regular HF
    # cross-entropy functions when in eval mode
    from mini_trainer.none_reduction_losses import (
        hf_fixed_cross_entropy_none_reduction,
        liger_fixed_fused_linear_cross_entropy_none_reduction,
    )
    from transformers import AutoModelForCausalLM

    # We patch HF loss unconditionally, since its usage will reappear in other places.
    # For example: when liger is being used and we switch the model into eval mode, it still uses the
    # HF CE loss instead of the Liger Fused Cross-entropy
    patch_target_module(
        "transformers.loss.loss_utils.fixed_cross_entropy",
        hf_fixed_cross_entropy_none_reduction,
    )
    ModelClass = AutoModelForCausalLM

    # ensures liger is available when requested
    if use_liger_kernels:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Tried to use liger kernels, but they are not installed. Please make sure you have installed the necessary cuda dependencies, or disable liger kernels."
            ) from e
        else:
            """need to patch the loss function to not reduce, so we can reduce across all GPUs"""
            patch_target_module(
                "liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy",
                liger_fixed_fused_linear_cross_entropy_none_reduction,
            )
            ModelClass = AutoLigerKernelForCausalLM

    def load_standard_model():
        """Load a standard model (non-OSFT) with memory-efficient distributed loading when available."""
        if dist.is_available() and dist.is_initialized():
            # distributed path: use memory-efficient loading
            return setup_sft_model_distributed(
                model_name_or_path=model_name_or_path,
                base_model_args=base_model_args,
                tokenizer=tokenizer,
                ModelClass=ModelClass,
                train_dtype=train_dtype,
            )
        else:
            # non-distributed path: direct loading
            model = ModelClass.from_pretrained(**base_model_args)
            return align_model_and_tokenizer(model, tokenizer)

    def load_osft_model():
        """Load a model with OSFT (Orthogonal Subspace Fine-Tuning) support."""
        log_rank_0("loading OSFT model")
        # If osft_output_dtype is not specified, use train_dtype for consistency
        effective_osft_output_dtype = (
            osft_output_dtype if osft_output_dtype is not None else train_dtype
        )

        # We monkey-patch the HF model class OSFT will wrap ahead of time so that liger can be properly loaded.
        # This is necessary because OSFT has to set up the model in a very specfic way which is incompatible with the
        # simple load path for SFT.
        _apply_liger_kernels_if_requested(
            use_liger_kernels, model_config, base_model_args
        )

        # Since OSFT requires modifying the base model architecture, we have to write our own
        # model loading logic to prevent wasteful memory usage on CPU and GPU.
        #
        # The general loading procedure works like this:
        # 1. the model's state dict is loaded into CPU memory by rank 0, we refer to this as the OG state dict.
        # 2. every process (including rank 0) initializes the model onto the meta device (no data gets loaded, only metadata)
        # 3. OSFT model builds an internal mapping from the OSFT weights to their original counterparts
        # 4. Model is then wrapped w/ activation checkpointing + FSDP2 modules
        # 5. Rank 0 moves the non-OSFT weights from the OG state dict to the global sharded model,
        #    so each rank only needs to load the pieces FSDP2 assigned to them, avoiding large memory spikes.
        # 6. Now the OG state dict only contains the params that need conversion into OSFT weights. Rank 0
        #    distributes the computation across the world by assigining each process a set of weights to compute SVD
        #    on.
        # 7. Once all processes complete SVD computation, their results are sent back to the global rank 0 process.
        # 8. Rank 0 places the results of the global SVD computation into a partial state dict which it then
        #    distributes into the OSFT model just like in step 5.
        # 9. Finally, all processes have a complete shard of the OSFT model and are able to start training.
        #
        # Something that's particularly annoying with HF Transformers is some models (Phi3) will sometimes store buffers
        # as direct tensor attributes on modules but not register them on the model. To handle these cases, we have to do some special
        # processing on our end in order to populate them with their original data if they're set as meta devices.

        model = None
        if dist.is_available() and dist.is_initialized():
            model = setup_osft_model_distributed(
                model_name_or_path=model_name_or_path,
                base_model_args=base_model_args,
                tokenizer=tokenizer,
                rank=local_rank,
                osft_rank_ratio=osft_rank_ratio,
                osft_target_patterns=osft_target_patterns,
                osft_upcast_dtype=osft_upcast_dtype,
                osft_output_dtype=effective_osft_output_dtype,
            )
        else:
            # non-distributed path: direct OSFT model creation
            actual_model_class = get_model_class_from_config(model_name_or_path)
            osft_cls = create_osft_model_class(actual_model_class)

            # prepare kwargs for OSFT loading
            osft_kwargs = _build_osft_kwargs(osft_rank_ratio, osft_target_patterns)
            model = osft_cls.from_pretrained(
                model_name_or_path,
                fsdp2_lazy_init=False,  # never use lazy init for non-distributed
                initialize_osft=False,  # initialize outside from_pretrained for consistency
                **osft_kwargs,
                **base_model_args,
            )

            model.reinitialize_osft(decompose_existing_weights=True)

        # capture impossible state
        if not model:
            raise RuntimeError("model is still None after OSFT model loading")

        # final configuration
        model = align_model_and_tokenizer(model, tokenizer)
        _set_osft_dtypes(model, osft_upcast_dtype, effective_osft_output_dtype)

        log_rank_0("OSFT model loaded successfully")
        return model

    # Choose whether to apply orthogonal subspace learning (OSL) based on `osft` flag
    # OSL enables continual fine-tuning by constraining updates to low-rank directions orthogonal to critical knowledge that is to be preserved
    model = load_osft_model() if osft else load_standard_model()

    # here we handle configuring the save_dtype
    model.config.torch_dtype = get_model_save_dtype(save_dtype, model_name_or_path)
    if not model.config.torch_dtype:
        raise ValueError(
            "error: model does not have a `torch_dtype` setting, cannot save model in this dtype"
        )

    # Freeze GPT-OSS router parameters BEFORE FSDP2 setup to avoid uniformity issues
    if is_gpt_oss:
        freeze_router_params(model)

    # Convert all trainable parameters to specified training dtype
    log_rank_0(f"🔧 Converting trainable parameters to {train_dtype} for training")
    converted_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != train_dtype:
            param.data = param.data.to(train_dtype)
            converted_count += 1
    if converted_count > 0:
        log_rank_0(f"✅ Converted {converted_count} parameters to {train_dtype}")
    else:
        log_rank_0(f"✅ All parameters already in {train_dtype}")

    # Get the base class name (strip WithOSFT suffix if present for OSFT models)
    class_name = model.__class__.__name__
    if class_name.endswith("WithOSFT"):
        class_name = class_name[:-8]  # Remove "WithOSFT"

    # List of supported architectures
    if class_name not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM",
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "Qwen2ForCausalLM",
        "Phi3ForCausalLM",  # covers phi3 and phi4
        "Qwen3ForCausalLM",
    ]:
        log_rank_0(
            f"\033[38;2;255;255;0mWarning: Model class name: {class_name} is not in the list of supported models.\033[0m",
            to_print=True,
        )

    # NOTE: Don't enable HuggingFace gradient checkpointing with FSDP2
    # It causes conflicts. TorchTitan applies PyTorch's checkpoint wrapper
    # BEFORE FSDP2 wrapping if needed.
    # model.gradient_checkpointing_enable()
    # torch.compile(model)
    return model


def setup_training_components(
    model: torch.nn.Module,
    learning_rate: float,
    num_warmup_steps: int,
    lr_scheduler: str,
    num_training_steps: Optional[int] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    # AdamW optimizer parameters
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    """
    Set up training components including model wrapping, optimizer, and learning rate scheduler.

    This function orchestrates the three-phase model initialization pipeline:
    1. Phase 1: Prepare model for FSDP2 (extract state dicts, materialize buffers)
    2. Phase 2: Pure FSDP2 wrapping (activation checkpointing + sharding)
    3. Phase 3: Finalize initialization (distribute weights, compute SVD for OSFT)

    Args:
        model: The model to be trained
        learning_rate: Peak learning rate for the optimizer
        num_warmup_steps: Number of warmup steps for the LR scheduler
        lr_scheduler: Type of learning rate scheduler to use
        num_training_steps: Total number of training steps (required for some schedulers)
        scheduler_kwargs: Additional scheduler-specific keyword arguments

    Returns:
        Tuple of (wrapped_model, optimizer, lr_scheduler)
    """
    from transformers import get_scheduler

    log_rank_0("=" * 80)
    log_rank_0("Starting three-phase model initialization pipeline")
    log_rank_0("=" * 80)

    # Phase 1: Prepare model for FSDP2 wrapping
    init_context = prepare_model_for_fsdp2(model)

    # Phase 2: Pure FSDP2 wrapping
    model = wrap_fsdp2(model)

    # Phase 3: Finalize model initialization (distribute weights)
    model = finalize_model_initialization(model, init_context)

    log_rank_0("=" * 80)
    log_rank_0("Model initialization pipeline complete")
    log_rank_0("=" * 80)
    log_rank_0("Using FSDP2 wrapper")

    # Filter parameters to only include those that require gradients
    # This handles cases where some parameters (e.g., frozen router params) have requires_grad=False
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Count trainable parameters for logging
    total_params = sum(1 for _ in model.parameters())
    trainable_count = len(trainable_params)
    if total_params != trainable_count:
        log_rank_0(
            f"📊 Using {trainable_count}/{total_params} trainable parameters in optimizer"
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    optimizer = osft_utils.optim_wrapper(optimizer, model)
    # Prepare scheduler kwargs
    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    lr_scheduler = get_scheduler(
        name=lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_kwargs,
    )
    return model, optimizer, lr_scheduler
