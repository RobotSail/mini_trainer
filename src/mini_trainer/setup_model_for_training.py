import json
import gc
import math
import os
from typing import Optional, Dict, Any
import psutil
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from transformers import AutoTokenizer, AutoConfig, Mxfp4Config
from mini_trainer.utils import get_model_class_from_config, log_rank_0, patch_target_module
from mini_trainer.osft_utils import OSFTModel, _build_osft_kwargs, create_osft_model_class, _configure_osft_model
from mini_trainer.gpt_oss_utils import freeze_router_params, is_gpt_oss_model
from mini_trainer.osft_utils import optim_wrapper



# New simple HF-only activation-checkpointing + FSDP2 wrapper
# This mirrors TorchTitan: checkpoint each block, then shard each block and the full model.
def _sanitize_meta_rope_tensors(model: torch.nn.Module) -> int:
    """Fixes meta tensors in RoPE modules by aliasing to existing buffers or recomputing.

    Returns the number of modules sanitized.
    """
    repaired = 0
    # best-effort device fallback
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    for module in model.modules():
        # identify rotary embedding-like modules without relying on specific class names
        has_rope = hasattr(module, "rope_init_fn") and hasattr(module, "config")
        if not has_rope:
            continue

        orig = getattr(module, "original_inv_freq", None)
        if not isinstance(orig, torch.Tensor):
            continue

        if orig.device.type != "meta":
            continue

        inv = getattr(module, "inv_freq", None)
        if isinstance(inv, torch.Tensor) and inv.device.type != "meta":
            module.original_inv_freq = inv
            repaired += 1
            continue

        # determine target device to recompute
        target_device = None
        # prefer any local non-meta parameter/buffer
        for p in module.parameters(recurse=False):
            if isinstance(p, torch.Tensor) and p.device.type != "meta":
                target_device = p.device
                break
        if target_device is None:
            for _, b in module.named_buffers(recurse=False):
                if isinstance(b, torch.Tensor) and b.device.type != "meta":
                    target_device = b.device
                    break
        if target_device is None:
            target_device = model_device

        try:
            inv_freq, _ = module.rope_init_fn(module.config, target_device)
            # keep inv_freq buffer as-is; only fix the non-buffer attribute
            module.original_inv_freq = inv_freq
            repaired += 1
        except Exception:
            # leave untouched if recomputation is not possible
            pass

    return repaired


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
    default_device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    def _is_osft_owned_attribute(module: torch.nn.Module, name: str) -> bool:
        if name.startswith("osft_") or name in {"U_low", "S_low", "V_low", "rank_high"}:
            return True
        return hasattr(module, "osft_params") and name in {"U_low", "S_low", "V_low", "rank_high"}

    for module in model.modules():
        # collect available candidates from this module
        buf_map = dict(module._buffers) if hasattr(module, "_buffers") else {}
        param_map = dict(module._parameters) if hasattr(module, "_parameters") else {}

        # helper to pick a candidate tensor by name
        def _get_by_name(name: str) -> torch.Tensor | None:
            t = buf_map.get(name)
            if t is not None:
                return t
            p = param_map.get(name)
            if p is not None:
                return p
            return None

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
                    if hasattr(model, 'output_dtype'):
                        expected_dtype = model.output_dtype
                    elif hasattr(model, 'dtype'):
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
                    module.__dict__[attr_name] = value.to(device=target_device, dtype=value.dtype)
                    repaired += 1
                except Exception:
                    # leave as-is if movement is unsafe
                    pass

    return repaired


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
        raise RuntimeError("_synchronize_state_dict_fsdp2 requires torch.distributed to be initialized")
    
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
        )
    )
    log_rank_0("✅ State dict synchronized across all ranks")



def wrap_fsdp2(model: torch.nn.Module) -> torch.nn.Module:
    # Check for SFT lazy initialization
    rank0_sft_state_dict = None
    sft_fsdp2_lazy_init = False
    if getattr(model, '_requires_fsdp2_init', False):
        sft_fsdp2_lazy_init = True
        # Extract state dict from rank 0
        rank0_sft_state_dict = getattr(model, '_fsdp2_pending_state_dict', None)
        if dist.get_rank() == 0:
            if rank0_sft_state_dict is None or len(rank0_sft_state_dict) == 0:
                raise RuntimeError("Rank 0 must have a non-empty state dict for SFT lazy init")
            log_rank_0("📦 [SFT FSDP2] Rank 0 has state dict for lazy init")
        else:
            if rank0_sft_state_dict is not None and len(rank0_sft_state_dict) > 0:
                raise RuntimeError("Non-rank 0 should not have state dict for SFT lazy init")

        # Clean up the attributes from the model
        model._requires_fsdp2_init = False
        model._fsdp2_pending_state_dict = None

    # This is where we must pay careful attention to OSFT, if it's using the FSDP2 lazy initialization
    rank0_og_osft_state_dict = None
    osft_fsdp2_lazy_init = False
    if getattr(model, 'requires_fsdp2_initialization', False):
        osft_fsdp2_lazy_init = True
        # shifts model into OSFT format across all procs.
        # this process is not parallelized, but it should be synchronized since each proc
        # should have an identical meta state-dict

        # pull the original state dict, but only the rank 0 should have this populated
        rank0_og_osft_state_dict = model.eject_og_state_dict()
        if dist.get_rank() == 0:
            assert rank0_og_osft_state_dict is not None and len(rank0_og_osft_state_dict) > 0
            # assert the lm head is in it originally
        else:
            assert rank0_og_osft_state_dict is None or len(rank0_og_osft_state_dict) == 0
        



    # 4) Mixed-precision policy using bfloat16 for Flash Attention compatibility
    # Flash Attention requires bfloat16 for proper operation
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.float32,
    )

    # ------------------------------------------------------------------------------------
    # CONTINUE STANDARD FSDP2 INITIALIZATION
    # ------------------------------------------------------------------------------------

    if hasattr(model, 'config'):
        try:
            model.config.use_cache = False
        except Exception as e:
            print(
                f"WARNING: Failed to disable HuggingFace cache for model {model.__class__.__name__}: {e}"
            )
            pass
    # Materialize SFT buffers BEFORE FSDP2 wrapping (if needed)
    if sft_fsdp2_lazy_init and hasattr(model, '_fsdp2_pending_buffers'):
        buffer_dict = model._fsdp2_pending_buffers
        if buffer_dict:
            log_rank_0(f"🔧 [SFT FSDP2] Materializing {len(buffer_dict)} buffers before FSDP2 wrapping")

            def get_module_by_name(model, name):
                """Helper to traverse and retrieve a module and its attribute by name."""
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

            # Materialize each buffer
            for buf_name, buf_data in buffer_dict.items():
                mod, attr = get_module_by_name(model, buf_name)
                if mod is not None:
                    # Verify current buffer is on meta device
                    curr_buff = getattr(mod, attr, None)
                    if curr_buff is not None and curr_buff.device.type == "meta":
                        # Check if dtype conversion is needed
                        expected_dtype = curr_buff.dtype
                        if buf_data.dtype != expected_dtype:
                            buf_data = buf_data.to(dtype=expected_dtype)

                        # Clone the buffer data and register it
                        new_data = buf_data.detach().clone()
                        mod.register_buffer(attr, new_data, persistent=True)

            log_rank_0("✅ [SFT FSDP2] Buffers materialized successfully")

        # Clean up buffer dict
        model._fsdp2_pending_buffers = None

    # 1) Find the HF transformer block container (GPT2: transformer.h, Llama: model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2, GPT-J, etc.: model.transformer.h
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find transformer block container on model. This likely means we need to update the code to support this model.")


    # 2) Activation checkpoint each block
    for idx, block in enumerate(layers):
        layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)


    # 3) Build a 1D device mesh over all ranks
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", [world_size], mesh_dim_names=["fsdp"])


    # 4) FSDP2 wrap each block
    log_rank_0("🔄 [OSFT FSDP2] Step 4: Wrapping model blocks with FSDP2")
    for idx, block in enumerate(layers):
        reshard = idx < len(layers) - 1
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard,
        )
    log_rank_0(f"   • Wrapped {len(layers)} blocks with FSDP2")

    # 5) FSDP2 wrap full model
    log_rank_0("🔄 [OSFT FSDP2] Step 5: Wrapping full model with FSDP2")
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    log_rank_0("   • Full model wrapped with FSDP2")

    # Handle SFT lazy initialization (distribute state dict)
    if sft_fsdp2_lazy_init:
        log_rank_0("🔄 [SFT FSDP2] Distributing state dict to all ranks")

        # Convert dtypes on rank 0 before broadcasting (set_model_state_dict doesn't cast like load_state_dict)
        if dist.get_rank() == 0 and rank0_sft_state_dict:
            expected_dtype = getattr(model, '_fsdp2_train_dtype', model.config.torch_dtype)
            converted_state_dict = {}
            conversions = 0

            for key, value in rank0_sft_state_dict.items():
                if isinstance(value, torch.Tensor) and value.dtype != expected_dtype:
                    converted_state_dict[key] = value.to(dtype=expected_dtype)
                    conversions += 1
                else:
                    converted_state_dict[key] = value

            if conversions > 0:
                log_rank_0(f"🔧 [SFT FSDP2] Converted {conversions} parameters to {expected_dtype}")
            rank0_sft_state_dict = converted_state_dict

        _synchronize_state_dict_fsdp2(
            model=model,
            state_dict=rank0_sft_state_dict if dist.get_rank() == 0 else {},
            strict=False,  # Use strict=False since buffers are handled separately
        )
        log_rank_0("✅ [SFT FSDP2] State dict distributed successfully")

        # Clean up temporary attributes
        if hasattr(model, '_fsdp2_train_dtype'):
            del model._fsdp2_train_dtype

        fixed_generic = _sanitize_meta_attribute_aliases(model)
        if fixed_generic:
            log_rank_0(f"🧩 [FSDP2] Sanitized {fixed_generic} meta tensor attributes")
        log_rank_0("✅ [SFT FSDP2] SFT lazy init complete, returning model")
        return model

    # Normal nodes can exit by now (non-OSFT, non-SFT lazy init)
    if not osft_fsdp2_lazy_init:
        fixed_generic = _sanitize_meta_attribute_aliases(model)
        if fixed_generic:
            log_rank_0(f"🧩 [FSDP2] Sanitized {fixed_generic} meta tensor attributes")
        log_rank_0("✅ [FSDP2] Non-lazy-init path complete, returning model")
        return model
    
    # ------------------------------------------------------------------------------------
    # OSFT FSDP2 INITIALIZATION STUB 
    # ------------------------------------------------------------------------------------
    # ****** FSDP2 Initialization for OSFT *****
    # 
    # In this codeblock, we have to run a specific algorithm to prepare the model's
    # OSFT parameters when working with FSDP2. 
    # 
    # To avoid duplicating CPU memory for each process, it's recommended to load the model
    # once on the main process and have all other procs in the group load it onto the meta device.
    # Once sharded in FSDP2, we can simply broadcast the state dict to every other process.
    # 
    # To make this work with OSFT, the computation of OSFT params can only be loaded **after**
    # we've already sharded the model, or otherwise we'd be wasting computation time. However;
    # since FSDP2 cannot modify the architecture once everything is sharded, the model
    # must be prepared to receive the SVD data in the expected format.
    # 
    # The following algorithm therefore does the following:
    #   1. prepares each process to receive the OSFT params
    #   2. shards the model with fsdp2
    #   3. shares all non-OSFT params with the sharded models from rank 0 master state dict
    #   4. distributes SVD computation across all procs
    #   5. gathers resulting OSFT params on rank 0
    #   6. distributes the final OSFT params across the state dict
 
    # now we need to share out all of the non-osft params
    model.post_fsdp2_wrap_synchronize_state_dict_across_procs(model, rank0_og_osft_state_dict)

    # ------------------------------------------------------------------------------------
    # CONTINUE POST-FSDP2 OSFT CONFIGURATION:
    # Here is where the main process broadcasts tensor data to all processes and
    # handles distributed SVD calculation.
    # ------------------------------------------------------------------------------------

    model.compute_distributed_svd(model, rank0_og_osft_state_dict)

    fixed_generic = _sanitize_meta_attribute_aliases(model)
    if fixed_generic:
        log_rank_0(f"🧩 [FSDP2] Sanitized {fixed_generic} meta tensor attributes")

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

    # Fix any discrepancy between model and tokenizer
    special_tokens = {
        'pad': ('pad_token_id', 'Fixing model pad token id'),
        'bos': ('bos_token_id', 'Fixing model bos token id'),
        'eos': ('eos_token_id', 'Fixing model eos token id')
    }

    for token_type, (token_attr, message) in special_tokens.items():
        model_token = getattr(model.config, token_attr)
        tokenizer_token = getattr(tokenizer, token_attr)
        
        if (model_token is not None and tokenizer_token is not None 
            and model_token != tokenizer_token):
            log_rank_0(
                "\033[38;5;226m"
                f"WARNING: There is a mismatch between {token_type} token id of "
                f"model({model_token}) and tokenizer({tokenizer_token}). "
                f"{message} to be same as tokenizer's {token_type} token id"
                "\033[0m"
            )
            setattr(model.config, token_attr, tokenizer_token)

    return model


def get_model_save_dtype(save_dtype: str | torch.dtype | None, model_name_or_path: str) -> torch.dtype:
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
        log_rank_0(f"⚠️ Model does not have a setting for `torch_dtype` and not `save_dtype` was provided, falling back to '{default_dtype}'")
        return default_dtype

    # handles the case save_dtype > original_dtype
    if not save_dtype:
        return original_dtype
    
    # by now we know that we are going to use a custom data type, so we just validate
    if not isinstance(save_dtype, (str, torch.dtype)):
        raise ValueError(f"error: could not recognize '{save_dtype}' as a supported dtype for saving model checkpoints")
 
    # convert dtype to a str
    if isinstance(save_dtype, str):
        if save_dtype not in dtype_map:
            raise ValueError(f"error: could not recognize '{save_dtype}' as a supported dtype for saving model checkpoints")
        save_dtype = dtype_map[save_dtype]
    
    # alert the user when the dtype differs
    if original_dtype and original_dtype != save_dtype:
        log_rank_0(f"⚠️ Model's original dtype is '{original_dtype}', but new checkpoints will be saved as '{save_dtype}'. ⚠️")
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
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "setup_osft_model_distributed requires torch.distributed to be available and initialized. "
            "For non-distributed training, use the model's from_pretrained method directly with fsdp2_lazy_init=False."
        )

    osft_kwargs = _build_osft_kwargs(osft_rank_ratio, osft_target_patterns)


    # Determine the actual model class and config
    actual_model_class = get_model_class_from_config(model_name_or_path)


    # Create OSFT model class and load model
    log_rank_0("📦 [setup_osft_model] Creating OSFT model class and loading pretrained weights")
    osft_cls = create_osft_model_class(actual_model_class)
    model_load_args = {
        **base_model_args,
        "initialize_osft": True,
        "fsdp2_lazy_init": True,
        **osft_kwargs,
    }
 

    log_rank_0("🚀 [setup_osft_model] Calling osft_cls.from_pretrained (initialize_osft=False)")
    model: OSFTModel = osft_cls.from_pretrained(
        **model_load_args,
    )
    
    # apply common OSFT configuration
    model = _configure_osft_model(model, tokenizer, osft_upcast_dtype, osft_output_dtype)


    # Print CPU memory utilization on each node
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_rss_gb = mem_info.rss / (1024 ** 3)  # Convert bytes to GB
    mem_vms_gb = mem_info.vms / (1024 ** 3)  # Convert bytes to GB
    
    
    print(f"[Rank {rank}] CPU Memory - Process RSS: {mem_rss_gb:.2f} GB, VMS: {mem_vms_gb:.2f} GB")

    # only global rank 0 should have this state dict
    if dist.get_rank() == 0:
        assert model._lazy_init_pending and model._lazy_init_og_state_dict
    else:
        assert model._lazy_init_pending and not model._lazy_init_og_state_dict

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
        log_rank_0("📦 [setup_sft_model] Rank 0: Loading model to CPU")
        try:
            with torch.no_grad():
                # Load model with device_map='cpu' to keep it on CPU
                cpu_model = ModelClass.from_pretrained(**base_model_args, device_map='cpu')
                config = cpu_model.config
                state_dict = cpu_model.state_dict()
                buffer_dict = dict(cpu_model.named_buffers())  # Extract all buffers
        finally:
            # Clean up immediately to free memory
            if 'cpu_model' in locals():
                del cpu_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_rank_0("✅ [setup_sft_model] Rank 0: State dict and buffers extracted, model deleted")

    # Broadcast config and buffer_dict to all ranks
    dist.barrier()
    mailbox = [config, buffer_dict]
    dist.broadcast_object_list(mailbox, src=0)
    if dist.get_rank() != 0:
        config, buffer_dict = mailbox
    log_rank_0("✅ [setup_sft_model] Config and buffers broadcast to all ranks")

    # All ranks: Create model on meta device
    log_rank_0("🏗️ [setup_sft_model] Creating model on meta device (all ranks)")
    with torch.device("meta"):
        model = ModelClass.from_config(config)

    # Align model with tokenizer
    model = align_model_and_tokenizer(model, tokenizer)

    # Store state dict and buffers for post-FSDP loading
    model._fsdp2_pending_state_dict = state_dict if dist.get_rank() == 0 else None
    model._fsdp2_pending_buffers = buffer_dict  # All ranks have buffer_dict
    model._fsdp2_train_dtype = train_dtype  # Store train_dtype for dtype conversion
    model._requires_fsdp2_init = True

    log_rank_0("✅ [setup_sft_model] Meta model created, ready for FSDP2 wrapping")
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
            if hasattr(quantization_config, 'torch_dtype'):
                quantization_config.torch_dtype = train_dtype
            # Pass quantization_config to from_pretrained
            base_model_args["quantization_config"] = quantization_config
            log_rank_0("🎯 Detected GPT-OSS model - applying dequantization for training")
        except ImportError:
            log_rank_0("⚠️ GPT-OSS model detected but Mxfp4Config not available - using default config")
    
    # Check if flash_attn is available and set appropriate attention implementation
    try:
        import flash_attn
        if is_gpt_oss:
            base_model_args["attn_implementation"] = "kernels-community/vllm-flash-attn3"
            log_rank_0("Set attention implementation to vllm-flash-attn3 for GPT-OSS")
        else:
            base_model_args["attn_implementation"] = "flash_attention_2"

    except ImportError as e:
        if os.environ.get("TESTING", "false").lower() == "true":
            base_model_args["attn_implementation"] = "eager"
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
            raise ImportError("Tried to use liger kernels, but they are not installed. Please make sure you have installed the necessary cuda dependencies, or disable liger kernels.") from e
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
        # If osft_output_dtype is not specified, use train_dtype for consistency
        effective_osft_output_dtype = osft_output_dtype if osft_output_dtype is not None else train_dtype
        
        if dist.is_available() and dist.is_initialized():
            # distributed path: always use memory-efficient loading
            return setup_osft_model_distributed(
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
                initialize_osft=True,   # initialize OSFT immediately
                **osft_kwargs,
                **base_model_args,
            )
            
            # apply common configuration
            model = _configure_osft_model(model, tokenizer, osft_upcast_dtype, effective_osft_output_dtype)
            
            return model
    
    # Choose whether to apply orthogonal subspace learning (OSL) based on `osft` flag
    # OSL enables continual fine-tuning by constraining updates to low-rank directions orthogonal to critical knowledge that is to be preserved
    model = load_osft_model() if osft else load_standard_model()

    # here we handle configuring the save_dtype
    model.config.torch_dtype = get_model_save_dtype(save_dtype, model_name_or_path)
    if not model.config.torch_dtype:
        raise ValueError("error: model does not have a `torch_dtype` setting, cannot save model in this dtype")

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
        # NEED TO CHECK QWEN3
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
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Set up training components including model wrapping, optimizer, and learning rate scheduler.
    
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
    
    # Using FSDP2 wrapper
    log_rank_0("Using FSDP2 wrapper")
    model = wrap_fsdp2(model)
    
    # Filter parameters to only include those that require gradients
    # This handles cases where some parameters (e.g., frozen router params) have requires_grad=False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Count trainable parameters for logging
    total_params = sum(1 for _ in model.parameters())
    trainable_count = len(trainable_params)
    if total_params != trainable_count:
        log_rank_0(f"📊 Using {trainable_count}/{total_params} trainable parameters in optimizer")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        foreach=False
    )
    optimizer = optim_wrapper(optimizer, model)
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
    lr_scheduler.split_batches = True
    lr_scheduler.step() #the scheduler starts at 0 and there's no learning.
    return model, optimizer, lr_scheduler

