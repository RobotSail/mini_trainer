import json
import gc
import math
import os
from sre_parse import State
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import set_model_state_dict, set_state_dict, StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Mxfp4Config
from mini_trainer.utils import get_model_class_from_config, log_rank_0, patch_target_module
from mini_trainer.osft_utils import OSFTModel, _build_osft_kwargs, _initialize_osft_with_distribution, _set_osft_dtypes, create_osft_model_class
from mini_trainer.gpt_oss_utils import freeze_router_params, is_gpt_oss_model


def load_and_distribute_config(
    model_class,
    base_model_args: dict,
    rank: int,
) -> Any:
    """
    Load model config on rank 0 and distribute to all ranks.
    
    This function ensures memory-efficient config loading by:
    1. Only rank 0 loads the actual model onto CPU
    2. Config is extracted and deep-copied via serialization
    3. Original model is deleted and garbage collected
    4. Config dict is broadcast to all other ranks
    
    Args:
        model_class: The model class to use for loading
        model_name_or_path: Path to the pretrained model
        base_model_args: Arguments for model loading
        rank: Current process rank
        
    Returns:
        The model config object (available on all ranks)
    """
    if rank == 0:
        log_rank_0("📥 [Rank 0] Loading model to extract config...")
        # load the model on CPU only on rank 0
        tmp = model_class.from_pretrained(**base_model_args)
        
        # extract config as JSON string (this creates a deep copy disconnected from the model)
        config_json = tmp.config.to_json_string()
        
        # delete the model immediately to free memory
        del tmp
        torch.cuda.empty_cache()
        log_rank_0("🗑️ [Rank 0] Model deleted, config extracted")
    else:
        config_json = None
    
    # broadcast the config JSON from rank 0 to all other ranks
    log_rank_0("📡 Broadcasting config from rank 0 to all ranks...")
    config_json_list = [config_json]
    dist.broadcast_object_list(config_json_list, src=0)
    config_json = config_json_list[0]
    
    # reconstruct config from JSON on all ranks
    config_dict = json.loads(config_json)
    model_type = config_dict.get("model_type")
    if model_type is None:
        raise ValueError("Config dict does not contain 'model_type' field")
    
    config = AutoConfig.for_model(**config_dict)
    
    import gc
    gc.collect()

    
    log_rank_0(f"✅ Config distributed to all ranks (rank {rank})")
    return config


# New simple HF-only activation-checkpointing + FSDP2 wrapper
# This mirrors TorchTitan: checkpoint each block, then shard each block and the full model.
def wrap_fsdp2(model: torch.nn.Module) -> torch.nn.Module:
    # TODO: use the efficient initalization for SFT as well


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
    # 1) Find the HF transformer block container (GPT2: transformer.h, Llama: model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2, GPT-J, etc.: model.transformer.h
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find transformer block container on model. This likely means we need to update the code to support this model.")
 

    # # TODO: make this work again
    # # # 2) Activation checkpoint each block
    # for idx, block in enumerate(layers):
    #     layers[idx] = ptd_checkpoint_wrapper(block, preserve_rng_state=False)


    # 3) Build a 1D device mesh over all ranks
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", [world_size], mesh_dim_names=["fsdp"])


    # 4) FSDP2 wrap each block
    log_rank_0("🔄 [OSFT FSDP2] Step 4: Wrapping model blocks with FSDP2")
    for idx, block in enumerate(layers):
        reshard = idx < len(layers) - 1
        fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard)
    log_rank_0(f"   • Wrapped {len(layers)} blocks with FSDP2")

    # 5) FSDP2 wrap full model
    log_rank_0("🔄 [OSFT FSDP2] Step 5: Wrapping full model with FSDP2")
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    log_rank_0("   • Full model wrapped with FSDP2")
    
    # Normal nodes can exit by now
    if not osft_fsdp2_lazy_init:
        log_rank_0("✅ [OSFT FSDP2] Non-OSFT path complete, returning model")
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

def setup_osft_model_meta(
    model_name_or_path: str,
    base_model_args: dict,
    tokenizer,
    # this was used to figure out whether to set a config or not, but no config is currently being set
    is_gpt_oss: bool,  
    rank: int,
    osft_rank_ratio=None,
    osft_target_patterns=None,
    osft_upcast_dtype=torch.float32,
    osft_output_dtype=None,
):
    """
    This function initializes an OSFT model, but in an efficient way that scales for distributed
    environments. This requires some additional complexity which doesn't work well for the non-distributed
    path.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("the setup_osft_model_meta` initialization method is not supported for non-distributed methods")

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
    model = align_model_and_tokenizer(model, tokenizer)

    # Set OSFT dtype attributes
    _set_osft_dtypes(model, osft_upcast_dtype, osft_output_dtype)


    # Print CPU memory utilization on each node
    import psutil
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




def setup_osft_model(
    model_name_or_path: str,
    base_model_args: dict,
    tokenizer,
    is_gpt_oss: bool,
    rank: int,
    osft_rank_ratio=None,
    osft_target_patterns=None,
    osft_upcast_dtype=torch.float32,
    osft_output_dtype=None,
):
    """
    High-level function to set up an OSFT model with all necessary configuration.

    This function handles both GPT-OSS and standard model paths with minimal
    duplication.

    Args:
        model_class: The base model class to use
        base_model_args: Arguments for model loading
        tokenizer: Tokenizer for model alignment
        is_gpt_oss: Whether this is a GPT-OSS model
        rank: Current process rank
        osft_rank_ratio: Rank ratio for OSFT decomposition
        osft_target_patterns: Target patterns for OSFT
        osft_upcast_dtype: Upcast dtype for OSFT computations
        osft_output_dtype: Output dtype for OSFT results

    Returns:
        Fully configured and initialized OSFT model
    """
    log_rank_0("🔧 [setup_osft_model] Starting OSFT model setup")
    log_rank_0(f"   • model_name_or_path: {model_name_or_path}")
    log_rank_0(f"   • is_gpt_oss: {is_gpt_oss}")
    log_rank_0(f"   • osft_rank_ratio: {osft_rank_ratio}")
    log_rank_0(f"   • osft_upcast_dtype: {osft_upcast_dtype}")
    log_rank_0(f"   • osft_output_dtype: {osft_output_dtype}")
    

    osft_kwargs = _build_osft_kwargs(osft_rank_ratio, osft_target_patterns)

    # Determine the actual model class and config
    actual_model_class = get_model_class_from_config(model_name_or_path)

    # Create OSFT model class and load model
    log_rank_0("📦 [setup_osft_model] Creating OSFT model class and loading pretrained weights")
    osft_cls = create_osft_model_class(actual_model_class)
    model_load_args = {
        **base_model_args,
        "initialize_osft": False,  # We initialize later via _initialize_osft_with_distribution
        **osft_kwargs,
    }
    

    log_rank_0("🚀 [setup_osft_model] Calling osft_cls.from_pretrained (initialize_osft=False)")
    model: OSFTModel = osft_cls.from_pretrained(**model_load_args)
    model = align_model_and_tokenizer(model, tokenizer)

    # Set OSFT dtype attributes
    _set_osft_dtypes(model, osft_upcast_dtype, osft_output_dtype)

    # Handle initialization based on memory_efficient_init flag
    device = torch.device("cuda", rank)
    log_rank_0(f"📍 [setup_osft_model] Ready to initialize OSFT parameters")
    log_rank_0(f"   • target device: {device}")

    # Memory-efficient: Initialize OSFT on CPU, then move to GPU
    log_rank_0("🧠 [setup_osft_model] Using memory-efficient OSFT initialization (CPU → GPU)")
    log_rank_0("🚀 [setup_osft_model] Calling _initialize_osft_with_distribution (model on CPU)")
    model = _initialize_osft_with_distribution(model)
    log_rank_0("✅ [setup_osft_model] OSFT initialization complete, keeping model on CPU until FSDP2 sharding")

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
            if 'granite-4' not in model_name_or_path:
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
        model = ModelClass.from_pretrained(**base_model_args)
        return align_model_and_tokenizer(model, tokenizer)
    
    def load_osft_model():
        """Load a model with OSFT (Orthogonal Subspace Fine-Tuning) support."""
        # If osft_output_dtype is not specified, use train_dtype for consistency
        effective_osft_output_dtype = osft_output_dtype if osft_output_dtype is not None else train_dtype
        if dist.is_available() and dist.is_initialized():
            return setup_osft_model_meta(
                # model_class=ModelClass,
                model_name_or_path=model_name_or_path,
                base_model_args=base_model_args,
                tokenizer=tokenizer,
                is_gpt_oss=is_gpt_oss,
                rank=local_rank,
                osft_rank_ratio=osft_rank_ratio,
                osft_target_patterns=osft_target_patterns,
                osft_upcast_dtype=osft_upcast_dtype,
                osft_output_dtype=effective_osft_output_dtype,
            )
        else:
            return setup_osft_model(
                # TODO(osilkin): some commit has updated the behavior of OSFT initialization,
                #                so the `model_class` field isn't being read anymore. This means that
                #                liger kernels are not currently working in OSFT. We need to fix this.
                # model_class=ModelClass,
                model_name_or_path=model_name_or_path,
                base_model_args=base_model_args,
                tokenizer=tokenizer,
                is_gpt_oss=is_gpt_oss,
                rank=local_rank,
                osft_rank_ratio=osft_rank_ratio,
                osft_target_patterns=osft_target_patterns,
                osft_upcast_dtype=osft_upcast_dtype,
                osft_output_dtype=effective_osft_output_dtype,
            )
    
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
    
    if class_name not in [
        "MistralForCausalLM",
        "GPTDolomiteForCausalLM", 
        "LlamaForCausalLM",
        "Starcoder2ForCausalLM",
        "GemmaForCausalLM",
        "MixtralForCausalLM",
        "GraniteForCausalLM",
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
    from mini_trainer.osft_utils import optim_wrapper
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

