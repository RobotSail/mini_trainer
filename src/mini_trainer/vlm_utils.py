"""Utilities for detecting and extracting CausalLM text backbones from VLM models.

Vision-Language Models (VLMs) like Mistral3ForConditionalGeneration wrap a
CausalLM text backbone (e.g. Ministral3ForCausalLM).  This module provides
helpers to detect that wrapping and extract the text backbone so mini-trainer
can treat it as a standard CausalLM for SFT / OSFT training.
"""

import torch
import torch.nn as nn
from transformers.models.auto import MODEL_FOR_CAUSAL_LM_MAPPING

from mini_trainer.utils import log_rank_0


def is_vlm_with_causal_lm(config) -> bool:
    """Check if a model config is a VLM wrapping a CausalLM text backbone.

    Returns True only when the top-level config is NOT in the CausalLM
    mapping but its nested ``text_config`` IS.  Models that are directly
    registered as CausalLM (even if they also have a text_config) return
    False.

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if the model is a VLM wrapping a CausalLM text backbone.
    """
    if config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING:
        return False
    text_config = getattr(config, "text_config", None)
    return text_config is not None and text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING


def _find_text_backbone(vlm_model: nn.Module) -> nn.Module:
    """Auto-detect the text backbone inside a VLM model.

    Tries well-known attribute names first (``language_model``,
    ``text_model``, ``llm``), then falls back to searching
    ``named_children`` for class names containing ``ForCausalLM`` or
    ``TextModel``.

    Args:
        vlm_model: The loaded VLM model.

    Returns:
        The text backbone module.

    Raises:
        ValueError: If no text backbone can be found.
    """
    inner = vlm_model.model if hasattr(vlm_model, "model") else vlm_model

    # Well-known attribute names
    for attr_name in ("language_model", "text_model", "llm"):
        if hasattr(inner, attr_name):
            return getattr(inner, attr_name)

    # Fallback: search named children for common class-name patterns
    for name, child in inner.named_children():
        cls_name = child.__class__.__name__
        if "ForCausalLM" in cls_name or "TextModel" in cls_name:
            return child

    available = [name for name, _ in inner.named_children()]
    raise ValueError(
        f"Cannot find text backbone in {type(vlm_model).__name__}. "
        f"Available sub-modules on inner model: {available}"
    )


def extract_causal_lm_from_vlm(model_path: str, load_kwargs: dict) -> nn.Module:
    """Load a VLM and extract the CausalLM text backbone.

    Loads the full VLM via ``AutoModelForImageTextToText``, auto-detects
    the text backbone using :func:`_find_text_backbone`, then creates a
    standalone CausalLM model by transferring weights.

    Args:
        model_path: HuggingFace model name or local path.
        load_kwargs: Keyword arguments forwarded to ``from_pretrained``.

    Returns:
        A standalone CausalLM model with the VLM's text weights.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    log_rank_0("🔄 VLM detected – loading full VLM to extract CausalLM text backbone")

    # Filter out None quantization_config to avoid interfering with
    # the model's built-in quantization handling (e.g. FP8 auto-dequant)
    vlm_kwargs = {
        k: v for k, v in load_kwargs.items()
        if not (k == "quantization_config" and v is None)
    }
    vlm = AutoModelForImageTextToText.from_pretrained(model_path, **vlm_kwargs)

    # Auto-detect text backbone
    backbone = _find_text_backbone(vlm)

    # Resolve text_config and create standalone CausalLM
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = config.text_config
    causal_lm_class = MODEL_FOR_CAUSAL_LM_MAPPING[text_config.__class__]

    log_rank_0(f"   Extracting {causal_lm_class.__name__} from {type(vlm).__name__}")
    text_model = causal_lm_class(text_config)

    # Transfer backbone weights
    text_model.model = backbone

    # Transfer lm_head
    if hasattr(vlm, "lm_head"):
        text_model.lm_head = vlm.lm_head
    else:
        raise ValueError(f"Cannot extract lm_head from {type(vlm).__name__}")

    del vlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_rank_0(f"   ✅ Extracted {causal_lm_class.__name__} successfully")
    return text_model


def has_mrope(config) -> bool:
    """Check if a model config uses M-RoPE (multimodal rotary position embeddings).

    Inspects both the top-level config and its ``text_config`` (if present)
    for ``rope_scaling`` or ``rope_parameters`` dicts containing the
    ``mrope_section`` key.

    Args:
        config: An already-loaded HuggingFace model config object.

    Returns:
        True if M-RoPE is detected.
    """
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in ("rope_scaling", "rope_parameters"):
            rope_dict = getattr(cfg, attr, None)
            if isinstance(rope_dict, dict) and "mrope_section" in rope_dict:
                return True
    return False
