from typing import Literal, cast

import torch.nn as nn

FSDP2_LAZY_INIT_ATTR = "_fsdp2_lazy_init_mode"
FSDP2LazyInitMode = Literal["sft", "osft"]
FSDP2_LAZY_INIT_SFT: FSDP2LazyInitMode = "sft"
FSDP2_LAZY_INIT_OSFT: FSDP2LazyInitMode = "osft"


def set_fsdp2_lazy_init_mode(
    model: nn.Module,
    mode: FSDP2LazyInitMode | None,
) -> None:
    """
    Tag a model with the FSDP2 lazy-init mode so later phases know which
    initialization pipeline (SFT vs OSFT) to execute.
    """
    if mode is None:
        if hasattr(model, FSDP2_LAZY_INIT_ATTR):
            delattr(model, FSDP2_LAZY_INIT_ATTR)
        return

    if mode not in (FSDP2_LAZY_INIT_SFT, FSDP2_LAZY_INIT_OSFT):
        raise ValueError(f"Unsupported FSDP2 lazy init mode: {mode}")
    setattr(model, FSDP2_LAZY_INIT_ATTR, mode)


def get_fsdp2_lazy_init_mode(model: nn.Module) -> FSDP2LazyInitMode | None:
    """Return the tagged lazy-init mode, if any."""
    mode = getattr(model, FSDP2_LAZY_INIT_ATTR, None)
    if isinstance(mode, str) and mode in (FSDP2_LAZY_INIT_SFT, FSDP2_LAZY_INIT_OSFT):
        return cast(FSDP2LazyInitMode, mode)
    return None
