# Changelog

All notable changes to mini_trainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.4.0] - 2025-11-19

### Added
- **Distributed Initialization Documentation**: Comprehensive architecture documentation explaining the 3-phase distributed initialization process for SFT and OSFT models in `docs/distributed_initialization.md`
- **OSFT Orthogonalization Tests**: Extensive regression tests for OSFT orthogonalization properties (`regression_tests/test_osft_orthogonalization.py`, `tests/test_utils/orthogonality.py`)
- **Model Support**: Added support for Qwen3 and GPT-2 models
- **Activation Checkpointing**: Enabled activation checkpointing support for memory-efficient training
- **FSDP2 Lazy Initialization**: New `fsdp2_lazy_init.py` module for managing lazy initialization state

### Changed
- **Major OSFT Refactor**: Complete refactoring of OSFT implementation for improved memory efficiency and production readiness
  - Restructured model initialization into 3 clear phases: prepare, wrap, and finalize
  - Significantly improved memory efficiency during distributed OSFT model loading
  - Cleaner separation of concerns between SFT and OSFT initialization paths
  - Better handling of tensor attributes and dtype conversions
- **Memory-Efficient Loading Extended to SFT**: SFT models now also benefit from the memory-efficient distributed loading path
- **CI/Linting Improvements**:
  - Linting workflow optimized to only run on changed files between target and base branch
  - Updated GitHub Actions workflows for better efficiency
  - Improved test stability and coverage
- **Code Quality**: Extensive code cleanup, removed dead code, improved error handling with try/finally blocks
- **Dependencies**: Updated to transformers v2.14.0

### Deprecated
- **`osft_memory_efficient_init` parameter**: This flag is deprecated and will be removed in v0.5.0.
  Memory-efficient initialization is now automatically enabled when distributed training is detected
  (when `torch.distributed.is_initialized()` returns `True`). The flag has no effect and can be
  safely removed from training configurations.

### Fixed
- **Non-distributed OSFT**: Fixed implementation and tests for non-distributed OSFT setups
- **Dtype Consistency**: Fixed dtype mismatch issues when loading state dicts from GPT-OSS models
- **Linting Errors**: Resolved various linting issues across the codebase
- **Memory Leaks**: Added try/finally blocks to prevent memory leakage during model initialization
- **Bias Handling**: Fixed pulling bias from modules in OSFT setup
- **Progress Bar**: Fixed epoch counter in progress bar display
- **Tensor Attributes**: Enabled distributed OSFT to properly carry over tensor attributes that are hidden/not registered as parameters or buffers
- **Gloo Backend**: Uses gloo backend when communicating CPU-based objects in distributed setup

## [v0.2.0] - 2025-09-16

### Added
- **GPT-OSS Model Support**
  - Full support for OpenAI's new open-weight GPT-OSS models (20B and 120B variants)
  - Native MXFP4 quantization implementation
  - New `gpt_oss_utils.py` module (430+ lines)
- **Memory-Efficient OSFT Initialization**
  - New `osft_memory_efficient_init` flag for optimized initialization of large models
  - Significant memory savings during model loading
- **Training Dtype Control**
  - New `train_dtype` parameter for switching models to bf16/fp16 training
  - Reduces memory usage (use sparingly as lower precision may impact results)
- **Pretraining Data Conversion**
  - New `convert_to_pretrain.py` script for converting conversation datasets
- **OSFT Dtype Controls**
  - `osft_upcast_dtype` for computation precision (default: float32)
  - `osft_output_dtype` for output precision control
- **Enhanced Data Processing**
  - Improved `process_data.py` with additional functionality
- **Weights & Biases (wandb) integration** for experiment tracking
  - New `wandb_wrapper.py` module
  - Automatic logging of training/validation metrics, gradients, and system stats
  - Opt-in via `--wandb` CLI flag or corresponding config entry
- **Train/Validation Split Support**
  - Deterministic split into train and validation shards in sampler
  - New `--validation-split` argument (default 0.05) controls hold-out fraction
  - Validation loop runs every `validation_frequency` steps
- **Validation Loss Tracking**
  - Validation loss computation and reporting
  - Integration with console logs and wandb dashboards

### Changed
- **Dependencies**
  - Updated transformers to `>=4.55.0`
  - Added liger-kernel for optimized operations
  - Added kernels package for flash-attention-3 support
- Simplified implementation of memory efficient + GPT-OSS loading
- Enhanced test coverage for validation and sampler behavior
- Updated dependencies in `pyproject.toml`

### Fixed
- Various test case failures
- Code optimization and cleanup based on PR feedback
- GPT-OSS checkpoint saving during SFT
- Distributed torch tests stability by mocking `torch.distributed` checks
- Dtype conversion edge-cases
- Default `validation_frequency` is now `None` instead of `0`

## [v0.1.1] - Previous Release

[Previous release details would go here]

---

## Usage Examples

### GPT-OSS-20B Training
```python
from mini_trainer.api_train import run_training
from mini_trainer.training_types import TrainingArgs, TorchrunArgs

train_args = TrainingArgs(
    model_name="openai/gpt-oss-20b",
    # osft_memory_efficient_init=True,  # DEPRECATED: No longer needed, automatic in distributed mode
    train_dtype="bfloat16",
    wandb=True,  # Enable wandb logging
    validation_split=0.05,  # 5% validation split
    validation_frequency=100,  # Validate every 100 steps
    ...  # other training arguments
)

run_training(torch_args, train_args)
```

### Upgrade Notes
- v0.2.0: No breaking API changes. Primary focus on GPT-OSS 20B model support (120B variant potentially supported but not extensively tested). WandB logging requires `wandb>=0.16`.

### Contributors
- @NikhilNayak-debug 
- @Maxusmusti 
- @RobotSail

### Links
- [Full Changelog v0.1.1...v0.2.0](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer/compare/v0.1.1...v0.2.0)
