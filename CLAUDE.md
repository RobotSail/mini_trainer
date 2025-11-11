# Mini Trainer: OSFT & High-Performance LLM Training Framework

## Repository Overview

This repository implements **Orthogonal Subspace Fine-Tuning (OSFT)**, a sophisticated training technique designed to prevent catastrophic forgetting during continual learning. The framework is built for high-performance, multi-node distributed training of large language models.

### Core Capabilities
- **OSFT Training**: SVD-based parameter decomposition with selective training
- **Standard SFT**: Full fine-tuning without decomposition
- **GPT-OSS Support**: Special handling for OpenAI's quantized open-weight models
- **Multi-Node Distributed Training**: FSDP2 sharding across GPU clusters
- **Memory-Efficient Operations**: Factorized computation, aggressive caching, padding-free training

### Design Principles
- **Modularity**: Clean separation between OSFT logic, data loading, training loop, and model-specific code
- **Type Safety**: Extensive use of dataclasses, protocols, and type hints for robust configuration
- **Performance**: Numba JIT compilation, GPU-optimized operations, minimal Python overhead
- **Memory Efficiency**: Factorized forward passes, incremental checkpoint conversion, strategic caching
- **Flexibility**: Support for 10+ model architectures and multiple training modes

---

## Core Concepts

### 1. OSFT (Orthogonal Subspace Fine-Tuning)

OSFT prevents catastrophic forgetting by decomposing weight matrices into two orthogonal subspaces:

**High-Rank Subspace (Frozen)**:
- Contains the most important singular values (top-k)
- Preserves critical prior knowledge
- Never updated during training

**Low-Rank Subspace (Trainable)**:
- Contains remaining singular values
- Learns new task-specific information
- Updated with orthogonally-projected gradients

#### How It Works

1. **SVD Decomposition**:
   ```
   W = U·S·Vᵀ
   W_high = U[:, :k] @ S[:k, :k] @ V[:, :k].T  (frozen)
   W_low = U[:, k:] @ S[k:, k:] @ V[:, k:].T   (trainable)
   ```
   - Parameter `osft_unfreeze_rank_ratio` controls k (e.g., 0.25 = train 25% least important)
   - Uses high-precision computation (`float32`) for numerical stability
   - Supports memory-efficient output dtypes (`bfloat16`) for storage

2. **Gradient Projection**:
   - Ensures low-rank gradients remain orthogonal to high-rank subspace
   - Prevents gradient updates from interfering with preserved knowledge
   - Formula: `dU_low = dU_low - U_high @ (U_high.T @ dU_low)`
   - Applied during backward pass via hooks

3. **Factorized Forward Pass**:
   - Avoids reconstructing full weight matrix (massive memory savings)
   - Computation: `y = (x @ V_high.T) @ (S_high @ U_high.T) + (x @ V_low.T) @ (S_low @ U_low.T)`
   - Uses only rank-sized intermediate tensors
   - Dynamically replaces linear layer forward methods

4. **Dynamic Model Wrapping**:
   - Any HuggingFace model can be wrapped with OSFT capabilities
   - Uses multiple inheritance to extend model classes
   - Pattern: `OSFTModel = create_osft_model_class(BaseModel)`
   - No modification to original model code required

#### Model-Specific Parameter Patterns

OSFT targets specific linear layers based on architecture:
- **Llama/Mistral/Qwen**: Attention projections (q/k/v/o) + MLP projections (gate/up/down)
- **GPT-2**: Combined attention (`c_attn`, `c_proj`) + MLP (`c_fc`)
- **Phi-3/Phi-4**: Fused projections (`qkv_proj`, `gate_up_proj`, `down_proj`)
- **GPT-OSS**: Only attention layers (experts excluded due to MoE complexity)

#### Distributed Initialization

For multi-GPU setups:
- SVD work partitioned across all ranks (round-robin assignment)
- Each rank computes subset of SVD decompositions
- Results broadcast via NCCL with device-aware handling
- Significantly faster than single-rank initialization (near-linear speedup)

#### Checkpoint Conversion

When saving OSFT models:
- SVD components reconstructed into dense weights for HuggingFace compatibility
- Processed incrementally to minimize peak memory (one parameter at a time)
- GPU cache cleared periodically (configurable interval)
- Takes ~5 minutes for large models but necessary for standard loading

---

### 2. GPT-OSS Support (OpenAI's Open-Weight Models)

GPT-OSS models (20B and 120B) use a unique quantization format requiring special handling:

#### MXFP4 Quantization

**Format Specification**:
- **Block size**: 32 elements (last dimension grouping)
- **Precision**: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
- **Scale**: Per-block power-of-2 (stored as int8 exponent)
- **Packing**: Interleaved nibbles in uint8 array
- **Value range**: 16 discrete values from -6.0 to 6.0

**Critical Implementation Details**:
- **Signed zero handling**: Tie-breaking prefers index 8 over index 0 for `-0.0`
- **Vectorized processing**: Quantizes entire expert tensor simultaneously (100x+ speedup)
- **Memory management**: GPU computation → immediate CPU transfer → cache clearing
- **Fidelity**: Bit-exact matching of OpenAI's reference format

#### Training Workflow

1. **Initialization**: Dequantize MXFP4 → `bfloat16`/`float32`
2. **Training**: Standard gradient updates on dequantized weights
3. **Checkpointing**: Re-quantize to MXFP4 for storage

#### Router Parameter Handling

**Critical requirement**: MoE router parameters must be frozen BEFORE FSDP2 setup
- Prevents gradient uniformity issues in distributed training
- Essential for training stability
- Applied automatically when GPT-OSS models detected

---

### 3. Distributed Training with FSDP2

#### Sharding Strategy

**Per-Block Sharding**:
- Each transformer block wrapped independently
- Activation checkpointing applied per block
- 1D device mesh spanning all ranks
- Mixed precision: `param_dtype=bfloat16`, `reduce_dtype=float32`

**Reshard Optimization**:
- Most blocks: `reshard_after_forward=True` (memory-optimal)
- Final block: `reshard_after_forward=False` (communication-optimal)
- Balances memory usage vs network overhead

#### OSFT + FSDP2 Interaction

**Memory-Efficient Initialization Mode**:
- OSFT models kept on CPU during SVD decomposition
- FSDP2 handles GPU placement during sharding
- Prevents OOM errors for large models
- Enabled via `osft_memory_efficient_init=True`

**Standard Mode**:
- SVD computed directly on GPU
- Faster but requires sufficient GPU memory
- Suitable for smaller models

---

### 4. Data Loading & Batching

#### Data Format

Expected JSONL format:
```json
{"input_ids": [1, 2, 3, ...], "labels": [1, 2, 3, ...], "len": 128, "num_loss_counted_tokens": 100}
```

- `labels`: Use `-100` for tokens to ignore in loss computation
- `len`: Sequence length (computed if missing)
- `num_loss_counted_tokens`: Number of loss-contributing tokens (computed if missing)

#### Multi-Stage Batching Pipeline

**Stage 1: Initial Batching**
- Groups `batch_size` samples from dataset
- Example: `batch_size=128` → initial batch of 128 samples

**Stage 2: Dynamic Minibatching**
- Splits initial batch into minibatches based on token count
- Constraint: Each rank must not exceed `max_tokens_per_gpu`
- Uses LPT (Longest Processing Time) algorithm for load balancing
- Number of minibatches varies per batch (adaptive to sequence lengths)

**Stage 3: Gradient Accumulation**
- Each minibatch = one gradient accumulation step
- Loss normalized across ranks and tokens: `loss * world_size / batch_num_loss_counted_tokens`
- Ensures consistent gradient scale regardless of batching decisions

#### Packing Algorithm (LPT with Min-Heap)

**Algorithm**:
1. Sort sequences by length (descending)
2. Maintain min-heap of current rank loads
3. Assign each sequence to least-loaded rank
4. Binary search for maximum sequences per minibatch
5. Pad with `-1` indices for ranks with no assignment

**Properties**:
- Time complexity: O(n log n log k) where n=sequences, k=ranks
- Near-optimal load balancing across GPUs
- JIT-compiled with Numba (10-100x speedup over pure Python)

**Example**:
```
Sequences: [2048, 1024, 1024, 512, 512]
max_tokens_per_rank: 2560
num_ranks: 2

Minibatch 1:
  Rank 0: [2048, 512] (2560 tokens)
  Rank 1: [1024, 1024] (2048 tokens)
```

#### Padding-Free Training

**Packed Sequence Format**:
```python
{
    "input_ids": [seq1..., seq2..., seq3...],  # Concatenated
    "labels": [seq1..., seq2..., seq3...],
    "position_ids": [0,1,2,...,0,1,2,...,0,1,2,...]  # Reset per sequence
}
```

**Benefits**:
- Zero padding tokens → maximum compute efficiency
- Flash Attention uses `position_ids` for sequence boundary detection
- Block-diagonal attention prevents cross-sequence leakage
- Significant throughput improvement vs padded batches

---

### 5. Flash Attention Integration

**Version Selection**:
- Standard models: `flash_attention_2` (HuggingFace default)
- GPT-OSS: `kernels-community/vllm-flash-attn3` (optimized for MoE)
- Testing mode: Falls back to `eager` attention

**Requirements**:
- Inputs must be `bfloat16` precision
- Position IDs required for packed sequences
- Automatically configured during model setup

---

### 6. Training Modes

Four distinct training modes controlled by `training_mode`:

1. **EPOCH**: Train for fixed number of epochs (`max_epochs`)
2. **STEP**: Train for fixed number of gradient steps (`max_steps`)
3. **TOKEN**: Train until processing fixed number of loss-counted tokens (`max_tokens`)
4. **INFINITE**: Train indefinitely until manual interruption

All modes support:
- Validation splits with configurable frequency
- Multiple checkpoint types (epoch, sample, final, best-val-loss)
- Progress tracking and logging

---

## Code Organization

The codebase is organized into modular components:

### Core Training Components
- **Training Loop**: Main training logic, validation, checkpointing, progress tracking
- **API Interface**: Programmatic entry point for launching training jobs
- **Configuration Types**: Type-safe dataclasses for all training parameters

### OSFT Implementation
- **SVD Decomposition**: High-precision decomposition with configurable output types
- **Gradient Projection**: Orthogonalization hooks for backward pass
- **Model Wrapping**: Dynamic class creation for any HuggingFace model
- **Factorized Computation**: Memory-efficient forward pass implementation
- **Distributed Initialization**: Multi-rank SVD computation and broadcasting
- **Checkpoint Conversion**: Incremental reconstruction of dense weights

### GPT-OSS Implementation
- **Quantization**: MXFP4 encoding with tie-breaking and vectorization
- **Dequantization**: Conversion to training-compatible formats
- **Router Management**: Freezing logic for MoE stability

### Data & Batching
- **Data Loading**: JSONL parsing, validation, preprocessing
- **Sampler**: Multi-stage batching pipeline coordination
- **Collation**: Padding-free sequence packing for Flash Attention
- **Batch Packer**: Numba-optimized LPT algorithm

### Distributed Training
- **Model Setup**: FSDP2 wrapping, Flash Attention configuration
- **Device Mesh**: Multi-node topology management
- **Gradient Scaling**: Distributed loss normalization

### Utilities
- **Logging**: Async structured JSONL + WandB + rich console output
- **Type Definitions**: Training modes, torchrun arguments, protocols
- **Data Processing Scripts**: Utilities for dataset preparation

### Testing
- **Unit Tests**: Component-level testing (OSFT, GPT-OSS, batching, logging)
- **GPU Tests**: Distributed utilities, mixed precision, Flash Attention
- **Regression Tests**: Performance benchmarks, orthogonalization verification
- **Integration Tests**: End-to-end training with small models

---

## Key Architectural Patterns

### 1. Dynamic Model Class Creation

OSFT uses runtime class generation to wrap any HuggingFace model:

```python
# Pattern used internally
BaseModelClass = AutoModelForCausalLM.from_config(config).__class__
OSFTModelClass = create_osft_model_class(BaseModelClass)
osft_model = OSFTModelClass.from_pretrained(model_name)
```

**How it works**:
- Multiple inheritance: `class ModelWithOSFT(BaseModel, OSFTMixin)`
- Replaces forward methods of targeted linear layers
- No modification to original model code
- Supports any architecture with linear layers

### 2. Factorized Linear Computation

Memory-efficient alternative to weight reconstruction:

```python
# Instead of: y = x @ (U_high @ S_high @ V_high.T + U_low @ S_low @ V_low.T)
# Compute as:
y_high = (x @ V_high.T) @ (S_high @ U_high.T)
y_low = (x @ V_low.T) @ (S_low @ U_low.T)
y = y_high + y_low
```

**Benefits**:
- Intermediate tensors are rank-sized (not full matrix)
- 10-100x memory reduction for low-rank configurations
- Essential for large models with OSFT

### 3. Incremental Checkpoint Conversion

When saving OSFT models to HuggingFace format:

```python
# Process one parameter at a time
for param_name, (u_high, s_high, v_high, u_low, s_low, v_low) in osft_params:
    # Reconstruct full weight
    weight = reconstruct_from_svd(u_high, s_high, v_high, u_low, s_low, v_low)
    state_dict[param_name] = weight

    # Clear cache every N parameters
    if i % CACHE_CLEAR_INTERVAL == 0:
        torch.cuda.empty_cache()
```

**Why needed**:
- Prevents OOM during checkpoint saving
- HuggingFace expects dense weight format
- Configurable via `OSFT_CACHE_CLEAR_INTERVAL` environment variable

### 4. Two-Stage Batching

Decouples logical batch size from GPU memory constraints:

```python
# Stage 1: Logical batch (for optimizer)
initial_batch = dataset.sample(batch_size)  # e.g., 128 samples

# Stage 2: Physical minibatches (for GPU memory)
minibatches = split_by_token_count(initial_batch, max_tokens_per_gpu)
# Results in variable number of minibatches

# Gradient accumulation over minibatches
for minibatch in minibatches:
    loss = compute_loss(minibatch)
    # Normalized across ranks and total tokens
    loss = loss * world_size / total_loss_counted_tokens
    loss.backward()
```

**Advantages**:
- Batch size independent of sequence length distribution
- Optimal GPU memory utilization
- Automatic load balancing across ranks

### 5. Distributed SVD Initialization

Work distribution for OSFT setup:

```python
# Pseudocode for distributed initialization
my_rank = get_rank()
world_size = get_world_size()

# Round-robin assignment
my_params = [p for i, p in enumerate(all_params) if i % world_size == my_rank]

# Each rank computes subset
for param in my_params:
    u, s, v = torch.svd(param.data.float())
    # Store results

# Broadcast all results
for rank in range(world_size):
    broadcast_from_rank(rank)
```

**Speedup**:
- Near-linear scaling with number of GPUs
- Critical for large models with many parameters
- Automatic coordination via NCCL

---

## Special Considerations

### OSFT-Specific

1. **Precision Management**:
   - SVD computation: Always `float32` (numerical stability)
   - Storage dtype: Configurable via `osft_output_dtype` (memory efficiency)
   - Training dtype: Inherited from FSDP2 (`bfloat16`)

2. **Memory Efficiency Modes**:
   - Standard: SVD on GPU (fast but memory-intensive)
   - Memory-efficient: SVD on CPU, FSDP2 moves to GPU (slower but always works)
   - Choose based on model size and available GPU memory

3. **Gradient Hook Timing**:
   - Hooks registered AFTER FSDP2 setup
   - Ensures gradient projection happens on sharded tensors
   - Critical for correctness in distributed setting

4. **Checkpoint Size**:
   - OSFT checkpoints store 6 tensors per parameter (U_high, S_high, V_high, U_low, S_low, V_low)
   - 2-3x larger than dense checkpoints during training
   - Converted back to dense format for final saving

### GPT-OSS-Specific

1. **Router Freezing**:
   - MUST freeze router parameters BEFORE FSDP2 setup
   - Failure causes gradient uniformity issues
   - Automatically handled when GPT-OSS model detected

2. **Quantization Fidelity**:
   - Signed zero tie-breaking essential for bit-exact matching
   - Use `signbit()` to detect `-0.0` vs `+0.0`
   - Critical for checkpoint compatibility with OpenAI's format

3. **Expert Exclusion from OSFT**:
   - MoE experts excluded from OSFT decomposition
   - Only attention layers decomposed
   - Avoids complexity of SVD on sparse activations

4. **Flash Attention Version**:
   - Must use `kernels-community/vllm-flash-attn3` for GPT-OSS
   - Standard Flash Attention 2 not optimized for MoE

### Distributed Training

1. **FSDP2 + OSFT Initialization Order**:
   ```
   Correct: SVD decomposition → FSDP2 wrapping → Hook registration
   Incorrect: FSDP2 wrapping → SVD decomposition (causes sharding issues)
   ```

2. **Loss Normalization**:
   - Must account for both distributed ranks and variable minibatch sizes
   - Formula: `loss * world_size / batch_num_loss_counted_tokens`
   - Ensures consistent gradient scale

3. **Checkpoint Aggregation**:
   - FSDP2 automatically handles weight gathering from all ranks
   - OSFT conversion happens AFTER gathering (on rank 0)
   - Saves network bandwidth by avoiding redundant conversion

### Performance Optimization

1. **Numba JIT Warmup**:
   - First call to batch packer triggers compilation (~1 second)
   - Subsequent calls near-instant
   - Worth the warmup cost for multi-epoch training

2. **Flash Attention Requirements**:
   - Inputs must be contiguous and `bfloat16`
   - Position IDs required for packed sequences
   - Falls back to eager if requirements not met (with warning)

3. **Gradient Checkpointing Trade-off**:
   - Saves memory by recomputing activations
   - Increases training time by ~20%
   - Essential for large models, optional for small ones

---

## Development Guidelines

### Adding Support for New Architectures

1. **Identify target parameters**: Find linear layer names for attention and MLP
2. **Add pattern to OSFT utilities**: Register layer name pattern for architecture
3. **Test with small model**: Verify forward pass correctness and gradient flow
4. **Benchmark memory**: Compare OSFT vs standard SFT memory usage

### Testing Conventions

1. **Unit tests**: Pure CPU tests for algorithm correctness
2. **GPU tests**: Single-GPU tests for distributed utilities and precision
3. **Regression tests**: Multi-run benchmarks for performance verification
4. **Integration tests**: End-to-end training with small models (GPT-2, TinyLlama)

### Configuration Best Practices

1. **OSFT rank ratio**: Start with 0.25 (train 25% of parameters)
   - Lower = more preservation, less adaptation
   - Higher = less preservation, more adaptation

2. **Batch sizing**:
   - `batch_size`: Logical batch size (optimizer perspective)
   - `max_tokens_per_gpu`: Physical constraint (GPU memory)
   - Rule of thumb: `max_tokens_per_gpu ≈ 100K-150K` for 80GB A100

3. **Learning rate**:
   - OSFT: Typically 2-5x higher than full fine-tuning (fewer trainable params)
   - Standard SFT: Standard values (1e-5 to 5e-5)

4. **Validation frequency**:
   - More frequent for small datasets (every 50-100 steps)
   - Less frequent for large datasets (every 500-1000 steps)

### Common Patterns

**Loading OSFT checkpoint for inference**:
- OSFT training checkpoints contain SVD components
- Final checkpoint automatically converted to dense format
- Use final checkpoint for deployment (HuggingFace compatible)

**Extending to new quantization formats**:
- Follow GPT-OSS pattern: separate module for quantization logic
- Implement dequantization for training, re-quantization for saving
- Ensure bit-exact fidelity with reference implementation

**Custom data preprocessing**:
- Create JSONL with `input_ids`, `labels`, `len`, `num_loss_counted_tokens`
- Use `-100` in labels for tokens to ignore
- Data processing scripts available in `scripts/` directory

---

## Entry Points

### Command-Line Training

```bash
torchrun --nnodes=1 --nproc-per-node=8 -m mini_trainer.train \
    --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./data.jsonl \
    --batch-size 128 \
    --max-tokens-per-gpu 128000 \
    --learning-rate 5e-6 \
    --osft \
    --osft-unfreeze-rank-ratio 0.25
```

### Programmatic API

```python
from mini_trainer import run_training, TorchrunArgs, TrainingArgs, TrainingMode

torch_args = TorchrunArgs(nnodes=2, nproc_per_node=8)
train_args = TrainingArgs(
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data.jsonl",
    batch_size=128,
    max_tokens_per_gpu=128000,
    learning_rate=5e-6,
    output_dir="./checkpoints",
    osft=True,
    osft_unfreeze_rank_ratio=0.25,
    osft_memory_efficient_init=True,
    training_mode=TrainingMode.STEP,
    max_steps=10000,
)

run_training(torch_args, train_args)
```

---

## Key Dependencies

- **PyTorch**: ≥2.6 (FSDP2 support)
- **Transformers**: ≥4.55.0 (GPT-OSS support)
- **Flash Attention**: ≥2.8.2
- **Numba**: ≥0.62.0 (JIT compilation)
- **Liger Kernel**: ≥0.5.10 (optional, memory-efficient kernels)
- **WandB**: Optional experiment tracking

---

## For Future Agents

When working with this codebase:

1. **OSFT modifications**: Look for SVD decomposition logic, gradient projection hooks, and factorized computation
2. **Data loading changes**: Focus on batching pipeline, collation logic, and packing algorithms
3. **New model support**: Add parameter patterns, test with small model first
4. **Performance issues**: Check Numba JIT compilation, Flash Attention configuration, batch packer efficiency
5. **Memory issues**: Verify OSFT memory-efficient init, FSDP2 sharding, checkpoint conversion strategy
6. **GPT-OSS bugs**: Validate quantization fidelity, router freezing, Flash Attention version

**Architecture Note**: This codebase represents production-grade research infrastructure. The OSFT implementation is particularly sophisticated, with careful attention to numerical stability, memory efficiency, and distributed correctness. When in doubt, consult existing tests for usage patterns.

