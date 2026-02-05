"""
Sequence Length Experiment Script

Loads a model in distributed setting and runs forward/backward passes
to measure memory usage and throughput vs sequence length.

Single mode: Starts with max sequences at min length, then doubles sequence length
(halving number of sequences) until reaching minimum sequences. Then increases
token budget and repeats until OOM.
"""

import time
import os
import sys
import json
from pathlib import Path
from typing import Dict, List

import torch
from typer import Typer, Option

# Add parent directory to path to import from training modules
sys.path.append(str(Path(__file__).parent.parent))

from setup_model_for_training import setup_model, setup_training_components
from utils import init_distributed_environment, log_rank_0

app = Typer()


def create_uniform_batch(
    num_sequences: int, sequence_length: int, vocab_size: int = 32000
) -> Dict[str, torch.Tensor]:
    """Create a batch with uniform sequence lengths in packed format."""
    total_tokens = num_sequences * sequence_length

    # Create random input_ids and labels
    input_ids = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.long)
    labels = torch.randint(0, vocab_size, (total_tokens,), dtype=torch.long)

    # Create position_ids that reset for each sequence
    position_ids = []
    for _ in range(num_sequences):
        position_ids.extend(range(sequence_length))
    position_ids = torch.tensor(position_ids, dtype=torch.long)

    return {
        "input_ids": input_ids.unsqueeze(0),  # Add batch dimension
        "labels": labels.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
    }


def measure_iteration(
    model,
    optimizer,
    batch: Dict[str, torch.Tensor],
    num_iterations: int = 5,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run forward/backward passes and measure performance."""

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    total_tokens = batch["input_ids"].shape[1]

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Measure iterations
    start_time = time.time()

    for i in range(num_iterations):
        # Forward pass
        output = model(**batch, use_cache=False)
        loss = output.loss.float().sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    tokens_per_second = (total_tokens * num_iterations) / total_time

    return {
        "peak_memory_gb": peak_memory_gb,
        "tokens_per_second": tokens_per_second,
        "time_per_iteration": total_time / num_iterations,
        "total_time": total_time,
    }


def combined_experiment(
    model,
    optimizer,
    vocab_size: int,
    start_max_tokens: int,
    budget_increase: int,
    min_length: int,
    num_iterations: int,
    device: str = "cuda",
) -> List[Dict]:
    """Combined experiment: vary budgets and sequence configurations until OOM."""
    results = []
    max_tokens = start_max_tokens

    while True:
        log_rank_0(f"\n=== Testing budget: {max_tokens} tokens ===")

        # Find N such that max_tokens // 2^N is closest to min_length
        import math

        N = round(math.log2(max_tokens / min_length))
        initial_sequence_length = max_tokens // (2**N)

        log_rank_0(f"Found N={N}, initial sequence length: {initial_sequence_length}")

        sequence_length = initial_sequence_length
        budget_success = False  # Track if any config in this budget succeeded

        # Double sequence length each time until num_sequences = 1
        while True:
            num_sequences = max_tokens // sequence_length
            actual_tokens = num_sequences * sequence_length

            # Stop if we only have 1 sequence
            if num_sequences < 1:
                break

            log_rank_0(
                f"\nTesting {num_sequences} sequences of length {sequence_length} (total: {actual_tokens} tokens, budget: {max_tokens})"
            )

            try:
                # Create batch
                batch = create_uniform_batch(num_sequences, sequence_length, vocab_size)

                # Measure performance
                metrics = measure_iteration(
                    model, optimizer, batch, num_iterations, device
                )

                # Record result
                result = {
                    "max_tokens_budget": max_tokens,
                    "num_sequences": num_sequences,
                    "sequence_length": sequence_length,
                    "total_tokens": actual_tokens,
                    **metrics,
                    "status": "success",
                }
                results.append(result)
                budget_success = True

                log_rank_0(
                    f"  ✓ Memory: {metrics['peak_memory_gb']:.2f}GB, Tokens/sec: {metrics['tokens_per_second']:.1f}"
                )

            except torch.cuda.OutOfMemoryError:
                result = {
                    "max_tokens_budget": max_tokens,
                    "num_sequences": num_sequences,
                    "sequence_length": sequence_length,
                    "total_tokens": actual_tokens,
                    "status": "oom",
                }
                results.append(result)
                log_rank_0(f"  ✗ Out of memory!")
                torch.cuda.empty_cache()

                # If we OOM on the first config of a new budget, stop entirely
                if not budget_success:
                    log_rank_0(
                        f"\nStopping: OOM on first configuration of budget {max_tokens}"
                    )
                    return results

            except Exception as e:
                result = {
                    "max_tokens_budget": max_tokens,
                    "num_sequences": num_sequences,
                    "sequence_length": sequence_length,
                    "total_tokens": actual_tokens,
                    "status": "error",
                    "error": str(e),
                }
                results.append(result)
                log_rank_0(f"  ✗ Error: {e}")
                torch.cuda.empty_cache()

            # Double sequence length for next iteration
            sequence_length *= 2

        # Move to next budget
        max_tokens += budget_increase

        # Safety check: if budget gets unreasonably large, stop
        if max_tokens > 1000000:  # 1M tokens
            log_rank_0(f"\nStopping: Budget reached safety limit of 1M tokens")
            break

    return results


@app.command()
def main(
    model_name_or_path: str = Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Model name or path"
    ),
    start_max_tokens: int = Option(
        60000, help="Starting maximum tokens per GPU budget"
    ),
    budget_increase: int = Option(
        20000, help="Amount to increase budget each iteration"
    ),
    min_length: int = Option(512, help="Minimum sequence length"),
    num_iterations: int = Option(
        5, help="Number of forward/backward iterations per test"
    ),
    output_file: str = Option(
        "experiment_results.json", help="Output file for results"
    ),
    use_liger_kernels: bool = Option(False, help="Whether to use Liger kernels"),
    fsdp_version: str = Option("fsdp1", help="FSDP version to use: 'fsdp1' or 'fsdp2'"),
):
    """Run combined sequence length experiment."""

    # Initialize distributed environment
    init_distributed_environment()

    # Setup model and training components
    log_rank_0("Setting up model...")
    model = setup_model(
        model_name_or_path=model_name_or_path, use_liger_kernels=use_liger_kernels
    )

    vocab_size = model.config.vocab_size
    log_rank_0(f"Model vocab size: {vocab_size}")

    log_rank_0("Setting up training components...")
    model, optimizer, lr_scheduler = setup_training_components(
        model,
        learning_rate=0.001,  # Fixed dummy learning rate
        num_warmup_steps=0,
        lr_scheduler="constant_with_warmup",
        fsdp_version=fsdp_version,
    )

    device = next(model.parameters()).device
    log_rank_0(f"Model loaded on device: {device}")
    log_rank_0(f"Model dtype: {model.dtype}")

    model.train()

    # Run combined experiment
    log_rank_0(f"\nRunning combined experiment...")
    log_rank_0(f"Starting budget: {start_max_tokens} tokens")
    log_rank_0(f"Budget increase: {budget_increase} tokens")
    log_rank_0(f"Min length: {min_length}")

    results = combined_experiment(
        model=model,
        optimizer=optimizer,
        vocab_size=vocab_size,
        start_max_tokens=start_max_tokens,
        budget_increase=budget_increase,
        min_length=min_length,
        num_iterations=num_iterations,
        device=device,
    )

    # Save results
    experiment_data = {
        "model": model_name_or_path,
        "experiment_mode": "combined",
        "parameters": {
            "start_max_tokens": start_max_tokens,
            "budget_increase": budget_increase,
            "min_length": min_length,
            "num_iterations": num_iterations,
            "use_liger_kernels": use_liger_kernels,
            "fsdp_version": fsdp_version,
        },
        "device": str(device),
        "results": results,
    }

    # Only rank 0 saves results and prints summary
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(experiment_data, f, indent=2)

        print(f"\nResults saved to {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total configurations tested: {len(results)}")

        # Group by budget
        budgets = set(r["max_tokens_budget"] for r in results)
        for budget in sorted(budgets):
            budget_results = [r for r in results if r["max_tokens_budget"] == budget]
            successful = [r for r in budget_results if r["status"] == "success"]
            oom = [r for r in budget_results if r["status"] == "oom"]

            print(f"\nBudget {budget} tokens:")
            print(f"  Successful: {len(successful)}, OOM: {len(oom)}")

            if successful:
                print(
                    f"  {'Sequences':<10} {'Length':<10} {'Tokens':<12} {'Memory (GB)':<12} {'Tokens/sec':<12}"
                )
                print("  " + "-" * 60)
                for r in successful:
                    print(
                        f"  {r['num_sequences']:<10} {r['sequence_length']:<10} {r['total_tokens']:<12} "
                        f"{r['peak_memory_gb']:<12.2f} {r['tokens_per_second']:<12.1f}"
                    )

        print("=" * 80)


if __name__ == "__main__":
    app()
