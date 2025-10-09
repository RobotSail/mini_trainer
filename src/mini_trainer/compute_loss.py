import os
import json
from typing import Annotated

import torch
import torch.distributed as dist
from tqdm import tqdm
from typer import Typer, Option

from mini_trainer.batch_metrics import BatchMetrics
from mini_trainer.sampler import get_data_loader
from mini_trainer.setup_model_for_training import setup_model, wrap_fsdp2
from mini_trainer.utils import (
    init_distributed_environment,
    destroy_distributed_environment,
    log_rank_0,
    setup_logger,
)


app = Typer(
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


def evaluate_loss(model: torch.nn.Module, data_loader, device: torch.device) -> dict:
    """Compute average loss over a dataset in distributed mode.

    The function mirrors the logic used in `compute_validation_loss`, but it does
    not switch the model back to training mode.
    """
    if data_loader is None:
        return {}

    log_rank_0("Computing loss over dataset...")
    model.eval()

    val_batch_totals = BatchMetrics()
    total_val_batches = 0
    total_num_tokens = 0
    total_overall_loss = 0.0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    total_batches = len(data_loader)

    with torch.no_grad():
        data_loader.sampler.set_epoch(0)
        data_iter = iter(data_loader)

        pbar = tqdm(
            total=total_batches,
            desc="Eval",
            disable=not is_main_process,
            unit="batch",
        )

        for batch in data_iter:
            val_batch_totals.reset_batch()

            for _, mb in enumerate(batch):
                mb_num_loss_counted_tokens = mb["num_loss_counted_tokens"]
                mb_num_samples = mb["num_samples"]
                batch_num_loss_counted_tokens = mb["batch_num_loss_counted_tokens"]

                model_inputs = {
                    "input_ids": mb["input_ids"].to(device),
                    "labels": mb["labels"].to(device),
                    "position_ids": mb["position_ids"].to(device),
                }

                output = model(**model_inputs)
                loss = output.loss.float().sum()
                loss_item = loss.detach().item()

                torch.cuda.empty_cache()

                val_batch_totals.accumulate_minibatch_metrics(
                    num_loss_counted_tokens=mb_num_loss_counted_tokens,
                    num_total_tokens=mb["input_ids"].shape[1],
                    num_samples=mb_num_samples,
                    loss=loss_item,
                    loss_backward=0.0,
                    time_per_minibatch=0.0,
                )

            torch.distributed.barrier()
            val_batch_totals.reduce_batch_metrics(device)
            total_val_batches += 1
            total_overall_loss += val_batch_totals.totals["loss"]

            assert len(batch) > 0, "validation batch was empty"
            total_num_tokens += batch[0]["batch_num_loss_counted_tokens"]

            if is_main_process:
                current_loss = (
                    total_overall_loss / total_num_tokens if total_num_tokens > 0 else 0.0
                )
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                pbar.update(1)

            dist.barrier()

        if is_main_process:
            pbar.close()

    vbm = val_batch_totals.totals
    if total_val_batches > 0 and vbm.get("num_loss_counted_tokens", 0) > 0:
        avg_loss = total_overall_loss / total_num_tokens
        metrics = {
            "loss": avg_loss,
            "num_samples": vbm["num_samples"],
            "num_loss_counted_tokens": vbm["num_loss_counted_tokens"],
            "num_batches": total_val_batches,
        }
        log_rank_0(f"Eval loss: {avg_loss:.6f}")
    else:
        metrics = {}
        log_rank_0("No evaluation data processed")

    return metrics


@app.command()
def main(
    model_name_or_path: Annotated[str, Option(help="Model name or path")] = ...,
    data_path: Annotated[str, Option(help="Path to JSONL dataset with input_ids and labels")] = ...,
    batch_size: Annotated[int, Option(help="Initial batch size before dynamic splitting")] = ...,
    max_tokens_per_gpu: Annotated[int, Option(help="Max tokens per GPU per minibatch")] = ...,
    seed: Annotated[int, Option(help="Random seed")] = 67,
    use_liger_kernels: Annotated[bool, Option(help="Use Liger kernels if available")] = False,
):
    setup_logger(level="INFO")
    init_distributed_environment()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)



    model = setup_model(
        model_name_or_path=model_name_or_path,
        use_liger_kernels=use_liger_kernels,
        local_rank=local_rank,
    )

    log_rank_0("Using FSDP2 wrapper for evaluation")
    model = wrap_fsdp2(model)

    data_loader = get_data_loader(
        data_path=data_path,
        batch_size=batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        seed=seed,
    )

    metrics = evaluate_loss(model=model, data_loader=data_loader, device=device)

    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        print(json.dumps({"metrics": metrics, "model": model_name_or_path}, indent=2))

    destroy_distributed_environment()


if __name__ == "__main__":
    app()


