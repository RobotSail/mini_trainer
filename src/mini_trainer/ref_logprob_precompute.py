"""Precompute reference model log probabilities for KL divergence tracking."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import datasets
from tqdm import tqdm

from mini_trainer.utils import log_rank_0


@torch.no_grad()
def precompute_reference_logprobs(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 4,
    device: str = "cuda:0",
) -> str:
    """
    Compute reference model logprobs for each token in the dataset.

    Args:
        model_path: Path to reference model
        data_path: Input JSONL with input_ids and labels
        output_path: Output JSONL with added ref_logprobs field
        batch_size: Batch size for inference
        device: Device for model inference

    Returns:
        Path to output file with ref_logprobs
    """
    log_rank_0(f"Loading reference model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    log_rank_0(f"Loading dataset from {data_path}")
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")

    def compute_logprobs_batch(examples):
        # Pad sequences to same length within batch
        input_ids_list = [torch.tensor(ids) for ids in examples["input_ids"]]
        max_len = max(len(ids) for ids in input_ids_list)

        input_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids_list):
            input_ids[i, : len(ids)] = ids

        input_ids = input_ids.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits.float()  # (B, T, V)

        # Compute log probs for each position
        # For position i, logits[i] predicts token at position i+1
        # So we gather using input_ids shifted by 1
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Gather the logprob of the actual next token
        # shift_labels[i] = input_ids[i+1]
        shift_labels = input_ids[:, 1:]  # (B, T-1)
        shift_logprobs = log_probs[:, :-1, :]  # (B, T-1, V)

        gathered = shift_logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
            -1
        )  # (B, T-1)

        # Pad to original length (position 0 has no ref logprob, use 0)
        ref_logprobs = F.pad(gathered, (1, 0), value=0.0)  # (B, T)

        # Convert back to lists, trimming to original lengths
        result = []
        for i, ids in enumerate(input_ids_list):
            result.append(ref_logprobs[i, : len(ids)].cpu().tolist())

        return {"ref_logprobs": result}

    log_rank_0("Computing reference logprobs...")
    dataset = dataset.map(
        compute_logprobs_batch,
        batched=True,
        batch_size=batch_size,
        desc="Computing ref logprobs",
    )

    log_rank_0(f"Saving to {output_path}")
    dataset.to_json(output_path)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return output_path


def dataset_has_ref_logprobs(data_path: str) -> bool:
    """Check if dataset already has ref_logprobs field."""
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    return "ref_logprobs" in dataset.column_names
