"""
GSM8K task-specific evaluation for SFT training.

This module provides pass@1 evaluation on GSM8K-format math datasets
during training. It evaluates the model's ability to generate correct
numerical answers within <answer>...</answer> tags.
"""

import re
import torch
import torch.distributed as dist
import datasets
from typing import Optional
from tqdm import tqdm

# Regex pattern to match <answer>...</answer> tags (case-insensitive)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> float:
    """
    Parse a string into a float, handling common formats from GSM8K answers.

    Handles:
    - Whitespace (leading/trailing/internal)
    - Percentage signs (42% -> 42.0)
    - Currency symbols ($100, EUR50, etc.)
    - Comma separators (1,000,000 -> 1000000)
    - Negative numbers (-42, negative prefix)
    - Decimal numbers (3.14)

    Returns:
        float: The parsed number

    Raises:
        ValueError: If no valid number can be parsed
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {text}")

    # Strip whitespace
    text = text.strip()

    # Remove currency symbols ($, EUR, GBP, JPY, etc.)
    text = re.sub(r"[$\u20AC\u00A3\u00A5\u20B9]", "", text)

    # Remove percentage sign (keep the number)
    text = text.replace("%", "")

    # Remove commas (thousand separators)
    text = text.replace(",", "")

    # Strip remaining whitespace after removals
    text = text.strip()

    # Check for digits
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in answer: {text}")

    # Extract the numeric portion (handles cases like "42 dollars" -> "42")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        raise ValueError(f"Could not extract number from: {text}")

    return float(match.group())


def grade_response(
    response: str, expected_answer: float, tolerance: float = 1e-6
) -> dict:
    """
    Grade a model response against the expected answer.

    Uses the LAST <answer>...</answer> tag if multiple are present
    (final answer after chain-of-thought reasoning).

    Args:
        response: Model's generated response
        expected_answer: Expected numerical answer
        tolerance: Tolerance for floating point comparison

    Returns:
        dict with keys:
            - is_parsable: bool - whether answer format was valid
            - is_correct: bool - whether answer matched expected
            - parsed_answer: float | None - the parsed answer if parsable
    """
    matches = ANSWER_PATTERN.findall(response)

    if not matches:
        return {"is_parsable": False, "is_correct": False, "parsed_answer": None}

    # Take the LAST answer (final answer after reasoning)
    last_match = matches[-1]

    try:
        parsed_answer = parse_number(last_match)
        is_correct = abs(parsed_answer - expected_answer) < tolerance
        return {
            "is_parsable": True,
            "is_correct": is_correct,
            "parsed_answer": parsed_answer,
        }
    except ValueError:
        return {"is_parsable": False, "is_correct": False, "parsed_answer": None}


def load_gsm8k_eval_dataset(
    data_path: str,
    max_samples: Optional[int] = None,
) -> datasets.Dataset:
    """
    Load GSM8K evaluation dataset.

    Expected format (JSONL):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "answer": 42.0
    }

    Args:
        data_path: Path to JSONL file
        max_samples: Maximum samples to load (None = all)

    Returns:
        HuggingFace Dataset
    """
    dataset = datasets.load_dataset("json", data_files=data_path, split="train")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    return dataset


@torch.no_grad()
def evaluate_gsm8k(
    model: torch.nn.Module,
    tokenizer,
    eval_dataset: datasets.Dataset,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """
    Evaluate model on GSM8K dataset using pass@1 (single response per prompt).

    This function handles FSDP2 distributed models by:
    1. Setting model to eval mode
    2. Clearing CUDA cache before/after generation
    3. Using model.generate() which works with FSDP2
    4. Aggregating metrics across all ranks with all_reduce

    Args:
        model: The model to evaluate (may be FSDP2-wrapped)
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset with 'messages' and 'answer' fields
        device: Device for generation
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)

    Returns:
        dict with evaluation metrics:
            - gsm8k_accuracy: float - percentage of correct answers
            - gsm8k_parsable_rate: float - percentage with valid format
            - gsm8k_num_samples: int - number of samples evaluated
            - gsm8k_correct: int - number of correct answers
            - gsm8k_parsable: int - number of parsable answers
    """
    from mini_trainer.utils import log_rank_0

    log_rank_0("Running GSM8K evaluation...")

    # Clear cache before evaluation
    torch.cuda.empty_cache()

    was_training = model.training
    model.eval()

    # Prepare generation config
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0

    correct_count = 0
    parsable_count = 0
    total_count = len(eval_dataset)

    # Get rank info for distributed training
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # In distributed mode, each rank evaluates a subset of samples
    if is_distributed:
        # Split samples across ranks
        samples_per_rank = (total_count + world_size - 1) // world_size
        start_idx = rank * samples_per_rank
        end_idx = min(start_idx + samples_per_rank, total_count)
        local_indices = list(range(start_idx, end_idx))
    else:
        local_indices = list(range(total_count))

    # Process samples assigned to this rank
    iterator = tqdm(
        local_indices,
        desc=f"GSM8K eval (rank {rank})",
        disable=rank != 0,  # Only show progress on rank 0
    )

    for i in iterator:
        sample = eval_dataset[i]
        messages = sample["messages"]

        # Strip assistant response if present (SFT format has assistant, GRPO format doesn't)
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]

        expected_answer = float(sample["answer"])

        # Tokenize prompt (system + user only)
        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        # Generate response
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        # Decode only the new tokens
        new_tokens = outputs[0, input_ids.shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Grade the response
        result = grade_response(response, expected_answer)

        if result["is_parsable"]:
            parsable_count += 1
        if result["is_correct"]:
            correct_count += 1

        # Clear cache periodically to prevent OOM
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Aggregate metrics across all ranks if distributed
    if is_distributed:
        # Create tensors for reduction
        correct_tensor = torch.tensor(
            [correct_count], dtype=torch.float32, device=device
        )
        parsable_tensor = torch.tensor(
            [parsable_count], dtype=torch.float32, device=device
        )

        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(parsable_tensor, op=dist.ReduceOp.SUM)

        correct_count = int(correct_tensor.item())
        parsable_count = int(parsable_tensor.item())

    # Calculate metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    parsable_rate = parsable_count / total_count if total_count > 0 else 0.0

    metrics = {
        "gsm8k_accuracy": accuracy,
        "gsm8k_parsable_rate": parsable_rate,
        "gsm8k_num_samples": total_count,
        "gsm8k_correct": correct_count,
        "gsm8k_parsable": parsable_count,
    }

    log_rank_0(f"GSM8K Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    log_rank_0(
        f"GSM8K Parsable Rate: {parsable_rate:.4f} ({parsable_count}/{total_count})"
    )

    # Clear cache after evaluation
    torch.cuda.empty_cache()

    # Restore training mode if it was training
    if was_training:
        model.train()

    return metrics


def evaluate_gsm8k_vllm(
    checkpoint_path: str,
    eval_dataset: datasets.Dataset,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.8,
    eval_device: int = 7,  # Dedicated GPU for evaluation (not used by training)
) -> dict:
    """
    Evaluate a saved checkpoint on GSM8K using vLLM for fast batched inference.

    This function runs vLLM in a completely separate subprocess to avoid
    conflicts with FSDP's NCCL process group.

    Args:
        checkpoint_path: Path to saved HuggingFace-format checkpoint
        eval_dataset: Dataset with 'messages' and 'answer' fields
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache

    Returns:
        dict with evaluation metrics
    """
    import os
    import json
    import subprocess
    import tempfile
    from mini_trainer.utils import log_rank_0

    log_rank_0(f"Running GSM8K evaluation with vLLM on checkpoint: {checkpoint_path}")

    # Save dataset to temp file for subprocess
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_data_path = f.name
        for sample in eval_dataset:
            json.dump({"messages": sample["messages"], "answer": sample["answer"]}, f)
            f.write('\n')

    # Create temp file for results
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_results_path = f.name

    # Python script to run in subprocess - write to a script file to avoid arg length issues
    eval_script = f'''
import os
import sys
import json

# Force single GPU mode before any CUDA imports - use dedicated eval GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "{eval_device}"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def parse_number(text):
    if not text or not isinstance(text, str):
        raise ValueError(f"Empty or invalid input: {{text}}")
    text = text.strip()
    text = re.sub(r"[$\\u20AC\\u00A3\\u00A5\\u20B9]", "", text)
    text = text.replace("%", "").replace(",", "").strip()
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in answer: {{text}}")
    match = re.search(r"-?\\d+\\.?\\d*", text)
    if not match:
        raise ValueError(f"Could not extract number from: {{text}}")
    return float(match.group())

def grade_response(response, expected_answer, tolerance=1e-6):
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return {{"is_parsable": False, "is_correct": False}}
    last_match = matches[-1]
    try:
        parsed_answer = parse_number(last_match)
        is_correct = abs(parsed_answer - expected_answer) < tolerance
        return {{"is_parsable": True, "is_correct": is_correct}}
    except ValueError:
        return {{"is_parsable": False, "is_correct": False}}

# Load data
data = []
with open("{temp_data_path}", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("{checkpoint_path}")
llm = LLM(
    model="{checkpoint_path}",
    tensor_parallel_size=1,
    gpu_memory_utilization={gpu_memory_utilization},
    dtype="bfloat16",
    enforce_eager=True,
)

sampling_params = SamplingParams(
    max_tokens={max_new_tokens},
    temperature={temperature if temperature > 0 else 0},
    top_p=1.0,
)

# Prepare prompts
prompts = []
expected_answers = []
for sample in data:
    messages = sample["messages"]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts.append(prompt)
    expected_answers.append(float(sample["answer"]))

# Generate
outputs = llm.generate(prompts, sampling_params)

# Grade
correct_count = 0
parsable_count = 0
for output, expected in zip(outputs, expected_answers):
    response = output.outputs[0].text
    result = grade_response(response, expected)
    if result["is_parsable"]:
        parsable_count += 1
    if result["is_correct"]:
        correct_count += 1

# Save results
total_count = len(data)
results = {{
    "gsm8k_accuracy": correct_count / total_count if total_count > 0 else 0.0,
    "gsm8k_parsable_rate": parsable_count / total_count if total_count > 0 else 0.0,
    "gsm8k_num_samples": total_count,
    "gsm8k_correct": correct_count,
    "gsm8k_parsable": parsable_count,
}}
with open("{temp_results_path}", "w") as f:
    json.dump(results, f)

print(f"GSM8K Accuracy: {{results['gsm8k_accuracy']:.4f}} ({{correct_count}}/{{total_count}})")
print(f"GSM8K Parsable Rate: {{results['gsm8k_parsable_rate']:.4f}} ({{parsable_count}}/{{total_count}})")
'''

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_script_path = f.name
        f.write(eval_script)

    try:
        # Run in subprocess with clean environment - remove all distributed training vars
        log_rank_0("Spawning vLLM evaluation subprocess...")
        clean_env = {k: v for k, v in os.environ.items() if k not in {
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID",
            "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "TORCHELASTIC_ERROR_FILE", "OMP_NUM_THREADS",
        }}
        clean_env["CUDA_VISIBLE_DEVICES"] = str(eval_device)
        clean_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True,
            text=True,
            env=clean_env,
            start_new_session=True,  # Fully detach from parent process group
        )

        if result.returncode != 0:
            log_rank_0(f"vLLM subprocess failed: {result.stderr}")
            return None

        # Print subprocess output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                log_rank_0(line)

        # Load results
        with open(temp_results_path, 'r') as f:
            metrics = json.load(f)

        return metrics

    finally:
        # Cleanup temp files
        for path in [temp_data_path, temp_results_path, temp_script_path]:
            if os.path.exists(path):
                os.unlink(path)
