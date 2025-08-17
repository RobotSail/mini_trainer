#!/usr/bin/env bash
# ============================================================================
# Simple Training Launcher
# -----------------------------------------------------------------------------
# This script performs **one** straightforward workflow:
#   1. (Optionally) preprocess a raw JSONL dataset with `data_process.py`
#   2. Fine-tune a model with `mini_trainer/train.py`
#
# The goal is to expose a **minimal** surface for experimentation while hiding
# the complexity found in `incremental-train.sh`.
#
# Example:
#   ./minified-train.sh \
#       --data-path /path/to/raw.jsonl \
#       --model meta-llama/Llama-3.1-8B-Instruct \
#       --output-dir /path/to/checkpoints \
#       --epochs 3 --batch-size 8 --learning-rate 1e-5
# ============================================================================

set -eo pipefail

# ----------------------------------------
# Default hyper-parameters / paths
# ----------------------------------------
PYTHON=${PYTHON:-python}
MAX_SEQ_LEN="8196"
BATCH_SIZE=128
LEARNING_RATE=''
EPOCHS=2
SEED=67
NPROC_PER_NODE=8
MAX_TOKENS_PER_GPU="45000"
ORTHOGONAL=0
RANK_RATIO="0.5"
UPCAST_DTYPE="float32"
SKIP_PROCESS=0
OPTIMIZER='adamw'
MUON_MOMENTUM=0.95
ADAMW_LEARNING_RATE="2e-5"
MUON_LEARNING_RATE="1.25e-4"
WANDB_PROJECT="quality-muon"
WANDB_RUN_NAME=""
WANDB_ENTITY=""

# Resolve helper script paths relative to this file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DATA_PROCESSING="${SCRIPT_DIR}/../training/.venv/bin/python"
DEFAULT_DATA_PROCESS_SCRIPT="${SCRIPT_DIR}/../training/src/instructlab/training/data_process.py"
DEFAULT_TRAINER_SCRIPT="${SCRIPT_DIR}/train.py"
DATA_PROCESS_SCRIPT="${DATA_PROCESS_SCRIPT:-${DEFAULT_DATA_PROCESS_SCRIPT}}"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-${DEFAULT_TRAINER_SCRIPT}}"

# Working directory for processed dataset
DATA_OUTPUT_PATH="/dev/shm/minified-train-ds"

# ----------------------------------------
# Usage helper
# ----------------------------------------
function show_usage() {
    cat << EOF
Usage: $0 --data-path PATH --model MODEL --output-dir DIR [OPTIONS]

Required arguments:
  --data-path PATH         Raw dataset (.jsonl) to preprocess & train on
  --model MODEL            Base model name or local path
  --output-dir DIR         Where to save training checkpoints / artifacts

Optional arguments:
  --epochs N               Number of epochs (default: ${EPOCHS})
  --batch-size N           Global batch size per device (default: ${BATCH_SIZE})
  --learning-rate LR       Learning rate (default: ${LEARNING_RATE})
  --max-seq-len N          Max sequence length for tokenization (default: ${MAX_SEQ_LEN})
  --max-tokens-per-gpu N   Tokens per GPU (default: ${MAX_TOKENS_PER_GPU})
  --nproc N                Number of GPUs / processes (default: ${NPROC_PER_NODE})
  --seed N                 RNG seed (default: ${SEED})
  --orthogonal             Enable orthogonal-subspace fine-tuning
  --rank-ratio R           OSFT rank ratio (default: ${RANK_RATIO})
  --upcast-dtype DT        OSFT upcast dtype (default: ${UPCAST_DTYPE})
  --skip-process           Assume dataset already tokenised; skip preprocessing
  --python PATH            Python interpreter to use (default: "${PYTHON}")
  --optimizer OPT          Optimizer to use (default: ${OPTIMIZER})
  --muon-momentum M        Momentum for Muon optimizer (default: ${MUON_MOMENTUM})
  --adamw-learning-rate LR Learning rate for AdamW optimizer (default: ${ADAMW_LEARNING_RATE})
  --wandb-project PROJECT  Project name for Weights & Biases logging
  --wandb-run-name NAME    Run name for W&B (optional)
  --wandb-entity ENTITY    W&B entity/team name (optional)
  --help                   Show this message
EOF
}

# ----------------------------------------
# Parse CLI
# ----------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"; shift 2;;
        --model)
            MODEL="$2"; shift 2;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2;;
        --epochs)
            EPOCHS="$2"; shift 2;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2;;
        --learning-rate)
            LEARNING_RATE="$2"; shift 2;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"; shift 2;;
        --max-tokens-per-gpu)
            MAX_TOKENS_PER_GPU="$2"; shift 2;;
        --nproc)
            NPROC_PER_NODE="$2"; shift 2;;
        --seed)
            SEED="$2"; shift 2;;
        --orthogonal)
            ORTHOGONAL=1; shift;;
        --rank-ratio)
            RANK_RATIO="$2"; shift 2;;
        --upcast-dtype)
            UPCAST_DTYPE="$2"; shift 2;;
        --skip-process)
            SKIP_PROCESS=1; shift;;
        --python)
            PYTHON="$2"; shift 2;;
        --optimizer)
            OPTIMIZER="$2"; shift 2;;
        --muon-momentum)
            MUON_MOMENTUM="$2"; shift 2;;
        --adamw-learning-rate)
            ADAMW_LEARNING_RATE="$2"; shift 2;;
        --muon-learning-rate)
            MUON_LEARNING_RATE="$2"; shift 2;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2;;
        --wandb-run-name)
            WANDB_RUN_NAME="$2"; shift 2;;
        --wandb-entity)
            WANDB_ENTITY="$2"; shift 2;;
        --help|-h)
            show_usage; exit 0;;
        *)
            echo "Error: Unknown option '$1'" >&2
            show_usage; exit 1;;
    esac
done

# ----------------------------------------
# Validate required args
# ----------------------------------------
if [[ -z "${DATA_PATH}" || -z "${MODEL}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Error: --data-path, --model and --output-dir are required." >&2
    show_usage; exit 1
fi

# ----------------------------------------
# Set default learning rate based on optimizer if not provided
# ----------------------------------------
if [[ -z "${LEARNING_RATE}" ]]; then
    if [[ "${OPTIMIZER}" == "muon" ]]; then
        LEARNING_RATE="${MUON_LEARNING_RATE}"
        echo "Using Muon learning rate: ${LEARNING_RATE}"
    else
        LEARNING_RATE="${ADAMW_LEARNING_RATE}"
        echo "Using AdamW learning rate: ${LEARNING_RATE}"
    fi
fi

# ----------------------------------------
# Pre-processing
# ----------------------------------------
mkdir -p "${DATA_OUTPUT_PATH}" "${OUTPUT_DIR}"

if [[ ${SKIP_PROCESS} -eq 0 ]]; then
    echo "\n>>> Preprocessing dataset..."
    "${PYTHON_DATA_PROCESSING}" "${DATA_PROCESS_SCRIPT}" \
        --data_path "${DATA_PATH}" \
        --data_output_path "${DATA_OUTPUT_PATH}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --model_name_or_path "${MODEL}" \
        --num_cpu_procs 24
fi

PROCESSED_JSONL="${DATA_OUTPUT_PATH}/data.jsonl"
if [[ ! -f "${PROCESSED_JSONL}" ]]; then
    echo "Error: Processed dataset not found at ${PROCESSED_JSONL}." >&2
    exit 1
fi

# ----------------------------------------
# Training
# ----------------------------------------
CMD=(torchrun --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" "${TRAINER_SCRIPT}" \
    --data-path "${PROCESSED_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name-or-path "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --max-epochs "${EPOCHS}" \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --optimizer "${OPTIMIZER}")

if [[ ${OPTIMIZER} == "muon" ]]; then
    CMD+=(--muon-momentum "${MUON_MOMENTUM}" --adamw-learning-rate "${ADAMW_LEARNING_RATE}")
fi

if [[ ${ORTHOGONAL} -eq 1 ]]; then
    CMD+=(--orthogonal-subspace-learning --osft-rank-ratio "${RANK_RATIO}" --osft-upcast-dtype "${UPCAST_DTYPE}")
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
    CMD+=(--wandb-project "${WANDB_PROJECT}")
    [[ -n "${WANDB_RUN_NAME}" ]] && CMD+=(--wandb-run-name "${WANDB_RUN_NAME}")
    [[ -n "${WANDB_ENTITY}" ]] && CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "\n>>> Launching training:"
printf '  %q ' "${CMD[@]}"; echo -e "\n"

CUDA_LAUNCH_BLOCKING=1 "${CMD[@]}"

echo -e "\n>>> Training complete! Artifacts saved to: ${OUTPUT_DIR}\n"
