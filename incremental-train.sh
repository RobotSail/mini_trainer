#!/bin/bash

# stops early in case of problem
set -eo pipefail

# ================================================================================
# SET VARIABLES
# ================================================================================
# BASE_MODEL='qwen/Qwen2.5-7B-Instruct'
BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'

# datasets 
FULL_DATASET='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/train_p07.jsonl'
DATASET_CHUNK_1='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_0.jsonl'
DATASET_CHUNK_2='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_1.jsonl'
DATASET_CHUNK_3='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_2.jsonl'

# outputs 
EXPERIMENT_DIR="/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/${BASE_MODEL}"
CHUNKED_OUTPUT_DIR="${EXPERIMENT_DIR}/output/chunked-dataset-0"
STANDARD_OUTPUT_DIR="${EXPERIMENT_DIR}/output/full-dataset-0"
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${CHUNKED_OUTPUT_DIR}"
mkdir -p "${STANDARD_OUTPUT_DIR}"

# hyperparams
LEARNING_RATE='2e-5'
EPOCHS=7
BATCH_SIZE=128
SEED=67
WARMUP_STEPS=0
MAX_TOKENS_PER_GPU='65536'
RANK_RATIO='0.5'


# data processing hyperparams
MAX_SEQ_LEN='8196'
STD_DATA_OUTPUT_PATH='/dev/shm/standard-ds'
DATASET_CHUNK_1_OUTPUT_PATH='/dev/shm/chunk-1'
DATASET_CHUNK_2_OUTPUT_PATH='/dev/shm/chunk-2'
DATASET_CHUNK_3_OUTPUT_PATH='/dev/shm/chunk-3'


# STANDARD-SPECIFIC HYPERPARAMS
STANDARD_SAVE_FREQUENCY=10856  # save per-epoch
STANDARD_MAX_STEPS=595  # there are 10856 samples in std ds, so ceil(10856/128)*7 = 595

# Script locations
DATA_PROCESS_PYTHON='/mnt/7TB-a/osilkin/training/.venv/bin/python'
DATA_PROCESS_SCRIPT='/mnt/7TB-a/osilkin/training/src/instructlab/training/data_process.py'
MINI_TRAINER_SCRIPT='/mnt/7TB-a/osilkin/mini_trainer/train.py'

# ================================================================================
# PARSE INPUT ARGUMENTS TO THE SCRIPT
# ================================================================================
# Parse arguments for --skip-process or -s
for arg in "$@"; do
    if [[ "$arg" == "--skip-process" || "$arg" == "-s" ]]; then
        SKIP_PROCESS=1
        # Remove the skip flag from the arguments so it doesn't get passed to python scripts
        set -- "${@/"$arg"}"
    fi
done

# ================================================================================
# DATASET PARSING
# ================================================================================
if [[ $SKIP_PROCESS -eq 0 ]]; then

    # first, we process the standard dataset
    "${DATA_PROCESS_PYTHON}" "${DATA_PROCESS_SCRIPT}" \
        --data_path="${FULL_DATASET}" \
        --data_output_path="${STD_DATA_OUTPUT_PATH}" \
        --max_seq_len="${MAX_SEQ_LEN}" \
        --model_name_or_path="${BASE_MODEL}" \
        --num_cpu_procs=24

    # process chunk 1 here
    "${DATA_PROCESS_PYTHON}" "${DATA_PROCESS_SCRIPT}" \
        --data_path="${DATASET_CHUNK_1}" \
        --data_output_path="${DATASET_CHUNK_1_OUTPUT_PATH}" \
        --max_seq_len="${MAX_SEQ_LEN}" \
        --model_name_or_path="${BASE_MODEL}" \
        --num_cpu_procs=24

    # process chunk 2 here
    "${DATA_PROCESS_PYTHON}" "${DATA_PROCESS_SCRIPT}" \
        --data_path="${DATASET_CHUNK_2}" \
        --data_output_path="${DATASET_CHUNK_2_OUTPUT_PATH}" \
        --max_seq_len="${MAX_SEQ_LEN}" \
        --model_name_or_path="${BASE_MODEL}" \
        --num_cpu_procs=24

    # process chunk 3 here
    "${DATA_PROCESS_PYTHON}" "${DATA_PROCESS_SCRIPT}" \
        --data_path="${DATASET_CHUNK_3}" \
        --data_output_path="${DATASET_CHUNK_3_OUTPUT_PATH}" \
        --max_seq_len="${MAX_SEQ_LEN}" \
        --model_name_or_path="${BASE_MODEL}" \
        --num_cpu_procs=24
fi


# ================================================================================
# Launch complete dataset training script
# ================================================================================
CUDA_LAUNCH_BLOCKING=1 torchrun \
    --nnodes=1 \
    --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
    --data-path "${STD_DATA_OUTPUT_PATH}/data.jsonl" \
    --output-dir "${STANDARD_OUTPUT_DIR}" \
    --model-name-or-path "${BASE_MODEL}" \
    --min-samples-per-checkpoint "${STANDARD_SAVE_FREQUENCY}" \
    --num-warmup-steps "${WARMUP_STEPS}" \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --seed="${SEED}" \
    --orthogonal-subspace-learning \
    --max-steps="${STANDARD_MAX_STEPS}" \
    --osft-rank-ratio="${RANK_RATIO}"



# ================================================================================
# Launch multi-chunk training script here
# ================================================================================
# < implementation left as exercise to the reader >
