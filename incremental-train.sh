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
#  UTILITY FUNCTIONS
# ================================================================================
function get_most_recent_checkpoint() {
    local directory="$1"
    
    # Check if directory exists
    if [ ! -d "$directory" ]; then
        echo "Directory does not exist: $directory" >&2
        return 1
    fi

    # Find most recent subdirectory by modification time
    # -mindepth 1: Skip the root directory
    # -maxdepth 1: Only look at immediate children
    # -type d: Only look at directories
    # -printf '%T@ %p\n': Print modification timestamp and path
    local most_recent=$(find "$directory" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$most_recent" ]; then
        echo "No subdirectories found in: $directory" >&2
        return 1
    fi

    # Convert to absolute path
    echo "$(cd "$most_recent" && pwd)"
}


# ================================================================================
# DATASET PARSING
# ================================================================================
function process_full_dataset() {
    # Process the dataset if we need to 
    if [[ "${SKIP_PROCESS}" -eq 0 ]]; then
        # this function processes the full dataset
        "${DATA_PROCESS_PYTHON}" "${DATA_PROCESS_SCRIPT}" \
            --data_path="${FULL_DATASET}" \
            --data_output_path="${STD_DATA_OUTPUT_PATH}" \
            --max_seq_len="${MAX_SEQ_LEN}" \
            --model_name_or_path="${BASE_MODEL}" \
            --num_cpu_procs=24
    fi

}

function process_chunked_datasets() {
    # Process the dataset if we need to 
    if [[ "${SKIP_PROCESS}" -eq 0 ]]; then
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
}


# utility function to process everything as-needed
function process_all_datasets() {
    process_full_dataset
    process_chunked_datasets
}
# ================================================================================
# Launch complete dataset training script
# ================================================================================
function complete_dataset_training() {
    # process dataset full
    process_full_dataset

    # then launch training 
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
}


# Version with dummy values for testing
function complete_dataset_training_test() {
    # process dataset full
    process_full_dataset

    # then launch training with dummy values
    CUDA_LAUNCH_BLOCKING=1 torchrun \
        --nnodes=1 \
        --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
        --data-path "${STD_DATA_OUTPUT_PATH}/data.jsonl" \
        --output-dir "/mnt/7TB-a/models/test_mini-trainer-new-changes" \
        --model-name-or-path "meta-llama/Llama-3.2-1B-Instruct" \
        --num-warmup-steps 100 \
        --max-tokens-per-gpu 2048 \
        --batch-size 32 \
        --learning-rate 1e-4 \
        --seed=42 \
        --max-epochs 3 \
        --save-on-epoch
}

# ================================================================================
# Launch multi-chunk training script here
# ================================================================================
# < implementation left as exercise to the reader >
function multi_chunk_training() {
    process_chunked_datasets

    # now we just need to launch training for each dataset that we processed
    DATASET_CHUNK_1_OUTPUT_PATH='/dev/shm/chunk-1'
    DATASET_CHUNK_2_OUTPUT_PATH='/dev/shm/chunk-2'
    DATASET_CHUNK_3_OUTPUT_PATH='/dev/shm/chunk-3'

    # checkpoint save directories
    local chunk_1_checkpoints="${CHUNKED_OUTPUT_DIR}/first_chunk"
    local chunk_2_checkpoints="${CHUNKED_OUTPUT_DIR}/second_chunk"
    local chunk_3_checkpoints="${CHUNKED_OUTPUT_DIR}/third_chunk"

    # Create checkpoint directories
    mkdir -p "${chunk_1_checkpoints}"
    mkdir -p "${chunk_2_checkpoints}" 
    mkdir -p "${chunk_3_checkpoints}"

    # Create dataset output directories
    mkdir -p "${DATASET_CHUNK_1_OUTPUT_PATH}"
    mkdir -p "${DATASET_CHUNK_2_OUTPUT_PATH}"
    mkdir -p "${DATASET_CHUNK_3_OUTPUT_PATH}"

    # Launch the first training job
    CUDA_LAUNCH_BLOCKING=1 torchrun \
        --nnodes=1 \
        --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
        --data-path "${DATASET_CHUNK_1_OUTPUT_PATH}/data.jsonl" \
        --output-dir "${chunk_1_checkpoints}" \
        --model-name-or-path "${BASE_MODEL}" \
        --num-warmup-steps "${WARMUP_STEPS}" \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --seed="${SEED}" \
        --orthogonal-subspace-learning \
        --osft-rank-ratio="${RANK_RATIO}" \
        --max-epochs="${EPOCHS}" \
        --save-last-checkpoint

    # then get the most recent checkpoint
    local model_chunk_1=$(get_most_recent_checkpoint "${chunk_1_checkpoints}/hf_format")

    # Launch the second training job
    CUDA_LAUNCH_BLOCKING=1 torchrun \
        --nnodes=1 \
        --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
        --data-path "${DATASET_CHUNK_2_OUTPUT_PATH}/data.jsonl" \
        --output-dir "${chunk_2_checkpoints}" \
        --model-name-or-path "${model_chunk_1}" \
        --num-warmup-steps "${WARMUP_STEPS}" \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --seed="${SEED}" \
        --orthogonal-subspace-learning \
        --osft-rank-ratio="${RANK_RATIO}" \
        --max-epochs="${EPOCHS}" \
        --save-last-checkpoint

    # then get the most recent checkpoint
    local model_chunk_2=$(get_most_recent_checkpoint "${chunk_2_checkpoints}/hf_format")

    # Launch the third and final training job
    CUDA_LAUNCH_BLOCKING=1 torchrun \
        --nnodes=1 \
        --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
        --data-path "${DATASET_CHUNK_3_OUTPUT_PATH}/data.jsonl" \
        --output-dir "${chunk_3_checkpoints}" \
        --model-name-or-path "${model_chunk_2}" \
        --num-warmup-steps "${WARMUP_STEPS}" \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --seed="${SEED}" \
        --orthogonal-subspace-learning \
        --osft-rank-ratio="${RANK_RATIO}" \
        --max-epochs="${EPOCHS}" \
        --save-last-checkpoint

    # final checkpoint
    local model_chunk_3=$(get_most_recent_checkpoint "${chunk_3_checkpoints}/hf_format")
    printf '\033[0;33mFinal checkpoint: %s\033[0m\n' "${model_chunk_3}"

}
# complete_dataset_training_test
# recent=$(get_most_recent_checkpoint)
# printf 'got most recent: "%s"\n' "$(get_most_recent_checkpoint '/mnt/7TB-a/models/test_mini-trainer-new-changes/hf_format')"
multi_chunk_training
