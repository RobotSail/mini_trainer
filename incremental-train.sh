#!/bin/bash
#
# Incremental Training Script
# Supports BMO and Finance-Bench experiment modes with Qwen2.5-7B or Llama-3.1-8B models
# Usage: ./incremental-train.sh [--mode MODE] [--model MODEL] [TRAINING_FLAGS] [--skip-process]
# Training flags: --full-sft, --complete-dataset, --multi-chunk
#

# stops early in case of problem
set -eo pipefail

# ================================================================================
# SET VARIABLES
# ================================================================================
# Default values - can be overridden by command line arguments
BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
EXPERIMENT_MODE='bmo'

# Training function flags - control which functions to run
RUN_FULL_SFT=0
RUN_COMPLETE_DATASET=0
RUN_MULTI_CHUNK=0

# datasets - will be populated based on mode
export FULL_DATASET=''
export DATASET_CHUNK_1=''
export DATASET_CHUNK_2=''
export DATASET_CHUNK_3=''

# BMO datasets
export BMO_FULL_DATASET='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/train_p07.jsonl'
export BMO_DATASET_CHUNK_1='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_0.jsonl'
export BMO_DATASET_CHUNK_2='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_1.jsonl'
export BMO_DATASET_CHUNK_3='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/bmo/3-chunks/chunk_2.jsonl'

# Finance-Bench Datasets
export FIN_BENCH_FULL_DATASET="/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/finance-bench/training_combined_cut_50x.jsonl"
export FIN_BENCH_DATASET_CHUNK_1='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/finance-bench/3-chunks/chunk_0.jsonl'
export FIN_BENCH_DATASET_CHUNK_2='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/finance-bench/3-chunks/chunk_1.jsonl'
export FIN_BENCH_DATASET_CHUNK_3='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0/finance-bench/3-chunks/chunk_2.jsonl'


# outputs - will be updated based on experiment mode
BASE_EXPERIMENT_DIR='/mnt/nvme1n1/experiments/os-cl-scenario-1-experiment-0'
EXPERIMENT_DIR=''  # Will be set after mode is determined



# Directory variables will be set by configure_experiment_mode()
CHUNKED_OUTPUT_DIR=''
FULL_OUTPUT_DIR=''
SFT_BASELINE_OUTPUT_DIR=''

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
# HELPER FUNCTIONS
# ================================================================================
function show_usage() {
    echo "Usage: $0 [OPTIONS] [TRAINING_FLAGS]"
    echo ""
    echo "Configuration Options:"
    echo "  --mode, -m MODE      Set experiment mode (bmo or finance-bench). Default: bmo"
    echo "  --model, -b MODEL    Set base model (qwen or llama). Default: llama"
    echo "  --skip-process, -s   Skip data processing step"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Training Functions (at least one required):"
    echo "  --full-sft          Run full SFT baseline training"
    echo "  --complete-dataset   Run complete dataset training"
    echo "  --multi-chunk       Run multi-chunk incremental training"
    echo ""
    echo "Available models:"
    echo "  qwen    - qwen/Qwen2.5-7B-Instruct"
    echo "  llama   - meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    echo "Examples:"
    echo "  $0 --mode bmo --model qwen --full-sft"
    echo "  $0 -m finance-bench -b llama --multi-chunk --complete-dataset"
    echo "  $0 --model qwen --full-sft --complete-dataset --multi-chunk"
    echo "  $0 --skip-process --full-sft"
    echo ""
}

# ================================================================================
# PARSE INPUT ARGUMENTS TO THE SCRIPT
# ================================================================================
# Parse arguments for configuration and training flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-process|-s)
            SKIP_PROCESS=1
            shift
            ;;
        --mode|-m)
            if [[ -n "$2" && "$2" != --* ]]; then
                if [[ "$2" == "bmo" || "$2" == "finance-bench" ]]; then
                    EXPERIMENT_MODE="$2"
                    shift 2
                else
                    echo "Error: Invalid mode '$2'. Must be 'bmo' or 'finance-bench'" >&2
                    show_usage
                    exit 1
                fi
            else
                echo "Error: --mode requires a value (bmo or finance-bench)" >&2
                show_usage
                exit 1
            fi
            ;;
        --model|-b)
            if [[ -n "$2" && "$2" != --* ]]; then
                if [[ "$2" == "qwen" ]]; then
                    BASE_MODEL='qwen/Qwen2.5-7B-Instruct'
                    shift 2
                elif [[ "$2" == "llama" ]]; then
                    BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
                    shift 2
                else
                    echo "Error: Invalid model '$2'. Must be 'qwen' or 'llama'" >&2
                    show_usage
                    exit 1
                fi
            else
                echo "Error: --model requires a value (qwen or llama)" >&2
                show_usage
                exit 1
            fi
            ;;
        --full-sft)
            RUN_FULL_SFT=1
            shift
            ;;
        --complete-dataset)
            RUN_COMPLETE_DATASET=1
            shift
            ;;
        --multi-chunk)
            RUN_MULTI_CHUNK=1
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Check if at least one training function is specified
if [[ $RUN_FULL_SFT -eq 0 && $RUN_COMPLETE_DATASET -eq 0 && $RUN_MULTI_CHUNK -eq 0 ]]; then
    echo "Error: At least one training function must be specified" >&2
    echo "Use --full-sft, --complete-dataset, or --multi-chunk" >&2
    echo ""
    show_usage
    exit 1
fi


# ================================================================================
#  CONFIGURATION FUNCTIONS
# ================================================================================
function configure_experiment_mode() {
    echo "Configuring experiment:"
    echo "  Mode: $EXPERIMENT_MODE"
    echo "  Base Model: $BASE_MODEL"
    
    # Show which training functions will run
    echo "  Training functions to run:"
    if [[ $RUN_FULL_SFT -eq 1 ]]; then
        echo "    - Full SFT Baseline"
    fi
    if [[ $RUN_COMPLETE_DATASET -eq 1 ]]; then
        echo "    - Complete Dataset Training"
    fi
    if [[ $RUN_MULTI_CHUNK -eq 1 ]]; then
        echo "    - Multi-Chunk Training"
    fi
    
    # Configure datasets based on mode
    if [[ "$EXPERIMENT_MODE" == "bmo" ]]; then
        export FULL_DATASET="$BMO_FULL_DATASET"
        export DATASET_CHUNK_1="$BMO_DATASET_CHUNK_1"
        export DATASET_CHUNK_2="$BMO_DATASET_CHUNK_2"
        export DATASET_CHUNK_3="$BMO_DATASET_CHUNK_3"
    elif [[ "$EXPERIMENT_MODE" == "finance-bench" ]]; then
        export FULL_DATASET="$FIN_BENCH_FULL_DATASET"
        export DATASET_CHUNK_1="$FIN_BENCH_DATASET_CHUNK_1"
        export DATASET_CHUNK_2="$FIN_BENCH_DATASET_CHUNK_2"
        export DATASET_CHUNK_3="$FIN_BENCH_DATASET_CHUNK_3"
    else
        echo "Error: Unknown experiment mode '$EXPERIMENT_MODE'" >&2
        exit 1
    fi
    
    # Configure directory structure with mode subdirectory
    EXPERIMENT_DIR="${BASE_EXPERIMENT_DIR}/${EXPERIMENT_MODE}/${BASE_MODEL}"
    
    # Update dependent directories
    CHUNKED_OUTPUT_DIR="${EXPERIMENT_DIR}/output/chunked-dataset-0"
    FULL_OUTPUT_DIR="${EXPERIMENT_DIR}/output/full-dataset-0"
    SFT_BASELINE_OUTPUT_DIR="${EXPERIMENT_DIR}/output/sft-baseline-0"
    
    echo "  Experiment directory: $EXPERIMENT_DIR"
    echo "Configuration complete!"
    
    # Create directories
    mkdir -p "${EXPERIMENT_DIR}"
    mkdir -p "${CHUNKED_OUTPUT_DIR}"
    mkdir -p "${FULL_OUTPUT_DIR}"
    mkdir -p "${SFT_BASELINE_OUTPUT_DIR}"
}
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
        echo "Processing full dataset with arguments:"
        echo "  --data_path=${FULL_DATASET}"
        echo "  --data_output_path=${STD_DATA_OUTPUT_PATH}" 
        echo "  --max_seq_len=${MAX_SEQ_LEN}"
        echo "  --model_name_or_path=${BASE_MODEL}"
        echo "  --num_cpu_procs=24"

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
        --output-dir "${FULL_OUTPUT_DIR}" \
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

function full_sft_baseline() {
    # process dataset full
    process_full_dataset

    # then launch training 
    CUDA_LAUNCH_BLOCKING=1 torchrun \
        --nnodes=1 \
        --nproc-per-node=8 "${MINI_TRAINER_SCRIPT}" \
        --data-path "${STD_DATA_OUTPUT_PATH}/data.jsonl" \
        --output-dir "${SFT_BASELINE_OUTPUT_DIR}" \
        --model-name-or-path "${BASE_MODEL}" \
        --num-warmup-steps "${WARMUP_STEPS}" \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --seed="${SEED}" \
        --max-epochs="${EPOCHS}" \
        --save-last-checkpoint
}


# complete_dataset_training_test
# recent=$(get_most_recent_checkpoint)
# printf 'got most recent: "%s"\n' "$(get_most_recent_checkpoint '/mnt/7TB-a/models/test_mini-trainer-new-changes/hf_format')"


# Configure experiment first
configure_experiment_mode

# Execute selected training functions
echo ""
echo "Starting training functions..."

if [[ $RUN_FULL_SFT -eq 1 ]]; then
    echo ""
    echo "==================== Running Full SFT Baseline ===================="
    full_sft_baseline
fi

if [[ $RUN_COMPLETE_DATASET -eq 1 ]]; then
    echo ""
    echo "================== Running Complete Dataset Training ================="
    complete_dataset_training
fi

if [[ $RUN_MULTI_CHUNK -eq 1 ]]; then
    echo ""
    echo "================== Running Multi-Chunk Training ===================="
    multi_chunk_training
fi

echo ""
echo "All selected training functions completed!"
