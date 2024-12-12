#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./run_wandb_docker.sh gpu=<GPU_ID> sweep=<SWEEP_ID> wandb_key=<WANDB_API_KEY>

# Default Values
GPU_ID=""
SWEEP_ID=""
WANDB_API_KEY=""

# Parse Arguments
for arg in "$@"; do
    case "$arg" in
        gpu=*) GPU_ID="${arg#*=}" ;;
        sweep=*) SWEEP_ID="${arg#*=}" ;;
        wandb_key=*) WANDB_API_KEY="${arg#*=}" ;;
        *)
            echo "Invalid argument: $arg"
            echo "Usage: $0 gpu=<GPU_ID> sweep=<SWEEP_ID> wandb_key=<WANDB_API_KEY>"
            exit 1
            ;;
    esac
done

# Validate Inputs
if [[ -z "$GPU_ID" || -z "$SWEEP_ID" || -z "$WANDB_API_KEY" ]]; then
    echo "Error: gpu=<GPU_ID>, sweep=<SWEEP_ID>, and wandb_key=<WANDB_API_KEY> must be provided."
    echo "Usage: $0 gpu=<GPU_ID> sweep=<SWEEP_ID> wandb_key=<WANDB_API_KEY>"
    exit 1
fi

# Check CUDA Version
cuda_version=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+")
if [[ "$cuda_version" -eq 12 ]]; then
    DOCKER_IMAGE="bic4907/pcgrl:cu12"
elif [[ "$cuda_version" -eq 11 ]]; then
    DOCKER_IMAGE="bic4907/pcgrl:cu11"
else
    echo "Unsupported CUDA version: $cuda_version"
    exit 1
fi

# Generate Current Date and Time
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")

# Generate Container Name: Replace '/' in SWEEP_ID with '_'
CONTAINER_NAME="$(echo "${SWEEP_ID}" | sed 's|/|_|g')_gpu${GPU_ID}_${CURRENT_DATE}"

# Run Docker with Specified GPU, CUDA_VISIBLE_DEVICES, and WandB Key
echo "Running WandB Agent with GPU=${GPU_ID}, Sweep ID=${SWEEP_ID}, and Docker Image=${DOCKER_IMAGE}..."
docker run --rm --gpus all \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --name "${CONTAINER_NAME}" \
    -e CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    "${DOCKER_IMAGE}" \
    bash -c "wandb agent ${SWEEP_ID}"
