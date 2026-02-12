#!/bin/bash
#SBATCH --job-name=swuggy
#SBATCH --output=logs/swuggy/%x_%j.out
#SBATCH --error=logs/swuggy/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Usage: sbatch scripts/swuggy/run.slurm <script> [args...]
#   script:  prepare_swuggy, evaluate, or plots
#
# Examples:
#   sbatch scripts/swuggy/run.slurm prepare_swuggy hubert-500
#   sbatch scripts/swuggy/run.slurm evaluate hubert-500 lstm_h256_l2_d0.0_09feb13
#   sbatch scripts/swuggy/run.slurm plots --dataset swuggy

SCRIPT_NAME="${1:?Usage: sbatch scripts/swuggy/run.slurm <script> [args...]}"
shift 1

# Plots script doesn't need encoder, others do
if [ "$SCRIPT_NAME" != "plots" ]; then
    ENCODER="${1:?Usage: sbatch scripts/swuggy/run.slurm <script> <encoder>}"
    shift 1  # remaining args passed to script
fi

SCRIPT_PATH="scripts/swuggy/${SCRIPT_NAME}.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Display info
if [ "$SCRIPT_NAME" = "plots" ]; then
    echo "sWuggy: ${SCRIPT_NAME}.py"
else
    echo "sWuggy: ${SCRIPT_NAME}.py (encoder: ${ENCODER})"
fi
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# Only prepare_swuggy with legacy encoders needs the textless environment
if [ "$SCRIPT_NAME" = "prepare_swuggy" ] && [ "$ENCODER" != "spidr_base" ]; then
    uv run --project envs/textless --frozen "$SCRIPT_PATH" --encoder "$ENCODER" "$@"
# Everything else uses main uv environment
elif [ "$SCRIPT_NAME" = "plots" ]; then
    uv run --frozen "$SCRIPT_PATH" "$@"
else
    uv run --frozen "$SCRIPT_PATH" --encoder "$ENCODER" "$@"
fi

echo ""
echo "Completed: $(date)"
