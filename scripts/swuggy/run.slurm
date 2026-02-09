#!/bin/bash
#SBATCH --job-name=swuggy
#SBATCH --output=logs/swuggy/%x_%j.out
#SBATCH --error=logs/swuggy/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Usage: sbatch scripts/swuggy/run.slurm <script> <encoder> [extra args...]
#   script:  prepare_swuggy or evaluate
#   encoder: spidr_base, hubert-500, mhubert
#
# Examples:
#   sbatch scripts/swuggy/run.slurm prepare_swuggy hubert-500
#   sbatch scripts/swuggy/run.slurm evaluate hubert-500 --dataset swuggy --model lstm_h256_l2_d0.0_09feb13

SCRIPT_NAME="${1:?Usage: sbatch scripts/swuggy/run.slurm <script> <encoder> [extra args...]}"
ENCODER="${2:?Usage: sbatch scripts/swuggy/run.slurm <script> <encoder>}"
shift 2  # remaining args passed to script

SCRIPT_PATH="scripts/swuggy/${SCRIPT_NAME}.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "sWuggy: ${SCRIPT_NAME}.py (encoder: ${ENCODER})"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# Legacy encoders (mhubert, hubert-500) need the textless conda env
if [ "$ENCODER" = "spidr_base" ]; then
    uv run --frozen "$SCRIPT_PATH" --encoder "$ENCODER" "$@"
else
    source ~/.bashrc
    conda activate textless
    python -u "$SCRIPT_PATH" --encoder "$ENCODER" "$@"
fi

echo ""
echo "Completed: $(date)"
