#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --output=logs/encode/encode_%A_%a.out
#SBATCH --error=logs/encode/encode_%A_%a.err
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Usage: sbatch scripts/encode/run.slurm <encoder> <manifest>
# Example: sbatch scripts/encode/run.slurm spidr_base manifests/chunks30.csv
#          sbatch scripts/encode/run.slurm mhubert manifests/chunks30.csv
#          sbatch scripts/encode/run.slurm hubert-500 manifests/chunks30.csv

ENCODER="${1:?Usage: sbatch scripts/encode/run.slurm <encoder> <manifest>}"
MANIFEST="${2:?Usage: sbatch scripts/encode/run.slurm <encoder> <manifest>}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Encode: $ENCODER"
echo "Manifest: $MANIFEST"
echo "Task: $TASK_ID / $((NUM_TASKS - 1))"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1

# SPIDR-based encoders use main uv environment
if [ "$ENCODER" = "spidr_base" ]; then
    uv run scripts/encode/encode.py \
        --encoder "$ENCODER" \
        --manifest "$MANIFEST" \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS" \
        --device cuda
# Legacy encoders (mhubert, hubert-500) use separate textless uv environment
else
    uv run --project envs/textless scripts/encode/encode.py \
        --encoder "$ENCODER" \
        --manifest "$MANIFEST" \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS" \
        --device cuda
fi

echo ""
echo "Task $TASK_ID completed: $(date)"
