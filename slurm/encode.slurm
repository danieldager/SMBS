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

# Usage: sbatch slurm/encode.slurm <encoder> <dataset>
# Example: sbatch slurm/encode.slurm spidr_base chunks30
#          sbatch slurm/encode.slurm mhubert chunks30
#          sbatch slurm/encode.slurm hubert-500 chunks30

ENCODER="${1:?Usage: sbatch slurm/encode.slurm <encoder> <dataset>}"
DATASET="${2:?Usage: sbatch slurm/encode.slurm <encoder> <dataset>}"

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Encode: $ENCODER | Dataset: $DATASET"
echo "Task: $TASK_ID / $((NUM_TASKS - 1))"
echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

export PYTHONUNBUFFERED=1

# SPIDR-based encoders use main uv environment
if [ "$ENCODER" = "spidr_base" ]; then
    uv run --frozen smbs encode \
        --encoder "$ENCODER" \
        --dataset "$DATASET" \
        --device cuda \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS"
# Legacy encoders (mhubert, hubert-500) use separate textless environment
else
    uv run --project envs/textless --frozen smbs encode \
        --encoder "$ENCODER" \
        --dataset "$DATASET" \
        --device cuda \
        --task-id "$TASK_ID" \
        --num-tasks "$NUM_TASKS"
fi

echo ""
echo "Task $TASK_ID completed: $(date)"
