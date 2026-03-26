#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --output=logs/train/grid_%j.out
#SBATCH --error=logs/train/grid_%j.err
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch slurm/grid.slurm [encoder]
# Example: sbatch slurm/grid.slurm spidr_base

ENCODER="${1:-spidr_base}"

NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)

export PYTHONUNBUFFERED=1

MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 100)))

echo "Grid search started at: $(date)"
echo "Encoder: $ENCODER | GPUs: $NUM_GPUS | Node: $(hostname)"
echo ""

uv run --frozen torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    -m smbs.train.grid \
    --encoder "$ENCODER"

echo ""
echo "Grid search completed at: $(date)"
