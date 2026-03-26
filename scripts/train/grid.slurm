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

# Usage: sbatch scripts/train/grid.slurm

PROJECT_ROOT="$(pwd)"
NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
export PROJECT_ROOT="$PROJECT_ROOT"

MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 100)))

echo "Grid search started at: $(date)"
echo "GPUs: $NUM_GPUS | Node: $(hostname)"
echo ""

uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    scripts/train/grid.py

echo ""
echo "Grid search completed at: $(date)"
