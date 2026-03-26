#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err
#SBATCH --partition=erc-dupoux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch slurm/train.slurm <encoder> [arch]
# Example: sbatch slurm/train.slurm spidr_base gpt2

ENCODER="${1:?Usage: sbatch slurm/train.slurm <encoder> [arch]}"
ARCH="${2:-gpt2}"

NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)

export PYTHONUNBUFFERED=1

# Use SLURM job ID to avoid port collisions
MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 100)))

uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    -m smbs.train.run \
    --encoder "$ENCODER" \
    --arch "$ARCH"

echo "Completed at: $(date)"
