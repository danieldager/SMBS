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

# Usage: sbatch scripts/train/run.slurm <encoder> [arch]
# Example: sbatch scripts/train/run.slurm spidr_base gpt2

ENCODER="${1:?Usage: sbatch scripts/train/run.slurm <encoder> [arch]}"
ARCH="${2:-gpt2}"

# Token directories derived from encoder name
PROJECT_ROOT="$(pwd)"
TRAIN_DIR="${PROJECT_ROOT}/tokens/chunks30_${ENCODER}/"
EVAL_DIR="${PROJECT_ROOT}/tokens/chunks-eval_${ENCODER}/"

# Extract number of GPUs from SLURM gres parameter
NUM_GPUS=$(echo $SLURM_STEP_GPUS | tr "," "\n" | wc -l)

# Add current working directory to Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Use SLURM job ID to avoid port collisions when multiple jobs run on same node
MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 100)))

uv run --frozen torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_port=$MASTER_PORT \
    scripts/train/train.py \
    --arch $ARCH \
    --encoder "$ENCODER" \
    --train_dir "$TRAIN_DIR" \
    --eval_dir "$EVAL_DIR"

echo "Completed at: $(date)"