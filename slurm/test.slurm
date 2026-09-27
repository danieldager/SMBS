#!/bin/bash
#SBATCH --job-name=test_smbs
#SBATCH --output=logs/test/test_%j.out
#SBATCH --error=logs/test/test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
##SBATCH --partition=<your-partition>   # set for your cluster, or pass -p to smbs

echo "Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

export PYTHONUNBUFFERED=1

mkdir -p logs/test

uv run python tests/test_gpu.py

echo ""
echo "Finished: $(date)"
