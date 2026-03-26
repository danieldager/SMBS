#!/bin/bash
#SBATCH --job-name=swuggy
#SBATCH --output=logs/swuggy/%x_%j.out
#SBATCH --error=logs/swuggy/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux,gpu-p1,gpu-p2

# Usage:
#   Prepare:  sbatch slurm/evaluate.slurm prepare <encoder> <parquet_pattern>
#   Evaluate: sbatch slurm/evaluate.slurm evaluate <encoder> <model>
#   Plots:    sbatch slurm/evaluate.slurm plots

ACTION="${1:?Usage: sbatch slurm/evaluate.slurm <prepare|evaluate|plots> [args...]}"
shift 1

echo "sWuggy: $ACTION"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

export PYTHONUNBUFFERED=1

case "$ACTION" in
    prepare)
        ENCODER="${1:?Usage: ... prepare <encoder> <parquet_pattern>}"
        PATTERN="${2:?Usage: ... prepare <encoder> <parquet_pattern>}"
        uv run python -m smbs.evaluate.swuggy prepare --encoder "$ENCODER" --parquet-pattern "$PATTERN"
        ;;
    evaluate)
        ENCODER="${1:?Usage: ... evaluate <encoder> <model>}"
        MODEL="${2:?Usage: ... evaluate <encoder> <model>}"
        uv run python -m smbs.evaluate.swuggy evaluate --encoder "$ENCODER" --model "$MODEL"
        ;;
    plots)
        uv run python -m smbs.evaluate.plots
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Usage: sbatch slurm/evaluate.slurm <prepare|evaluate|plots> [args...]"
        exit 1
        ;;
esac

echo ""
echo "Completed: $(date)"
