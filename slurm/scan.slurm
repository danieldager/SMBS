#!/bin/bash
#SBATCH --job-name=scan
#SBATCH --output=logs/scan/scan_%j.out
#SBATCH --error=logs/scan/scan_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
##SBATCH --partition=<your-partition>   # set for your cluster, or pass -p to smbs

# Usage: sbatch slurm/scan.slurm <dataset_dir>
# Example: sbatch slurm/scan.slurm /path/to/audio/

DATASET="${1:?Usage: sbatch slurm/scan.slurm <dataset_dir>}"

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset directory not found: $DATASET"
    exit 1
fi

echo "Scan Job | Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK"
echo "Dataset: $DATASET"
echo "Started: $(date)"
echo ""

export PYTHONUNBUFFERED=1

uv run python -m smbs.scan "$DATASET" --workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "Completed: $(date)"
