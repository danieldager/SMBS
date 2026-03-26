#!/bin/bash
#SBATCH --job-name=vad-ten
#SBATCH --output=logs/vad/ten_%j.out
#SBATCH --error=logs/vad/ten_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu,erc-dupoux,gpu-p1,gpu-p2

# Usage: sbatch slurm/vad_ten.slurm <manifest>
# Example: sbatch slurm/vad_ten.slurm chunks5

MANIFEST="${1:?Usage: sbatch slurm/vad_ten.slurm <manifest>}"

echo "TenVAD Pipeline"
echo "Job ID: $SLURM_JOB_ID | Manifest: $MANIFEST"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: ${SLURM_MEM_PER_NODE}"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/llvm-17.0.6-ibcxgmojxutzsgeoyywyraq5fmvycsfl/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
export POLARS_SKIP_CPU_CHECK=1
export PYTHONUNBUFFERED=1

module purge
module load ffmpeg llvm

uv run python -m smbs.vad.tenvad "$MANIFEST" \
    --workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "VAD completed: $(date)"
