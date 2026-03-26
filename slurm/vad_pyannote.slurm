#!/bin/bash
#SBATCH --job-name=vad-pyannote
#SBATCH --output=logs/vad/pyannote_%A_%a.out
#SBATCH --error=logs/vad/pyannote_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --array=0

# Usage: sbatch slurm/vad_pyannote.slurm <manifest>
# Example: sbatch slurm/vad_pyannote.slurm chunks5

MANIFEST="${1:?Usage: sbatch slurm/vad_pyannote.slurm <manifest>}"

echo "VAD PyAnnote | Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Manifest: $MANIFEST"
echo "Array Task: ${SLURM_ARRAY_TASK_ID} / $((SLURM_ARRAY_TASK_COUNT - 1))"
echo "Started: $(date)"

export LD_LIBRARY_PATH=/shared/opt/linux-rocky9-x86_64/gcc-11.4.1/ffmpeg-6.1.1-gynsavpssxgp4ewikkmsa6jswfgi3ycg/lib:$LD_LIBRARY_PATH

module purge
module load ffmpeg

export PYTHONUNBUFFERED=1

uv run --frozen smbs vad-pyannote \
    --manifest "$MANIFEST" \
    --array-id "$SLURM_ARRAY_TASK_ID" \
    --array-count "$SLURM_ARRAY_TASK_COUNT"

EXIT_CODE=$?

echo ""
echo "Completed: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
