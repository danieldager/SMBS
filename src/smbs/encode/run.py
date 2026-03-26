"""Encode audio files into token sequences and write to WebDataset shards."""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio  # type: ignore
import webdataset as wds  # type: ignore

from smbs.config import (
    SAMPLE_RATE,
    MIN_AUDIO_DURATION,
    MAX_SHARD_SIZE,
    MAX_SHARD_COUNT,
    TOKENS_DIR,
)
from smbs.encode.base import AudioEncoder
from smbs.encode.registry import load_encoder
from smbs.utils.manifest import load_manifest

warnings.filterwarnings("ignore")


# =============================================================================
# Shard Writing
# =============================================================================


def setup_writer(dataset_name: str, encoder_name: str, task_id: int) -> wds.ShardWriter:  # type: ignore
    """Create output directory and WebDataset shard writer."""
    output_dir = TOKENS_DIR / f"{dataset_name}_{encoder_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    pattern = str(output_dir / f"task{task_id:03d}-shard%03d.tar")
    return wds.ShardWriter(pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT)  # type: ignore


def write_tokens(
    sink: wds.ShardWriter,  # type: ignore
    file_id: str,
    segment_id: int,
    tokens: np.ndarray,
    audio_filepath: str,
) -> None:
    """Write a single tokenized segment to the shard."""
    file_stem = Path(file_id).stem
    key = f"{file_stem}_s{segment_id:03d}"

    sink.write(
        {
            "__key__": key,
            "tokens.npy": tokens.astype(np.int16),
            "json": {
                "file_id": file_stem,
                "segment_id": segment_id,
                "token_count": len(tokens),
                "audio_filepath": str(audio_filepath),
            },
        }
    )


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressTracker:
    """Track processing speed and statistics."""

    def __init__(self):
        self.start_time = time.time()
        self.processed = 0
        self.skipped_short = 0
        self.skipped_error = 0

    def elapsed_min(self) -> float:
        return (time.time() - self.start_time) / 60

    def rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0

    def log_progress(self, counter: int, total: int) -> None:
        if counter % 1000 == 0 or counter == 100:
            rate = self.rate()
            remaining = (total - counter) / rate if rate > 0 else 0
            eta_h, eta_m = int(remaining // 3600), int((remaining % 3600) // 60)
            print(
                f"  [{counter:6d}/{total}] | "
                f"{self.processed} tokens written | "
                f"{rate:.1f} files/sec | "
                f"ETA: {eta_h}h{eta_m}m",
                flush=True,
            )

    def log_summary(self, task_id: int) -> None:
        print(f"\n{'='*60}")
        print(f"Task {task_id} complete: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Written:  {self.processed} samples")
        print(
            f"  Skipped:  {self.skipped_short} too short, {self.skipped_error} errors"
        )
        print(f"  Time:     {self.elapsed_min():.1f} min ({self.rate():.1f} files/sec)")
        print(f"{'='*60}\n")


# =============================================================================
# Core Processing
# =============================================================================


def process_file(
    encoder: AudioEncoder,
    file_id: str,
    audio_filepath: str,
    sink: wds.ShardWriter,  # type: ignore
    tracker: ProgressTracker,
) -> None:
    """Load, encode, and write one audio file."""
    waveform, sample_rate = torchaudio.load(audio_filepath)

    if waveform.shape[-1] == 0:
        print(f"  ERROR: zero-length waveform: {audio_filepath}", file=sys.stderr)
        tracker.skipped_error += 1
        return

    duration = waveform.shape[-1] / sample_rate
    if duration < MIN_AUDIO_DURATION:
        tracker.skipped_short += 1
        return

    tokens = encoder.encode(waveform, sample_rate)

    if len(tokens) == 0:
        print(f"  ERROR: zero tokens: {audio_filepath}", file=sys.stderr)
        tracker.skipped_error += 1
        return

    write_tokens(
        sink, file_id, segment_id=0, tokens=tokens, audio_filepath=audio_filepath
    )
    tracker.processed += 1


def run_encode(
    encoder_name: str,
    dataset: str,
    manifest_path: str,
    device: str = "cuda",
    task_id: int = 0,
    num_tasks: int = 1,
) -> None:
    """Main tokenization pipeline."""
    encoder = load_encoder(encoder_name, device=device)

    df = load_manifest(manifest_path)
    df = df[task_id::num_tasks]
    print(f"Processing {len(df)} files (task {task_id}/{num_tasks})\n")

    tracker = ProgressTracker()

    with setup_writer(dataset, encoder_name, task_id) as sink:
        for counter, row in enumerate(df.iter_rows(named=True)):
            try:
                process_file(
                    encoder=encoder,
                    file_id=str(row["file_id"]),
                    audio_filepath=str(row["audio_filepath"]),
                    sink=sink,
                    tracker=tracker,
                )
            except Exception as e:
                msg = str(e)[:100]
                if "Cannot subsample F0" not in msg:
                    print(f"  ERROR [{row['file_id']}]: {msg}", file=sys.stderr)
                tracker.skipped_error += 1

            tracker.log_progress(counter, len(df))

    tracker.log_summary(task_id)
