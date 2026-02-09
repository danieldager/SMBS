#!/usr/bin/env python3
"""Tokenize audio files using a speech encoder and write to WebDataset shards.

Usage:
    python scripts/encode/encode.py --encoder spidr_base --manifest manifests/chunks30.csv
    python scripts/encode/encode.py --encoder mhubert --manifest manifests/chunks30.csv

Output is written to: tokens/{manifest_stem}_{encoder_name}/
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
import torchaudio  # type: ignore
import webdataset as wds  # type: ignore

from scripts.encode.encoders import load_encoder, AudioEncoder

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 16000
MIN_DURATION = 3.0  # seconds — skip segments shorter than this
MAX_SHARD_SIZE = 1 * 1024**3  # 1 GB per shard
MAX_SHARD_COUNT = 10000  # max samples per shard


# =============================================================================
# Manifest Loading
# =============================================================================


def load_manifest(manifest_path: str) -> pl.DataFrame:
    """Load manifest from CSV or Parquet. Requires 'file_id' and 'audio_filepath'."""
    if manifest_path.endswith(".parquet"):
        df = pl.read_parquet(manifest_path)
    else:
        df = pl.read_csv(manifest_path)

    # Normalize column names
    if "path" in df.columns:
        df = df.rename({"path": "audio_filepath"})

    required = {"file_id", "audio_filepath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# =============================================================================
# Shard Writing
# =============================================================================


def setup_writer(
    manifest_path: str, encoder_name: str, task_id: int
) -> wds.ShardWriter:
    """Create output directory and WebDataset shard writer.

    Output: {project_root}/tokens/{dataset}_{encoder}/task{task_id}-shard{N}.tar
    """
    manifest = Path(manifest_path)
    project_root = manifest.parent.parent  # manifests/ → project root
    dataset_name = manifest.stem.split("_")[0]  # "chunks30" from "chunks30.csv"
    output_dir = project_root / "tokens" / f"{dataset_name}_{encoder_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    pattern = str(output_dir / f"task{task_id:03d}-shard%03d.tar")
    return wds.ShardWriter(pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT)  # type: ignore


def write_tokens(
    sink: wds.ShardWriter,
    file_id: str,
    segment_id: int,
    tokens: np.ndarray,
    audio_filepath: str,
) -> None:
    """Write a single tokenized segment to the shard."""
    file_stem = Path(file_id).stem
    key = f"{file_stem}_s{segment_id:03d}"

    sink.write({
        "__key__": key,
        "tokens.npy": tokens.astype(np.int16),
        "json": {
            "file_id": file_stem,
            "segment_id": segment_id,
            "token_count": len(tokens),
            "audio_filepath": str(audio_filepath),
        },
    })


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
        print(f"  Skipped:  {self.skipped_short} too short, {self.skipped_error} errors")
        print(f"  Time:     {self.elapsed_min():.1f} min ({self.rate():.1f} files/sec)")
        print(f"{'='*60}\n")


# =============================================================================
# Core Processing
# =============================================================================


def process_file(
    encoder: AudioEncoder,
    file_id: str,
    audio_filepath: str,
    sink: wds.ShardWriter,
    tracker: ProgressTracker,
) -> None:
    """Load, encode, and write one audio file."""
    waveform, sample_rate = torchaudio.load(audio_filepath)

    if waveform.shape[-1] == 0:
        print(f"  ERROR: zero-length waveform: {audio_filepath}", file=sys.stderr)
        tracker.skipped_error += 1
        return

    duration = waveform.shape[-1] / sample_rate
    if duration < MIN_DURATION:
        tracker.skipped_short += 1
        return

    tokens = encoder.encode(waveform, sample_rate)

    if len(tokens) == 0:
        print(f"  ERROR: zero tokens: {audio_filepath}", file=sys.stderr)
        tracker.skipped_error += 1
        return

    write_tokens(sink, file_id, segment_id=0, tokens=tokens, audio_filepath=audio_filepath)
    tracker.processed += 1


def tokenize(
    encoder: AudioEncoder,
    manifest_path: str,
    encoder_name: str,
    task_id: int = 0,
    num_tasks: int = 1,
) -> None:
    """Main tokenization pipeline."""
    df = load_manifest(manifest_path)
    df = df[task_id::num_tasks]
    print(f"Processing {len(df)} files (task {task_id}/{num_tasks})\n")

    tracker = ProgressTracker()

    with setup_writer(manifest_path, encoder_name, task_id) as sink:
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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio files to WebDataset shards")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder name (e.g. spidr_base, mhubert, hubert-500)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV/Parquet with file_id and audio_filepath")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--task-id", type=int, default=0, help="Task ID for array jobs")
    parser.add_argument("--num-tasks", type=int, default=1, help="Total parallel tasks")

    args = parser.parse_args()

    print(f"Encoder: {args.encoder}")
    print(f"Manifest: {args.manifest}")
    print(f"Device: {args.device}")

    encoder = load_encoder(args.encoder, device=args.device)

    with torch.no_grad():
        tokenize(
            encoder=encoder,
            manifest_path=args.manifest,
            encoder_name=args.encoder,
            task_id=args.task_id,
            num_tasks=args.num_tasks,
        )
