"""TenVAD pipeline — CPU-based voice activity detection with multiprocessing.

Outputs two parquet files:
  - metadata.parquet  per-file summary (duration, speech ratio, etc.)
  - segments.parquet  per-segment rows (file_id, onset, offset, duration)

Files containing any speech segment >= threshold are flagged.
"""

import sys
import time
import shutil
import random
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import torch
import torchaudio

from ten_vad import TenVad

from smbs.config import SAMPLE_RATE, METADATA_DIR
from smbs.utils.manifest import get_task_shard

LONG_SEGMENT_THRESHOLD = 10.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_runs(flags: np.ndarray):
    """Return (speech_runs, silence_runs) as Nx2 arrays of frame indices."""
    if len(flags) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int)

    edges = np.flatnonzero(np.diff(flags))
    edges = np.r_[0, edges + 1, len(flags)]
    pairs = np.column_stack((edges[:-1], edges[1:]))

    if flags[0] == 1:
        speech_runs, silence_runs = pairs[0::2], pairs[1::2]
    else:
        speech_runs, silence_runs = pairs[1::2], pairs[0::2]

    return speech_runs, silence_runs


def runs_to_segments(runs: np.ndarray, hop_size: int, sr: int) -> list[dict]:
    """Convert frame-index runs to list of {onset, offset, duration} in seconds."""
    if runs.size == 0:
        return []
    factor = hop_size / sr
    onsets = np.round(runs[:, 0] * factor, 3)
    offsets = np.round(runs[:, 1] * factor, 3)
    durations = np.round(offsets - onsets, 3)
    return [
        {"onset": float(o), "offset": float(off), "duration": float(d)}
        for o, off, d in zip(onsets, offsets, durations)
    ]


def segment_stats(durations: list[float]) -> dict:
    if not durations:
        return {"max": 0.0, "min": 0.0, "sum": 0.0, "num": 0, "avg": 0.0}
    arr = np.asarray(durations)
    return {
        "max": float(arr.max()),
        "min": float(arr.min()),
        "sum": float(arr.sum()),
        "num": len(durations),
        "avg": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Per-file processing (runs inside worker processes)
# ---------------------------------------------------------------------------


def _error_meta(path: str, error: str) -> dict:
    return {
        "success": False,
        "path": path,
        "file_id": Path(path).stem,
        "duration": 0.0,
        "original_sr": 0,
        "speech_ratio": 0.0,
        "n_speech_segments": 0,
        "n_silence_segments": 0,
        "speech_max": 0.0, "speech_min": 0.0, "speech_sum": 0.0,
        "speech_num": 0, "speech_avg": 0.0,
        "nospch_max": 0.0, "nospch_min": 0.0, "nospch_sum": 0.0,
        "nospch_num": 0, "nospch_avg": 0.0,
        "has_long_segment": False,
        "error": error,
    }


def process_file(args: tuple) -> tuple[dict, list[dict]]:
    """Process one WAV file. Returns (metadata_row, segment_rows)."""
    wav_path, hop_size, threshold = args

    try:
        vad = TenVad(hop_size=hop_size, threshold=threshold)
    except Exception as e:
        return _error_meta(str(wav_path), f"TenVad init: {e}"), []

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        original_sr = sr

        if waveform.size(0) > 1:
            waveform = waveform[0:1, :]

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            sr = SAMPLE_RATE

        data = (waveform.squeeze().numpy() * 32767).astype(np.int16)
        duration = round(len(data) / sr, 3)

        n_frames = len(data) // hop_size
        frames = data[: n_frames * hop_size].reshape(-1, hop_size)
        flags = np.empty(n_frames, dtype=np.uint8)
        proc = vad.process
        for i in range(n_frames):
            _, flags[i] = proc(frames[i])

        speech_runs, silence_runs = get_runs(flags)
        speech_segs = runs_to_segments(speech_runs, hop_size, sr)
        silence_segs = runs_to_segments(silence_runs, hop_size, sr)

        speech_durs = [s["duration"] for s in speech_segs]
        silence_durs = [s["duration"] for s in silence_segs]

        sp = segment_stats(speech_durs)
        ns = segment_stats(silence_durs)

        has_long = any(d >= LONG_SEGMENT_THRESHOLD for d in speech_durs)
        file_id = Path(wav_path).stem

        meta = {
            "success": True,
            "path": str(wav_path),
            "file_id": file_id,
            "duration": duration,
            "original_sr": int(original_sr),
            "speech_ratio": round(float(flags.mean()), 3),
            "n_speech_segments": sp["num"],
            "n_silence_segments": ns["num"],
            **{f"speech_{k}": v for k, v in sp.items()},
            **{f"nospch_{k}": v for k, v in ns.items()},
            "has_long_segment": has_long,
            "error": "",
        }

        seg_rows = [{"file_id": file_id, **s} for s in speech_segs]
        return meta, seg_rows

    except Exception as e:
        return _error_meta(str(wav_path), str(e)), []


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


def _log_progress(done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    eta = f"{remaining / 60:.0f}m" if remaining < 3600 else f"{remaining / 3600:.1f}h"
    print(f"  [{done:>7}/{total}]  {rate:.1f} files/s  ETA {eta}")


def _log_interval(n: int) -> int:
    if n < 10_000:
        return 1_000
    if n < 50_000:
        return 5_000
    if n < 100_000:
        return 10_000
    return 20_000


def process_parallel(
    wavs: list[Path],
    hop_size: int,
    threshold: float,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    """Run VAD on all files using a process pool."""
    tasks = [(w, hop_size, threshold) for w in wavs]
    meta_rows: list[dict] = []
    seg_rows: list[dict] = []
    errors = 0
    total = len(tasks)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, t): t[0] for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                meta, segs = future.result()
                meta_rows.append(meta)
                seg_rows.extend(segs)
                if not meta.get("success", False):
                    errors += 1
                    print(f"  WARN: {Path(path).name}: {meta['error']}", file=sys.stderr)
            except Exception as e:
                errors += 1
                print(f"  ERROR: {Path(path).name}: {e}", file=sys.stderr)

            if i % _log_interval(i) == 0 or i == total:
                _log_progress(i, total, t0)

    elapsed = time.time() - t0
    print(f"Processed {len(meta_rows)}/{total} files in {elapsed:.1f}s  ({errors} errors)")
    return meta_rows, seg_rows


def copy_long_segment_files(meta_df: pl.DataFrame, dest: Path) -> int:
    """Copy files with speech segments >= LONG_SEGMENT_THRESHOLD to dest/."""
    long = meta_df.filter(pl.col("has_long_segment"))
    if long.is_empty():
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    paths = long.get_column("path").to_list()
    copied = 0
    for p in paths:
        src = Path(p)
        if src.exists():
            shutil.copy2(src, dest / src.name)
            copied += 1
    return copied


def run_tenvad(
    manifest: str,
    hop_size: int = 256,
    threshold: float = 0.5,
    workers: int | None = None,
) -> None:
    """Run TenVAD on a manifest."""
    set_seeds(42)

    workers = workers or mp.cpu_count()
    print(f"Workers: {workers}")

    _, _, paths = get_task_shard(manifest, 0, 1)
    wavs = [Path(p) for p in paths]
    print(f"Files:   {len(wavs)}")

    if not wavs:
        print("Nothing to process.")
        return

    manifest_stem = Path(manifest).stem
    out_dir = METADATA_DIR / manifest_stem / "ten"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows, seg_rows = process_parallel(wavs, hop_size, threshold, workers)

    if not meta_rows:
        print("ERROR: all files failed", file=sys.stderr)
        sys.exit(1)

    meta_rows.sort(key=lambda r: r["path"])
    seg_rows.sort(key=lambda r: (r["file_id"], r["onset"]))

    meta_df = pl.DataFrame(meta_rows)
    meta_df.write_parquet(out_dir / "metadata.parquet", compression="zstd")
    print(f"Saved metadata.parquet ({len(meta_df)} rows)")

    seg_df = pl.DataFrame(seg_rows) if seg_rows else pl.DataFrame(
        schema={"file_id": pl.Utf8, "onset": pl.Float64, "offset": pl.Float64, "duration": pl.Float64}
    )
    seg_df.write_parquet(out_dir / "segments.parquet", compression="zstd")
    print(f"Saved segments.parquet ({len(seg_df)} rows)")

    ok = meta_df.filter(pl.col("success"))
    fail = meta_df.filter(~pl.col("success"))
    print(f"\nSuccess: {len(ok)}/{len(meta_df)}")
    if not fail.is_empty():
        print(f"Failed:  {len(fail)}", file=sys.stderr)
        for row in fail.head(5).iter_rows(named=True):
            print(f"  {Path(row['path']).name}: {row['error']}", file=sys.stderr)

    from smbs.config import PROJECT_ROOT
    data_dir = PROJECT_ROOT / "data" / manifest_stem / "long_segments"
    n_copied = copy_long_segment_files(meta_df, data_dir)
    if n_copied:
        print(f"Copied {n_copied} files with speech >= {LONG_SEGMENT_THRESHOLD}s to {data_dir}")
