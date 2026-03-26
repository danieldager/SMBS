"""Parallel directory scanning to produce audio file manifests."""

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

from smbs.config import MANIFESTS_DIR

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}


def scan_directory_recursive(dirpath: str) -> list[Path]:
    """Recursively scan a directory tree for audio files."""
    results = []
    try:
        for root, _, files in os.walk(dirpath):
            for filename in files:
                if Path(filename).suffix.lower() in AUDIO_EXTENSIONS:
                    results.append(Path(root, filename).resolve())
    except (PermissionError, OSError):
        pass
    return results


def iter_audio_files_parallel(root: Path, num_workers: int | None) -> list[Path]:
    """Parallel scan by distributing top-level subdirectories across workers."""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"Finding top-level directories to distribute across {num_workers} workers...")
    try:
        subdirs = [
            str(Path(root, entry.name)) for entry in os.scandir(root) if entry.is_dir()
        ]
    except (PermissionError, OSError):
        subdirs = []

    if not subdirs:
        print("No subdirectories found, scanning root directly...")
        return scan_directory_recursive(str(root))

    print(f"Found {len(subdirs)} top-level directories, scanning in parallel...")

    all_files = []
    with Pool(num_workers) as pool:
        for i, results in enumerate(
            pool.imap_unordered(scan_directory_recursive, subdirs, chunksize=1), 1
        ):
            all_files.extend(results)
            print(f"  Completed {i}/{len(subdirs)} dirs, found {len(all_files)} files so far...")

    return all_files


def run_scan(dataset_dir: str, workers: int | None = None) -> None:
    """Scan a dataset directory and write a manifest to manifests/."""
    dataset = Path(dataset_dir).expanduser().resolve()
    if not dataset.exists() or not dataset.is_dir():
        raise SystemExit(f"ERROR: not a directory: {dataset}")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (MANIFESTS_DIR / f"{dataset.name}.txt").resolve()

    print(f"Scanning {dataset}...")
    paths = iter_audio_files_parallel(dataset, num_workers=workers)

    print(f"\nFound {len(paths):,} total files")
    print("Sorting by (directory, filename) for NFS cache locality...")
    paths.sort(key=lambda p: (str(p.parent), p.name))

    print(f"Writing {len(paths):,} paths to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths, 1):
            f.write(str(p) + "\n")
            if i % 100_000 == 0:
                print(f"  Written {i:,}/{len(paths):,} paths...")

    print(f"\nWrote {len(paths):,} paths to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_scan(args.directory, args.workers)
