"""Manifest loading and task sharding utilities."""

from pathlib import Path

import polars as pl

from smbs.config import MANIFESTS_DIR


def resolve_manifest(dataset: str) -> Path:
    """Resolve a dataset name to a manifest file path.

    Tries, in order: manifests/{dataset}.csv, .parquet
    If dataset is already an existing path, returns it directly.
    """
    p = Path(dataset)
    if p.exists():
        return p

    for ext in (".csv", ".parquet"):
        candidate = MANIFESTS_DIR / f"{dataset}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No manifest found for '{dataset}'. "
        f"Searched: {MANIFESTS_DIR}/{dataset}.{{csv,parquet}}"
    )


def load_manifest(manifest_path: str | Path) -> pl.DataFrame:
    """Load manifest from CSV or Parquet.

    Expects 'file_id' and 'audio_filepath' (or 'path') columns.
    """
    path = Path(manifest_path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pl.read_parquet(path)
    elif suffix == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {suffix}. Expected .csv or .parquet")

    # Normalize column names
    if "path" in df.columns and "audio_filepath" not in df.columns:
        df = df.rename({"path": "audio_filepath"})

    required = {"file_id", "audio_filepath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_task_shard(
    manifest_path: str | Path, array_id: int, array_count: int
) -> tuple[int, int, list[str]]:
    """Parse manifest and return (total_files, chunk_size, file_paths) for an array task."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif suffix == ".csv":
        lf = pl.scan_csv(path)
    else:
        raise ValueError(f"Unsupported manifest extension: {suffix}")

    schema = lf.collect_schema()
    if "path" in schema.names():
        col = "path"
    elif "audio_filepath" in schema.names():
        col = "audio_filepath"
    else:
        raise ValueError(f"Manifest must contain 'path' or 'audio_filepath' column")

    total = lf.select(pl.len()).collect().item()
    chunk = total // array_count
    start = array_id * chunk
    length = total - start if array_id == array_count - 1 else chunk

    paths = (
        lf.sort(col)
        .slice(start, length)
        .select(col)
        .collect()
        .get_column(col)
        .to_list()
    )
    return total, length, paths
