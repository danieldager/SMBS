"""Ingest and tokenize the sWuggy dataset.

This is the ONLY swuggy-specific file. It reads the raw sWuggy parquet
(which has embedded audio bytes in positive/negative struct columns),
unpivots into individual samples, encodes audio → tokens, and writes
the standard format expected by evaluate.py and analysis.py:

    Tokens:   WebDataset .tar files with {__key__: file_id, tokens.npy: int16[]}
    Metadata: Parquet with columns [group_id, file_id, positive, ...extras]
"""

import io
import time
import warnings
from pathlib import Path

import polars as pl
import torch
import torchaudio  # type: ignore
import webdataset as wds

from scripts.encode.encoders import load_encoder, SAMPLE_RATE

# Suppress noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

MAX_SHARD_SIZE = 1 * 1024**3
MAX_SHARD_COUNT = 10000


def prepare_and_encode(
    raw_parquet_pattern: str,
    output_tokens_dir: str,
    output_metadata_path: str,
    encoder_name: str = "spidr_base",
    device: str = "cuda",
) -> None:
    """Load raw sWuggy parquets, encode audio, write standard format.

    The raw sWuggy schema has one row per word pair:
        id, positive (struct with bytes), negative (struct with bytes),
        positive_word, negative_word, positive_phones, negative_phones, voice

    We unpivot into one row per audio sample with a group_id linking each
    positive to its negative(s), then encode and write to tar + metadata.
    """
    print("=" * 60)
    print("SWUGGY: PREPARE & ENCODE")
    print(f"Encoder: {encoder_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and unpivot
    # ------------------------------------------------------------------
    print(f"\nLoading raw data from {raw_parquet_pattern}...")
    df_raw = pl.read_parquet(raw_parquet_pattern)
    print(f"Loaded {len(df_raw)} word pairs")

    df_positive = df_raw.select(
        pl.col("id").alias("group_id"),
        pl.col("positive").struct.field("bytes").alias("audio_bytes"),
        pl.col("positive_word").alias("word"),
        pl.col("positive_phones").alias("phones"),
        pl.col("voice"),
        pl.lit(True).alias("positive"),
    )

    df_negative = df_raw.select(
        pl.col("id").alias("group_id"),
        pl.col("negative").struct.field("bytes").alias("audio_bytes"),
        pl.col("negative_word").alias("word"),
        pl.col("negative_phones").alias("phones"),
        pl.col("voice"),
        pl.lit(False).alias("positive"),
    )

    df = pl.concat([df_positive, df_negative])

    # file_id: unique key for each audio sample
    df = df.with_columns(
        pl.concat_str([
            pl.col("group_id").cast(pl.Utf8),
            pl.lit("_"),
            pl.col("voice"),
            pl.lit("_"),
            pl.when(pl.col("positive")).then(pl.lit("pos")).otherwise(pl.lit("neg")),
        ]).alias("file_id")
    )

    print(f"Unpivoted into {len(df)} samples "
          f"({len(df_positive)} positive + {len(df_negative)} negative)")

    # ------------------------------------------------------------------
    # 2. Encode audio → tokens and write to WebDataset tars
    # ------------------------------------------------------------------
    print(f"\nLoading {encoder_name} encoder on {device}...")
    encoder = load_encoder(encoder_name, device=device)

    output_dir = Path(output_tokens_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / "shard-%03d.tar")

    print(f"Encoding and writing to {output_dir}...")

    processed = 0
    skipped = 0
    start_time = time.time()

    with wds.ShardWriter(shard_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:  # type: ignore
        for row in df.iter_rows(named=True):
            file_id = row["file_id"]
            audio_bytes = row["audio_bytes"]

            try:
                waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
                duration = waveform.shape[-1] / sample_rate

                if duration < 0.5:
                    skipped += 1
                    continue

                tokens = encoder.encode(waveform, sample_rate)

                if len(tokens) == 0:
                    skipped += 1
                    continue

                sink.write({"__key__": file_id, "tokens.npy": tokens})
                processed += 1

                if processed % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {processed}/{len(df)} | {processed / elapsed:.1f} samples/sec")

            except Exception as e:
                skipped += 1
                print(f"  Error {file_id}: {str(e)[:100]}")

    # ------------------------------------------------------------------
    # 3. Save metadata (without audio bytes)
    # ------------------------------------------------------------------
    print(f"\nSaving metadata to {output_metadata_path}...")
    df_meta = df.select("group_id", "file_id", "word", "phones", "voice", "positive")
    Path(output_metadata_path).parent.mkdir(parents=True, exist_ok=True)
    df_meta.write_parquet(output_metadata_path)

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'=' * 60}")
    print(f"Done: {processed} encoded, {skipped} skipped, {elapsed_min:.1f} min")
    print(f"Tokens:   {output_tokens_dir}")
    print(f"Metadata: {output_metadata_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and encode sWuggy dataset")
    parser.add_argument("--encoder", type=str, default="spidr_base",
                        help="Encoder name (e.g. spidr_base, hubert-500, mhubert)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path.cwd()

    prepare_and_encode(
        raw_parquet_pattern="/store/projects/lexical-benchmark/swuggy/data/*.parquet",
        output_tokens_dir=str(root / "tokens" / f"swuggy_{args.encoder}"),
        output_metadata_path=str(root / "metadata" / "swuggy.parquet"),
        encoder_name=args.encoder,
        device=device,
    )
