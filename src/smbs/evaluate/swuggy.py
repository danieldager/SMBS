"""sWuggy lexical discrimination benchmark.

Includes both dataset preparation (raw parquet → tokens) and evaluation
(score tokens with model, compute discrimination accuracy).
"""

import io
import time
from pathlib import Path

import polars as pl
import torch
import torchaudio  # type: ignore
import webdataset as wds

from smbs.config import (
    MIN_SWUGGY_DURATION,
    MAX_SHARD_SIZE,
    MAX_SHARD_COUNT,
    TOKENS_DIR,
    METADATA_DIR,
    WEIGHTS_DIR,
)
from smbs.encode import load_encoder, get_encoder_config
from smbs.evaluate.metrics import discrimination_accuracy, per_voice_accuracy
from smbs.train.utils import load_checkpoint, find_latest_checkpoint


# ───────────────────── Preparation ─────────────────────


def prepare_swuggy(
    raw_parquet_pattern: str,
    encoder_name: str,
    device: str = "cuda",
) -> None:
    """Load raw sWuggy parquets, encode audio, write standard format.

    The raw sWuggy schema has one row per word pair with embedded audio bytes.
    We unpivot into individual samples, encode audio → tokens, and write to
    WebDataset .tar files + metadata parquet.
    """
    output_tokens_dir = TOKENS_DIR / f"swuggy_{encoder_name}"
    output_metadata_path = METADATA_DIR / "swuggy.parquet"

    print("=" * 60)
    print("SWUGGY: PREPARE & ENCODE")
    print(f"Encoder: {encoder_name}")
    print("=" * 60)

    # Load and unpivot
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

    # Encode
    print(f"\nLoading {encoder_name} encoder on {device}...")
    encoder = load_encoder(encoder_name, device=device)

    output_tokens_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_tokens_dir / "shard-%03d.tar")

    print(f"Encoding and writing to {output_tokens_dir}...")

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

                if duration < MIN_SWUGGY_DURATION:
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

    # Save metadata (without audio bytes)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta = df.select("group_id", "file_id", "word", "phones", "voice", "positive")
    df_meta.write_parquet(output_metadata_path)

    elapsed_min = (time.time() - start_time) / 60
    print(f"\n{'=' * 60}")
    print(f"Done: {processed} encoded, {skipped} skipped, {elapsed_min:.1f} min")
    print(f"Tokens:   {output_tokens_dir}")
    print(f"Metadata: {output_metadata_path}")
    print(f"{'=' * 60}\n")


# ───────────────────── Scoring ─────────────────────


def calculate_sequence_log_probability(model, tokens, device):
    """Compute log P(sequence) via autoregressive factorization."""
    tokens = tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=tokens)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        seq_log_prob = target_log_probs.sum().item()
        n = target_log_probs.shape[1]
        norm_log_prob = seq_log_prob / n if n > 0 else 0.0

    return seq_log_prob, norm_log_prob, n


def load_tokens_from_tar(tokens_dir: str, bos_token_id: int, eos_token_id: int) -> dict:
    """Load all token arrays from .tar shards, wrap with BOS/EOS."""
    urls = sorted(str(p) for p in Path(tokens_dir).glob("*.tar"))
    if not urls:
        raise ValueError(f"No tar files found in {tokens_dir}")

    print(f"Loading tokens from {len(urls)} tar files...")
    tokens_dict = {}

    for sample in wds.WebDataset(urls, shardshuffle=False).decode():  # type: ignore
        key = sample["__key__"]
        raw = sample.get("tokens.npy")
        if raw is None:
            continue
        token_list = [bos_token_id] + raw.tolist() + [eos_token_id]
        tokens_dict[key] = torch.tensor(token_list, dtype=torch.long)

    print(f"Loaded {len(tokens_dict)} token sequences")
    return tokens_dict


def score_samples(model, df: pl.DataFrame, tokens_dict: dict, device: str) -> pl.DataFrame:
    """Score every sample in df, return df with log_prob columns added."""
    print("\nCalculating log probabilities...")

    total = len(df)
    log_probs, log_probs_norm, seq_lengths = [], [], []
    missing = 0
    start_time = time.time()
    log_interval = max(1, total // 20)

    for idx, row in enumerate(df.iter_rows(named=True), 1):
        tokens = tokens_dict.get(row["file_id"])

        if tokens is None:
            log_probs.append(None)
            log_probs_norm.append(None)
            seq_lengths.append(None)
            missing += 1
            continue

        lp, lpn, n = calculate_sequence_log_probability(model, tokens, device)
        log_probs.append(lp)
        log_probs_norm.append(lpn)
        seq_lengths.append(n)

        if idx % log_interval == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total - idx) / rate if rate > 0 else 0
            print(f"  {idx}/{total} ({100*idx/total:.0f}%) | "
                  f"{rate:.1f} s/s | ETA {remaining/60:.1f}m")

    if missing:
        print(f"Warning: {missing}/{total} samples had no tokens")

    return df.with_columns(
        pl.Series("log_prob", log_probs),
        pl.Series("log_prob_norm", log_probs_norm),
        pl.Series("num_tokens", seq_lengths),
    )


def print_analysis(df_scored: pl.DataFrame, output_path: Path):
    """Print discrimination accuracy results."""
    group_col = "group_id" if "group_id" in df_scored.columns else "word_id"

    print(f"\n{'=' * 60}")
    print("DISCRIMINATION ACCURACY")
    print(f"{'=' * 60}")
    print(f"  Source:  {output_path}")
    print(f"  Samples: {len(df_scored)}")
    print(f"  Groups:  {df_scored[group_col].n_unique()} (column: {group_col})")

    for label, col in [("Raw", "log_prob"), ("Normalized", "log_prob_norm")]:
        if col not in df_scored.columns:
            continue
        acc = discrimination_accuracy(df_scored, col, group_col)
        print(f"\n  {label}: {acc:.4f}")

        if "voice" in df_scored.columns:
            for voice, vacc in per_voice_accuracy(df_scored, col, group_col).items():
                print(f"    {voice:<12s} {vacc:.4f}")

    print(f"\n{'=' * 60}")


# ───────────────────── Main entry points ─────────────────────


def run_evaluate(
    encoder: str,
    model: str,
    dataset: str = "swuggy",
    force: bool = False,
) -> None:
    """Score samples with log-probs and compute discrimination accuracy."""
    metadata_path = METADATA_DIR / f"{dataset}.parquet"
    tokens_dir = TOKENS_DIR / f"{dataset}_{encoder}"
    model_dir = WEIGHTS_DIR / encoder / model
    output_path = METADATA_DIR / dataset / f"{encoder}_{model}.parquet"

    # Check for existing results
    if output_path.exists() and not force:
        print(f"Found existing results: {output_path}")
        print("Skipping scoring, running analysis only. Use --force to re-score.\n")
        df_scored = pl.read_parquet(output_path)
        print_analysis(df_scored, output_path)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = find_latest_checkpoint(model_dir)

    print(f"{'=' * 60}")
    print("EVALUATE + ANALYSE")
    print(f"{'=' * 60}")
    print(f"  Dataset:    {dataset}")
    print(f"  Encoder:    {encoder}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Metadata:   {metadata_path}")
    print(f"  Tokens:     {tokens_dir}")
    print(f"  Output:     {output_path}")
    print(f"  Device:     {device}")
    print(f"{'=' * 60}\n")

    # Load model
    print("Loading model...")
    loaded_model, model_config = load_checkpoint(str(checkpoint_path), device)
    n_params = sum(p.numel() for p in loaded_model.parameters())
    print(f"  {model_config.model_type.upper()} | {n_params/1e6:.1f}M params")

    # Load data
    print("Loading metadata...")
    df = pl.read_parquet(metadata_path)
    print(f"  {len(df)} samples")

    enc_config = get_encoder_config(encoder)
    tokens_dict = load_tokens_from_tar(
        str(tokens_dir), enc_config.bos_token_id, enc_config.eos_token_id)

    # Score
    run_start = time.time()
    df_scored = score_samples(loaded_model, df, tokens_dict, device)
    scoring_time = time.time() - run_start

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.write_parquet(output_path)

    # Summary
    scored = df_scored.filter(pl.col("log_prob").is_not_null())
    lp = scored["log_prob"]
    lpn = scored["log_prob_norm"]
    ntok = scored["num_tokens"]

    print(f"\n{'=' * 60}")
    print("SCORING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Samples:      {len(df)} total, {len(scored)} scored, "
          f"{len(df) - len(scored)} missing tokens")
    print(f"  Token stats:  min={ntok.min()}, median={ntok.median():.0f}, "
          f"max={ntok.max()}, mean={ntok.mean():.1f}")
    print(f"  Log-prob:     mean={lp.mean():.2f}, std={lp.std():.2f}")
    print(f"  Log-prob/tok: mean={lpn.mean():.4f}, std={lpn.std():.4f}")
    print(f"  Scoring time: {scoring_time:.1f}s ({len(scored)/scoring_time:.1f} samples/s)")
    print(f"  Saved to:     {output_path}")

    print_analysis(df_scored, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--encoder", required=True)
    p_prep.add_argument("--parquet-pattern", required=True)
    p_prep.add_argument("--device", default="cuda")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--encoder", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--dataset", default="swuggy")
    p_eval.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.action == "prepare":
        prepare_swuggy(args.parquet_pattern, args.encoder, args.device)
    else:
        run_evaluate(args.encoder, args.model, args.dataset, args.force)
