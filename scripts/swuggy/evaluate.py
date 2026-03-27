"""
Evaluate language models on lexical discrimination tasks.

Combined pipeline: scores every sample with log-probabilities, saves
the evaluated CSV, then computes and prints discrimination accuracy.

If the output CSV already exists, skips scoring and goes straight to analysis.

Path conventions (all derived from dataset + encoder + model):
    Metadata in:  metadata/{dataset}.csv
    Tokens:       tokens/{dataset}_{encoder}/
    Output:       metadata/{dataset}_{encoder}_{model}.csv
    Checkpoint:   weights/{encoder}/{model}/checkpoint-<latest>

Usage:
    python -m scripts.swuggy.evaluate --encoder hubert-500 --dataset swuggy --model lstm_h256_l2_d0.0_09feb13
"""

import time
from pathlib import Path

import polars as pl
import torch
import webdataset as wds

from scripts.encode.encoders import get_encoder_config
from scripts.swuggy.utils import load_checkpoint


# ───────────────────── Core computation ─────────────────────

def calculate_sequence_log_probability(model, tokens, device):
    """Compute log P(sequence) via autoregressive factorization.

    Returns (log_prob, log_prob_normalized, num_predicted_tokens).
    """
    tokens = tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=tokens, labels=tokens)

        # Get logits [1, seq_len, vocab_size]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # Next-token prediction: predict tokens[1:] from tokens[:-1]
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


def add_log_probabilities(model, df: pl.DataFrame, tokens_dict: dict, device: str) -> pl.DataFrame:
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


# ───────────────────── Analysis ─────────────────────


def discrimination_accuracy(
    df: pl.DataFrame,
    prob_column: str = "log_prob",
    group_column: str = "group_id",
) -> float:
    """Proportion of groups where positives beat negatives.

    For each group, computes: mean over positives of (fraction of negatives beaten).
    Returns macro-average across groups. Works with N-positive × M-negative.
    """
    accuracies = []

    for group_id in df[group_column].unique().to_list():
        group = df.filter(pl.col(group_column) == group_id)
        pos = group.filter(pl.col("positive"))[prob_column].drop_nulls().to_list()
        neg = group.filter(~pl.col("positive"))[prob_column].drop_nulls().to_list()

        if not pos or not neg:
            continue

        scores = [sum(p > n for n in neg) / len(neg) for p in pos]
        accuracies.append(sum(scores) / len(scores))

    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def per_voice_accuracy(
    df: pl.DataFrame, prob_column: str = "log_prob", group_column: str = "group_id"
) -> dict[str, float]:
    """Break down accuracy by voice (requires 'voice' column)."""
    results = {}
    for voice in sorted(df["voice"].unique().to_list()):
        subset = df.filter(pl.col("voice") == voice)
        results[voice] = discrimination_accuracy(subset, prob_column, group_column)
    return results


# ───────────────────── Path helpers ─────────────────────

ROOT = Path.cwd()

def find_latest_checkpoint(model_dir: Path) -> Path:
    """Return the checkpoint subdirectory with the highest step number."""
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return checkpoints[-1]


def resolve_paths(dataset: str, encoder: str, model: str):
    """Derive all paths from the three user-provided names."""
    metadata = ROOT / "metadata" / f"{dataset}.csv"
    tokens_dir = ROOT / "tokens" / f"{dataset}_{encoder}"
    model_dir = ROOT / "weights" / encoder / model
    output = ROOT / "metadata" / "swuggy" / f"{encoder}_{model}.csv"
    return metadata, tokens_dir, model_dir, output


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


# ───────────────────── CLI ─────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score samples with log-probs and compute discrimination accuracy. "
                    "Skips scoring if output already exists.")
    parser.add_argument("--encoder", type=str, required=True,
                        help="Encoder name (e.g. spidr_base, hubert-500, mhubert)")
    parser.add_argument("model", type=str, nargs="?", default=None,
                        help="Model directory name under weights/{encoder}/... "
                             "(e.g. lstm_h256_l2_d0.0_09feb13). Latest checkpoint is used.")
    parser.add_argument("--dataset", type=str, default="swuggy",
                        help="Dataset name (default: swuggy). "
                             "Reads metadata/{dataset}.csv, tokens from tokens/{dataset}_{encoder}/")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if output CSV exists")
    args = parser.parse_args()
    
    if args.model is None:
        parser.error("model argument is required")

    metadata_path, tokens_dir, model_dir, output_path = \
        resolve_paths(args.dataset, args.encoder, args.model)

    # ── Check for existing results ───────────────────────────────
    if output_path.exists() and not args.force:
        print(f"Found existing results: {output_path}")
        print("Skipping scoring, running analysis only. Use --force to re-score.\n")
        df_scored = pl.read_csv(output_path)
        group_col = "group_id" if "group_id" in df_scored.columns else "word_id"
        print_analysis(df_scored, output_path)
        raise SystemExit(0)

    # ── Full evaluation run ──────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = find_latest_checkpoint(model_dir)

    print(f"{'=' * 60}")
    print("EVALUATE + ANALYSE")
    print(f"{'=' * 60}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Encoder:    {args.encoder}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Metadata:   {metadata_path}")
    print(f"  Tokens:     {tokens_dir}")
    print(f"  Output:     {output_path}")
    print(f"  Device:     {device}")
    print(f"{'=' * 60}\n")

    # Load model
    print("Loading model...")
    model, model_config = load_checkpoint(str(checkpoint_path), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_config.model_type.upper()} | {n_params/1e6:.1f}M params")

    # Load data
    print("Loading metadata...")
    df = pl.read_csv(metadata_path)
    print(f"  {len(df)} samples")

    enc_config = get_encoder_config(args.encoder)
    tokens_dict = load_tokens_from_tar(
        str(tokens_dir), enc_config.bos_token_id, enc_config.eos_token_id)

    # Score
    run_start = time.time()
    df_scored = add_log_probabilities(model, df, tokens_dict, device)
    scoring_time = time.time() - run_start

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.write_csv(output_path)

    # ── Scoring summary ─────────────────────────────────────────
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

    # ── Discrimination accuracy ──────────────────────────────────
    group_col = "group_id" if "group_id" in df_scored.columns else "word_id"
    print_analysis(df_scored, output_path)
