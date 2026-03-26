"""Evaluation metrics — shared across all benchmarks."""

import polars as pl


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
    df: pl.DataFrame,
    prob_column: str = "log_prob",
    group_column: str = "group_id",
) -> dict[str, float]:
    """Break down accuracy by voice (requires 'voice' column)."""
    results = {}
    for voice in sorted(df["voice"].unique().to_list()):
        subset = df.filter(pl.col("voice") == voice)
        results[voice] = discrimination_accuracy(subset, prob_column, group_column)
    return results
