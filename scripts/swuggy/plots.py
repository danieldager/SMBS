"""Generate comparison plots across models and encoders.

Scans metadata/ for all evaluated parquet files and creates bar charts
comparing discrimination accuracy across different encoder+model combinations.

Usage:
    python scripts/swuggy/plots.py [--raw] [--dataset DATASET]
    
    # Compare all models for all datasets
    python scripts/swuggy/plots.py
    
    # Use raw log-prob instead of normalized
    python scripts/swuggy/plots.py --raw
    
    # Only plot specific dataset
    python scripts/swuggy/plots.py --dataset swuggy
"""

from pathlib import Path
import re

import polars as pl


# ============================================================================
# Analysis functions (shared with evaluate.py)
# ============================================================================


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


# ============================================================================
# Plotting
# ============================================================================


def plot_accuracy(output_path: Path, df_scored: pl.DataFrame, group_col: str, 
                 output_dir: Path = None):
    """Generate and save a bar chart of discrimination accuracy for a single model.
    
    Creates a figure showing raw and normalized accuracy with value labels.
    
    Args:
        output_path: Path object used to generate filename (uses .stem)
        df_scored: DataFrame with log_prob and log_prob_norm columns
        group_col: Column name for grouping (e.g., "group_id")
        output_dir: Where to save figure (default: figures/)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ⚠ Matplotlib not installed, skipping plot generation")
        return
    
    if output_dir is None:
        output_dir = Path.cwd() / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute accuracies
    raw_acc = discrimination_accuracy(df_scored, "log_prob", group_col)
    norm_acc = discrimination_accuracy(df_scored, "log_prob_norm", group_col) \
        if "log_prob_norm" in df_scored.columns else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.array([0])
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, [raw_acc], width, label="Raw", color="#1f77b4")
    if norm_acc is not None:
        bars2 = ax.bar(x + width/2, [norm_acc], width, label="Length-normalized", color="#aec7e8")
    
    # Add value labels
    ax.text(x[0] - width/2, raw_acc + 0.02, f"{raw_acc:.3f}", 
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    if norm_acc is not None:
        ax.text(x[0] + width/2, norm_acc + 0.02, f"{norm_acc:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    # Styling
    model_name = output_path.stem  # e.g., swuggy_hubert-500_lstm_...
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Lexical Discrimination Accuracy\n{model_name}", fontsize=13)
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Chance")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    # Save
    figure_path = output_dir / f"{output_path.stem}.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Plot saved:   {figure_path}")


def plot_dataset_comparison(dataset: str, results: dict[str, Path], 
                           use_raw: bool = False, output_dir: Path = None):
    """Create bar chart comparing models for a single dataset.
    
    Args:
        dataset: Dataset name (e.g., "swuggy")
        results: {model_label: parquet_path} dict
        use_raw: Use raw log_prob instead of log_prob_norm
        output_dir: Where to save figure (default: figures/)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if output_dir is None:
        output_dir = Path.cwd() / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prob_col = "log_prob" if use_raw else "log_prob_norm"
    
    # Compute accuracies for each model
    model_names = []
    accuracies = []
    
    for label, path in sorted(results.items()):
        df = pl.read_parquet(path)
        
        # Detect grouping column
        group_col = "group_id" if "group_id" in df.columns else "word_id"
        
        # Skip if this result doesn't have the requested prob column
        if prob_col not in df.columns:
            print(f"  Skipping {label}: missing {prob_col} column")
            continue
        
        acc = discrimination_accuracy(df, prob_col, group_col)
        model_names.append(label)
        accuracies.append(acc)
    
    if not accuracies:
        print(f"  No valid results found for {dataset}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), 6))
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, accuracies, color="#1f77b4", alpha=0.8)
    
    # Value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Styling
    prob_type = "Raw" if use_raw else "Length-Normalized"
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Lexical Discrimination Accuracy: {dataset}\n({prob_type})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Chance")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    # Save
    suffix = "_raw" if use_raw else ""
    output_path = output_dir / f"{dataset}_comparison{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# File discovery
# ============================================================================


def find_evaluated_results(metadata_dir: Path = None) -> dict[str, dict[str, Path]]:
    """Scan metadata directory for evaluated parquet files.
    
    Returns nested dict: {dataset: {encoder_model: path}}
    
    Parses filenames like: swuggy_hubert-500_lstm_h256_l2_d0.0_09feb13.parquet
    """
    if metadata_dir is None:
        metadata_dir = Path.cwd() / "metadata"
    
    # Pattern: {dataset}_{encoder}_{model}.parquet
    # We need at least 3 underscore-separated parts
    pattern = re.compile(r"^(.+?)_([^_]+)_(.+)\.parquet$")
    
    results = {}
    
    for path in metadata_dir.glob("*.parquet"):
        match = pattern.match(path.name)
        if not match:
            continue
        
        dataset, encoder, model = match.groups()
        
        # Skip raw metadata files (those without model suffix)
        # Check if file has required columns for evaluated results
        try:
            df = pl.read_parquet(path)
            if "log_prob" not in df.columns:
                continue  # Not an evaluated result
        except Exception:
            continue
        
        # Group by dataset
        if dataset not in results:
            results[dataset] = {}
        
        # Label: encoder + architecture (e.g., "hubert-500_lstm" or "spidr_base_gpt2")
        # Extract architecture from model name (e.g., "lstm_h256_l2_d0.0_09feb13" -> "lstm")
        arch = model.split("_")[0]  # First component is the architecture
        label = f"{encoder}_{arch}"
        results[dataset][label] = path
    
    return results


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for lexical discrimination results")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw log_prob instead of length-normalized")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Only plot specific dataset (default: plot all)")
    parser.add_argument("--metadata-dir", type=str, default=None,
                        help="Metadata directory (default: ./metadata)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: ./figures)")
    args = parser.parse_args()
    
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else Path.cwd() / "metadata"
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "figures"
    
    print(f"Scanning {metadata_dir} for evaluated results...")
    results = find_evaluated_results(metadata_dir)
    
    if not results:
        print("No evaluated results found.")
        print("Run evaluate.py first to generate results.")
        raise SystemExit(1)
    
    print(f"Found {len(results)} dataset(s):")
    for dataset, models in results.items():
        print(f"  {dataset}: {len(models)} model(s)")
    print()
    
    # Filter by dataset if requested
    if args.dataset:
        if args.dataset not in results:
            print(f"Dataset '{args.dataset}' not found in results.")
            raise SystemExit(1)
        results = {args.dataset: results[args.dataset]}
    
    # Generate plots
    prob_type = "raw log-prob" if args.raw else "normalized log-prob"
    print(f"Generating comparison plots ({prob_type})...")
    
    for dataset, models in results.items():
        print(f"\n{dataset}:")
        plot_dataset_comparison(dataset, models, args.raw, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")
