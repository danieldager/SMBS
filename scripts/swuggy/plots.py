"""Generate comparison plots across models and encoders.

Creates a single unified bar chart showing all LSTM and GPT-2 models.

Usage:
    python scripts/swuggy/plots.py [--raw]
"""

import re
from datetime import datetime
import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", font_scale=1.2)


def discrimination_accuracy(df: pl.DataFrame, prob_column: str = "log_prob", 
                           group_column: str = "group_id"):
    """Calculate discrimination accuracy: proportion of groups where positives beat negatives."""
    # Separate positive and negative examples
    pos_df = df.filter(pl.col("positive")).select([group_column, prob_column]).drop_nulls()
    neg_df = df.filter(~pl.col("positive")).select([group_column, prob_column]).drop_nulls()
    
    # Join on group_id to create all positive-negative pairs
    pairs = pos_df.join(neg_df, on=group_column, suffix="_neg")
    
    # Calculate: for each pair, is positive > negative?
    pairs = pairs.with_columns((pl.col(prob_column) > pl.col(f"{prob_column}_neg")).alias("correct"))
    
    # Group by group_id and positive example, calculate accuracy per positive
    group_acc = pairs.group_by([group_column, prob_column]).agg(
        pl.col("correct").mean().alias("pos_acc")
    )
    
    # Average across positive examples within each group, then across groups
    final_acc = group_acc.group_by(group_column).agg(
        pl.col("pos_acc").mean().alias("group_acc")
    )
    
    return final_acc["group_acc"].mean() if len(final_acc) > 0 else 0.0


def parse_model_info(filepath: Path):
    """Extract encoder, architecture, and size from result CSV filename."""
    # e.g., spidr_base_lstm_h256_l2_d0.0.csv
    # or   hubert-500_gpt2_e768_l12_h12_09feb13.csv
    stem = filepath.stem
    
    # Determine architecture by searching for keywords
    if "lstm" in stem:
        arch = "lstm"
        # Extract hidden size
        match = re.search(r"_h(\d+)", stem)
        size = int(match.group(1)) if match else 0
    elif "gpt2" in stem:
        arch = "gpt2"
        # Extract embedding size
        match = re.search(r"_e(\d+)", stem)
        size = int(match.group(1)) if match else 0
    else:
        return None, "unknown", 0
    
    # Extract encoder name (everything before arch keyword)
    if arch == "lstm":
        encoder = stem.split("_lstm")[0]
    elif arch == "gpt2":
        encoder = stem.split("_gpt2")[0]
    else:
        encoder = "unknown"
    
    return encoder, arch, size


def create_unified_plot(metadata_dir: Path, output_dir: Path, use_raw: bool = False):
    """Create single plot comparing all models across all encoders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prob_col = "log_prob" if use_raw else "log_prob_norm"
    
    # Collect all models
    models = []
    
    for csv_file in metadata_dir.glob("*.csv"):
        # Read and check if it's an evaluated result
        try:
            df = pl.read_csv(csv_file)
            if prob_col not in df.columns or "positive" not in df.columns:
                continue
        except Exception:
            continue
        
        # Parse model info
        encoder, arch, size = parse_model_info(csv_file)
        
        # Skip if not LSTM or GPT2
        if arch not in ["lstm", "gpt2"]:
            continue
        
        # Calculate accuracy
        group_col = "group_id" if "group_id" in df.columns else "word_id"
        accuracy = discrimination_accuracy(df, prob_col, group_col)
        
        models.append({
            "encoder": encoder,
            "arch": arch,
            "size": size,
            "accuracy": accuracy,
            "filename": csv_file.stem,
        })
    
    if not models:
        print("No valid models found")
        return
    
    # Sort: LSTMs first (by encoder, size), then GPT2s (by encoder, size)
    lstm_models = sorted([m for m in models if m["arch"] == "lstm"], 
                        key=lambda x: (x["encoder"], x["size"]))
    gpt2_models = sorted([m for m in models if m["arch"] == "gpt2"], 
                        key=lambda x: (x["encoder"], x["size"]))
    all_models = lstm_models + gpt2_models
    
    # Prepare plot data
    labels = []
    accuracies = []
    colors = []
    
    encoder_colors = {
        "spidr_base": "#2E86AB",
        "spidr": "#2E86AB", 
        "hubert-500": "#A23B72",
        "mhubert": "#F18F01",
    }
    
    for model in all_models:
        if model["arch"] == "lstm":
            labels.append(f"{model['encoder']}\nh={model['size']}")
        else:
            labels.append(f"{model['encoder']}\ne={model['size']}")
        
        accuracies.append(model["accuracy"])
        colors.append(encoder_colors.get(model["encoder"], "#808080"))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.9), 8))
    
    # Create custom x positions with tighter spacing within groups, larger gap between
    x_positions = []
    current_x = 0
    for i, model in enumerate(all_models):
        x_positions.append(current_x)
        # If next model is different architecture, add extra gap
        if i < len(all_models) - 1 and all_models[i]["arch"] != all_models[i+1]["arch"]:
            current_x += 1.0 + 0.25  # bar width + extra gap
        else:
            current_x += 1.0  # normal spacing within group
    
    x = np.array(x_positions)
    bars = ax.bar(x, accuracies, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5, width=0.8)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
               f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    # Style plot
    prob_type = "Raw" if use_raw else "Length-Normalized"
    ax.set_ylabel("Discrimination Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"Lexical Discrimination\n({prob_type} Log-Probability)", 
                fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    
    # Add architecture labels below x-axis
    lstm_x_center = x[:len(lstm_models)].mean()
    gpt2_x_center = x[len(lstm_models):].mean()
    ax.text(lstm_x_center, -0.09, "LSTM", ha="center", va="top", fontsize=13, 
           fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(gpt2_x_center, -0.09, "GPT-2", ha="center", va="top", fontsize=13,
           fontweight="bold", transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="red", ls=":", lw=2, alpha=0.7, label="Chance (50%)")
    
    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = []
    for encoder in sorted(set(m["encoder"] for m in all_models)):
        color = encoder_colors.get(encoder, "#808080")
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=encoder))
    legend_elements.append(Line2D([0], [0], color="red", ls=":", lw=2, label="Chance"))  # type: ignore 
    
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%b%d").lower()  # e.g., "feb12"
    suffix = "_raw" if use_raw else ""
    output_path = output_dir / f"lexical_discrimination_{timestamp}{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved unified plot: {output_path}")
    print(f"  Models included: {len(all_models)} ({len(lstm_models)} LSTM, {len(gpt2_models)} GPT-2)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate unified comparison plot")
    parser.add_argument("--raw", action="store_true",
                       help="Use raw log_prob instead of normalized (default: normalized)")
    args = parser.parse_args()
    
    metadata_dir = Path.cwd() / "metadata" / "swuggy"
    output_dir = Path.cwd() / "figures"
    
    print(f"Scanning {metadata_dir} for models...")
    
    prob_type = "raw" if args.raw else "normalized"
    print(f"Creating unified plot ({prob_type} log-probability)...\n")
    
    create_unified_plot(metadata_dir, output_dir, args.raw)
    
    print(f"\nPlot saved to {output_dir}")
