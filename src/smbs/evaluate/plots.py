"""Generate comparison plots across models and encoders."""

import re
from datetime import datetime

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from smbs.config import METADATA_DIR, FIGURES_DIR
from smbs.evaluate.metrics import discrimination_accuracy

sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", font_scale=1.2)


def parse_model_info(filepath: Path):
    """Extract encoder, architecture, and size from parquet filename."""
    stem = filepath.stem

    if "lstm" in stem:
        arch = "lstm"
        match = re.search(r"_h(\d+)", stem)
        size = int(match.group(1)) if match else 0
    elif "gpt2" in stem:
        arch = "gpt2"
        match = re.search(r"_e(\d+)", stem)
        size = int(match.group(1)) if match else 0
    else:
        return None, "unknown", 0

    if arch == "lstm":
        encoder = stem.split("_lstm")[0]
    else:
        encoder = stem.split("_gpt2")[0]

    return encoder, arch, size


def create_unified_plot(use_raw: bool = False):
    """Create single plot comparing all models across all encoders."""
    metadata_dir = METADATA_DIR / "swuggy"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prob_col = "log_prob" if use_raw else "log_prob_norm"

    models = []

    for parquet_file in metadata_dir.glob("*.parquet"):
        try:
            df = pl.read_parquet(parquet_file)
            if prob_col not in df.columns or "positive" not in df.columns:
                continue
        except Exception:
            continue

        encoder, arch, size = parse_model_info(parquet_file)
        if arch not in ["lstm", "gpt2"]:
            continue

        group_col = "group_id" if "group_id" in df.columns else "word_id"
        accuracy = discrimination_accuracy(df, prob_col, group_col)

        models.append({
            "encoder": encoder,
            "arch": arch,
            "size": size,
            "accuracy": accuracy,
            "filename": parquet_file.stem,
        })

    if not models:
        print("No valid models found")
        return

    lstm_models = sorted([m for m in models if m["arch"] == "lstm"],
                         key=lambda x: (x["encoder"], x["size"]))
    gpt2_models = sorted([m for m in models if m["arch"] == "gpt2"],
                         key=lambda x: (x["encoder"], x["size"]))
    all_models = lstm_models + gpt2_models

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

    fig, ax = plt.subplots(figsize=(max(12, len(all_models) * 0.9), 8))

    x_positions = []
    current_x = 0
    for i, model in enumerate(all_models):
        x_positions.append(current_x)
        if i < len(all_models) - 1 and all_models[i]["arch"] != all_models[i + 1]["arch"]:
            current_x += 1.25
        else:
            current_x += 1.0

    x = np.array(x_positions)
    bars = ax.bar(x, accuracies, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5, width=0.8)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.008,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    prob_type = "Raw" if use_raw else "Length-Normalized"
    ax.set_ylabel("Discrimination Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"Lexical Discrimination\n({prob_type} Log-Probability)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)

    if lstm_models:
        lstm_x_center = x[:len(lstm_models)].mean()
        ax.text(lstm_x_center, -0.09, "LSTM", ha="center", va="top", fontsize=13,
                fontweight="bold", transform=ax.get_xaxis_transform())
    if gpt2_models:
        gpt2_x_center = x[len(lstm_models):].mean()
        ax.text(gpt2_x_center, -0.09, "GPT-2", ha="center", va="top", fontsize=13,
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color="red", ls=":", lw=2, alpha=0.7, label="Chance (50%)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = []
    for enc in sorted(set(m["encoder"] for m in all_models)):
        color = encoder_colors.get(enc, "#808080")
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=enc))
    legend_elements.append(Line2D([0], [0], color="red", ls=":", lw=2, label="Chance"))  # type: ignore

    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%b%d").lower()
    suffix = "_raw" if use_raw else ""
    output_path = FIGURES_DIR / f"lexical_discrimination_{timestamp}{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved unified plot: {output_path}")
    print(f"  Models included: {len(all_models)} ({len(lstm_models)} LSTM, {len(gpt2_models)} GPT-2)")
