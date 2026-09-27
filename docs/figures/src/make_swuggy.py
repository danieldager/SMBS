"""Draw docs/figures/swuggy.png. No network.
    uv run --with matplotlib python docs/figures/src/make_swuggy.py

Values are the bar labels of the committed result plots (length-normalised
sWuggy accuracy; the per-run CSVs they were drawn from are not in the repo):
  Feb  <- figures/lexical_discrimination_feb12.png
  Mar  <- figures/lexical_discrimination_mar26.png
          (same six values as figures/lexical_discrimination_feb15.png)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).parent.parent
# (model, encoder, feb12, mar26); None = no run in that plot
ROWS = [
    ("LSTM", "HuBERT-500", 0.549, 0.565),
    ("LSTM", "mHuBERT", 0.567, 0.602),
    ("LSTM", "SpidR", 0.575, 0.645),
    ("GPT-2", "HuBERT-500", 0.583, 0.600),
    ("GPT-2", "mHuBERT", 0.576, 0.625),
    ("GPT-2", "SpidR", None, 0.658),
]
SIZE = {"LSTM": "hidden 256 in Feb, 1024 in Mar", "GPT-2": "768-dim, 12 layers"}

# dataviz reference palette (light surface), as in the other landing pages
COL = {"HuBERT-500": "#2a78d6", "mHuBERT": "#eb6834", "SpidR": "#1baf7a"}
NEUTRAL = "#b8b7b0"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
W, DPI = 8.0, 200  # 1600 px wide
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 11, "axes.labelcolor": INK2, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "xtick.color": INK2, "ytick.color": INK2, "legend.fontsize": 10.5,
    "legend.frameon": False, "text.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    fig, ax = plt.subplots(figsize=(W, 4.6), layout="constrained")
    ys, labels, y = [], [], 0.0
    for i, (model, enc, feb, mar) in enumerate(ROWS):
        if i and ROWS[i - 1][0] != model:
            y -= 0.7
        if i == 0 or ROWS[i - 1][0] != model:
            t = ax.text(0.4505, y + 0.62, model, fontsize=11.5, fontweight="bold", color=INK, va="center")
            ax.annotate(SIZE[model], xy=(1, 0.5), xycoords=t, xytext=(8, 0), textcoords="offset points",
                        fontsize=9.5, color=MUTED, va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
        c = COL[enc]
        if feb is not None:
            ax.plot([feb, mar], [y, y], color=c, linewidth=2, alpha=0.35, solid_capstyle="butt", zorder=1)
            ax.scatter([feb], [y], s=70, color=NEUTRAL, edgecolor="white", linewidth=2, zorder=2)
            ax.text(feb - 0.004, y, f"{feb:.3f}", ha="right", va="center", fontsize=9.5, color=MUTED)
        else:
            ax.text(mar - 0.012, y, "no February run", ha="right", va="center", fontsize=9.5, color=MUTED)
        ax.scatter([mar], [y], s=95, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(mar + 0.005, y, f"{mar:.3f}", ha="left", va="center", fontsize=10.5, color=INK,
                fontweight="bold" if mar == 0.658 else "normal")
        ys.append(y); labels.append(enc)
        y -= 0.75
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    ax.text(0.5015, ys[-1] - 0.55, "chance 0.5", fontsize=9.5, color=MUTED, va="center")
    ax.set_yticks(ys, labels)
    ax.set_xlim(0.45, 0.70)
    ax.set_ylim(ys[-1] - 0.8, ys[0] + 0.95)
    ax.set_xlabel("sWuggy accuracy (length-normalised log-probability)")
    ax.set_title("sWuggy accuracy, February to March 2026")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=NEUTRAL,
                      markeredgecolor="white", label="12 Feb runs"),
               Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=INK2,
                      markeredgecolor="white", label="26 Mar results (colour = encoder)")]
    ax.legend(handles=handles, loc="upper right", ncols=2, borderaxespad=0.1, frameon=True, framealpha=1, facecolor="white", edgecolor="white")
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2)
    fig.savefig(OUT / "swuggy.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
