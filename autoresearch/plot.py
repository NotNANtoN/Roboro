"""Generate autoresearch progress plot from results.tsv.

Reads results.tsv and produces progress.png:
  - Green dots + line: kept improvements (running best)
  - Red X: discarded experiments
  - Gray X: crashed experiments
  - Labels on kept experiments with descriptions

Usage:
    python plot.py              # reads results.tsv, writes progress.png
    python plot.py results.tsv  # custom input file
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_progress(tsv_path: str = "results.tsv", out_path: str = "progress.png") -> None:
    rows = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if not rows:
        print("No results to plot.")
        return

    kept_x, kept_y, kept_labels = [], [], []
    discard_x, discard_y, discard_labels = [], [], []
    crash_x, crash_y, crash_labels = [], [], []

    for i, row in enumerate(rows):
        exp_num = i + 1
        status = row["status"].strip()
        score = float(row["score"])
        desc = row.get("description", "").strip()

        if status == "keep":
            kept_x.append(exp_num)
            kept_y.append(score)
            kept_labels.append(desc)
        elif status == "crash":
            crash_x.append(exp_num)
            crash_y.append(0.0)
            crash_labels.append(desc)
        else:  # discard
            discard_x.append(exp_num)
            discard_y.append(score)
            discard_labels.append(desc)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    import re

    def short_desc(desc, max_len=28):
        """Strip 'expNN: ' prefix and truncate."""
        desc = re.sub(r"^exp\d+[:/]?\s*", "", desc)
        return (desc[:max_len-1] + "…") if len(desc) > max_len else desc

    # Discarded: red X with sparse truncated labels (every other to reduce clutter)
    if discard_x:
        ax.scatter(
            discard_x, discard_y,
            marker="x", color="#e74c3c", s=50, linewidths=1.5,
            alpha=0.4, zorder=2, label="Discarded",
        )
        for i, (x, y, label) in enumerate(zip(discard_x, discard_y, discard_labels)):
            if label:
                ax.annotate(
                    short_desc(label, 22),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, -7),
                    fontsize=4,
                    color="#e74c3c",
                    alpha=0.35,
                    rotation=30,
                )

    # Crashed: gray X with sparse labels
    if crash_x:
        ax.scatter(
            crash_x, crash_y,
            marker="x", color="#95a5a6", s=50, linewidths=1.5,
            alpha=0.4, zorder=2, label="Crashed",
        )
        for i, (x, y, label) in enumerate(zip(crash_x, crash_y, crash_labels)):
            if label:
                ax.annotate(
                    short_desc(label, 22),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, -7),
                    fontsize=4,
                    color="#95a5a6",
                    alpha=0.35,
                    rotation=30,
                )

    # Kept: green dots connected by line
    if kept_x:
        ax.plot(
            kept_x, kept_y,
            color="#27ae60", linewidth=2, zorder=3, label="Running best",
        )
        ax.scatter(
            kept_x, kept_y,
            color="#27ae60", s=80, zorder=4, edgecolors="white", linewidths=0.5,
        )

        # Spread labels with cycling offsets to avoid overlap
        offsets = [(10, 22), (-60, -24), (30, 16), (-5, -30), (20, 22), (-15, -24), (10, 24)]
        for idx, (x, y, label) in enumerate(zip(kept_x, kept_y, kept_labels)):
            if label:
                dx, dy = offsets[idx % len(offsets)]
                ax.annotate(
                    short_desc(label),
                    (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=7,
                    fontweight="bold",
                    color="#2c3e50",
                    alpha=0.85,
                    ha="left",
                    arrowprops=dict(arrowstyle="-", color="#2c3e50", alpha=0.3, lw=0.5),
                )

    n_total = len(rows)
    n_kept = len(kept_x)

    ax.set_title(
        f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Score (higher is better)", fontsize=12)

    ax.set_xlim(0, n_total + 1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    tsv = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
    out = sys.argv[2] if len(sys.argv) > 2 else "progress.png"
    plot_progress(tsv, out)
