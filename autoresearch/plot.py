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

    # Discarded: red X, not connected, with labels
    if discard_x:
        ax.scatter(
            discard_x, discard_y,
            marker="x", color="#e74c3c", s=50, linewidths=1.5,
            alpha=0.5, zorder=2, label="Discarded",
        )
        for x, y, label in zip(discard_x, discard_y, discard_labels):
            if label:
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, -10),
                    fontsize=6,
                    color="#e74c3c",
                    alpha=0.6,
                    rotation=20,
                )

    # Crashed: gray X, with labels
    if crash_x:
        ax.scatter(
            crash_x, crash_y,
            marker="x", color="#95a5a6", s=50, linewidths=1.5,
            alpha=0.5, zorder=2, label="Crashed",
        )
        for x, y, label in zip(crash_x, crash_y, crash_labels):
            if label:
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, -10),
                    fontsize=6,
                    color="#95a5a6",
                    alpha=0.6,
                    rotation=20,
                )

    # Kept: green dots connected by line (running best)
    if kept_x:
        ax.plot(
            kept_x, kept_y,
            color="#27ae60", linewidth=2, zorder=3, label="Running best",
        )
        ax.scatter(
            kept_x, kept_y,
            color="#27ae60", s=80, zorder=4, edgecolors="white", linewidths=0.5,
        )

        # Labels on kept experiments
        for x, y, label in zip(kept_x, kept_y, kept_labels):
            if label:
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=7,
                    color="#2c3e50",
                    alpha=0.8,
                    rotation=20,
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
    plot_progress(tsv)
