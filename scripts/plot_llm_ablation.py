#!/usr/bin/env python3
"""Render the article's synthetic 2x2 interaction from its shared example data.

Run with Python and Matplotlib. No training measurements or network required.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/llm-training-metrics"
sys.path.insert(0, str(BUNDLE))
from metrics_lab import ABLATION_SCORES


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17})
    fig, ax = plt.subplots(figsize=(9.6, 6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcff")
    for keys, label, color, marker in [
        (("A", "C"), "No deduplication", "#3976b8", "o"),
        (("B", "D"), "With deduplication", "#be7427", "s"),
    ]:
        values = [ABLATION_SCORES[k] for k in keys]
        ax.plot([0, 1], values, label=label, color=color, linewidth=2.5,
                marker=marker, markersize=10)
        for x, key, y in zip([0, 1], keys, values):
            ax.annotate(f"{key}: {y:.0f}%", (x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=18, weight="bold")
    ax.set_xticks([0, 1], ["Full-sequence loss", "Response-only loss"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(68, 78.5)
    ax.set_yticks([68, 70, 72, 74, 76, 78])
    ax.set_ylabel("Task accuracy (%)", labelpad=10)
    ax.grid(axis="y", color="#dce3ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9aa9bb")
    ax.tick_params(axis="both", length=0, pad=10)
    ax.legend(loc="upper left", frameon=True, facecolor="white",
              edgecolor="#dce3ec", fontsize=14)
    fig.suptitle("Two factors, one interaction", fontsize=24, weight="bold", y=.96)
    fig.text(.5, .88, "Synthetic example: not measured model performance",
             ha="center", fontsize=13, color="#526176")
    fig.subplots_adjust(left=.14, right=.96, bottom=.16, top=.82)
    destination = BUNDLE / "assets/llm-ablation-interaction.png"
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
