#!/usr/bin/env python3
"""Deterministic figures: one illustrative calculation and one paper data chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from survey_lab import PAPER_COUNTS, chain_success


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9ca9b9")
    ax.grid(axis="y", color="#e4eaf2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#334155")


def main():
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 19, "axes.labelsize": 14,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    steps = list(range(1, 21))
    for prob, color, marker in ((0.90, "#c57437", "s"),
                                (0.95, "#7257a8", "o"),
                                (0.99, "#2d8b70", "^")):
        ax.plot(steps, [chain_success(prob, n) for n in steps], color=color,
                marker=marker, markevery=[0, 4, 9, 14, 19], linewidth=2.3,
                label=f"Per-stage conditional success = {prob:.0%}")
    ax.set(title="Long tasks amplify local failures", xlabel="Number of required stages",
           ylabel="Probability of completing every stage", xlim=(1, 20), ylim=(0, 1.03))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    style(ax)
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))
    fig.text(0.5, 0.018, "Illustrative calculation: constant conditional probability; no recovery.",
             ha="center", fontsize=10, color="#526174")
    fig.savefig(assets / "long-task-success.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    bars = ax.bar([str(y) for y in PAPER_COUNTS], list(PAPER_COUNTS.values()),
                  color=["#b7cee9"] * 5 + ["#9b86bf"], width=0.62,
                  edgecolor="#536d91", linewidth=0.8)
    ax.bar_label(bars, padding=5, fontsize=13)
    ax.set(title="Papers in the survey's retrieval set", xlabel="Publication year",
           ylabel="Number of papers", ylim=(0, 330))
    style(ax)
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.935))
    fig.text(0.5, 0.018, "Source: arXiv:2405.14093v8, Fig. 11(a). Retrieval period: 2020–2025; total: 393.",
             ha="center", fontsize=10, color="#526174")
    fig.savefig(assets / "survey-paper-counts.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
