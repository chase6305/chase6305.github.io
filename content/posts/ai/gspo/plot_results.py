"""Optional: plot_results.py --input results.json --output figures (Matplotlib 3.10.6)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results.json"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    x = np.arange(2)
    for shift, kind, color in ((-.18, "grpo", "#d97706"), (.18, "gspo", "#2563eb")):
        values = report["cancellation"]["1.0"][kind]["gradient"]
        bars = ax.bar(x + shift, values, .34, color=color, label=kind.upper())
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(x, ["Token 1 (ratio 0.5)", "Token 2 (ratio 2.0)"])
    ax.set_ylabel("Derivative of maximized objective w.r.t. token logp")
    ax.set_ylim(0, .65)
    ax.set_title("Same response: A = +1, epsilon = 0.2, sequence ratio = 1")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "ratio-gradient.png", dpi=180)
    plt.close(fig)
    values = report["length_normalization"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot([r["length"] for r in values], [r["raw_ratio"] for r in values], "o-",
            color="#d97706", label="Full sequence ratio: 1.01^L")
    ax.plot([r["length"] for r in values], [r["gspo_ratio"] for r in values], "o-",
            color="#2563eb", label="GSPO ratio: 1.01")
    ax.set_yscale("log")
    ax.set_xlabel("Response length L (tokens)")
    ax.set_ylabel("Likelihood ratio (log scale)")
    ax.set_title("Uniform per-token ratio of 1.01; illustrative values")
    ax.grid(alpha=.2)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "length-normalization.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
