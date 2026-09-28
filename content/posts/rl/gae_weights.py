"""Plot exact GAE residual coefficients; no training measurements.

Run: python -B gae_weights.py --output gae-residual-weights.png
Requires NumPy and Matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lag = np.arange(21)
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for lam, color, style in [(0, "#9865b5", ":"), (0.5, "#cf8737", "--"),
                               (0.95, "#3978b4", "-"), (1, "#438a70", "-.")]:
        coefficient = (0.99 * lam) ** lag
        ax.plot(lag, coefficient, label=f"lambda = {lam:g}", color=color,
                linestyle=style, marker="o", markersize=3, linewidth=2)
        print(f"lambda={lam:g}, weight at lag 10={coefficient[10]:.9f}")
    ax.set(title="GAE residual weights (gamma = 0.99)",
           xlabel="Residual lag l (steps)", ylabel="Coefficient (gamma * lambda)^l",
           xlim=(-0.3, 20.3), ylim=(-0.03, 1.06))
    ax.set_xticks(np.arange(0, 21, 2))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
