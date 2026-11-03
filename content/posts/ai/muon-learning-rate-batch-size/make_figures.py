"""Recreate the three local-model figures. Requires NumPy and Matplotlib."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batch_math import gaussian_curvature, mean_field_lr

OUT = Path(__file__).parent / "assets"
BLUE, PURPLE, ORANGE, GREEN = "#3976ae", "#8661aa", "#c2772c", "#348474"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.18, "savefig.dpi": 180})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    x = np.logspace(-3, 3, 1000)
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    ax.plot(x, [mean_field_lr(float(b), 1) for b in x], color=PURPLE, lw=2.5, label="Muon mean-field model")
    ax.plot(x, 1 / (1 + 1 / x), color=BLUE, lw=2, ls="-.", label="SGD local quadratic model")
    ax.plot(x, np.sqrt(x), color=ORANGE, ls="--", label="Small-batch square-root extrapolation")
    ax.axhline(1, color=GREEN, ls=":", label="Large-batch limit")
    ax.set(xscale="log", xlabel="Batch / critical batch", ylabel="Local optimal LR / large-batch limit",
           title="Square-root growth has a limited range", ylim=(0, 1.5), xlim=(1e-3, 1e3))
    ax.legend(fontsize=11, loc="upper left")
    save(fig, "batch-scaling-law.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    spectra = [[1, 1], [np.sqrt(1.99), 0.1]]
    colors = [BLUE, PURPLE]
    for ax, sigmas, color, title in zip(axes, spectra, colors, ["Flat spectrum", "Uneven spectrum"]):
        vals = [s / np.sqrt(s*s + 0.1) for s in sigmas]
        bars = ax.bar([0, 1], vals, color=color, width=0.5, hatch=["", "//"])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_xticks([0, 1], [f"sigma = {s:.3g}" for s in sigmas])
        ax.set(title=title, ylim=(0, 1.12), ylabel="Direction amplitude in Gram approximation")
        ax.set_axisbelow(True)
    fig.suptitle("Same squared Frobenius norm = 2; noise Gram term = 0.1 I")
    save(fig, "gram-anisotropy.png")

    batches = np.logspace(0, 7, 160)
    results = [gaussian_curvature(float(b), noise_std=100) for b in batches]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    axes[0].plot(batches, [r["lr"] for r in results], color=PURPLE, lw=2.5,
                 label="Gaussian integral: a(B) / b(B)")
    axes[0].axhline(0.01, color=GREEN, ls="--", label="Large-batch limit = 0.01")
    axes[0].set(xscale="log", xlabel="Batch B", ylabel="Local optimal LR", ylim=(0, 0.025),
                title="A nonmonotone local optimum")
    axes[0].legend(fontsize=10, loc="lower right")
    axes[1].plot(batches, [r["alignment"] for r in results], color=BLUE, lw=2,
                 label="Alignment a(B)")
    axes[1].plot(batches, [r["curvature"] / 100 for r in results], color=ORANGE, ls="--", lw=2,
                 label="Curvature b(B) / 100")
    axes[1].set(xscale="log", xlabel="Batch B", ylabel="Normalized quantity", ylim=(0, 1.05),
                title="Alignment and curvature both change")
    axes[1].legend(fontsize=10, loc="lower right")
    fig.suptitle("G = (1, 0); Hessian = diag(100, 1); per-sample noise std s = 100")
    save(fig, "curvature-counterexample.png")


if __name__ == "__main__":
    main()
