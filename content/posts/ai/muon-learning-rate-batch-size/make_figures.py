"""Recreate the four local-model figures. Requires NumPy and Matplotlib."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batch_math import gaussian_curvature, gaussian_sign_lr, local_budget_scores, mean_field_lr
from mean_field_probe import report as gaussian_matrix_report

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    ax = axes[0]
    ax.plot(x, [mean_field_lr(float(b), 1) for b in x], color=PURPLE, lw=2.5, label="Muon mean-field model")
    ax.plot(x, 1 / (1 + 1 / x), color=BLUE, lw=2, ls="-.", label="SGD local quadratic model")
    ax.plot(x, np.sqrt(x), color=ORANGE, ls="--", label="Small-batch square-root extrapolation")
    ax.axhline(1, color=GREEN, ls=":", label="Large-batch limit")
    ax.set(xscale="log", xlabel="Batch / model noise scale", ylabel="Local optimal LR / large-batch limit",
           title="Conditional scaling models", ylim=(0, 1.5), xlim=(1e-3, 1e3))
    ax.legend(fontsize=13, loc="upper left")
    ax = axes[1]
    ax.plot(x, [mean_field_lr(float(b), 1) for b in x], color=PURPLE, lw=2.5, label="Mean-field approximation")
    ax.plot(x, [gaussian_sign_lr(float(b), 1) for b in x], color=GREEN, ls="--", lw=2, label="Exact Gaussian sign expectation")
    ax.set(xscale="log", xlabel="Batch / critical batch", ylabel="Local optimal LR / large-batch limit",
           title="A 1 x 1 Gaussian check", ylim=(0, 1.05), xlim=(1e-3, 1e3))
    ax.text(0.04, 0.73, "At B = Bc:\nexact 0.683\nmean field 0.707", transform=ax.transAxes,
            fontsize=13, va="top", bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.4"})
    ax.legend(fontsize=13, loc="lower right")
    save(fig, "batch-scaling-law.png")

    probe = gaussian_matrix_report()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), layout="constrained")
    spectra = [[1, 1], [np.sqrt(1.99), 0.1]]
    for ax, sigmas, name, title in zip(axes, spectra, ["flat", "uneven"], ["Flat signal spectrum", "Uneven signal spectrum"]):
        case = probe["cases"][name]
        vals = np.diag(case["gram_prediction"])
        means = np.diag(case["mean_polar"])
        errors = 1.96*np.diag(case["standard_error"])
        predicted = ax.bar(np.array([0,1])-0.15, vals, color=BLUE, width=0.29, label="Gram approximation")
        measured = ax.bar(np.array([0,1])+0.15, means, color=GREEN, width=0.29, hatch="//",
                          yerr=errors, capsize=4, label="SVD mean + approx. 95% interval")
        ax.bar_label(predicted, fmt="%.3f", padding=5, fontsize=13)
        ax.bar_label(measured, fmt="%.3f", padding=5, fontsize=13)
        ax.set_xticks([0, 1], [f"sigma = {s:.3g}" for s in sigmas])
        ax.set(title=title, ylim=(0, 1.12), ylabel="Mean update along signal singular direction")
        ax.set_axisbelow(True)
    fig.suptitle("Same squared Frobenius norm = 2\n200,000 Gaussian SVD draws per case", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=13)
    save(fig, "gram-anisotropy.png")

    batches = np.logspace(0, 7, 160)
    results = [gaussian_curvature(float(b), noise_std=100) for b in batches]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    axes[0].plot(batches, [r["lr"] for r in results], color=PURPLE, lw=2.5,
                 label="Gaussian integral: a(B) / b(B)")
    axes[0].axhline(0.01, color=GREEN, ls="--", label="Large-batch limit = 0.01")
    axes[0].set(xscale="log", xlabel="Batch B", ylabel="Local optimal LR", ylim=(0, 0.025),
                title="A nonmonotone local optimum")
    axes[0].legend(fontsize=13, loc="lower right")
    axes[1].plot(batches, [r["alignment"] for r in results], color=BLUE, lw=2,
                 label="Alignment a(B)")
    axes[1].plot(batches, [r["curvature"] / 100 for r in results], color=ORANGE, ls="--", lw=2,
                 label="Curvature b(B) / 100")
    axes[1].set(xscale="log", xlabel="Batch B", ylabel="Normalized quantity", ylim=(0, 1.05),
                title="Alignment and curvature both change")
    axes[1].legend(fontsize=13, loc="lower right")
    fig.suptitle("G = (1, 0); Hessian = diag(100, 1); per-sample noise std s = 100")
    save(fig, "curvature-counterexample.png")

    batches = np.logspace(0, 5, 800)
    scores = [local_budget_scores(float(b), 4096, 256) for b in batches]
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    settings = [("per_update", "Per optimizer update", "Gain / large-batch one-step gain", PURPLE, 1),
                ("per_sample", "Per sample", "Gain per sample\n(x 10,000)", BLUE, 10000),
                ("per_cost", "Per cost unit", "Gain per cost unit\n(x 10,000)", ORANGE, 10000)]
    for ax, (key, title, label, color, factor) in zip(axes, settings):
        ax.plot(batches, [r[key]*factor for r in scores], color=color, lw=2.5)
        ax.axvline(4096, color="#666666", ls=":", lw=1.2, label="Bc = 4096")
        ax.set(xscale="log", xlabel="Batch B", ylabel=label, title=title, xlim=(1, 1e5))
    axes[0].legend(loc="upper left", fontsize=13)
    peak = local_budget_scores(1024, 4096, 256)["per_cost"]*10000
    axes[2].plot([1024], [peak], "o", color="black", label="Maximum at B = 1024")
    axes[2].legend(loc="lower center", fontsize=13)
    fig.suptitle("Frozen local model; illustrative per-step cost C(B) = B + 256", fontsize=15)
    save(fig, "batch-budget-efficiency.png")


if __name__ == "__main__":
    main()
