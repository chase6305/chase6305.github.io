"""Plot the saved constructed experiment; requires NumPy and Matplotlib only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(reference, output):
    result = json.loads(Path(reference).read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelcolor": "#26364c", "text.color": "#26364c", "axes.edgecolor": "#64748b",
        "xtick.color": "#475569", "ytick.color": "#475569", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 8.8))
    figure.subplots_adjust(left=.15, right=.96, top=.89, bottom=.15, hspace=.72)
    figure.suptitle("Two-link identification: held-out checks", fontsize=16, weight="bold", y=.98)
    singular = np.asarray(result["scaled_singular_values"])
    relative = singular / singular[0]
    indices = np.arange(1, len(singular) + 1)
    retained = np.arange(len(singular)) < result["retained_rank"]
    floor = 1e-18
    axes[0].scatter(indices[retained], relative[retained], s=45, color="#3978ae", marker="o", zorder=3,
                    label="Retained directions")
    below = relative <= floor
    axes[0].scatter(indices[~retained & ~below], relative[~retained & ~below], s=42, color="#64748b", marker="x", zorder=3)
    axes[0].scatter(indices[below], np.full(np.sum(below), floor), s=45, color="#64748b", marker="v", zorder=3)
    axes[0].axhline(result["svd_relative_cutoff"], color="#b86518", linestyle="--", linewidth=1.5,
                    label="Relative cutoff: 1e-10")
    axes[0].set(yscale="log", ylim=(3e-19, 3), xlim=(.3, 20.7), xticks=[1, 6, 7, 10, 15, 20],
                yticks=[1, 1e-5, 1e-10, 1e-15, 1e-18], xlabel="Singular-value index", ylabel="Relative singular value")
    axes[0].set_title("A. 6 retained directions from 20 columns", fontsize=14, pad=12, loc="left")
    axes[0].legend(loc="center right", fontsize=11, frameon=True, framealpha=1, facecolor="white", edgecolor="#d6dee8")
    axes[0].grid(axis="y", alpha=.18)
    axes[0].text(0., -.38, "Down triangles: values at or below the 1e-18 display floor.",
                 transform=axes[0].transAxes, fontsize=10.5)
    positions = np.arange(2)
    before = result["per_joint_nominal_urdf_validation_rmse_nm"]
    after = result["per_joint_validation_rmse_nm"]
    bars_before = axes[1].bar(positions - .18, before, width=.32, color="#f4dfc2", edgecolor="#b86518",
                               linewidth=1.2, label="Nominal URDF")
    bars_after = axes[1].bar(positions + .18, after, width=.32, color="#d5e7f7", edgecolor="#3978ae",
                              linewidth=1.2, hatch="//", label="Identified prediction")
    axes[1].bar_label(bars_before, labels=[f"{value:.4f}" for value in before], padding=4, fontsize=12)
    axes[1].bar_label(bars_after, labels=[f"{value:.4f}" for value in after], padding=4, fontsize=12)
    axes[1].set(xticks=positions, xticklabels=["Joint 1", "Joint 2"], ylabel="Torque RMSE (N m)", ylim=(0, .32))
    axes[1].set_title("B. Same held-out noisy torque samples", fontsize=14, pad=12, loc="left")
    axes[1].legend(loc="upper left", fontsize=11, frameon=True, framealpha=1, facecolor="white", edgecolor="#d6dee8")
    axes[1].grid(axis="y", alpha=.18)
    axes[1].set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(.15, .035, "Constructed data; inertia mismatch + 0.02 N m torque noise.\n"
                "Prediction accuracy does not make the 20 parameters unique.", fontsize=11, linespacing=1.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, metadata={"Software": "Matplotlib; saved identification reference data"})
    plt.close(figure)


if __name__ == "__main__":
    bundle = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=bundle / "assets/pinocchio-reference.json")
    parser.add_argument("--output", type=Path, default=bundle / "assets/identification-held-out.png")
    args = parser.parse_args()
    make_figure(args.reference, args.output)
