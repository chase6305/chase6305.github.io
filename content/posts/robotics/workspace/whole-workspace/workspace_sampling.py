"""Compare joint-uniform and area-uniform samples for a planar 2R annulus.

Python 3.10+, NumPy and Matplotlib. No joint limits, obstacles or orientation
constraints. The area sampler describes this analytic annulus only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def joint_radial_cdf(radius, l1=1.0, l2=0.6):
    cosine = (np.asarray(radius) ** 2 - l1**2 - l2**2) / (2 * l1 * l2)
    return 1 - np.arccos(np.clip(cosine, -1, 1)) / np.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("workspace-sampling"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    count = 20000
    l1, l2 = 1.0, 0.6
    inner, outer = abs(l1 - l2), l1 + l2
    q = rng.uniform(0, 2 * np.pi, (count, 2))
    joint_xy = np.column_stack((l1 * np.cos(q[:, 0]) + l2 * np.cos(q.sum(axis=1)),
                                l1 * np.sin(q[:, 0]) + l2 * np.sin(q.sum(axis=1))))
    joint_radius = np.linalg.norm(joint_xy, axis=1)
    area_radius = np.sqrt(rng.uniform(inner**2, outer**2, count))
    angle = rng.uniform(0, 2 * np.pi, count)
    area_xy = area_radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    assert np.all((joint_radius >= inner - 1e-12) & (joint_radius <= outer + 1e-12))
    sorted_radius = np.sort(joint_radius)
    cdf = joint_radial_cdf(sorted_radius)
    ecdf_after = np.arange(1, count + 1) / count
    ecdf_before = np.arange(count) / count
    ks_error = max(np.max(ecdf_after - cdf), np.max(cdf - ecdf_before))
    assert ks_error < .02
    np.testing.assert_allclose(joint_radial_cdf([inner, outer]), [0, 1], atol=1e-7)
    threshold = 1.5
    report = {
        "seed": 42, "samples_per_distribution": count, "link_lengths_m": [l1, l2],
        "radial_bounds_m": [inner, outer], "joint_cdf_max_error": float(ks_error),
        "outer_band_threshold_m": threshold,
        "joint_outer_band_observed_fraction": float(np.mean(joint_radius >= threshold)),
        "joint_outer_band_analytic_fraction": float(1 - joint_radial_cdf(threshold)),
        "area_outer_band_observed_fraction": float(np.mean(area_radius >= threshold)),
        "area_outer_band_analytic_fraction": (outer**2 - threshold**2) / (outer**2 - inner**2),
    }
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    for ax, xy, title, color in zip(axes[:2], [joint_xy, area_xy],
            ["Uniform joint angles", "Uniform annulus area"], ["#397ec0", "#8d6bbe"]):
        ax.scatter(xy[:, 0], xy[:, 1], s=.7, alpha=.35, c=color, rasterized=True)
        for radius in (inner, outer):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#293c55", linewidth=.7))
        ax.set(xlabel="x [m]", ylabel="y [m]", title=title, aspect="equal",
               xlim=(-1.7, 1.7), ylim=(-1.7, 1.7))
        ax.grid(alpha=.12)
    radius = np.linspace(inner, outer, 500)
    axes[2].plot(sorted_radius, ecdf_after, color="#397ec0", lw=2, label="Joint samples")
    axes[2].plot(radius, joint_radial_cdf(radius), color="#23354d", ls="--", lw=1.4,
                 label="Joint analytic CDF")
    axes[2].plot(radius, (radius**2 - inner**2) / (outer**2 - inner**2),
                 color="#8d6bbe", lw=2, label="Area analytic CDF")
    axes[2].set(xlabel="Radius [m]", ylabel="Fraction with distance <= radius",
                title="Same support, different density", ylim=(0, 1), xlim=(inner, outer))
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=9, loc="upper left", frameon=False)
    fig.suptitle("Planar 2R: L1 = 1.0 m, L2 = 0.6 m | 20,000 samples per cloud | seed 42", fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "workspace-sampling.png", dpi=150, facecolor="white")
    plt.close(fig)
    (args.output_dir / "workspace-sampling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
