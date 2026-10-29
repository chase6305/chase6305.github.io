"""Plot the constructed doorway geometry; Matplotlib, no robot dependencies."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tamp_constraints import BLOCKER_RADIUS, GRASPS, HALF_OPENING, PARKING, refine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)
    for axis, (name, parked) in zip(axes, PARKING.items()):
        radius = GRASPS["narrow"]
        result = refine(parked, radius)
        valid = result["status"] == "feasible"
        for low, high in [(-1.4, -HALF_OPENING), (HALF_OPENING, 1.6)]:
            axis.plot([0, 0], [low, high], color="#4b5563", linewidth=8, solid_capstyle="butt")
        axis.add_patch(Circle(parked, BLOCKER_RADIUS, facecolor="#f5d4b6", edgecolor="#9d531b", linewidth=2,
                              label="Parked blocker (R = 0.40 m)"))
        axis.plot(*parked, marker="+", color="#713f16", markersize=8)
        axis.add_patch(Circle((-1.6, 0), radius, facecolor="#e3d8f3", edgecolor="#70539a", linewidth=1.8))
        axis.plot([-2, 2], [0, 0], linestyle="-" if valid else "--",
                  color="#327b4d" if valid else "#aa553b", linewidth=2.1,
                  label="Verified path" if valid else "Rejected candidate")
        if not valid:
            axis.plot(0, 0, marker="x", color="#8b2e21", markersize=10, markeredgewidth=2)
        axis.scatter([-2, 2], [0, 0], marker="s", color="#355875", s=26, zorder=5)
        axis.text(-2, -.29, "Start", ha="center", fontsize=10)
        axis.text(2, -.29, "Goal", ha="center", fontsize=10)
        axis.text(-1.6, .37, "Carry radius 0.25 m", ha="center", color="#604181", fontsize=9)
        axis.set(xlim=(-2.4, 2.4), ylim=(-1.4, 1.65), xlabel="x [m]", ylabel="y [m]",
                 title=f"{name.capitalize()} parking + narrow grasp: {'verified' if valid else 'blocked'}")
        axis.set_aspect("equal")
        axis.grid(alpha=.15)
        axis.legend(loc="lower left", fontsize=9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir/"doorway-refinement.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
