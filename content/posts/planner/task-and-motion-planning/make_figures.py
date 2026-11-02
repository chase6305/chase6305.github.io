"""Plot the constructed doorway geometry; Matplotlib, no robot dependencies."""
import argparse
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from tamp_constraints import BLOCKER_RADIUS, GRASPS, HALF_OPENING, PARKING, refine, verify_path
from tamp_search import GRID_BOUNDS, GRID_STEP, grid_motion
from tamp_kinematics import GRASPS as KIN_GRASPS, PLACEMENTS, LINKS, demo as kinematics_demo, joint_positions


def kinematics_figure(output):
    result = kinematics_demo()
    with plt.rc_context({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(12.6, 10.2))
        fig.subplots_adjust(left=.06, right=.98, top=.85, bottom=.14, hspace=.55, wspace=.28)
        for row, grasp in enumerate(KIN_GRASPS):
            for column, (name, placement) in enumerate(PLACEMENTS.items()):
                axis = axes[row, column]
                record = result["endpoint_witnesses"][name][grasp]
                solutions = record["solutions_rad"]
                axis.add_patch(Circle((0., 0.), LINKS[0] + LINKS[1],
                    facecolor="#f3f7fd", edgecolor="#9bb4d1", linewidth=1.1, linestyle="--"))
                for index, q in enumerate(solutions):
                    points = joint_positions(q)
                    axis.plot(*zip(*points), marker="o", markersize=5,
                        color="#356d9c" if index == 0 else "#8363a6",
                        linestyle="-" if index == 0 else "--", linewidth=2.6 if index == 0 else 1.7,
                        label="IK branch 1" if index == 0 else "IK branch 2")
                target = record["hand_target"]
                wrist = (target["x"] - LINKS[2] * math.cos(target["yaw"]),
                         target["y"] - LINKS[2] * math.sin(target["yaw"]))
                axis.scatter(*wrist, marker="x" if not solutions else "s", s=70,
                    color="#a34d37" if not solutions else "#405d76", zorder=6, label="Required wrist")
                axis.scatter(0., 0., marker="^", s=75, color="#334155", zorder=7)
                axis.scatter(placement.x, placement.y, facecolors="#f9dfbc", edgecolors="#9d6125",
                    s=110, linewidths=1.5, zorder=7, label="Object / hand origin")
                axis.annotate("", (target["x"] + .4 * math.cos(target["yaw"]),
                    target["y"] + .4 * math.sin(target["yaw"])),
                    (target["x"], target["y"]), arrowprops={"arrowstyle": "->", "color": "#9d6125", "lw": 1.8})
                status = "IK: 2 verified branches" if solutions else "No IK: wrist beyond 2 m"
                axis.set_title(f"{name.capitalize()} | {grasp} grasp\n{status}", fontsize=14,
                    color="#245a40" if solutions else "#8f4632")
                axis.text(.03, .04, f"Required wrist radius: {record['wrist_radius_m']:.3f} m",
                    transform=axis.transAxes, fontsize=14,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#c4ced8"})
                axis.set(xlim=(-2.6, 2.6), ylim=(-2.35, 2.35), xlabel="x [m]", ylabel="y [m]")
                axis.set_aspect("equal")
                axis.grid(alpha=.12)
        handles, labels = axes[0, 1].get_legend_handles_labels()
        handles.append(Line2D([], [], marker="x", color="#a34d37", linestyle="none", markersize=7))
        labels.append("Unreachable wrist")
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncols=3, fontsize=14)
        fig.suptitle("One grasp must work at both endpoints | Planar 3R: 1 + 1 + 0.25 m\n"
            "Dashed circle: wrist reach of the first two links. Arrows: required hand x-axis. Endpoint IK only.", fontsize=14)
        fig.savefig(output / "shared-grasp-ik.png", dpi=160, facecolor="white")
        plt.close(fig)


def grid_figure(output):
    parked, radius = (1., 0.), .25
    result = grid_motion(parked, radius)
    if result["status"] != "feasible":
        raise RuntimeError("Reference grid search did not return a path")
    with plt.rc_context({"font.size": 13}):
        fig, axis = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
        points = [(x * GRID_STEP, y * GRID_STEP)
                  for x in range(GRID_BOUNDS[0], GRID_BOUNDS[1] + 1)
                  for y in range(GRID_BOUNDS[2], GRID_BOUNDS[3] + 1)]
        free = [p for p in points if verify_path([p, p], radius, parked)["valid"]]
        axis.scatter(*zip(*free), color="#b5bdc7", s=10, label="Collision-free grid centers")
        for low, high in [(-1.4, -HALF_OPENING), (HALF_OPENING, 1.4)]:
            axis.plot([0, 0], [low, high], color="#4b5563", linewidth=8, solid_capstyle="butt")
        axis.add_patch(Circle(parked, BLOCKER_RADIUS, facecolor="#f5d4b6", edgecolor="#9d531b",
                              linewidth=2, label="Blocker: radius 0.40 m"))
        axis.add_patch(Circle(parked, BLOCKER_RADIUS + radius, facecolor="none", edgecolor="#9d531b",
                              linewidth=1.8, linestyle="--", label="Forbidden center radius: 0.65 m"))
        axis.add_patch(Circle((-2, 0), radius, facecolor="#e3d8f3", edgecolor="#70539a", linewidth=1.8))
        axis.plot([-2, 2], [0, 0], linestyle="--", color="#aa553b", linewidth=1.7,
                  label="First candidate: collision")
        axis.plot(*zip(*result["path_m"]), color="#327b4d", linewidth=2.8, marker="o", markersize=3.5,
                  label="A* retry: verified segments")
        axis.scatter([-2, 2], [0, 0], marker="s", color="#355875", s=30, zorder=5)
        axis.text(-2, -.37, "Start", ha="center")
        axis.text(2, .16, "Goal", ha="center")
        axis.set(xlim=(-2.4, 2.4), ylim=(-1.4, 1.4), xlabel="x [m]", ylabel="y [m]",
                 title="Finite-grid A* retry (spacing 0.20 m, carry radius 0.25 m)")
        axis.set_aspect("equal")
        axis.legend(loc="upper left", fontsize=12, ncols=2)
        fig.savefig(output / "grid-detour.png", dpi=160)
        plt.close(fig)


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
    grid_figure(args.output_dir)
    kinematics_figure(args.output_dir)


if __name__ == "__main__":
    main()
