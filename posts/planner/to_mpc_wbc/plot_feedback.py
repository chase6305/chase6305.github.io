"""Plot feedback_demo.py outputs; optional dependency: Matplotlib 3.10.6."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results-feedback"))
    parser.add_argument("--output", type=Path, default=Path("results-feedback/feedback-comparison.png"))
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, layout="constrained")
    for name, color, label in (("replay", "#d97706", "Replay nominal commands"),
                                ("feedback", "#2563eb", "Replan from measured state")):
        with (args.input / f"{name}.csv").open() as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        times, positions = [0.], [0.]
        for row in rows:
            if row["offset"]:
                times.append(row["time"] - .1)
                positions.append(row["measured_position"])
            times.append(row["time"])
            positions.append(row["position"])
        axes[0].plot(times, positions, color=color, linewidth=2, label=label)
        axes[1].step([r["time"] - .1 for r in rows] + [rows[-1]["time"]],
                     [r["velocity"] for r in rows] + [rows[-1]["velocity"]],
                     where="post", color=color, linewidth=2)
    axes[0].axhline(.3, color="#475569", linestyle=":", label="Target: 0.3 rad")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Position (rad)")
    axes[0].set_ylim(.20, .325)
    axes[1].set_ylabel("Velocity command (rad/s)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_xlim(2.5, 5.)
    for ax in axes:
        ax.axvline(3., color="#64748b", linestyle="--", linewidth=1)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feedback after a synthetic -0.08 rad state offset at t = 3 s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
