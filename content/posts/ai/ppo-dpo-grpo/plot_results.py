"""Plot measured CSV output from rl_lab.py; requires matplotlib==3.10.6."""
import argparse
import csv
import math
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("training-curves.svg"))
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, sharey=True)
    colors = {"ppo": "#2563eb", "dpo": "#b45309", "grpo": "#047857"}
    for ax, (name, color) in zip(axes, colors.items()):
        with (args.input / f"{name}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        report_path = args.input / f"{name}.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        seed = report.get("settings", {}).get("seed", "not recorded")
        steps = [int(row["step"]) for row in rows]
        values = [float(row["expected_reward"]) for row in rows]
        if not rows or steps != sorted(set(steps)) or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid curve data: {name}")
        ax.plot(steps, values, color=color, linewidth=2.5, label=f"{name.upper()} (seed {seed})")
        ax.axhline(values[0], color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Expected reward")
        ax.legend(loc="lower right", frameon=False)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        last = rows[-1]
        cost = (f'{last["pair_presentations"]} pair presentations' if name == "dpo"
                else f'{last["sampled_actions"]} sampled actions')
        ax.text(.02, .82, cost + f'; {last["optimizer_steps"]} optimizer steps',
                transform=ax.transAxes, fontsize=9, color="#475569")
    axes[-1].set_xlabel("Outer iteration within each method (different update/data budgets)")
    fig.suptitle("CPU contextual-bandit experiments\n"
                 "Learning checks — not an equal-cost algorithm benchmark", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Path glyphs make the SVG independent of the reader's installed fonts.
    with plt.rc_context({"svg.fonttype": "path", "svg.hashsalt": "chase-rl-lab"}):
        fig.savefig(args.output, bbox_inches="tight", facecolor="white",
                    metadata={"Date": None} if args.output.suffix == ".svg" else {})
    plt.close(fig)
    if args.output.suffix == ".svg":
        lines = args.output.read_text().splitlines()
        args.output.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
