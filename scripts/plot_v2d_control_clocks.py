#!/usr/bin/env python3
"""Plot exact configured durations for the article's Vega/GR00T recipe."""

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "content/posts/ai/video-to-data/assets/control-clock-durations.webp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    physics_dt = Fraction(1, 100)
    control_dt = 5 * physics_dt
    rows = [
        ("Arm delay\n4 physics substeps", 4 * physics_dt, "#fae7d4", "#c3732b"),
        ("Control interval\n5 physics substeps", control_dt, "#dcebf8", "#3173b8"),
        ("Execution length\n4 control intervals", 4 * control_dt, "#e2efdf", "#39814a"),
        ("Prediction horizon\n16 control intervals", 16 * control_dt, "#e8e0f5", "#8463b1"),
    ]
    assert [duration for _, duration, _, _ in rows] == [
        Fraction(1, 25), Fraction(1, 20), Fraction(1, 5), Fraction(4, 5)
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": "#25354a", "axes.labelcolor": "#25354a",
                         "axes.edgecolor": "#ccd5df", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.subplots_adjust(left=.32, right=.98, bottom=.24, top=.79)
    for index, (_, duration, face, edge) in enumerate(rows):
        y = len(rows) - 1 - index
        value = float(duration)
        ax.barh(y, value, height=.50, color=face, edgecolor=edge, linewidth=1.5, zorder=3)
        ax.text(value + .018, y, f"{value:.2f} s", va="center", color=edge,
                fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])
    ax.tick_params(axis="y", length=0, pad=10, labelsize=11.5)
    ax.set_xlim(0, .98)
    ax.set_ylim(-.6, len(rows) - .4)
    ax.set_xticks([0, .1, .2, .4, .6, .8], ["0", "0.1", "0.2", "0.4", "0.6", "0.8"])
    ax.set_xlabel("Configured duration (seconds)", labelpad=12)
    ax.grid(axis="x", color="#e1e6ed", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Four clocks in the Vega / GR00T recipe", fontsize=16,
                 fontweight="bold", y=.97)
    fig.text(.32, .85, "100 Hz physics / 20 Hz control\n"
             "Execution length 4, prediction horizon 16", fontsize=10, color="#57677a")
    fig.text(.055, .065, "Configured durations, not measured latency or joint response.\n"
             "16 actions cover 0.80 s; first-to-last timestamps span 0.75 s.",
             fontsize=10, color="#57677a")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
