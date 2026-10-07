#!/usr/bin/env python3
"""Rebuild the analytic two-mode figure: Python 3.8+ and matplotlib.

All values come from the constructed distribution in vla_lab.py. No robot
measurements or learned model outputs are used. Run with python -B.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vla_lab import bimodal_flow


def build(destination):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 19, "axes.labelsize": 15,
        "svg.fonttype": "path", "svg.hashsalt": "vla-two-mode-flow-v1",
        "axes.edgecolor": "#66758a", "text.color": "#16243a",
        "axes.labelcolor": "#16243a", "xtick.color": "#42536b",
        "ytick.color": "#42536b", "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.55]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.30, wspace=0.32)
    fig.suptitle("One observation, two valid action modes", fontsize=26, fontweight="bold", y=0.96)
    fig.text(0.5, 0.88, "Target action A is -1 or +1 with equal probability", ha="center", fontsize=17)
    a, b = axes
    a.bar([-1, 1], [0.5, 0.5], width=0.20, color=["#a7c6ed", "#c9b6e8"],
          edgecolor=["#356dac", "#7953aa"], linewidth=1.4)
    a.scatter([0], [0], marker="x", s=125, color="#cc731d", linewidths=2.8, zorder=4)
    a.annotate("L2 optimum: 0", (0, 0.015), (0, 0.23), ha="center", fontsize=15,
               arrowprops={"arrowstyle": "->", "color": "#cc731d", "lw": 1.4})
    a.set(xlim=(-1.65, 1.65), ylim=(-0.045, 0.65), xticks=[-1, 0, 1], yticks=[0, 0.25, 0.5],
          xlabel="Final action", ylabel="Target probability mass")
    a.set_title("Direct final-action regression", pad=16)
    a.text(0.5, -0.22, "The conditional mean lies\nbetween both target modes.",
           transform=a.transAxes, ha="center", va="top", fontsize=14, linespacing=1.4)
    for noise in [-2.2, -1.7, -1.2, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.2, 1.7, 2.2]:
        path = bimodal_flow(noise, steps=1000, end_time=0.02)
        b.plot([p[0] for p in path], [p[1] for p in path],
               color="#356dac" if noise < 0 else "#7953aa", linewidth=1.65, alpha=0.78)
    for endpoint in [-1, 1]:
        b.axhline(endpoint, color="#97a7b7", linestyle=(0, (3, 4)), linewidth=0.8, zorder=0)
    b.set(xlim=(1, 0), ylim=(-2.45, 2.45), xticks=[1, 0.75, 0.5, 0.25, 0],
          yticks=[-2, -1, 0, 1, 2], xlabel="Flow time s: noise to data", ylabel="Action-space coordinate")
    b.set_title("Samples follow the same vector field", pad=16)
    b.text(0.5, -0.22, "Different initial noise retains different outcomes.\nEuler integration stops at s = 0.02.",
           transform=b.transAxes, ha="center", va="top", fontsize=14, linespacing=1.4)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4eaf0", linewidth=0.7)
        axis.set_axisbelow(True)
    fig.text(0.5, 0.035, "Analytic scalar example; no learned policy or robot experiment.",
             ha="center", fontsize=13, color="#52647c")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Date": None}, facecolor="white")
    plt.close(fig)
    if destination.suffix.lower() == ".svg":
        # Matplotlib emits trailing spaces in multi-line path attributes.
        # Keep the vector coordinates intact while normalizing file whitespace.
        text = destination.read_text(encoding="utf-8")
        destination.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n",
                               encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "assets/flow-mode-transport.svg")
    print(build(parser.parse_args().output))
