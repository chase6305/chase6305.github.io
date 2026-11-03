#!/usr/bin/env python3
"""Plot the constructed slot-readout example; requires Matplotlib only for plotting."""
from __future__ import annotations

import argparse
from pathlib import Path

from onejev_examples import arithmetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "assets/onejev-probability-readout.png")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    values = arithmetic()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "text.color": "#172337", "axes.labelcolor": "#172337",
                         "xtick.color": "#172337", "ytick.color": "#172337"})
    fig = plt.figure(figsize=(12, 6.4), facecolor="white")
    fig.suptitle("From slot logits to a typed Choice answer", fontsize=23, weight="bold", y=.95)
    fig.text(.5, .86, "Constructed example  |  K = 3  |  T = 1", ha="center", fontsize=15)
    boxes = [(.055, .25, .25, .49), (.375, .25, .25, .49), (.695, .25, .265, .49)]
    for x, y, width, height in boxes:
        fig.add_artist(FancyBboxPatch((x-.018, y-.04), width+.036, height+.065,
                                     boxstyle="round,pad=0.008,rounding_size=0.018",
                                     transform=fig.transFigure, facecolor="#fcfcfe",
                                     edgecolor="#c2cad6", linewidth=1.2, zorder=0))
    for start, end in [(.326, .35), (.646, .67)]:
        fig.add_artist(FancyArrowPatch((start, .49), (end, .49), arrowstyle="-|>",
                                      mutation_scale=17, linewidth=1.7, color="#344359",
                                      transform=fig.transFigure))
    ax1 = fig.add_axes([.09, .34, .18, .27])
    ax2 = fig.add_axes([.41, .34, .18, .27])
    for ax in [ax1, ax2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a5afbe")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e7ee", linewidth=.7)
        ax.tick_params(length=0, pad=6)
        ax.set_xticks([0, 1, 2], ["A", "B", "C"])
        ax.set_xlim(-.6, 2.6)
    ax1.bar([0, 1, 2], values["logits"], width=.55, color="#a695cf", edgecolor="#705699")
    ax1.scatter([1, 2], [0, 0], s=18, color="#705699", zorder=3)
    ax1.set_ylim(-.03, 2.65)
    ax1.set_yticks([0, 1, 2])
    ax1.set_ylabel("Logit", fontsize=13)
    for i, label in enumerate(["ln 8", "0", "0"]):
        ax1.text(i, values["logits"][i]+.13, label, ha="center", fontsize=15, weight="bold")
    ax2.bar([0, 1, 2], values["probabilities_T1"], width=.55,
            color="#8fbc9e", edgecolor="#46805c")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    ax2.set_ylabel("Probability", fontsize=13)
    for i, value in enumerate(values["probabilities_T1"]):
        ax2.text(i, value+.055, f"{value:.2f}", ha="center", fontsize=15, weight="bold")
    for x, title in [(.18, "1  Slot logits"), (.5, "2  Softmax"), (.8275, "3  Typed fields")]:
        fig.text(x, .675, title, ha="center", weight="bold", fontsize=17)
    fig.text(.18, .27, "Candidate labels", ha="center", fontsize=13)
    fig.text(.5, .27, "Normalize over A, B, C", ha="center", fontsize=13)
    fig.text(.825, .57, "slot A maps to enable", ha="center", fontsize=15)
    fig.text(.825, .495, "choice = enable", ha="center", fontsize=18, weight="bold")
    fig.text(.825, .42, "max probability = 0.80", ha="center", fontsize=15)
    fig.text(.825, .345, f"confidence = {values['choice_confidence']:.2f}",
             ha="center", fontsize=18, weight="bold", color="#355e46")
    fig.text(.825, .27, "Confidence is a separate field", ha="center", fontsize=12)
    fig.text(.5, .12, r"$c_{\mathrm{choice}}=(p_{\max}-1/3)/(1-1/3)=0.7$",
             ha="center", fontsize=19)
    fig.text(.5, .055, "Candidate-relative arithmetic; not a measured task-success rate",
             ha="center", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
