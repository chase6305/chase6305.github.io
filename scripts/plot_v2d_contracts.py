#!/usr/bin/env python3
"""Plot explicit V2D contract indices and constructed height examples."""

import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/video-to-data"
spec = importlib.util.spec_from_file_location("v2d_contract_lab", BUNDLE / "data_contract_lab.py")
lab = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lab
spec.loader.exec_module(lab)


def main():
    out = BUNDLE / "assets"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": "#25354a", "axes.labelcolor": "#25354a",
                         "savefig.facecolor": "white", "axes.edgecolor": "#ccd5df"})
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.10, hspace=0.48)
    titles = {"stable_place": "Smooth return: success",
              "held_aloft": "Held aloft: lift/hold passes, placement fails",
              "abrupt_drop": "Abrupt drop: final window stable, descent limit fails"}
    colors = {"stable_place": "#39814a", "held_aloft": "#8463b1", "abrupt_drop": "#c3732b"}
    for ax, (name, trace) in zip(axes, lab.example_traces().items()):
        result = lab.height_result(trace)
        t = [i / 20 for i in range(len(trace))]
        ax.axhspan(-0.01, 0.03, color="#e8f3e9", alpha=0.75)
        ax.axhline(0.10, color="#3173b8", linewidth=1, linestyle="--")
        ax.axhline(0.05, color="#8463b1", linewidth=1, linestyle=":")
        ax.axvspan(t[-20], t[-1], color="#e9e7f4", alpha=0.55)
        ax.plot(t, trace, color=colors[name], linewidth=2.5)
        ax.set_title(titles[name], loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.text(0.99, 0.88, f"Peak {result['max_lift_m']:.2f} m | "
                f"hold {result['max_hold_samples']} samples | "
                f"max downward step {result['max_descent_m_per_sample']:.3f} m",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95})
        ax.set_ylim(-0.015, 0.145)
        ax.set_yticks([0, 0.05, 0.10])
        ax.set_ylabel("Lift (m)")
        ax.grid(axis="x", color="#e1e6ed", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s), 20 Hz sampling; sample 0 sets the baseline", labelpad=10)
    fig.suptitle("The same lift can lead to different episode outcomes", fontsize=18,
                 fontweight="bold", y=0.97)
    fig.text(0.10, 0.905, "Constructed traces, not measured trials. Dashed: lift 0.10 m; dotted: hold 0.05 m.\n"
             "Green band: final-height range; purple band: last 20 samples. Other checks also apply.",
             fontsize=10.5, color="#57677a")
    fig.savefig(out / "height-criteria-examples.webp", format="webp", dpi=180,
                pil_kwargs={"lossless": True})
    plt.close(fig)

    fields = {"R arm": (7, "#dcebf8", "#3173b8"),
              "L arm": (7, "#e8e0f5", "#8463b1"),
              "R fingers": (22, "#f9e8d5", "#c3732b"),
              "L fingers": (22, "#e2efdf", "#39814a")}
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.17)
    rows = [(2.1, ["R arm", "L arm", "R fingers", "L fingers"], "State"),
            (0.6, ["R arm", "R fingers", "L arm", "L fingers"], "Action")]
    for y, order, title in rows:
        start = 0
        ax.text(-1, y + 0.30, title, ha="right", va="center", fontsize=13, fontweight="bold")
        for name in order:
            width, face, edge = fields[name]
            ax.add_patch(FancyBboxPatch((start, y), width, 0.60,
                         boxstyle="round,pad=0,rounding_size=0.11",
                         facecolor=face, edgecolor=edge, linewidth=1.2))
            ax.text(start + width / 2, y + 0.30, name, ha="center", va="center", fontsize=11)
            ax.text(start, y - 0.13, str(start), ha="center", va="top", fontsize=10)
            start += width
        ax.text(start, y - 0.13, str(start), ha="center", va="top", fontsize=10)
    ax.annotate("Different field order", xy=(22, 1.46), ha="center", fontsize=11,
                color="#526170")
    ax.text(29, -0.14, "Slice boundaries: zero-based, end-exclusive. Widths are proportional to field size.",
            ha="center", fontsize=10, color="#57677a")
    ax.set_xlim(-1, 59)
    ax.set_ylim(-0.3, 3.1)
    ax.axis("off")
    fig.suptitle("58 values do not imply the same index contract", fontsize=18,
                 fontweight="bold", y=0.97)
    fig.text(0.12, 0.84, "Vega / Sharpa joint contract: measured positions vs absolute position targets",
             fontsize=11, color="#57677a")
    fig.savefig(out / "state-action-layout.webp", format="webp", dpi=180,
                pil_kwargs={"lossless": True})
    plt.close(fig)
    print("Created height-criteria-examples.webp and state-action-layout.webp")


if __name__ == "__main__":
    main()
