#!/usr/bin/env python3
"""Plot the pinned Vega configuration's two VOC clocks, not training measurements."""

from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/video-to-data"


def main():
    lab = runpy.run_path(str(BUNDLE / "voc_schedule_lab.py"))
    updates = list(range(0, 20001, 25))
    steps = list(range(71))
    target = [lab["global_target"](i) for i in updates]
    late = [lab["reset_scale"](i, 0.0) for i in steps]
    free = [lab["reset_scale"](i, 0.0, freeze=0, decay=0, reset=0) for i in steps]
    assert target[updates.index(12500)] == 0
    assert late[40] == 1 and late[45] == .5 and late[50] == 0
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": "#25354a", "axes.labelcolor": "#25354a",
                         "axes.edgecolor": "#ccd5df", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8))
    fig.subplots_adjust(left=.10, right=.97, top=.83, bottom=.13, hspace=.72)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-.06, 1.12)
        ax.set_yticks([0, .5, 1])
        ax.set_ylabel("VOC scale")
        ax.grid(color="#e1e6ed", linewidth=.8)
        ax.set_axisbelow(True)
    ax = axes[0]
    ax.step(updates, target, where="post", color="#3173b8", linewidth=2.6)
    ax.axvspan(12500, 20000, color="#e5f1e6", zorder=0)
    ax.axvline(12500, color="#39814a", linestyle="--", linewidth=1.2)
    ax.text(13300, .58, "Scheduled scale = 0\nReset assistance still exists", color="#27663b")
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 1250, 5000, 10000, 12500, 20000],
                  ["0", "1,250", "5,000", "10,000", "12,500", "20,000"])
    ax.set_xlabel("Environment steps / 24 (24 steps collected per PPO update)")
    ax.set_title("1. Configured global schedule (checked on reset)", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax = axes[1]
    ax.axvspan(0, 50, color="#f0eafa", zorder=0)
    ax.plot(steps, late, color="#3173b8", linewidth=2.6,
            label="Training reset, after global target reaches zero")
    ax.plot(steps, free, color="#d68434", linewidth=2.2, linestyle="--",
            label="Collection / closed-loop evaluation")
    ax.scatter([0, 40, 45, 50], [1, 1, .5, 0], color="#3173b8", s=28, zorder=4)
    ax.axvline(50, color="#8463b1", linestyle=":", linewidth=1.6)
    ax.text(7, .49, "Training reference held fixed\nthrough command update 50", color="#715299")
    ax.annotate("Reference advances\nfrom update 51", xy=(51, 0), xytext=(54, .50),
                fontsize=11, color="#715299", arrowprops={"arrowstyle": "->", "color": "#8463b1"})
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 50, 60, 70])
    ax.set_xlabel("Command updates since resample (20 Hz cadence)")
    ax.set_title("2. Per-environment scale after command updates", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.37), frameon=False, fontsize=10)
    fig.suptitle("Two VOC clocks in the Vega recipe", fontsize=20, fontweight="bold", y=.98)
    fig.text(.1, .935, "Configured schedules from pinned source; scale factors are not measured forces.",
             fontsize=11, color="#57677a")
    fig.text(.1, .025, "Training: freeze=50, decay=10, reset scale=1.0. Collection/evaluation: all three are zero.\n"
             "The lower blue curve uses a fixed global target of zero; actual wrench also depends on state error and clamps.\n"
             "Reference advancement assumes that the trajectory has not ended.",
             fontsize=10, color="#57677a")
    out = BUNDLE / "assets/voc-two-clocks.webp"
    fig.savefig(out, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
