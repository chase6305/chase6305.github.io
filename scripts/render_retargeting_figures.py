#!/usr/bin/env python3
"""Draw the quantitative examples from the retargeting articles."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path


POSTS = Path(__file__).resolve().parents[1] / "content/posts/robotics"
HAND_LIMIT = 120
DEMO_HEIGHT = 135
BOX_PRIOR = Fraction(1, 2)
TARGET_POINT = 0
PALM_POINT = 20
WINDOWS = ((0, 60, 0), (57, 117, 3), (58, 118, 3))
SHORT_TAIL = (114, 118)
SOURCE_FPS = 24
WRITTEN_FPS = 30
SOURCE_FRAMES = 60
CONTACT_FRAME = 24


def height_terms(box_height):
    gap = max(0, box_height - HAND_LIMIT)
    return gap * gap, BOX_PRIOR * (box_height - DEMO_HEIGHT) ** 2


def source_and_written_time(source_frame):
    return Fraction(source_frame, SOURCE_FPS), Fraction(source_frame - 1, WRITTEN_FPS)


def check():
    optimum = Fraction(HAND_LIMIT + BOX_PRIOR * DEMO_HEIGHT, 1 + BOX_PRIOR)
    assert optimum == 125
    assert height_terms(optimum) == (25, 50)
    assert height_terms(DEMO_HEIGHT) == (225, 0)
    for height in (Fraction(index, 10) for index in range(1120, 1421)):
        if height != optimum:
            assert sum(height_terms(height)) > 75
    assert abs(PALM_POINT - TARGET_POINT) == 20
    assert all(end - start == 60 for start, end, _ in WINDOWS)
    assert SHORT_TAIL[0] == WINDOWS[1][1] - 3 and SHORT_TAIL[1] - SHORT_TAIL[0] < 5
    assert [(start, start + pins) for start, _, pins in WINDOWS if pins] == [(57, 60), (58, 61)]
    assert source_and_written_time(1) == (Fraction(1, 24), 0)
    assert source_and_written_time(CONTACT_FRAME) == (1, Fraction(23, 30))
    source, written = source_and_written_time(SOURCE_FRAMES - 1)
    assert source - written == Fraction(21, 40)


def render_height(plt, output):
    figure, ax = plt.subplots(figsize=(7.0, 4.3))
    figure.subplots_adjust(left=.15, right=.94, bottom=.19, top=.93)
    x = [i / 20 for i in range(112 * 20, 141 * 20 + 1)]
    costs = [height_terms(height) for height in x]
    ax.plot(x, [sum(terms) for terms in costs], color="#225985", linewidth=2.6, label="Total")
    ax.plot(x, [gap for gap, _ in costs], color="#a24e20", linestyle="--", linewidth=1.8, label="Hand gap²")
    ax.plot(x, [prior for _, prior in costs], color="#604c87", linestyle=":", linewidth=2.1, label="Box prior")
    ax.axvline(HAND_LIMIT, color="#637181", alpha=.6, linewidth=1, linestyle="--")
    ax.text(HAND_LIMIT - .4, 25, "Reach limit", size=10.5, ha="right", va="bottom")
    for center, caption, offset, alignment in ((125, "Free: 125 cm", 0, "center"), (135, "Fixed: 135 cm", -.5, "right")):
        score = sum(height_terms(center))
        ax.scatter([center], [score], color="#225985", edgecolor="white", linewidth=1.2, s=80, zorder=4)
        ax.annotate(caption, (center, score), (center + offset, score + 36), ha=alignment, fontsize=12,
                    arrowprops={"arrowstyle": "-", "color": "#334350"})
    ax.set(xlim=(112, 141), ylim=(0, 460), xticks=[115, 120, 125, 130, 135, 140],
           xlabel="Box grasp height (cm)", ylabel="Cost (cm²)")
    ax.grid(axis="y", color="#e2e8ed", linewidth=.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    figure.savefig(output, dpi=170, pil_kwargs={"lossless": True})
    plt.close(figure)


def render_source_time(plt, output):
    figure, ax = plt.subplots(figsize=(7.0, 4.1))
    figure.subplots_adjust(left=.14, right=.94, bottom=.20, top=.94)
    indices = range(1, SOURCE_FRAMES)
    source, written = zip(*(tuple(map(float, source_and_written_time(i))) for i in indices))
    ax.plot(indices, source, color="#225985", linewidth=2.7, label="Source time (24 FPS)")
    ax.plot(indices, written, color="#a24e20", linestyle="--", linewidth=2.4, label="File index ÷ 30 FPS")
    at_source, at_written = map(float, source_and_written_time(CONTACT_FRAME))
    ax.scatter([CONTACT_FRAME, CONTACT_FRAME], [at_source, at_written], zorder=4,
               color=["#225985", "#a24e20"], edgecolor="white", linewidth=1.2, s=65)
    ax.annotate("", xy=(CONTACT_FRAME + 1.1, at_source), xytext=(CONTACT_FRAME + 1.1, at_written),
                arrowprops={"arrowstyle": "<->", "color": "#334350", "shrinkA": 1, "shrinkB": 1})
    error_ms = round(1000 * (at_source - at_written))
    ax.annotate(f"Contact at source frame {CONTACT_FRAME}\n{error_ms} ms early in file time",
                (CONTACT_FRAME + 1.1, (at_written + at_source) / 2), (31, .45),
                fontsize=11.5, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#526477"})
    ax.set(xlim=(0, SOURCE_FRAMES), ylim=(-.045, 2.56), xticks=[0, 12, 24, 36, 48, 59],
           xlabel="Source frame index", ylabel="Time (s)")
    ax.grid(axis="y", color="#e2e8ed", linewidth=.9)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)
    figure.savefig(output, dpi=170, pil_kwargs={"lossless": True})
    plt.close(figure)


def render_surface(plt, output):
    figure, ax = plt.subplots(figsize=(6.8, 3.45))
    figure.subplots_adjust(left=.11, right=.89, bottom=.08, top=.95)
    ax.set(xlim=(-12, 32), ylim=(-9, 13), aspect="equal")
    ax.set_axis_off()
    ax.fill_between([-12, 32], -9, 0, color="#e6eef5")
    ax.plot([-12, 32], [0, 0], color="#3b5368", linewidth=1.5)
    ax.text(10, -5.5, "Box face", ha="center", va="center", color="#2b4356", fontsize=14)
    ax.scatter([TARGET_POINT], [0], marker="D", s=80, facecolor="white", edgecolor="#604c87", linewidth=2, zorder=5)
    ax.scatter([PALM_POINT], [0], marker="o", s=100, facecolor="#225985", edgecolor="white", linewidth=1.2, zorder=5)
    for x, label in ((TARGET_POINT, "Target anchor"), (PALM_POINT, "Palm point")):
        ax.text(x, 10.4, label, ha="center", va="center", fontsize=14)
        ax.plot([x, x], [0, 7.9], color="#6b7987", linestyle=":", linewidth=1.2)
    ax.annotate("", xy=(PALM_POINT, 5.2), xytext=(TARGET_POINT, 5.2),
                arrowprops={"arrowstyle": "<->", "color": "#3b5368", "linewidth": 1.3})
    ax.text(10, 6.2, "20 cm", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.text(PALM_POINT, -9.0, "Surface gap = 0 cm", ha="center", va="top", fontsize=12)
    figure.savefig(output, dpi=170, pil_kwargs={"lossless": True})
    plt.close(figure)


def render_windows(plt, output):
    figure, (overview, seam) = plt.subplots(2, 1, figsize=(7.2, 4.75), height_ratios=(3.3, 2.0))
    figure.subplots_adjust(left=.15, right=.96, bottom=.12, top=.88, hspace=.62)

    def bar(ax, row, start, end, pins):
        ax.barh(row, end - start - pins, left=start + pins, height=.50,
                color="#b9d6ea", edgecolor="#2d5a75", linewidth=1)
        if pins:
            ax.barh(row, pins, left=start, height=.50,
                    color="#f8deb5", edgecolor="#875323", hatch="///", linewidth=1)

    for row, (start, end, pins) in zip([3, 2, 1], WINDOWS):
        bar(overview, row, start, end, pins)
        overview.text((start + end) / 2, row, f"[{start}, {end})", va="center", ha="center", fontsize=11)
        if pins:
            bar(seam, row, start, end, pins)

    overview.barh(0, SHORT_TAIL[1] - SHORT_TAIL[0], left=SHORT_TAIL[0], height=.50,
                  fill=False, edgecolor="#526477", linewidth=1.2, linestyle="--")
    overview.text(SHORT_TAIL[0] - 1.2, 0, "[114, 118) is extended", va="center", ha="right", fontsize=10.5)
    overview.set(xlim=(0, 120), xticks=[0, 30, 60, 90, 118], ylim=(-.52, 3.60),
                 yticks=[3, 2, 1, 0], yticklabels=["First", "Next", "Tail", ""], title="118-frame clip")
    seam.set(xlim=(55, 64), xticks=range(55, 65), ylim=(.50, 2.62), yticks=[2, 1],
             yticklabels=["Next", "Tail"], title="Overlap near the start of each window", xlabel="Frame index")
    for row, start, end in ((2, 57, 60), (1, 58, 61)):
        seam.text((start + end) / 2, row, "Fixed", ha="center", va="center", fontsize=10.5,
                  bbox={"facecolor": "#f8deb5", "edgecolor": "none", "pad": .5})
        seam.text(62.4, row, "Solve", ha="center", va="center", fontsize=10.5)
    for axis in (overview, seam):
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=6)
        axis.grid(axis="x", color="#e2e8ed", linewidth=.75)
        axis.set_axisbelow(True)
    figure.savefig(output, dpi=170, pil_kwargs={"lossless": True})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check the arithmetic using the standard library")
    mode.add_argument("--render", action="store_true", help="Regenerate article images with Matplotlib and Pillow")
    args = parser.parse_args()
    check()
    if args.render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": "#192632",
                             "axes.labelcolor": "#192632", "axes.edgecolor": "#526477",
                             "axes.spines.top": False, "axes.spines.right": False,
                             "xtick.color": "#334350", "ytick.color": "#334350"})
        files = ((render_height, "motion-retargeting", "box-height.webp"),
                 (render_source_time, "motion-retargeting", "source-time.webp"),
                 (render_windows, "contact-preserving-retargeting", "window-overlap.webp"),
                 (render_surface, "contact-preserving-retargeting", "contact-distance.webp"))
        for draw, route, name in files:
            output = POSTS / route / "assets" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            draw(plt, output)
    print("Checked the box-height, source-time, window and contact-distance examples.")


if __name__ == "__main__":
    main()
