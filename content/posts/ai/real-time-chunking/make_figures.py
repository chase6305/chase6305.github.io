#!/usr/bin/env python3
"""Rebuild the article's original SVG diagrams using stdlib and rtc_lab/NumPy."""

from html import escape
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from rtc_lab import prefix_weights


OUT = Path(__file__).resolve().parent / "assets"
INK = "#16334d"
MUTED = "#52687b"
ORANGE = "#f6c58a"
BLUE = "#bddcf4"
TEAL = "#b8e2d3"
PALE = "#edf2f6"


class Figure:
    def __init__(self, title, subtitle, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
            f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<rect width="100%" height="100%" rx="20" fill="#f8fafc"/>',
            '<g font-family="Arial, sans-serif">',
        ]
        self.text(42, 53, title, 29, weight="bold")
        self.text(42, 85, subtitle, 18, MUTED)

    def text(self, x, y, value, size=18, color=INK, anchor="start", weight="normal"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill, radius=5):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, color=MUTED, width=2, dash=""):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ["</g></svg>\n"]))


def timeline():
    f = Figure("RTC: align chunks before switching", "H = 16   |   s = 5   |   d = 4   |   zero-based indices", 520)
    start, step = 220, 38
    f.text(42, 175, "Old chunk", 20, weight="bold")
    f.text(42, 198, "origin: t - 5", 16, MUTED)
    f.text(42, 280, "New chunk", 20, weight="bold")
    f.text(42, 303, "origin: t", 16, MUTED)
    for j in range(16):
        color = PALE if j < 5 else ORANGE if j < 9 else BLUE
        f.rect(start + j * step, 150, 35, 48, color)
    for i in range(16):
        color = ORANGE if i < 4 else BLUE if i < 11 else TEAL
        x = start + (i + 5) * step
        f.rect(x, 250, 35, 48, color)
        f.text(x + 17.5, 280, i, 16, anchor="middle")
    for j, label in [(0, "t - 5"), (5, "t"), (9, "t + 4"), (16, "t + 11"), (21, "t + 16")]:
        x = start + j * step
        f.line(x - 2, 122, x - 2, 326, "#9aafbf", 1, "4 5")
        f.text(x - 2, 352, label, 17, anchor="middle")
    f.text(start + 5 * step, 117, "request", 16)
    f.text(start + 9 * step, 225, "result ready / use new[4:]", 17, weight="bold")
    for x, color, title, detail in [
        (42, ORANGE, "Committed prefix: 4", "Execute old actions while waiting"),
        (404, BLUE, "Remaining overlap: 7", "Old plan can guide the new plan"),
        (766, TEAL, "New future: 5", "No old counterpart"),
    ]:
        f.rect(x, 393, 24, 24, color)
        f.text(x + 34, 413, title, 19, weight="bold")
        f.text(x, 451, detail, 17, MUTED)
    f.save("chunk-timeline.svg")


def masks():
    f = Figure("Prefix weights: commitment and freedom", "H = 8   |   s = 3   |   d = 2   |   overlap ends at index 5", 680)
    schedules = [("ones", "#637487"), ("zeros", "#d88b36"),
                 ("linear", "#368975"), ("exp", "#2879bc")]
    for row, (schedule, color) in enumerate(schedules):
        y = 158 + row * 118
        f.text(42, y + 25, schedule, 22, color, weight="bold")
        weights = prefix_weights(2, 5, 8, schedule)
        for i, w in enumerate(weights):
            x = 200 + i * 106
            region = ORANGE if i < 2 else BLUE if i < 5 else TEAL
            f.rect(x, y - 22, 94, 78, PALE)
            f.rect(x, y + 48 - 60 * float(w), 94, 60 * float(w), color, 2)
            f.rect(x, y + 59, 94, 5, region, 1)
            f.text(x + 47, y + 84, f"{w:.3f}", 17, anchor="middle")
    for i in range(8):
        f.text(247 + i * 106, 123, f"i = {i}", 17, MUTED, anchor="middle")
    f.text(42, 643, "Orange: committed prefix     Blue: mutable overlap     Green: new future", 18, MUTED)
    f.save("prefix-weights.svg")


def training():
    f = Figure("Training-time RTC: learn the continuation", "H = 8   |   d = 2   |   clean prefix + noisy suffix + per-action flow time", 650)
    labels = [(157, "Input x"), (255, "Flow time"), (512, "Loss mask")]
    for y, label in labels:
        f.text(42, y + 34, label, 20, weight="bold")
        for i in range(8):
            prefix = i < 2
            x = 215 + i * 105
            f.rect(x, y, 96, 54, ORANGE if prefix else BLUE)
            value = (f"A{i}" if prefix else f"x{i}(tau)") if y == 157 else (
                ("1.0" if prefix else "tau") if y == 255 else ("0" if prefix else "1"))
            f.text(x + 48, y + 34, value, 20, anchor="middle")
    f.text(312, 136, "Known / clean", 19, anchor="middle")
    f.text(733, 136, "To be generated / noisy", 19, anchor="middle")
    f.rect(215, 361, 831, 86, "#dfebf3", 12)
    f.text(630, 396, "Conditional action network", 24, anchor="middle", weight="bold")
    f.text(630, 426, "observation o + action x + time for each action token", 18, MUTED, anchor="middle")
    for x in (311, 733):
        f.line(x, 318, x, 350)
        f.line(x, 350, x - 6, 341)
        f.line(x, 350, x + 6, 341)
        f.line(x, 454, x, 497)
        f.line(x, 497, x - 6, 488)
        f.line(x, 497, x + 6, 488)
    f.text(42, 612, "Sampling: restore the prefix before every network call; execute only the live suffix.", 19, MUTED)
    f.save("training-conditioning.svg")


if __name__ == "__main__":
    timeline()
    masks()
    training()
    print(f"Wrote three SVG teaching diagrams to {OUT}")
