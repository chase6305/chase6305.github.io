#!/usr/bin/env python3
"""Render exact constructed LetterBox/NMS examples with standard-library SVG."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/ultralytics"
spec = importlib.util.spec_from_file_location("ultralytics_geometry", BUNDLE / "detection_geometry.py")
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)


def text(x, y, label, size=19, weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(str(label))}</text>')


def rectangle(x, y, w, h, fill, stroke="#334155", extra=""):
    return (f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" {extra}/>')


def document(width, height, title, description, elements):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}</style>'
            f'<rect width="{width}" height="{height}" fill="white"/>'
            + "".join(elements) + '</svg>\n')


def letterbox_figure():
    elements = [text(30, 42, "Same image, different padding", 28, "bold"),
                text(30, 76, "Continuous box edges; target 640 x 640; centered padding; stride 32", 18)]
    original = geometry.Box(320, 180, 960, 540)
    square = geometry.letterbox(1280, 720)
    minimal = geometry.letterbox(1280, 720, minimal=True)
    panels = [
        ("Original", (1280, 720), original, None, "r = 1", "No padding"),
        ("Fixed square", square.output_size, square.forward(original), square,
         "r = 0.5", "Top / bottom: 140 px each"),
        ("Minimal rectangle", minimal.output_size, minimal.forward(original), minimal,
         "r = 0.5", "Top / bottom: 12 px each"),
    ]
    for i, (name, size, box, placement, gain_label, padding_label) in enumerate(panels):
        x = 20 + i * 370
        elements += [rectangle(x, 104, 354, 470, "#fafbff", "#cbd5e1", 'rx="14"'),
                     text(x + 18, 138, name, 23, "bold"),
                     text(x + 18, 168, f"{size[0]} x {size[1]} px", 20)]
        scale = 310 / size[0]
        y = 198
        elements.append(rectangle(x + 22, y, 310, size[1] * scale,
                                  "#ffedd5" if placement else "#eff6ff"))
        if placement:
            elements.append(rectangle(x + 22 + placement.left * scale,
                                      y + placement.top * scale,
                                      placement.resized_width * scale,
                                      placement.resized_height * scale, "#eff6ff", "#64748b"))
        elements.append(rectangle(x + 22 + box.x1 * scale, y + box.y1 * scale,
                                  (box.x2-box.x1) * scale, (box.y2-box.y1) * scale,
                                  "#bfdbfe", "#1d4ed8", 'stroke-dasharray="8 4"'))
        elements += [text(x + 18, 537, gain_label + "; " + padding_label, 17),
                     text(x + 18, 560, f"Box: ({box.x1:g},{box.y1:g})-({box.x2:g},{box.y2:g})", 17)]
    elements += [text(30, 609, "Blue dashed box = same object; orange area = padding. Panels are fitted to equal display widths.", 18),
                 text(30, 638, "Invert using the transform actually applied: x = (x' - left) / r; y = (y' - top) / r.", 18)]
    return document(1134, 662, "LetterBox square versus stride-minimal padding",
                    "1280 by 720 becomes 640 by 360 content. Square padding is 140 pixels per side; stride-minimal padding is 12 per side with a 640 by 384 output.", elements)


def nms_figure():
    elements = [text(28, 42, "NMS: overlap, class and threshold are separate decisions", 26, "bold"),
                text(28, 74, "A: class 0, score 0.95; B: class 0, score 0.90; C: class 1, score 0.85", 19),
                text(28, 102, "A = (0,0)-(100,100); B and C = (10,10)-(110,110); IoU(A,B) = 81/119 = 0.680672...", 18)]
    panels = [("Class-aware, IoU > 0.5", "Keep A and C", True, False),
              ("Class-agnostic, IoU > 0.5", "Keep A only", False, False),
              ("Class-aware, IoU > 0.7", "Keep A, B and C", True, True)]
    for i, (title, decision, keep_c, keep_b) in enumerate(panels):
        x = 20 + 370*i
        elements += [rectangle(x, 130, 354, 340, "#fafbff", "#cbd5e1", 'rx="14"'),
                     text(x+16, 167, title, 19, "bold")]
        ox, oy, scale = x+50, 196, 1.7
        elements += [rectangle(ox, oy, 100*scale, 100*scale, "#dbeafe", "#1d4ed8"),
                     text(ox+12, oy+27, "A", 22, "bold")]
        if keep_b:
            elements.append(rectangle(ox+10*scale, oy+10*scale, 100*scale, 100*scale,
                                      "none", "#7c3aed", 'stroke-dasharray="3 4"'))
        if keep_c:
            elements.append(rectangle(ox+10*scale, oy+10*scale, 100*scale, 100*scale,
                                      "none", "#9a3412", 'stroke-dasharray="10 5"'))
        elements += [text(x+16, 424, decision, 23, "bold"),
                     text(x+16, 452, "B retained" if keep_b else "B suppressed by A", 19)]
    elements += [text(28, 510, "Confidence filtering happens before this example; all three scores already pass it.", 19),
                 text(28, 540, "These are constructed boxes. NMS-free end-to-end filtering follows a different path.", 19)]
    return document(1134, 568, "Class-aware and class-agnostic greedy NMS example",
                    "Three candidate boxes illustrate how changing class handling or IoU threshold changes suppression. B and C occupy the same coordinates but represent different classes.", elements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    geometry.check()
    figures = {"letterbox-coordinates.svg": letterbox_figure(), "nms-decisions.svg": nms_figure()}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in figures.items():
            (args.output/name).write_text(data, encoding="utf-8")
    if args.check:
        import xml.etree.ElementTree as ET
        for name, data in figures.items():
            ET.fromstring(data)
            if (BUNDLE/"assets"/name).read_text() != data:
                raise SystemExit(f"Stale SVG: {name}")
    print("LetterBox/NMS arithmetic, geometry invariants and SVG documents passed.")


if __name__ == "__main__":
    main()
