#!/usr/bin/env python3
"""Render the fixed-duration backward/communication example; no GPU experiment.

--check needs only Python's standard library. Rendering additionally needs
Matplotlib and Pillow in a separate plotting environment.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

def schedule(mode):
    """Four layers, 2 ms backward each, one 1 ms bucket per layer."""
    stride = 3 if mode == "Serial" else 2
    return [{"layer": 4 - index,
             "compute": [stride * index, stride * index + 2],
             "communication": [stride * index + 2, stride * index + 3]}
            for index in range(4)]


def check():
    result = {}
    for mode in ("Serial", "Overlap"):
        rows = schedule(mode)
        for index, row in enumerate(rows):
            start, ready = row["compute"]
            send, finish = row["communication"]
            if ready - start != 2 or finish - send != 1 or send < ready:
                raise AssertionError("A bucket must follow its complete gradient")
            if index and (start < rows[index - 1]["compute"][1]
                          or send < rows[index - 1]["communication"][1]):
                raise AssertionError("Operations on one stream must not overlap")
        end_compute = rows[-1]["compute"][1]
        end = max(end_compute, rows[-1]["communication"][1])
        exposed = sum(max(0, finish - max(start, end_compute))
                      for start, finish in (row["communication"] for row in rows))
        result[mode] = {"rows": rows, "finish_ms": end,
                        "last_compute_ms": end_compute,
                        "communication_after_compute_ms": exposed}
    if (result["Serial"]["finish_ms"] != 4 * (2 + 1)
            or result["Overlap"]["finish_ms"] != 4 * 2 + 1
            or result["Overlap"]["communication_after_compute_ms"] != 1):
        raise AssertionError("The fixed durations do not match the figure labels")
    return result


def render(directory, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "ddp-overlap-fixed-ms"})
    colors = {"compute": ("#e2ecfa", "#6285b4"),
              "comm": ("#faead7", "#bd8b53"),
              "exposed": ("#f8dedb", "#bc706b"),
              "wait": ("#eef1f5", "#9ba6b1")}
    figure, axes = plt.subplots(2, 1, figsize=(13, 7.1), sharex=True)
    for axis, mode in zip(axes, ("Serial", "Overlap")):
        rows = data[mode]["rows"]
        finish = data[mode]["finish_ms"]
        axis.set_title(f"{mode}: {finish} ms", loc="left", fontsize=20,
                       fontweight="bold", color="#24384a", pad=13)
        axis.set_xlim(0, 12.3)
        axis.set_ylim(-.15, 1.95)
        axis.set_yticks([1.25, .4], ["Backward\ncompute", "Gradient\nall-reduce"],
                       fontsize=16, color="#34485b")
        axis.tick_params(axis="y", length=0, pad=18)
        axis.set_xticks(range(0, 13, 2))
        axis.tick_params(axis="x", labelsize=15, labelcolor="#526679",
                         bottom=True, labelbottom=True)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#e7edf3", linewidth=.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.spines["bottom"].set_visible(True)
        axis.spines["bottom"].set_color("#9aaaba")

        def bar(start, end, y, kind, label, hatch=None):
            face, edge = colors[kind]
            axis.add_patch(FancyBboxPatch(
                (start, y - .24), end - start, .48,
                boxstyle="round,pad=0,rounding_size=0.05", facecolor=face,
                edgecolor=edge, linewidth=1.3, hatch=hatch, zorder=3))
            axis.text((start + end) / 2, y, label, ha="center", va="center",
                      fontsize=17, color="#24384a", zorder=4)

        for row in rows:
            start, ready = row["compute"]
            send, end = row["communication"]
            label = f"L{row['layer']}"
            bar(start, ready, 1.25, "compute", label)
            kind = "exposed" if mode == "Overlap" and send >= 8 else "comm"
            bar(send, end, .4, kind, label)
            axis.annotate("", xy=(send, .67), xytext=(ready, 1.0),
                          arrowprops={"arrowstyle": "-|>", "color": "#8394a5",
                                      "lw": 1.1, "mutation_scale": 10}, zorder=5)
        axis.axvline(finish, color="#697b8c", linestyle=(0, (3, 3)),
                     linewidth=1.2, ymax=.89, zorder=2)
        if mode == "Overlap":
            bar(8, 9, 1.25, "wait", "Wait", "//")
            axis.annotate("Exposed\n1 ms", xy=(8.75, .64), xytext=(10.4, 1.15),
                          ha="center", va="center", fontsize=17, color="#9b4842",
                          arrowprops={"arrowstyle": "-|>", "color": "#aa615a",
                                      "lw": 1.4, "connectionstyle": "angle3,angleA=0,angleB=90"},
                          zorder=5)
        axis.set_xlabel("Time [ms]", fontsize=16, color="#425b70", labelpad=7)
    figure.subplots_adjust(left=.15, right=.98, top=.92, bottom=.09, hspace=.68)
    buffer = BytesIO()
    # Include both x-axis labels and the two-line stream names in the image.
    figure.savefig(buffer, format="png", dpi=160, facecolor="white",
                   bbox_inches="tight", pad_inches=.2)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.convert("RGB").save(directory / "compute-communication-overlap.webp",
                                  quality=94, method=6)
    figure.savefig(directory / "compute-communication-overlap.svg",
                   facecolor="white", bbox_inches="tight", pad_inches=.2,
                   metadata={"Date": None})
    plt.close(figure)
    (directory / "compute-communication-overlap.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check fixed durations without plotting dependencies")
    parser.add_argument("--output", type=Path, help="Write the WebP, SVG and interval JSON here")
    args = parser.parse_args()
    data = check()
    if args.output:
        render(args.output, data)
    elif not args.check:
        parser.error("Choose --check or --output DIRECTORY")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
