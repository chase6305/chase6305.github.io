"""Recreate the three quantitative figures. Requires NumPy and Matplotlib."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from muon_math import ns_response, polar_rms, qk_example, shape_factors

OUT = Path(__file__).parent / "assets"
COLORS = ["#3976ae", "#8661aa", "#c2772c", "#348474"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.18, "savefig.dpi": 180})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    x = np.linspace(0, 1, 1601)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(x, [ns_response(float(z), 1) for z in x], color=COLORS[0], lw=2, label="One fast NS step")
    ax.plot(x, [ns_response(float(z), 5) for z in x], color=COLORS[1], lw=2, ls="-.", label="Five fast NS steps")
    ax.plot(x[1:], np.ones(len(x) - 1), "--", color=COLORS[3], label="Exact polar (x > 0)")
    ax.plot([0], [0], "o", color="black", label="Zero stays zero")
    ax.set(xlabel="Initial normalized singular value x", ylabel="Output singular value",
           title="Fast NS is a finite polynomial response", xlim=(0, 1), ylim=(-0.03, 1.4))
    ax.legend(loc="lower right", fontsize=13)
    save(fig, "spectral-response.png")

    shapes = [(4096, 1024), (4096, 4096), (1024, 4096)]
    labels = ["Expansion\n4096 x 1024", "Square\n4096 x 4096", "Compression\n1024 x 4096"]
    names = ["Plain", "Jordan", "muP factor", "Moonlight"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for i, (key, name) in enumerate(zip(shape_factors(1, 1), names)):
        factors = np.array([shape_factors(*s)[key] for s in shapes])
        for j, values in enumerate((factors, factors * [polar_rms(*s) for s in shapes])):
            bars = axes[j].bar(np.arange(3) + (i - 1.5) * 0.19, values, 0.18,
                               color=COLORS[i], hatch=["", "//", "..", "\\"][i], label=name)
            axes[j].bar_label(bars, fmt="%.3g", padding=4, fontsize=13,
                              rotation=90 if j else 0)
    for ax in axes:
        ax.set_xticks(np.arange(3), labels, fontsize=13)
        ax.set_axisbelow(True)
    axes[0].set(title="Shape factor s(W)", ylabel="Factor before outer LR", ylim=(0, 15))
    axes[1].set(title="Ideal full-rank update RMS", ylabel="RMS of s(W) polar(G)", ylim=(0, 0.24))
    fig.legend(names, loc="outside upper center", ncol=4, fontsize=13)
    save(fig, "shape-scaling.png")

    qk = qk_example()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    for i, (label, vals) in enumerate((("Original", [200, 50]), ("Per head", qk["per_head"]),
                                      ("Whole layer", qk["whole_layer"]))):
        bars = axes[0].bar(np.arange(2) + (i - 1) * 0.24, vals, 0.23,
                           color=COLORS[i], label=label, hatch=["", "//", ".."][i])
        axes[0].bar_label(bars, padding=3, fontsize=13)
    axes[0].set_xticks([0, 1], ["Head A", "Head B"])
    axes[0].set(title="MHA: avoid shrinking the quiet head", ylabel="Maximum score", ylim=(0, 250))
    axes[0].legend(fontsize=13, loc="upper right")
    content = [120, qk["mla_correct_content_rotary"][0], qk["mla_wrong_content_rotary"][0]]
    rotary = [80, qk["mla_correct_content_rotary"][1], qk["mla_wrong_content_rotary"][1]]
    axes[1].bar(range(3), content, color=COLORS[0], label="Content")
    axes[1].bar(range(3), rotary, bottom=content, color=COLORS[1], hatch="//", label="Rotary")
    for i, (c, r) in enumerate(zip(content, rotary)):
        axes[1].text(i, c / 2, f"{c:g}", ha="center", va="center", color="white")
        axes[1].text(i, c + r / 2, f"{r:.1f}", ha="center", va="center", color="white")
        axes[1].text(i, c + r + 4, f"{c+r:.1f}", ha="center", fontsize=13)
    axes[1].set_xticks(range(3), ["Original", "Correct", "Wrong sqrt\non rotary Q"], fontsize=13)
    axes[1].set(title="MLA: scale both score terms by 0.5", ylabel="Score at one fixed token pair", ylim=(0, 250))
    axes[1].legend(fontsize=13, loc="upper right")
    for ax in axes:
        ax.axhline(100, color="#444444", ls="--", lw=1)
        ax.set_axisbelow(True)
    save(fig, "qk-clip-example.png")


if __name__ == "__main__":
    main()
