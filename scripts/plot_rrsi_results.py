#!/usr/bin/env python3
"""Plot published RRSI PDF values; does not run agents or infer raw-trial CIs."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/rrsi"
COLORS = {"Coding": "#3173b8", "Workspace": "#8463b1", "Engineering": "#d68434"}


def main():
    data = json.loads((BUNDLE / "results.json").read_text())
    rows = data["main_results"]
    for row in rows:
        assert math.isclose(row["rrsi"] - row["h0"], row["gain"], abs_tol=1e-9)
        assert row["ci_low"] <= row["gain"] <= row["ci_high"]
    out = BUNDLE / "assets"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelcolor": "#25354a", "text.color": "#25354a",
                         "axes.edgecolor": "#ccd5df", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(12.8, 6.9))
    fig.subplots_adjust(left=0.28, right=0.82, top=0.87, bottom=0.15)
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        color = COLORS[row["domain"]]
        if i % 2 == 0:
            ax.axhspan(y - 0.48, y + 0.48, color="#f4f6f9", zorder=0)
        ax.errorbar(row["gain"], y,
                    xerr=[[row["gain"] - row["ci_low"]],
                          [row["ci_high"] - row["gain"]]],
                    fmt="s" if row["role"] == "Evolve" else "o", color=color,
                    capsize=5, markersize=8, linewidth=2.0, zorder=3)
        ax.text(13.0, y, f"+{row['gain']:.1f}  [{row['ci_low']:.1f}, {row['ci_high']:.1f}]",
                va="center", fontsize=10.5, clip_on=False)
    labels = [f"{r['benchmark']}  |  {r['role']}" for r in rows]
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.tick_params(axis="y", length=0, pad=10, labelsize=11)
    ax.set_xlim(-0.2, 12.6)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xticks(range(0, 13, 2))
    ax.axvline(0, color="#526170", linewidth=1)
    ax.grid(axis="x", color="#e1e6ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("RRSI minus H0: reported benchmark-score points", labelpad=12)
    fig.suptitle("RRSI gains and paired task-level uncertainty", fontsize=18,
                 fontweight="bold", x=0.51, y=0.97)
    fig.text(0.28, 0.91, "Paper Fig. 3 + Table 6  |  10,000 paired task bootstrap resamples",
             fontsize=10, color="#57677a")
    ax.text(13.0, len(rows) - 0.25, "Gain [95% CI]", fontsize=11, fontweight="bold", clip_on=False)
    handles = [Line2D([], [], color=c, marker="o", linestyle="", label=d)
               for d, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, -0.005), fontsize=10)
    fig.savefig(out / "heldout-gains.webp", format="webp", dpi=180,
                pil_kwargs={"lossless": True})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 6.7))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.86)
    styles = {
        "H0": ("#7d8895", "s", "H0", (-18, 14)),
        "Unregularized": ("#d68434", "o", "Unregularized", (-116, -34)),
        "No proposal regularizers": ("#8463b1", "o", "No proposal\nregularizers", (-115, 35)),
        "No acceptance regularizers": ("#3173b8", "o", "No acceptance\nregularizers", (-127, 12)),
        "RRSI": ("#39814a", "*", "RRSI", (-20, 18)),
    }
    for row in data["ablation"]:
        color, marker, label, offset = styles[row["variant"]]
        ax.scatter(row["tokens_million"], row["ood"], c=color, marker=marker,
                   s=230 if row["variant"] == "RRSI" else 90, zorder=3)
        ax.annotate(f"{label}\n({row['tokens_million']:.2f}M, {row['ood']:.1f})",
                    (row["tokens_million"], row["ood"]), xytext=offset,
                    textcoords="offset points", color=color, fontsize=12,
                    fontweight="bold" if row["variant"] == "RRSI" else "normal")
    ax.set_xlim(1.3, 4.1)
    ax.set_ylim(38.6, 45.0)
    ax.set_xlabel("Final harness: million policy tokens per trial (evolve split)", labelpad=12)
    ax.set_ylabel("Workspace OOD average score", labelpad=12)
    ax.grid(color="#e1e6ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Better transfer with controlled inference cost", fontsize=18,
                 fontweight="bold", y=0.97)
    fig.text(0.12, 0.90, "Paper Table 2  |  OOD mean: JobBench, GDPval, APEX-Agents",
             color="#57677a", fontsize=10)
    fig.text(0.12, 0.025, "Published rounded values; no raw-trial error bars available for this table.\n"
             "This is final-harness inference cost, not the total cost of the evolution search.",
             fontsize=10, color="#57677a")
    fig.savefig(out / "cost-transfer-ablation.webp", format="webp", dpi=180,
                pil_kwargs={"lossless": True})
    plt.close(fig)
    print(json.dumps({"figures": 2, "main_rows": len(rows),
                      "ablation_rows": len(data["ablation"]), "output": str(out)}))


if __name__ == "__main__":
    main()
