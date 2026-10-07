"""Synthetic coordinate and metric checks for the MolmoMotion article.

No model weights, third-party datasets, network or GPU are used.
This strict example parser is a verification aid, not the repository parser.
Run: python -B trajectory_contract_lab.py [--output-dir assets]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


def serialize(points, anchor, start_index):
    """points: (P,F,3) absolute XYZ meters; one shared anchor: (3,)."""
    points, anchor = np.asarray(points), np.asarray(anchor)
    if (points.ndim != 3 or points.shape[-1] != 3 or anchor.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(anchor).all()):
        raise ValueError("finite points (P,F,3) and anchor (3,) required")
    quantized = np.rint((points-anchor)*1000).astype(np.int64)
    frames = []
    for f in range(points.shape[1]):
        fields = [f"{start_index+f:.1f}"]
        for p in range(points.shape[0]):
            fields.extend([str(p+1), *map(str, quantized[p, f])])
        frames.append(" ".join(fields))
    return '<tracks coords="' + ';'.join(frames) + '">3d object trajectories</tracks>'


def parse_checked(text, *, points, frames, start_index, anchor):
    """Return XYZ and coverage separately. Reject malformed or duplicate entries.

    Missing entries stay NaN. A valid numeric zero is never treated as missing.
    """
    match = re.fullmatch(r'<tracks\s+coords="([^"]*)">[^<]*</tracks>', text.strip())
    if match is None:
        raise ValueError("incomplete or malformed tracks block")
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("invalid shared anchor")
    if min(points, frames) <= 0:
        raise ValueError("positive point/frame counts required")
    xyz = np.full((points, frames, 3), np.nan)
    present = np.zeros((points, frames), dtype=bool)
    seen_frames = set()
    for row in match.group(1).split(';'):
        fields = row.split()
        if not fields or (len(fields)-1) % 4:
            raise ValueError("invalid point tuple")
        stamp = float(fields[0])
        if not np.isfinite(stamp) or stamp != int(stamp):
            raise ValueError("frame index must be finite and integral")
        f = int(stamp)-start_index
        if not 0 <= f < frames or f in seen_frames:
            raise ValueError("duplicate or out-of-range frame")
        seen_frames.add(f)
        for i in range(1, len(fields), 4):
            p = int(fields[i])-1
            if not 0 <= p < points or present[p, f]:
                raise ValueError("duplicate or out-of-range point id")
            xyz[p, f] = np.array([int(v) for v in fields[i+1:i+4]])/1000 + anchor
            present[p, f] = True
    return xyz, present


def metrics(prediction, truth, visible, *, end="last_visible", inclusive=True):
    """Global point-frame ADE/PWT; selectable FDE convention.

    The visibility mask comes from the evaluation protocol, never from model
    success. Nonfinite predictions at scored entries fail instead of disappearing.
    """
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    visible = np.asarray(visible, dtype=bool)
    if (truth.ndim != 3 or truth.shape[-1] != 3
            or prediction.shape != truth.shape or visible.shape != truth.shape[:2]):
        raise ValueError("incompatible prediction, truth and mask shapes")
    if not visible.any():
        raise ValueError("no scored entries; an empty benchmark is not zero error")
    if not np.isfinite(prediction[visible]).all() or not np.isfinite(truth[visible]).all():
        raise ValueError("nonfinite scored coordinate")
    error = np.linalg.norm(prediction-truth, axis=-1)
    if end == "last_visible":
        final = [error[p, np.flatnonzero(v)[-1]]
                 for p, v in enumerate(visible) if v.any()]
    elif end == "common_final":
        final = error[:, -1][visible[:, -1]]
    else:
        raise ValueError("unknown FDE convention")
    if len(final) == 0:
        raise ValueError("no visible endpoints for this FDE convention")
    thresholds = np.array([.01, .02, .05, .10, .20])
    hits = error[visible, None] <= thresholds if inclusive else error[visible, None] < thresholds
    per_threshold = hits.mean(axis=0)
    return {"ADE_m": float(error[visible].mean()), "FDE_m": float(np.mean(final)),
            "PWT": float(per_threshold.mean()), "PWT_per_threshold": per_threshold.tolist(),
            "scored_point_frames": int(visible.sum())}


def transform(points, rotation, translation):
    """Row-vector array convention: p_new = p_old @ R.T + t."""
    return np.asarray(points) @ np.asarray(rotation).T + translation


def small_example():
    truth = np.zeros((2, 3, 3))
    prediction = truth.copy()
    prediction[:, :, 0] = [[.015, .025, .035], [.045, .085, .300]]
    visible = np.array([[True, True, True], [True, True, False]])
    return prediction, truth, visible


def checks():
    passed = []
    anchor = np.array([.4, .1, .8])
    rng = np.random.default_rng(20260925)
    points = anchor + rng.uniform(-.1, .1, (8, 30, 3))
    text = serialize(points, anchor, start_index=3)
    decoded, coverage = parse_checked(text, points=8, frames=30, start_index=3, anchor=anchor)
    assert coverage.all()
    assert np.max(np.abs(decoded-points)) <= .0005 + 1e-12
    assert np.max(np.linalg.norm(decoded-points, axis=-1)) <= np.sqrt(3)*.0005+1e-12
    passed.append("8x30x3 shared-anchor roundtrip and millimeter quantization bound")

    single = '<tracks coords="3.0 1 0 0 0">3d object trajectories</tracks>'
    decoded, coverage = parse_checked(single, points=2, frames=2, start_index=3, anchor=anchor)
    np.testing.assert_allclose(decoded[0, 0], anchor)
    assert coverage.sum() == 1 and np.isnan(decoded[1, 0]).all()
    try:
        parse_checked(single[:-9], points=2, frames=2, start_index=3, anchor=anchor)
    except ValueError:
        passed.append("legal zero delta differs from missing data; truncated block rejected")
    else:
        raise AssertionError("truncated text accepted")

    prediction, truth, visible = small_example()
    last = metrics(prediction, truth, visible)
    common = metrics(prediction, truth, visible, end="common_final")
    np.testing.assert_allclose([last['ADE_m'],last['FDE_m'],common['FDE_m'],last['PWT']],
                               [.041,.060,.035,.6])
    passed.append("ADE/PWT use five valid entries; two FDE endpoint conventions differ")
    prediction[1, -1] = 100000.0  # invisible point never contributes
    assert metrics(prediction, truth, visible) == last
    prediction[0, 0] = np.nan  # visible prediction failure must remain a failure
    try:
        metrics(prediction, truth, visible)
    except ValueError:
        passed.append("ground-truth mask is independent of prediction success")
    else:
        raise AssertionError("failed prediction was silently masked")

    prediction, truth, visible = small_example()
    angle = .7
    rotation = np.array([[np.cos(angle),-np.sin(angle),0],
                         [np.sin(angle),np.cos(angle),0],[0,0,1]])
    shift = np.array([2,-1,.5])
    transformed = metrics(transform(prediction,rotation,shift),transform(truth,rotation,shift),visible)
    np.testing.assert_allclose(transformed['ADE_m'],last['ADE_m'],atol=1e-14)
    passed.append("shared rigid frame change preserves Euclidean trajectory errors")

    world = np.array([1.0, 0.0, 2.0])
    camera_centers = np.array([[0,0,0],[.1,0,0],[.2,0,0]])
    local = world-camera_centers
    restored = local+camera_centers
    np.testing.assert_allclose(restored, np.tile(world, (3,1)))
    assert np.ptp(local[:,0]) > .19
    passed.append("moving camera makes a static world point move unless frames are aligned")
    return passed, last, common


def plot_example(destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prediction, truth, visible = small_example()
    error_mm = np.linalg.norm(prediction-truth, axis=-1)*1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.2)
    for p, (color, name) in enumerate([('#397fba','Point A'),('#9161b1','Point B')]):
        indexes = np.flatnonzero(visible[p])
        axes[0].plot(indexes+1,error_mm[p,indexes],'-o' if p==0 else '--^',label=name,color=color,lw=2)
        axes[0].scatter(indexes[-1]+1,error_mm[p,indexes[-1]],s=100,color=color,marker='s',zorder=3)
    axes[0].text(2.95,55,'B: not visible\nat the final frame',ha='right',fontsize=10,color='#626b75')
    axes[0].set(xticks=[1,2,3],xticklabels=['t1','t2','t3'],xlabel='Future frame',
                ylabel='3D position error (mm)',title='Only ground-truth-visible entries are scored',ylim=(0,100))
    axes[0].legend(loc='upper left')
    labels=['ADE','FDE\ncommon final','FDE\nlast visible']
    bars=axes[1].bar(labels,[41,35,60],color=['#559667','#397fba','#9161b1'],width=.6)
    axes[1].bar_label(bars,fmt='%.0f mm',padding=5)
    axes[1].set(ylabel='Error (mm)',ylim=(0,80),title='Same trajectories, different endpoint conventions')
    fig.savefig(destination/'metric-mask-example.png',dpi=170,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    passed,last,common=checks()
    report={'experiment':'synthetic; not model evaluation','checks':passed,
            'last_visible_endpoint':last,'common_final_endpoint':common}
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        plot_example(args.output_dir)
        (args.output_dir/'trajectory-lab-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
