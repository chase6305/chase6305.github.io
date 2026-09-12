#!/usr/bin/env python3
"""Exact-arithmetic teaching lab for X-VLA's H+1 target-time sampling.

No model, dataset, controller, or third-party dependency is required.
--write-figure also regenerates assets/action-time-grid.svg.
"""

import argparse
from bisect import bisect_left, bisect_right
from fractions import Fraction as F
from html import escape
from pathlib import Path


def sample_times(start, end, window, horizon):
    """Return state time followed by H strictly future target times."""
    start, end, window = F(start), F(end), F(window)
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if end <= start or window <= 0:
        raise ValueError("the example requires a nonempty future window")
    duration = min(window, end - start)
    return [start + F(i, horizon) * duration for i in range(horizon + 1)]


def next_target_index(times, now, *, allow_equal):
    """Choose an upcoming target, not a zero-order-held historical command.

The caller decides whether a target exactly at 'now' can still be dispatched.
This helper does not define the robot's interpolation or actuator semantics.
"""
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("target times must be nonempty and strictly increasing")
    index = (bisect_left if allow_equal else bisect_right)(times, F(now))
    if index == len(times):
        raise ValueError("no future target remains")
    return index


def checks():
    cases = [(F(1), F(10), F(1, 30)),
             (F(4), F(10), F(2, 15)),
             (F(1), F(2, 5), F(1, 75))]
    for window, end, spacing in cases:
        grid = sample_times(0, end, window, 30)
        state, actions = grid[0], grid[1:]
        assert len(grid) == 31 and len(actions) == 30 and state == 0
        assert all(b - a == spacing for a, b in zip(grid, grid[1:]))
        assert actions[0] == spacing and actions[-1] == min(end, window)
        print(f"window={actions[-1]} s; first target={actions[0]} s; spacing={spacing} s")
    # Translation of the time origin must not change spacing or action index.
    shifted = sample_times(100, 104, 4, 30)
    assert shifted == [t + 100 for t in sample_times(0, 4, 4, 30)]
    targets = sample_times(0, F(2, 5), F(2, 5), 4)[1:]
    assert targets == [F(1, 10), F(1, 5), F(3, 10), F(2, 5)]
    assert next_target_index(targets, F(1, 4), allow_equal=True) == 2
    assert next_target_index(targets, F(1, 5), allow_equal=True) == 1
    assert next_target_index(targets, F(1, 5), allow_equal=False) == 2
    assert next_target_index(targets, F(2, 5), allow_equal=True) == 3
    for now, allow_equal in [(F(2, 5), False), (F(1, 2), True)]:
        try:
            next_target_index(targets, now, allow_equal=allow_equal)
        except ValueError:
            pass
        else:
            raise AssertionError("expired target sequence was accepted")
    playback = F(30, 30)
    assert playback == 1 and F(4) / playback == 4
    print("30 targets at 30 Hz: 1 s nominal cycle; 4 s labels compressed by factor 4")
    print("at 0.20 s: index 1 if equality allowed; index 2 if strictly future")
    print("at 0.25 s: next target is index 2 at 0.30 s; expiry checks passed")
    # Distinct anchor frames do not make future-label windows independent.
    train_targets = set(sample_times(10, 20, 4, 30)[1:])
    validation_targets = set(sample_times(12, 20, 4, 30)[1:])
    shared = train_targets & validation_targets
    assert len(shared) == 15 and min(shared) == F(182, 15) and max(shared) == 14
    separate_targets = set(sample_times(15, 20, 4, 30)[1:])
    assert not train_targets & separate_targets
    print("split windows: anchors 10/12 s share 15 of 30 future-label times; 10/15 s do not")
    print("All timing checks passed. No robot timing or learned policy was measured.")


def write_figure():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="570" '
        'viewBox="0 0 1120 570" role="img" aria-labelledby="title desc">',
        '<title id="title">Same action count, different target times</title>',
        '<desc id="desc">Two sets of thirty future actions span one and four '
        'seconds on the same time axis. The state is a separate sample at zero.</desc>',
        '<rect width="1120" height="570" rx="20" fill="#f8fafc"/>',
        '<g font-family="Arial,sans-serif" fill="#16334d">',
    ]

    def text(x, y, value, size=19, anchor="start", bold=False):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                     f'text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}">'
                     f'{escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color="#9aafbf", dash=""):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')

    text(42, 53, "Same action count, different target times", 29, bold=True)
    text(42, 86, "H = 30 future targets   |   one state sample at time 0   |   shared time scale", 18)
    for second in range(5):
        x = 220 + second * 205
        line(x, 133, x, 360, "#c8d6e2", "5 5")
        text(x, 399, f"{second} s", 18, "middle")
    for duration, y, color in [(1, 190, "#bf7429"), (4, 310, "#278a86")]:
        text(42, y - 6, f"Q = {duration} s", 23, bold=True)
        text(42, y + 23, f"spacing: {F(duration, 30)} s", 17)
        grid = sample_times(0, duration, duration, 30)
        line(220, y, 1040, y, "#cad5df")
        for target in grid[1:]:
            x = 220 + 205 * float(target)
            parts.append(f'<circle cx="{x:.4f}" cy="{y}" r="2.6" fill="{color}"/>')
        parts.append(f'<circle cx="220" cy="{y}" r="5" fill="#f8fafc" stroke="#16334d" stroke-width="2"/>')
        text(220 + 205 * duration, y - 22, "action[29]", 17, "middle")
    text(220, 434, "Open circle: state     Filled dots: action[0] ... action[29]", 19)
    line(42, 467, 1078, 467, "#dce5ed")
    text(42, 502, "First target: +1/30 s in the upper row, +2/15 s in the lower row.", 20, bold=True)
    text(42, 535, "Target-time spacing is a data convention; controller playback must be defined separately.", 18)
    parts.append("</g></svg>\n")
    out = Path(__file__).resolve().parent / "assets" / "action-time-grid.svg"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-figure", action="store_true")
    args = parser.parse_args()
    checks()
    if args.write_figure:
        write_figure()
