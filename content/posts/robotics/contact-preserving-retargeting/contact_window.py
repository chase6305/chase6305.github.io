#!/usr/bin/env python3
"""One-dimensional contact timing model in integer centimeters per interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULTS = {"distance": 12, "step": 4, "intervals": 3, "pinned": 0}


def compare_paths(distance: int, step: int, intervals: int, pinned: int) -> dict:
    """Return paths for a one-step adjustment and a window with a zero fixed prefix."""
    bounds = ((distance, 0, 100), (step, 0, 100), (intervals, 1, 12), (pinned, 0, intervals))
    if any(type(value) is not int or not lower <= value <= upper for value, lower, upper in bounds):
        raise ValueError("Require integer distance/step 0–100, intervals 1–12, pinned 0–intervals")

    free_intervals = intervals - pinned
    late_end = min(distance, step) if free_intervals else 0
    window_end = min(distance, free_intervals * step)
    return {
        "free_intervals": free_intervals,
        "late_gap": distance - late_end,
        "window_gap": distance - window_end,
        "late_path": [0] * intervals + [late_end],
        "window_path": [max(0, window_end - (intervals - index) * step) for index in range(intervals + 1)],
    }


def reference() -> dict:
    return {"input": DEFAULTS, "result": compare_paths(**DEFAULTS)}


def check() -> int:
    """Independently enumerate integer states and keep the least-motion history."""
    samples = 0
    for distance in range(11):
        for step in range(5):
            for intervals in range(1, 7):
                for pinned in range(intervals + 1):
                    histories = {0: (0, [0])}
                    for index in range(1, intervals + 1):
                        positions = [0] if index <= pinned else range(distance + 1)
                        next_histories = {}
                        for position in positions:
                            candidates = [
                                (cost + position**2, [*path, position])
                                for previous, (cost, path) in histories.items()
                                if abs(position - previous) <= step
                            ]
                            if candidates:
                                next_histories[position] = min(candidates)
                        histories = next_histories

                    reached = max(histories)
                    one_step = max(p for p in range(distance + 1) if p <= step and (p == 0 or pinned < intervals))
                    result = compare_paths(distance, step, intervals, pinned)
                    if result != {
                        "free_intervals": intervals - pinned,
                        "late_gap": distance - one_step,
                        "window_gap": distance - reached,
                        "late_path": [0] * intervals + [one_step],
                        "window_path": histories[reached][1],
                    }:
                        raise AssertionError((distance, step, intervals, pinned, result))
                    samples += 1
    default_path = Path(__file__).with_name("assets") / "window-default.json"
    if default_path.exists() and json.loads(default_path.read_text(encoding="utf-8")) != reference():
        raise AssertionError("Article fallback differs from the default calculation")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with independent discrete state enumeration")
    parser.add_argument("--distance", type=int, default=DEFAULTS["distance"], help="Contact goal in centimeters")
    parser.add_argument("--step", type=int, default=DEFAULTS["step"], help="Maximum centimeters per interval")
    parser.add_argument("--intervals", type=int, default=DEFAULTS["intervals"], help="Total intervals to contact")
    parser.add_argument("--pinned", type=int, default=DEFAULTS["pinned"], help="Intervals fixed at the initial position")
    args = parser.parse_args()
    try:
        if args.check:
            print(f"Checked {check()} cases against discrete state enumeration.")
        else:
            values = {name: getattr(args, name) for name in DEFAULTS}
            print(json.dumps({"input": values, "result": compare_paths(**values)}, indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
