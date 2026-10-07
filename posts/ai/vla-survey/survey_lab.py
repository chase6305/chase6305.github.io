#!/usr/bin/env python3
"""Recompute illustrative examples in the VLA survey article; no robot control."""

import argparse
import json
import math


PAPER_COUNTS = {2020: 2, 2021: 2, 2022: 9, 2023: 22, 2024: 63, 2025: 295}


def quantize_midpoint(value, lower=-1.0, upper=1.0, bins=256):
    """Equal-width bins, decoded to centers; endpoints belong to edge bins."""
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if not all(math.isfinite(v) for v in (value, lower, upper)):
        raise ValueError("values must be finite")
    if not lower < upper or not lower <= value <= upper:
        raise ValueError("require lower < upper and value inside the range")
    width = (upper - lower) / bins
    token = min(bins - 1, int((value - lower) / width))
    decoded = lower + (token + 0.5) * width
    return token, decoded


def chain_success(conditional_success, stages):
    """Constant conditional success, sequential stages, no recovery."""
    if not math.isfinite(conditional_success) or not 0 <= conditional_success <= 1:
        raise ValueError("success must lie in [0, 1]")
    if not isinstance(stages, int) or isinstance(stages, bool) or stages < 0:
        raise ValueError("stages must be a nonnegative integer")
    return conditional_success ** stages


def results():
    skills = [("place_cup", 0.80, 0.05), ("open_door", 0.50, 0.90),
              ("wipe_table", 0.10, 0.95)]
    scores = {name: relevance * feasibility for name, relevance, feasibility in skills}
    return {
        "provenance": {
            "paper_counts": "2405.14093v8, PDF page 53, Figure 11(a)",
            "all_other_numbers": "Illustrative calculations, not robot measurements",
        },
        "quantization": {"range_m": [-1, 1], "bins": 256,
                         "bin_width_m": 2 / 256,
                         "max_in_range_encoding_error_mm": 1000 / 256},
        "skill_scores": scores,
        "selected_skill": max(scores, key=scores.get),
        "chain_success_p095": {str(n): chain_success(0.95, n) for n in (5, 10, 20)},
        "aggregation": {"task_counts": [[90, 100], [1, 10]],
                        "micro": 91 / 110, "macro": (0.9 + 0.1) / 2},
        "paper_counts": PAPER_COUNTS,
        "paper_count_total": sum(PAPER_COUNTS.values()),
    }


def check():
    # Check the entire in-range grid plus endpoint behavior, not just one example.
    for bins in (1, 2, 17, 256):
        for step in range(2001):
            value = -1 + step / 1000
            token, decoded = quantize_midpoint(value, bins=bins)
            assert 0 <= token < bins
            assert abs(decoded - value) <= 1 / bins + 1e-12
        assert quantize_midpoint(-1, bins=bins)[0] == 0
        assert quantize_midpoint(1, bins=bins)[0] == bins - 1
    for value in (-1.01, 1.01, math.nan, math.inf):
        try:
            quantize_midpoint(value)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range / invalid input was accepted")
    assert chain_success(0.95, 0) == 1
    assert chain_success(0, 3) == 0
    assert chain_success(1, 20) == 1
    assert chain_success(0.95, 20) < chain_success(0.95, 10)
    r = results()
    assert r["selected_skill"] == "open_door"
    assert math.isclose(r["aggregation"]["micro"], 0.8272727272727273)
    assert r["paper_count_total"] == 393
    # The target velocity of the article's straight path is A - epsilon.
    action, noise, tau, delta = 0.7, -0.3, 0.4, 1e-5
    path = lambda t: (1 - t) * noise + t * action
    derivative = (path(tau + delta) - path(tau)) / delta
    assert math.isclose(derivative, action - noise, rel_tol=1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check bounds and formulas")
    args = parser.parse_args()
    if args.check:
        check()
    print(json.dumps(results(), ensure_ascii=False, indent=2))
