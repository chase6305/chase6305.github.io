"""Recompute the article's scalar and shape examples; Python standard library only.

This is a mathematical reference, not a training optimizer. Dimensions follow
PyTorch Linear: W.shape == (d_out, d_in). All numbers are constructed examples.
"""

import json
import math


def ns_response(x, steps=5):
    """Apply the published fast NS polynomial to a normalized singular value."""
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("initial singular value must be finite and in [0, 1]")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    for _ in range(steps):
        x = 3.4445 * x - 4.775 * x**3 + 2.0315 * x**5
    return x


def shape_factors(d_out, d_in):
    for d in (d_out, d_in):
        if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
            raise ValueError("dimensions must be positive integers")
    return {
        "plain": 1.0,
        "jordan": math.sqrt(max(1, d_out / d_in)),
        "muP_factor": math.sqrt(d_out / d_in),
        "moonlight": 0.2 * math.sqrt(max(d_out, d_in)),
    }


def polar_rms(d_out, d_in, rank=None):
    """RMS of the canonical partial-isometry polar factor, including rank zero."""
    shape_factors(d_out, d_in)
    rank = min(d_out, d_in) if rank is None else rank
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= min(d_out, d_in):
        raise ValueError("rank must be an integer between zero and min(shape)")
    return math.sqrt(rank / (d_out * d_in))


def convert_convention(lr, decay, factor_from, factor_to):
    """Keep lr * shape_factor and lr * decay fixed for one update."""
    if not all(math.isfinite(x) and x > 0 for x in (lr, factor_from, factor_to)):
        raise ValueError("learning rate and factors must be positive and finite")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("decay must be nonnegative and finite")
    return lr * factor_from / factor_to, decay * factor_to / factor_from


def clip_ratio(max_score, threshold):
    if not math.isfinite(max_score) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("scores must be finite and threshold positive")
    return threshold / max_score if max_score > threshold else 1.0


def qk_example():
    gamma = clip_ratio(200, 100)
    return {
        "per_head": [200 * gamma, 50.0],
        "whole_layer": [200 * gamma, 50 * gamma],
        "mla_correct_content_rotary": [120 * gamma, 80 * gamma],
        "mla_wrong_content_rotary": [120 * gamma, 80 * math.sqrt(gamma)],
        "stale_statistic_new_score": 400 * gamma,
    }


def reference():
    return {
        "ns_one_at_one": ns_response(1, 1),
        "ns_five_at_half": ns_response(0.5),
        "expansion_4096_by_1024": shape_factors(4096, 1024),
        "compression_1024_by_4096": shape_factors(1024, 4096),
        "full_rank_moonlight_rms": 12.8 * polar_rms(4096, 1024),
        "rank_256_moonlight_rms": 12.8 * polar_rms(4096, 1024, 256),
        "moonlight_to_plain_lr_decay": convert_convention(0.001, 0.1, 12.8, 1),
        "qk": qk_example(),
    }


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, allow_nan=False))
