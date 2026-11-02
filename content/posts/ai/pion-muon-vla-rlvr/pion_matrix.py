"""Float64 NumPy illustration of Pion's polynomial core, not an optimizer.

No momentum state, weight decay, Nesterov, mixed precision or dimension scale.
The default epsilon=0 is the ideal scale-invariant normalization; the optional
epsilon argument models the additive denominator separately from BF16 effects.
Per-head axes follow PyTorch Linear weight layout (out_features, in_features).
"""
import argparse
import json
import numpy as np
from pion_spectral import PROMOTION, SUPPRESSION, TOTAL_STEPS, check_steps, finite_nonnegative


def transform(momentum, promotion_steps=1, epsilon=0.):
    """Normalize each matrix by ||M||_F + epsilon, then apply five polynomials."""
    promotion_steps = check_steps(promotion_steps)
    epsilon = finite_nonnegative(epsilon, "epsilon")
    source = np.asarray(momentum)
    if source.ndim != 2 or min(source.shape) < 1 or not np.isrealobj(source):
        raise ValueError("momentum must be a nonempty real matrix")
    source = np.asarray(source, dtype=np.float64)
    if not np.isfinite(source).all():
        raise ValueError("momentum must be finite")
    # Scale first: normalization should not overflow for finite large entries.
    maximum = np.max(np.abs(source))
    if maximum == 0:
        return np.zeros_like(source)
    scaled = source / maximum
    norm = np.linalg.norm(scaled)
    if epsilon > maximum:
        ratio = maximum/epsilon
        value = (scaled*ratio)/(1+norm*ratio)
    else:
        value = scaled/(norm+epsilon/maximum)
    transpose = value.shape[0] > value.shape[1]
    if transpose:
        value = value.T
    schedule = [PROMOTION] * promotion_steps + [SUPPRESSION] * (TOTAL_STEPS-promotion_steps)
    for a, b, c in schedule:
        gram = value @ value.T
        value = a*value + (b*gram + c*(gram @ gram)) @ value
    return value.T if transpose else value


def per_head_transform(momentum, heads, axis=0, promotion_steps=1, epsilon=0.):
    source = np.asarray(momentum)
    if source.ndim != 2 or type(heads) is not int or heads < 1 or type(axis) is not int or axis not in (0, 1):
        raise ValueError("Use a matrix, positive integer heads, and axis 0 or 1")
    if source.shape[axis] % heads:
        raise ValueError("head count must divide the selected projection dimension")
    pieces = np.split(source, heads, axis=axis)
    return np.concatenate([transform(piece, promotion_steps, epsilon) for piece in pieces], axis=axis)


def head_report():
    """Two equal-shaped spectra at a 10:1 raw amplitude ratio; no weight data."""
    source = np.diag([8., 1., .8, .1])
    whole = transform(source, 1)
    per_head = per_head_transform(source, 2, 0, 1)
    return {"scope": "Constructed unsharded Float64 matrix; no optimizer dimension scale or pretrained weights",
            "promotion_steps": 1, "suppression_steps": 4, "epsilon": 0.,
            "input_diagonal": np.diag(source).tolist(),
            "whole_diagonal": np.diag(whole).tolist(), "per_head_diagonal": np.diag(per_head).tolist(),
            "input_block_norms": [float(np.linalg.norm(block)) for block in np.split(source, 2)],
            "whole_block_norms": [float(np.linalg.norm(block)) for block in np.split(whole, 2)],
            "per_head_block_norms": [float(np.linalg.norm(block)) for block in np.split(per_head, 2)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-report", action="store_true")
    args = parser.parse_args()
    if args.head_report:
        print(json.dumps(head_report(), indent=2, allow_nan=False))
    else:
        sample = np.diag([.8, .4, .2, .02])
        print("Input singular values:", np.linalg.svd(sample, compute_uv=False))
        print("Pion 1+4 singular values:", np.linalg.svd(transform(sample, 1), compute_uv=False))
