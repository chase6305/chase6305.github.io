"""Inspect one Muon matrix update in Float64. Requires NumPy.

The reference uses an EMA buffer, optional Nesterov, and decoupled decay.
It deliberately separates an SVD polar factor from finite-step fast NS.
It is not a production optimizer and does not reproduce BF16 arithmetic.
"""

import argparse
import json
import math

import numpy as np

from muon_math import shape_factors


def matrix(value):
    if np.iscomplexobj(value):
        raise ValueError("use a real matrix")
    out = np.array(value, dtype=np.float64, copy=True)
    if out.ndim != 2 or 0 in out.shape or not np.isfinite(out).all():
        raise ValueError("use a finite, nonempty, two-dimensional matrix")
    return out


def normalized(value, eps=1e-7, convention="clamp"):
    out = matrix(value)
    if not math.isfinite(eps) or eps < 0 or convention not in ("add", "clamp"):
        raise ValueError("use finite eps >= 0 and add or clamp normalization")
    scale = float(np.abs(out).max())
    if scale == 0:
        return out
    unit = out/scale
    unit_norm = float(np.linalg.norm(unit))
    # Avoid squaring original values, or materializing a norm that can exceed
    # Float64 even though every matrix entry is finite.
    if scale >= eps:
        denominator = unit_norm + eps/scale if convention == "add" else unit_norm
        return unit/denominator
    denominator = 1+(scale/eps)*unit_norm if convention == "add" else max(1, (scale/eps)*unit_norm)
    return (out/eps)/denominator


def fast_ns(value, steps=5, eps=1e-7, convention="clamp"):
    """Five-step default, with the smaller Gram matrix and Float64 products."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    out = normalized(value, eps, convention)
    transpose = out.shape[0] > out.shape[1]
    if transpose:
        out = out.T
    for _ in range(steps):
        gram = out @ out.T
        out = 3.4445*out + (-4.775*gram + 2.0315*(gram @ gram)) @ out
    return out.T.copy() if transpose else out


def polar(value, relative_cutoff=None):
    """Canonical polar factor with numerical rank selection made explicit.

    The cutoff is max(shape)*Float64-epsilon by default, relative to the
    largest singular value. This is a numerical decision, not an exact rank
    oracle for an unknown real matrix.
    """
    out = matrix(value)
    u, singular, vh = np.linalg.svd(out, full_matrices=False)
    cutoff = max(out.shape)*np.finfo(np.float64).eps if relative_cutoff is None else relative_cutoff
    if not math.isfinite(cutoff) or not 0 <= cutoff < 1:
        raise ValueError("relative_cutoff must be in [0, 1)")
    keep = singular > cutoff * singular[0]
    return (u * keep) @ vh


def rms(value):
    return float(np.linalg.norm(value) / math.sqrt(value.size))


def step(weight, buffer, gradient, *, lr=0.01, decay=0.1, beta=0.95,
         nesterov=True, scaling="moonlight", transform="ns", eps=1e-7,
         convention="clamp", steps=5):
    """Return a trace without mutating inputs. Missing gradient skips this matrix.

    All gradients must already represent the correctly weighted global update.
    For the polar option, rank selection is documented in polar().
    """
    weight = matrix(weight)
    if not math.isfinite(lr) or lr < 0 or not math.isfinite(decay) or decay < 0:
        raise ValueError("lr and decay must be nonnegative and finite")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if transform not in ("ns", "polar"):
        raise ValueError("transform must be ns or polar")
    if scaling not in shape_factors(*weight.shape):
        raise ValueError("unknown shape convention")
    if buffer is not None:
        buffer = matrix(buffer)
        if buffer.shape != weight.shape:
            raise ValueError("buffer shape must match weight")
    if gradient is None:
        return {"skipped": True, "weight": weight, "buffer": buffer}
    gradient = matrix(gradient)
    if gradient.shape != weight.shape:
        raise ValueError("gradient shape must match weight")
    old_buffer = np.zeros_like(weight) if buffer is None else buffer
    new_buffer = beta*old_buffer + (1-beta)*gradient
    direction = (1-beta)*gradient + beta*new_buffer if nesterov else new_buffer
    ortho = (fast_ns(direction, steps, eps, convention) if transform == "ns" else polar(direction))
    factor = shape_factors(*weight.shape)[scaling]
    update = -lr * factor * ortho
    result = (1-lr*decay)*weight + update
    return {"skipped": False, "weight": result, "buffer": new_buffer,
            "gradient": gradient, "transform_input": direction,
            "transform_output": ortho, "shape_factor": factor,
            "decay_delta": -lr*decay*weight, "gradient_delta": update,
            "transform_rms": rms(ortho), "scaled_rms": rms(factor*ortho),
            "singular_values": np.linalg.svd(ortho, compute_uv=False).tolist()}


def demonstration(transform="ns"):
    weight = np.array([[1., 0.2], [0., 2.], [0.1, -0.1]])
    buffer = None
    gradients = [np.array([[1., 0.], [0., 0.1], [0., 0.]]),
                 np.array([[0., 1.], [0.1, 0.], [0., 0.]]),
                 np.zeros_like(weight), None]
    rows = []
    for label, grad in zip(("first", "second", "zero", "missing"), gradients):
        before = weight.copy()
        trace = step(weight, buffer, grad, lr=0.05, beta=0.5, decay=0.1, transform=transform)
        weight, buffer = trace["weight"], trace["buffer"]
        row = {"label": label, "skipped": trace["skipped"],
               "weight_change_norm": float(np.linalg.norm(weight-before)),
               "buffer": None if buffer is None else buffer.tolist()}
        if not trace["skipped"]:
            row.update({key: trace[key] for key in ("transform_rms", "scaled_rms", "singular_values")})
        rows.append(row)
    return {"precision": "Float64", "transform": transform,
            "normalization": "clamp", "shape": [3, 2], "scaling": "moonlight",
            "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", choices=("ns", "polar"), default="ns")
    args = parser.parse_args()
    print(json.dumps(demonstration(args.transform), indent=2, allow_nan=False))
