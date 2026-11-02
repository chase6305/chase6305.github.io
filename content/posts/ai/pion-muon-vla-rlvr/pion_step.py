"""Trace one unsharded Float64 Pion matrix step, without a training framework.

The EMA and Nesterov conventions follow the pinned Pion implementations.
This is a pure calculation: caller arrays are never changed. No AdamW,
autograd, BF16, convolution flattening or distributed execution is included.
"""
import argparse
import json
import math

import numpy as np

from pion_matrix import transform
from pion_spectral import check_steps, finite_nonnegative


def matrix(value, name, shape=None):
    """Return a finite, real, independent Float64 matrix."""
    source = np.asarray(value)
    if source.ndim != 2 or min(source.shape) < 1 or not np.isrealobj(source):
        raise ValueError(f"{name} must be a nonempty real matrix")
    try:
        result = np.array(source, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite and match the weight shape")
    return result


def step(weight, gradient, buffer=None, *, lr=.01, weight_decay=.1,
         beta=.5, nesterov=True, promotion_steps=1, epsilon=0.,
         scale_factor=.2, missing_gradient="skip"):
    """Return (new weight, new buffer, trace) for one whole-matrix step.

    ``gradient=None`` keeps the buffer. ``skip`` also keeps the weight, as in
    RL/verl; ``decay`` applies only weight decay, as in VLA-Adapter. An actual
    zero matrix is a gradient observation and can yield a momentum update.
    All scalar arguments are explicit example settings, not training advice.
    """
    before = matrix(weight, "weight")
    previous = np.zeros_like(before) if buffer is None else matrix(buffer, "buffer", before.shape)
    lr = finite_nonnegative(lr, "lr")
    weight_decay = finite_nonnegative(weight_decay, "weight_decay")
    beta = finite_nonnegative(beta, "beta")
    epsilon = finite_nonnegative(epsilon, "epsilon")
    scale_factor = finite_nonnegative(scale_factor, "scale_factor")
    promotion_steps = check_steps(promotion_steps)
    if beta >= 1 or type(nesterov) is not bool:
        raise ValueError("beta must be below 1 and nesterov must be boolean")
    if missing_gradient not in ("skip", "decay"):
        raise ValueError("missing_gradient must be 'skip' or 'decay'")
    decay_factor = 1 - lr * weight_decay
    scale = scale_factor * math.sqrt(max(before.shape))
    if not math.isfinite(decay_factor) or not math.isfinite(scale):
        raise ValueError("decay and dimension factors must be finite")

    if gradient is None:
        after = before.copy() if missing_gradient == "skip" else before * decay_factor
        current = previous.copy()
        direction = None
        filtered = np.zeros_like(before)
        gradient_delta = np.zeros_like(before)
        status = "missing_skip" if missing_gradient == "skip" else "missing_decay_only"
    else:
        observed = matrix(gradient, "gradient", before.shape)
        current = beta * previous + (1-beta) * observed
        direction = (1-beta) * observed + beta * current if nesterov else current.copy()
        if not np.isfinite(current).all() or not np.isfinite(direction).all():
            raise ValueError("momentum calculation must remain finite")
        filtered = transform(direction, promotion_steps, epsilon)
        gradient_delta = -(lr * scale) * filtered
        after = before * decay_factor + gradient_delta
        status = "gradient_observed"
    if not np.isfinite(after).all():
        raise ValueError("weight update must remain finite")
    # A finite coordinate RMS can have an unrepresentable intermediate norm.
    # Scale before taking the norm; roundoff must not make RMS exceed max |x|.
    largest_delta = float(np.max(np.abs(gradient_delta)))
    gradient_rms = 0. if largest_delta == 0 else largest_delta * min(
        1., math.hypot(*(gradient_delta/largest_delta).flat) / math.sqrt(before.size))
    trace = {
        "status": status,
        "weight_before": before.tolist(),
        "buffer_before": previous.tolist(),
        "buffer_after": current.tolist(),
        "direction_before_normalization": None if direction is None else direction.tolist(),
        "filtered_direction": filtered.tolist(),
        "dimension_scale": scale,
        "gradient_update_rms": gradient_rms,
        "configured_decay_factor": decay_factor,
        "applied_decay_factor": 1. if status == "missing_skip" else decay_factor,
        "weight_after": after.tolist(),
    }
    return after, current, trace


def report(missing_gradient="skip"):
    """Three observations: nonzero gradient, zero gradient, missing gradient."""
    weight = np.diag([1., 2.])
    buffer = np.zeros_like(weight)
    observations = [np.diag([.8, .2]), np.zeros_like(weight), None]
    rows = []
    for index, gradient in enumerate(observations, 1):
        weight, buffer, trace = step(weight, gradient, buffer, missing_gradient=missing_gradient)
        rows.append({"step": index, "gradient": None if gradient is None else gradient.tolist(), **trace})
    return {
        "scope": "Constructed Float64 whole-matrix trace; no model, BF16, AdamW or distributed training",
        "settings": {"lr": .01, "weight_decay": .1, "beta": .5, "nesterov": True,
                     "promotion_steps": 1, "suppression_steps": 4, "epsilon": 0.,
                     "scale_factor": .2, "missing_gradient": missing_gradient},
        "steps": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-gradient", choices=("skip", "decay"), default="skip")
    args = parser.parse_args()
    print(json.dumps(report(args.missing_gradient), indent=2, allow_nan=False))
