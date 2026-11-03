"""Local batch-scaling models and data-clock examples, not a tuning prescription.

Only Python's standard library is required. Gaussian expectations use adaptive
Simpson integration on [0, 12], doubled by symmetry; omitted probability < 4e-33.
"""

import json
import math


def positive(*values):
    if not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("arguments must be positive and finite")


def mean_field_lr(batch, critical_batch, limit=1):
    positive(batch, critical_batch, limit)
    return limit / math.sqrt(1 + critical_batch / batch)


def ema_variance_factor(beta):
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    return (1 - beta) / (1 + beta)


def nesterov_variance_factor(beta):
    ema_variance_factor(beta)
    # h_t=(1-beta)g_t+beta*m_t: current coefficient 1-beta**2;
    # older coefficients (1-beta)*beta**(j+1), j=1,2,... .
    return (1 - beta**2)**2 + (1 - beta)**2 * beta**4 / (1 - beta**2)


def data_clock_beta(beta, batch_ratio):
    ema_variance_factor(beta)
    positive(batch_ratio)
    return beta**batch_ratio


def data_clock_decay(lr_old, decay_old, lr_new, batch_ratio):
    """Match only cumulative shrinkage at equal samples, for constant rates."""
    positive(lr_old, lr_new, batch_ratio)
    if not math.isfinite(decay_old) or decay_old < 0 or lr_old * decay_old >= 1:
        raise ValueError("require 0 <= lr_old * decay_old < 1")
    return -math.expm1(batch_ratio * math.log1p(-lr_old * decay_old)) / lr_new


def noise_estimate(gradients, microbatch):
    """Independent, equally weighted gradients at one parameter position.

    Each input gradient is a flat nonempty list. A nonpositive corrected signal
    returns None for the ratio; no artificial positive floor is imposed.
    """
    positive(microbatch)
    gradients = [list(g) for g in gradients]
    k = len(gradients)
    if k < 2 or not gradients[0] or any(len(g) != len(gradients[0]) for g in gradients):
        raise ValueError("need at least two equal-sized nonempty gradients")
    if not all(math.isfinite(x) for g in gradients for x in g):
        raise ValueError("gradient entries must be finite")
    mean = [math.fsum(col) / k for col in zip(*gradients)]
    variance = math.fsum((g[i] - mean[i])**2 for g in gradients for i in range(len(mean))) / (k - 1)
    signal = math.fsum(x*x for x in mean) - variance / k
    noise = microbatch * variance
    return {"signal": signal, "single_sample_noise": noise,
            "critical_batch_estimate": noise / signal if signal > 0 else None}


def _integrate(f, left=0.0, right=12.0, tolerance=1e-11):
    def refine(a, b, fa, fm, fb, whole, tol, depth):
        m = (a + b) / 2
        fl, fr = f((a + m) / 2), f((m + b) / 2)
        lo = (m - a) * (fa + 4 * fl + fm) / 6
        hi = (b - m) * (fm + 4 * fr + fb) / 6
        delta = lo + hi - whole
        if abs(delta) <= 15 * tol:
            return lo + hi + delta / 15
        if depth == 0:
            raise ArithmeticError("integration tolerance not reached")
        return (refine(a, m, fa, fl, fm, lo, tol / 2, depth - 1)
                + refine(m, b, fm, fr, fb, hi, tol / 2, depth - 1))
    mid = (left + right) / 2
    fa, fm, fb = f(left), f(mid), f(right)
    return refine(left, right, fa, fm, fb,
                  (right - left) * (fa + 4 * fm + fb) / 6, tolerance, 35)


def gaussian_curvature(batch, noise_std=100, tolerance=1e-11):
    """Exact local optimum for G=(1,0), H=diag(100,1), noise=(0,sZ/sqrt(B))."""
    positive(batch, noise_std, tolerance)
    scale2 = noise_std**2 / batch
    def moment(power):
        return _integrate(lambda z: math.sqrt(2 / math.pi) * math.exp(-z*z / 2)
                          / (1 + scale2 * z*z)**power, tolerance=tolerance)
    alignment, first_square = moment(0.5), moment(1)
    curvature = 1 + 99 * first_square
    return {"alignment": alignment, "curvature": curvature, "lr": alignment / curvature}


def reference():
    return {
        "lr_ratio_256_to_1024_Bc4096": mean_field_lr(1024, 4096) / mean_field_lr(256, 4096),
        "ema_noise_factor_beta095": ema_variance_factor(0.95),
        "nesterov_noise_factor_beta095": nesterov_variance_factor(0.95),
        "beta_for_fourfold_batch": data_clock_beta(0.95, 4),
        "decay_for_fourfold_batch_double_lr": data_clock_decay(0.001, 0.1, 0.002, 4),
        "noise_estimate": noise_estimate([[3, 1], [1, 3], [3, 3], [1, 1]], 8),
        "gaussian_s100": {str(b): gaussian_curvature(b) for b in (1, 10, 100, 1000, 1000000)},
    }


if __name__ == "__main__":
    print(json.dumps(reference(), indent=2, allow_nan=False))
