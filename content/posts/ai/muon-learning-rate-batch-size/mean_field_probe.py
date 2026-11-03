"""Estimate ideal polar expectations for two constructed Gaussian matrix models.

Requires NumPy. These are independent noisy gradient draws at one fixed matrix,
not optimizer steps or model training. Output uses Float64 SVD, not finite NS.
"""

import argparse
import json
import math

import numpy as np


class Moments:
    """Merge centered moments across chunks without subtracting large squares."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values):
        count = len(values)
        mean = values.mean(axis=0)
        m2 = ((values-mean)**2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        total = self.count+count
        delta = mean-self.mean
        self.m2 += m2 + delta**2*(self.count*count/total)
        self.mean += delta*(count/total)
        self.count = total

    def standard_error(self):
        return np.sqrt(self.m2/((self.count-1)*self.count))


def observe(signal, noise_std, draws, rng, chunk=2048):
    """Full-column-rank signal; noise_std belongs to a batch-gradient entry."""
    if np.iscomplexobj(signal):
        raise ValueError("use a real signal matrix")
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or 0 in signal.shape or signal.shape[0] < signal.shape[1]:
        raise ValueError("signal must be a nonempty matrix with at least as many rows as columns")
    if not np.isfinite(signal).all() or not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("signal and nonnegative noise_std must be finite")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("draws must be an integer >= 2")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    m, n = signal.shape
    left, singular, right = np.linalg.svd(signal, full_matrices=False)
    if singular[-1] <= max(signal.shape)*np.finfo(float).eps*singular[0]:
        raise ValueError("this probe requires a numerically full-column-rank signal")
    nuclear = singular.sum()
    gram = signal.T@signal
    delta = m*noise_std**2
    # SVD evaluates the same Gram formula without squaring the condition number
    # in a numerical inverse square root of signal.T @ signal.
    prediction = (left*(singular/np.sqrt(singular**2+delta)))@right
    phi_stats, score_stats, gram_stats = Moments(), Moments(), Moments()
    rank_deficient = 0
    for start in range(0, draws, chunk):
        count = min(chunk, draws-start)
        samples = signal+rng.normal(0, noise_std, size=(count,m,n))
        u, s, vh = np.linalg.svd(samples, full_matrices=False)
        # Record whether any sample approaches the numerical rank cutoff.
        deficient = s[:,-1] <= max(signal.shape)*np.finfo(float).eps*s[:,0]
        rank_deficient += int(deficient.sum())
        phi = u@vh
        phi_stats.add(phi)
        score_stats.add(np.einsum('ij,kij->k',signal,phi)/nuclear)
        gram_stats.add(samples.transpose(0,2,1)@samples)
    simple_ratio = m*n*noise_std**2/float((signal*signal).sum())
    return {
        "shape": [m,n], "draws": draws, "noise_std_per_batch_gradient_entry": noise_std,
        "signal": signal.tolist(), "mean_polar": phi_stats.mean.tolist(),
        "standard_error": phi_stats.standard_error().tolist(),
        "gram_prediction": prediction.tolist(),
        "expected_gradient_gram": (gram+delta*np.eye(n)).tolist(),
        "observed_gradient_gram": gram_stats.mean.tolist(),
        "eta_ratio_special_curvature": float(score_stats.mean),
        "eta_ratio_standard_error": float(score_stats.standard_error()),
        "eta_ratio_gram_prediction": float((signal*prediction).sum()/nuclear),
        "eta_ratio_scalar_prediction": 1/math.sqrt(1+simple_ratio),
        "numerically_rank_deficient_draws": rank_deficient,
    }


def report(draws=200000, seed=20261103):
    rng = np.random.default_rng(seed)
    cases = {"flat": np.diag([1.,1.]), "uneven": np.diag([np.sqrt(1.99),0.1])}
    return {"seed": seed, "precision": "Float64 SVD", "noise_gram_term": 0.1,
            "interval": "mean +/- 1.96 standard errors; approximate Monte Carlo interval",
            "scope": "constructed iid Gaussian gradient distributions; no training trajectory",
            "cases": {name: observe(signal, math.sqrt(0.05), draws, rng) for name, signal in cases.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    print(json.dumps(report(args.draws,args.seed), indent=2, allow_nan=False))
