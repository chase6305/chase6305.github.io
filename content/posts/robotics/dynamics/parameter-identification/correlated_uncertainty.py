"""Known temporal noise correlation changes OLS uncertainty, not its objective.

Python 3.10+, NumPy; optional Matplotlib figure. No measured robot data.
Run: python -B correlated_uncertainty.py --output correlated-uncertainty.json
"""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("correlated-uncertainty.json"))
    parser.add_argument("--figure", type=Path, help="Optional PNG plot; needs Matplotlib")
    args = parser.parse_args()
    samples, trials, seed = 200, 2500, 20260928
    dt, amplitude, frequency, sigma, rho = .01, .03, .7, .05, .9
    time = np.arange(samples) * dt
    omega = 2 * np.pi * frequency
    velocity = amplitude * omega * np.cos(omega * time)
    acceleration = -amplitude * omega**2 * np.sin(omega * time)
    design = np.column_stack((acceleration, velocity))
    truth = np.array([2., .4])  # kg, N*s/m; force = mass*a + damping*v
    operator, _, rank, _ = np.linalg.lstsq(design, np.eye(samples), rcond=None)
    assert rank == 2
    np.testing.assert_allclose(operator @ design, np.eye(2), atol=1e-14)
    separation = abs(np.arange(samples)[:, None] - np.arange(samples)[None, :])
    noise_covariance = sigma**2 * rho**separation
    covariance = operator @ noise_covariance @ operator.T
    independent_covariance = sigma**2 * operator @ operator.T

    # Generate stationary AR(1) noise by recurrence, independently of the
    # covariance-matrix calculation. sigma is the marginal, not innovation, std.
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((samples, trials))
    noise = np.empty_like(white)
    noise[0] = sigma * white[0]
    for i in range(1, samples):
        noise[i] = rho * noise[i - 1] + sigma * np.sqrt(1 - rho**2) * white[i]
    force = (design @ truth)[:, None] + noise
    estimates = (operator @ force).T
    independent_fit = np.linalg.lstsq(design, force[:, :3], rcond=None)[0].T
    np.testing.assert_allclose(estimates[:3], independent_fit, atol=1e-13)
    errors = estimates - truth
    expected_std = np.sqrt(np.diag(covariance))
    naive_std = np.sqrt(np.diag(independent_covariance))
    empirical_std = estimates.std(axis=0, ddof=1)
    z = NormalDist().inv_cdf(.975)
    coverage = (abs(errors) <= z * expected_std).mean(axis=0)
    naive_coverage = (abs(errors) <= z * naive_std).mean(axis=0)
    np.testing.assert_allclose(empirical_std, expected_std, rtol=.08)
    assert ((coverage > .92) & (coverage < .98)).all()
    assert (naive_coverage < .55).all()
    assert (expected_std / naive_std > 3.5).all()
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), constrained_layout=True)
        for i, (ax, label) in enumerate(zip(axes, ("Mass estimate [kg]", "Damping estimate [N s/m]"))):
            ax.hist(estimates[:, i], bins=45, density=True, color="#c8dcec",
                    edgecolor="white", label="2500 independent trials")
            xx = np.linspace(truth[i] - 4 * expected_std[i], truth[i] + 4 * expected_std[i], 501)
            for std, color, style, name in [
                    (expected_std[i], "#3b78ae", "-", "Known correlated noise"),
                    (naive_std[i], "#cf772c", "--", "Incorrect independence")]:
                density = np.exp(-.5 * ((xx - truth[i]) / std)**2) / (std * np.sqrt(2 * np.pi))
                ax.plot(xx, density, color=color, linestyle=style, linewidth=2, label=name)
            ax.axvline(truth[i], color="#64748b", linestyle=":", label="True parameter")
            ax.set(xlabel=label, ylabel="Probability density",
                   title="Same OLS estimates; different uncertainty models")
            ax.legend(fontsize=8.5)
            ax.grid(alpha=.15)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=160)
        plt.close(fig)
    rows = []
    for i, (name, unit) in enumerate([("mass", "kg"), ("damping", "N*s/m")]):
        rows.append({"parameter": name, "unit": unit, "truth": float(truth[i]),
                     "mean_estimate": float(estimates[:, i].mean()),
                     "std_if_noise_independent": float(naive_std[i]),
                     "std_with_temporal_covariance": float(expected_std[i]),
                     "monte_carlo_std": float(empirical_std[i]),
                     "nominal_95pct_coverage_if_independent": float(naive_coverage[i]),
                     "nominal_95pct_coverage_with_known_covariance": float(coverage[i])})
    result = {"numpy": np.__version__, "seed": seed, "trials": trials,
              "samples_per_trial": samples, "sample_period_s": dt,
              "position_amplitude_m": amplitude, "motion_frequency_hz": frequency,
              "force_noise_marginal_std_N": sigma, "noise_ar1_coefficient": rho,
              "noise_innovation_std_N": float(sigma * np.sqrt(1 - rho**2)),
              "regressor_rank": int(rank), "true_OLS_covariance": covariance.tolist(),
              "independent_noise_covariance": independent_covariance.tolist(),
              "scope": "fixed exact regressors; known Gaussian noise covariance; no ridge or constraints",
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
