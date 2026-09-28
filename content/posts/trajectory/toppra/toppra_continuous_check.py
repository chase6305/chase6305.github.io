"""TOPPRA 0.6.3: polynomial interval extrema versus discrete constraints.

Run: python -B toppra_continuous_check.py --output-dir results
Requires toppra, numpy, matplotlib. No robot commands or collision checking.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint


COEFFICIENTS = [[0, 1, 0, 0, 0], [0, 0, 16, -32, 16]]


def candidates(poly, start, end):
    # All stationary points plus both one-sided interval endpoints.
    roots = poly.deriv().roots()
    real = [float(r.real) for r in roots if abs(r.imag) < 1e-9
            and start < r.real < end]
    return np.asarray([start, end, *real])


def interval_peaks(grid, speeds):
    velocity_max = np.zeros(2)
    acceleration_max = np.zeros(2)
    for i, (left, right) in enumerate(zip(grid[:-1], grid[1:])):
        # Use the actual returned speed profile, matching ParametrizeConstAccel.
        u = (speeds[i + 1]**2 - speeds[i]**2) / (2 * (right - left))
        squared_speed = Polynomial([speeds[i]**2 - 2 * u * left, 2 * u])
        for joint, coeff in enumerate(COEFFICIENTS):
            q = Polynomial(coeff)
            squared_velocity = q.deriv()**2 * squared_speed
            acceleration = q.deriv(2) * squared_speed + q.deriv() * u
            v2 = squared_velocity(candidates(squared_velocity, left, right))
            acc = acceleration(candidates(acceleration, left, right))
            velocity_max[joint] = max(velocity_max[joint], np.sqrt(max(0., v2.max())))
            acceleration_max[joint] = max(acceleration_max[joint], abs(acc).max())
    return velocity_max, acceleration_max


def plan(grid_count):
    path = ta.PolynomialPath(COEFFICIENTS)
    grid = np.linspace(0, 1, grid_count)
    planner = algo.TOPPRA(
        [constraint.JointVelocityConstraint([1., 1.]),
         constraint.JointAccelerationConstraint([2., 2.],
             discretization_scheme=constraint.DiscretizationType.Interpolation)],
        path, gridpoints=grid, solver_wrapper="seidel",
        parametrizer="ParametrizeConstAccel")
    trajectory = planner.compute_trajectory(0., 0.)
    if trajectory is None:
        raise RuntimeError("no feasible parameterization")
    speeds = np.asarray(planner.problem_data.sd_vec)
    if not np.isfinite(speeds).all():
        raise RuntimeError("nonfinite path-speed profile")
    velocity, acceleration = interval_peaks(grid, speeds)
    times = np.linspace(0, trajectory.duration, 50001)
    q, dq, ddq = [trajectory(times, order) for order in (0, 1, 2)]
    np.testing.assert_allclose(q[:, 1], 16 * q[:, 0]**2 * (1 - q[:, 0])**2, atol=1e-10)
    np.testing.assert_allclose(q[[0, -1]], [[0, 0], [1, 0]], atol=1e-10)
    np.testing.assert_allclose(dq[[0, -1]], 0, atol=1e-8)
    # Dense samples can underestimate peaks; they must not exceed the extrema.
    assert np.all(abs(dq).max(axis=0) <= velocity + 1e-8)
    assert np.all(abs(ddq).max(axis=0) <= acceleration + 1e-8)
    # Uniform slowing preserves this geometric path and its zero endpoint speeds.
    scale = max(1., velocity.max(), np.sqrt(acceleration.max() / 2.)) * (1 + 1e-6)
    assert (velocity / scale <= 1 + 1e-10).all()
    assert (acceleration / scale**2 <= 2 + 1e-10).all()
    result = {"grid_points": grid_count, "duration_s": float(trajectory.duration),
              "continuous_max_velocity_rad_s": velocity.tolist(),
              "continuous_max_acceleration_rad_s2": acceleration.tolist(),
              "dense_max_velocity_rad_s": abs(dq).max(axis=0).tolist(),
              "dense_max_acceleration_rad_s2": abs(ddq).max(axis=0).tolist(),
              "uniform_time_scale": float(scale),
              "scaled_duration_s": float(scale * trajectory.duration)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows = []
    for count in (11, 21, 51, 101, 401):
        rows.append(plan(count))
    # These are fixture-specific counterexamples, not arbitrary-path guarantees.
    assert max(rows[0]["continuous_max_velocity_rad_s"]) > 1.04
    assert max(rows[0]["continuous_max_acceleration_rad_s2"]) > 2.06
    assert max(rows[-1]["continuous_max_acceleration_rad_s2"]) < 2.001
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    s = np.linspace(0, 1, 1001)
    axes[0].plot(s, 16 * s**2 * (1 - s)**2, color="#3b78ae", linewidth=2)
    axes[0].scatter([0, 1], [0, 0], color="#d07b2c")
    axes[0].set(xlabel="Joint 1 = s [rad]", ylabel="Joint 2 [rad]",
                title="The same geometric path for every grid")
    count = [row["grid_points"] for row in rows]
    v = [100 * (max(row["continuous_max_velocity_rad_s"]) - 1) for row in rows]
    a = [100 * (max(row["continuous_max_acceleration_rad_s2"]) / 2 - 1) for row in rows]
    axes[1].loglog(count, v, "o-", label="Velocity excess", color="#3b78ae")
    axes[1].loglog(count, a, "s-", label="Acceleration excess", color="#d07b2c")
    axes[1].set(xlabel="Solver grid points", ylabel="Continuous peak above limit [%]",
                title="Interval extrema, before uniform slowing")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "toppra-grid-extrema.png", dpi=160)
    plt.close(fig)
    report = {"toppra_version": version("toppra"), "path_polynomial_coefficients": COEFFICIENTS,
              "path_interval": [0, 1], "joint_velocity_limits_rad_s": [1, 1],
              "joint_acceleration_limits_rad_s2": [2, 2], "endpoint_path_speeds": [0, 0],
              "constraint_discretization": "Interpolation", "parametrizer": "ParametrizeConstAccel",
              "verification": "polynomial stationary points and one-sided segment endpoints, float64",
              "results": rows}
    (args.output_dir / "toppra-grid-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
