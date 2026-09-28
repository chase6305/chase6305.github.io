"""Planar 2R calibration: base yaw and first-joint zero share a gauge.

Run with Python 3.10+ and NumPy. This is a noiseless algebra check, not a
measurement-accuracy benchmark. Lengths are metres; angles are radians.
"""
import json

import numpy as np


def position(q, parameters):
    beta, offset1, offset2 = parameters
    a = q[:, 0] + beta + offset1
    b = a + q[:, 1] + offset2
    return np.column_stack((np.cos(a) + 0.6 * np.cos(b),
                            np.sin(a) + 0.6 * np.sin(b)))


def parameter_jacobian(q, parameters):
    beta, offset1, offset2 = parameters
    a = q[:, 0] + beta + offset1
    b = a + q[:, 1] + offset2
    common = np.column_stack((-np.sin(a) - 0.6 * np.sin(b),
                              np.cos(a) + 0.6 * np.cos(b)))
    second = np.column_stack((-0.6 * np.sin(b), 0.6 * np.cos(b)))
    # Flatten in sample order: x0, y0, x1, y1, ...
    return np.stack((common, common, second), axis=-1).reshape(-1, 3)


def main():
    rng = np.random.default_rng(42)
    q_train = rng.uniform(-2.5, 2.5, (30, 2))
    q_holdout = rng.uniform(-2.5, 2.5, (12, 2))
    truth = np.array([0.07, -0.02, 0.04])
    measurements = position(q_train, truth)
    jacobian = parameter_jacobian(q_train, truth)
    epsilon = 1e-6
    finite_difference = np.column_stack([
        ((position(q_train, truth + epsilon * basis)
          - position(q_train, truth - epsilon * basis)) / (2 * epsilon)).ravel()
        for basis in np.eye(3)
    ])
    np.testing.assert_allclose(jacobian, finite_difference, atol=1e-9)
    assert np.linalg.matrix_rank(jacobian) == 2
    np.testing.assert_allclose(jacobian @ [1, -1, 0], 0, atol=1e-14)
    equivalent = truth + np.array([0.3, -0.3, 0.0])
    np.testing.assert_allclose(position(q_holdout, equivalent),
                               position(q_holdout, truth), atol=1e-14)

    # Fix beta=0 as a coordinate convention. The fitted first angle is beta+offset1,
    # not the original physical offset1. No held-out samples enter this loop.
    estimate = np.zeros(3)
    for _ in range(10):
        residual = (measurements - position(q_train, estimate)).ravel()
        reduced = parameter_jacobian(q_train, estimate)[:, 1:]
        step, _, rank, _ = np.linalg.lstsq(reduced, residual, rcond=None)
        assert rank == 2
        estimate[1:] += step
        if np.linalg.norm(step) < 1e-12:
            break
    np.testing.assert_allclose(estimate, [0, 0.05, 0.04], atol=1e-10)
    holdout_errors = np.linalg.norm(position(q_holdout, estimate)
                                    - position(q_holdout, truth), axis=1)
    assert max(holdout_errors) < 1e-10
    print(json.dumps({
        "seed": 42, "train_samples": len(q_train), "held_out_samples": len(q_holdout),
        "full_jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "full_singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist(),
        "fixed_gauge_singular_values": np.linalg.svd(jacobian[:, 1:], compute_uv=False).tolist(),
        "truth_beta_offset1_offset2_rad": truth.tolist(),
        "fitted_beta_offset1_offset2_rad": estimate.tolist(),
        "held_out_max_position_error_m": float(max(holdout_errors)),
        "finite_difference": "passed", "gauge_equivalence": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
