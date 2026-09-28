"""Planar 3R zero offsets: an unknown contact point leaves a rotation gauge.

Python 3.10+, NumPy. Units: metres and radians; noiseless synthetic observations.
This checks identifiability, not measured calibration precision.
"""
import json

import numpy as np


LENGTHS = np.array([0.8, 0.6, 0.4])


def fk_jacobian(q):
    angles = np.cumsum(q, axis=-1)
    segments = np.stack((np.cos(angles), np.sin(angles)), axis=-1) * LENGTHS[None, :, None]
    position = segments.sum(axis=1)
    tangents = np.stack((-segments[:, :, 1], segments[:, :, 0]), axis=-1)
    jacobian = np.flip(np.cumsum(np.flip(tangents, axis=1), axis=1), axis=1)
    return position, np.swapaxes(jacobian, 1, 2)


def touching_poses(point, first_angles):
    # Choose q1, then solve the remaining 2R arm to touch the same point.
    wrist = point - LENGTHS[0] * np.column_stack((np.cos(first_angles), np.sin(first_angles)))
    cosine = ((wrist * wrist).sum(axis=1) - LENGTHS[1]**2 - LENGTHS[2]**2) / (
        2 * LENGTHS[1] * LENGTHS[2])
    if np.any(abs(cosine) >= 1):
        raise ValueError("fixture reaches a singular or unreachable 2R configuration")
    q3 = np.arccos(cosine)
    q2 = (np.arctan2(wrist[:, 1], wrist[:, 0]) - first_angles
          - np.arctan2(LENGTHS[2] * np.sin(q3), LENGTHS[1] + LENGTHS[2] * np.cos(q3)))
    q = np.column_stack((first_angles, q2, q3))
    np.testing.assert_allclose(fk_jacobian(q)[0], np.tile(point, (len(q), 1)), atol=1e-14)
    return q


def fit_known(encoder, point):
    offsets = np.zeros(3)
    for _ in range(30):
        position, jacobian = fk_jacobian(encoder + offsets)
        step, _, rank, _ = np.linalg.lstsq(jacobian.reshape(-1, 3), (point - position).ravel(), rcond=None)
        assert rank == 3
        offsets += step
        if np.linalg.norm(step) < 1e-12:
            return offsets
    raise RuntimeError("known-point fit did not converge")


def fit_unknown_fixed_first(encoder):
    offsets = np.zeros(3)
    point = fk_jacobian(encoder)[0].mean(axis=0)
    for _ in range(30):
        position, jacobian = fk_jacobian(encoder + offsets)
        matrix = np.concatenate((jacobian[:, :, 1:],
                                 np.tile(-np.eye(2), (len(encoder), 1, 1))), axis=2).reshape(-1, 4)
        step, _, rank, _ = np.linalg.lstsq(matrix, (point - position).ravel(), rcond=None)
        assert rank == 4
        offsets[1:] += step[:2]
        point += step[2:]
        if np.linalg.norm(step) < 1e-12:
            return offsets, point
    raise RuntimeError("fixed-gauge fit did not converge")


def main():
    point = np.array([1.25, .35])
    truth = np.array([.02, -.03, .025])
    physical = touching_poses(point, np.linspace(-.3, .8, 40))
    encoder = physical - truth
    position, jacobian = fk_jacobian(physical)
    known = jacobian.reshape(-1, 3)
    joint = np.concatenate((jacobian, np.tile(-np.eye(2), (len(encoder), 1, 1))), axis=2).reshape(-1, 5)
    difference = (jacobian[1:] - jacobian[:1]).reshape(-1, 3)
    assert np.linalg.matrix_rank(known) == 3
    assert np.linalg.matrix_rank(joint) == 4
    assert np.linalg.matrix_rank(difference) == 2
    null_direction = [1., 0., 0., -point[1], point[0]]
    np.testing.assert_allclose(joint @ null_direction, 0, atol=1e-14)
    np.testing.assert_allclose(difference[:, 0], 0, atol=1e-14)

    # A finite rotation gives exactly the same zero contact residual, not merely
    # a small first-order residual. The unknown point rotates with the arm.
    alpha = .4
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)],
                         [np.sin(alpha), np.cos(alpha)]])
    rotated = fk_jacobian(encoder + truth + [alpha, 0, 0])[0]
    np.testing.assert_allclose(rotated, np.tile(rotation @ point, (len(encoder), 1)), atol=1e-14)

    fitted_known = fit_known(encoder, point)
    fitted_unknown, fitted_point = fit_unknown_fixed_first(encoder)
    np.testing.assert_allclose(fitted_known, truth, atol=1e-10)
    np.testing.assert_allclose(fitted_unknown, [0, truth[1], truth[2]], atol=1e-10)
    held = touching_poses(point, np.linspace(-.27, .77, 11)) - truth
    predicted = fk_jacobian(held + fitted_unknown)[0]
    internal_error = float(np.linalg.norm(predicted - fitted_point, axis=1).max())
    external_error = float(np.linalg.norm(predicted - point, axis=1).max())
    assert internal_error < 1e-10
    np.testing.assert_allclose(external_error, 2 * np.linalg.norm(point) * np.sin(abs(truth[0]) / 2), atol=1e-10)
    print(json.dumps({
        "training_poses": 40, "held_out_poses": 11, "units": "m, rad",
        "known_point_rank": int(np.linalg.matrix_rank(known)),
        "unknown_point_joint_rank": int(np.linalg.matrix_rank(joint)),
        "unknown_point_difference_rank": int(np.linalg.matrix_rank(difference)),
        "joint_singular_values_unscaled": np.linalg.svd(joint, compute_uv=False).tolist(),
        "true_offsets_rad": truth.tolist(), "known_point_fit_rad": fitted_known.tolist(),
        "fixed_gauge_fit_rad": fitted_unknown.tolist(), "fitted_unknown_point_m": fitted_point.tolist(),
        "held_out_contact_residual_m": internal_error,
        "held_out_error_to_external_point_m": external_error,
        "finite_rotation_equivalence": "passed", "null_direction": null_direction,
        "note": "No noise; mixed parameter units; singular values only diagnose exact rank here."
    }, indent=2))


if __name__ == "__main__":
    main()
