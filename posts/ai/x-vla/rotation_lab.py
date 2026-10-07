"""Original CPU exercise: distinguish the two rotation layouts used in X-VLA.

Run: python rotation_lab.py
Dependency: numpy. This does not load a policy or command a robot.
"""

import numpy as np


def encode_rotation(matrix, layout):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Expected one 3x3 rotation matrix")
    if layout == "interleaved":
        return matrix[:, :2].reshape(6)
    if layout == "columns":
        return matrix[:, :2].T.reshape(6)
    raise ValueError("Unknown rotation layout")


def decode_rotation(values, layout):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Expected six finite values")
    if layout == "interleaved":
        first, second = values[::2], values[1::2]
    elif layout == "columns":
        first, second = values[:3], values[3:]
    else:
        raise ValueError("Unknown rotation layout")
    norm = np.linalg.norm(first)
    if norm < 1e-8:
        raise ValueError("First rotation vector is degenerate")
    first = first / norm
    second = second - np.dot(first, second) * first
    norm = np.linalg.norm(second)
    if norm < 1e-8:
        raise ValueError("Rotation vectors are collinear")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def rotation_error_degrees(expected, actual):
    cosine = (np.trace(expected.T @ actual) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def check_interpolation_and_targets():
    canonical = encode_rotation(np.eye(3), "interleaved")
    redundant = np.array([2, 1, 0, 3, 0, 0], dtype=float)
    mse = np.mean((canonical - redundant) ** 2)
    recovered = decode_rotation(redundant, "interleaved")
    np.testing.assert_allclose(mse, 1.0)
    np.testing.assert_allclose(recovered, np.eye(3))
    print(f"6D MSE={mse:.1f}, decoded angle error="
          f"{rotation_error_degrees(np.eye(3), recovered):.1f} degrees")

    # Exact 180-degree endpoint avoids a floating-point sine residue.
    half_turn = encode_rotation(np.diag([-1., -1., 1.]), "interleaved")
    midpoint = (canonical + half_turn) / 2
    np.testing.assert_array_equal(midpoint, np.zeros(6))
    try:
        decode_rotation(midpoint, "interleaved")
    except ValueError:
        pass
    else:
        raise AssertionError("Linear 6D midpoint should be degenerate")
    quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(quarter_turn.T @ quarter_turn, np.eye(3))
    np.testing.assert_allclose(np.linalg.det(quarter_turn), 1)
    print("180-degree boundary: linear 6D midpoint is degenerate; Rz(90) is valid")

    target = np.interp(.5, [0., 1.], [0., 1.])
    np.testing.assert_allclose(target, .5)
    # BCEWithLogits(z,g) = softplus(z) - g*z; verify its derivative.
    for logit in (-2., 0., 2.):
        eps = 1e-5
        upper = np.logaddexp(0, logit + eps) - target * (logit + eps)
        lower = np.logaddexp(0, logit - eps) - target * (logit - eps)
        gradient = 1 / (1 + np.exp(-logit)) - target
        np.testing.assert_allclose((upper - lower) / (2 * eps), gradient, atol=1e-10)
    print("gripper: interpolated target=0.5; BCE logit gradient verified as sigmoid(z)-g")


def check_relative_pose_frames():
    # Pose maps tool coordinates into the base frame, with column vectors.
    current_r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    local_dr = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    current_p = np.array([1., 2., 0.])
    target_p = np.array([1., 2.1, 0.])
    target_r = current_r @ local_dr
    base_dp = target_p - current_p
    local_dp = current_r.T @ base_dp
    base_dr = target_r @ current_r.T
    np.testing.assert_allclose(local_dp, [.1, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(current_p + current_r @ local_dp, target_p)
    np.testing.assert_allclose(base_dr @ current_r, target_r)
    assert not np.allclose(current_p + local_dp, target_p)
    assert not np.allclose(local_dr @ current_r, target_r)
    # Left multiplication of a complete SE(3) transform needs a different translation.
    current = np.eye(4)
    current[:3, :3], current[:3, 3] = current_r, current_p
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = target_r, target_p
    local_delta = np.linalg.inv(current) @ target
    base_delta = target @ np.linalg.inv(current)
    np.testing.assert_allclose(current @ local_delta, target, atol=1e-12)
    np.testing.assert_allclose(base_delta @ current, target, atol=1e-12)
    np.testing.assert_allclose(base_delta[:3, 3], target_p - base_dr @ current_p)
    assert not np.allclose(base_delta[:3, 3], base_dp)
    print("pose frames: base displacement [0, 0.1, 0] equals tool displacement [0.1, 0, 0]")
    print("SE(3): both composition orders reconstruct the target; naive translation does not")


def main():
    identity = np.eye(3)
    angle = np.deg2rad(37.0)
    c, s = np.cos(angle), np.sin(angle)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    angle_y = np.deg2rad(-23.0)
    c, s = np.cos(angle_y), np.sin(angle_y)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    angle_x = np.deg2rad(19.0)
    c, s = np.cos(angle_x), np.sin(angle_x)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # Include a general 3D rotation; identity alone cannot catch every mismatch.
    for layout in ("interleaved", "columns"):
        print(f"identity ({layout}): {encode_rotation(identity, layout).tolist()}")
        for matrix in (identity, rz, rz @ ry @ rx):
            recovered = decode_rotation(encode_rotation(matrix, layout), layout)
            np.testing.assert_allclose(recovered, matrix, atol=1e-12)
            np.testing.assert_allclose(recovered.T @ recovered, identity, atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(recovered), 1.0, atol=1e-12)
        print(f"round trips ({layout}): OK")

    matrix = rz @ ry @ rx
    wrong = decode_rotation(encode_rotation(matrix, "interleaved"), "columns")
    error = rotation_error_degrees(matrix, wrong)
    print(f"wrong-layout orientation error: {error:.2f} degrees")
    assert error > 1.0

    for invalid in (np.zeros(6), np.ones(6), np.full(6, np.nan)):
        try:
            decode_rotation(invalid, "interleaved")
        except ValueError:
            pass
        else:
            raise AssertionError("A degenerate input was accepted")
    print("degenerate inputs: rejected")
    check_interpolation_and_targets()
    check_relative_pose_frames()
    print(f"one prompt: {32 * 1024:,} parameters")
    print(f"30-domain prompt table: {30 * 32 * 1024:,} parameters")


if __name__ == "__main__":
    main()
