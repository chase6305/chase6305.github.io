"""Coordinate-chain checks, independent of any robot's DH parameters.

Requires NumPy. Run: python -B frame_chain_check.py
All lengths are metres; angles are radians; vectors are columns.
"""
import numpy as np


def transform_z(angle, translation):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


def inverse_pose(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def main():
    a_from_b = transform_z(np.pi / 2, [1, 0, 0])
    point_b = np.array([0.2, 0, 0, 1])
    point_a = a_from_b @ point_b
    np.testing.assert_allclose(point_a, [1, 0.2, 0, 1], atol=1e-12)
    np.testing.assert_allclose(inverse_pose(a_from_b) @ point_a, point_b,
                               atol=1e-12)

    # Construct a target from a known base-to-flange pose, then recover it.
    world_from_base = transform_z(0.4, [1, -0.2, 0.3])
    base_from_flange = transform_z(-0.7, [0.4, 0.1, 0.5])
    flange_from_tcp = transform_z(0.2, [0.08, 0, 0.12])
    world_from_tcp = world_from_base @ base_from_flange @ flange_from_tcp
    recovered = (inverse_pose(world_from_base) @ world_from_tcp
                 @ inverse_pose(flange_from_tcp))
    np.testing.assert_allclose(recovered, base_from_flange, atol=1e-12)
    # Independent inverse implementation checks the closed-form expression.
    for pose in (a_from_b, world_from_base, base_from_flange, flange_from_tcp):
        np.testing.assert_allclose(inverse_pose(pose), np.linalg.inv(pose),
                                   atol=1e-12)
    print("point in A [m]:", point_a[:3].tolist())
    print("base / flange / TCP chain: passed")


if __name__ == "__main__":
    main()
