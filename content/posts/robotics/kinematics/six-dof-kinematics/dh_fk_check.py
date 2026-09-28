"""Check the article's six-axis standard-DH model, not a calibrated robot.

Run: python -B dh_fk_check.py
Dependency: NumPy. Lengths are metres, angles are radians.
"""
import json
import numpy as np

A = np.array([0.0, 0.455, 0.0, 0.0, 0.0, 0.0])
D = np.array([0.220, 0.0, 0.0, 0.495, 0.0, -0.155])
ALPHA = np.array([np.pi / 2, 0, np.pi / 2, np.pi / 2, np.pi / 2, 0])


def standard_dh(theta, d, a, alpha):
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s * ca, s * sa, a * c],
                     [s, c * ca, -c * sa, a * s],
                     [0, sa, ca, d], [0, 0, 0, 1]], dtype=float)


def fk(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Expected six finite joint angles in radians")
    transform = np.eye(4)
    origins, axes = [], []
    for i in range(6):
        origins.append(transform[:3, 3].copy())
        axes.append(transform[:3, 2].copy())
        transform = transform @ standard_dh(q[i], D[i], A[i], ALPHA[i])
    return transform, np.asarray(origins), np.asarray(axes)


def check():
    zero, _, _ = fk(np.zeros(6))
    np.testing.assert_allclose(zero[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(zero[:3, 3], [0.455, 0.0, -0.430], atol=1e-14)
    rng = np.random.default_rng(42)
    step, maximum_error = 1e-6, 0.0
    for q in rng.uniform(-np.pi, np.pi, size=(100, 6)):
        transform, origins, axes = fk(q)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-13)
        assert abs(np.linalg.det(rotation) - 1) < 1e-13
        # Geometric point-velocity Jacobian: z_(i-1) cross (p_end-o_(i-1)).
        analytic = np.cross(axes, transform[:3, 3] - origins).T
        numeric = np.column_stack([
            (fk(q + step * e)[0][:3, 3] - fk(q - step * e)[0][:3, 3])
            / (2 * step) for e in np.eye(6)
        ])
        maximum_error = max(maximum_error, float(abs(analytic - numeric).max()))
        np.testing.assert_allclose(analytic, numeric, atol=2e-9, rtol=1e-7)
        # The wrist-centre point lies at frame 4's origin in this DH geometry.
        wrist = transform[:3, 3] - D[-1] * rotation[:, 2]
        prefix = np.eye(4)
        for i in range(4):
            prefix = prefix @ standard_dh(q[i], D[i], A[i], ALPHA[i])
        np.testing.assert_allclose(wrist, prefix[:3, 3], atol=1e-13)
    return {"zero_position_m": zero[:3, 3].tolist(),
            "random_poses": 100, "maximum_jacobian_error": maximum_error,
            "wrist_centre_identity": "passed", "calibrated_robot": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
