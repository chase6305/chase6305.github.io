"""Bounded scalar-joint frame IK with an explicitly scaled SE(3) residual.

Python 3.10+, NumPy and pin (verified with Pinocchio 4.1.0).
No robot commands. Clipped steps may stagnate; this is not constrained global IK.
The linear log component is divided by position_scale, rotation by rotation_scale.
"""
from numbers import Integral

import numpy as np
import pinocchio as pin


def solve_ik(model, frame_name, target, q0, max_iter=500,
             position_tol=1e-4, rotation_tol=1e-4, damping=1e-3,
             position_scale=1.0, rotation_scale=1.0):
    if model.nv == 0 or any(j.nq != 1 or j.nv != 1 for j in model.joints[1:]):
        raise ValueError("Only fixed-base scalar joints are supported")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    settings = np.array([position_tol, rotation_tol, damping,
                         position_scale, rotation_scale], dtype=float)
    if not np.isfinite(settings).all() or np.any(settings <= 0):
        raise ValueError("Tolerances, damping and scales must be finite and positive")
    row_scale = 1.0 / np.array([position_scale] * 3 + [rotation_scale] * 3)
    if not np.isfinite(row_scale).all():
        raise ValueError("Residual scales are too small")
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise ValueError(f"Unknown end-effector frame: {frame_name}")

    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()
            and np.all(lower < upper)):
        raise ValueError("Finite, ordered joint bounds are required")
    q = np.asarray(q0, dtype=float).copy()
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError("q0 must be a finite vector of size model.nq")
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("q0 is outside joint limits")
    if not (np.isfinite(target.homogeneous).all()
            and np.allclose(target.rotation.T @ target.rotation,
                            np.eye(3), atol=1e-6, rtol=0)
            and np.isclose(np.linalg.det(target.rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("target must contain a valid rigid rotation")

    data = model.createData()

    def evaluate(configuration):
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        relative = current.actInv(target)
        error = pin.log6(relative).vector
        position_error = np.linalg.norm(current.translation - target.translation)
        rotation_error = np.linalg.norm(pin.log3(relative.rotation))
        return relative, error, position_error, rotation_error

    status = "iteration_limit"
    for iteration in range(max_iter):
        relative, error, pos_err, rot_err = evaluate(q)
        if pos_err <= position_tol and rot_err <= rotation_tol:
            status = "converged"
            break

        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL
        )
        error_jacobian = -pin.Jlog6(relative.inverse()) @ jacobian
        error_jacobian = row_scale[:, None] * error_jacobian
        weighted_error = row_scale * error
        delta = -error_jacobian.T @ np.linalg.solve(
            error_jacobian @ error_jacobian.T + damping**2 * np.eye(6),
            weighted_error,
        )
        if not np.isfinite(delta).all():
            status = "nonfinite_step"
            break

        cost = weighted_error @ weighted_error
        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = np.clip(pin.integrate(model, q, step * delta),
                                lower, upper)
            _, candidate_error, _, _ = evaluate(candidate)
            candidate_error = row_scale * candidate_error
            if candidate_error @ candidate_error < cost:
                q = candidate
                accepted = True
                break
        if not accepted:
            status = "stagnation"
            break

    _, _, pos_err, rot_err = evaluate(q)
    success = bool(pos_err <= position_tol and rot_err <= rotation_tol
                   and np.all(q >= lower) and np.all(q <= upper))
    return {
        "success": success,
        "status": "converged" if success else status,
        "q": q,
        "iterations": iteration + 1,
        "position_error": float(pos_err),
        "rotation_error": float(rot_err),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    parser.add_argument("frame")
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    frame_id = model.getFrameId(args.frame)
    if frame_id >= len(model.frames):
        raise ValueError(f"Unknown frame: {args.frame}")
    q0 = np.clip(pin.neutral(model),
                 model.lowerPositionLimit, model.upperPositionLimit)
    known_q = np.clip(q0 + 0.1,
                     model.lowerPositionLimit, model.upperPositionLimit)
    data = model.createData()
    pin.forwardKinematics(model, data, known_q)
    pin.updateFramePlacements(model, data)
    target = data.oMf[frame_id].copy()

    result = solve_ik(model, args.frame, target, q0)
    print(result)
    if not result["success"]:
        raise SystemExit("IK failed: inspect residuals and seed")
