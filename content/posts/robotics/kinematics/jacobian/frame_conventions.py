"""Check Pinocchio reference points, wrench duality and log-residual derivatives.

Python 3.10+, NumPy and pin (verified with Pinocchio 4.1.0).
Uses a built-in model and a fixed tool offset; no URDF, GUI or robot required.
Run: python -B frame_conventions.py --output frame-conventions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


def placement(model, frame_id, q):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('frame-conventions.json'))
    args = parser.parse_args()
    model = pin.buildSampleModelManipulator()
    flange_id = model.nframes - 1
    flange = model.frames[flange_id]
    offset = pin.SE3(pin.exp3(np.array([.2, -.1, .3])), np.array([.12, -.05, .18]))
    tool_id = model.addFrame(pin.Frame('offset_tool', flange.parentJoint, flange_id,
                                       flange.placement * offset, pin.FrameType.OP_FRAME))
    q = np.array([.35, -.45, .6, -.7, .4, .25])
    dq = np.array([.2, -.1, .15, .3, -.25, .12])
    data = model.createData()
    pose = placement(model, tool_id, q)
    rotation, position = pose.rotation, pose.translation
    matrices = {name: pin.computeFrameJacobian(model, data, q, tool_id, reference).copy()
                for name, reference in [('LOCAL', pin.LOCAL),
                                         ('LOCAL_WORLD_ALIGNED', pin.LOCAL_WORLD_ALIGNED),
                                         ('WORLD', pin.WORLD)]}
    local, aligned, world = [matrices[k] for k in ('LOCAL', 'LOCAL_WORLD_ALIGNED', 'WORLD')]
    pure_rotation = np.zeros((6, 6))
    pure_rotation[:3, :3] = pure_rotation[3:, 3:] = rotation
    np.testing.assert_allclose(aligned, pure_rotation @ local, atol=1e-13)
    np.testing.assert_allclose(world, pose.action @ local, atol=1e-13)
    expected_world = aligned.copy()
    expected_world[:3] += pin.skew(position) @ aligned[3:]
    np.testing.assert_allclose(world, expected_world, atol=1e-13)

    # Derive end-point velocity from independent FK samples, not Jacobian APIs.
    epsilon = 1e-6
    finite_difference = np.empty_like(aligned)
    for i in range(model.nv):
        step = np.eye(model.nv)[i] * epsilon
        plus = placement(model, tool_id, pin.integrate(model, q, step))
        minus = placement(model, tool_id, pin.integrate(model, q, -step))
        finite_difference[:3, i] = (plus.translation - minus.translation) / (2 * epsilon)
        finite_difference[3:, i] = pin.log3(plus.rotation @ minus.rotation.T) / (2 * epsilon)
    np.testing.assert_allclose(aligned, finite_difference, atol=1e-8, rtol=1e-8)

    flange_pose = placement(model, flange_id, q)
    flange_j = pin.computeFrameJacobian(model, data, q, flange_id,
                                        pin.LOCAL_WORLD_ALIGNED).copy()
    tool_from_flange_world = flange_pose.rotation @ offset.translation
    expected_tool = flange_j.copy()
    expected_tool[:3] -= pin.skew(tool_from_flange_world) @ flange_j[3:]
    np.testing.assert_allclose(aligned, expected_tool, atol=1e-13)

    wrench_at_tool = np.array([4., -3., 2., .2, -.1, .3])  # N, N m; world axes
    wrench_local = pure_rotation.T @ wrench_at_tool
    wrench_at_world_origin = wrench_at_tool.copy()
    wrench_at_world_origin[3:] += np.cross(position, wrench_at_tool[:3])
    torques = [local.T @ wrench_local, aligned.T @ wrench_at_tool,
               world.T @ wrench_at_world_origin]
    for torque in torques[1:]:
        np.testing.assert_allclose(torque, torques[0], atol=1e-12)
    wrench_at_flange = wrench_at_tool.copy()
    wrench_at_flange[3:] += np.cross(tool_from_flange_world, wrench_at_tool[:3])
    np.testing.assert_allclose(flange_j.T @ wrench_at_flange, torques[0], atol=1e-12)
    powers = [float(w @ (j @ dq)) for w, j in
              [(wrench_local, local), (wrench_at_tool, aligned),
               (wrench_at_world_origin, world)]]
    np.testing.assert_allclose(powers, [torques[0] @ dq] * 3, atol=1e-12)
    wrong_torque_error = np.linalg.norm(world.T @ wrench_at_tool - torques[0])
    assert wrong_torque_error > .1

    # Move only the world origin; the physical motion and wrench are unchanged.
    translated_position = position - np.array([.4, -.3, .2])
    translated_j = aligned.copy()
    translated_j[:3] += pin.skew(translated_position) @ aligned[3:]
    translated_wrench = wrench_at_tool.copy()
    translated_wrench[3:] += np.cross(translated_position, wrench_at_tool[:3])
    np.testing.assert_allclose(translated_j.T @ translated_wrench, torques[0], atol=1e-12)

    target = pose * pin.SE3(pin.exp3(np.array([1., -.8, .6])), np.array([.3, -.2, .1]))
    relative = pose.actInv(target)
    analytic_error_j = -pin.Jlog6(relative.inverse()) @ local
    numeric_error_j = np.empty_like(local)
    for i in range(model.nv):
        step = np.eye(model.nv)[i] * epsilon
        def residual(delta):
            current = placement(model, tool_id, pin.integrate(model, q, delta))
            return pin.log6(current.actInv(target)).vector
        numeric_error_j[:, i] = (residual(step) - residual(-step)) / (2 * epsilon)
    np.testing.assert_allclose(analytic_error_j, numeric_error_j, atol=1e-8, rtol=1e-8)
    result = {
        'pinocchio': pin.__version__, 'numpy': np.__version__, 'nq': model.nq, 'nv': model.nv,
        'q_rad': q.tolist(), 'qdot_rad_s': dq.tolist(), 'tool_offset_m': offset.translation.tolist(),
        'tool_position_world_m': position.tolist(),
        'twists_linear_then_angular': {key: (matrix @ dq).tolist() for key, matrix in matrices.items()},
        'aligned_jacobian_finite_difference_max_error': float(abs(aligned - finite_difference).max()),
        'world_linear_block_vs_position_derivative_max_difference': float(abs(world[:3] - finite_difference[:3]).max()),
        'tool_vs_flange_linear_jacobian_max_difference': float(abs(aligned[:3] - flange_j[:3]).max()),
        'joint_torque_Nm': torques[0].tolist(), 'power_W_in_three_conventions': powers,
        'wrong_unshifted_wrench_torque_error_Nm': float(wrong_torque_error),
        'log_residual_jacobian_max_error': float(abs(analytic_error_j - numeric_error_j).max()),
        'naive_negative_geometric_jacobian_max_error': float(abs(-local - numeric_error_j).max()),
        'checks': ['rotation versus adjoint', 'FK finite differences', 'fixed tool offset',
                   'wrench moment shift', 'joint power', 'translated world origin', 'Jlog6 derivative'],
        'scope': 'Fixed synthetic configuration, no robot data; log derivative tested away from the pi branch cut.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
