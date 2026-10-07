"""Independent checks for frame_ik.py; download both files to one directory.

Python 3.10+, NumPy and Pinocchio. Uses the built-in six-axis manipulator.
Run: python -B ik_validation.py --output ik-validation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from frame_ik import solve_ik


def fk(model, name, q):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(name)].copy()


def assert_solution(model, name, target, result):
    assert result['success'] and result['status'] == 'converged', result
    actual = fk(model, name, result['q'])
    position_error = np.linalg.norm(actual.translation - target.translation)
    rotation_error = np.linalg.norm(pin.log3(actual.rotation.T @ target.rotation))
    assert position_error <= 1e-4 and rotation_error <= 1e-4
    assert np.all(result['q'] >= model.lowerPositionLimit)
    assert np.all(result['q'] <= model.upperPositionLimit)
    return float(position_error), float(rotation_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('ik-validation.json'))
    args = parser.parse_args()
    model = pin.buildSampleModelManipulator()
    model.lowerPositionLimit[:] = -np.pi
    model.upperPositionLimit[:] = np.pi
    name = model.frames[-1].name
    rng = np.random.default_rng(20260928)
    errors = []
    for _ in range(60):
        known = rng.uniform(-.8, .8, model.nq)
        seed = known + rng.uniform(-.08, .08, model.nq)
        target = fk(model, name, known)
        result = solve_ik(model, name, target, seed, position_scale=.1, rotation_scale=.5)
        errors.append(assert_solution(model, name, target, result))

    known = np.array([.3, -.5, .6, -.8, .4, .2])
    seed = known + np.array([.06, -.03, .04, -.05, .02, -.04])
    target = fk(model, name, known)
    metres = solve_ik(model, name, target, seed, position_scale=.1, rotation_scale=.5)
    # This rescales kinematic lengths only; no dynamics or inertia claim is made.
    millimetre_model = pin.Model(model)
    for transform in millimetre_model.jointPlacements:
        transform.translation *= 1000
    for frame in millimetre_model.frames:
        frame.placement.translation *= 1000
    target_mm = fk(millimetre_model, name, known)
    np.testing.assert_allclose(target_mm.translation, target.translation * 1000, atol=1e-10)
    millimetres = solve_ik(millimetre_model, name, target_mm, seed,
                           position_scale=100., rotation_scale=.5, position_tol=.1)
    assert metres['success'] and millimetres['success']
    np.testing.assert_allclose(metres['q'], millimetres['q'], atol=1e-10, rtol=1e-10)
    assert metres['iterations'] == millimetres['iterations']

    invalid = []
    for key, value in [('max_iter', 0), ('max_iter', 2.5), ('max_iter', True),
                       ('damping', float('nan')), ('position_tol', float('nan')),
                       ('rotation_tol', float('inf')), ('position_scale', 0.),
                       ('rotation_scale', -1.)]:
        try:
            solve_ik(model, name, target, seed, **{key: value})
        except ValueError:
            invalid.append(key + '=' + str(value))
        else:
            raise AssertionError('invalid option accepted: ' + key)
    for bad_name, bad_seed in [('missing_frame', seed), (name, np.full(model.nq, 100.))]:
        try:
            solve_ik(model, bad_name, target, bad_seed)
        except ValueError:
            invalid.append('missing frame' if bad_name != name else 'out-of-bounds seed')
        else:
            raise AssertionError('invalid frame or seed accepted')
    bad_rotation = pin.SE3(np.diag([-1., 1., 1.]), np.zeros(3))
    try:
        solve_ik(model, name, bad_rotation, seed)
    except ValueError:
        invalid.append('reflection instead of rotation')
    else:
        raise AssertionError('improper rotation accepted')
    broken_limits = pin.Model(model)
    broken_limits.upperPositionLimit[0] = broken_limits.lowerPositionLimit[0]
    try:
        solve_ik(broken_limits, name, target, seed)
    except ValueError:
        invalid.append('zero-width joint interval')
    else:
        raise AssertionError('invalid joint interval accepted')

    unreachable = solve_ik(model, name, pin.SE3(np.eye(3), np.array([100., 0., 0.])),
                             pin.neutral(model), max_iter=5)
    assert not unreachable['success'] and unreachable['status'] != 'converged'
    result = {'pinocchio': pin.__version__, 'numpy': np.__version__, 'seed': 20260928,
              'reachable_near_seed_cases': len(errors),
              'max_position_error_m': max(x[0] for x in errors),
              'max_rotation_error_rad': max(x[1] for x in errors),
              'unit_conversion': {'length_factor': 1000, 'position_scale_m': .1,
                  'position_scale_mm': 100., 'position_tol_m': 1e-4, 'position_tol_mm': .1,
                  'max_joint_difference_rad': float(abs(metres['q'] - millimetres['q']).max()),
                  'iterations_in_both_units': metres['iterations']},
              'rejected_inputs': invalid,
              'unreachable': {k: v for k, v in unreachable.items() if k != 'q'},
              'scope': 'Synthetic FK targets near known seeds; no global IK, collision, hardware or dynamics test.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
