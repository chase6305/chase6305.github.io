#!/usr/bin/env python3
"""Replay pinned UMI interpolation branches with numerical recording state only.

Read source from immutable Git objects. Import the pure NumPy/SciPy trajectory
module; extract only the scheduling branch from each hardware controller.
Never import a controller, robot SDK, IPC queue or device module.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

COMMIT = 'd095ba9590df789df5189eea5ee7e431689038a6'
FILES = {
    'umi/common/pose_trajectory_interpolator.py': '34662c3e495ef6a22f47055784b7a2c7dc37b11c8ecbd393244fb69778cbce23',
    'umi/real_world/franka_interpolation_controller.py': 'a3e058338782796dc2f1d30ac04c58d8553767fa7b18b297250eeb1a33f9f24b',
    'umi/real_world/rtde_interpolation_controller.py': '6a4bb56ea8ce660eb00afac29b5e5f5d5caf29c7f038a1f5afeb65dc5b32e5b3',
    'umi/real_world/wsg_controller.py': '627fe6867ddb2d012c4a6a408fa1ba438a54e6903b8802dbda4e06cfb56cb6fd',
}


def sources(root):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f'Expected {COMMIT}; found {revision}')
    result = {}
    for name, digest in FILES.items():
        raw = subprocess.check_output(['git', '-C', str(root), 'show', COMMIT+':'+name])
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError('Git-object source hash mismatch: '+name)
        result[name] = raw.decode('utf-8')
    return result


def run(verified):
    import numpy as np
    import scipy

    # This file contains only NumPy/SciPy interpolation and Python utilities.
    name = 'umi/common/pose_trajectory_interpolator.py'
    namespace = {'__name__': 'verified_umi_trajectory'}
    exec(compile(verified[name], name, 'exec'), namespace)
    interpolator = namespace['PoseTrajectoryInterpolator']

    def initial(offset=0):
        return interpolator([4.+offset, 4.5+offset],
                            [[0., 0., 0., 0., 0., 0.], [.5, 0., 0., 0., 0., 0.]])

    # Independent 1D reference: constant unit-speed old segment followed by a
    # straight new segment. Position speed, not a robot dynamics model.
    def expected_end(requested, current, last, maximum_speed):
        if requested <= current:
            return 4.5
        retained_end = current if requested <= last else max(last, current)
        retained_position = min(.5, max(0., retained_end-4.))
        minimum_duration = abs(1.-retained_position)/maximum_speed
        return retained_end+max(requested-retained_end, minimum_duration)

    pure = initial()
    target = np.array([1., 0., 0., 0., 0., 0.])
    for requested in (3.91, 4.01):
        assert pure.schedule_waypoint(target, requested, curr_time=4.01,
                                      last_waypoint_time=4.5) is pure
    retimed = pure.schedule_waypoint(target, 4.3, max_pos_speed=.2,
                                     curr_time=4.01, last_waypoint_time=4.5)
    assert math.isclose(retimed.times[-1], 8.96, abs_tol=2e-14)
    np.testing.assert_allclose(retimed(4.01), pure(4.01), atol=1e-14)
    np.testing.assert_allclose(retimed(8.96), target, atol=1e-14)
    for t in np.linspace(4.01, 8.96, 19):
        expected_x = .01+.2*(t-4.01)
        assert math.isclose(float(retimed(t)[0]), expected_x, abs_tol=3e-14)

    rows = []
    for controller in ('franka', 'rtde', 'wsg'):
        filename = f'umi/real_world/{controller}_'+('controller.py' if controller == 'wsg' else 'interpolation_controller.py')
        tree = ast.parse(verified[filename])
        branches = [node for node in ast.walk(tree)
                    if isinstance(node, ast.If) and
                    ast.unparse(node.test) == 'cmd == Command.SCHEDULE_WAYPOINT.value']
        assert len(branches) == 1, filename
        unit = ast.fix_missing_locations(ast.Module(body=branches[0].body, type_ignores=[]))
        compiled = compile(unit, filename+' scheduling branch', 'exec')
        for offset in (0., 100.):
            # Offset changes the monotonic clock's origin while preserving the
            # wall-clock command times. Equality is tested at offset zero.
            current = 4. if controller == 'wsg' else 4.01
            requests = [3.91, 4.3, 4.8]+([current] if offset == 0 else [])
            for requested in requests:
                before = initial(offset)
                command = {'target_time': requested, 'target_pose': target, 'target_pos': 1.}
                fake = SimpleNamespace(max_pos_speed=.2, max_rot_speed=.4,
                                       move_max_speed=.4, scale=1., verbose=False)
                state = {'self': fake, 'command': command, 'pose_interp': before,
                         'last_waypoint_time': 4.5+offset, 't_now': 4.+offset, 'dt': .01,
                         'time': SimpleNamespace(monotonic=lambda offset=offset: 4.+offset,
                                                 time=lambda: 4.)}
                exec(compiled, state)
                after = state['pose_interp']
                maximum = math.inf if controller == 'franka' else (.2 if controller == 'rtde' else .4)
                expected = expected_end(requested, current, 4.5, maximum)+offset
                assert math.isclose(float(after.times[-1]), expected, abs_tol=5e-13), (controller, requested, offset)
                ignored = requested <= current
                assert (after is before) == ignored
                # The outer controller records the requested time even when
                # interpolation is unchanged or a finite speed retimes arrival.
                assert math.isclose(state['last_waypoint_time'], requested+offset, abs_tol=1e-13)
                np.testing.assert_allclose(after(current+offset), before(current+offset), atol=1e-13)
                if not ignored:
                    np.testing.assert_allclose(after(after.times[-1]), target, atol=1e-13)
                rows.append({'controller': controller, 'clock_offset': offset,
                             'requested_wall_time': requested, 'current_wall_time_equivalent': current,
                             'interpolator_same_object': after is before,
                             'actual_interpolator_end_wall_equivalent': float(after.times[-1])-offset,
                             'recorded_last_requested_wall_equivalent': state['last_waypoint_time']-offset,
                             'finite_speed_argument': None if math.isinf(maximum) else maximum})
    return {'commit': COMMIT, 'source_sha256': FILES, 'source_origin': 'Verified immutable Git objects',
            'numpy': np.__version__, 'scipy': scipy.__version__, 'controller_branch_cases': len(rows),
            'pure_interpolator': {'past_and_equal_return_original': True,
                                  'requested_time': 4.3, 'current_time': 4.01,
                                  'position_speed': .2, 'realized_end': float(retimed.times[-1]),
                                  'independent_1d_samples': 19},
            'cases': rows,
            'scope': 'Extracted branch replay and pure interpolation only. No controller imports, queues, device connection, servo loop, joint dynamics, clock drift or hardware timing validation. WSG branch uses constructed scale=1 position units.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(sources(args.source_root))
    text = json.dumps(result, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
