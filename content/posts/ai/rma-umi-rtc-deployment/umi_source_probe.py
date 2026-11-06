#!/usr/bin/env python3
"""Read-only probes of two hash-pinned UMI source interfaces; NumPy only.

No cameras, robot SDK, policy weights or device/network connections are used.
One pure module is imported; one verified method is extracted with ast and
executed against recording stubs, without importing the hardware module.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

COMMIT = 'd095ba9590df789df5189eea5ee7e431689038a6'
FILES = {
    'diffusion_policy/common/pose_repr_util.py':'6438d8746c8ee026a556d7df8a034ad54306746ee17b10973bb25f536894f348',
    'umi/real_world/bimanual_umi_env.py':'a25d8d3ae5cea69cfacc7e3f334a96ff985ae621fc42e8a26a4b728dcfba4fb1',
}


def verified_sources(root):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f'Expected UMI commit {COMMIT}; found {revision}')
    result = {}
    for name, expected in FILES.items():
        path = root/name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Source hash mismatch: {name}')
        result[name] = path
    return result


def pose_probe(source):
    import numpy as np
    spec = importlib.util.spec_from_file_location('verified_umi_pose', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(20261106)

    def make_pose():
        # Unit quaternion -> rotation, independently of the UMI helpers.
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        out = np.eye(4)
        out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        out[:3, 3] = rng.uniform(-2, 2, 3)
        return out

    errors = []
    for _ in range(200):
        base, target, common = make_pose(), make_pose(), make_pose()
        actual = module.convert_pose_mat_rep(target, base, pose_rep='relative')
        expected = np.eye(4)
        expected[:3, :3] = base[:3, :3].T @ target[:3, :3]
        expected[:3, 3] = base[:3, :3].T @ (target[:3, 3]-base[:3, 3])
        errors.append(float(np.max(np.abs(actual-expected))))
        restored = module.convert_pose_mat_rep(actual, base, pose_rep='relative', backward=True)
        errors.append(float(np.max(np.abs(restored-target))))
        changed = module.convert_pose_mat_rep(common@target, common@base, pose_rep='relative')
        errors.append(float(np.max(np.abs(changed-actual))))
    if max(errors) >= 2e-12:
        raise AssertionError(f'Relative pose disagreement: {max(errors)}')
    base = np.array([[0., -1., 0., 1.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    target = base.copy()
    target[0, 3] += 1
    relative = module.convert_pose_mat_rep(target, base, pose_rep='relative')
    legacy = module.convert_pose_mat_rep(target, base, pose_rep='rel')
    delta = module.convert_pose_mat_rep(target[None], base, pose_rep='delta')[0]
    if not np.allclose(relative[:3, 3], [0., -1., 0.]):
        raise AssertionError('Body-frame fixture failed')
    if not np.allclose(legacy[:3, 3], [1., 0., 0.]) or not np.allclose(delta[:3, 3], [1., 0., 0.]):
        raise AssertionError('Legacy / delta fixture changed')
    return {'random_cases':200, 'matrix_comparisons':len(errors), 'maximum_absolute_error':max(errors),
            'relative_position':relative[:3, 3].tolist(), 'legacy_rel_position':legacy[:3, 3].tolist(),
            'delta_position':delta[:3, 3].tolist(),
            'scope':'The current relative path agrees with rigid-frame composition; rel and componentwise delta have different semantics.'}


def action_probe(source):
    import numpy as np
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BimanualUmiEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'exec_actions')
    namespace = {'np':np, 'time':SimpleNamespace(time=lambda:4.)}
    unit = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(unit, str(source), 'exec'), namespace)
    execute = namespace['exec_actions']

    class Recorder:
        def __init__(self):
            self.calls = []

        def schedule_waypoint(self, **kwargs):
            self.calls.append(kwargs)

    times = [3.99, 4., 4.01, 4.5]
    actions = np.arange(len(times)*14, dtype=float).reshape(len(times), 14)/100
    rows, timing_modes = [], []
    for compensate in (False, True):
        robots, grippers = [Recorder(), Recorder()], [Recorder(), Recorder()]
        recorder = SimpleNamespace(put=lambda a, t: rows.append({'kept_times':t.tolist(), 'kept_action_rows':a.tolist()}))
        fake = SimpleNamespace(is_ready=True, robots=robots, grippers=grippers,
                               robots_config=[{'robot_action_latency':.1}, {'robot_action_latency':.2}],
                               grippers_config=[{'gripper_action_latency':.03}, {'gripper_action_latency':.04}],
                               action_accumulator=recorder)
        execute(fake, actions, times, compensate_latency=compensate)
        for i in range(2):
            if len(robots[i].calls) != 2 or len(grippers[i].calls) != 2:
                raise AssertionError('Strict deadline filter did not keep exactly two rows')
            for j, original in enumerate((2, 3)):
                r, g = robots[i].calls[j], grippers[i].calls[j]
                if not np.array_equal(r['pose'], actions[original, i*7:i*7+6]) or g['pos'] != actions[original, i*7+6]:
                    raise AssertionError('Bimanual action slicing changed')
                r_delay = fake.robots_config[i]['robot_action_latency'] if compensate else 0
                g_delay = fake.grippers_config[i]['gripper_action_latency'] if compensate else 0
                if abs(r['target_time']-(times[original]-r_delay)) > 1e-14 or abs(g['target_time']-(times[original]-g_delay)) > 1e-14:
                    raise AssertionError('Per-device scheduling offset changed')
        scheduled = [float(call['target_time']) for device in robots+grippers for call in device.calls]
        before_receive = sum(t < 4. for t in scheduled)
        if before_receive != (4 if compensate else 0):
            raise AssertionError('The synthetic lead-time boundary behavior changed')
        timing_modes.append({'compensate_latency':compensate, 'minimum_scheduled_time':min(scheduled),
                             'first_robot_first_scheduled_time':float(robots[0].calls[0]['target_time']),
                             'scheduled_before_receive_time':before_receive})
    if any(row['kept_times'] != [4.01, 4.5] for row in rows):
        raise AssertionError('Action recording retained outdated rows')
    return {'modes':2, 'robots_per_mode':2, 'waypoint_calls':16, 'strictly_kept_times':[4.01, 4.5],
            'equality_at_current_time_rejected':True,
            'timing_modes':timing_modes,
            'scope':'Only the verified method against recording stubs; no proof of lower-level handling of past scheduled times, full caller fallback or physical latency.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paths = verified_sources(args.source_root)
    import numpy as np
    result = {'commit':COMMIT, 'source_sha256':FILES, 'numpy':np.__version__,
              'pose':pose_probe(paths['diffusion_policy/common/pose_repr_util.py']),
              'scheduling':action_probe(paths['umi/real_world/bimanual_umi_env.py'])}
    data = json.dumps(result, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()
