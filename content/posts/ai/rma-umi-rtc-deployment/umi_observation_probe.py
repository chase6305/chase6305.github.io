#!/usr/bin/env python3
"""Hash-pinned UMI observation alignment against synthetic recording stubs.

Requires NumPy/SciPy, imports only a pure interpolation module, and extracts
get_obs with ast. No robot/camera module, device or policy is started.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

COMMIT = 'd095ba9590df789df5189eea5ee7e431689038a6'
FILES = {
    'umi/real_world/bimanual_umi_env.py': 'a25d8d3ae5cea69cfacc7e3f334a96ff985ae621fc42e8a26a4b728dcfba4fb1',
    'umi/common/interpolation_util.py': 'cf53a3edb321a3c31db32a98cf13b74d6ae53b47dcca9cf26e297d2c547fa212',
}


def verified_sources(root):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f'Expected UMI commit {COMMIT}; found {revision}')
    result = {}
    for name, expected in FILES.items():
        path = root/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source hash mismatch: '+name)
        result[name] = path
    return result


def run(sources):
    import numpy as np
    import scipy
    from scipy.spatial.transform import Rotation

    spec = importlib.util.spec_from_file_location('verified_umi_interpolation', sources['umi/common/interpolation_util.py'])
    interpolation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interpolation)
    tree = ast.parse(sources['umi/real_world/bimanual_umi_env.py'].read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BimanualUmiEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_obs')
    namespace = {'np': np, 'math': math, 'PoseInterpolator': interpolation.PoseInterpolator,
                 'get_interp1d': interpolation.get_interp1d}
    unit = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(unit, str(sources['umi/real_world/bimanual_umi_env.py']), 'exec'), namespace)
    get_obs = namespace['get_obs']

    class Camera:
        def __init__(self, phases):
            self.calls, self.data = [], {}
            for index, phase in enumerate(phases):
                times = 99.6+np.arange(32)/60+phase
                # Pixel values encode the original frame number and camera.
                colors = np.array([np.full((1, 1, 3), index*100+i, dtype=np.uint16) for i in range(32)])
                self.data[index] = {'timestamp': times, 'color': colors}

        def get(self, k, out):
            self.calls.append({'k': k, 'out_was_none': out is None})
            return {i: {key: value[-k:] for key, value in data.items()} for i, data in self.data.items()}

    def pose_curve(times, index):
        t = np.asarray(times)-100
        return np.column_stack([t+index, 2*t-index, np.full_like(t, .1),
                                np.zeros_like(t), np.zeros_like(t), .3*t])

    def fixture(robots, extra_camera, phases, horizon=2, downsample=1, stale=False):
        camera = Camera(phases[:robots+extra_camera])
        devices, grippers = [], []
        for index in range(robots):
            times = np.linspace(98.5, 99. if stale else 101., 300)
            poses = pose_curve(times, index)
            data = {'robot_timestamp': times, 'ActualTCPPose': poses}
            grip = {'gripper_timestamp': times, 'gripper_position': .03+.001*(times-100)+index*.01}
            devices.append(SimpleNamespace(get_all_state=lambda data=data: data))
            grippers.append(SimpleNamespace(get_all_state=lambda data=grip: data))
        return SimpleNamespace(is_ready=True, frequency=20, camera=camera, robots=devices, grippers=grippers,
                               camera_obs_horizon=horizon, robot_obs_horizon=2, gripper_obs_horizon=2,
                               camera_down_sample_steps=downsample, robot_down_sample_steps=1,
                               gripper_down_sample_steps=1, last_camera_data=None, obs_accumulator=None)

    rng = np.random.default_rng(20261106)
    errors, frame_checks, cases, selected = [], 0, 0, []
    for robots in (1, 2):
        for extra in (0, 1):
            for horizon, downsample in ((2, 1), (2, 2), (3, 1)):
                for _ in range(10):
                    phases = rng.uniform(-.008, .008, size=robots+extra)
                    fake = fixture(robots, extra, phases, horizon, downsample)
                    observed = get_obs(fake)
                    k = math.ceil(horizon*downsample*3)+2
                    assert fake.camera.calls == [{'k': k, 'out_was_none': True}]
                    buffers = fake.last_camera_data
                    # Select the latest candidate whose sum of differences to
                    # strictly earlier frames in other observation cameras is
                    # smallest. This oracle scans the entire available buffer.
                    costs = []
                    for i in range(robots):
                        candidate = float(buffers[i]['timestamp'][-1])
                        cost = math.fsum(candidate-max(float(t) for t in buffers[j]['timestamp'] if t < candidate)
                                         for j in range(robots) if j != i)
                        costs.append(cost)
                    chosen = min(range(robots), key=lambda i: costs[i])
                    anchor = float(buffers[chosen]['timestamp'][-1])
                    queries = np.array([anchor-i*downsample/20 for i in reversed(range(horizon))])
                    np.testing.assert_allclose(observed['timestamp'], queries, atol=1e-13, rtol=0)
                    selected.append(chosen)
                    for i, data in buffers.items():
                        for position, query in enumerate(queries):
                            nearest = min(range(len(data['timestamp'])), key=lambda j: abs(float(data['timestamp'][j])-query))
                            assert np.array_equal(observed[f'camera{i}_rgb'][position], data['color'][nearest])
                            frame_checks += 1
                    low_dim_queries = np.array([anchor-.05, anchor])
                    for index in range(robots):
                        expected = pose_curve(low_dim_queries, index)
                        actual = np.concatenate([observed[f'robot{index}_eef_pos'], observed[f'robot{index}_eef_rot_axis_angle']], axis=1)
                        errors.append(float(np.max(np.abs(actual-expected))))
                        widths = .03+.001*(low_dim_queries-100)+index*.01
                        errors.append(float(np.max(np.abs(observed[f'robot{index}_gripper_width'][:, 0]-widths))))
                    cases += 1
    assert max(errors) < 1e-12

    # A rotation branch counterexample: interpolating +/-179 degree rotvecs
    # componentwise would pass through zero; shortest-arc Slerp passes pi.
    poses = np.array([[0., 0., 0., 0., 0., math.radians(179)],
                      [1., 2., 3., 0., 0., math.radians(-179)]])
    queries = np.array([-1., 0., .25, .5, .75, 1., 2.])
    curve = interpolation.PoseInterpolator(np.array([0., 1.]), poses)(queries)
    clipped = np.clip(queries, 0, 1)
    np.testing.assert_allclose(curve[:, :3], clipped[:, None]*[1., 2., 3.], atol=1e-14)
    rotation_errors = []
    for i, amount in enumerate(clipped):
        angle = math.radians(179+2*amount)
        expected = [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        rotation_errors.append(float(np.max(np.abs(Rotation.from_rotvec(curve[i, 3:]).as_matrix()-expected))))
    assert max(rotation_errors) < 1e-14

    stale = fixture(2, 1, [0., .003, -.002], stale=True)
    observed = get_obs(stale)
    assert float(observed['timestamp'][-1]) > 100
    # The source clamps low-dimensional interpolators to the available endpoint.
    for index in range(2):
        expected = pose_curve([99., 99.], index)
        np.testing.assert_allclose(observed[f'robot{index}_eef_pos'], expected[:, :3], atol=1e-14)
        np.testing.assert_allclose(observed[f'robot{index}_eef_rot_axis_angle'], expected[:, 3:], atol=1e-14)
        np.testing.assert_allclose(observed[f'robot{index}_gripper_width'], .029+index*.01, atol=1e-14)
    return {'commit': COMMIT, 'source_sha256': FILES, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'alignment': {'fixtures': cases, 'nearest_frame_checks': frame_checks,
                          'low_dimensional_comparisons': len(errors), 'maximum_absolute_error': max(errors),
                          'selected_camera_counts': {str(i): selected.count(i) for i in sorted(set(selected))}},
            'rotation': {'queries_including_clamped_endpoints': len(queries), 'maximum_matrix_error': max(rotation_errors),
                         'midpoint_abs_z_angle_degrees': abs(math.degrees(curve[3, 5]))},
            'stale_proprioception': {'camera_anchor': float(observed['timestamp'][-1]), 'last_robot_timestamp': 99.,
                                     'endpoint_clamping_observed': True},
            'scope': 'Synthetic timestamped buffers and verified pure interpolation; no device clocks, network, tracking or physical latency validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(verified_sources(args.source_root))
    data = json.dumps(result, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    print(data, end='')
