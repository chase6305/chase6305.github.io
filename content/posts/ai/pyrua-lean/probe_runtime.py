#!/usr/bin/env python3
"""Probe a pinned PyRUA-Lean runtime using a recording object, without a robot.

Requires only Python >=3.10 and a local official checkout at COMMIT. All seven
imported source modules are SHA-256 verified BEFORE any of them is executed.
No simulator, model, network connection, camera, or hardware module is loaded.
Output is printed; optional --output uses exclusive creation, never overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

COMMIT = 'e754cf43c822b91f4728777a78071fa4e012c371'
HASHES = {
    '_backend': 'd3acdfd846dd0469d025fe29c7c74a1c88057201e1e2a9803e6d9740feaf22d4',
    '_robot': 'f4b9da3a4150bed9a3c2c709b341b2188c9b76fb9e455dafbe2b5cbbc058d170',
    'robots': '61e719aaa27abf6e3b9d6b015dfeed962cd56fe3b3c48a3ce69b3c46b84b26fa',
    'libero': 'b63ada08c1a098cbf71d4d481ba2e7186288b94fc4945c8efb416869dd052d96',
    'runner': 'd5cbb10dedd60cbd9dee7ac945f5e302bda7584064d76e1bc3a29ecc41ce123b',
    'sandbox': '22f679b8e2dbea75b1a0bccef59808e1aadeb25a03eb6ec9b3e72ff67a3aad66',
    'arms': '30e5cac55389bb49b8b81c695d315301fc46b9beb0ece69604586d5d656d1315',
}


class NoThirdParty(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'PIL', 'scipy', 'torch', 'mujoco', 'rpent'):
            raise ImportError('Third-party package deliberately disabled by the probe')
        return None


def probe(source):
    source = Path(source).resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError(f'Expected official source commit {COMMIT}; got {head}')
    package_dir = source/'src/pyrualean'
    for name, expected in HASHES.items():
        actual = hashlib.sha256((package_dir/(name+'.py')).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source SHA-256 mismatch: {name}.py')
    # Do not execute package __init__ or discover/install any dependencies.
    # Relative imports can only reach the verified module set used below.
    package_name = '_article_pyrua_probe'
    if package_name in sys.modules:
        raise RuntimeError('Run this probe in a fresh Python process')
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    sys.modules[package_name] = package
    blocker = NoThirdParty()
    sys.meta_path.insert(0, blocker)
    sys.dont_write_bytecode = True
    try:
        arms = importlib.import_module(package_name+'.arms')
        base = importlib.import_module(package_name+'._robot')

        class RecordingBackend:
            steps = 0
            cancelled = False
            def env_steps(self): return self.steps
            def solved(self): return False
            def cancel(self): self.cancelled = True

        class RecordingRobot(base.RobotBase):
            CAMERAS = ('view',)
            IMAGE_ARTIFACTS = {('view', 'high'): 'view.png'}
            ON_MOTION_IMAGES = ('view.png',)

            def bump(self):
                self._guard()
                record = self._ledger.open('bump', {}, stateful=True)
                self._backend.steps += 1
                record.ok = True
                record.env_steps = 1
                return self._backend.steps

            def artifact(self, name, *, step=-1):
                # Opaque synthetic byte markers, not photographs or depth.
                return f'frame:{self._backend.steps}'.encode()

            def _summary_dict(self):
                return {'recording_step': self._backend.steps}

        def toolkit(path, **options):
            robot = RecordingRobot(RecordingBackend())
            return robot, arms.ArmToolkit(robot, arm='cells', workspace=path, **options)

        def cell(tk, code):
            return tk.execute_tool('python', {'code': code})

        with tempfile.TemporaryDirectory(prefix='pyrua-runtime-probe-') as temporary:
            root = Path(temporary)
            robot, tk = toolkit(root/'pure', images='on-demand', feedback='pure')
            failed = cell(tk, 'saved = 7\nrobo.bump()\nprint("before error")\n1/0')
            assert 'ZeroDivisionError' in failed.result['exception']
            assert failed.result['stdout'] == 'before error\n'
            assert robot._backend.steps == 1 and not failed.images
            resumed = cell(tk, 'saved + 1')
            assert resumed.result['value'] == '8'
            assert 'state' not in resumed.result and 'primitive_calls' not in resumed.result
            # The official show() records a camera request. Attachment reads
            # its latest artifact at cell end, after the intervening bump.
            requested = cell(tk, 'robo.show("view")\nrobo.bump()')
            assert requested.images == [b'frame:2']
            saved_frame = cell(tk, 'frame = robo.artifact("view.png")\nrobo.bump()\ndisplay(frame)')
            assert saved_frame.images == [b'frame:2']
            assert robot._backend.steps == 3
            assert not cell(tk, '0').images  # no request leaks into the next cell

            budget_robot, budget = toolkit(root/'budget', max_turns=2, images='none', feedback='pure')
            cell(budget, 'for _ in range(5):\n    robo.bump()')
            cell(budget, 'print("one remaining cell")')
            refused = cell(budget, 'robo.bump()')
            assert 'budget exhausted' in refused.result['error']
            assert budget_robot._backend.steps == 5 and budget_robot._backend.cancelled

            _, defaults = toolkit(root/'defaults')
            rich = cell(defaults, 'robo.bump()')
            assert 'state' in rich.result and 'primitive_calls' in rich.result
            assert len(rich.images) == 1
            # This is a host-side test setup, not a proposed policy operation.
            robot._done = True
            stopped = cell(tk, 'robo.bump()')
            assert stopped.result['episode_finished'] == 'terminated'
            assert robot._backend.steps == 3
            loaded = sorted(name.removeprefix(package_name+'.') for name in sys.modules
                            if name.startswith(package_name+'.'))
            assert set(loaded) == set(HASHES)
            assert not any(name == 'numpy' or name.startswith('numpy.') for name in sys.modules)
        return {
            'commit': head, 'verified_modules': loaded,
            'persistent_assignment_after_exception': True,
            'side_effects_are_not_rolled_back': True,
            'show_then_move_attaches_end_of_cell_frame': True,
            'materialized_display_preserves_earlier_bytes': True,
            'cell_budget_differs_from_primitive_count': True,
            'defaults_are_rich_and_on_motion': True,
            'simulator_model_hardware_or_network_used': False,
        }
    finally:
        sys.meta_path.remove(blocker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('--output already exists; refusing to overwrite')
    try:
        result = probe(args.source)
        text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(text)
        print(text, end='')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()
