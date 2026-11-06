#!/usr/bin/env python3
"""Negative controls for the downloadable UMI interface probe.

Only disposable fixtures are modified. The supplied checkout is read-only.
Use --numpy-python for the separate NumPy-only interpreter.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PROBES = {
    'action': ROOT/'content/posts/ai/rma-umi-rtc-deployment/umi_source_probe.py',
    'observation': ROOT/'content/posts/ai/rma-umi-rtc-deployment/umi_observation_probe.py',
}


def command(*args):
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def check(source, numpy_python, kind):
    probe_path = PROBES[kind]
    spec = importlib.util.spec_from_file_location('umi_interface_probe', probe_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    class BlockThirdParty(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('numpy', 'scipy', 'torch'):
                raise ImportError('Third-party imports blocked during source verification')
    blocker = BlockThirdParty()
    sys.meta_path.insert(0, blocker)
    report = {'kind': kind, 'checks': [], 'scope': 'Source identity, verification order and exclusive output; no hardware.'}
    before = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in probe.FILES}
    try:
        verified = probe.verified_sources(source)
        assert set(verified) == set(probe.FILES)
        report['checks'].append('Source verification works before any NumPy/SciPy/PyTorch import.')
        git_dir = subprocess.check_output(['git', '-C', str(source), 'rev-parse', '--absolute-git-dir'], text=True).strip()
        with tempfile.TemporaryDirectory(prefix='chase-umi-negative-') as tmp:
            directory = Path(tmp)
            tampered = directory/'tampered'
            tampered.mkdir()
            (tampered/'.git').write_text('gitdir: '+git_dir+'\n')
            for name in probe.FILES:
                target = tampered/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source/name).read_bytes())
            first = next(iter(probe.FILES))
            (tampered/first).write_bytes((tampered/first).read_bytes()+b'\nraise RuntimeError("TAMPERED SOURCE WAS EXECUTED")\n')
            try:
                probe.verified_sources(tampered)
            except RuntimeError as error:
                assert str(error) == 'Source hash mismatch: '+first
            else:
                raise AssertionError('Modified source was accepted')
            report['checks'].append('A changed source file is rejected before its top-level code can execute.')

            wrong = directory/'wrong-revision'
            wrong.mkdir()
            initialized = command('git', '-C', str(wrong), 'init', '--quiet')
            assert initialized.returncode == 0, initialized.stderr
            committed = command('git', '-C', str(wrong), '-c', 'user.name=Probe fixture', '-c',
                                'user.email=probe@example.invalid', '-c', 'commit.gpgsign=false',
                                'commit', '--allow-empty', '--quiet', '-m', 'Disposable wrong revision')
            assert committed.returncode == 0, committed.stderr
            try:
                probe.verified_sources(wrong)
            except RuntimeError as error:
                assert str(error).startswith('Expected UMI commit '+probe.COMMIT+'; found ')
            else:
                raise AssertionError('Wrong revision was accepted')
            report['checks'].append('A different revision is rejected before missing or modified source files are inspected.')

            output = directory/'preserved.json'
            sentinel = b'Preserve this existing reader record.\n'
            output.write_bytes(sentinel)
            run = command(str(numpy_python), '-I', '-B', str(probe_path), '--source-root', str(source), '--output', str(output))
            assert run.returncode != 0 and 'FileExistsError' in run.stderr, run.stderr
            assert output.read_bytes() == sentinel
            report['checks'].append('An existing output file is preserved and the probe fails explicitly.')
            fresh = directory/'fresh.json'
            run = command(str(numpy_python), '-I', '-B', str(probe_path), '--source-root', str(source), '--output', str(fresh))
            assert run.returncode == 0, run.stderr
            saved = json.loads(fresh.read_text())
            assert saved == json.loads(run.stdout)
            assert saved['commit'] == probe.COMMIT
            if kind == 'action':
                assert saved['scheduling']['waypoint_calls'] == 16
            else:
                assert saved['alignment']['fixtures'] == 120
            report['checks'].append('A fresh exclusive output agrees with stdout and contains the pinned source identity.')
            report['successful_probe'] = {'commit': saved['commit'], 'numpy': saved['numpy']}
            if kind == 'action':
                report['successful_probe'].update(pose_matrix_comparisons=saved['pose']['matrix_comparisons'],
                                                  waypoint_calls=saved['scheduling']['waypoint_calls'])
            else:
                report['successful_probe'].update(scipy=saved['scipy'], alignment_fixtures=saved['alignment']['fixtures'],
                                                  frame_checks=saved['alignment']['nearest_frame_checks'])
    finally:
        sys.meta_path.remove(blocker)
    after = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in probe.FILES}
    assert before == after
    report['source_unchanged'] = True
    report['source_sha256'] = after
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--numpy-python', type=Path, required=True)
    parser.add_argument('--kind', choices=sorted(PROBES), default='action')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Resolving the venv interpreter symlink would bypass pyvenv.cfg and
    # silently run the base interpreter without the isolated NumPy install.
    result = check(args.source_root.resolve(), args.numpy_python.absolute(), args.kind)
    data = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.write_text(data)
    print(data, end='')
