#!/usr/bin/env python3
"""Check Git-object probe rejection order, output preservation and source purity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT/'content/posts/ai/rma-umi-rtc-deployment/umi_waypoint_probe.py'


def command(args, **kwargs):
    return subprocess.run([str(x) for x in args], text=True, capture_output=True,
                          timeout=60, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--numpy-python', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Keep the virtual-environment symlink rather than resolving to system Python.
    runtime = args.numpy_python.absolute()
    source = args.source_root.absolute()
    before = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'])
    source_digest = hashlib.sha256(PROBE.read_bytes()).hexdigest()
    results = {}
    with tempfile.TemporaryDirectory(prefix='chase-umi-waypoint-contracts-') as temporary:
        root = Path(temporary)
        wrong = root/'wrong-revision'
        wrong.mkdir()
        assert command(['git', 'init', '-q', wrong]).returncode == 0
        assert command(['git', '-C', wrong, '-c', 'user.name=Source Fixture',
                        '-c', 'user.email=source-fixture@example.invalid',
                        'commit', '-q', '--allow-empty', '-m', 'fixture']).returncode == 0
        rejected = command([runtime, '-I', '-S', '-B', PROBE, '--source-root', wrong])
        assert rejected.returncode != 0 and 'Expected d095ba9' in rejected.stderr
        assert 'ModuleNotFoundError' not in rejected.stderr
        results['wrong_revision_before_site_packages'] = True

        # Corrupt only the expected digest in a disposable probe copy. Git
        # objects and the user's source checkout remain untouched.
        altered = root/'altered_expected_hash.py'
        altered.write_text(PROBE.read_text().replace(
            '34662c3e495ef6a22f47055784b7a2c7dc37b11c8ecbd393244fb69778cbce23', '0'*64, 1))
        rejected = command([runtime, '-I', '-S', '-B', altered, '--source-root', source])
        assert rejected.returncode != 0 and 'Git-object source hash mismatch' in rejected.stderr
        assert 'ModuleNotFoundError' not in rejected.stderr
        results['wrong_expected_hash_before_site_packages'] = True

        existing = root/'existing.json'
        existing.write_bytes(b'preserve this earlier result\n')
        rejected = command([runtime, '-I', '-B', PROBE, '--source-root', source,
                            '--output', existing])
        assert rejected.returncode != 0 and 'FileExistsError' in rejected.stderr
        assert existing.read_bytes() == b'preserve this earlier result\n'
        results['existing_output_preserved'] = True

        fresh = root/'fresh.json'
        accepted = command([runtime, '-I', '-B', PROBE, '--source-root', source,
                            '--output', fresh])
        assert accepted.returncode == 0, accepted.stderr
        assert accepted.stdout == fresh.read_text()
        report = json.loads(accepted.stdout)
        assert report['controller_branch_cases'] == 21
        assert report['pure_interpolator']['independent_1d_samples'] == 19
        results['fresh_output_matches_stdout'] = True
    after = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'])
    assert before == after
    assert hashlib.sha256(PROBE.read_bytes()).hexdigest() == source_digest
    results.update(source_checkout_unchanged=True, published_probe_unchanged=True,
                   probe_sha256=source_digest, source_origin='immutable Git objects')
    text = json.dumps(results, indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
