#!/usr/bin/env python3
"""Build/check the data article archive; --test runs extracted teaching code.

Run from any directory. The optional NumPy/external-repository probe is not run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'content/posts/ai/vla-world-model-data'
PAYLOAD = sorted([
    'README-lab.txt', 'data_contract_lab.py', 'lab-results.json',
    'make_figures.py', 'source_probe.py', 'source-probe-results.json',
    'source-map.json', 'assets/supervision-map.svg',
    'assets/scene-to-rollout.svg', 'assets/observation-time.svg',
])
PREFIX = 'data-contract-lab/'


def verify(bundle, *, check=False, test=False):
    payload = {name: (bundle / name).read_bytes() for name in PAYLOAD}
    sums = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                   for name, data in payload.items()).encode('utf-8')
    members = dict(sorted({**payload, 'SHA256SUMS': sums}.items()))
    output = bundle / 'data-contract-lab.zip'
    if check:
        if (bundle / 'SHA256SUMS').read_bytes() != sums:
            raise ValueError('SHA256SUMS is stale; rebuild the archive')
    else:
        (bundle / 'SHA256SUMS').write_bytes(sums)
        with zipfile.ZipFile(output, 'w') as archive:
            for name, data in members.items():
                entry = zipfile.ZipInfo(PREFIX + name, (2026, 9, 14, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        # Reject extra, duplicate and traversal entries before extraction.
        if archive.namelist() != [PREFIX + name for name in members]:
            raise ValueError('unexpected archive members')
        for name, data in members.items():
            # Compare decompressed content, independent of zlib version.
            if archive.read(PREFIX + name) != data:
                raise ValueError(f'stale archive member: {name}')
        checks = None
        if test:
            with tempfile.TemporaryDirectory(prefix='robot-data-lab-') as temporary:
                archive.extractall(temporary)
                extracted = Path(temporary) / PREFIX
                result = subprocess.run(
                    [sys.executable, '-B', 'data_contract_lab.py'], cwd=extracted,
                    check=True, capture_output=True, text=True, timeout=60,
                )
                actual = json.loads(result.stdout)
                if actual != json.loads((extracted / 'lab-results.json').read_text()):
                    raise ValueError('recorded lab results do not match extracted program')
                subprocess.run(
                    [sys.executable, '-B', 'make_figures.py'], cwd=extracted,
                    check=True, capture_output=True, timeout=60,
                )
                for name, data in members.items():
                    if (extracted / name).read_bytes() != data:
                        raise ValueError(f'reproduction changed {name}')
                checks = actual['checks_passed']
    return {'archive': output.name, 'members': len(members), 'lab_checks_passed': checks,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify without changing files')
    parser.add_argument('--test', action='store_true', help='run lab and reproduce SVGs after extraction')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(BUNDLE, check=args.check, test=args.test), indent=2))
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
