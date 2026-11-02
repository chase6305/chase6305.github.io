"""Check the served Pion ZIP with the documented commands in a spaced path."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('python')
    parser.add_argument('output', type=Path)
    parser.add_argument('--stdlib', action='store_true', help='Run the dependency-free default checks')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root/'content/posts/ai/pion-muon-vla-rlvr/pion-spectral-lab.zip'
    url = args.base.rstrip('/')+'/posts/ai/pion-muon-vla-rlvr/pion-spectral-lab.zip'
    with urllib.request.urlopen(url, timeout=10) as response:
        data = response.read()
        headers = dict(response.headers)
    if data != source.read_bytes():
        raise RuntimeError('Served ZIP differs from the current source archive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    references = {}
    with tempfile.TemporaryDirectory(prefix='pion 读者 ', dir=args.output.parent) as temporary:
        folder = Path(temporary)/'中文 空格目录'
        folder.mkdir()
        archive = folder/'pion-spectral-lab.zip'
        archive.write_bytes(data)
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if len(names) != 32 or any(Path(name).is_absolute() or '..' in Path(name).parts
                                      or name.endswith('.pyc') for name in names):
                raise RuntimeError('Unexpected archive inventory')
            handle.extractall(folder)
        methods = 25 if args.stdlib else 35
        command = [args.python, '-B', 'check_examples.py']+([] if args.stdlib else ['--numpy'])
        checked = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=60)
        args.output.with_suffix('.log').write_text(checked.stdout+checked.stderr)
        if checked.returncode or f'Ran {methods} tests' not in checked.stderr:
            raise RuntimeError('Extracted checks failed; see the saved log')
        for script, reference in [
            ('pion_spectral.py', 'spectral-reference.json'),
            ('pion_alignment.py', 'alignment-reference.json'),
            ('pion_snr.py', 'snr-reference.json'),
            ('snr_correlated_groups.py', 'snr-correlated-groups-reference.json'),
            ('grpo_group_signal.py', 'grpo-group-signal-reference.json'),
            ('vla_gradient_rank.py', 'vla-gradient-rank-reference.json'),
            ('flat_spectrum.py', 'flat-spectrum-reference.json'),
        ]:
            actual = subprocess.check_output([args.python, '-B', script], cwd=folder, timeout=30)
            if json.loads(actual) != json.loads((folder/'assets'/reference).read_text()):
                raise RuntimeError('Saved reference changed: '+script)
            references[script] = {'matches_saved_reference': True,
                                  'output_sha256': hashlib.sha256(actual).hexdigest()}
        if list(folder.rglob('*.pyc')):
            raise RuntimeError('Published commands created bytecode')
        path = str(folder)
    record = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
        'headers': headers, 'members': names, 'member_count': len(names),
        'temporary_directory': path, 'temporary_directory_removed': True,
        'command': command, 'test_methods': methods, 'numpy_checks': not args.stdlib,
        'references': references,
        'bytecode_files': 0, 'download_matches_current_source': True,
        'pip_installations': 0,
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'members': len(names), 'test_methods': methods,
                      'references': len(references), 'sha256': record['sha256']}))


if __name__ == '__main__':
    main()
