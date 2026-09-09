#!/usr/bin/env python3
"""Build or byte-check two self-contained article bundles; optionally test extracted files."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {
    'gspo': ['README.txt', 'requirements.txt', 'gspo_lab.py', 'gspo_update.py',
             'gspo_online.py', 'test_gspo_lab.py', 'plot_results.py', 'plot_online.py',
             'assets/results.json', 'assets/update-results.json', 'assets/online-results.json'],
    'gen-1-5': ['README.txt', 'prompt_eval.py', 'test_prompt_eval.py', 'assets/example-results.json'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    for slug, files in BUNDLES.items():
        source = ROOT / 'content/posts/ai' / slug
        prefix = slug + '-lab/'
        archive = source / (slug + '-lab.zip')
        if not args.check:
            with zipfile.ZipFile(archive, 'w') as out:
                for name in files:
                    entry = zipfile.ZipInfo(prefix + name, date_time=(2026, 9, 9, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o100644 << 16
                    out.writestr(entry, (source / name).read_bytes())
        with zipfile.ZipFile(archive) as bundle:
            if bundle.namelist() != [prefix + name for name in files]:
                raise SystemExit(f'{archive.name}: unexpected members; rebuild')
            for name in files:
                if bundle.read(prefix + name) != (source / name).read_bytes():
                    raise SystemExit(f'{archive.name}: stale member {name}; rebuild')
            if args.test:
                with tempfile.TemporaryDirectory(prefix='blog-lab-') as temp:
                    bundle.extractall(temp)  # member list matched the fixed safe allowlist above
                    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                                    '-p', 'test_*.py', '-v'], cwd=Path(temp) / (slug + '-lab'), check=True)
        print(f'{archive.name}: {len(files)} members match source', flush=True)


if __name__ == '__main__':
    main()
