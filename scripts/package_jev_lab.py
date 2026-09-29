#!/usr/bin/env python3
"""Create/check the fixed Jev offline archive and optionally test its contents."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/posts/ai/jev-decision-layer"
FILES = ("README.txt", "decision_gate.py", "routing_lab.py",
         "assets/gate-example-results.json", "assets/routing-results.json",
         "assets/routing-fixtures.json", "assets/routing-fixtures.csv",
         "assets/routing-tradeoff.png")
PREFIX = "jev-decision-lab/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    archive = SOURCE / "jev-decision-lab.zip"
    if not args.check:
        with zipfile.ZipFile(archive, "w") as out:
            for name in FILES:
                entry = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 9, 29, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                out.writestr(entry, (SOURCE / name).read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [PREFIX + name for name in FILES]:
            raise SystemExit("Unexpected archive members; rebuild")
        for name in FILES:
            if bundle.read(PREFIX + name) != (SOURCE / name).read_bytes():
                raise SystemExit("Stale archive member: " + name)
        if args.test:
            with tempfile.TemporaryDirectory(prefix="jev-offline-lab-") as temp:
                bundle.extractall(temp)  # Exact safe member names were checked above.
                work = Path(temp) / PREFIX
                for script in ("decision_gate.py", "routing_lab.py"):
                    subprocess.run([sys.executable, "-B", script], cwd=work,
                                   stdout=subprocess.DEVNULL, check=True)
    print(f"{archive.name}: {len(FILES)} members match source")


if __name__ == "__main__":
    main()
