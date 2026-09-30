#!/usr/bin/env python3
"""Build/check the VLA article's deterministic, self-contained example archive."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content/posts/ai/vla-evolution"
FILES = ("README.txt", "vla_lab.py", "plot_flow_modes.py", "flow_matching_lab.py",
         "assets/flow-learning-results.json", "assets/flow-learned-distribution.png",
         "assets/flow-solver-comparison.png", "assets/flow-sample-paths.png")
PREFIX = "vla-labs/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Run the dependency-free examples after extraction")
    args = parser.parse_args()
    archive = SOURCE / "vla-labs.zip"
    if not args.check:
        with zipfile.ZipFile(archive, "w") as bundle:
            for name in FILES:
                member = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 9, 30, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                bundle.writestr(member, (SOURCE / name).read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [PREFIX + name for name in FILES]:
            raise SystemExit("Unexpected archive members; rebuild")
        for name in FILES:
            if bundle.read(PREFIX + name) != (SOURCE / name).read_bytes():
                raise SystemExit("Stale archive member: " + name)
        if args.test:
            with tempfile.TemporaryDirectory(prefix="vla-offline-labs-") as temporary:
                bundle.extractall(temporary)  # Exact safe member names checked above.
                work = Path(temporary) / PREFIX
                subprocess.run([sys.executable, "-B", "vla_lab.py"], cwd=work,
                               stdout=subprocess.DEVNULL, check=True)
                subprocess.run([sys.executable, "-B", "vla_lab.py", "--self-test"],
                               cwd=work, check=True)
    print(f"{archive.name}: {len(FILES)} members match source")


if __name__ == "__main__":
    main()
