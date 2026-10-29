"""Build or check the deterministic, standard-library-only TAMP lab download."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/planner/task-and-motion-planning"
ARCHIVE = BUNDLE / "tamp-lab.zip"
FILES = ("README.txt", "tamp_constraints.py", "tamp_search.py",
         "assets/constraint-results.json", "assets/search-results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.check:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if archive.namelist() != list(FILES):
                raise SystemExit("TAMP ZIP member list is stale")
            for name in FILES:
                if archive.read(name) != (BUNDLE / name).read_bytes():
                    raise SystemExit(f"Stale TAMP ZIP member: {name}")
    else:
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name in FILES:
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (BUNDLE / name).read_bytes())
    if args.test:
        with tempfile.TemporaryDirectory(prefix="tamp-lab-") as directory:
            with zipfile.ZipFile(ARCHIVE) as archive:
                archive.extractall(directory)
            for source, reference in (("tamp_constraints.py", "constraint-results.json"),
                                      ("tamp_search.py", "search-results.json")):
                result = subprocess.run([sys.executable, "-B", source, "--check"],
                    cwd=directory, check=True, capture_output=True, timeout=60)
                expected = Path(directory, "assets", reference).read_bytes()
                if result.stdout != expected:
                    raise SystemExit(f"Reference output mismatch: {source}")
    print("TAMP archive matches source" + ("; extracted examples passed" if args.test else ""))


if __name__ == "__main__":
    main()
