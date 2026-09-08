#!/usr/bin/env python3
"""Build or verify the control tutorial ZIP; --test runs extracted NumPy CLIs.

python scripts/package_control_lab.py
python scripts/package_control_lab.py --check
python scripts/package_control_lab.py --check --test
WholeBodyX integration and optional plots are outside this NumPy-only check.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BUNDLE = Path(__file__).resolve().parents[1] / "content/posts/planner/to_mpc_wbc"
FILES = {name: name for name in ("atomic_control.py", "feedback_demo.py", "plot_feedback.py",
                                 "wholebodyx_demo.py", "reference_failures.py")}
FILES.update({"CONTROL_README.txt": "README.txt", "requirements-atomic.txt": "requirements.txt"})


def check_archive():
    with zipfile.ZipFile(BUNDLE / "control-lab.zip") as archive:
        names = archive.namelist()
        expected = {"control-lab/" + name for name in FILES.values()}
        if len(names) != len(expected) or set(names) != expected:
            raise SystemExit("Unexpected ZIP members; rebuild with scripts/package_control_lab.py")
        for source, member in FILES.items():
            if archive.read("control-lab/" + member) != (BUNDLE / source).read_bytes():
                raise SystemExit(f"Stale ZIP member: {member}; rebuild with scripts/package_control_lab.py")
    print(f"Verified {len(FILES)} ZIP members against article sources.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check ZIP without modifying it")
    parser.add_argument("--test", action="store_true", help="Implies --check; run extracted NumPy examples")
    args = parser.parse_args()
    if not (args.check or args.test):
        with zipfile.ZipFile(BUNDLE / "control-lab.zip", "w") as archive:
            for source, member in FILES.items():
                info = zipfile.ZipInfo("control-lab/" + member, date_time=(2026, 9, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (BUNDLE / source).read_bytes())
    check_archive()
    if args.test:
        with tempfile.TemporaryDirectory(prefix="chase-control-download-") as directory:
            with zipfile.ZipFile(BUNDLE / "control-lab.zip") as archive:
                archive.extractall(directory)  # Exact member allowlist checked above.
            root = Path(directory) / "control-lab"
            for script, output in (("atomic_control.py", "results-atomic"),
                                   ("feedback_demo.py", "results-feedback")):
                result = subprocess.run([sys.executable, "-B", script, "--output", output],
                                        cwd=root, text=True, capture_output=True, timeout=60)
                if result.returncode:
                    raise SystemExit(f"{script} failed:\n{result.stdout}\n{result.stderr}")
                report = json.loads((root / output / "report.json").read_text())
                if report != json.loads(result.stdout):
                    raise SystemExit(f"{script}: saved report differs from stdout")
                print(f"Passed extracted {script}: assertions and report consistency.", flush=True)


if __name__ == "__main__":
    main()
