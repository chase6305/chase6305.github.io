#!/usr/bin/env python3
"""Build or verify the control tutorial ZIP; --test runs extracted NumPy CLIs.

python scripts/package_control_lab.py
python scripts/package_control_lab.py --check
python scripts/package_control_lab.py --check --test
WholeBodyX integration and optional plots are outside this NumPy-only check.
"""
import argparse
import json
import math
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


def compare_report(actual, expected, path="report"):
    """Check published default results with numeric tolerance; ignore root NumPy version.

    Versions are still recorded in each report. This checks numerical examples,
    not equivalence of dependency environments or real-time performance.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        ignored = {"numpy"} if path == "report" else set()
        if set(actual) - ignored != set(expected) - ignored:
            raise SystemExit(f"Published report keys differ at {path}")
        for key in expected.keys() - ignored:
            compare_report(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise SystemExit(f"Published report length differs at {path}")
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            compare_report(got, wanted, f"{path}[{index}]")
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not (math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)):
            raise SystemExit(f"Published numeric result differs at {path}: {actual} != {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise SystemExit(f"Published result differs at {path}: {actual!r} != {expected!r}")


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
            for script, output, snapshot in (("atomic_control.py", "results-atomic", "atomic-control-report.json"),
                                             ("feedback_demo.py", "results-feedback", "feedback-report.json")):
                result = subprocess.run([sys.executable, "-B", script, "--output", output],
                                        cwd=root, text=True, capture_output=True, timeout=60)
                if result.returncode:
                    raise SystemExit(f"{script} failed:\n{result.stdout}\n{result.stderr}")
                report = json.loads((root / output / "report.json").read_text())
                if report != json.loads(result.stdout):
                    raise SystemExit(f"{script}: saved report differs from stdout")
                expected = json.loads((BUNDLE / "assets" / snapshot).read_text())
                compare_report(report, expected)
                print(f"Passed extracted {script}: assertions, output and published report consistency.", flush=True)


if __name__ == "__main__":
    main()
