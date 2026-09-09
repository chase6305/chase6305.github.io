#!/usr/bin/env python3
"""Build or verify the control tutorial ZIP; --test runs extracted NumPy CLIs.

python scripts/package_control_lab.py
python scripts/package_control_lab.py --check
python scripts/package_control_lab.py --check --test
WholeBodyX integration and optional plots are outside this NumPy-only check.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BUNDLE = Path(__file__).resolve().parents[1] / "content/posts/planner/to_mpc_wbc"
FILES = {name: name for name in ("atomic_control.py", "feedback_demo.py", "plot_feedback.py",
                                 "wholebodyx_demo.py", "reference_failures.py", "test_atomic_control.py")}
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


def check_trajectory(path, published, final):
    """Check the default 0.1 s, target=0.3 rad integrator experiment's CSV."""
    def read(source):
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames
            if not names or len(set(names)) != len(names):
                raise SystemExit(f"Invalid CSV header: {source}")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise SystemExit(f"Invalid CSV row: {source}")
                try:
                    rows.append({key: float(value) for key, value in row.items()})
                except ValueError as exc:
                    raise SystemExit(f"Non-numeric CSV row: {source}") from exc
            return names, rows

    names, rows = read(path)
    expected_names, expected = read(published)
    if names != expected_names or not rows:
        raise SystemExit(f"CSV columns changed or trajectory empty: {path.name}")
    compare_report(rows, expected, path.name)
    q, velocity = 0., 0.
    for step, row in enumerate(rows, 1):
        compare_report(row["step"], step, f"{path.name}.step")
        compare_report(row["time"], .1 * step, f"{path.name}[{step}].time")
        measured = q + row.get("offset", 0.)
        if "measured_position" in row:
            compare_report(row["measured_position"], measured, f"{path.name}[{step}].measurement")
        compare_report(row["position"], measured + .1 * row["velocity"], f"{path.name}[{step}].integration")
        compare_report(row["absolute_error"], abs(.3 - row["position"]), f"{path.name}[{step}].error")
        if (abs(row["position"]) > .5 + 1e-8 or abs(row["velocity"]) > 1. + 1e-8
                or abs(row["velocity"] - velocity) > .2 + 1e-8):
            raise SystemExit(f"Default integrator bound violated: {path.name}, step {step}")
        q, velocity = row["position"], row["velocity"]
    compare_report(rows[-1], final, f"{path.name}.final")
    print(f"Verified {path.name}: {len(rows)} rows, dynamics, bounds and report endpoint.", flush=True)


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
            subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", "test_atomic_control.py"],
                           cwd=root, check=True, timeout=60)
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
                if script == "atomic_control.py":
                    check_trajectory(root / output / "rollout.csv", BUNDLE / "assets/atomic-control-rollout.csv",
                                     report["rollout"]["final"])
                else:
                    for name, published in (("replay", "feedback-replay.csv"),
                                            ("feedback", "feedback-replan.csv")):
                        check_trajectory(root / output / f"{name}.csv", BUNDLE / "assets" / published,
                                         report["cases"][name]["final"])
                print(f"Passed extracted {script}: assertions, output and published report consistency.", flush=True)


if __name__ == "__main__":
    main()
