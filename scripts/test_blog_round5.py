#!/usr/bin/env python3
"""Offline regression for the downloadable impedance log analysis example."""
import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/robotics/control/impedance-control"
# Article bundles are published by Hugo; never put test bytecode inside them.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("blog_impedance_analysis", BUNDLE / "analyze_impedance_log.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rejected(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def main():
    acceptance = analysis.load_acceptance(BUNDLE / "impedance_acceptance_example.json")
    api_cases = csv_cases = metric_cases = 0
    with tempfile.TemporaryDirectory(prefix="chase-blog-round5-impedance-") as folder:
        directory = Path(folder)
        subprocess.run([sys.executable, str(BUNDLE / "impedance_1d.py")],
                       cwd=directory, check=True, capture_output=True, text=True, timeout=10)
        path = directory / "impedance_response.csv"
        rows = analysis.load_rows(path)
        metrics = analysis.analyze(rows, .5, 1.5)
        assert abs(metrics["estimated_stiffness_n_m"] - 1000) < 1
        assert not analysis.acceptance_failures(metrics, acceptance)
        for field in ("time_s", "force_n", "position_m"):
            for value in (float("nan"), float("inf"), -float("inf"), None, True):
                bad = copy.deepcopy(rows)
                bad[10][field] = value
                rejected(analysis.analyze, bad, .5, 1.5)
                api_cases += 1
        for bad in ([], rows[:2], list(reversed(rows)), rows[:2] + [rows[1]] + rows[2:]):
            rejected(analysis.analyze, bad, .5, 1.5)
            api_cases += 1
        missing = copy.deepcopy(rows)
        del missing[10]["position_m"]
        rejected(analysis.analyze, missing, .5, 1.5)
        api_cases += 1

        for field in ("estimated_stiffness_n_m", "sample_period_mean_s", "overshoot_percent",
                      "released_position_rms_m", "settling_time_2pct_s", "release_settling_time_2pct_s"):
            for value in (float("nan"), float("inf"), -float("inf")):
                bad = dict(metrics, **{field: value})
                assert analysis.acceptance_failures(bad, acceptance), (field, value)
                metric_cases += 1
        header = "time_s,force_n,position_m\n"
        invalid_csv = [
            header + f"0,0,0\n1,10,{value}\n2,0,0\n"
            for value in ("nan", "inf", "-inf", "", "not-a-number")
        ] + [
            "time_s,force_n\n0,0\n1,10\n2,0\n",
            "time_s,force_n,position_m,force_n\n0,0,0,0\n1,10,.01,10\n2,0,0,0\n",
            header + "0,0,0\n1,10,.01,extra\n2,0,0\n",
            header + "0,0,0\n1,10\n2,0,0\n",
            header + "0,0,0\n0,10,.01\n2,0,0\n"
        ]
        bad_path = directory / "invalid.csv"
        for content in invalid_csv:
            bad_path.write_text(content)
            rejected(analysis.load_rows, bad_path)
            csv_cases += 1

        def cli(target):
            return subprocess.run([sys.executable, str(BUNDLE / "analyze_impedance_log.py"),
                str(target), "--acceptance", str(BUNDLE / "impedance_acceptance_example.json")],
                cwd=directory, capture_output=True, text=True, timeout=10)
        normal = cli(path)
        assert normal.returncode == 0 and "PASS:" in normal.stdout
        corrupt = copy.deepcopy(rows)
        corrupt[70]["position_m"] = math.nan
        with bad_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["time_s","force_n","position_m"])
            writer.writeheader()
            writer.writerows(corrupt)
        invalid = cli(bad_path)
        assert invalid.returncode != 0 and "PASS:" not in invalid.stdout
    print(json.dumps({"valid_samples":len(rows), "estimated_stiffness_n_m":metrics["estimated_stiffness_n_m"],
        "invalid_api_cases":api_cases, "invalid_csv_cases":csv_cases,
        "nonfinite_metric_cases":metric_cases, "cli_valid_and_invalid":"passed",
        "hardware":"not used"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
