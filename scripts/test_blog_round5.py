#!/usr/bin/env python3
"""Offline regression for the downloadable impedance log analysis example."""
import copy
import csv
import importlib.util
import json
import math
import random
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
    api_cases = csv_cases = metric_cases = timing_cases = 0
    with tempfile.TemporaryDirectory(prefix="chase-blog-round5-impedance-") as folder:
        directory = Path(folder)
        subprocess.run([sys.executable, str(BUNDLE / "impedance_1d.py")],
                       cwd=directory, check=True, capture_output=True, text=True, timeout=10)
        path = directory / "impedance_response.csv"
        rows = analysis.load_rows(path)
        metrics = analysis.analyze(rows, .5, 1.5)
        assert abs(metrics["estimated_stiffness_n_m"] - 1000) < 1
        assert not analysis.acceptance_failures(metrics, acceptance)
        intervals = [b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:])]
        assert metrics["sample_period_min_s"] == min(intervals)
        assert metrics["sample_period_max_s"] == max(intervals)

        # The endpoints and sample count stay fixed. The old mean-only check
        # accepted both an isolated long gap and a very short catch-up interval.
        delayed = copy.deepcopy(rows)
        delayed[20]["time_s"] += .009
        delayed_metrics = analysis.analyze(delayed, .5, 1.5)
        assert delayed_metrics["sample_period_mean_s"] == metrics["sample_period_mean_s"]
        failures = analysis.acceptance_failures(delayed_metrics, acceptance)
        assert len(failures) == 2 and all("_relative_deviation" in x for x in failures)
        legacy = dict(acceptance)
        del legacy["max_sample_period_relative_deviation"]
        assert not analysis.acceptance_failures(delayed_metrics, legacy)
        timing_cases += 1

        # Compare checking the two extremes against checking every interval.
        rng = random.Random(20261029)
        expected = acceptance["expected_sample_period_s"]
        tolerance = acceptance["max_sample_period_relative_deviation"]
        for index in range(300):
            amplitude = (.01, .04, .06, .9)[index % 4]
            periods = [expected * rng.uniform(1 - amplitude, 1 + amplitude) for _ in range(40)]
            probe = dict(metrics, sample_period_min_s=min(periods), sample_period_max_s=max(periods))
            got = analysis.acceptance_failures(probe, acceptance)
            assert bool(got) == any(abs(period / expected - 1) > tolerance for period in periods)
            timing_cases += 1
        for field in ("sample_period_min_s", "sample_period_max_s"):
            # Each extreme is independently enforced, including NaN/Inf.
            for value in (0., .001, .019, math.nan, math.inf, -math.inf):
                assert analysis.acceptance_failures(dict(metrics, **{field: value}), acceptance)
                timing_cases += 1
        for sign in (-1, 1):
            value = expected * (1 + sign * tolerance)
            probe = dict(metrics, sample_period_min_s=value, sample_period_max_s=value)
            assert not analysis.acceptance_failures(probe, acceptance)
            outside = math.nextafter(value, 0 if sign < 0 else math.inf)
            probe = dict(metrics, sample_period_min_s=outside, sample_period_max_s=outside)
            assert analysis.acceptance_failures(probe, acceptance)
            timing_cases += 1

        acceptance_path = directory / "acceptance.json"
        for value in (-.01, math.nan, math.inf, -math.inf, True, "0.05", None):
            acceptance_path.write_text(json.dumps(dict(acceptance, max_sample_period_relative_deviation=value)))
            rejected(analysis.load_acceptance, acceptance_path)
            timing_cases += 1
        acceptance_path.write_text(json.dumps(legacy))
        assert analysis.load_acceptance(acceptance_path) == legacy

        for on, off in ((True, 1.5), (.5, True), (math.nan, 1.5), (.5, math.inf)):
            rejected(analysis.analyze, rows, on, off)
            api_cases += 1
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
        delayed_path = directory / "delayed.csv"
        with delayed_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["time_s", "force_n", "position_m"])
            writer.writeheader()
            writer.writerows(delayed)
        timing_failure = cli(delayed_path)
        assert timing_failure.returncode == 1 and "PASS:" not in timing_failure.stdout
        assert "sample_period_min_s_relative_deviation" in timing_failure.stdout
        assert "sample_period_max_s_relative_deviation" in timing_failure.stdout
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
        "record_interval_cases":timing_cases,
        "hardware":"not used"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
