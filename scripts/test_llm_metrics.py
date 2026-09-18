#!/usr/bin/env python3
"""Offline numerical and input-contract checks for the LLM metrics article."""
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/llm-training-metrics"
sys.path.insert(0, str(BUNDLE))
import metrics_lab as metrics
import paired_eval


class MetricsArticleTests(unittest.TestCase):
    def test_pass_at_k_against_exhaustive_subsets(self):
        for n in range(1, 9):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    subsets = list(itertools.combinations(range(n), k))
                    exact = sum(any(i < c for i in subset) for subset in subsets) / len(subsets)
                    self.assertAlmostEqual(metrics.pass_at_k(n, c, k), exact)
        for args in [(0, 0, 1), (3, 4, 1), (3, 1, 4), (3, 1, 0), (3.0, 1, 1)]:
            with self.assertRaises(ValueError):
                metrics.pass_at_k(*args)

    def test_nll_targets_not_batch_means(self):
        self.assertEqual(metrics.weighted_nll([2, 24], [2, 8]), 2.6)
        self.assertEqual(metrics.weighted_nll([0, 24], [0, 8]), 3)
        for args in [([], []), ([1], [0]), ([1, 2], [1]), ([1], [-1]),
                     ([math.nan], [1]), ([math.inf], [1]), ([-1], [1]), ([1], [1.5])]:
            with self.assertRaises(ValueError):
                metrics.weighted_nll(*args)

    def test_wilson_reference_values_and_symmetry(self):
        # Independent reference values from scipy.stats.binomtest(...).proportion_ci(method='wilson').
        for successes, trials, expected in [
            (0, 100, (0.0, 0.03699349820698568)),
            (70, 100, (0.6041514536665333, 0.7810511470506722)),
            (75, 100, (0.656955364519384, 0.8245478863771232)),
        ]:
            got = metrics.wilson_interval(successes, trials)
            for a, b in zip(got, expected):
                self.assertAlmostEqual(a, b, places=12)
        for n in [1, 5, 20, 100]:
            for c in range(n + 1):
                low, high = metrics.wilson_interval(c, n)
                reverse_low, reverse_high = metrics.wilson_interval(n - c, n)
                self.assertLessEqual(low, c / n)
                self.assertGreaterEqual(high, c / n)
                self.assertAlmostEqual(low, 1 - reverse_high)
                self.assertAlmostEqual(high, 1 - reverse_low)
        for args in [(0, 0), (2, 1), (-1, 10), (1.5, 10), (1, 10, 1)]:
            with self.assertRaises(ValueError):
                metrics.wilson_interval(*args)

    def test_binary_event_calibration(self):
        y = [1] * 8 + [0] * 2
        ece, brier = metrics.binary_calibration([0.8] * 10, y)
        self.assertAlmostEqual(ece, 0)
        self.assertAlmostEqual(brier, 0.16)
        ece, brier = metrics.binary_calibration([1.0] * 10, y)
        self.assertAlmostEqual(ece, 0.2)
        self.assertAlmostEqual(brier, 0.2)
        self.assertEqual(metrics.binary_calibration([0, 1], [0, 1]), (0, 0))
        for args in [([], []), ([0.1], []), ([math.nan], [1]), ([1.1], [1]),
                     ([0.5], [2]), ([0.5], [1], 0)]:
            with self.assertRaises(ValueError):
                metrics.binary_calibration(*args)

    def test_qwen_against_pinned_weight_index(self):
        snapshot = json.loads((BUNDLE / "qwen2.5-7b-budget.json").read_text())
        parts = metrics.qwen_parameter_breakdown(snapshot["config"])
        self.assertEqual(sum(parts.values()), snapshot["expected_parameters"])
        self.assertEqual(sum(parts.values()) * 2, snapshot["expected_weight_bytes"])
        self.assertEqual(parts["embedding"], parts["lm_head"])
        self.assertRegex(snapshot["revision"], r"^[a-f0-9]{40}$")

    def test_pairing_and_order_invariance(self):
        a, b, ids = paired_eval.demo()
        report = paired_eval.compare(a, b, ids, repeats=1000)
        shuffled = paired_eval.compare(a[::-1], b[::2] + b[1::2], ids[::-1], repeats=1000)
        self.assertEqual(report, shuffled)
        self.assertEqual(report["difference_pp"], 5)
        self.assertEqual(report["contingency"], {
            "both_correct": 60, "only_baseline_correct": 10,
            "only_candidate_correct": 15, "both_wrong": 15,
        })
        self.assertEqual(metrics.paired_bootstrap([0, 1], [0, 1], 100), (0, 0, 0))
        self.assertEqual(metrics.paired_bootstrap([0, 0], [1, 1], 100), (1, 1, 1))

    def test_completeness_and_invalid_annotations(self):
        a, b, ids = paired_eval.demo()
        for left, right, manifest in [(a + [a[0]], b, ids), (a[:-1], b, ids),
                                      (a, b[:-1], ids), (a, b, ids[:-1]),
                                      (a, b, ids + ["extra"]), (a, b, ids + [ids[0]])]:
            with self.assertRaises(ValueError):
                paired_eval.compare(left, right, manifest, repeats=100)
        for change in [{"correct": "true"}, {"correct": 1}, {"correct": None},
                       {"status": "error", "correct": True}, {"status": "unknown"}, {"id": ""}]:
            with self.assertRaises(ValueError):
                paired_eval.compare([{**a[0], **change}] + a[1:], b, ids, repeats=100)

    def test_cli_and_errors_remain_in_denominator(self):
        a = {"id": "x", "correct": False, "status": "error"}
        b = {"id": "x", "correct": True, "status": "ok"}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "a.jsonl").write_text(json.dumps(a) + "\n")
            (directory / "b.jsonl").write_text(json.dumps(b) + "\n")
            (directory / "ids.json").write_text('["x"]')
            args = [sys.executable, "-B", str(BUNDLE / "paired_eval.py"), "--baseline", str(directory / "a.jsonl"),
                    "--candidate", str(directory / "b.jsonl"), "--ids", str(directory / "ids.json"), "--repeats", "100"]
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["questions"], 1)
            self.assertEqual(report["difference_pp"], 100)
            self.assertEqual(report["errors_counted_in_denominator"]["baseline"], 1)
            self.assertTrue(any("degenerate" in warning for warning in report["warnings"]))
            (directory / "b.jsonl").write_text('{bad json}\n')
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
