#!/usr/bin/env python3
"""ACT timing weights: endpoint limits and an independent decimal oracle."""
import ast
import contextlib
import decimal
import importlib.util
import io
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/lerobot-act"
spec = importlib.util.spec_from_file_location("act_timing", BUNDLE / "act_timing_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

# Execute the published function rather than maintaining a second copy in tests.
from validate_blog import code_blocks
snippet = next(code for language, code, _ in code_blocks((BUNDLE / "index.md").read_text())
               if language == "python" and "def ensemble(" in code)
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(ast.parse(snippet), "<published ACT example>", "exec"), namespace)
FUNCTIONS = (lab.weighted_average, namespace["ensemble"])


class ACTWeightsTests(unittest.TestCase):
    def test_extreme_finite_coefficients_choose_expected_endpoint(self):
        for function in FUNCTIONS:
            for size in (1, 3, 100):
                values = [i / 10 for i in range(size)]
                self.assertEqual(function(values, 1e308), values[0])
                self.assertEqual(function(values, -1e308), values[-1])
            with self.assertRaises(ValueError):
                function([], 0)

    def test_published_direction_and_uniform_mean(self):
        for function in FUNCTIONS:
            values = [.2, .5, .8]
            self.assertEqual(f"{function(values, 0):.4f}", "0.5000")
            self.assertEqual(f"{function(values, .5):.4f}", "0.4040")
            self.assertEqual(f"{function(values, -.5):.4f}", "0.5960")
            for coefficient in (-1000, -.5, 0, .5, 1000):
                self.assertAlmostEqual(function([3.5] * 20, coefficient), 3.5)

    def test_random_weighted_means_against_decimal(self):
        rng = random.Random(20261029)
        with decimal.localcontext() as context:
            context.prec = 60
            for _ in range(500):
                values = [rng.uniform(-10, 10) for _ in range(rng.randrange(1, 40))]
                coefficient = rng.uniform(-100, 100)
                c = decimal.Decimal.from_float(coefficient)
                weights = [(-c * i).exp() for i in range(len(values))]
                expected = float(sum(w * decimal.Decimal.from_float(v)
                                     for w, v in zip(weights, values)) / sum(weights))
                for function in FUNCTIONS:
                    actual = function(values, coefficient)
                    self.assertTrue(math.isfinite(actual))
                    self.assertTrue(min(values) - 1e-12 <= actual <= max(values) + 1e-12)
                    self.assertTrue(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                                    (coefficient, values, expected, actual))

    def test_cli_rejects_nonfinite_coefficients(self):
        for coefficient in ("nan", "inf", "-inf"):
            result = subprocess.run([sys.executable, "-B", str(BUNDLE / "act_timing_lab.py"),
                                     "--coeff=" + coefficient], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coeff must be finite", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
