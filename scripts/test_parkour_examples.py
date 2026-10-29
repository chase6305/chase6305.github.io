#!/usr/bin/env python3
"""Check scalar phase-weight boundaries without PyTorch or robot dependencies."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "content/posts/ai/light-loco-parkour"
spec = importlib.util.spec_from_file_location("parkour_example", BUNDLE / "parkour_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class PhaseWeightsTests(unittest.TestCase):
    def test_far_positions_and_sharp_transitions_saturate(self):
        for x in (-200., -1000., -1e308):
            with self.subTest(x=x):
                self.assertEqual(lab.phase_weights(x), [1., 0., 0.])
        for x in (200., 1000., 1e308):
            with self.subTest(x=x):
                self.assertEqual(lab.phase_weights(x), [0., 0., 1.])
        self.assertEqual(lab.phase_weights(0., inverse_temperature=1e308), [1., 0., 0.])
        self.assertEqual(lab.phase_weights(3., inverse_temperature=1e308), [0., 0., 1.])

    def test_finite_sorted_triggers_form_a_partition(self):
        rng = random.Random(20261029)
        for _ in range(1000):
            triggers = sorted(rng.sample(range(-1000, 1001), rng.randrange(8)))
            weights = lab.phase_weights(rng.uniform(-2000, 2000), triggers,
                                        inverse_temperature=10 ** rng.uniform(-3, 3))
            self.assertEqual(len(weights), len(triggers) + 1)
            self.assertTrue(all(math.isfinite(w) and 0 <= w <= 1 for w in weights))
            self.assertAlmostEqual(sum(weights), 1., places=14)

    def test_single_phase_and_one_shot_triggers(self):
        self.assertEqual(lab.phase_weights(0., triggers=()), [1.])
        self.assertEqual(lab.phase_weights(1., triggers=iter([1., 2.])),
                         lab.phase_weights(1., triggers=[1., 2.]))
        self.assertEqual(lab.phase_weights(1., triggers=[1.]), [.5, .5])

    def test_invalid_parameters_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, "1", None, 10**1000):
            with self.subTest(position=value), self.assertRaises(ValueError):
                lab.phase_weights(value)
            with self.subTest(temperature=value), self.assertRaises(ValueError):
                lab.phase_weights(0., inverse_temperature=value)
            with self.subTest(trigger=value), self.assertRaises(ValueError):
                lab.phase_weights(0., triggers=[value])
        for triggers in ([2., 1.], [1., 1.]):
            with self.subTest(triggers=triggers), self.assertRaises(ValueError):
                lab.phase_weights(0., triggers=triggers)
        for beta in (0., -1.):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                lab.phase_weights(0., inverse_temperature=beta)

    def test_published_weights_remain_numerically_identical(self):
        reference = json.loads((BUNDLE / "teaching-results.json").read_text())
        for x, expected in zip([.9, 1., 1.1, 2., 2.1], reference["smooth_phase_weights"]):
            for actual, value in zip(lab.phase_weights(x), expected):
                self.assertAlmostEqual(actual, value, places=15)


if __name__ == "__main__":
    unittest.main()
