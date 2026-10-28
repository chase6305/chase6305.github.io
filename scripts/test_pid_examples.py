#!/usr/bin/env python3
"""Exercise PID parameter changes and rejected samples without robot I/O."""
import copy
import math
import random
import runpy
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PID = runpy.run_path(str(ROOT / 'content/posts/pid/pid_controller.py'))['PID']


def state(controller):
    return tuple(getattr(controller, name) for name in
                 ('integral', 'previous_y', 'filtered_dy', 'last_raw', 'last_output'))


INVALID_SETTINGS = [
    *((name, value) for name in ('kp', 'ki', 'kd')
      for value in (math.nan, math.inf, -math.inf)),
    *(('limit', value) for value in (math.nan, math.inf, -math.inf, 0., -1.)),
    *(('tau', value) for value in (math.nan, math.inf, -math.inf, -.01)),
    *(('anti_windup', value) for value in (0, 1, None, 'false')),
]


class PIDExamples(unittest.TestCase):
    def test_constructor_and_updated_settings_share_the_same_contract(self):
        for name, value in INVALID_SETTINGS:
            with self.subTest(name=name, value=value):
                settings = dict(kp=2., ki=.5, kd=.1, limit=3.)
                settings[name] = value
                with self.assertRaises(ValueError):
                    PID(**settings)
                controller = PID(2., .5, .1, 3.)
                controller.update(1., 0., .01)
                before = state(controller)
                setattr(controller, name, value)
                with self.assertRaises(ValueError):
                    controller.update(1., .2, .01)
                self.assertEqual(state(controller), before)

    def test_repairing_a_rejected_setting_preserves_the_next_update(self):
        for name, value in INVALID_SETTINGS:
            with self.subTest(name=name, value=value):
                controller = PID(2., .5, .1, 3.)
                controller.update(1., 0., .01)
                expected = copy.deepcopy(controller)
                previous = getattr(controller, name)
                setattr(controller, name, value)
                with self.assertRaises(ValueError):
                    controller.update(1., .2, .01)
                setattr(controller, name, previous)
                self.assertEqual(controller.update(.7, .2, .03),
                                 expected.update(.7, .2, .03))
                self.assertEqual(state(controller), state(expected))

    def test_valid_retuning_uses_the_new_limit_and_preserves_integral_units(self):
        controller = PID(2., .5, .1, 3.)
        self.assertAlmostEqual(controller.update(1., 0., .01), 2.005)
        controller.kp, controller.ki, controller.kd = 4., 1., 0.
        controller.limit, controller.tau = 1., 0.
        self.assertEqual(controller.update(1., 0., .01), 1.)
        self.assertAlmostEqual(controller.integral, .005)
        self.assertAlmostEqual(controller.last_raw, 4.005)
        self.assertAlmostEqual(controller.update(0., 0., .01), .005)
        self.assertAlmostEqual(PID(-2., -.5, -.1, 3.).update(1., 0., .01), -2.005)

    def test_rejected_samples_and_overflow_do_not_change_history(self):
        controller = PID(2., .5, .1, 3.)
        controller.update(1., .1, .01)
        for sample in ((1., 0., 0.), (1., 0., -.1), (math.nan, 0., .01),
                       (1., math.inf, .01), (1e308, -1e308, .01)):
            with self.subTest(sample=sample):
                before = state(controller)
                with self.assertRaises(ValueError):
                    controller.update(*sample)
                self.assertEqual(state(controller), before)

    def test_derivative_and_reset(self):
        controller = PID(0., 0., 2., 100.)
        self.assertEqual(controller.update(0., .5, .01), 0.)
        self.assertEqual(controller.update(100., .5, .01), 0.)
        controller.reset(measured=.5, integral_output=1.)
        self.assertEqual(controller.update(.5, .5, .01), 1.)
        for settings in (dict(measured=math.nan), dict(integral_output=math.inf)):
            before = state(controller)
            with self.assertRaises(ValueError):
                controller.reset(**settings)
            self.assertEqual(state(controller), before)

    def test_finite_output_bounds_during_valid_parameter_changes(self):
        rng = random.Random(20261028)
        controller = PID(2., .5, .1, 3.)
        for _ in range(1000):
            controller.kp = rng.uniform(-3., 3.)
            controller.ki = rng.uniform(-2., 2.)
            controller.kd = rng.uniform(-1., 1.)
            controller.limit = 10 ** rng.uniform(-3., 2.)
            controller.tau = rng.choice((0., .02, 1.))
            controller.anti_windup = rng.choice((True, False))
            output = controller.update(rng.uniform(-10., 10.), rng.uniform(-10., 10.),
                                       10 ** rng.uniform(-4., -1.))
            self.assertTrue(math.isfinite(output))
            self.assertLessEqual(abs(output), controller.limit)
            self.assertEqual(controller.last_output, output)


if __name__ == '__main__':
    unittest.main()
