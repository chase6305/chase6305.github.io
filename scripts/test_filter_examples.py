#!/usr/bin/env python3
"""One Euro scalar/rotation configuration and history contracts; NumPy/SciPy only."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

# Imported example files live in published page bundles.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "blog_filters", ROOT / "content/posts/robotics/control/robot-filters/robot_filters_lab.py")
filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters)
CLASSES = (filters.OneEuro, filters.RotationOneEuro)
INVALID = {"minimum": (0., -1., math.nan, math.inf, -math.inf),
           "beta": (-.01, math.nan, math.inf, -math.inf),
           "dc": (0., -1., math.nan, math.inf, -math.inf)}


def sample(cls, angle=0.):
    return angle if cls is filters.OneEuro else Rotation.from_rotvec([0., 0., angle]).as_quat()


def state(filt):
    if isinstance(filt, filters.OneEuro):
        return {"t": filt.t, "y": filt.y, "d": filt.d}
    return {"t": filt.t, "r": None if filt.r is None else filt.r.as_quat(), "speed": filt.speed}


class OneEuroContracts(unittest.TestCase):
    def assert_state(self, filt, expected):
        actual = state(filt)
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value)
            else:
                self.assertEqual(actual[key], value, key)

    def test_constructor_rejects_invalid_configuration(self):
        names = {"minimum": "min_cutoff", "beta": "beta", "dc": "derivative_cutoff"}
        for cls in CLASSES:
            for field, values in INVALID.items():
                for value in values:
                    with self.subTest(cls=cls.__name__, field=field, value=value):
                        with self.assertRaises(ValueError):
                            cls(**{names[field]: value})

    def test_mutation_rejected_without_advancing_history(self):
        for cls in CLASSES:
            for initialized in (False, True):
                for field, values in INVALID.items():
                    for value in values:
                        with self.subTest(cls=cls.__name__, initialized=initialized, field=field, value=value):
                            filt, reference = cls(), cls()
                            if initialized:
                                filt.update(0., sample(cls)); reference.update(0., sample(cls))
                            previous = state(filt)
                            valid = getattr(filt, field)
                            setattr(filt, field, value)
                            with self.assertRaises(ValueError):
                                filt.update(.01, sample(cls, .2))
                            self.assert_state(filt, previous)
                            setattr(filt, field, valid)
                            actual = filt.update(.02, sample(cls, .3))
                            expected = reference.update(.02, sample(cls, .3))
                            np.testing.assert_array_equal(actual, expected)

    def test_invalid_sample_preserves_history(self):
        for cls in CLASSES:
            filt = cls(); filt.update(0., sample(cls)); previous = state(filt)
            bad_value = math.nan if cls is filters.OneEuro else [0., 0., 0., 0.]
            for timestamp, value in [(0., sample(cls, .2)), (math.nan, sample(cls)), (.01, bad_value)]:
                with self.subTest(cls=cls.__name__, timestamp=timestamp):
                    with self.assertRaises(ValueError):
                        filt.update(timestamp, value)
                    self.assert_state(filt, previous)

    def test_valid_retuning_matches_constant_cutoff(self):
        # With beta=0, the gain is independently determined by the new cutoff.
        gain = .01 / (.01 + 1. / (2. * math.pi * 2.))
        for cls in CLASSES:
            filt = cls(beta=0.); filt.update(0., sample(cls))
            filt.minimum = 2.
            actual = filt.update(.01, sample(cls, .2))
            actual = actual if cls is filters.OneEuro else Rotation.from_quat(actual).as_rotvec()[2]
            self.assertAlmostEqual(actual, gain * .2, places=14)

    def test_repeated_valid_retuning_keeps_finite_output(self):
        for cls in CLASSES:
            filt = cls()
            for index in range(1000):
                filt.minimum = .3 + (index % 17) * .1
                filt.beta = (index % 11) * .02
                filt.dc = .5 + (index % 7) * .2
                value = math.sin(index * .07)
                if cls is filters.RotationOneEuro:
                    value = Rotation.from_rotvec([value, .2 * math.cos(index * .03), index * .001]).as_quat()
                result = filt.update(index * .01, value)
                self.assertTrue(np.isfinite(result).all())
                if cls is filters.RotationOneEuro:
                    self.assertAlmostEqual(np.linalg.norm(result), 1., places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
