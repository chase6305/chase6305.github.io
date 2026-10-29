#!/usr/bin/env python3
"""Check the planar 2R radial CDF using only NumPy and forward geometry."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workspace_example",
    ROOT / "content/posts/robotics/workspace/whole-workspace/workspace_sampling.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class RadialCDFTests(unittest.TestCase):
    def test_negative_queries_and_support_endpoints(self):
        np.testing.assert_array_equal(lab.joint_radial_cdf([-np.inf, -2., -1., -.5, 0., .4]), np.zeros(6))
        np.testing.assert_array_equal(lab.joint_radial_cdf([1.6, 2., np.inf]), np.ones(3))
        self.assertTrue(np.isscalar(lab.joint_radial_cdf(-1.)))
        actual = lab.joint_radial_cdf([[-2., 0.], [1.6, 2.]])
        np.testing.assert_array_equal(actual, [[0., 0.], [1., 1.]])

    def test_extreme_queries_are_clipped_before_squaring(self):
        with np.errstate(over='raise', invalid='raise'):
            actual = lab.joint_radial_cdf([-1e308, 1e308])
        np.testing.assert_array_equal(actual, [0., 1.])

    def test_known_joint_angles_and_global_monotonicity(self):
        for l1, l2 in [(1., .6), (.7, .3), (1., 1.), (3., 2.)]:
            inner, outer = abs(l1-l2), l1+l2
            midpoint = math.sqrt(l1*l1+l2*l2)
            np.testing.assert_allclose(lab.joint_radial_cdf([inner, midpoint, outer], l1, l2),
                                       [0., .5, 1.], atol=1e-14, rtol=0)
            values = lab.joint_radial_cdf(np.linspace(-2*outer, 2*outer, 10001), l1, l2)
            self.assertTrue(np.all(np.diff(values) >= -1e-15))
            self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_forward_joint_geometry_matches_cdf(self):
        # Uniform angular quadrature uses forward kinematics, independently of
        # the inverse arccos expression. Each crossing contributes <=1/N error.
        count = 131072
        q2 = (np.arange(count)+.5)*(2*np.pi/count)
        for l1, l2 in [(1., .6), (.7, .3), (1., 1.), (3., 2.)]:
            radii = np.sort(np.hypot(l1+l2*np.cos(q2), l2*np.sin(q2)))
            queries = np.linspace(-2*(l1+l2), 2*(l1+l2), 129)
            empirical = np.searchsorted(radii, queries, side='right')/count
            np.testing.assert_allclose(lab.joint_radial_cdf(queries, l1, l2), empirical,
                                       atol=2/count, rtol=0)

    def test_scaling_and_link_order_preserve_probability(self):
        radii = np.linspace(-3., 3., 1000)
        expected = lab.joint_radial_cdf(radii, 1., .6)
        np.testing.assert_allclose(lab.joint_radial_cdf(radii, .6, 1.), expected,
                                   atol=1e-12, rtol=0)
        for scale in [.001, 1000.]:
            np.testing.assert_allclose(lab.joint_radial_cdf(radii*scale, scale, .6*scale),
                                       expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
