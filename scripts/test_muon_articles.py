"""Independent checks for the two general Muon articles; no ML runtime needed.

Optional --numpy checks matrix/SVD geometry. This does not train a model or
validate a distributed optimizer, hardware precision, or a universal batch law.
"""

import importlib.util
import itertools
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, bundle):
    spec = importlib.util.spec_from_file_location(name, ROOT / "content/posts/ai" / bundle / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


muon = load("muon_math", "muon-optimizer")
batch = load("batch_math", "muon-learning-rate-batch-size")
NUMPY = "--numpy" in sys.argv
if NUMPY:
    sys.argv.remove("--numpy")


class MuonArticleTests(unittest.TestCase):
    def test_scalar_response_decimal(self):
        with localcontext() as ctx:
            ctx.prec = 65
            for initial in ("0", "0.00001", "0.01", "0.1", "0.5", "1"):
                x = Decimal(initial)
                for steps in range(6):
                    self.assertAlmostEqual(muon.ns_response(float(initial), steps), float(x), places=12)
                    x = Decimal("3.4445") * x - Decimal("4.775") * x**3 + Decimal("2.0315") * x**5
        self.assertAlmostEqual(muon.ns_response(1, 1), 0.701)
        self.assertNotAlmostEqual(muon.ns_response(0.5), 1)

    def test_layout_and_rank(self):
        forward = muon.shape_factors(4096, 1024)
        backward = muon.shape_factors(1024, 4096)
        self.assertEqual(forward["jordan"], 2)
        self.assertEqual(backward["jordan"], 1)
        self.assertEqual(backward["muP_factor"], 0.5)
        self.assertEqual(forward["moonlight"], backward["moonlight"])
        # Independent count of r unit singular values: sum of squared entries = r.
        for m, n, r in ((4096, 1024, 1024), (1024, 4096, 256), (7, 9, 0)):
            got = muon.polar_rms(m, n, r)
            self.assertAlmostEqual(m*n*got**2, r)
        self.assertAlmostEqual(forward["moonlight"] * muon.polar_rms(4096, 1024), 0.2)
        self.assertAlmostEqual(forward["moonlight"] * muon.polar_rms(4096, 1024, 256), 0.1)

    def test_same_parameter_update_after_conversion(self):
        eta, wd = 0.001, 0.1
        new_eta, new_wd = muon.convert_convention(eta, wd, 12.8, 1)
        for weight, direction in ((3, -0.5), (-4, 0.25), (0, 1)):
            original = (1 - eta*wd)*weight - eta*12.8*direction
            converted = (1 - new_eta*new_wd)*weight - new_eta*direction
            self.assertAlmostEqual(original, converted)

    def test_qk_bilinear_and_shared_key(self):
        # Independent Q/K vectors, not only multiplying the reported scores.
        q, k = [3, 4], [8, 6]
        gamma = muon.clip_ratio(sum(x*y for x, y in zip(q, k)), 24)
        q2 = [x * math.sqrt(gamma) for x in q]
        k2 = [x * math.sqrt(gamma) for x in k]
        self.assertAlmostEqual(sum(x*y for x, y in zip(q2, k2)), 24)
        ex = muon.qk_example()
        self.assertEqual(ex["per_head"], [100, 50])
        self.assertEqual(ex["whole_layer"], [100, 25])
        self.assertEqual(sum(ex["mla_correct_content_rotary"]), 100)
        self.assertAlmostEqual(sum(ex["mla_wrong_content_rotary"]), 116.5685424949238)
        self.assertEqual(ex["stale_statistic_new_score"], 200)

    def test_batch_ratio_and_limits(self):
        ratio = batch.mean_field_lr(1024, 4096) / batch.mean_field_lr(256, 4096)
        self.assertAlmostEqual(ratio**2, float(Fraction(17, 5)))
        self.assertAlmostEqual(batch.mean_field_lr(1e16, 1), 1)
        self.assertAlmostEqual(batch.mean_field_lr(1e-12, 1) / 1e-6, 1)

    def test_momentum_as_explicit_filter_weights(self):
        for beta in (0, 0.5, 0.95):
            ema = [(1-beta)*beta**j for j in range(2000)]
            nest = [1-beta**2] + [(1-beta)*beta**(j+1) for j in range(1, 2000)]
            self.assertAlmostEqual(sum(ema), 1)
            self.assertAlmostEqual(sum(nest), 1)
            self.assertAlmostEqual(sum(x*x for x in ema), batch.ema_variance_factor(beta))
            self.assertAlmostEqual(sum(x*x for x in nest), batch.nesterov_variance_factor(beta))
        self.assertAlmostEqual(batch.data_clock_beta(0.95, 4)**25, 0.95**100)

    def test_decay_equal_sample_budget(self):
        # Independently apply 100 old updates versus 25 updates at fourfold batch.
        eta, wd, new_eta = 0.001, 0.1, 0.002
        new_wd = batch.data_clock_decay(eta, wd, new_eta, 4)
        w0 = w1 = 2.0
        for _ in range(100):
            w0 *= 1-eta*wd
        for _ in range(25):
            w1 *= 1-new_eta*new_wd
        self.assertAlmostEqual(w0, w1, places=13)
        self.assertLess(new_wd, 0.2)  # small-step approximation is not exact.
        self.assertEqual(batch.data_clock_decay(eta, 0, new_eta, 4), 0)

    def test_noise_estimator_exact_enumeration(self):
        # True scalar mean=2, single-sample noise=+/-1. Enumerate K=2 iid records.
        signals, noises = [], []
        for eps in itertools.product((-1, 1), repeat=2):
            got = batch.noise_estimate([[2+x] for x in eps], 1)
            signals.append(got["signal"])
            noises.append(got["single_sample_noise"])
        self.assertEqual(sum(signals)/4, 4)
        self.assertEqual(sum(noises)/4, 1)
        negative = batch.noise_estimate([[-1], [1]], 1)
        self.assertLess(negative["signal"], 0)
        self.assertIsNone(negative["critical_batch_estimate"])

    def test_gaussian_integral_independent_quadrature(self):
        # A separate fixed-grid Simpson rule checks the adaptive integrator.
        # At r=100 the narrow feature has width ~0.01; h=1e-4 resolves it.
        panels = 120000
        h = 12 / panels
        for B in (1, 10, 100, 10000):
            r = 100 / math.sqrt(B)
            def f(z):
                return math.sqrt(2/math.pi)*math.exp(-z*z/2)/math.sqrt(1+(r*z)**2)
            alignment = h / 3 * (f(0)+f(12)+math.fsum((4 if i%2 else 2)*f(i*h) for i in range(1, panels)))
            # The squared first component has a closed-form Gaussian integral.
            second = math.sqrt(math.pi/2)/r*math.exp(1/(2*r*r))*math.erfc(1/(math.sqrt(2)*r))
            got = batch.gaussian_curvature(B)
            self.assertAlmostEqual(got["alignment"], alignment, places=10)
            self.assertAlmostEqual(got["curvature"], 1+99*second, places=9)
            optimum = alignment / (1+99*second)
            def expected_change(eta):
                return -eta*alignment + eta**2*(1+99*second)/2
            self.assertLess(expected_change(got["lr"]), expected_change(optimum*0.9))
            self.assertLess(expected_change(got["lr"]), expected_change(optimum*1.1))
        lrs = [batch.gaussian_curvature(b)["lr"] for b in (1, 10, 100)]
        self.assertGreater(lrs[1], lrs[0])
        self.assertGreater(lrs[1], lrs[2])
        self.assertAlmostEqual(batch.gaussian_curvature(1e10)["lr"], 0.01, places=7)

    def test_invalid_inputs(self):
        for call in (lambda: muon.ns_response(float("nan")), lambda: muon.ns_response(-1),
                     lambda: muon.shape_factors(0, 3), lambda: muon.polar_rms(2, 3, 3),
                     lambda: muon.convert_convention(1, 1, 0, 1),
                     lambda: muon.clip_ratio(1, 0), lambda: batch.ema_variance_factor(1),
                     lambda: batch.mean_field_lr(0, 1), lambda: batch.gaussian_curvature(1, 0),
                     lambda: batch.data_clock_decay(1, 1, 1, 4),
                     lambda: batch.noise_estimate([[1]], 1),
                     lambda: batch.noise_estimate([[1], [float("inf")]], 1)):
            with self.assertRaises(ValueError):
                call()

    @unittest.skipUnless(NUMPY, "optional independent NumPy matrix/SVD checks")
    def test_matrix_polynomial_svd_and_accumulation(self):
        import numpy as np
        rng = np.random.default_rng(20261103)
        for shape in ((3, 7), (7, 3), (4, 4)):
            A = rng.normal(size=shape)
            for original in (A, A[:, :1] @ A[:1, :], np.zeros(shape)):
                norm = np.linalg.norm(original)
                X = original / norm if norm else original.copy()
                u, s, vh = np.linalg.svd(X, full_matrices=False)
                for _ in range(5):
                    gram = X @ X.T
                    X = 3.4445*X + (-4.775*gram+2.0315*gram@gram)@X
                expected = (u * [muon.ns_response(float(x)) for x in np.clip(s, 0, 1)]) @ vh
                np.testing.assert_allclose(X, expected, atol=1e-12)
        def polar(A):
            u, s, vh = np.linalg.svd(A, full_matrices=False)
            return (u * (s > 1e-12)) @ vh
        g1, g2 = np.diag([4., 1]), np.diag([-1., 4])
        np.testing.assert_allclose(polar((g1+g2)/2), np.eye(2))
        np.testing.assert_allclose((polar(g1)+polar(g2))/2, np.diag([0, 1]))
        for A in (rng.normal(size=(7, 3)), rng.normal(size=(3, 7))):
            O = polar(A)
            np.testing.assert_allclose(O.T@O if A.shape[0]>A.shape[1] else O@O.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
