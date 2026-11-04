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
DOWNLOADED = (ROOT/"muon-download.json").is_file()


def source_path(name, bundle):
    folder = (ROOT/("optimizer" if bundle == "muon-optimizer" else "batch")
              if DOWNLOADED else ROOT/"content/posts/ai"/bundle)
    source = folder/f"{name}.py"
    if not source.is_file():
        raise SystemExit("Missing local Muon example source: " + str(source))
    return source


def load(name, bundle):
    spec = importlib.util.spec_from_file_location(name, source_path(name, bundle))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


muon = load("muon_math", "muon-optimizer")
batch = load("batch_math", "muon-learning-rate-batch-size")
# A partial download must not silently pass by skipping its optional files.
source_path("matrix_update", "muon-optimizer")
source_path("toy_training", "muon-optimizer")
source_path("gradient_noise", "muon-optimizer")
source_path("mean_field_probe", "muon-learning-rate-batch-size")
NUMPY = "--numpy" in sys.argv
if NUMPY:
    sys.argv.remove("--numpy")


class MuonArticleTests(unittest.TestCase):
    def test_saved_gradient_record_without_ml_imports(self):
        import tempfile
        noise = load("gradient_noise", "muon-optimizer")
        record = {
            "format":"muon-gradient-records-v1", "sampling":"iid-with-replacement",
            "layer":"constructed.matrix", "torch":"not-used-in-standard-library-fixture",
            "parameter_shape":[2,1], "dataset_rows":32, "batch_size":8, "batch_count":4,
            "parameter_sha256_before":"0"*64, "parameter_sha256_after":"0"*64,
            "gradients":[[3,1],[1,3],[3,3],[1,1]],
            "sampled_indices":[list(range(8*k,8*k+8)) for k in range(4)],
        }
        result = noise.summarize(record)
        self.assertAlmostEqual(result["observed_batch_gradient_variance"],8/3)
        self.assertAlmostEqual(result["observed_mean_gradient_power"],8)
        self.assertAlmostEqual(result["estimate"]["signal"],22/3)
        self.assertAlmostEqual(result["estimate"]["single_sample_noise"],64/3)
        self.assertAlmostEqual(result["estimate"]["critical_batch_estimate"],32/11)
        with tempfile.TemporaryDirectory(prefix="muon-gradient-record-") as folder:
            target = Path(folder)/"record.json"
            noise.save_records(target,record)
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                noise.save_records(target,record)
            self.assertEqual(target.read_bytes(),before)

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

    def test_normalization_conventions(self):
        self.assertEqual(muon.normalization_gain(1e-7, 1e-7, "add"), 0.5)
        self.assertEqual(muon.normalization_gain(1e-7, 1e-7, "clamp"), 1)
        for name in ("add", "clamp"):
            self.assertEqual(muon.normalization_gain(0, 0, name), 0)
            self.assertEqual(muon.normalization_gain(7, 0, name), 1)
        self.assertEqual(muon.normalization_gain(1e308, 1e308, "add"), 0.5)
        for norm, eps, name in ((-1, 0, "add"), (1, -1, "add"), (1, 1, "other")):
            with self.assertRaises(ValueError):
                muon.normalization_gain(norm, eps, name)

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
        self.assertEqual(batch.mean_field_lr(7, 0, 0.3), 0.3)
        self.assertAlmostEqual(batch.mean_field_lr(1e308, 1e308), 1/math.sqrt(2))
        self.assertGreater(batch.mean_field_lr(1e-300, 1e300), 0)
        self.assertEqual(batch.gaussian_sign_lr(7, 0, 0.3), 0.3)
        self.assertEqual(batch.gaussian_curvature(7, 0), {"alignment": 1.0, "curvature": 100.0, "lr": 0.01})

    def test_exact_gaussian_sign_integral(self):
        # Integrate the scalar Gaussian density independently of erf().
        panels = 10000
        for ratio in (0.01, 0.1, 1, 10):
            endpoint = math.sqrt(ratio)
            h = endpoint / panels
            def density(z):
                return math.exp(-z*z/2) / math.sqrt(2*math.pi)
            integral = 2*h/3*(density(0)+density(endpoint)+math.fsum(
                (4 if i%2 else 2)*density(i*h) for i in range(1, panels)))
            self.assertAlmostEqual(batch.gaussian_sign_lr(ratio, 1), integral, places=12)
        self.assertNotAlmostEqual(batch.gaussian_sign_lr(1, 1), batch.mean_field_lr(1, 1), places=2)

    def test_effective_counts_from_independent_sign_enumeration(self):
        # Weighted sum of four iid +/-1 noise values, using exact rational math.
        weights = [1, 1, 1, 3]
        values = [sum(Fraction(w*x, sum(weights)) for w, x in zip(weights, eps))
                  for eps in itertools.product((-1, 1), repeat=4)]
        variance = sum(x*x for x in values)/len(values)
        self.assertAlmostEqual(batch.independent_weighted_batch(weights), float(1/variance))
        self.assertAlmostEqual(batch.independent_weighted_batch([10*w for w in weights]), 3)
        self.assertAlmostEqual(batch.independent_weighted_batch([1e300*w for w in weights]), 3)
        self.assertEqual(batch.equicorrelated_batch(16, 0.2), 4)
        self.assertEqual(batch.equicorrelated_batch(16, 0), 16)
        self.assertEqual(batch.equicorrelated_batch(16, 1), 1)

    def test_local_budget_score_by_completed_square(self):
        for B in (256, 1024, 4096):
            # In a quadratic with b=1 and a^2=B/(B+Bc), the best decrease is a^2/2.
            a = math.sqrt(B/(B+4096))
            expected_gain = -(-a*a + 0.5*a*a)
            scores = batch.local_budget_scores(B, 4096, 256)
            self.assertAlmostEqual(scores["per_update"], 2*expected_gain)
            self.assertAlmostEqual(scores["per_cost"], 2*expected_gain/(B+256))
        peak = batch.local_budget_scores(1024, 4096, 256)["per_cost"]
        for B in (512, 2048):
            self.assertGreater(peak, batch.local_budget_scores(B, 4096, 256)["per_cost"])

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
                     lambda: batch.mean_field_lr(0, 1), lambda: batch.gaussian_curvature(1, -1),
                     lambda: batch.mean_field_lr(1, -1), lambda: batch.independent_weighted_batch([0, 0]),
                     lambda: batch.equicorrelated_batch(0, 0.1),
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


@unittest.skipUnless(NUMPY, "optional NumPy matrix update reference")
class MatrixUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np
        import numpy as np
        cls.ref = load("matrix_update", "muon-optimizer")

    def test_rotated_directions_and_rank_selection(self):
        G = np.array([[5.5, 4.5], [4.5, 5.5]])
        np.testing.assert_allclose(self.ref.polar(G), np.eye(2), atol=1e-14)
        np.testing.assert_array_equal(np.sign(G), np.ones((2, 2)))
        np.testing.assert_allclose(np.linalg.svd(G, compute_uv=False), [10, 1])
        # Rank tolerance is explicit and can change the chosen numerical rank.
        diagonal = np.diag([1, 1e-16])
        np.testing.assert_allclose(self.ref.polar(diagonal), np.diag([1, 0]))
        np.testing.assert_allclose(self.ref.polar(diagonal, relative_cutoff=0), np.eye(2))
        np.testing.assert_array_equal(self.ref.polar(np.zeros((3, 2))), np.zeros((3, 2)))

    def test_ns_scalar_oracle_and_transpose(self):
        rng = np.random.default_rng(507)
        for shape in ((3, 7), (7, 3), (4, 4)):
            G = rng.normal(size=shape)
            for norm_kind in ("add", "clamp"):
                epsilon = np.linalg.norm(G)  # maximally exposes the denominator difference.
                O = self.ref.fast_ns(G, eps=epsilon, convention=norm_kind)
                u, s, vh = np.linalg.svd(G, full_matrices=False)
                denominator = np.linalg.norm(G) + epsilon if norm_kind == "add" else max(np.linalg.norm(G), epsilon)
                expected = (u * [muon.ns_response(float(z / denominator)) for z in s]) @ vh
                np.testing.assert_allclose(O, expected, atol=1e-12)
                np.testing.assert_allclose(O, self.ref.fast_ns(G.T, eps=epsilon, convention=norm_kind).T, atol=1e-12)
            np.testing.assert_allclose(self.ref.fast_ns(G, eps=0), self.ref.fast_ns(G*3.7, eps=0), atol=1e-12)

    def test_normalization_without_squaring_extreme_entries(self):
        for scale in (1e-250, 1e250, 1e308):
            matrix = np.ones((2, 2))*scale
            np.testing.assert_allclose(self.ref.normalized(matrix, eps=0), np.full((2, 2), 0.5), atol=1e-15)
            np.testing.assert_allclose(self.ref.normalized(matrix, eps=scale, convention="add"),
                                       np.full((2, 2), 1/3), atol=1e-15)
            np.testing.assert_allclose(self.ref.fast_ns(matrix, eps=0),
                                       self.ref.fast_ns(np.ones((2, 2)), eps=0), atol=1e-12)
        small = np.eye(2)*1e-250
        np.testing.assert_allclose(self.ref.normalized(small, eps=1e-7), small/1e-7, rtol=1e-14, atol=0)
        for epsilon, convention in ((-1, "clamp"), (float("nan"), "add"), (0, "other")):
            with self.assertRaises(ValueError):
                self.ref.normalized(np.eye(2), eps=epsilon, convention=convention)

    def test_instantaneous_shampoo_identity_and_history(self):
        def inverse_quarter(gram):
            values, basis = np.linalg.eigh(gram)
            keep = values > max(gram.shape)*np.finfo(float).eps*max(values)
            weights = np.zeros_like(values)
            weights[keep] = values[keep]**(-0.25)
            return (basis*weights)@basis.T
        for gradient in (np.array([[5.5,4.5],[4.5,5.5]]),
                         np.array([[1.,2.],[3.,-1.]]),
                         np.array([[2.,0.],[0.,1.],[0.,0.]]),
                         np.ones((2,2))):
            left, right = gradient@gradient.T, gradient.T@gradient
            direction = inverse_quarter(left)@gradient@inverse_quarter(right)
            np.testing.assert_allclose(direction,self.ref.polar(gradient),atol=1e-12)
        current, previous = np.diag([2.,1.]), np.diag([0.,3.])
        left = current@current.T+previous@previous.T
        right = current.T@current+previous.T@previous
        direction = inverse_quarter(left)@current@inverse_quarter(right)
        np.testing.assert_allclose(direction,np.diag([1.,1/math.sqrt(10)]),atol=1e-14)
        self.assertGreater(np.linalg.norm(direction-self.ref.polar(current)),0.5)

    def test_stateful_diagonal_update_and_input_preservation(self):
        W, M, G = np.diag([2., 3.]), np.diag([0.4, -0.2]), np.diag([1., 2.])
        copies = [a.copy() for a in (W, M, G)]
        trace = self.ref.step(W, M, G, lr=0.1, decay=0.2, beta=0.5, transform="polar")
        np.testing.assert_allclose(trace["buffer"], np.diag([0.7, 0.9]))
        np.testing.assert_allclose(trace["transform_input"], np.diag([0.85, 1.45]))
        np.testing.assert_allclose(trace["transform_output"], np.eye(2))
        expected = np.diag([1.96, 2.94]) - 0.1*0.2*math.sqrt(2)*np.eye(2)
        np.testing.assert_allclose(trace["weight"], expected, atol=1e-14)
        for value, original in zip((W, M, G), copies):
            np.testing.assert_array_equal(value, original)
        zero = self.ref.step(trace["weight"], trace["buffer"], np.zeros_like(G),
                             lr=0.1, decay=0.2, beta=0.5, transform="polar")
        np.testing.assert_allclose(zero["buffer"], np.diag([0.35, 0.45]))
        np.testing.assert_allclose(zero["transform_output"], np.eye(2))
        self.assertGreater(np.linalg.norm(zero["gradient_delta"]), 0)
        missing = self.ref.step(trace["weight"], trace["buffer"], None)
        self.assertTrue(missing["skipped"])
        np.testing.assert_array_equal(missing["weight"], trace["weight"])
        np.testing.assert_array_equal(missing["buffer"], trace["buffer"])

    def test_zero_buffer_decay_and_rejected_shapes(self):
        W = np.ones((3, 2))
        for method in ("ns", "polar"):
            zero = self.ref.step(W, None, np.zeros_like(W), beta=0, lr=0.2, decay=0.1, transform=method)
            np.testing.assert_array_equal(zero["transform_output"], np.zeros_like(W))
            np.testing.assert_allclose(zero["weight"], W*0.98)
        with self.assertRaises(ValueError):
            self.ref.step(W, np.zeros((2, 3)), W)
        with self.assertRaises(ValueError):
            self.ref.step(W, None, np.zeros((2, 3)))


@unittest.skipUnless(NUMPY, "optional Gaussian matrix expectation probe")
class MeanFieldProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np
        import numpy as np
        cls.ref = load("mean_field_probe", "muon-learning-rate-batch-size")

    def test_chunked_moments_against_direct_centered_sum(self):
        values = np.arange(48,dtype=float).reshape(12,2,2)/7
        stats = self.ref.Moments()
        for group in np.array_split(values,[3,8]):
            stats.add(group)
        np.testing.assert_allclose(stats.mean,values.mean(axis=0),atol=1e-14)
        expected = np.sqrt(((values-values.mean(axis=0))**2).sum(axis=0)/(12*11))
        np.testing.assert_allclose(stats.standard_error(),expected,atol=1e-14)

    def test_noise_free_polar_and_stable_ill_conditioned_signal(self):
        with self.assertRaises(ValueError):
            self.ref.observe(np.eye(2, dtype=complex), 0, 8, np.random.default_rng(3))
        for signal in (np.diag([1.,0.1]),np.array([[0.8,-0.6],[0.6,0.8]])@np.diag([1.,1e-10])):
            result = self.ref.observe(signal,0,8,np.random.default_rng(3),chunk=3)
            u,s,vh = np.linalg.svd(signal,full_matrices=False)
            np.testing.assert_allclose(result['mean_polar'],u@vh,atol=1e-14)
            np.testing.assert_allclose(result['gram_prediction'],u@vh,atol=1e-14)
            self.assertAlmostEqual(result['eta_ratio_special_curvature'],1)
            self.assertTrue(np.isfinite(result['gram_prediction']).all())
            self.assertLess(np.max(result['standard_error']),1e-14)

    def test_seed_and_chunk_size_preserve_the_distribution(self):
        signal = np.diag([1.,0.1])
        rows = [self.ref.observe(signal,0.2,1024,np.random.default_rng(71),chunk=k) for k in (1,17,2048)]
        for row in rows:
            np.testing.assert_allclose(row['expected_gradient_gram'],np.diag([1.08,0.09]),atol=1e-14)
            for key in ('mean_polar','standard_error','observed_gradient_gram','eta_ratio_special_curvature'):
                np.testing.assert_allclose(row[key],rows[0][key],atol=1e-12)

    def test_saved_gaussian_probe_reference(self):
        path = source_path('mean_field_probe','muon-learning-rate-batch-size').parent/'assets/mean-field-probe-reference.json'
        import json
        expected = json.loads(path.read_text())
        actual = self.ref.report()
        for name,row in actual['cases'].items():
            self.assertEqual(row['draws'],200000)
            self.assertEqual(row['numerically_rank_deficient_draws'],0)
            for key in ('mean_polar','standard_error','gram_prediction','eta_ratio_special_curvature'):
                np.testing.assert_allclose(row[key],expected['cases'][name][key],rtol=1e-10,atol=1e-12)
        flat,uneven = actual['cases']['flat'],actual['cases']['uneven']
        self.assertGreater(flat['eta_ratio_special_curvature'],uneven['eta_ratio_special_curvature'])
        self.assertAlmostEqual(flat['eta_ratio_scalar_prediction'],uneven['eta_ratio_scalar_prediction'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
