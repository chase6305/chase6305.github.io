"""Offline identification examples: independent equations and data contracts.

Use --only common, pinocchio, or drake to test one dependency group.
The downloadable copy is named check_examples.py and locates its own bundle.
"""
import argparse
import ast
import contextlib
import csv
from decimal import Decimal, localcontext
import importlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUNDLE = HERE if (HERE / "two_link_id.urdf").exists() else HERE.parent / "content/posts/robotics/dynamics/parameter-identification"
sys.path.insert(0, str(BUNDLE))
from identification_common import digest, format_and_save_report, load_processed_csv, rmse, scaled_svd_fit, unseen_fraction, write_processed_csv, write_raw_csv
import preprocess_identification as preprocessing


class DataTests(unittest.TestCase):
    def test_scaled_svd_extreme_column_units_and_real_contract(self):
        base = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
        coefficients = np.array([2., -3.])
        target = base @ coefficients
        for units in ([1e-200, 1e-200], [1e200, 1e200], [1e-200, 1e200], [1e200, 1e-200]):
            with self.subTest(units=units), np.errstate(all="raise"):
                matrix = base * units
                fit = scaled_svd_fit(matrix, target)
                self.assertEqual(fit["rank"], 2)
                np.testing.assert_allclose(fit["parameters"] * units, coefficients, rtol=2e-14)
                np.testing.assert_allclose(matrix @ fit["parameters"], target, rtol=2e-14, atol=2e-14)
                np.testing.assert_allclose(fit["scale"] / units, [math.sqrt(6), math.sqrt(3)], rtol=2e-14)
                self.assertLess(unseen_fraction(matrix, fit), 2e-14)
        # Finite entries do not guarantee a representable norm or solution.
        for matrix, values in ((np.full((4, 1), 1e308), np.ones(4)),
                               (np.full((4, 1), 1e-309), np.ones(4)),
                               (base.astype(complex), target), (base, target.astype(complex))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                scaled_svd_fit(matrix, values)

    def test_rmse_requires_aligned_samples_and_avoids_square_overflow(self):
        for predicted, observed in (([0., 1.], [[0.], [1.]]), ([], []),
                                    ([math.nan], [0.]), ([0.], [math.inf]),
                                    (np.array([1.j]), np.array([0.j])), ([1.e308], [-1.e308])):
            with self.subTest(predicted=predicted, observed=observed), self.assertRaises(ValueError):
                rmse(predicted, observed)
        self.assertEqual(rmse([1., 2.], [1., 2.]), 0.)
        self.assertAlmostEqual(rmse(np.arange(6.).reshape(3, 2), np.zeros((3, 2))), math.sqrt(55/6), places=14)
        with localcontext() as context:
            context.prec = 70
            factor = (Decimal(14)/3).sqrt()
            for scale in (1.e-200, 1.e200):
                expected = float(factor * Decimal(str(scale)))
                actual = rmse([scale, 2*scale, 3*scale], [0., 0., 0.])
                self.assertTrue(math.isfinite(actual))
                self.assertAlmostEqual(actual / expected, 1., places=14)

    def test_closed_form_timing_counterexample(self):
        import timing_counterexample as timing
        reference = json.loads((BUNDLE / "assets/timing-reference.json").read_text())
        result = timing.run()
        for key in ("mass_kg", "damping_n_s_per_m", "validation_reference_rmse_n", "validation_logged_rmse_n"):
            self.assertAlmostEqual(result[key], reference[key], places=12)
        self.assertLess(result["train_rmse_n"], 1e-12)
        corrected = timing.run(correct_training_delay=True)
        self.assertLess(corrected["validation_reference_rmse_n"], 1e-12)
        self.assertGreater(corrected["validation_logged_rmse_n"], .27)
        same = timing.run(validation_hz=.7)
        self.assertLess(same["validation_logged_rmse_n"], 1e-12)
        self.assertGreater(same["validation_reference_rmse_n"], .01)
        for options in ({"delay_ms": True}, {"delay_ms": math.nan}, {"train_hz": 0},
                        {"validation_hz": math.inf}, {"correct_training_delay": "false"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                timing.run(**options)

    def test_report_cannot_replace_model_data_or_file_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory, "model.urdf")
            original.write_text("model input")
            symbolic, hard = Path(directory, "alias.json"), Path(directory, "hard.json")
            symbolic.symlink_to(original)
            os.link(original, hard)
            for target in (original, symbolic, hard):
                with self.assertRaises(ValueError):
                    format_and_save_report({"test": 1}, target, (None, original))
                self.assertEqual(original.read_text(), "model input")
            result = Path(directory, "results/report.json")
            text = format_and_save_report({"test": 1}, result, (original,))
            self.assertEqual(result.read_text(), text)
            with self.assertRaises(ValueError):
                format_and_save_report({"bad": float("nan")}, result)
            self.assertEqual(result.read_text(), text)

    def test_csv_joint_names_reorder_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "data.csv")
            t = np.arange(3.)
            q = np.array([[1., 2.], [3., 4.], [5., 6.]])
            write_processed_csv(path, ["b", "a"], t, q, q + 10, q + 20, q + 30)
            actual = load_processed_csv(path, ["a", "b"])
            np.testing.assert_array_equal(actual[0], t)
            for offset, values in zip((0, 10, 20, 30), actual[1:]):
                np.testing.assert_array_equal(values, (q + offset)[:, ::-1])
            bom = Path(directory, "data-bom.csv")
            bom.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
            for plain, marked in zip(actual, load_processed_csv(bom, ["a", "b"])):
                np.testing.assert_array_equal(plain, marked)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_processed_csv(path, ["a", "b"], t, q, q, q, q)
            self.assertEqual(path.read_bytes(), original)

    def test_csv_rejects_bad_names_values_and_time(self):
        header = ["time_s", "q/j", "dq/j", "ddq/j", "tau/j"]
        good = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
        cases = [([*header, "q/j"], [row + [0] for row in good]),
                 ([*header, "q/other"], [row + [0] for row in good]),
                 (header[:-1], [row[:-1] for row in good]),
                 (header, []), (header, good[:1]),
                 (header, [good[0], [0, 2, 3, 4, 5]]),
                 (header, [good[0], [-1, 2, 3, 4, 5]]),
                 (header, [good[0], [1, 2, 3, 4]]),
                 (header, [good[0], [1, 2, 3, 4, 5, 6]])]
        for value in ("nan", "inf", "-inf", "", "bad"):
            for column in range(5):
                row = good[1].copy()
                row[column] = value
                cases.append((header, [good[0], row]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bad.csv")
            for fields, rows in cases:
                with self.subTest(fields=fields, rows=rows):
                    with path.open("w", newline="") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(fields)
                        writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        load_processed_csv(path, ["j"])
            for names in (None, "j", [""], [" "], [1], ["j", "j"]):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    load_processed_csv(path, names)

    def test_export_validates_schema_before_creating_a_file(self):
        times = np.arange(3.)
        values = np.ones((3, 1))
        cases = [([""], times, [values]*4), (["j", "j"], times, [values]*4),
                 (["j"], times, [np.ones((3, 2))]*4),
                 (["j"], times, [np.ones((2, 1)), values, values, values]),
                 (["j"], times, [values, values, values, np.full((3, 1), math.nan)]),
                 (["j"], [0., 0., 1.], [values]*4),
                 (["j"], [0., math.inf, 2.], [values]*4),
                 (["j"], times.astype(complex), [values]*4),
                 (["j"], times, [values, values, values, values.astype(complex)]),
                 (["j"], [], [np.ones((0, 1))]*4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "export.csv")
            for names, time, signals in cases:
                with self.subTest(names=names, times=time), self.assertRaises(ValueError):
                    write_processed_csv(path, names, time, *signals)
                self.assertFalse(path.exists(), "Invalid export left a partial CSV")
            write_processed_csv(path, ["j"], times, *(np.arange(3.) for _ in range(4)))
            self.assertEqual(load_processed_csv(path, ["j"])[1].shape, (3, 1))

    def test_scaled_svd_rank_prediction_and_new_directions(self):
        x = np.linspace(-2., 2., 31)
        matrix = np.column_stack((x, 1e8 * x, np.zeros_like(x)))
        fit = scaled_svd_fit(matrix, 3 * x)
        self.assertEqual(fit["rank"], 1)
        self.assertEqual(fit["raw_column_l2_norms"][2], 0.)
        self.assertEqual(fit["scale"][2], 1.)
        np.testing.assert_allclose(matrix @ fit["parameters"], 3 * x, atol=1e-13)
        self.assertLess(unseen_fraction(matrix, fit), 1e-14)
        self.assertGreater(unseen_fraction(np.array([[0., 0., 1.]]), fit), .99)
        # Scaled-coordinate minimum norm splits the two equal scaled columns.
        np.testing.assert_allclose(fit["parameters"][:2], [1.5, 1.5e-8], atol=1e-13)
        for matrix, target in [(np.zeros((3, 2)), np.ones(3)),
                               (np.array([[np.nan, 1.]]), np.ones(1)),
                               (np.ones((3, 2)), np.ones(2))]:
            with self.assertRaises(ValueError):
                scaled_svd_fit(matrix, target)


class PreprocessTests(unittest.TestCase):
    def test_irregular_polynomial_derivatives_and_time_origin(self):
        rng = np.random.default_rng(20261106)
        for offset in (0., 10000.):
            times = offset + np.cumsum(rng.uniform(.007, .018, 120))
            relative = times - times[0]
            coefficients = np.array([[.1, .2], [.2, -.1], [-.3, .2], [.04, -.03], [.05, .02], [-.02, .01]])
            q = np.polynomial.polynomial.polyval(relative, coefficients).T
            indices, estimated_q, estimated_v, estimated_a, details = preprocessing.local_polynomial_states(times, q, 13, 5)
            self.assertEqual(len(indices), len(times) - 12)
            self.assertEqual(indices[0], 6)
            self.assertEqual(indices[-1], len(times) - 7)
            for derivative, actual in enumerate((estimated_q, estimated_v, estimated_a)):
                expected = np.polynomial.polynomial.polyval(relative[indices],
                    np.polynomial.polynomial.polyder(coefficients, m=derivative)).T
                np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
            self.assertGreater(details["window_span_max_s"], details["window_span_min_s"])

    def test_signed_delay_and_no_endpoint_extrapolation(self):
        times = np.arange(61) / 8
        positions = (times**2)[:, None]
        for delay in (-.9375, -.875, -.3125, 0., .3125, .875, .9375):
            logged = np.column_stack((3 * (times - delay) + 2, -2 * (times - delay) + 1))
            q = np.column_stack((positions, .5 * positions))
            values, report = preprocessing.preprocess_arrays(times, q, logged, 7, 2, delay, .126)
            t, fitted, v, a, aligned = values
            expected_indices = np.arange(3, 58)
            valid = (times[expected_indices] + delay >= times[0]) & (times[expected_indices] + delay <= times[-1])
            np.testing.assert_array_equal(t, times[expected_indices[valid]])
            np.testing.assert_allclose(aligned, np.column_stack((3 * t + 2, -2 * t + 1)), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(v, np.column_stack((2 * t, t)), atol=1e-10)
            np.testing.assert_allclose(a, np.tile([2., 1.], (len(t), 1)), atol=1e-10)
            self.assertEqual(report["retained_samples"], len(t))
        irregular = np.cumsum(np.random.default_rng(20261030).uniform(.07, .18, 80))
        for delay in (-.13, .02, .13):
            logged = np.column_stack((3 * (irregular - delay) + 2, -2 * (irregular - delay) + 1))
            q = np.column_stack((irregular**2, .5 * irregular**2))
            (t, _, v, a, aligned), _ = preprocessing.preprocess_arrays(irregular, q, logged, 7, 2, delay)
            np.testing.assert_allclose(aligned, np.column_stack((3 * t + 2, -2 * t + 1)), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(v, np.column_stack((2 * t, t)), atol=1e-10)
            np.testing.assert_allclose(a, np.tile([2., 1.], (len(t), 1)), atol=1e-10)
        for delay in (-100., 100.):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(times, positions, positions, 7, 2, delay)

    def test_preprocessing_rejects_bad_windows_gaps_and_states(self):
        times = np.arange(30.) * .01
        q = np.column_stack((times**2, times))
        for window, degree in ((2, 2), (6, 3), (True, 2), (7, 1), (7, 7), (7, True), (7., 3), (7, 3.)):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(times, q, q, window, degree)
        for delay in (True, "0.01", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(times, q, q, 7, 3, delay)
        for gap in (0, -1, True, float("nan"), float("inf"), "0.02", .009):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(times, q, q, 7, 3, max_gap=gap)
        for signal in (0, 1, 2):
            values = [times.copy(), q.copy(), q.copy()]
            values[signal].flat[4] = float("nan")
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(*values, 7, 3)
        for modified in (times[::-1], np.zeros_like(times)):
            with self.assertRaises(ValueError):
                preprocessing.preprocess_arrays(modified, q, q, 7, 3)
        with self.assertRaises(ValueError):
            preprocessing.preprocess_arrays(times[:7], q[:7], q[:7], 7, 3)
        with self.assertRaises(ValueError):
            preprocessing.preprocess_arrays(times, q, q[:, :1], 7, 3)

    def test_preprocessed_files_hashes_names_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.csv", root / "processed.csv"
            times = np.arange(31.) / 8
            q = np.column_stack((times**2, 2 * times**2))
            write_raw_csv(source, ["b", "a"], times, q, 3 * q)
            with source.open(newline="") as stream:
                rows = list(csv.reader(stream))
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(rows[0] + ["dq/a", "ddq/a", "note"])
                writer.writerows(row + [9999, -9999, "unused"] for row in rows[1:])
            original = source.read_bytes()
            report = preprocessing.process_file(source, output, ["a", "b"], 7, 3, max_gap=.126)
            values = load_processed_csv(output, ["a", "b"])
            np.testing.assert_allclose(values[1], q[3:-3, ::-1], atol=1e-12)
            self.assertEqual(report["input_sha256"], digest(source))
            self.assertEqual(report["output_sha256"], digest(output))
            self.assertEqual(report["algorithm_sha256"], digest(BUNDLE / "preprocess_identification.py"))
            self.assertEqual(report["helper_sha256"], digest(BUNDLE / "identification_common.py"))
            metadata = output.with_suffix(".meta.json")
            self.assertEqual(json.loads(metadata.read_text()), report)
            before = source.read_bytes(), output.read_bytes(), metadata.read_bytes()
            for target in (source, output):
                with self.assertRaises(ValueError):
                    preprocessing.process_file(source, target, ["a", "b"], 7, 3)
            self.assertEqual(before, (source.read_bytes(), output.read_bytes(), metadata.read_bytes()))
            self.assertEqual(original, source.read_bytes())


class PinocchioTests(unittest.TestCase):
    def test_window_comparison_keeps_common_train_and_validation_times(self):
        import preprocessing_study
        result = preprocessing_study.run()
        self.assertEqual(result["common_support"]["train"]["samples"], 321)
        self.assertEqual(result["common_support"]["validation"]["samples"], 221)
        self.assertEqual(result["common_support"]["train"]["scalar_equations"], 642)
        self.assertEqual(result["common_support"]["validation"]["scalar_equations"], 442)
        reference = json.loads((BUNDLE / "assets/preprocessing-study.json").read_text())
        self.assertEqual(len(result["results"]), 5)
        self.assertEqual(len(reference["results"]), 5)
        for row, expected in zip(result["results"], reference["results"]):
            self.assertEqual(row["retained_rank"], 6)
            for key, value in row.items():
                self.assertTrue(math.isclose(value, expected[key], rel_tol=1e-8, abs_tol=1e-10), (key,value,expected[key]))
            self.assertGreater(row["validation_window_span_s"], row["train_window_span_s"])
        by_setting = {(r["window_samples"], r["polynomial_degree"]): r for r in result["results"]}
        narrow, broad = by_setting[21, 5], by_setting[81, 5]
        self.assertGreater(broad["validation_acceleration_rmse_rad_s2"], 50 * narrow["validation_acceleration_rmse_rad_s2"])
        self.assertLess(broad["validation_measured_torque_rmse_nm"], .023)

    @classmethod
    def setUpClass(cls):
        cls.lab = importlib.import_module("pinocchio_identification")
        cls.model = cls.lab.build_model(BUNDLE / "two_link_id.urdf")

    def test_regressor_against_independent_two_link_equations(self):
        rng = np.random.default_rng(20261101)
        q, v, a = rng.uniform(-2, 2, (200, 2)), rng.uniform(-3, 3, (200, 2)), rng.uniform(-10, 10, (200, 2))
        q1, q2 = q.T
        v1, v2 = v.T
        a1, a2 = a.T
        m1, m2, l1, c1, c2, j1, j2, g = 1.2, .8, .6, .3, .225, .03627, .01368, 9.81
        m11 = j1 + j2 + m1 * c1**2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2 * np.cos(q2))
        m12 = j2 + m2 * (c2**2 + l1 * c2 * np.cos(q2))
        m22 = j2 + m2 * c2**2
        coupling = m2 * l1 * c2 * np.sin(q2)
        tau1 = m11 * a1 + m12 * a2 - coupling * (2 * v1 * v2 + v2**2) - g * (
            (m1 * c1 + m2 * l1) * np.cos(q1) + m2 * c2 * np.cos(q1 + q2))
        tau2 = m12 * a1 + m22 * a2 + coupling * v1**2 - g * m2 * c2 * np.cos(q1 + q2)
        matrix = self.lab.regressors(self.model, q, v, a)
        theta = np.concatenate([inertia.toDynamicParameters() for inertia in self.model.inertias[1:]])
        np.testing.assert_allclose((matrix @ theta).reshape(-1, 2), np.column_stack((tau1, tau2)), rtol=1e-12, atol=2e-13)

    def test_gravity_root_frame_against_independent_potential_gradient(self):
        q = np.random.default_rng(20261030).uniform(-2., 2., (100, 2))
        for gravity in ([3.2, 1.7, -8.4], [0., 0., 0.], [-9.81, 0., 0.]):
            model = self.lab.build_model(BUNDLE / "two_link_id.urdf", gravity)
            q1, q2 = q.T
            gx, _, gz = gravity
            proximal = (1.2 * .3 + .8 * .6) * (gx * np.sin(q1) + gz * np.cos(q1))
            distal = .8 * .225 * (gx * np.sin(q1 + q2) + gz * np.cos(q1 + q2))
            matrix = self.lab.regressors(model, q, np.zeros_like(q), np.zeros_like(q))
            theta = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
            np.testing.assert_allclose((matrix @ theta).reshape(-1, 2), np.column_stack((proximal + distal, distal)),
                                       rtol=1e-12, atol=2e-13)
            np.testing.assert_array_equal(model.gravity.linear, gravity)
        np.testing.assert_array_equal(self.model.gravity.linear, [0., 0., -9.81])
        for invalid in ([0., 0.], [0., 0., 0., 1.], [0., 0., math.nan], [math.inf, 0., 0.],
                        [True, 0., -9.81], [10**1000, 0., 0.], [[0.], [0.], [-9.81]]):
            with self.subTest(gravity=invalid), self.assertRaises(ValueError):
                self.lab.build_model(BUNDLE / "two_link_id.urdf", invalid)
        result = self.lab.run(BUNDLE / "two_link_id.urdf", gravity=[3.2, 1.7, -8.4])
        self.assertEqual(result["gravity_m_s2"], [3.2, 1.7, -8.4])
        self.assertLess(result["validation_measured_rmse_nm"], .023)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "gravity.json")
            command = [sys.executable, "-B", str(BUNDLE / "pinocchio_identification.py"),
                       "--gravity", "3.2", "1.7", "-8.4", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, timeout=20)
            self.assertEqual(json.loads(output.read_text())["gravity_m_s2"], [3.2, 1.7, -8.4])

    def test_regressor_and_rnea_actuator_terms_are_distinct(self):
        import pinocchio as pin
        model = pin.Model(self.model)
        q, v, a = np.random.default_rng(20261031).uniform(-1., 1., (3, 50, 2))
        before = self.lab.regressors(model, q, v, a)
        model.armature = np.array([.3, .2])
        model.friction = np.array([.4, .1])
        model.damping = np.array([.2, .3])
        after = self.lab.regressors(model, q, v, a)
        np.testing.assert_array_equal(after, before)
        theta = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
        data = model.createData()
        rnea = np.vstack([pin.rnea(model, data, qi, vi, ai).copy() for qi, vi, ai in zip(q, v, a)])
        # Pinned Pinocchio 4.1 RNEA adds armature*a, but does not automatically
        # add the model's friction/damping fields. The inertial regressor does neither.
        np.testing.assert_allclose(rnea, (after @ theta).reshape(-1, 2) + a * model.armature,
                                   rtol=1e-12, atol=2e-13)
        np.testing.assert_array_equal(self.model.armature, [0., 0.])

    def test_synthetic_fit_derivatives_and_poor_excitation(self):
        truth_model = self.lab.synthetic_truth_model(self.model)
        np.testing.assert_allclose([inertia.mass for inertia in self.model.inertias[1:]], [1.2, .8])
        np.testing.assert_allclose([inertia.mass for inertia in truth_model.inertias[1:]], [1.38, .72])
        for inertia in truth_model.inertias[1:]:
            moments = np.linalg.eigvalsh(inertia.inertia)
            self.assertGreater(inertia.mass, 0)
            self.assertGreater(moments[0], 0)
            self.assertLess(moments[-1], sum(moments[:-1]))
        result = self.lab.run(BUNDLE / "two_link_id.urdf")
        self.assertEqual(result["retained_rank"], 6)
        self.assertEqual(result["unobserved_directions"], 14)
        self.assertLess(result["regressor_rnea_max_error_nm"], 1e-12)
        self.assertLess(result["validation_noise_free_rmse_nm"], .004)
        self.assertTrue(.018 < result["validation_measured_rmse_nm"] < .023)
        self.assertGreater(result["nominal_urdf_validation_measured_rmse_nm"], .18)
        self.assertLess(max(result["per_joint_validation_rmse_nm"]), .023)
        self.assertLess(max(result["state_derivative_finite_difference_max_errors"].values()), 1e-7)
        _, q, v, a, _, _ = self.lab.synthetic_data(self.model, validation=True)
        static = self.lab.regressors(self.model, np.tile(q[0], (20, 1)), np.zeros((20, 2)), np.zeros((20, 2)))
        poor = scaled_svd_fit(static, np.ones(len(static)))
        self.assertLess(poor["rank"], result["retained_rank"])
        self.assertGreater(unseen_fraction(self.lab.regressors(self.model, q, v, a), poor), .1)

    def test_csv_replay_independence_and_preserved_export(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.lab.run(BUNDLE / "two_link_id.urdf", csv_dir=directory)
            train, validation = Path(directory, "train.csv"), Path(directory, "validation.csv")
            second = self.lab.run(BUNDLE / "two_link_id.urdf", train, validation)
            for key in ("retained_rank", "train_measured_rmse_nm", "validation_measured_rmse_nm"):
                self.assertEqual(first[key], second[key])
            # A known held-out sensor bias must move observed-minus-predicted
            # mean residuals in the same direction, without refitting training.
            times, q, v, a, torque = load_processed_csv(validation, ["joint1", "joint2"])
            biased = Path(directory, "biased-validation.csv")
            bias = np.array([.25, -.1])
            write_processed_csv(biased, ["joint1", "joint2"], times, q, v, a, torque + bias)
            shifted = self.lab.run(BUNDLE / "two_link_id.urdf", train, biased)
            np.testing.assert_allclose(np.asarray(shifted["per_joint_validation_mean_residual_nm"]) -
                second["per_joint_validation_mean_residual_nm"], bias, atol=1e-14)
            self.assertEqual(shifted["prediction_parameter_vector"], second["prediction_parameter_vector"])
            self.assertEqual(shifted["residual_definition"], "measured torque minus predicted torque")
            original = train.read_bytes(), validation.read_bytes()
            with self.assertRaises(ValueError):
                self.lab.run(BUNDLE / "two_link_id.urdf", csv_dir=directory)
            self.assertEqual(original, (train.read_bytes(), validation.read_bytes()))
            with self.assertRaises(ValueError):
                self.lab.run(BUNDLE / "two_link_id.urdf", train, train)
            with self.assertRaises(ValueError):
                self.lab.run(BUNDLE / "two_link_id.urdf", train, None)

    def test_model_adapter_rejects_continuous_and_prismatic(self):
        original = (BUNDLE / "two_link_id.urdf").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "unsupported.urdf")
            for kind in ("continuous", "prismatic"):
                path.write_text(original.replace('type="revolute"', f'type="{kind}"'))
                with self.assertRaises(ValueError):
                    self.lab.build_model(path)

    def test_raw_position_torque_pipeline_and_independent_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo = preprocessing.make_demo(root)
            self.assertEqual(demo["data_kind"], "constructed_position_and_joint_torque")
            outputs = []
            for name, count in (("train", 381), ("validation", 281)):
                output = root / (name + ".csv")
                report = preprocessing.process_file(root / (name + "-raw.csv"), output,
                    ["joint1", "joint2"], 21, 5, max_gap=.03)
                self.assertEqual(report["retained_samples"], count)
                outputs.append(output)
            result = self.lab.run(BUNDLE / "two_link_id.urdf", *outputs)
            self.assertEqual(result["retained_rank"], 6)
            self.assertLess(result["validation_measured_rmse_nm"], .025)
            self.assertGreater(result["nominal_urdf_validation_measured_rmse_nm"], .18)
            for output in outputs:
                self.assertEqual(result["input_sha256"]["train" if output.stem == "train" else "validation"], digest(output))
            original = {path.name: path.read_bytes() for path in root.iterdir()}
            with self.assertRaises(ValueError):
                preprocessing.make_demo(root)
            self.assertEqual(original, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_cli_preserves_model_and_exported_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("identification_common.py", "pinocchio_identification.py", "two_link_id.urdf"):
                shutil.copy2(BUNDLE / name, root / name)
            model = root / "two_link_id.urdf"
            original = model.read_bytes()
            command = [sys.executable, "-B", str(root / "pinocchio_identification.py")]
            collision = subprocess.run(command + ["--output", str(model)], capture_output=True, text=True, timeout=20)
            self.assertEqual(collision.returncode, 2)
            self.assertIn("must not replace a model or input data file", collision.stderr)
            self.assertEqual(model.read_bytes(), original)
            exported = root / "exported"
            collision = subprocess.run(command + ["--csv-dir", str(exported), "--output", str(exported / "train.csv")],
                capture_output=True, text=True, timeout=20)
            self.assertEqual(collision.returncode, 2)
            self.assertIn("must not replace a model or input data file", collision.stderr)
            self.assertEqual(len(load_processed_csv(exported / "train.csv", ["joint1", "joint2"])[0]), 401)
            self.assertEqual(len(load_processed_csv(exported / "validation.csv", ["joint1", "joint2"])[0]), 301)


class DrakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = importlib.import_module("drake_identification")
        cls.plant = cls.lab.build_plant()
        cls.symbols = cls.lab.symbolic_regressor(cls.plant)

    def test_symbolic_and_numeric_dynamics_against_closed_form(self):
        rng = np.random.default_rng(20261102)
        q, v, a = [rng.uniform(-2, 2, (200, 1)) for _ in range(3)]
        matrix = self.lab.regressors(self.symbols, q, v, a)
        expected = np.column_stack((a[:, 0], v[:, 0], 9.81 * np.sin(q[:, 0])))
        np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-14)
        closed = .67 * a + .08 * v + 9.81 * np.sin(q)
        np.testing.assert_allclose(self.lab.numerical_torques(self.plant, q, v, a), closed, rtol=1e-12, atol=3e-14)
        for m, ell, central in ((2., .5, .17), (2.5, .4, .27)):
            parameters = dict(zip(self.symbols["physical"], (m, ell, central / m, .08)))
            actual = [self.symbols["torque"].Evaluate(parameters | dict(zip(self.symbols["state"], row)))
                      for row in zip(q[:, 0], v[:, 0], a[:, 0])]
            np.testing.assert_allclose(actual, closed[:, 0], rtol=1e-12, atol=3e-14)

    def test_held_out_fit_and_csv_replay(self):
        result = self.lab.run()
        self.assertEqual(result["retained_rank"], 3)
        self.assertLess(result["decomposition_max_error_nm"], 1e-12)
        self.assertLess(result["symbolic_numeric_inverse_dynamics_max_error_nm"], 1e-12)
        self.assertLess(result["validation_noise_free_rmse_nm"], .004)
        self.assertTrue(.018 < result["validation_measured_rmse_nm"] < .023)
        np.testing.assert_allclose(result["lumped_parameter_estimates"], [.67, .08, 1.], atol=.003)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, name + ".csv") for name in ("train", "validation")]
            for path, validation in zip(paths, (False, True)):
                values = self.lab.synthetic_data(self.plant, validation)
                write_processed_csv(path, ["joint1"], *values[:5])
            replay = self.lab.run(*paths)
            for key in ("retained_rank", "train_measured_rmse_nm", "validation_measured_rmse_nm"):
                self.assertEqual(result[key], replay[key])
            with self.assertRaises(ValueError):
                self.lab.run(paths[0], paths[0])
            with self.assertRaises(ValueError):
                self.lab.run(paths[0], None)


    def test_known_com_distance_recovers_only_assumed_axis_model(self):
        for distance, expected_mass, expected_inertia in ((.5, 2., .17), (.4, 2.5, .27)):
            result = self.lab.recover_axis_parameters([.67, .08, 1.], distance)
            self.assertAlmostEqual(result["mass_kg"], expected_mass)
            self.assertAlmostEqual(result["central_y_inertia_kg_m2"], expected_inertia)
            self.assertTrue(all(result["necessary_axis_checks"].values()))
        invalid = self.lab.recover_axis_parameters([.2, -.1, 1.], .5)
        self.assertAlmostEqual(invalid["central_y_inertia_kg_m2"], -.3)
        self.assertFalse(invalid["necessary_axis_checks"]["central_y_inertia_positive"])
        self.assertFalse(invalid["necessary_axis_checks"]["damping_nonnegative"])
        for distance in (0, -1, True, float("nan"), float("inf"), "0.5"):
            with self.assertRaises(ValueError):
                self.lab.run(known_com_distance=distance)
        result = self.lab.run(known_com_distance=.5)
        self.assertTrue(result["axis_model_unique_given_known_com_distance"])
        self.assertAlmostEqual(result["axis_parameters_given_known_com_distance"]["mass_kg"], 2.000363083857756, places=10)
        self.assertFalse(result["physical_parameters_uniquely_identified"])


class CrossBackendTests(unittest.TestCase):
    def test_spatial_axes_fixed_link_and_named_csv(self):
        from pinocchio_identification import build_model, regressors, run
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.multibody.tree import MultibodyForces
        robot = ET.Element("robot", name="spatial-identification")
        for index, name in enumerate(("base", "upper", "elbow", "branch", "tool")):
            inertia = ET.SubElement(ET.SubElement(robot, "link", name=name), "inertial")
            ET.SubElement(inertia, "origin", xyz=f"{.03*index} {-.015*index} {.01*index}", rpy=".2 -.3 .4")
            mass = .5 + .2 * index
            ET.SubElement(inertia, "mass", value=str(mass))
            ET.SubElement(inertia, "inertia", ixx=str(.02*mass), iyy=str(.03*mass), izz=str(.04*mass),
                          ixy="0", ixz="0", iyz="0")
        for name, kind, parent, child, xyz, rpy, axis in (
            ("shoulder", "revolute", "base", "upper", ".1 .2 .3", ".2 .1 -.3", "1 0 0"),
            ("elbow_joint", "revolute", "upper", "elbow", ".6 0 .1", "-.1 .3 .2", "0 0 1"),
            ("branch_joint", "revolute", "upper", "branch", ".2 .1 .2", ".4 -.2 .1", "1 2 2"),
            ("tool_fixed", "fixed", "elbow", "tool", ".25 -.05 .1", ".2 .4 -.1", "0 0 1")):
            joint = ET.SubElement(robot, "joint", name=name, type=kind)
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
            if kind == "revolute":
                direction = np.asarray([float(value) for value in axis.split()])
                ET.SubElement(joint, "axis", xyz=" ".join(str(value) for value in direction / np.linalg.norm(direction)))
                ET.SubElement(joint, "limit", lower="-2", upper="2", velocity="4", effort="100")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "spatial.urdf"
            ET.ElementTree(robot).write(model_path)
            model = build_model(model_path)
            names = list(model.names)[1:]
            self.assertEqual(model.nv, 3)
            self.assertEqual(len(model.inertias) - 1, 3)  # Fixed tool merges into its parent joint body.
            plant = MultibodyPlant(0.)
            Parser(plant).AddModels(model_path)
            plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
            plant.Finalize()
            positions = [plant.GetJointByName(name).position_start() for name in names]
            velocities = [plant.GetJointByName(name).velocity_start() for name in names]
            context, forces = plant.CreateDefaultContext(), MultibodyForces(plant)
            q, v, a = np.random.default_rng(20261106).uniform(-1, 1, (3, 400, 3))
            expected = []
            for qi, vi, ai in zip(q, v, a):
                qd, vd, ad = np.zeros(3), np.zeros(3), np.zeros(3)
                qd[positions], vd[velocities], ad[velocities] = qi, vi, ai
                plant.SetPositions(context, qd)
                plant.SetVelocities(context, vd)
                plant.CalcForceElementsContribution(context, forces)
                expected.append(plant.CalcInverseDynamics(context, ad, forces)[velocities])
            expected = np.asarray(expected)
            theta = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
            np.testing.assert_allclose((regressors(model, q, v, a) @ theta).reshape(-1, 3), expected,
                                       rtol=1e-12, atol=2e-13)
            # Reverse the CSV columns; both backends' native joint orders may differ.
            for split, subset in (("train", slice(0, 200)), ("validation", slice(200, None))):
                write_processed_csv(root / (split + ".csv"), names[::-1], np.arange(200)*.01,
                    q[subset, ::-1], v[subset, ::-1], a[subset, ::-1], expected[subset, ::-1])
            result = run(model_path, root / "train.csv", root / "validation.csv")
            self.assertEqual(result["parameter_count"], 30)
            self.assertLess(result["validation_measured_rmse_nm"], 1e-11)

    def test_same_two_link_urdf_pinocchio_and_drake(self):
        from pinocchio_identification import build_model, synthetic_truth_model
        import pinocchio as pin
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.multibody.tree import MultibodyForces
        model = build_model(BUNDLE / "two_link_id.urdf")
        rng = np.random.default_rng(20261103)
        for model in (model, synthetic_truth_model(model)):
            robot = ET.fromstring((BUNDLE / "two_link_id.urdf").read_text())
            # Write CENTRAL inertias and COM positions using URDF semantics;
            # this independently checks the nonzero-COM frame convention too.
            for index, link in enumerate(("link1", "link2"), 1):
                inertia = model.inertias[index]
                node = robot.find(f"link[@name='{link}']/inertial")
                node.find("mass").set("value", str(inertia.mass))
                node.find("origin").set("xyz", " ".join(str(v) for v in inertia.lever))
                for label, row, column in (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                                           ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)):
                    node.find("inertia").set(label, str(inertia.inertia[row, column]))
            data, plant = model.createData(), MultibodyPlant(0.)
            Parser(plant).AddModels(file_contents=ET.tostring(robot, encoding="unicode"), file_type="urdf")
            plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
            plant.Finalize()
            context, forces = plant.CreateDefaultContext(), MultibodyForces(plant)
            for _ in range(200):
                q, v, a = rng.uniform(-2, 2, (3, 2))
                plant.SetPositions(context, q)
                plant.SetVelocities(context, v)
                plant.CalcForceElementsContribution(context, forces)
                expected = plant.CalcInverseDynamics(context, a, forces)
                np.testing.assert_allclose(pin.rnea(model, data, q, v, a), expected, rtol=1e-12, atol=2e-13)


class PublishedTests(unittest.TestCase):
    @unittest.skipUnless((BUNDLE / "index.md").exists(), "Published snippets are checked in the source repository")
    def test_published_raw_pipeline_commands(self):
        text = (BUNDLE / "index.md").read_text()
        examples = [part.split("\n```", 1)[0] for part in text.split("```bash\n")[1:]
                    if "--make-demo raw-demo" in part.split("\n```", 1)[0]]
        self.assertEqual(len(examples), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("identification_common.py", "pinocchio_identification.py", "preprocess_identification.py", "two_link_id.urdf"):
                shutil.copy2(BUNDLE / name, root / name)
            code = examples[0].replace(".venv-pin/bin/python", shlex.quote(sys.executable))
            subprocess.run(["bash", "-eu"], input=code, cwd=root, text=True,
                check=True, capture_output=True, timeout=30)
            result = json.loads((root / "raw-pipeline.json").read_text())
            self.assertEqual(result["retained_rank"], 6)
            self.assertEqual(result["train_samples"], 381)
            self.assertEqual(result["validation_samples"], 281)
            self.assertLess(result["validation_measured_rmse_nm"], .025)

    @unittest.skipUnless((BUNDLE / "index.md").exists(), "Published snippets are checked in the source repository")
    def test_published_complete_examples(self):
        text = (BUNDLE / "index.md").read_text()
        examples = [part.split("\n```", 1)[0] for part in text.split("```python\n")[1:]
                    if "from identification_common import rmse, scaled_svd_fit" in part.split("\n```", 1)[0]]
        self.assertEqual(len(examples), 2)
        previous = Path.cwd()
        try:
            os.chdir(BUNDLE)
            for code in examples:
                with contextlib.redirect_stdout(io.StringIO()) as stream:
                    exec(compile(ast.parse(code), "<published identification example>", "exec"), {})
                self.assertIn("validation RMSE:", stream.getvalue())
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["common", "pinocchio", "drake"])
    args = parser.parse_args()
    groups = {"common": [DataTests, PreprocessTests], "pinocchio": [PinocchioTests], "drake": [DrakeTests]}
    classes = groups[args.only] if args.only else [DataTests, PreprocessTests, PinocchioTests, DrakeTests, CrossBackendTests, PublishedTests]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in classes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
