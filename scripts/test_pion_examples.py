"""Independent arithmetic checks for the Pion article; optional --numpy group."""
import argparse
import json
from decimal import Decimal, localcontext
import importlib.util
import math
from pathlib import Path
import random
import sys
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
# A downloaded check_examples.py must test its neighboring files even when
# extracted inside a checkout that also contains the original article bundle.
BUNDLE = HERE.parent/"content/posts/ai/pion-muon-vla-rlvr" if HERE.name=="scripts" and Path(__file__).resolve().name=="test_pion_examples.py" else HERE
for name in ('pion_spectral.py', 'pion_alignment.py', 'pion_matrix.py', 'pion_step.py', 'flat_spectrum.py', 'pion_snr.py', 'snr_correlated_groups.py', 'grpo_group_signal.py', 'vla_gradient_rank.py'):
    if not (BUNDLE/name).is_file():
        raise FileNotFoundError("Missing local Pion example source: "+str(BUNDLE/name))
sys.path.insert(0, str(BUNDLE))
from pion_spectral import (PROMOTION, SUPPRESSION, effective_rank, half_response,
                          muon, pion, polynomial, report, normalize_spectrum, evaluate_spectrum,
                          response_trace, MUON)


class ScalarTests(unittest.TestCase):
    def test_correlated_power_pair_identity_and_negative_covariance(self):
        from fractions import Fraction
        from itertools import product
        from pion_snr import estimate
        def dot(a, b):
            return sum(x*y for x, y in zip(a, b))
        def exact_power(rows):
            count = len(rows)
            mean = [sum(row[j] for row in rows)/count for j in range(len(rows[0]))]
            centered = [[value-average for value, average in zip(row, mean)] for row in rows]
            variance = sum(dot(row, row) for row in centered)/(count-1)
            return dot(mean, mean)-variance/count
        rng = random.Random(20261102)
        for count in range(2, 9):
            for width in (1, 2, 5):
                rows = [[Fraction(rng.randint(-17, 17), 7) for _ in range(width)] for _ in range(count)]
                pairs = sum(dot(rows[i], rows[j]) for i in range(count) for j in range(count) if i != j)
                exact = exact_power(rows)
                self.assertEqual(exact, pairs/(count*(count-1)))
                actual = estimate(rows)
                self.assertAlmostEqual(actual['corrected_signal_power_signed'], float(exact), places=13)
                self.assertEqual(estimate(rows[::-1]), actual)
        for mean in ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(2))):
            states = []
            for u, v in product((-1, 1), repeat=2):
                noises = [(Fraction(u), Fraction(v)), (Fraction(-u), Fraction(2*v)),
                          (Fraction(2*u), Fraction(-v))]
                rows = [tuple(a+b for a, b in zip(mean, noise)) for noise in noises]
                states.append((noises, rows))
            covariances = [[sum(dot(noise[i], noise[j]) for noise, rows in states)/len(states)
                            for j in range(3)] for i in range(3)]
            off_diagonal = sum(covariances[i][j] for i in range(3) for j in range(3) if i != j)
            expected = sum(exact_power(rows) for noise, rows in states)/len(states)
            self.assertEqual([covariances[i][i] for i in range(3)], [2, 5, 5])
            self.assertEqual(off_diagonal, -4)
            self.assertEqual(expected, dot(mean, mean)+off_diagonal/6)
            self.assertEqual(expected, Fraction(-2, 3) if mean == (0, 0) else Fraction(13, 3))

    def test_grpo_group_moments_isolate_caller_decimal_context(self):
        from decimal import getcontext, Inexact, ROUND_FLOOR, Underflow
        from grpo_group_signal import group_moments
        expected = group_moments(8, '0.01')
        def signature(context):
            return (context.prec, context.rounding, context.Emin, context.Emax,
                    context.capitals, context.clamp, dict(context.flags), dict(context.traps))
        outer = signature(getcontext())
        with localcontext() as caller:
            caller.clear_flags()
            caller.prec = 9;caller.rounding = ROUND_FLOOR
            caller.Emin = -20;caller.Emax = 20;caller.capitals = 0
            caller.flags[Underflow] = True;caller.traps[Inexact] = True
            before = signature(caller)
            self.assertEqual(group_moments(8, '0.01'), expected)
            self.assertEqual(signature(caller), before)
            for invalid in ('bad', 'NaN', '1e-1000000'):
                with self.assertRaises(ValueError):group_moments(8, invalid)
                self.assertEqual(signature(caller), before)
        self.assertEqual(signature(getcontext()), outer)

    def test_scalar_action_hidden_gradient_rank_and_finite_differences(self):
        from fractions import Fraction
        from vla_gradient_rank import hidden_example
        for size in (2, 3, 8):
            result = hidden_example(size)
            weights, inputs = result['hidden_weights'], result['inputs']
            def loss(matrix):
                outputs = [math.fsum(max(0., math.fsum(w*x for w, x in zip(row, sample)))
                                     for row in matrix) for sample in inputs]
                return math.fsum(abs(value) for value in outputs)/len(inputs)
            self.assertEqual(loss(weights), 1.)
            step = 1e-6
            for i in range(size):
                for j in range(size):
                    plus, minus = [row[:] for row in weights], [row[:] for row in weights]
                    plus[i][j] += step;minus[i][j] -= step
                    derivative = (loss(plus)-loss(minus))/(2*step)
                    self.assertAlmostEqual(derivative, result['hidden_gradient'][i][j], delta=2e-10)
            # Exact elimination supplies an independent rank/determinant
            # certificate without using the helper's eigenvalue formulas.
            matrix = [[Fraction(value) for value in row] for row in result['hidden_gradient_exact']]
            determinant = Fraction(1)
            for col in range(size):
                pivot = next((row for row in range(col, size) if matrix[row][col]), None)
                self.assertIsNotNone(pivot)
                if pivot != col:
                    matrix[pivot], matrix[col] = matrix[col], matrix[pivot]
                    determinant = -determinant
                diagonal = matrix[col][col]
                determinant *= diagonal
                for row in range(col+1, size):
                    ratio = matrix[row][col]/diagonal
                    for j in range(col, size):
                        matrix[row][j] -= ratio*matrix[col][j]
            certificate = result['positive_eigenvalue_certificate']
            self.assertEqual(determinant, Fraction(certificate['determinant']))
            self.assertGreater(determinant, 0)
            self.assertEqual(result['hidden_rank'], size)
            self.assertEqual(result['scalar_output_dimension'], 1)
            self.assertEqual(result['output_head_rank'], 1)
        for size in (True, 1, 33, 8., -1):
            with self.assertRaises(ValueError):hidden_example(size)

    def test_rank_one_gradients_accumulate_distinct_momentum_directions(self):
        from fractions import Fraction
        from vla_gradient_rank import momentum_example, report as rank_report
        history = momentum_example()
        for row in history['steps']:
            step = row['step']
            # Closed history sum of beta^(t-j)*(1-beta)*G_j, evaluated
            # independently of the helper's iterative buffer update.
            exact = [Fraction(1, 2**(step-j)) if j < step else Fraction(0)
                     for j in range(3)]
            self.assertEqual(row['buffer_diagonal_exact'], [str(value) for value in exact])
            self.assertEqual(row['buffer_rank'], step)
            self.assertEqual(row['gradient_rank'], 1)
            self.assertTrue(1 <= row['buffer_effective_rank'] <= step)
        self.assertEqual(history['steps'][-1]['buffer_diagonal'], [.125, .25, .5])
        actual = json.loads(subprocess.check_output([sys.executable, '-B', str(BUNDLE/'vla_gradient_rank.py')], cwd=BUNDLE, timeout=10))
        self.assertEqual(actual, rank_report())

    def test_grpo_group_gradient_detached_advantages_and_finite_differences(self):
        from grpo_group_signal import one_group
        for outcomes in ([1, 0], [1, 0, 0], [0, 1, 1, 0, 1], [0, 0], [1, 1]):
            for p in (.01, .25, .5, .9):
                actual = one_group(iter(outcomes), p)
                size, successes = len(outcomes), sum(outcomes)
                expected = math.sqrt(successes*(size-successes))/size
                self.assertAlmostEqual(actual['ascent_gradient'], expected, places=14)
                self.assertAlmostEqual(actual['sum_advantages'], 0., places=13)
                theta = math.log(p/(1-p))
                # Differentiate log likelihood while holding the *sampled*
                # advantages fixed, exactly as a policy surrogate does.
                def objective(logit):
                    return math.fsum(a*(-math.log1p(math.exp(-logit)) if y else
                                        -math.log1p(math.exp(logit)))
                                     for y, a in zip(outcomes, actual['advantages']))/size
                step = 1e-5
                derivative = (objective(theta+step)-objective(theta-step))/(2*step)
                self.assertAlmostEqual(derivative, expected, delta=1e-10)
                def log_probability(logit, outcome):
                    return -math.log1p(math.exp(-logit)) if outcome else -math.log1p(math.exp(logit))
                old_logs = [log_probability(theta, y) for y in outcomes]
                def ratio_objective(logit):
                    return math.fsum(a*math.exp(log_probability(logit, y)-old)
                                     for y, a, old in zip(outcomes, actual['advantages'], old_logs))/size
                ratio_derivative = (ratio_objective(theta+step)-ratio_objective(theta-step))/(2*step)
                self.assertAlmostEqual(ratio_derivative, expected, delta=1e-10)
        mixed = one_group([1, 0], .5)
        self.assertEqual(mixed['advantages'], [1., -1.])
        self.assertEqual(mixed['scores_wrt_logit'], [.5, -.5])
        self.assertEqual(mixed['ascent_gradient'], .5)

    def test_grpo_group_moments_against_independent_outcome_enumeration(self):
        from fractions import Fraction
        from itertools import product
        from grpo_group_signal import group_moments
        for size in range(2, 9):
            for p in (Fraction(1, 100), Fraction(1, 4), Fraction(1, 2)):
                gradients, weights, kept = [], [], []
                for outcomes in product((0, 1), repeat=size):
                    successes = sum(outcomes)
                    weight = p**successes*(1-p)**(size-successes)
                    reward_mean = successes/size
                    std = math.sqrt(reward_mean*(1-reward_mean))
                    # Compute each response's score-weighted contribution;
                    # do not use the report's binomial count coefficients.
                    gradient = (math.fsum((value-reward_mean)/std*(value-float(p))
                                          for value in outcomes)/size if std else 0.)
                    gradients.append(gradient);weights.append(float(weight))
                    if 0 < successes < size:
                        kept.append((gradient, float(weight)))
                result = group_moments(size, str(float(p)))
                mean = math.fsum(g*w for g, w in zip(gradients, weights))
                second = math.fsum(g*g*w for g, w in zip(gradients, weights))
                variance = second-mean*mean
                self.assertAlmostEqual(float(result['full_group']['mean']), mean, places=13)
                self.assertAlmostEqual(float(result['full_group']['variance']), variance, places=13)
                self.assertTrue(math.isclose(float(result['full_group']['snr']), mean*mean/variance, rel_tol=1e-11))
                exact_q = 1-p**size-(1-p)**size
                exact_second = Fraction(size-1, size)*p*(1-p)
                with localcontext() as context:
                    context.prec = 100
                    decimal_q = Decimal(exact_q.numerator)/Decimal(exact_q.denominator)
                    self.assertLess(abs(Decimal(result['nondegenerate_probability'])-decimal_q), Decimal('1e-75'))
                    decimal_second = Decimal(exact_second.numerator)/Decimal(exact_second.denominator)
                    reconstructed = Decimal(result['full_group']['variance'])+Decimal(result['full_group']['mean'])**2
                    self.assertLess(abs(reconstructed-decimal_second), Decimal('1e-75'))
                    conditional = result['nondegenerate_only']
                    total_variance = (decimal_q*Decimal(conditional['variance'])+
                                      decimal_q*(1-decimal_q)*Decimal(conditional['mean'])**2)
                    self.assertLess(abs(total_variance-Decimal(result['full_group']['variance'])), Decimal('1e-75'))
                q = math.fsum(w for g, w in kept)
                kept_mean = math.fsum(g*w for g, w in kept)/q
                self.assertAlmostEqual(float(result['nondegenerate_only']['mean']), kept_mean, places=13)
                self.assertAlmostEqual(float(result['expected_proposal_groups_per_kept_group']), 1/q, places=12)

    def test_grpo_group_moment_boundaries_units_and_cli(self):
        from grpo_group_signal import group_moments, one_group, report as group_report
        for p in ('0', '1'):
            result = group_moments(8, p)
            self.assertEqual(Decimal(result['nondegenerate_probability']), 0)
            self.assertEqual(Decimal(result['full_group']['mean']), 0)
            self.assertEqual(Decimal(result['full_group']['variance']), 0)
            self.assertIsNone(result['full_group']['snr'])
            self.assertIsNone(result['nondegenerate_only'])
            self.assertIsNone(result['expected_proposal_groups_per_kept_group'])
        for size in (2, 3):
            for p in ('0.01', '0.5', '0.99'):
                result = group_moments(size, p)
                self.assertTrue(result['nondegenerate_only']['variance_zero'])
                self.assertIsNone(result['nondegenerate_only']['snr'])
        for p, opposite in (('0.01', '0.99'), ('0.1', '0.9')):
            left, right = group_moments(8, p), group_moments(8, opposite)
            for key in ('mean', 'variance', 'snr'):
                self.assertAlmostEqual(float(left['full_group'][key]), float(right['full_group'][key]), places=13)
        for p in ('1e-300', '0.999999999999999999999999999999'):
            result = group_moments(8, p)
            self.assertGreater(Decimal(result['full_group']['variance']), 0)
            self.assertGreater(Decimal(result['full_group']['snr']), 0)
            self.assertGreater(Decimal(result['nondegenerate_only']['variance']), 0)
        # A raw E[h^2]-E[h]^2 subtraction loses this conditional variance
        # even at80 digits. The k=2 versus k=1 rare-event expansion gives
        # an independent leading coefficient as p tends to zero.
        rare = group_moments(8, '1e-120')
        with localcontext() as context:
            context.prec = 100
            delta = (Decimal(12).sqrt()-Decimal(7).sqrt())/8
            leading = Decimal(7)/2*delta*delta
            measured = Decimal(rare['nondegenerate_only']['variance'])/Decimal('1e-120')
            self.assertLess(abs(measured-leading), Decimal('1e-70'))
            large = group_moments(128, '.5')
            second = (Decimal(large['full_group']['variance'])+
                      Decimal(large['full_group']['mean'])**2)
            self.assertLess(abs(second-Decimal(127)/512), Decimal('1e-75'))
        for size, p in ((1, '.5'), (129, '.5'), (True, '.5'), (8., '.5'),
                        (8, 'NaN'), (8, 'Infinity'), (8, '-.1'), (8, '1.1'),
                        (8, 'bad'), (8, '1e-1000000')):
            with self.assertRaises(ValueError):group_moments(size, p)
        for values, p in ((None, .5), ([], .5), ([1], .5), ([0, 2], .5),
                          ([True, 0], .5), ([1., 0], .5), ([1, 0], 0),
                          ([1, 0], 1), ([1, 0], math.nan)):
            with self.assertRaises(ValueError):one_group(values, p)
        # An unbounded outcome iterator is rejected after129 observations.
        bounded_code = ('from itertools import repeat; from grpo_group_signal import one_group\n'
                        'try: one_group(repeat(1), .5)\n'
                        'except ValueError: pass\n'
                        'else: raise RuntimeError("unbounded group accepted")\n')
        subprocess.run([sys.executable, '-B', '-c', bounded_code], cwd=BUNDLE, check=True, timeout=5)
        result = group_report()
        rare = result['cases'][1]
        self.assertGreater(float(rare['nondegenerate_only']['snr']), 300)
        self.assertLess(float(rare['full_group']['snr']), .084)
        self.assertGreater(float(rare['expected_proposal_groups_per_kept_group']), 12)
        actual = json.loads(subprocess.check_output([sys.executable, '-B', str(BUNDLE/'grpo_group_signal.py')], cwd=BUNDLE, timeout=10))
        self.assertEqual(actual, result)

    def test_correlated_group_expectations_against_exact_enumeration(self):
        from fractions import Fraction
        from itertools import product
        from pion_snr import estimate
        from snr_correlated_groups import known_expectations
        for groups in range(2, 7):
            for copies in (1, 2, 8, 32):
                totals = {key: Fraction(0) for key in (
                    'sample_mean_power', 'row_sample_variance',
                    'row_iid_corrected_signal_power',
                    'independent_group_sample_variance',
                    'independent_group_corrected_signal_power')}
                for signs in product((-1, 1), repeat=groups):
                    # Enumerate the joint distribution itself; do not use a
                    # covariance/bias formula from the reporting helper.
                    independent = [Fraction(1 + sign) for sign in signs]
                    rows = [value for value in independent for _ in range(copies)]
                    count = len(rows)
                    mean = sum(rows)/count
                    power = mean*mean
                    row_variance = sum((value-mean)**2 for value in rows)/(count-1)
                    group_variance = sum((value-mean)**2 for value in independent)/(groups-1)
                    exact = dict(zip(totals, (power, row_variance, power-row_variance/count,
                                             group_variance, power-group_variance/groups)))
                    for key, value in exact.items():
                        totals[key] += value
                    observed = estimate((value,) for value in rows)
                    grouped = estimate((value,) for value in independent)
                    self.assertEqual(observed['mean'], grouped['mean'])
                    self.assertAlmostEqual(observed['sample_trace_variance'], float(row_variance), places=13)
                    self.assertAlmostEqual(observed['corrected_signal_power_signed'], float(exact['row_iid_corrected_signal_power']), places=13)
                    self.assertAlmostEqual(grouped['corrected_signal_power_signed'], float(exact['independent_group_corrected_signal_power']), places=13)
                reference = known_expectations(groups, copies)
                for key, total in totals.items():
                    expectation = total/(2**groups)
                    self.assertEqual(Fraction(reference['expectations'][key]['exact']), expectation)
                    self.assertEqual(reference['expectations'][key]['float'], float(expectation))
                bias = totals['row_iid_corrected_signal_power']/(2**groups)-1
                self.assertEqual(Fraction(reference['expectations']['row_iid_correction_bias']['exact']), bias)
                self.assertEqual(reference['variance_of_sample_mean'], 1/groups)
                self.assertEqual(reference['recorded_rows'], groups*copies)
                self.assertEqual(bias == 0, copies == 1)

    def test_correlated_group_limits_cli_and_single_sample_scope(self):
        from snr_correlated_groups import known_expectations, report as group_report
        for groups, copies in ((True, 8), (4, False), (1, 8), (4, 0), (4., 8),
                               (4, 8.), (1_000_001, 8), (4, 1_000_001), (-1, 2)):
            with self.assertRaises(ValueError):
                known_expectations(groups, copies)
        large = known_expectations(1_000_000, 1_000_000)
        self.assertEqual(large['recorded_rows'], 10**12)
        result = group_report()
        self.assertEqual([row['recorded_rows'] for row in result['cases']], [4, 8, 32, 128])
        finite = result['finite_example']
        self.assertEqual(finite['row_based']['mean'], finite['independent_group_based']['mean'])
        # One balanced sample falls below the true power even though the
        # expectation of the incorrectly applied row correction is above it.
        self.assertLess(finite['row_based']['corrected_signal_power_signed'], 1.)
        self.assertLess(finite['independent_group_based']['corrected_signal_power_signed'], 1.)
        self.assertGreater(result['cases'][2]['expectations']['row_iid_corrected_signal_power']['float'], 1.)
        actual = json.loads(subprocess.check_output(
            [sys.executable, '-B', str(BUNDLE/'snr_correlated_groups.py')],
            cwd=BUNDLE, timeout=10))
        self.assertEqual(actual, json.loads(json.dumps(result)))

    def test_snr_correction_against_exact_independent_rademacher_enumeration(self):
        from fractions import Fraction
        from itertools import product
        from pion_snr import estimate
        for count in range(2, 8):
            powers, variances, corrected = [], [], []
            for signs in product((-1, 1), repeat=count):
                vectors = [(Fraction(1)+sign*Fraction(1,2), Fraction(2)-sign) for sign in signs]
                means = [sum(row[j] for row in vectors)/count for j in range(2)]
                exact_power = sum(value*value for value in means)
                exact_variance = sum((row[j]-means[j])**2 for row in vectors for j in range(2))/(count-1)
                exact_corrected = exact_power-exact_variance/count
                result = estimate(vectors)
                self.assertAlmostEqual(result['sample_mean_power'],float(exact_power),places=12)
                self.assertAlmostEqual(result['sample_trace_variance'],float(exact_variance),places=12)
                self.assertAlmostEqual(result['corrected_signal_power_signed'],float(exact_corrected),places=12)
                powers.append(result['sample_mean_power']);variances.append(result['sample_trace_variance']);corrected.append(result['corrected_signal_power_signed'])
            # Independent draws have mean(1,2), trace covariance 1/4+1.
            self.assertAlmostEqual(math.fsum(powers)/len(powers),5+1.25/count,places=12)
            self.assertAlmostEqual(math.fsum(variances)/len(variances),1.25,places=12)
            self.assertAlmostEqual(math.fsum(corrected)/len(corrected),5.,places=12)
        signed = [estimate([[0.,float(s)] for s in signs])['corrected_signal_power_signed'] for signs in product((-1,1),repeat=4)]
        self.assertAlmostEqual(math.fsum(signed)/len(signed),0.,places=14)
        self.assertGreater(math.fsum(max(0.,value)for value in signed)/len(signed),0.)

    def test_snr_edge_cases_scaling_and_single_pass_observations(self):
        from pion_snr import estimate, report as snr_report
        rows = [[1.,.5],[1.,-.5],[2.,.5],[2.,-.5]]
        original = [row[:]for row in rows]
        expected = estimate(rows)
        self.assertEqual(estimate(iter(rows)),expected)
        self.assertEqual(rows,original)
        for scale in (1e-60,1e60,8.):
            actual = estimate([[value*scale for value in row]for row in rows])
            self.assertAlmostEqual(actual['corrected_ratio_signed'],expected['corrected_ratio_signed'],places=12)
        result = estimate([[0.,1.],[0.,-1.]])
        self.assertEqual(result['corrected_signal_power_signed'],-1.)
        self.assertEqual(result['corrected_ratio_signed'],-.5)
        for samples in ([[1.],[1.]],[[0.],[0.]]):
            result=estimate(samples);self.assertTrue(result['sample_variance_zero'])
            self.assertIsNone(result['naive_ratio']);self.assertIsNone(result['corrected_ratio_signed'])
        for samples in ([],[[]],[[],[]],[[1.]],[[1.],[1.,2.]],[[math.nan],[1.]],[[math.inf],[1.]],[[1e200],[-1e200]],[[1e-200],[-1e-200]],[[1e-200],[1e-200]],[1.,2.]):
            with self.assertRaises(ValueError):estimate(samples)
        result=snr_report()
        self.assertEqual([c['population_snr']for c in result['cases']],[1.,8.,0.])
        self.assertLess(result['cases'][-1]['estimate']['corrected_ratio_signed'],0.)

    def test_flat_spectrum_isolates_caller_decimal_context(self):
        from decimal import getcontext, Inexact, ROUND_FLOOR, Underflow
        from flat_spectrum import rank_response
        cases = [(rank, kp) for rank in (1, 2, 8, 65536) for kp in range(6)]
        expected = {case: rank_response(*case) for case in cases}
        def signature(context):
            return (context.prec, context.rounding, context.Emin, context.Emax,
                    context.capitals, context.clamp, dict(context.flags), dict(context.traps))
        outer = signature(getcontext())
        with localcontext() as caller:
            caller.clear_flags()
            caller.prec = 9;caller.rounding = ROUND_FLOOR
            caller.Emin = -20;caller.Emax = 20;caller.capitals = 0
            caller.flags[Underflow] = True;caller.traps[Inexact] = True
            before = signature(caller)
            for case in cases:
                self.assertEqual(rank_response(*case), expected[case])
                self.assertEqual(signature(caller), before)
            for invalid in (True, 0, 65537, None):
                with self.assertRaises(ValueError):rank_response(invalid, 0)
                self.assertEqual(signature(caller), before)
        self.assertEqual(signature(getcontext()), outer)

    def test_flat_spectrum_decimal_values_with_exact_rational_composition(self):
        from fractions import Fraction
        from flat_spectrum import rank_response, report as flat_report
        # Perfect-square dimensions give rational normalized singular values.
        # Exact fractions and explicit polynomial formulas do not use the
        # report's coefficient table or its Decimal implementation.
        with localcontext() as context:
            context.prec = 120
            for rank in (1, 4, 16, 64, 256, 1024, 4096, 65536):
                for kp in (0, 1, 2):
                    value = Fraction(1, math.isqrt(rank))
                    for index in range(5):
                        value = (15*value-10*value**3+3*value**5)/8 if index < kp else (5*value**3-3*value**5)/2
                    exact = Decimal(value.numerator) / Decimal(value.denominator)
                    result = rank_response(rank, kp)
                    actual = Decimal(result['coefficient_decimal'])
                    self.assertLess(abs(actual-exact)/exact, Decimal('1e-95'))
                    self.assertAlmostEqual(result['log10_coefficient_decimal'], float(exact.log10()), places=12)
        rows = {row['rank']: row for row in flat_report()['rows']}
        self.assertEqual(rows[4096]['mathematical_entropy_rank'], 4096)
        self.assertGreater(Decimal(rows[4096]['pion']['0']['coefficient_decimal']), 0)
        self.assertEqual(rows[4096]['pion']['0']['coefficient_float64'], 0.)
        self.assertTrue(rows[4096]['pion']['0']['float64_underflowed'])
        self.assertEqual(rows[4096]['pion']['0']['float64_entropy_rank'], 0)
        self.assertLess(rows[8]['pion']['0']['log10_coefficient_decimal'], -60)
        for rank in (True, 0, -1, 65537, 4., None):
            with self.assertRaises(ValueError): rank_response(rank, 0)

    def test_optional_probes_reject_wrong_source_before_import_under_optimized_python(self):
        names = ['probe_official_pion.py', 'probe_official_steps.py',
                 'probe_official_vla_buffers.py', 'probe_official_vla_padding.py', 'flat_spectrum.py']
        with tempfile.TemporaryDirectory(prefix='pion-source-guard-') as folder:
            bad = Path(folder, 'not_official.py')
            bad.write_text('raise RuntimeError("UNTRUSTED_SOURCE_IMPORTED")\n')
            output = Path(folder, 'preserved.json')
            output.write_text('preserve this existing file\n')
            for optimized in (False, True):
                for name in names:
                    arguments = [sys.executable, '-B'] + (['-O'] if optimized else [])
                    arguments += [str(BUNDLE/name), '--official-file', str(bad)]
                    if name == 'probe_official_vla_padding.py': arguments += ['--output', str(output)]
                    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0, name)
                    self.assertIn('does not match', result.stderr, name)
                    self.assertIn('pinned Pion commit', result.stderr, name)
                    self.assertNotIn('UNTRUSTED_SOURCE_IMPORTED', result.stderr, name)
                    self.assertEqual(output.read_text(), 'preserve this existing file\n')

    def test_each_trace_stage_with_decimal_and_cli_contract(self):
        with localcontext() as context:
            context.prec = 90
            for sigma in (0., .001, .35, .5, math.sqrt(2/3), 1.):
                for kp in range(6):
                    result = response_trace(sigma,kp)
                    self.assertEqual(result['pion_stages'],['promotion']*kp+['suppression']*(5-kp))
                    for name in ['pion','muon']:
                        expected = Decimal.from_float(sigma)
                        self.assertEqual(len(result[name]),6)
                        self.assertEqual(result[name][0],sigma)
                        for index in range(5):
                            coefficients = MUON if name=='muon' else PROMOTION if index<kp else SUPPRESSION
                            a,b,c = [Decimal(str(v)) for v in coefficients]
                            expected = a*expected+b*expected**3+c*expected**5
                            self.assertTrue(math.isclose(result[name][index+1],float(expected),rel_tol=3e-12,abs_tol=5e-324))
                    self.assertEqual(result['pion'][-1],pion(sigma,kp))
                    self.assertEqual(result['muon'][-1],muon(sigma))
        first,second = response_trace(.35,1),response_trace(.35,2)
        self.assertLess(first['pion'][1],math.sqrt(2/3))
        self.assertGreater(second['pion'][2],math.sqrt(2/3))
        self.assertLess(first['pion'][-1],1e-5)
        self.assertGreater(second['pion'][-1],.98)
        script=BUNDLE/'pion_spectral.py'
        actual=json.loads(subprocess.check_output([sys.executable,'-B',str(script),'--sigma','.35','--kp','2','--trace'],text=True))
        self.assertEqual(actual['trace'],second)
        for arguments in [['--trace'],['--spectrum','1','2','--trace']]:
            result=subprocess.run([sys.executable,'-B',str(script),*arguments],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('error:',result.stderr)

    def test_decimal_composition_and_boundaries(self):
        rng = random.Random(20261030)
        with localcontext() as context:
            context.prec = 90
            for kp in range(6):
                for value in [0., 1., .001, .5, .816496580927726, *[rng.random() for _ in range(50)]]:
                    expected = Decimal.from_float(value)
                    for index in range(5):
                        a,b,c = map(lambda x: Decimal(str(x)), PROMOTION if index < kp else SUPPRESSION)
                        expected = a*expected+b*expected**3+c*expected**5
                    # An ordinary 1e-15 absolute tolerance would accept zero
                    # instead of the tiny positive coefficients discussed here.
                    self.assertTrue(math.isclose(pion(value,kp),float(expected),rel_tol=2e-12,abs_tol=5e-324))
                self.assertEqual(pion(0,kp),0)
                self.assertEqual(pion(1,kp),1)

    def test_monotonic_maps_and_suppression_fixed_point(self):
        threshold = math.sqrt(2/3)
        self.assertAlmostEqual(polynomial(threshold,SUPPRESSION),threshold)
        for coefficients in (PROMOTION,SUPPRESSION):
            values=[polynomial(i/2000,coefficients) for i in range(2001)]
            self.assertTrue(all(-2e-15<=x<=1+2e-15 for x in values))
            self.assertTrue(all(b>=a-2e-15 for a,b in zip(values,values[1:])))
        self.assertLess(polynomial(.8,SUPPRESSION),.8)
        self.assertGreater(polynomial(.9,SUPPRESSION),.9)
        self.assertAlmostEqual(pion(1,1),1)
        for index in range(2001):
            sigma=index/2000
            difference=polynomial(sigma,PROMOTION)-polynomial(sigma,SUPPRESSION)
            self.assertAlmostEqual(difference,1.875*sigma*(1-sigma*sigma)**2,places=14)
            values=[pion(sigma,kp)for kp in range(6)]
            self.assertTrue(all(b>=a-3e-15 for a,b in zip(values,values[1:])))
        # Finite Muon NS is not exact matrix sign at sigma=1.
        self.assertGreater(abs(muon(1)-1),.01)

    def test_finite_response_threshold_and_flat_spectrum(self):
        thresholds=[half_response(kp) for kp in range(6)]
        self.assertTrue(all(b<a for a,b in zip(thresholds,thresholds[1:])))
        for kp,value in enumerate(thresholds): self.assertAlmostEqual(pion(value,kp),.5,places=12)
        self.assertLess(pion(.5,0),1e-20)
        self.assertEqual(effective_rank([0,0]),0)
        self.assertAlmostEqual(effective_rank([1,1,1,1]),4)
        self.assertAlmostEqual(effective_rank([1,1,1,1,5e-324]),4)
        self.assertAlmostEqual(effective_rank([1,.5]),effective_rank([.5,.25]))

    def test_invalid_arguments(self):
        for value in [True,None,'0.5',float('nan'),float('inf'),-.1,1.1,10**10000]:
            with self.assertRaises(ValueError): pion(value)
        for value in [True,-1,6,1.,None]:
            with self.assertRaises(ValueError): pion(.5,value)
        with self.assertRaises(ValueError): effective_rank([])

    def test_frobenius_budget_for_half_response_directions(self):
        # Equal spectra attain the direction-count bound; adding one direction
        # makes every normalized coefficient fall below the same threshold.
        limits=[]
        for kp in range(6):
            threshold=half_response(kp)
            count=math.floor(1/threshold**2)
            limits.append(count)
            self.assertGreaterEqual(pion(1/math.sqrt(count),kp),.5)
            self.assertLess(pion(1/math.sqrt(count+1),kp),.5)
        self.assertEqual(limits[:2],[1,4])
        rng=random.Random(191)
        for kp in range(6):
            for width in [2,4,16,64]:
                for _ in range(12):
                    values=evaluate_spectrum([rng.random()**3 for _ in range(width)],kp)
                    count=sum(v>=.5 for v in values['pion'])
                    self.assertLessEqual(count,min(width,limits[kp]))

    def test_reported_table_arithmetic(self):
        result=report()
        self.assertAlmostEqual(result['vl_adapter_suite_averages_percent']['Pion'],97.25)
        self.assertAlmostEqual(result['vlanext_pion_minus_muon_percentage_points']['LIBERO-Plus'],3.59)
        self.assertEqual(sum(result['real_robot_success_counts_inferred_from_30_trials']['Pion']),77)
        self.assertAlmostEqual(sum(x*x for x in result['example_normalized_singular_values']),1)

    def test_normalization_against_decimal_norm_and_epsilon(self):
        with localcontext() as context:
            context.prec = 100
            for raw in [[.8,.4,.2,.02],[1e308]*4,[1e-200,2e-200],[1.,5e-324],[0.,0.]]:
                for epsilon in [0.,1e-7,1e308]:
                    values=[Decimal.from_float(v) for v in raw]
                    denominator=sum(v*v for v in values).sqrt()+Decimal.from_float(epsilon)
                    expected=[float(v/denominator) if denominator else 0. for v in values]
                    actual=normalize_spectrum(raw,epsilon)
                    for a,b in zip(actual,expected):self.assertTrue(math.isclose(a,b,rel_tol=5e-15,abs_tol=5e-324))
        self.assertAlmostEqual(effective_rank([1e308]*4),4)
        self.assertAlmostEqual(effective_rank([8,4,2,.2]),effective_rank([.8,.4,.2,.02]))

    def test_spectrum_evaluation_and_invalid_epsilon(self):
        value=evaluate_spectrum([.8,.4,.2,.02])
        self.assertAlmostEqual(value['pion_norm'],math.hypot(*value['pion']))
        self.assertGreater(value['input_erank'],value['pion_erank'])
        flat=evaluate_spectrum([1,1,1,1],0)
        self.assertLess(flat['pion_norm'],1e-30)
        self.assertAlmostEqual(flat['pion_erank'],4)
        self.assertTrue(evaluate_spectrum([0,0,0,0])['all_zero'])
        for epsilon in [-1,True,None,'0',float('nan'),float('inf'),10**10000]:
            with self.assertRaises(ValueError):evaluate_spectrum([1,1],epsilon=epsilon)

    def test_cli_spectrum_and_normalized_input_contract(self):
        script=BUNDLE/'pion_spectral.py'
        result=subprocess.run([sys.executable,'-B',str(script),'--spectrum','.8','.4','.2','.02','--kp','1','--epsilon','1e-7'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),evaluate_spectrum([.8,.4,.2,.02],1,1e-7))
        for arguments in [['--kp','0'],['--sigma','.5','--epsilon','1e-7'],['--sigma','.5','--epsilon','0'],['--epsilon','0'],['--spectrum','-1','1'],['--spectrum','1','--kp','6']]:
            result=subprocess.run([sys.executable,'-B',str(script),*arguments],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('error:',result.stderr)

    def test_known_signal_alignment_in_both_perturbation_regimes(self):
        from pion_alignment import report as alignment_report
        weak,dominant=alignment_report()['examples']
        self.assertEqual(weak['signal_diagonal'],dominant['signal_diagonal'])
        expected=[(.9747403576571587,.48745768043555915,1.),
                  (.19611613513818402,.7408948780944629,7.148384810287638e-22)]
        for row,reference in zip([weak,dominant],expected):
            for key,value in zip(['normalized_gradient','muon','pion'],reference):
                self.assertTrue(math.isclose(row['cosine_with_clean_signal'][key],value,rel_tol=2e-12,abs_tol=5e-324))
                with localcontext() as context:
                    context.prec=90
                    signal=[Decimal.from_float(v)for v in row['signal_diagonal']]
                    update=[Decimal.from_float(v)for v in row['update_diagonals'][key]]
                    numerator=sum(a*b for a,b in zip(signal,update))
                    denominator=(sum(v*v for v in signal)*sum(v*v for v in update)).sqrt()
                    self.assertTrue(math.isclose(row['cosine_with_clean_signal'][key],float(numerator/denominator),rel_tol=4e-15,abs_tol=5e-324))
            self.assertEqual(row['gradient_diagonal'],[a+b for a,b in zip(row['signal_diagonal'],row['perturbation_diagonal'])])
        self.assertGreater(weak['cosine_with_clean_signal']['pion'],weak['cosine_with_clean_signal']['muon'])
        self.assertLess(dominant['cosine_with_clean_signal']['pion'],dominant['cosine_with_clean_signal']['muon'])


class MatrixTests(unittest.TestCase):
    def test_scalar_output_hidden_gradient_against_matrix_backprop_and_svd(self):
        import numpy as np
        from vla_gradient_rank import hidden_example
        for size in (2, 3, 8, 32):
            result = hidden_example(size)
            inputs = np.array(result['inputs'])
            weights = np.array(result['hidden_weights'], dtype=float)
            head = np.array(result['output_weights'], dtype=float)
            preactivation = inputs @ weights.T
            hidden = np.maximum(preactivation, 0.)
            outputs = hidden @ head.T
            loss_derivative = np.sign(outputs)/size
            upstream = (loss_derivative @ head)*(preactivation > 0.)
            hidden_gradient = upstream.T @ inputs
            head_gradient = loss_derivative.T @ hidden
            np.testing.assert_allclose(hidden_gradient, result['hidden_gradient'], rtol=2e-15, atol=0.)
            np.testing.assert_allclose(head_gradient, result['output_head_gradient'], rtol=2e-15, atol=0.)
            self.assertEqual(np.linalg.matrix_rank(hidden_gradient), size)
            self.assertEqual(np.linalg.matrix_rank(head_gradient), 1)
            singular_values = np.linalg.svd(hidden_gradient, compute_uv=False)
            certificate = result['positive_eigenvalue_certificate']
            from fractions import Fraction
            expected = [float(Fraction(certificate['orthogonal_to_ones']))]*(size-1)
            expected.append(float(Fraction(certificate['parallel_to_ones'])))
            np.testing.assert_allclose(singular_values, expected, rtol=2e-14, atol=0.)

    def test_flat_spectrum_rms_matches_full_matrix_step(self):
        import numpy as np
        from flat_spectrum import rank_response
        from pion_step import step
        for rank in (1, 2, 4, 8, 16):
            for kp in (0, 1, 2):
                weight = np.zeros((rank, rank))
                updated, _, trace = step(weight, np.eye(rank), beta=0., nesterov=False,
                                         promotion_steps=kp, lr=.01, scale_factor=.2, weight_decay=0.)
                reference = rank_response(rank, kp)['coefficient_float64']
                expected_rms = .002 * reference
                self.assertTrue(math.isclose(trace['gradient_update_rms'], expected_rms,
                                             rel_tol=5e-13, abs_tol=5e-324))
                # The update stays diagonal for I_r; include every coordinate
                # in the RMS rather than averaging only the diagonal entries.
                expected = -math.sqrt(rank) * expected_rms * np.eye(rank)
                np.testing.assert_allclose(updated, expected, rtol=5e-13, atol=5e-324)

    def test_svd_factorization_rectangular_and_rank_deficient(self):
        import numpy as np
        from pion_matrix import transform
        rng=np.random.default_rng(20261030)
        for shape in [(3,7),(7,3),(4,4)]:
            for kp in range(6):
                source=rng.normal(size=shape)
                u,s,vt=np.linalg.svd(source,full_matrices=False)
                for epsilon in [0.,1e-7,10.]:
                    expected=(u*np.array([pion(float(x),kp) for x in s/(np.linalg.norm(s)+epsilon)]))@vt
                    np.testing.assert_allclose(transform(source,kp,epsilon),expected,atol=5e-13,rtol=5e-12)
        zero=np.zeros((3,5));np.testing.assert_array_equal(transform(zero),zero)
        rank_one=np.outer([1.,2.,3.],[1.,-2.,0.,4.])
        self.assertEqual(np.linalg.matrix_rank(transform(rank_one)),1)

    def test_head_axes_scale_and_equivariance(self):
        import numpy as np
        from pion_matrix import transform,per_head_transform
        rng=np.random.default_rng(37)
        source=rng.normal(size=(6,8))
        for axis,heads in [(0,3),(1,4)]:
            expected=np.concatenate([transform(p,1) for p in np.split(source,heads,axis=axis)],axis=axis)
            np.testing.assert_allclose(per_head_transform(source,heads,axis,1),expected,atol=1e-14)
        np.testing.assert_allclose(transform(source*1e200),transform(source),atol=5e-14)
        np.testing.assert_allclose(transform(source*1e-200),transform(source),atol=5e-14)
        np.testing.assert_allclose(transform(source.T),transform(source).T,atol=5e-14)
        for source in [np.ones((2,2))*1j,np.array([[float('nan')]]),np.array([])]:
            with self.assertRaises(ValueError): transform(source)
        with self.assertRaises(ValueError): per_head_transform(np.ones((3,4)),2,0)

    def test_independent_two_head_diagonal_and_epsilon_scale(self):
        import numpy as np
        from pion_matrix import transform,per_head_transform,head_report
        result=head_report()
        raw=result['input_diagonal']
        expected=[]
        for pair in [raw[:2],raw[2:]]:
            expected.extend([pion(value/math.hypot(*pair),1)for value in pair])
        np.testing.assert_allclose(result['per_head_diagonal'],expected,atol=0,rtol=2e-12)
        self.assertAlmostEqual(result['input_block_norms'][0]/result['input_block_norms'][1],10)
        self.assertAlmostEqual(result['per_head_block_norms'][0]/result['per_head_block_norms'][1],1)
        self.assertAlmostEqual(math.hypot(*result['whole_block_norms']),1)
        self.assertAlmostEqual(math.hypot(*result['per_head_block_norms']),math.sqrt(2))
        source=np.diag([.8,.4,.2,.02])*1e-8
        np.testing.assert_allclose(np.diag(transform(source,1,1e-7)),evaluate_spectrum(np.diag(source).tolist(),1,1e-7)['pion'],rtol=1e-12,atol=1e-300)
        self.assertLess(float(np.linalg.norm(transform(source,1,1e-7))),1e-50)
        # Large finite input and epsilon must not overflow their common norm.
        actual=transform(np.eye(2)*1e308,5,1e308)
        expected=np.eye(2)*pion(1/(math.sqrt(2)+1),5)
        np.testing.assert_allclose(actual,expected,atol=1e-14)

    def test_stateful_step_against_independent_diagonal_oracle(self):
        import numpy as np
        from pion_step import step
        # The independent oracle evolves scalar Decimal EMA coefficients and
        # composes the odd polynomials; it does not call the matrix core.
        for nesterov in (False, True):
            for beta in (0., .5, .95):
                for kp in (0, 1, 5):
                    weight = np.diag([1., 2.])
                    buffer = np.zeros_like(weight)
                    with localcontext() as context:
                        context.prec = 90
                        expected_weight = [Decimal(1), Decimal(2)]
                        expected_buffer = [Decimal(0), Decimal(0)]
                        for observed in ([.8, .2], [0., 0.], [-.3, .1]):
                            expected_buffer = [Decimal(str(beta))*b+(1-Decimal(str(beta)))*Decimal(str(g))
                                               for b,g in zip(expected_buffer, observed)]
                            direction = [(1-Decimal(str(beta)))*Decimal(str(g))+Decimal(str(beta))*b
                                         if nesterov else b for g,b in zip(observed, expected_buffer)]
                            norm = sum(v*v for v in direction).sqrt()
                            filtered = [v/norm if norm else Decimal(0) for v in direction]
                            for index in range(5):
                                coefficients = PROMOTION if index < kp else SUPPRESSION
                                a,b,c = [Decimal(str(v)) for v in coefficients]
                                filtered = [a*v+b*v**3+c*v**5 for v in filtered]
                            scale = Decimal('.2')*Decimal(2).sqrt()
                            expected_weight = [Decimal('.999')*w-Decimal('.01')*scale*f
                                               for w,f in zip(expected_weight, filtered)]
                            weight,buffer,trace = step(weight,np.diag(observed),buffer,beta=beta,
                                                      nesterov=nesterov,promotion_steps=kp)
                            np.testing.assert_allclose(np.diag(buffer),[float(v) for v in expected_buffer],rtol=3e-14,atol=1e-17)
                            np.testing.assert_allclose(np.diag(weight),[float(v) for v in expected_weight],rtol=3e-14,atol=1e-17)
                            for actual,expected in zip(np.diag(trace['filtered_direction']),filtered):
                                self.assertTrue(math.isclose(actual,float(expected),rel_tol=3e-11,abs_tol=5e-324))

    def test_rotations_and_tiny_full_rank_matrix_response(self):
        import numpy as np
        from pion_matrix import transform
        rng = np.random.default_rng(402)
        for width in (4,16,64,128):
            left,_ = np.linalg.qr(rng.normal(size=(width,width)))
            right,_ = np.linalg.qr(rng.normal(size=(width,width)))
            source = left @ right.T
            coefficient = pion(1/math.sqrt(width),0)
            self.assertGreater(coefficient,0.)
            actual = transform(source,0)
            # Ordinary absolute tolerances would accept an all-zero matrix.
            np.testing.assert_allclose(actual/coefficient,source,rtol=8e-11,atol=8e-11)
            self.assertTrue(math.isclose(math.hypot(*actual.flat),math.sqrt(width)*coefficient,rel_tol=8e-11))
        source = np.outer([1.,2.,3.],[1.,-2.,3.,-4.])
        expected = source/math.hypot(*source.flat)
        for kp in (0,1,2):np.testing.assert_allclose(transform(source,kp),expected,atol=3e-15,rtol=3e-15)

    def test_zero_gradient_and_missing_gradient_have_different_effects(self):
        import numpy as np
        from pion_step import step,report as step_report
        skip,decay = step_report('skip'),step_report('decay')
        first,zero,missing = skip['steps']
        self.assertEqual(zero['status'],'gradient_observed')
        self.assertEqual(missing['status'],'missing_skip')
        np.testing.assert_allclose(zero['buffer_after'],np.array(first['buffer_after'])*.5,atol=0)
        self.assertGreater(np.linalg.norm(zero['filtered_direction']),.99)
        self.assertEqual(missing['weight_after'],zero['weight_after'])
        self.assertEqual(missing['buffer_after'],zero['buffer_after'])
        self.assertEqual(missing['configured_decay_factor'],.999)
        self.assertEqual(missing['applied_decay_factor'],1.)
        self.assertEqual(decay['steps'][-1]['applied_decay_factor'],.999)
        self.assertAlmostEqual(first['gradient_update_rms'],.002/math.sqrt(2),places=15)
        self.assertAlmostEqual(zero['gradient_update_rms'],first['gradient_update_rms'],places=15)
        self.assertEqual(missing['gradient_update_rms'],0.)
        np.testing.assert_allclose(decay['steps'][-1]['weight_after'],np.array(zero['weight_after'])*.999,atol=0)
        zero_weight,zero_buffer,trace = step(np.eye(2),np.zeros((2,2)))
        np.testing.assert_array_equal(zero_buffer,np.zeros((2,2)))
        np.testing.assert_array_equal(trace['filtered_direction'],np.zeros((2,2)))
        self.assertEqual(trace['gradient_update_rms'],0.)
        np.testing.assert_array_equal(zero_weight,np.eye(2)*.999)

    def test_gradient_rms_for_rectangular_rank_one_updates(self):
        import numpy as np
        from pion_step import step
        # A single nonzero singular direction has filtered coefficient one.
        # Its coordinate RMS is eta*gamma/sqrt(min(rows,cols)), regardless
        # of which side the matrix core transposes internally.
        for shape in [(1,7),(7,1),(2,5),(5,2),(3,7),(7,3),(4,4)]:
            gradient=np.zeros(shape)
            gradient[0,0]=2.
            after,buffer,trace=step(np.zeros(shape),gradient,beta=0.,nesterov=False,weight_decay=0.)
            expected=np.zeros(shape)
            expected[0,0]=-.002*math.sqrt(max(shape))
            np.testing.assert_allclose(after,expected,atol=2e-17,rtol=0)
            self.assertTrue(math.isclose(trace['gradient_update_rms'],.002/math.sqrt(min(shape)),rel_tol=2e-14))
            self.assertTrue(math.isclose(trace['gradient_update_rms'],math.hypot(*after.flat)/math.sqrt(shape[0]*shape[1]),rel_tol=2e-14))
        after,buffer,trace=step(np.zeros((4,4)),np.eye(4),beta=0.,nesterov=False,
                                promotion_steps=5,lr=1e308,weight_decay=0.,scale_factor=.5)
        self.assertTrue(np.isfinite(after).all())
        self.assertTrue(math.isclose(trace['gradient_update_rms'],5e307,rel_tol=2e-14))
        json.dumps(trace,allow_nan=False)
        after,buffer,trace=step(np.zeros((1,2)),np.array([[1.,0.]]),beta=0.,nesterov=False,
                                lr=1e-300,weight_decay=0.,scale_factor=1/math.sqrt(2))
        self.assertTrue(math.isclose(trace['gradient_update_rms'],1e-300/math.sqrt(2),rel_tol=2e-14))

    def test_stateful_step_preserves_inputs_and_rejects_invalid_observations(self):
        import numpy as np
        from pion_step import step
        weight = np.eye(2)
        gradient = np.diag([.8,.2])
        buffer = np.diag([.2,.1])
        originals = [value.copy() for value in (weight,gradient,buffer)]
        after,current,trace = step(weight,gradient,buffer,lr=0.)
        np.testing.assert_array_equal(after,weight)
        self.assertFalse(np.array_equal(current,buffer))
        for result in (after,current):
            for source in (weight,gradient,buffer):self.assertFalse(np.shares_memory(result,source))
        cases = [{'beta':1.},{'nesterov':1},{'missing_gradient':'zero'},{'lr':-1},
                 {'scale_factor':float('inf')},{'epsilon':-1},{'promotion_steps':6}]
        for settings in cases:
            with self.assertRaises(ValueError):step(weight,gradient,buffer,**settings)
        for bad in (np.ones((3,2)),np.array([[float('nan'),0],[0,1]]),np.eye(2)*1j):
            with self.assertRaises(ValueError):step(weight,bad,buffer)
        for source,original in zip((weight,gradient,buffer),originals):np.testing.assert_array_equal(source,original)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--numpy',action='store_true');args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScalarTests)
    if args.numpy:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(MatrixTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
