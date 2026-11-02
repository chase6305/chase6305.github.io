"""Independent finite-policy audit of reward-conditional score residuals.

A response is a vector of independent Bernoulli bits. Its reward is the first
bit. The first logit has success probability p; every nuisance logit is zero.
T=1 and groups have two responses. This is a constructed policy, not a paper
experiment or a literal instantiation of all shared-variance assumptions.
"""
import argparse
from fractions import Fraction
from itertools import product
import json
import math


def norm2(values):
    return sum(x*x for x in values)


def log_sigmoid(value):
    return -max(0., -value)-math.log1p(math.exp(-abs(value)))


def log_probability(bits, logits):
    return math.fsum(log_sigmoid(t if bit else -t) for bit, t in zip(bits, logits))


def population(p, dimensions):
    rows = []
    for bits in product((0, 1), repeat=dimensions+1):
        weight = (p if bits[0] else 1-p)/2**dimensions
        score = (Fraction(bits[0])-p,)+tuple(Fraction(bit)-Fraction(1, 2) for bit in bits[1:])
        rows.append((bits[0], weight, score))
    mean = [sum(w*s[j] for reward, w, s in rows) for j in range(dimensions+1)]
    assert mean == [0]*(dimensions+1)
    conditional = {r: [sum(w*s[j] for reward, w, s in rows if reward == r)/(p if r else 1-p)
                       for j in range(dimensions+1)] for r in (0, 1)}
    residual = sum(w*norm2([a-b for a, b in zip(s, conditional[r])]) for r, w, s in rows)
    total = sum(w*norm2(s) for r, w, s in rows)
    between = sum((p if r else 1-p)*norm2(conditional[r]) for r in (0, 1))
    assert residual == Fraction(dimensions, 4)
    assert between == p*(1-p)
    assert total == residual+between
    return {'total_score_variance': str(total), 'conditional_residual_variance': str(residual),
            'between_reward_means_variance': str(between),
            'score_gap_norm_squared': str(norm2([a-b for a, b in zip(conditional[1], conditional[0])]))}


def groups(p, dimensions):
    mean = [Fraction(0)]*(dimensions+1)
    second = Fraction(0)
    kept_mean = [Fraction(0)]*(dimensions+1)
    kept_second = Fraction(0)
    q = Fraction(0)
    patterns = 0
    derivative_checks = 0
    max_error = 0.
    logits = [math.log(float(p/(1-p)))]+[0.]*dimensions
    for flat in product((0, 1), repeat=2*(dimensions+1)):
        bits = [flat[:dimensions+1], flat[dimensions+1:]]
        rewards = [row[0] for row in bits]
        weight = p**sum(rewards)*(1-p)**(2-sum(rewards))/2**(2*dimensions)
        reward_mean = Fraction(sum(rewards), 2)
        centered = [Fraction(r)-reward_mean for r in rewards]
        variance = sum(x*x for x in centered)/2
        if variance:
            numerator, denominator = math.isqrt(variance.numerator), math.isqrt(variance.denominator)
            std = Fraction(numerator, denominator)
            assert std*std == variance
            advantages = [v/std for v in centered]
        else:
            advantages = [Fraction(0)]*2
        scores = [[Fraction(row[0])-p]+[Fraction(bit)-Fraction(1, 2) for bit in row[1:]] for row in bits]
        gradient = [sum(advantages[i]*scores[i][j] for i in range(2))/2 for j in range(dimensions+1)]
        for j in range(dimensions+1):mean[j] += weight*gradient[j]
        second += weight*norm2(gradient)
        if variance:
            q += weight
            for j in range(dimensions+1):kept_mean[j] += weight*gradient[j]
            kept_second += weight*norm2(gradient)
        if dimensions <= 2:
            old = [log_probability(row, logits) for row in bits]
            def objective(parameters):
                return math.fsum(float(a)*math.exp(log_probability(row, parameters)-lp)
                                 for a, row, lp in zip(advantages, bits, old))/2
            for j in range(dimensions+1):
                for step in (1e-5, 1e-6):
                    plus, minus = logits[:], logits[:]
                    plus[j] += step;minus[j] -= step
                    observed = (objective(plus)-objective(minus))/(2*step)
                    error = abs(observed-float(gradient[j]))
                    assert error < 3e-9
                    max_error = max(max_error, error)
                    derivative_checks += 1
        patterns += 1
    full_variance = second-norm2(mean)
    conditional_mean = [value/q for value in kept_mean]
    conditional_variance = kept_second/q-norm2(conditional_mean)
    expected_q = 1-p*p-(1-p)**2
    assert q == expected_q
    assert mean == [q/2]+[0]*dimensions
    assert conditional_mean == [Fraction(1, 2)]+[0]*dimensions
    assert conditional_variance == Fraction(dimensions, 8)
    assert full_variance == q*(1-q)/4+q*dimensions/8
    assert full_variance == q*conditional_variance+q*(1-q)*norm2(conditional_mean)
    snr = norm2(mean)/full_variance
    assert snr == q/(1-q+Fraction(dimensions, 2))
    return {'p': str(p), 'nuisance_dimensions': dimensions, 'joint_groups_enumerated': patterns,
            'nondegenerate_probability': str(q), 'mean_reward_gradient': str(mean[0]),
            'full_gradient_trace_variance': str(full_variance), 'full_snr': str(snr),
            'conditional_trace_variance': str(conditional_variance),
            'conditional_snr': str(norm2(conditional_mean)/conditional_variance) if conditional_variance else None,
            'finite_difference_checks': derivative_checks, 'max_finite_difference_error': max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = []
    for p in (Fraction(1, 100), Fraction(1, 4), Fraction(1, 2)):
        for dimensions in range(5):
            cases.append({**groups(p, dimensions), 'population': population(p, dimensions)})
    asymptotic = []
    for p in (Fraction(1, 100), Fraction(1, 2)):
        q = 2*p*(1-p)
        for dimensions in (1, 16, 64, 4096):
            exact = q/(1-q+Fraction(dimensions, 2))
            approximation = 2*q/dimensions
            relative = (approximation-exact)/exact
            assert relative == 2*(1-q)/dimensions
            asymptotic.append({'p': str(p), 'nuisance_dimensions': dimensions,
                'calculation': 'Analytic independent-coordinate variance; not joint enumeration at this dimension.',
                'exact_snr': str(exact), 'residual_dominant_approximation': str(approximation),
                'approximation_relative_overestimate': str(relative)})
    result = {'construction': __doc__, 'optimizer_steps': 0,
        'cases': cases, 'joint_groups_enumerated': sum(r['joint_groups_enumerated'] for r in cases),
        'finite_difference_checks': sum(r['finite_difference_checks'] for r in cases),
        'max_finite_difference_error': max(r['max_finite_difference_error'] for r in cases),
        'analytic_noise_limit': asymptotic,
        'interpretation': 'Total score variance includes between-reward mean variation. It equals residual variance only when that variation vanishes, or approximately when it is negligible. With no nuisance dimension the conditional residual is zero and the residual-dominant formula is inapplicable. Large-dimensional analytic rows do not enumerate or train large models.'}
    from pathlib import Path
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ['joint_groups_enumerated','finite_difference_checks','max_finite_difference_error','optimizer_steps']}))


if __name__ == '__main__':main()
