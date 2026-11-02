"""One-step Bernoulli GRPO surrogate with population reward standard deviation.

At fixed old policy p=sigmoid(theta), rewards equal sampled outcomes y in {0,1}
and the score is y-p. Advantages are held constant during differentiation.
There is no clipping, KL term, multi-token score noise or model training.
"""
from decimal import (Context, Decimal, DivisionByZero, InvalidOperation,
                     Overflow, ROUND_HALF_EVEN, Underflow, localcontext)
from itertools import islice
import json
import math


def one_group(outcomes, success_probability):
    """Float64 realized ascent direction, with detached normalized advantages."""
    try:
        values = tuple(islice(iter(outcomes), 129))
    except TypeError as error:
        raise ValueError('outcomes must be an iterable of binary integers') from error
    try:
        p = float(success_probability)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('success_probability must be finite and in (0,1)') from error
    if (not 2 <= len(values) <= 128 or
            any(type(value) is not int or value not in (0, 1) for value in values) or
            not math.isfinite(p) or not 0 < p < 1):
        raise ValueError('require 2..128 binary integer outcomes and p in (0,1)')
    size = len(values)
    mean_reward = sum(values)/size
    std = math.sqrt(mean_reward*(1-mean_reward))
    advantages = [(value-mean_reward)/std if std else 0. for value in values]
    scores = [value-p for value in values]
    return {
        'outcomes': list(values), 'mean_reward': mean_reward,
        'population_reward_std': std, 'advantages': advantages,
        'scores_wrt_logit': scores,
        'sum_advantages': math.fsum(advantages),
        'ascent_gradient': math.fsum(a*s for a, s in zip(advantages, scores))/size,
    }


def group_moments(group_size, success_probability):
    """80-digit Decimal moments over proposed groups and retained mixed groups.

    Probabilities0 and1 are limiting cases, not finite Bernoulli logits.
    A private context fixes rounding/range and preserves the caller's context,
    including on rejected inputs. A zero-variance ratio is returned as null.
    An unresolved range raises ValueError rather than storing a zero SNR.
    """
    if type(group_size) is not int or not 2 <= group_size <= 128:
        raise ValueError('group_size must be an integer in [2,128]')
    try:
        private = Context(prec=80, rounding=ROUND_HALF_EVEN,
                          Emin=-999999, Emax=999999, capitals=1, clamp=0,
                          flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
        with localcontext(private) as context:
            p = Decimal(str(success_probability))
            if not p.is_finite() or not 0 <= p <= 1:
                raise ValueError('success_probability must be finite and in [0,1]')
            size = Decimal(group_size)
            # Integer k*(g-k) keeps the k/g and (g-k)/g cases symmetric.
            coefficients = [Decimal(k*(group_size-k)).sqrt()/size
                            for k in range(group_size+1)]
            if p == 0 or p == 1:
                weights = [Decimal(int(k == (group_size if p else 0)))
                           for k in range(group_size+1)]
            else:
                weights = [Decimal(math.comb(group_size, k))*p**k*(1-p)**(group_size-k)
                           for k in range(group_size+1)]
                total = sum(weights)
                weights = [weight/total for weight in weights]

            def summarize(values, probabilities):
                active = {value for value, weight in zip(values, probabilities) if weight}
                if len(active) == 1:
                    mean, variance = next(iter(active)), Decimal(0)
                else:
                    mean = sum(value*weight for value, weight in zip(values, probabilities))
                    variance = sum(weight*(value-mean)**2
                                   for value, weight in zip(values, probabilities))
                ratio = mean*mean/variance if variance else None
                return {'mean': str(mean), 'variance': str(variance),
                        'snr': str(ratio) if ratio is not None else None,
                        'variance_zero': variance == 0}

            nondegenerate = sum(weights[1:-1])
            full = summarize(coefficients, weights)
            kept = (summarize(coefficients[1:-1],
                              [weight/nondegenerate for weight in weights[1:-1]])
                    if nondegenerate else None)
            proposals = str(1/nondegenerate) if nondegenerate else None
            if context.flags[Underflow]:
                raise ValueError('probability is outside the supported Decimal numerical range')
            return {
                'group_size': group_size, 'success_probability': str(p),
                'decimal_precision': context.prec,
                'nondegenerate_probability': str(nondegenerate),
                'full_group': full, 'nondegenerate_only': kept,
                'expected_proposal_groups_per_kept_group': proposals,
            }
    except (ArithmeticError, TypeError) as error:
        raise ValueError('invalid probability or unresolved Decimal numerical range') from error


def report():
    return {
        'construction': 'Independent one-step Bernoulli outcomes, reward=y, '
                        'p=sigmoid(theta), score=y-p, population reward std, '
                        'epsilon=0 with zero advantages for constant rewards.',
        'scope': 'A constructed scalar on-policy surrogate ascent direction '
                 'with detached advantages; no clipping, KL, token residuals, '
                 'model, optimizer or training. It is not the paper low-SNR '
                 'approximation and not a model-gradient SNR prediction.',
        'mixed_two_response_example': one_group((1, 0), .5),
        'cases': [group_moments(8, p) for p in ('0', '0.01', '0.1', '0.5', '0.9', '0.99', '1')],
        'constant_kept_coefficient_example': group_moments(2, '0.5'),
        'interpretation': 'The two moment distributions use different sampling '
                          'units: every proposed complete group versus only '
                          'mixed groups after rejection. Proposal cost assumes '
                          'independent repeated draws. A null ratio at zero '
                          'variance is not a claim about noise in real training.',
    }


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, allow_nan=False))
