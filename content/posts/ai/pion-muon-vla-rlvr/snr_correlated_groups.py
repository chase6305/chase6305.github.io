"""Known expectations when independent scalar gradients are recorded repeatedly.

Each of B independent groups draws G=1+Z, with equiprobable Z in {-1,+1}.
The group's gradient is then copied n times, giving K=B*n rows but only B
independent observations. This construction is not a GRPO noise model.
"""
from fractions import Fraction
import json

from pion_snr import estimate


def known_expectations(group_count, copies):
    """Exact powers/variances for the fixed mean-one, variance-one construction.

    Ratios with an estimated variance are deliberately not averaged: they can
    be undefined and are not made unbiased by an unbiased numerator.
    """
    if (type(group_count) is not int or type(copies) is not int or
            not 2 <= group_count <= 1_000_000 or not 1 <= copies <= 1_000_000):
        raise ValueError('group_count must be an integer in [2,1000000]; '
                         'copies must be an integer in [1,1000000]')
    count = group_count * copies
    bias = Fraction(copies-1, count-1)
    powers = {
        'sample_mean_power': 1 + Fraction(1, group_count),
        'row_sample_variance': Fraction(copies*(group_count-1), count-1),
        'row_iid_corrected_signal_power': 1 + bias,
        'row_iid_correction_bias': bias,
        'independent_group_sample_variance': Fraction(1),
        'independent_group_corrected_signal_power': Fraction(1),
    }
    return {
        'independent_groups': group_count,
        'copies_per_group': copies,
        'recorded_rows': count,
        'true_signal_power': 1,
        'marginal_variance_per_row': 1,
        'variance_of_sample_mean': float(Fraction(1, group_count)),
        'expectations': {
            key: {'exact': str(value), 'float': float(value)}
            for key, value in powers.items()
        },
    }


def report():
    groups = (0., 0., 2., 2.)
    return {
        'construction': 'B independent scalar gradients G=1+Z with iid '
                        'equiprobable signs Z; n identical copies of each group.',
        'scope': 'Analytic expectations for this constructed distribution, '
                 'not training measurements or a model of GRPO dependence. '
                 'Grouping requires known independent sampling units.',
        'population_mean': 1,
        'population_trace_variance': 1,
        'population_snr_per_independent_group': 1,
        'cases': [known_expectations(4, n) for n in (1, 2, 8, 32)],
        'finite_example': {
            'group_gradients': groups,
            'copies_per_group': 8,
            'row_based': estimate((value,) for value in groups for _ in range(8)),
            'independent_group_based': estimate((value,) for value in groups),
        },
        'interpretation': 'Expectations are over all independent group draws. '
                          'A single corrected power can fall above or below '
                          'the true power. Copies leave the sample mean and its '
                          'true uncertainty unchanged. Averaging estimated '
                          'SNR ratios is not asserted to be unbiased.',
    }


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, allow_nan=False))
