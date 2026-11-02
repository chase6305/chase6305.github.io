"""Finite-sample gradient SNR diagnostics; standard library only.

Each row is one same-shape gradient flattened in a fixed coordinate order.
Population claims require iid observations with finite second moments at the
same model state and under the same sampling definition.
The default data are constructed Rademacher observations, not model gradients.
"""
import json
import math
import random


def estimate(samples):
    """Estimate trace variance and signed corrected signal power.

    A nonpositive corrected power is a finite-sample outcome, not a negative
    physical SNR. Zero sample variance leaves both ratios undefined here.
    Powers must be resolvable in Float64; extreme scales raise ValueError.
    """
    try:
        rows = [tuple(float(value) for value in row) for row in samples]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('samples must contain finite numeric vectors') from error
    count = len(rows)
    if count < 2 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('at least two equal, nonempty gradient vectors are required')
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError('gradient observations must be finite')
    dimension = len(rows[0])
    mean = []
    for j in range(dimension):
        scale = max(abs(row[j]) for row in rows)
        mean.append(scale * (math.fsum(row[j]/scale for row in rows)/count) if scale else 0.)
    try:
        mean_power = math.fsum(value*value for value in mean)
        variance = math.fsum((row[j]-mean[j])*(row[j]-mean[j])
                             for row in rows for j in range(dimension))/(count-1)
    except OverflowError as error:
        raise ValueError('gradient powers overflow Float64; use a common smaller scale') from error
    if not math.isfinite(mean_power) or not math.isfinite(variance):
        raise ValueError('gradient powers overflow Float64; use a common smaller scale')
    if ((mean_power == 0. and any(mean)) or
            (variance == 0. and any(row[j] != mean[j] for row in rows for j in range(dimension)))):
        raise ValueError('gradient powers underflow Float64; use a common larger scale')
    corrected = mean_power - variance/count
    naive = mean_power/variance if variance else None
    signed = corrected/variance if variance else None
    if any(value is not None and not math.isfinite(value) for value in (naive, signed)):
        raise ValueError('SNR ratio is outside the finite Float64 range')
    return {
        'observations': count, 'coordinates': dimension, 'mean': mean,
        'sample_mean_power': mean_power, 'sample_trace_variance': variance,
        'mean_noise_correction': variance/count,
        'corrected_signal_power_signed': corrected,
        'naive_ratio': naive, 'corrected_ratio_signed': signed,
        'sample_variance_zero': variance == 0.,
        'corrected_power_negative': corrected < 0.,
    }


def constructed_observations(count=256, averaging=1, signal=1., seed=2026):
    """Four coordinates: mean(signal,0,0,0), independent +/-0.5 noise."""
    rng = random.Random(seed)
    for _ in range(count):
        row = [signal, 0., 0., 0.]
        for j in range(4):
            row[j] += .5*sum(1 if rng.getrandbits(1) else -1 for _ in range(averaging))/averaging
        yield row


def report():
    cases = []
    for name, averaging, signal in [('single_observation', 1, 1.),
                                     ('average_of_eight_independent', 8, 1.),
                                     ('zero_mean', 1, 0.)]:
        cases.append({'name': name, 'independent_base_gradients_per_observation': averaging,
                      'population_mean': [signal, 0., 0., 0.],
                      'population_trace_variance': 1./averaging,
                      'population_snr': signal*signal*averaging,
                      'estimate': estimate(constructed_observations(averaging=averaging, signal=signal))})
    return {
        'construction': 'A 2x2 gradient flattened to four coordinates; independent Rademacher +/-0.5 noise, fixed mean and seed2026.',
        'scope': 'Constructed statistics only. No model, GRPO rollout, optimizer or training is run. Independence and fixed-checkpoint assumptions are not verified by the estimator.',
        'definition': 'SNR of one observation unit: squared norm of its population mean divided by trace covariance. The repetition count estimates this quantity; it does not redefine the observation as their average.',
        'correction': 'sample_mean_power - sample_trace_variance / observations has unbiased expectation under iid finite-second-moment assumptions. Its ratio with the random variance estimate is not generally unbiased.',
        'cases': cases,
        'edge_cases': {
            'balanced_two_observations': estimate([[0., 1., 0., 0.], [0., -1., 0., 0.]]),
            'identical_two_observations': estimate([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            'all_zero': estimate([[0., 0., 0., 0.], [0., 0., 0., 0.]]),
        },
    }


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, allow_nan=False))
