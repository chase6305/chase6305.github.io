#!/usr/bin/env python3
"""Coordinate and deadline examples for RMA / UMI / RTC; standard library.

SE(2) poses are (x, y, angle_radians). Times are milliseconds. This module
does not train RMA, run UMI hardware, or implement generative RTC inference.
"""
import argparse
from decimal import Decimal, ROUND_CEILING
import math
import random


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def pose(value):
    if len(value) != 3:
        raise ValueError("an SE(2) pose needs x, y and angle")
    return tuple(finite(v, "pose coordinate") for v in value)


def compose(first, second):
    x, y, angle = pose(first)
    u, v, theta = pose(second)
    c, s = math.cos(angle), math.sin(angle)
    return (x + c*u - s*v, y + s*u + c*v, angle + theta)


def inverse(value):
    x, y, angle = pose(value)
    c, s = math.cos(angle), math.sin(angle)
    return (-c*x - s*y, s*x - c*y, -angle)


def relative(anchor, target):
    return compose(inverse(anchor), target)


def reanchor(action, old_anchor, new_anchor):
    """Represent one unchanged world target relative to a different anchor."""
    return compose(inverse(new_anchor), compose(old_anchor, action))


def first_valid_index(observation_ms, inference_ms, execution_ms, period_ms):
    """First i with i*period >= total delay; equality is accepted.

    Decimal conversion preserves the caller's decimal numbers at boundaries.
    This is a deadline convention, not the paper's sub-timestep-free floor.
    """
    values = [finite(v, "time") for v in (observation_ms, inference_ms, execution_ms, period_ms)]
    if min(values[:3]) < 0 or values[3] <= 0:
        raise ValueError("delays must be nonnegative and period must be positive")
    delays = [Decimal(str(v)) for v in values[:3]]
    period = Decimal(str(values[3]))
    return int((sum(delays)/period).to_integral_value(rounding=ROUND_CEILING))


def _softmask_weights(horizon, overlap, delay):
    """RTC v2 Eq. 5: linear factor times normalized exponential factor."""
    result = []
    for i in range(horizon):
        if i < delay:
            result.append(1.)
        elif i < overlap:
            c = (overlap-i)/(overlap-delay+1)
            result.append(c*math.expm1(c)/math.expm1(1.))
        else:
            result.append(0.)
    return result


def rtc_weights(horizon=16, stride=5, delay=4):
    """Equation 5 of RTC v2, with explicit serial scheduling constraints."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (horizon, stride, delay)):
        raise ValueError("horizon, stride and delay must be integers")
    if not (horizon >= 1 and 1 <= stride and 0 <= delay <= stride <= horizon-delay):
        raise ValueError("expected 0 <= d <= s <= H-d, with H,s positive")
    return _softmask_weights(horizon, horizon-stride, delay)


def timing_budget(horizon=16, stride=5, period=20., observation=0., inference=80., execution=0.):
    """Fixed-latency extension: worker occupancy and target deadline differ."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (horizon, stride)) or not (8 <= horizon <= 64 and 1 <= stride <= horizon):
        raise ValueError("inline horizon/stride settings are out of range")
    times = [finite(v, "time") for v in (period, observation, inference, execution)]
    if not 1 <= times[0] <= 100 or any(not 0 <= v <= 1000 for v in times[1:]):
        raise ValueError("inline time settings are out of range")
    if any(abs(v*10-round(v*10)) > 1e-9 for v in times):
        raise ValueError("inline times must be multiples of 0.1 ms")
    # Quantized decimal sums and explicit ceiling are independent of the JS
    # implementation's integer division with a period-minus-one offset.
    period_d, obs_d, work_d, exec_d = [Decimal(str(v)).quantize(Decimal('.1')) for v in times]
    delay = int(((obs_d+work_d+exec_d)/period_d).to_integral_value(rounding=ROUND_CEILING))
    work_delay = int((work_d/period_d).to_integral_value(rounding=ROUND_CEILING))
    overlap = horizon-stride
    valid = work_delay <= stride <= horizon-delay
    weights = _softmask_weights(horizon, overlap, delay) if valid else []
    return {'horizon':horizon, 'stride':stride, 'delay':delay, 'workDelay':work_delay,
            'totalDelay':float(obs_d+work_d+exec_d), 'remaining':overlap, 'schedulable':valid,
            'frozen':delay if valid else 0, 'editable':overlap-delay if valid else 0,
            'future':stride if valid else 0, 'noStride':work_delay+delay > horizon, 'weights':weights}


def convolution_lengths(length=50, layers=((8, 4), (5, 1), (5, 1))):
    """No-padding, dilation-one temporal convolution lengths."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    result = [length]
    for kernel, stride in layers:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (kernel, stride)) or length < kernel:
            raise ValueError("kernel/stride invalid for the current length")
        length = (length-kernel)//stride+1
        result.append(length)
    return result


def uncovered_steps(horizon, stride, actual_delay):
    """One handover's missing old-plan slots; not a full RTC scheduler."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (horizon, stride, actual_delay)) or not (horizon > 0 and 0 <= stride <= horizon and actual_delay >= 0):
        raise ValueError("invalid handover counts")
    return max(0, actual_delay-(horizon-stride))


def guidance_strength(tau, beta=5.):
    """Clipped coefficient of RTC v2 Eq. 2, with explicit endpoint limits."""
    tau, beta = finite(tau, "denoising time"), finite(beta, "guidance cap")
    if not 0 <= tau <= 1 or beta <= 0:
        raise ValueError("invalid denoising time or guidance cap")
    return beta if tau in (0., 1.) else min(beta, tau/(1-tau)+(1-tau)/tau)


def linear_guidance(value=(.1, -.2, .3), tau=.4):
    """A fixed nonsymmetric linear velocity field, not a trained flow policy."""
    value = tuple(finite(v, "noise coordinate") for v in value)
    tau = finite(tau, "denoising time")
    if len(value) != 3 or not 0 <= tau <= 1:
        raise ValueError("expected three coordinates and tau in [0,1]")
    matrix = [[.2, -.1, .3], [.4, .1, -.2], [-.3, .2, .1]]
    offset, target, mask = [.1, -.2, .05], [.3, -.1, .4], [1., .5, 0.]
    velocity = [math.fsum(a*b for a, b in zip(row, value))+b for row, b in zip(matrix, offset)]
    final = [a+(1-tau)*v for a, v in zip(value, velocity)]
    jacobian = [[float(i == j)+(1-tau)*matrix[i][j] for j in range(3)] for i in range(3)]
    loss = .5*math.fsum(w*(a-y)**2 for w, a, y in zip(mask, final, target))
    correction = [math.fsum(jacobian[i][j]*mask[i]*(target[i]-final[i]) for i in range(3)) for j in range(3)]
    return loss, correction


def affine_metric_example(value=(.2, .2), scale=(.1, 1.),
                          offset=(.3, -.4), target=(.3, -.4),
                          tolerance=(.01, .2)):
    """Two local action coordinates: affine decoding and tolerance metric.

    Coordinate 0 may be translation in m, coordinate 1 rotation in rad.
    This diagonal affine example is not an SE(3) geodesic or RTC policy.
    """
    rows = [tuple(finite(v, 'metric coordinate') for v in row)
            for row in (value, scale, offset, target, tolerance)]
    if any(len(row) != 2 for row in rows):
        raise ValueError('expected two coordinates in every metric vector')
    value, scale, offset, target, tolerance = rows
    if min(scale+tolerance) <= 0:
        raise ValueError('scales and tolerances must be positive')
    physical = [b+s*x for b, s, x in zip(offset, scale, value)]
    error = [p-y for p, y in zip(physical, target)]
    loss = .5*math.fsum((e/t)**2 for e, t in zip(error, tolerance))
    normalized_loss = .5*math.fsum((x-(y-b)/s)**2
                                 for x, y, b, s in zip(value, target, offset, scale))
    gradient = [s*e/(t*t) for s, e, t in zip(scale, error, tolerance)]
    return {'physical':physical, 'error':error, 'loss':loss,
            'normalized_loss':normalized_loss, 'gradient':gradient}


def segment_distance(start, end, point):
    """Euclidean distance from a 2D point to the entire closed segment."""
    if any(len(p) != 2 for p in (start, end, point)):
        raise ValueError("expected 2D points")
    a, b, p = [tuple(finite(v, "point coordinate") for v in q) for q in (start, end, point)]
    delta = (b[0]-a[0], b[1]-a[1])
    length2 = sum(v*v for v in delta)
    t = 0. if length2 == 0 else max(0., min(1., sum((p[i]-a[i])*delta[i] for i in range(2))/length2))
    return math.hypot(*(a[i]+t*delta[i]-p[i] for i in range(2)))


def minimum_path_distance(points, point=(0., 0.)):
    points = list(points)
    if len(points) < 2:
        raise ValueError("a path needs at least two points")
    return min(segment_distance(a, b, point) for a, b in zip(points, points[1:]))


def check():
    rng = random.Random(20261106)
    # Independent homogeneous-matrix composition, not another call to compose.
    def matrix(p):
        x, y, a = p
        return [[math.cos(a), -math.sin(a), x], [math.sin(a), math.cos(a), y], [0., 0., 1.]]

    def mm(a, b):
        return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    def close(a, b):
        assert max(abs(u-v) for ra, rb in zip(matrix(a), matrix(b)) for u, v in zip(ra, rb)) < 2e-12

    for _ in range(200):
        old, new, target, world, tool = [tuple(rng.uniform(-2, 2) for _ in range(3)) for _ in range(5)]
        ab = compose(old, new)
        expected = mm(matrix(old), matrix(new))
        assert max(abs(expected[i][j]-matrix(ab)[i][j]) for i in range(3) for j in range(3)) < 2e-12
        close(compose(old, relative(old, target)), target)
        close(compose(new, reanchor(relative(old, target), old, new)), target)
        close(relative(compose(world, old), compose(world, target)), relative(old, target))
        close(relative(compose(old, tool), compose(target, tool)),
              compose(inverse(tool), compose(relative(old, target), tool)))
        mid = tuple(rng.uniform(-2, 2) for _ in range(3))
        close(compose(relative(old, mid), relative(mid, target)), relative(old, target))
    assert first_valid_index(40, 60, 20, 20) == 6
    assert first_valid_index(40, 60.001, 20, 20) == 7
    assert first_valid_index(.1, .2, 0, .1) == 3
    extended = timing_budget(16, 4, 20, 40, 60, 20)
    assert extended['delay'] == 6 and extended['workDelay'] == 3 and extended['schedulable']
    assert extended['weights'][:6] == [1.]*6 and extended['editable'] == 6
    assert convolution_lengths() == [50, 11, 7, 3]
    # Independent enumeration of complete windows checks each convolution.
    lengths = [50]
    for kernel, stride in ((8, 4), (5, 1), (5, 1)):
        lengths.append(len([i for i in range(0, lengths[-1], stride) if i+kernel <= lengths[-1]]))
    assert lengths == convolution_lengths()
    assert uncovered_steps(16, 5, 12) == 1
    assert uncovered_steps(16, 5, 11) == 0
    assert guidance_strength(0.) == guidance_strength(1.) == 5.
    assert guidance_strength(.5) == 2.
    for tau in (.001, .01, .1, .4, .5, .8, .99, .999):
        r2 = (1-tau)**2/(tau*tau+(1-tau)**2)
        assert abs(guidance_strength(tau)-min(5., (1-tau)/(tau*r2))) < 1e-12
        loss, correction = linear_guidance(tau=tau)
        for axis in range(3):
            plus, minus = [.1, -.2, .3], [.1, -.2, .3]
            plus[axis] += 1e-6
            minus[axis] -= 1e-6
            negative_gradient = -(linear_guidance(plus, tau)[0]-linear_guidance(minus, tau)[0])/2e-6
            assert abs(negative_gradient-correction[axis]) < 1e-10
    assert abs(linear_guidance()[1][2]) > .001  # W[2]=0 does not imply g[2]=0.
    metric = affine_metric_example()
    assert abs(metric['loss']-2.5) < 1e-13
    assert abs(metric['normalized_loss']-.04) < 1e-15
    assert all(abs(a-b) < 1e-12 for a, b in zip(metric['gradient'], (20., 5.)))
    metric_rng = random.Random(202611061)
    for _ in range(100):
        x = [metric_rng.uniform(-1., 1.) for _ in range(2)]
        scale = [metric_rng.uniform(.1, 2.) for _ in range(2)]
        tolerance = [metric_rng.uniform(.1, .5) for _ in range(2)]
        offset = [metric_rng.uniform(-1., 1.) for _ in range(2)]
        target = [metric_rng.uniform(-1., 1.) for _ in range(2)]
        result = affine_metric_example(x, scale, offset, target, tolerance)
        for axis in range(2):
            plus, minus = x.copy(), x.copy()
            plus[axis] += 1e-6
            minus[axis] -= 1e-6
            derivative = (affine_metric_example(plus, scale, offset, target, tolerance)['loss']
                          -affine_metric_example(minus, scale, offset, target, tolerance)['loss'])/2e-6
            assert abs(derivative-result['gradient'][axis])/max(1., abs(derivative)) < 2e-8
        # Convert coordinate 0 from m to mm, including its physical tolerance.
        converted = affine_metric_example(x, (1000*scale[0], scale[1]),
                                         (1000*offset[0], offset[1]),
                                         (1000*target[0], target[1]),
                                         (1000*tolerance[0], tolerance[1]))
        assert math.isclose(result['loss'], converted['loss'], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(result['normalized_loss'], converted['normalized_loss'],
                            rel_tol=1e-12, abs_tol=1e-12)
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                   for a, b in zip(result['gradient'], converted['gradient']))
    # An independent integer deadline scan checks all exact-tenth-ms fixtures.
    for delay_tenths in range(161):
        for period_tenths in (1, 2, 3, 7, 20):
            actual = first_valid_index(delay_tenths/10, 0, 0, period_tenths/10)
            expected = next(i for i in range(200) if i*period_tenths >= delay_tenths)
            assert actual == expected
    schedules = 0
    for h in range(1, 25):
        for s in range(1, h+1):
            for d in range(min(s, h-s)+1):
                weights = rtc_weights(h, s, d)
                assert len(weights) == h and weights[:d] == [1.]*d and weights[h-s:] == [0.]*s
                assert all(0 <= w <= 1 for w in weights)
                assert all(a >= b for a, b in zip(weights, weights[1:]))
                for i in range(d, h-s):
                    # Algebraically independent product c*(exp(c)-1)/(e-1).
                    c = (h-s-i)/(h-s-d+1)
                    expected = c*(1-math.exp(-c))/(math.exp(1-c)-math.exp(-c))
                    assert abs(weights[i]-expected) < 3e-15
                assert h-s >= d  # The old chunk covers the inference interval.
                schedules += 1
    # A fixed c=1/2 case catches omission of the multiplier printed in Eq. 5.
    half = rtc_weights(8, 3, 2)[3]
    assert abs(half-.5/(math.sqrt(math.e)+1)) < 1e-15
    assert abs(half-1/(math.sqrt(math.e)+1)) > .18
    upper = [(-1., 0.), (-.5, 1.), (.5, 1.), (1., 0.)]
    lower = [(x, -y) for x, y in upper]
    average = [((a[0]+b[0])/2, (a[1]+b[1])/2) for a, b in zip(upper, lower)]
    assert minimum_path_distance(upper) > .4 and minimum_path_distance(lower) > .4
    assert minimum_path_distance(average) == 0.
    invalid = [lambda: first_valid_index(-1, 0, 0, 20), lambda: first_valid_index(0, 0, 0, 0),
               lambda: first_valid_index(float('nan'), 0, 0, 20), lambda: rtc_weights(8, 2, 3),
               lambda: rtc_weights(8, 5, 4), lambda: rtc_weights(8, 0, 0), lambda: rtc_weights(True, 1, 0),
               lambda: affine_metric_example(tolerance=(0., .2)),
               lambda: affine_metric_example(scale=(-.1, 1.)),
               lambda: affine_metric_example(value=(math.nan, 0.))]
    for run in invalid:
        try:
            run()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print(f"PASS: 200 frame contracts, 805 exact deadline cases, {schedules} RTC schedules, "
          "24 guidance derivatives, convolution lengths, handover gap, path averaging counterexample")
    print('PASS: affine metric fixture, 200 metric derivatives and 100 unit conversions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
        return
    print('First valid index, 40 + 60 + 20 ms at 50 Hz:', first_valid_index(40, 60, 20, 20))
    print('Same delays plus 0.001 ms:', first_valid_index(40, 60.001, 20, 20))
    print('RTC H=16, s=5, d=4 weights:', ', '.join(f'{w:.4f}' for w in rtc_weights()))
    old, new, target = (1., 0., math.pi/2), (1., 1., 0.), (2., 1., math.pi/2)
    print('Old-anchor action:', relative(old, target))
    print('New-anchor action:', reanchor(relative(old, target), old, new))
    print('RMA conditional no-padding temporal lengths:', convolution_lengths())
    print('H=16, s=5, actual delay=12: uncovered slots:', uncovered_steps(16, 5, 12))
    loss, correction = linear_guidance()
    print('Constructed flow at tau=0.4: weighted loss=', f'{loss:.6f}',
          '; negative gradient=', ', '.join(f'{v:.6f}' for v in correction))
    metric = affine_metric_example()
    print('Affine local-action metric: loss=', f"{metric['loss']:.6f}",
          '; normalized Euclidean loss=', f"{metric['normalized_loss']:.6f}",
          '; positive gradient=', ', '.join(f'{v:.6f}' for v in metric['gradient']))


if __name__ == '__main__':
    main()
