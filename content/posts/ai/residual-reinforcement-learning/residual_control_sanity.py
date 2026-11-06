#!/usr/bin/env python3
"""Residual action bookkeeping and a scalar feedback example; no RL training.

Scalar plant: e[k+1] = e[k] + dt * (u[k] + bias). The nominal command is
-gain*e. Units are illustrative and must not be treated as a robot model.
"""
import argparse
import math
import random


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def clamp(value, low, high):
    value, low, high = [number(v, "clamp value") for v in (value, low, high)]
    if low > high:
        raise ValueError("lower bound exceeds upper bound")
    return min(high, max(low, value))


def feasible_residual(nominal, total_low=-1., total_high=1., radius=None):
    nominal, total_low, total_high = [number(v, "action") for v in (nominal, total_low, total_high)]
    if total_low > total_high:
        raise ValueError("invalid total action bounds")
    low, high = total_low-nominal, total_high-nominal
    if radius is not None:
        radius = number(radius, "residual radius")
        if radius < 0:
            raise ValueError("residual radius must be nonnegative")
        low, high = max(low, -radius), min(high, radius)
    return None if low > high else (low, high)


def executed_action(nominal, residual, low=-1., high=1.):
    nominal, residual = number(nominal, "nominal"), number(residual, "residual")
    return clamp(nominal+residual, low, high)


def clipped_exploration(nominal=.9, amplitude=.2, low=-1., high=1.):
    """Equal-probability +/- amplitude noise before total-action clipping."""
    nominal, amplitude, low, high = [number(v, 'exploration parameter')
                                     for v in (nominal, amplitude, low, high)]
    if amplitude < 0 or not low <= nominal <= high:
        raise ValueError('expected a nonnegative amplitude and feasible nominal action')
    actions = [executed_action(nominal, noise, low, high)
               for noise in (-amplitude, amplitude)]
    mean = math.fsum(actions)/2
    return {'actions':actions, 'mean':mean, 'mean_correction':mean-nominal,
            'variance':math.fsum((u-mean)**2 for u in actions)/2}


def scalar_transition(error, residual, gain=1., dt=.1, bias=0., total_limit=None):
    error, residual, gain, dt, bias = [number(v, "scalar parameter") for v in (error, residual, gain, dt, bias)]
    if gain <= 0 or dt <= 0:
        raise ValueError("gain and dt must be positive")
    total = -gain*error+residual
    if total_limit is not None:
        total_limit = number(total_limit, "total limit")
        if total_limit < 0:
            raise ValueError("total limit must be nonnegative")
        total = clamp(total, -total_limit, total_limit)
    return error+dt*(total+bias)


def trajectory(error=1., gain=1., dt=.1, bias=.2, constant_residual=0., residual_gain=0., steps=60, total_limit=None):
    error, constant_residual, residual_gain = [number(v, "trajectory parameter") for v in (error, constant_residual, residual_gain)]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    result = [error]
    for _ in range(steps):
        error = scalar_transition(error, constant_residual+residual_gain*error, gain, dt, bias, total_limit)
        result.append(error)
    return result


def bounded_residual(error, amplitude=.2, steepness=10.):
    """A bounded nonlinear residual for a separate continuous scalar example."""
    error, amplitude, steepness = [number(v, 'bounded residual parameter')
                                  for v in (error, amplitude, steepness)]
    if amplitude < 0 or steepness < 0:
        raise ValueError('amplitude and steepness must be nonnegative')
    return amplitude*math.tanh(steepness*error)


def bounded_counterexample():
    """Fixed dot(e)=-e+0.2*tanh(10e): unstable zero, bounded state."""
    low, high = .1, .2
    for _ in range(64):
        middle = (low+high)/2
        if -middle+bounded_residual(middle) > 0:
            low = middle
        else:
            high = middle
    root = (low+high)/2
    return {'local_slope': 1., 'ultimate_bound': .2,
            'positive_equilibrium': root,
            'equilibrium_slope': -1+2*(1-math.tanh(10*root)**2)}


def disturbance_bound(initial, gain, dt, disturbance, steps):
    initial, gain, dt, disturbance = [number(v, "bound parameter") for v in (initial, gain, dt, disturbance)]
    if gain <= 0 or dt <= 0 or disturbance < 0:
        raise ValueError("invalid disturbance bound parameters")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    q = abs(1-gain*dt)
    if q >= 1:
        raise ValueError("the nominal discrete-time system is not contractive")
    return q**steps*abs(initial)+dt*disturbance*(1-q**steps)/(1-q)


def td3_target(reward, gamma, terminated, q1, q2):
    reward, gamma, q1, q2 = [number(v, "target parameter") for v in (reward, gamma, q1, q2)]
    if not 0 <= gamma <= 1 or not isinstance(terminated, bool):
        raise ValueError("invalid discount or termination flag")
    return reward if terminated else reward+gamma*min(q1, q2)


def actor_chain_example(theta=(.2, -.3), saturated=False):
    """Fixed quadratic physical-action critic and a nonlinear 2D actor.

    Returns a surrogate objective and its analytic chain-rule gradient.
    The scaling matrix is deliberately non-diagonal to expose transposes.
    """
    if len(theta) != 2 or not isinstance(saturated, bool):
        raise ValueError("expected two actor parameters and a boolean flag")
    x, y = (number(v, "actor parameter") for v in theta)
    action = [x+2*y, x*y]
    scale = [[2., .5], [-.25, 3.]]
    nominal, target = [.4, -.2], [.1, .3]
    raw = [nominal[i]+sum(scale[i][j]*action[j] for j in range(2)) for i in range(2)]
    physical = [clamp(v, -1., 1.) for v in raw] if saturated else raw
    value = -(physical[0]-target[0])**2-2*(physical[1]-target[1])**2
    physical_gradient = [-2*(physical[0]-target[0]), -4*(physical[1]-target[1])]
    if saturated:
        physical_gradient = [g if -1 < u < 1 else 0. for g, u in zip(physical_gradient, raw)]
    action_gradient = [sum(scale[i][j]*physical_gradient[i] for i in range(2)) for j in range(2)]
    jacobian = [[1., 2.], [y, x]]
    gradient = [sum(jacobian[i][j]*action_gradient[i] for i in range(2)) for j in range(2)]
    return value, gradient, raw


def check():
    rng = random.Random(20261106)
    for _ in range(500):
        state = rng.uniform(-3, 3)
        residual = rng.uniform(-2, 2)
        bias = rng.uniform(-.5, .5)
        gain, dt = rng.uniform(.3, 2), rng.uniform(.01, .3)
        # Compare an induced residual transition with the original physical
        # plant and one explicitly summed command, including saturation.
        total = min(1., max(-1., -gain*state+residual))
        direct = state+dt*(total+bias)
        induced = scalar_transition(state, residual, gain, dt, bias, 1.)
        assert abs(direct-induced) < 1e-14
        radius = rng.uniform(0, 2)
        feasible = feasible_residual(-gain*state, radius=radius)
        for candidate in (-radius, -.1*radius, 0., .1*radius, radius):
            physical_ok = -1 <= -gain*state+candidate <= 1
            interval_ok = feasible is not None and feasible[0] <= candidate <= feasible[1]
            assert physical_ok == interval_ok
        # Either residual-action or physical-action Q bookkeeping gives the
        # same value when the baseline is converted exactly once.
        nominal = -.7*state
        q_residual = lambda a: -(a-.3*state)**2+.1*state
        q_physical = lambda u: -(u-nominal-.3*state)**2+.1*state
        assert abs(q_residual(residual)-q_physical(nominal+residual)) < 1e-12
        # Fixed linear-addition interfaces permit physical-action relabeling
        # after a baseline change. This identity does not invert saturation.
        new_nominal = .3*state
        relabeled = nominal+residual-new_nominal
        assert abs(new_nominal+relabeled-(nominal+residual)) < 1e-14
    assert feasible_residual(.9) == (-1.9, .09999999999999998)
    assert feasible_residual(1.5, radius=.2) is None
    assert feasible_residual(1., radius=0.) == (0., 0.)
    assert executed_action(.9, .4) == 1.
    assert clamp(.9, -1, 1)+clamp(.4, -1, 1) > 1.
    from fractions import Fraction
    noise_cases = 0
    for tenth in range(-10, 11):
        for magnitude in (Fraction(0), Fraction(1, 20), Fraction(1, 5),
                          Fraction(1, 2), Fraction(9, 10), Fraction(13, 10)):
            baseline = Fraction(tenth, 10)
            # Exact rational expectations, independent of the float helper.
            exact = [max(Fraction(-1), min(Fraction(1), baseline+sign*magnitude))
                     for sign in (-1, 1)]
            mean = sum(exact)/2
            variance = sum((u-mean)**2 for u in exact)/2
            result = clipped_exploration(float(baseline), float(magnitude))
            assert abs(result['mean']-float(mean)) < 1e-14
            assert abs(result['variance']-float(variance)) < 1e-14
            assert abs(result['mean_correction']-float(mean-baseline)) < 1e-14
            noise_cases += 1
    assert abs(clipped_exploration()['mean']-.85) < 1e-14
    for dt in (.01, .1, .9, 1.5):
        gain, bias, residual, initial = 1., .2, -.1, 1.
        values = trajectory(initial, gain, dt, bias, residual, steps=80)
        equilibrium = (bias+residual)/gain
        for i, value in enumerate(values):
            independent = equilibrium+(initial-equilibrium)*(1-gain*dt)**i
            assert abs(value-independent) < 3e-14
            assert abs(value) <= disturbance_bound(initial, gain, dt, abs(bias+residual), i)+3e-14
    # Nonconstant bounded disturbances exercise the geometric-series bound.
    for _ in range(100):
        gain, dt, disturbance = rng.uniform(.2, 2), .1, rng.uniform(0, .5)
        initial = error = rng.uniform(-2, 2)
        for i in range(1, 101):
            forcing = rng.uniform(-disturbance, disturbance)
            error = scalar_transition(error, forcing, gain, dt)
            assert abs(error) <= disturbance_bound(initial, gain, dt, disturbance, i)+1e-12
    corrected = trajectory(bias=.2, constant_residual=-.2)
    assert corrected[-1] < .002
    divergent = trajectory(bias=0., residual_gain=1.2)
    assert divergent[-1] > divergent[0]
    # A separate continuous example retains a global amplitude bound while
    # losing local stability at zero. Its nonzero equilibrium is independently
    # solved using Decimal exp rather than the published math.tanh function.
    bounded = bounded_counterexample()
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 70
        low, high = Decimal('.1'), Decimal('.2')
        for _ in range(160):
            middle = (low+high)/2
            exponential = (20*middle).exp()
            field = -middle+Decimal('.2')*(exponential-1)/(exponential+1)
            if field > 0:
                low = middle
            else:
                high = middle
        independent_root = float((low+high)/2)
    assert abs(bounded['positive_equilibrium']-independent_root) < 1e-15
    assert bounded['equilibrium_slope'] < 0
    h = 1e-6
    derivative = ((-h+bounded_residual(h))- (h+bounded_residual(-h)))/(2*h)
    assert abs(derivative-bounded['local_slope']) < 1e-8
    assert h*(-h+bounded_residual(h)) > 0
    bounded_rng = random.Random(202611062)
    for _ in range(1000):
        error = bounded_rng.uniform(-3, 3)
        amplitude, steepness, gain = bounded_rng.uniform(0, 1), bounded_rng.uniform(0, 20), bounded_rng.uniform(.1, 3)
        residual = bounded_residual(error, amplitude, steepness)
        assert abs(residual) <= amplitude
        vdot = error*(-gain*error+residual)
        assert vdot <= -gain*error**2+amplitude*abs(error)+1e-13
        if abs(error) > amplitude/gain:
            assert vdot < 0
    assert td3_target(-1., .99, True, 100., 100.) == -1.
    assert td3_target(-1., .9, False, 2., 3.) == .8
    actor_derivatives = 0
    for saturated in (False, True):
        for _ in range(150):
            theta = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
            value, gradient, raw = actor_chain_example(theta, saturated)
            if saturated and min(abs(abs(v)-1.) for v in raw) < 1e-4:
                continue  # No derivative is claimed exactly at a clipping kink.
            for axis in range(2):
                plus, minus = theta.copy(), theta.copy()
                plus[axis] += 1e-6
                minus[axis] -= 1e-6
                numerical = (actor_chain_example(plus, saturated)[0]-actor_chain_example(minus, saturated)[0])/2e-6
                assert abs(numerical-gradient[axis])/max(1., abs(gradient[axis])) < 2e-8
                actor_derivatives += 1
    bad = [lambda: feasible_residual(0, 1, -1), lambda: feasible_residual(0, radius=-1),
           lambda: scalar_transition(1, 0, dt=0), lambda: scalar_transition(1, float('nan')),
           lambda: trajectory(steps=True), lambda: disturbance_bound(1, 1, 2, 0, 10),
           lambda: td3_target(0, 1.1, False, 0, 0),
           lambda: bounded_residual(True), lambda: bounded_residual(0, amplitude=-1),
           lambda: bounded_residual(0, steepness=-1),
           lambda: clipped_exploration(amplitude=-.1),
           lambda: clipped_exploration(nominal=1.1)]
    for run in bad:
        try:
            run()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print('PASS: 500 residual-MDP/action contracts, 4 analytic trajectories, 10000 bounded-disturbance steps, '
          f'{actor_derivatives} actor derivatives, TD3 terminal targets, 1000 bounded nonlinear Lyapunov cases')
    print(f'PASS: {noise_cases} exact clipped-exploration expectations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
        return
    print('Total limits [-1,1], baseline 0.9: residual interval', feasible_residual(.9))
    print('Requested residual 0.4: executed total', executed_action(.9, .4), '; effective residual 0.1')
    for label, options in [('nominal, bias +0.2', {}), ('residual -0.2', {'constant_residual':-.2}),
                           ('wrong feedback +1.2*e', {'bias':0., 'residual_gain':1.2})]:
        values = trajectory(**options)
        print(f'{label}: e[0]={values[0]:.6f}, e[60]={values[-1]:.6f}')
    value, gradient, raw = actor_chain_example()
    print('Constructed physical-action critic: value=', f'{value:.6f}',
          '; actor gradient=', ', '.join(f'{v:.6f}' for v in gradient))
    bounded = bounded_counterexample()
    print('Separate continuous bounded residual: local slope=', bounded['local_slope'],
          '; positive equilibrium=', f"{bounded['positive_equilibrium']:.12f}",
          '; ultimate bound=', bounded['ultimate_bound'])
    noise = clipped_exploration()
    print('Equal +/-0.2 exploration at nominal0.9: executed=', noise['actions'],
          '; mean=', f"{noise['mean']:.6f}",
          '; mean correction=', f"{noise['mean_correction']:.6f}")


if __name__ == '__main__':
    main()
