#!/usr/bin/env python3
"""Standard-library teaching examples for the Light-Loco-Parkour article.

No third-party packages, model weights, simulator or robot are used.
The source-specific comparisons live separately in source_probe.py.
"""
import json
import math


def reward_memory(excess, dt=0.02, tau=0.06):
    if dt <= 0 or tau <= 0:
        raise ValueError('dt and tau must be positive')
    alpha = math.exp(-dt / tau)
    state = 0.0
    result = []
    for value in excess:
        if value < 0:
            raise ValueError('excess acceleration must be nonnegative')
        state = alpha * state + value
        result.append(state)
    return result


def phase_weights(position, triggers=(1., 2.), inverse_temperature=4.):
    if inverse_temperature <= 0 or any(a >= b for a, b in zip(triggers, triggers[1:])):
        raise ValueError('use increasing triggers and a positive inverse temperature')
    sigmoid = lambda value: 1. / (1. + math.exp(-value))
    switches = [1., *(sigmoid(inverse_temperature*(position-p)) for p in triggers), 0.]
    return [a-b for a, b in zip(switches, switches[1:])]


def gae(rewards, values, continuation, gamma=.9, lam=.8):
    if len(values) != len(rewards)+1 or len(continuation) != len(rewards):
        raise ValueError('supply T rewards, T continuation masks and T+1 values')
    advantage = 0.
    targets = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*continuation[t]*values[t+1]-values[t]
        advantage = delta+gamma*lam*continuation[t]*advantage
        targets.append(advantage+values[t])
    return list(reversed(targets))


def standardized(values, eps=1e-5):
    mean = sum(values)/len(values)
    variance = sum((x-mean)**2 for x in values)/len(values)
    return [(x-mean)/math.sqrt(variance+eps) for x in values]


def main():
    memory = reward_memory([20., 0., 0.])
    assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(memory,
        [20.,20.*math.exp(-1./3),20.*math.exp(-2./3)]))
    continuous = reward_memory([20.]*200)[-1]
    steady = 20./(1.-math.exp(-1./3))
    assert math.isclose(continuous, steady, rel_tol=1e-12)
    steady_100hz = 20./(1.-math.exp(-.01/.06))
    assert math.isclose(reward_memory([20.]*400, dt=.01)[-1], steady_100hz, rel_tol=1e-12)
    assert math.isclose(math.exp(-.02/.06)**3, math.exp(-.01/.06)**6)
    assert 1.84 < steady_100hz / steady < 1.85
    targets = gae([1.,2.],[.5,.4,.3],[1.,0.])
    assert all(math.isclose(a,b) for a,b in zip(targets,[2.512,2.]))
    timeout = gae([1.,2.+.9*.3],[.5,.4,.3],[1.,0.])
    assert math.isclose(timeout[-1],2.27)
    xs = [.9,1.,1.1,2.,2.1]
    weights = [phase_weights(x) for x in xs]
    assert all(all(w >= 0 for w in row) and math.isclose(sum(row),1.) for row in weights)
    hard = [sum(x>p for p in (1.,2.)) for x in xs]
    assert hard == [0,0,1,1,2]
    first, second = standardized([1.,2.,3.]), standardized([300.,200.,100.])
    weighted = [a+.5*b for a,b in zip(first,second)]
    assert weighted[0] < 0 and weighted[1] == 0 and weighted[2] > 0
    # Beta(2,2) under y=2x-1: density gains factor 1/2, entropy gains log(2).
    y = .4
    x = (y+1.)/2.
    shifted_density = 6.*x*(1.-x)/2.
    assert math.isclose(shifted_density,.75*(1.-y*y))
    weighted_mse=(1.*1.+9.*3.)/2.
    assert weighted_mse == 14. and weighted_mse != (1.+27.)/4.
    # Equal-mean Gaussians need not be equal distributions.
    kl_same_mean = math.log(2.)+1./8.-.5
    assert kl_same_mean > 0.
    result={
        'scope':'Deterministic teaching math; no model or robot performance',
        'acceleration_memory':memory,
        'continuous_excess_steady_value':steady,
        'same_tau_different_control_rates': {
            'steady_50hz': steady, 'steady_100hz': steady_100hz,
            'amplitude_ratio': steady_100hz / steady,
            'impulse_decay_after_60ms': math.exp(-1.),
        },
        'default_positive_weight_contribution':[.01*x for x in memory],
        'paper_negative_weight_contribution':[-.01*x for x in memory],
        'gae_value_targets':targets,
        'timeout_value_targets':timeout,
        'hard_phases':hard,
        'smooth_phase_weights':weights,
        'weighted_standardized_advantage':weighted,
        'weighted_distillation_valid_token_mean':weighted_mse,
        'weighted_distillation_sum_weight_mean':7.,
        'equal_mean_gaussian_KL':kl_same_mean,
        'shifted_beta_density_at_0_4':shifted_density,
        'shifted_beta_entropy':5./3.-math.log(3.),
        'passed':True,
    }
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
