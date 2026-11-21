#!/usr/bin/env python3
"""Constructed probability, pose and score checks for the PEAR article.

Standard library only. No model, simulator, device, network or training run.
The Bernoulli calculator assumes independent trials with a fixed success
probability. It does not estimate the paper's actual sampling cost.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math

DEFAULT = {"probability": 2.0, "group": 8, "groups": 128}
REAL_COUNTS = (("base", 0, 30), ("libero_only", 7, 30), ("final", 18, 30))


def finite(value, name):
    if type(value) not in (int, float):
        raise ValueError(name + ": finite number required")
    try:
        value=float(value)
    except (ValueError,OverflowError):
        raise ValueError(name + ": finite floating-point value required") from None
    if not math.isfinite(value):
        raise ValueError(name + ": finite number required")
    return value


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: integer in [{low}, {high}] required")
    return value


def sampling(probability=2.0, group=8, groups=128):
    """Probability is a PERCENT. Groups are sampled without policy updates.

    Expected candidate rollouts include all discarded groups. The conditional
    success fraction is over the mixed groups that are retained, not an
    evaluation success rate. None means no finite estimate is available.
    """
    p = finite(probability, "probability") / 100.0
    if not 0 <= p <= 1:
        raise ValueError("probability: percent in [0, 100] required")
    integer(group, "group", 2, 64)
    integer(groups, "groups", 1, 1024)
    q = min(p, 1-p)
    # Symmetric, stable near either endpoint; direct 1-p**G-(1-p)**G
    # loses tiny positive mixed probabilities by cancellation.
    mixed = 0.0 if q == 0 else -math.expm1(group * math.log1p(-q)) - q**group
    raw = groups/mixed if mixed else None
    rollouts = raw*group if raw is not None else None
    retained = p * -math.expm1((group-1)*math.log(p)) / mixed if mixed else None
    def available(value):
        return value if value is None or math.isfinite(value) else None
    return {"all_fail": (1-p)**group, "all_success": p**group,
            "mixed": mixed, "candidate_groups": available(raw),
            "candidate_rollouts": available(rollouts),
            "retained_success": available(retained), "accepted_rollouts": group*groups}


def binary_advantages(rewards):
    """Population standard deviation, epsilon=0 for this explicit toy model."""
    rewards = tuple(rewards)
    if not rewards or any(type(r) is not int or r not in (0, 1) for r in rewards):
        raise ValueError("rewards: nonempty binary integer sequence required")
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(mean*(1-mean))
    return tuple((r-mean)/std if std else 0.0 for r in rewards)


def retained_task_weights(probabilities, weights, group=8):
    """Fixed task-mixture illustration; each task has its own success percent."""
    probabilities,weights=tuple(probabilities),tuple(weights)
    if not probabilities or len(probabilities)!=len(weights):
        raise ValueError('one nonempty weight per task required')
    weights=tuple(finite(w,'task weight') for w in weights)
    if any(w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('nonnegative weights with positive total required')
    maximum=max(weights)
    retained=tuple(w/maximum*sampling(p,group,1)['mixed'] for p,w in zip(probabilities,weights))
    total=sum(retained)
    return tuple(w/total for w in retained) if total else None


def wilson(successes, trials, z=1.959963984540054):
    """Two-sided 95% Wilson interval UNDER an IID binomial assumption.

    Fixed object/scene pairs in the paper do not establish this assumption;
    these intervals illustrate sample size rather than certify robot reliability.
    """
    integer(trials, "trials", 1, 10**9)
    integer(successes, "successes", 0, trials)
    z = finite(z, "z")
    if z <= 0:
        raise ValueError("z must be positive")
    p = successes/trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denominator
    return max(0.0, center-half), min(1.0, center+half)


def rpy_matrix(roll, pitch, yaw):
    """Fixed-axis Rz(yaw) Ry(pitch) Rx(roll), angles in radians."""
    roll, pitch, yaw = (finite(v, "angle") for v in (roll, pitch, yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def matmul(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def world_target(current_cm, delta_cm, base_yaw, origin_m):
    """A constructed position example: additive base-frame quantities in cm.

    Only a z-rotated base is represented here; no IK, collision, orientation
    interpolation, robot dynamics or gripper actuation is implemented.
    """
    values = [tuple(finite(v, "coordinate") for v in row) for row in (current_cm, delta_cm, origin_m)]
    if any(len(row) != 3 for row in values):
        raise ValueError("three coordinates required")
    current, change, origin = values
    rotation = rpy_matrix(0, 0, base_yaw)
    local = tuple((a+b)/100 for a, b in zip(current, change))
    return tuple(origin[i]+sum(rotation[i][j]*local[j] for j in range(3)) for i in range(3))


def score_example():
    """One prefix, two logits, fixed rollout distribution mu.

    A centered score has zero expectation under mu. With action-dependent
    rewards, this alone does NOT equal the on-policy reward gradient under pi.
    This is an algebraic counterexample, not a full GRPO implementation.
    """
    pi, mu, rewards = (0.5, 0.5), (0.9, 0.1), (0.0, 1.0)
    scores = tuple(tuple(float(v == j)-pi[j] for j in range(2)) for v in range(2))
    offset = tuple(sum(mu[v]*scores[v][j] for v in range(2)) for j in range(2))
    centered = tuple(tuple(scores[v][j]-offset[j] for j in range(2)) for v in range(2))
    expectation = tuple(sum(mu[v]*centered[v][j] for v in range(2)) for j in range(2))
    rollout_reward_gradient = tuple(sum(mu[v]*rewards[v]*centered[v][j] for v in range(2)) for j in range(2))
    on_policy_reward_gradient = tuple(sum(pi[v]*rewards[v]*scores[v][j] for v in range(2)) for j in range(2))
    return {"pi": pi, "mu": mu, "raw_score_expectation": offset,
            "centered_score_expectation": expectation,
            "rollout_centered_reward_gradient": rollout_reward_gradient,
            "on_policy_reward_gradient": on_policy_reward_gradient}


def reference():
    return {"default_sampling": sampling(**DEFAULT),
            "low_probability_sampling": sampling(0.4),
            "task_mixture_illustration": {"success_percent":[2,50],"candidate_weights":[.5,.5],
                                          "retained_weights":retained_task_weights((2,50),(.5,.5))},
            "real_trials": [{"model": name, "successes": k, "trials": n,
                             "fraction": k/n, "iid_wilson_illustration": wilson(k,n)}
                            for name,k,n in REAL_COUNTS],
            "position_target_world_m": world_target((40,20,30),(0,0,-5),math.pi/2,(1,2,.5)),
            "score_example": score_example(),
            "decode_only_seconds_for_3230_tokens": {str(rate):3230/rate for rate in (52,221,747)},
            "scope": "Constructed IID sampling, coordinates and score algebra; reported trial counts only. No policy or robot execution."}


def check():
    cases = 0
    # Independent exact binomial summation, rather than the production shortcut.
    for g in range(2, 13):
        for p in (Fraction(0),Fraction(1,250),Fraction(1,50),Fraction(1,4),Fraction(1,2),Fraction(9,10),Fraction(1)):
            terms = [Fraction(math.comb(g,k))*p**k*(1-p)**(g-k) for k in range(g+1)]
            mixed = sum(terms[1:-1])
            result = sampling(float(p*100),g,3)
            assert math.isclose(result["mixed"],float(mixed),rel_tol=2e-13,abs_tol=1e-15)
            if mixed:
                retained = sum(Fraction(k,g)*terms[k] for k in range(1,g))/mixed
                assert math.isclose(result["retained_success"],float(retained),rel_tol=2e-13)
                assert math.isclose(result["candidate_rollouts"],float(Fraction(3*g)/mixed),rel_tol=2e-13)
            else:
                assert result['candidate_rollouts'] is None and result['retained_success'] is None
            cases += 1
    for p in (1e-200,1e-12,99.9999999999):
        r=sampling(p)
        assert r['mixed']>0 and r['candidate_rollouts']>0
    assert retained_task_weights((0,100),(.5,.5)) is None
    assert retained_task_weights((0,2,100),(1,1,1))==(0.0,1.0,0.0)
    assert retained_task_weights((2,50),(.5,.5))==retained_task_weights((2,50),(1e300,1e300))
    m1=1-Fraction(49,50)**8-Fraction(1,50)**8
    m2=1-2*Fraction(1,2)**8
    assert math.isclose(retained_task_weights((2,50),(.5,.5))[0],float(m1/(m1+m2)),rel_tol=1e-14)
    # Every group of eight, using an independent high-precision standard deviation.
    with localcontext() as ctx:
        ctx.prec=60
        for mask in range(256):
            rewards=tuple((mask>>i)&1 for i in range(8))
            mean=Decimal(sum(rewards))/8
            std=(sum((Decimal(r)-mean)**2 for r in rewards)/8).sqrt()
            expected=tuple((Decimal(r)-mean)/std if std else Decimal(0) for r in rewards)
            actual=binary_advantages(rewards)
            assert all(abs(a-float(b))<1e-14 for a,b in zip(actual,expected))
            assert abs(sum(actual))<1e-13
    target=world_target((40,20,30),(0,0,-5),math.pi/2,(1,2,.5))
    assert all(abs(a-b)<1e-14 for a,b in zip(target,(.8,2.4,.75)))
    for yaw in (-2,-.3,0,.7,math.pi):
        angle=math.pi/2+yaw
        point=world_target((40,20,30),(0,0,-5),angle,(0,0,0))
        assert math.isclose(sum(v*v for v in point),.4**2+.2**2+.25**2,rel_tol=1e-14)
    a,b=(.2,.3,.4),(.1,-.2,.2)
    summed=rpy_matrix(*(x+y for x,y in zip(a,b)))
    composed=matmul(rpy_matrix(*a),rpy_matrix(*b))
    assert max(abs(summed[i][j]-composed[i][j]) for i in range(3) for j in range(3))>.01
    # Wilson endpoints solve the score-test quadratic, independently of its closed form.
    z=1.959963984540054
    for k,n in [(0,30),(7,30),(18,30),(30,30),(1,1),(8,10)]:
        for endpoint in wilson(k,n):
            assert abs(n*(k/n-endpoint)**2-z*z*endpoint*(1-endpoint))<1e-12
    s=score_example()
    assert all(abs(x)<1e-14 for x in s['centered_score_expectation'])
    assert math.isclose(s['rollout_centered_reward_gradient'][1],.09,abs_tol=1e-14)
    assert s['on_policy_reward_gradient'][1]==.25
    invalid=[lambda:sampling(True),lambda:sampling(float('nan')),lambda:sampling(101),
             lambda:sampling(10**400),lambda:retained_task_weights((2,),(0,)),
             lambda:sampling(group=1),lambda:sampling(group=8.0),lambda:sampling(groups=0),
             lambda:binary_advantages((True,0)),lambda:wilson(31,30),
             lambda:world_target((1,2),(0,0,0),0,(0,0,0))]
    for operation in invalid:
        try: operation()
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    print(json.dumps({"binomial_cases":cases,"binary_groups":256,"pose_score_wilson_invalid_checks":"passed"}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check: check()
    else: print(json.dumps(reference(),ensure_ascii=False,indent=2,allow_nan=False))
