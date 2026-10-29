"""Two-step survival-weighted return and critic targets; Python standard library.

Fixed scalar functions illustrate SuperMimic Equations 2/3. No learned model,
simulator, checkpoints, robot control or reported performance is reproduced.
Run without arguments for a JSON example, or --check for mathematical checks.
"""
import argparse
import json
import math
import random


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _episode(rewards, dones, gamma):
    rewards, dones = tuple(rewards), tuple(dones)
    if not rewards or len(rewards) != len(dones):
        raise ValueError("Supply equal nonempty reward/probability sequences")
    if (not all(_finite(x) for x in (*rewards, *dones, gamma))
            or not 0 <= gamma <= 1 or any(not 0 <= d <= 1 for d in dones)):
        raise ValueError("Use finite rewards and probabilities/discount in [0,1]")
    return rewards, dones


def finite_return(rewards, dones, tail, gamma=.9):
    """r[t+1] receives c[t], then d[t+1] changes future survival."""
    rewards, dones = _episode(rewards, dones, gamma)
    if not _finite(tail):
        raise ValueError("Finite tail value required")
    survival, discount, total = 1., 1., 0.
    for reward, done in zip(rewards, dones):
        total += discount * survival * reward
        survival *= 1. - done
        discount *= gamma
    result = total + discount * survival * tail
    if not _finite(result):
        raise ValueError("Return arithmetic overflow")
    return result


def lambda_returns(rewards, dones, values, gamma=.9, lam=.95):
    """Fixed critic targets from a soft-probability trajectory, not a fit."""
    rewards, dones = _episode(rewards, dones, gamma)
    values = tuple(values)
    if (len(values) != len(rewards) + 1 or not all(_finite(v) for v in values)
            or not _finite(lam) or not 0 <= lam <= 1):
        raise ValueError("Supply T+1 finite values and lambda in [0,1]")
    target = values[-1]
    result = []
    for t in reversed(range(len(rewards))):
        target = rewards[t] + gamma * (1.-dones[t]) * (
            (1.-lam)*values[t+1] + lam*target)
        if not _finite(target):
            raise ValueError("Target arithmetic overflow")
        result.append(target)
    return list(reversed(result))


def _sigmoid(value):
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1. + exponential)


def _trace(theta):
    states = [0., 2.*theta, 4.*theta]
    rewards = [1.-(s-1.)**2 for s in states[1:]]
    dones = [_sigmoid(5.*(s-.6)) for s in states[1:]]
    values = [1.2-(s-.8)**2 for s in states]
    return states, rewards, dones, values


def evaluate(theta=.1, gamma=.9, *, stop_survival=False, stop_value=False):
    """Detach switches remove backward paths, keeping the forward value fixed."""
    if not _finite(theta) or not -1 <= theta <= 1:
        raise ValueError("This example uses finite theta in [-1,1]")
    if not _finite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Use finite gamma in [0,1]")
    if type(stop_survival) is not bool or type(stop_value) is not bool:
        raise ValueError("Detach switches must be booleans")
    states, rewards, dones, values = _trace(theta)
    s1, s2 = states[1:]
    r1, r2 = rewards
    d1, d2 = dones
    c1, c2 = 1.-d1, (1.-d1)*(1.-d2)
    dr1, dr2 = -4.*(s1-1.), -8.*(s2-1.)
    dd1, dd2 = 10.*d1*(1.-d1), 20.*d2*(1.-d2)
    dc1, dc2 = -dd1, -dd1*(1.-d2) - (1.-d1)*dd2
    dv2 = -8.*(s2-.8)
    paths = {
        "reward_state": dr1 + gamma*c1*dr2,
        "survival": 0. if stop_survival else gamma*dc1*r2 + gamma**2*dc2*values[-1],
        "tail_state": 0. if stop_value else gamma**2*c2*dv2,
    }
    epsilon = 1e-6
    def value_at(parameter):
        _, r, d, v = _trace(parameter)
        return finite_return(r, d, v[-1], gamma)
    return {
        "theta": theta, "gamma": gamma,
        "stop_survival": stop_survival, "stop_value": stop_value,
        "states": states, "rewards": rewards, "termination_probabilities": dones,
        "survival_weights": [1., c1, c2], "critic_values": values,
        "objective": finite_return(rewards, dones, values[-1], gamma),
        "gradient_paths": paths, "gradient": sum(paths.values()),
        "finite_difference_full": (value_at(theta+epsilon)-value_at(theta-epsilon))/(2.*epsilon),
        "lambda_targets_095": lambda_returns(rewards, dones, values, gamma),
        "lambda_targets_1": lambda_returns(rewards, dones, values, gamma, lam=1.),
    }


def self_check():
    def near(actual, expected, message):
        if not math.isclose(actual, expected, rel_tol=2e-7, abs_tol=2e-8):
            raise AssertionError(f"{message}: {actual} != {expected}")

    # A terminating transition retains its own reward and removes later values.
    near(finite_return([3., 5.], [1., 0.], 7.), 3., "first transition terminates")
    near(finite_return([3., 5.], [0., 1.], 7.), 7.5, "second transition terminates")
    near(finite_return([3., 5.], [0., 0.], 7.), 13.17, "no termination")
    if lambda_returns([3., 5.], [1., 1.], [9., 11., 7.]) != [3., 5.]:
        raise AssertionError("Terminated critic targets must be immediate rewards")
    full = evaluate()
    near(full["objective"], 1.4097724894448882, "default return")
    near(full["gradient"], 4.50505361235868, "default gradient")
    near(evaluate(stop_survival=True)["gradient"], 8.674069138494076, "detached survival")
    near(evaluate(stop_value=True)["gradient"], 2.8360278507290557, "detached critic")
    near(evaluate(gamma=0.)["objective"], .36, "zero discount keeps first reward")
    near(evaluate(gamma=0.)["gradient"], 3.2, "zero discount removes future paths")
    undiscounted = evaluate(gamma=1.)
    near(undiscounted["gradient"], undiscounted["finite_difference_full"], "finite undiscounted horizon")
    rng = random.Random(20261029)
    for _ in range(500):
        theta, gamma = rng.uniform(-1., 1.), rng.random()
        full = evaluate(theta, gamma)
        near(full["gradient"], full["finite_difference_full"], "full objective derivative")
        near(full["lambda_targets_1"][0], full["objective"], "lambda=1 unrolls full return")
        for stop_survival, stop_value in [(True, False), (False, True), (True, True)]:
            stopped = evaluate(theta, gamma, stop_survival=stop_survival, stop_value=stop_value)
            if stopped["objective"] != full["objective"]:
                raise AssertionError("Detach changed the forward value")
        _, rewards, dones, values = _trace(theta)
        one_step = lambda_returns(rewards, dones, values, gamma, lam=0.)
        for t in range(2):
            near(one_step[t], rewards[t]+gamma*(1.-dones[t])*values[t+1], "lambda=0 one-step target")
    for kwargs in ({"theta": math.nan}, {"gamma": math.inf}, {"theta": True},
                   {"gamma": -1.}, {"gamma": 1.1}, {"theta": 2.}, {"stop_value": 1}):
        try:
            evaluate(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid evaluation input was accepted")
    for rewards, dones in [([], []), ([1.], []), ([math.nan], [0.]),
                           ([1.], [-.1]), ([1.], [1.1]), ([1.], [True])]:
        try:
            finite_return(rewards, dones, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid episode was accepted")
    return {"random_trajectories": 500, "finite_difference": "passed",
            "lambda_0_and_1": "passed", "terminal_reward_index": "passed",
            "detach_preserves_forward": "passed", "invalid_inputs": "rejected"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=.1)
    parser.add_argument("--gamma", type=float, default=.9)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = {"scope": "constructed scalar functions; not a trained SuperMimic model",
                  "assumptions": "s0=0; a0=a1=theta; s_next=s+2a; r(s)=1-(s-1)^2; d(s)=sigmoid(5*(s-.6)); V(s)=1.2-(s-.8)^2; H=2; fixed dynamics/reward/termination/critic parameters",
                  "full": evaluate(args.theta, args.gamma),
                  "detach_survival": evaluate(args.theta, args.gamma, stop_survival=True),
                  "detach_value": evaluate(args.theta, args.gamma, stop_value=True)}
        if args.check:
            result["checks"] = self_check()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
