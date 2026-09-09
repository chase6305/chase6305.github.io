"""Original bookkeeping examples; no GEN-1.5 model, API or robot control."""
import json
import math


def context_budget(window, prompts, live, trajectory_hz):
    values = [window, live, trajectory_hz, *prompts]
    if any(not math.isfinite(x) or x < 0 for x in values) or window <= 0 or trajectory_hz <= 0:
        raise ValueError('Use finite nonnegative durations and positive window/rate')
    used = sum(prompts) + live
    if used > window:
        raise ValueError('Prompts plus live history exceed the proposed window')
    return dict(used_seconds=used, free_seconds=window-used,
                nominal_action_samples=used*trajectory_hz)


def wilson(successes, trials, z=1.959963984540054):
    if (type(trials) is not int or type(successes) is not int or
            trials <= 0 or not 0 <= successes <= trials):
        raise ValueError('Use integer counts with 0 <= successes <= trials and trials > 0')
    if not math.isfinite(z) or z <= 0:
        raise ValueError('z must be positive and finite')
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(tasks):
    if not tasks:
        raise ValueError('At least one task is required')
    for successes, trials in tasks:
        wilson(successes, trials)
    return dict(macro=sum(s/n for s, n in tasks)/len(tasks),
                micro=sum(s for s, n in tasks)/sum(n for s, n in tasks))


if __name__ == '__main__':
    print(json.dumps(dict(scope='invented examples; not reported GEN-1.5 trial counts',
        context=context_budget(30, [6, 8], 12, 100),
        evaluation=summarize([(9, 10), (10, 20)]),
        illustrative_19_of_30_wilson95=wilson(19, 30)), indent=2))
