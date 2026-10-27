#!/usr/bin/env python3
"""Recompute the article's constructed examples and rounded Table I comparisons.

Python standard library only. This does not train a policy, load robot assets,
run hardware or reproduce the authors' tracking evaluation.
"""
import argparse
import json
import math
from pathlib import Path

# Technical report, Table I: rounded published values, not trial-level records.
TABLE_I = [
    ('G1', 'full-motion', .9306, .9358, 99.797, 99.747),
    ('X2 Ultra', 'full-motion', .9253, .9304, 99.818, 99.891),
    ('T2', 'full-motion', .9448, .9398, 99.860, 99.886),
    ('G1', 'five-point', .8807, .8626, 99.849, 99.729),
    ('X2 Ultra', 'five-point', .8689, .8774, 99.841, 99.880),
    ('T2', 'five-point', .9014, .8890, 99.857, 99.852),
]


def canonical_position(native, permutation, signs, offsets):
    return [sign * (native[index] - offset)
            for index, sign, offset in zip(permutation, signs, offsets)]


def native_position(canonical, permutation, signs, offsets, default):
    result = list(default)
    for value, index, sign, offset in zip(canonical, permutation, signs, offsets):
        result[index] = sign * value + offset
    return result


def compare_record(expected, actual, path='$'):
    """Compare a numerical record without requiring identical float accumulation.

    Integer, boolean and structural fields stay exact. Float tolerance is far
    below the report's four-decimal precision and the 1e-9 gradient check bound.
    """
    if type(expected) is not type(actual):
        raise ValueError(f'{path}: type differs')
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f'{path}: keys differ')
        for key in expected:
            compare_record(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f'{path}: length differs')
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_record(left, right, f'{path}[{index}]')
    elif isinstance(expected, float):
        if not (math.isfinite(expected) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'{path}: number differs ({expected!r} versus {actual!r})')
    elif expected != actual:
        raise ValueError(f'{path}: value differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--output', type=Path, help='write JSON; omitted means stdout only')
    mode.add_argument('--check', action='store_true', help='verify the published JSON record')
    args = parser.parse_args()
    dimensions = dict(proprioception=3 * 29 + 6, history=10 * 93,
                      full_goal=10 * (29 + 29 + 3 + 6),
                      five_point=10 * 5 * 9 + 5 + 10,
                      morphology=4 * 29 + 9,
                      decoder_input=10 * 93 + 32 + 125,
                      world_model_input=268 + 17 + 29 + 33 + 125)
    assert dimensions == dict(proprioception=93, history=930, full_goal=670,
                              five_point=465, morphology=125,
                              decoder_input=1087, world_model_input=472)
    # Constructed three-joint example with one extra, untracked native joint.
    native = [.5, -.2, .9, 1.7]
    permutation, signs, offsets = [2, 0, 1], [-1, 1, -1], [.1, .2, -.3]
    canonical = canonical_position(native, permutation, signs, offsets)
    restored = native_position(canonical, permutation, signs, offsets, [0, 0, 0, 1.7])
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(restored, native))
    assert restored[3] == 1.7
    action_examples = []
    for kp, effort, sign, default in [(100, 50, -1, .2), (200, 80, 1, -.1)]:
        scale = .25 * effort / kp
        target = default + sign * scale * .4
        delta_torque = kp * (target - default)
        assert math.isclose(delta_torque / effort, sign * .25 * .4, abs_tol=1e-12)
        action_examples.append(dict(kp=kp, effort_limit=effort, sign=sign,
            default_radians=default, action=.4, scale=scale,
            target_radians=round(target, 8), proportional_torque_change_nm=round(delta_torque, 8)))
    # One paired reference frame, three synthetic two-dimensional latents.
    latents = [[.2, .1], [.6, -.1], [.4, 0.]]
    center = [sum(z[j] for z in latents) / 3 for j in range(2)]
    losses = [sum((x - c) ** 2 for x, c in zip(z, center)) for z in latents]
    gradients = [[2 * (x - c) for x, c in zip(z, center)] for z in latents]
    eps = 1e-6
    max_error = 0.
    for z, gradient in zip(latents, gradients):
        for j in range(2):
            plus, minus = list(z), list(z)
            plus[j] += eps
            minus[j] -= eps
            # Keep the stop-gradient target fixed when taking finite differences.
            loss = lambda values: sum((x-c)**2 for x,c in zip(values, center))
            numeric = (loss(plus) - loss(minus)) / (2 * eps)
            max_error = max(max_error, abs(numeric - gradient[j]))
    assert max_error < 1e-9
    comparisons = [dict(robot=robot, interface=interface, specialist_return=s,
        unified_return=u, return_difference=round(u-s, 4),
        specialist_survival_percent=ss, unified_survival_percent=us,
        survival_difference_pp=round(us-ss, 3), within_tolerance=abs(u-s)<=.02)
        for robot, interface, s, u, ss, us in TABLE_I]
    assert all(row['within_tolerance'] for row in comparisons)
    assert max(abs(row['return_difference']) for row in comparisons) == .0181
    early_return = 20 * .09 / (.1 * (101-1))
    assert math.isclose(early_return, .18)
    result = dict(dimensions=dimensions,
        canonical_example=dict(native=native, canonical=canonical, restored=restored),
        action_examples=action_examples,
        alignment=dict(latents=latents, center=center, per_robot_losses=losses,
            local_gradients_fixed_center=gradients, finite_difference_max_error=max_error),
        time_seconds=dict(control_interval=.02, preview_first_to_last=9*.02,
            model_fitting=8*.02, imagination=32*.02),
        early_termination_example=dict(clip_frames=101, executed_transitions=20,
            reward_per_transition=.09, maximum_reward_per_transition=.1,
            full_clip_normalized_return=early_return, executed_step_average_fraction=.9),
        reported_comparisons=comparisons,
        gpu_hours=dict(unified=16.3, specialists_total=13.9,
            relative_increase_percent=100 * (16.3 / 13.9 - 1)))
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.check:
        stored = Path(__file__).resolve().parent / 'assets/contracts-results.json'
        try:
            compare_record(json.loads(stored.read_text(encoding='utf-8')), result)
        except ValueError as error:
            raise SystemExit('Contract results differ from the article record: ' + str(error)) from error
        print('MorphMimic dimensions, mappings, gradients and Table I differences passed')
        return
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
