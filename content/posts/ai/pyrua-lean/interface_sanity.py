#!/usr/bin/env python3
"""Exact bookkeeping and constructed geometry for the PyRUA-Lean article.

Standard library only. No model, simulator, network, tokenizer or robot.
Run --check for independent arithmetic checks; otherwise print the reference.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import json
import math


MAIN = (
    ("LIBERO-PRO", 200, 166, 188, "83.0", "94.0"),
    ("RoboTwin 2.0", 250, 150, 173, "60.0", "69.2"),
    ("RC365 atomic", 90, 71, 78, "78.9", "86.7"),
    ("RC365 composite", 160, 55, 63, "34.4", "39.4"),
)
DOJO = (("Generalization", 600, 275), ("Precision", 400, 149),
        ("Long-Horizon", 400, 192), ("Memory", 300, 189), ("Open", 400, 256))
# Table 17, 50 episodes per original task; the two Generalization halves
# together still contribute 50. These independently recover each group total.
DOJO_TASK_SUCCESSES = (
    (39,38,16,35,1,9,4,3,43,37,0,50),
    (33,6,22,38,32,17,0,1),
    (38,7,49,47,1,17,0,33),
    (50,41,0,49,0,49),
    (39,49,50,45,46,17,0,10),
)
DEFAULT = dict(operations=12, group=3, reply=1200, retained=20)


def integer(value, name, low=0, high=100_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: integer in [{low}, {high}] required")
    return value


def cumulative_input(prefix, appended):
    """K calls: prefix in each; feedback j enters only calls AFTER call j.

    The final appended message is deliberately absent from this cost. Cache
    discounts, context compaction and provider image tokenization are omitted.
    """
    integer(prefix, "prefix", high=10**9)
    values = tuple(appended)
    for value in values:
        integer(value, "appended", high=10**9)
    k = len(values)
    return k*prefix + sum((k-1-j)*value for j, value in enumerate(values))


def accountant(operations=12, group=3, reply=1200, retained=20):
    """Same fixed primitive events, different feedback boundaries.

    Prefixes are fixed illustrative values: tool 2000, code 2400 tokens.
    Every group's retained feedback is rounded UP to a whole token.
    """
    integer(operations, "operations", 1, 100)
    integer(group, "group", 1, 100)
    integer(reply, "reply", 0, 100_000)
    integer(retained, "retained", 0, 100)
    sizes = [min(group, operations-i) for i in range(0, operations, group)]
    raw = [size*reply for size in sizes]
    selected = [(value*retained+99)//100 for value in raw]
    return {
        "primitive_events": operations,
        "tool_calls": operations,
        "code_calls": len(sizes),
        "tool_input": cumulative_input(2000, [reply]*operations),
        "grouped_input": cumulative_input(2400, raw),
        "selected_input": cumulative_input(2400, selected),
    }


def placement_target(eef, bowl, plate, bottom, clearance=.007):
    """Translation only, world metres, rigid grasp and synchronized readings.

    Does not validate the scene, rotation, collision, contact or reachability.
    """
    points = []
    for name, point in (("eef", eef), ("bowl", bowl), ("plate", plate)):
        try:
            values = tuple(float(v) for v in point)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}: three finite coordinates required") from exc
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name}: three finite coordinates required")
        points.append(values)
    bottom, clearance = float(bottom), float(clearance)
    if not math.isfinite(bottom) or not math.isfinite(clearance) or clearance < 0:
        raise ValueError("finite bottom and nonnegative finite clearance required")
    eef, bowl, plate = points
    return (eef[0]+plate[0]-bowl[0], eef[1]+plate[1]-bowl[1],
            plate[2]+eef[2]-bottom+clearance)


def binary_advantages(rewards):
    """Illustration: population std, zero advantage for a constant group.

    The paper says group-normalized reward; this explicit convention is not a
    claim about miles' epsilon, variance correction or token-loss reduction.
    """
    values = tuple(rewards)
    if not values or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError("nonempty binary integer rewards required")
    mean = sum(values)/len(values)
    std = math.sqrt(mean*(1-mean))
    return tuple((v-mean)/std if std else 0. for v in values)


def clipped_gain(ratio, advantage):
    """One-token surrogate gain; minimizing loss uses its NEGATIVE.

    Fixed asymmetric clipping from the paper. This does not specify training
    token reduction, optimizer state, KL or an entire GRPO implementation.
    """
    ratio, advantage = float(ratio), float(advantage)
    if not math.isfinite(ratio) or ratio < 0 or not math.isfinite(advantage):
        raise ValueError('nonnegative finite ratio and finite advantage required')
    return min(ratio*advantage, min(1.28,max(.8,ratio))*advantage)


def reference():
    n = sum(row[1] for row in MAIN)
    tool = sum(row[2] for row in MAIN)
    code = sum(row[3] for row in MAIN)
    macro = sum((Fraction(s, n) for _, n, s in DOJO), Fraction())/len(DOJO)
    pooled = Fraction(sum(s for _, _, s in DOJO), sum(n for _, n, _ in DOJO))
    return {
        "scope": "paper bookkeeping and constructed arithmetic; no model or robot",
        "main": {"instances": n, "tool_successes": tool, "code_successes": code,
                 "tool_percent": 100*tool/n, "code_percent": 100*code/n,
                 "percentage_points": 100*(code-tool)/n,
                 "relative_increase_percent": 100*(code-tool)/tool,
                 "both_solved": 406, "tool_only": 36, "code_only": 96, "both_failed": 162},
        "robodojo": {"macro_percent": float(100*macro), "pooled_percent": float(100*pooled),
                      "successes": 1061, "episodes": 2100},
        "accounting": {"input": DEFAULT, "result": accountant(**DEFAULT)},
        "placement_metres": placement_target((.4, .2, 1.12), (.42, .21, 1.09),
                                             (.45, .25, 1.00), 1.08),
        "group_advantages": binary_advantages([1, 1, 1, 0, 0, 0, 0, 0]),
        "clipped_gain_examples": {str(r): {"positive":clipped_gain(r,1),
                                           "negative":clipped_gain(r,-1)}
                                  for r in (.6,.8,1.,1.28,1.4)},
    }


def check():
    # Recover integer success counts independently from the printed, rounded
    # one-decimal percentages. Every row has a unique compatible count.
    for _, n, tool, code, tool_rate, code_rate in MAIN:
        for expected, printed in ((tool, tool_rate), (code, code_rate)):
            matches = [s for s in range(n+1)
                       if (Decimal(100)*s/n).quantize(Decimal('.1')) == Decimal(printed)]
            assert matches == [expected]
    result = reference()
    assert result['main']['tool_successes'] == 442
    assert result['main']['code_successes'] == 502
    assert 406+36+96+162 == 700
    assert 406+36 == 442 and 406+96 == 502
    assert Fraction(1061, 2100) != sum((Fraction(s, n) for _, n, s in DOJO), Fraction())/5
    for (_, count, successes), tasks in zip(DOJO,DOJO_TASK_SUCCESSES):
        assert len(tasks)*50 == count and sum(tasks) == successes
        assert all(0 <= n <= 50 for n in tasks)
    # A sign-based piecewise oracle, independent of the min/clip expression.
    for ratio in (0., .1, .6, .799, .8, 1., 1.28, 1.281, 2., 10.):
        for advantage in (-7., -1., 0., 1., 3.):
            expected = ratio*advantage
            if advantage > 0 and ratio > 1.28: expected = 1.28*advantage
            if advantage < 0 and ratio < .8: expected = .8*advantage
            assert abs(clipped_gain(ratio,advantage)-expected)<1e-13
    # Independently replay the running prompt, rather than use the weighted sum.
    for k in range(1, 14):
        for prefix in (0, 37, 2000):
            appended = [((j*71+13) % 103) for j in range(k)]
            prompt, expected = prefix, 0
            for value in appended:
                expected += prompt
                prompt += value
            assert expected == cumulative_input(prefix, appended)
            changed = appended[:-1]+[10**8]
            assert cumulative_input(prefix, changed) == expected
    assert accountant() == dict(primitive_events=12, tool_calls=12, code_calls=4,
                               tool_input=103200, grouped_input=31200, selected_input=13920)
    # Uneven groups and rounding are checked against direct rational arithmetic.
    for n in range(1, 25):
        for group in range(1, 9):
            for retained in (0, 1, 20, 99, 100):
                got = accountant(n, group, 137, retained)
                prompt, expected, left = 2400, 0, n
                while left:
                    count = min(group, left)
                    expected += prompt
                    prompt += math.ceil(Fraction(count*137*retained, 100))
                    left -= count
                assert got['selected_input'] == expected
    eef, bowl, plate = (.4, .2, 1.12), (.42, .21, 1.09), (.45, .25, 1.)
    target = placement_target(eef, bowl, plate, 1.08)
    assert all(abs(a-b)<1e-14 for a,b in zip(target, (.43, .24, 1.047)))
    # Independent rigid-translation witness: object centre reaches plate xy,
    # and its bottom reaches plate height + clearance.
    moved_bowl = [bowl[j]+target[j]-eef[j] for j in range(3)]
    assert all(abs(moved_bowl[j]-plate[j])<1e-14 for j in (0,1))
    assert abs(1.08+target[2]-eef[2]-1.007)<1e-14
    shift = (2., -3., 5.)
    shifted = placement_target(*(tuple(p[j]+shift[j] for j in range(3)) for p in (eef,bowl,plate)), 6.08)
    assert all(abs(shifted[j]-target[j]-shift[j])<1e-14 for j in range(3))
    # Exhaust all 256 groups; compare with a high-precision Decimal definition.
    with localcontext() as ctx:
        ctx.prec = 50
        for rewards in product((0,1), repeat=8):
            advantages = binary_advantages(rewards)
            mean = sum(map(Decimal, rewards))/8
            variance = sum((Decimal(r)-mean)**2 for r in rewards)/8
            std = variance.sqrt()
            oracle = [(Decimal(r)-mean)/std if std else Decimal(0) for r in rewards]
            assert all(abs(a-float(b))<2e-14 for a,b in zip(advantages,oracle))
            assert abs(sum(advantages))<2e-14
    for kwargs in ({'operations':0}, {'operations':True}, {'group':0}, {'group':1.5},
                   {'reply':-1}, {'retained':101}, {'reply':float('nan')}):
        try:
            accountant(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(('accepted invalid accounting input', kwargs))
    for bad in ((1,2), (1,2,float('nan')), (1,2,float('inf'))):
        try:
            placement_target(bad,bowl,plate,1.08)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted malformed geometry')
    print('PASS: rounded paper counts, 42-task aggregation, prompt replay, rigid placement, 256 reward groups, asymmetric clipping')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        print(json.dumps(reference(), ensure_ascii=False, indent=2))
