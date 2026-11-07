"""Constructed shared-grasp and evidence-context examples, not TAMPAgent's API.

No tools, robot, perception, geometric certification, or paper implementation is
executed. Revision numbers stand in for dependency tracking in a small model.
"""

import argparse
from dataclasses import dataclass, replace
from itertools import product


@dataclass(frozen=True)
class Context:
    scene: int
    base: int
    start: int


@dataclass(frozen=True)
class Evidence:
    factor: str
    grasp: int
    context: Context
    action: str = "pick:apple"


DEPENDENCIES = {
    "Grasp": ("scene",),
    "Kin": ("scene", "base"),
    "Motion": ("scene", "base", "start"),
}


def is_current(evidence, current):
    fields = DEPENDENCIES.get(evidence.factor)
    return fields is not None and all(
        getattr(evidence.context, field) == getattr(current, field) for field in fields
    )


def pick_ready(evidence, current, grasp):
    values = tuple(evidence)
    if len(values) != 3 or {v.factor for v in values} != set(DEPENDENCIES):
        return False
    return all(
        v.action == "pick:apple" and v.grasp == grasp and is_current(v, current)
        for v in values
    )


def holding_after_observation(previous, object_seen_in_hand):
    # In this constructed example, a command return alone is not confirmation.
    return "apple" if object_seen_in_hand else previous


def check():
    assert {1, 2} & {2, 3} == {2}
    old = Context(0, 0, 0)
    witnesses = tuple(Evidence(f, 2, old) for f in DEPENDENCIES)
    assert pick_ready(iter(witnesses), old, 2)
    for scene, base, start, grasp in product((0, 1), (0, 1), (0, 1), (2, 3)):
        expected = (scene, base, start, grasp) == (0, 0, 0, 2)
        assert pick_ready(witnesses, Context(scene, base, start), grasp) == expected

    moved = Context(0, 1, 1)
    assert [is_current(w, moved) for w in witnesses] == [True, False, False]
    refreshed = tuple(
        w if w.factor == "Grasp" else replace(w, context=moved) for w in witnesses
    )
    assert pick_ready(refreshed, moved, 2)
    moved_again = replace(moved, start=2)
    assert [is_current(w, moved_again) for w in refreshed] == [True, True, False]
    mixed_grasps = (witnesses[0], witnesses[1], replace(witnesses[2], grasp=3))
    assert not pick_ready(mixed_grasps, old, 2)
    assert not pick_ready(witnesses[:2], old, 2)
    assert not pick_ready(witnesses + witnesses[:1], old, 2)
    assert not pick_ready(tuple(replace(w, action="pick:cup") for w in witnesses), old, 2)
    assert holding_after_observation(None, False) is None
    assert holding_after_observation(None, True) == "apple"
    print("Shared pick/place grasps: {2}")
    print("After base move, reusable Grasp/Kin/Motion: True / False / False")
    print("After start-only change: True / True / False")
    print("Mixed grasp evidence rejected; dispatched command alone does not confirm holding.")
    print("Constructed dependency checks passed; no physical constraint was certified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else parser.print_help()
