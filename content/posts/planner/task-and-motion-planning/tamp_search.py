"""Finite TAMP-style outer search around a verified 2D carrying-stage oracle.

Standard library only. Relocating the blocker and picking/placing the target
are abstract primitives: their trajectories, IK and dynamics are NOT verified.
This script illustrates scheduling and scoped proofs, not a full robot planner.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import tamp_constraints as geometry


@dataclass(frozen=True)
class Candidate:
    skeleton: str
    parking: tuple
    grasp: str
    attempt: int = 0


def context_key(candidate):
    """Every modeled geometric dependency; no cache survives a world change."""
    return (geometry.HALF_OPENING, geometry.BLOCKER_RADIUS, geometry.EPS,
            (-2., 0.), (2., 0.), candidate.parking,
            geometry.GRASPS[candidate.grasp])


def width_key(candidate):
    # The necessary door-width condition does not depend on blocker placement.
    return (geometry.HALF_OPENING, geometry.EPS,
            geometry.GRASPS[candidate.grasp])


def candidates(initial_parking, allow_relocation):
    modes = [("direct", tuple(initial_parking))]
    if allow_relocation:
        modes += [("relocate-near", geometry.PARKING["near"]),
                  ("relocate-far", geometry.PARKING["far"])]
    return deque(Candidate(name, parked, grasp)
                 for name, parked in modes for grasp in geometry.GRASPS)


def refine_one(candidate):
    if candidate.attempt == 0:
        return geometry.refine(candidate.parking,
                               geometry.GRASPS[candidate.grasp])
    # A second, explicitly constructed candidate, not a general path planner.
    path = [(-2., 0.), (-1., 0.), (.3, 0.), (.3, 1.), (2., 1.), (2., 0.)]
    checks = geometry.verify_path(path, geometry.GRASPS[candidate.grasp],
                                  candidate.parking)
    return {"status": "feasible" if checks["valid"] else "unknown",
            "reason": "verified_detour" if checks["valid"] else "detour_collision",
            "path_m": path, **checks}


def search(initial_parking=(0., .05), allow_relocation=True, reuse=True,
           max_attempts=2, max_calls=20):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("max_calls must be a nonnegative integer")
    queue = candidates(initial_parking, allow_relocation)
    exact_proofs, width_proofs = set(), set()
    trace, unresolved, calls = [], [], 0
    while queue:
        candidate = queue.popleft()
        row = {"skeleton": candidate.skeleton, "parking_m": candidate.parking,
               "grasp": candidate.grasp, "attempt": candidate.attempt + 1}
        exact, width = context_key(candidate), width_key(candidate)
        if reuse and (width in width_proofs or exact in exact_proofs):
            trace.append({**row, "status": "pruned", "reason": "cached_proof"})
            continue
        if calls >= max_calls:
            unresolved.extend([candidate, *queue])
            break
        result = refine_one(candidate)
        calls += 1
        trace.append({**row, "status": result["status"], "reason": result["reason"]})
        if result["status"] == "feasible":
            checks = geometry.verify_path(result["path_m"],
                geometry.GRASPS[candidate.grasp], candidate.parking)
            if not checks["valid"]:
                raise RuntimeError("Refinement returned an invalid witness")
            actions = (["relocate blocker"] if candidate.skeleton != "direct" else [])
            actions += ["pick target", "carry target", "place target"]
            return {"status": "carry_witness_found", "refinement_calls": calls,
                    "trace": trace, "skeleton": candidate.skeleton,
                    "grasp": candidate.grasp, "abstract_actions": actions,
                    "carry_path_m": result["path_m"], "carry_checks": checks,
                    "unverified": ["blocker relocation", "arm IK", "pick/place",
                                   "dynamics", "hardware execution"]}
        if result["status"] == "infeasible":
            exact_proofs.add(exact)
            if result["reason"] == "door_width_certificate":
                width_proofs.add(width)
        elif candidate.attempt + 1 < max_attempts:
            # Round-robin: let other skeletons run before another path attempt.
            queue.append(Candidate(candidate.skeleton, candidate.parking,
                                   candidate.grasp, candidate.attempt + 1))
        else:
            unresolved.append(candidate)
    return {"status": "unknown" if unresolved else "finite_candidates_infeasible",
            "refinement_calls": calls, "trace": trace,
            "unresolved_candidates": len(unresolved),
            "scope": "Only these finite skeleton/parking/grasp candidates; no global robot infeasibility claim."}


def demo():
    return {"scope": "Constructed carrying-stage search, not full robot planning.",
            "blocked_without_cache": search(reuse=False),
            "blocked_with_cache": search(),
            "detour_with_one_attempt": search((1., 0.), False, max_attempts=1),
            "detour_with_revisit": search((1., 0.), False)}


def self_check():
    result = demo()
    uncached, cached = result["blocked_without_cache"], result["blocked_with_cache"]
    if (uncached["refinement_calls"], cached["refinement_calls"]) != (6, 3):
        raise AssertionError("Scoped proof reuse changed unexpectedly")
    for key in ("skeleton", "grasp", "abstract_actions", "carry_path_m"):
        if uncached[key] != cached[key]:
            raise AssertionError("Proof reuse changed the verified witness")
    if cached["skeleton"] != "relocate-far" or cached["grasp"] != "narrow":
        raise AssertionError("Wrong abstract carrying choice")
    if result["detour_with_one_attempt"]["status"] != "unknown":
        raise AssertionError("Candidate exhaustion became an infeasibility proof")
    revisit = result["detour_with_revisit"]
    if revisit["status"] != "carry_witness_found" or revisit["refinement_calls"] != 3:
        raise AssertionError("Unknown was permanently pruned instead of revisited")
    if [row["status"] for row in revisit["trace"]] != ["infeasible", "unknown", "feasible"]:
        raise AssertionError("Unexpected tri-state trace")
    for budget in range(6):
        partial = search(max_calls=budget)
        if partial["refinement_calls"] > budget:
            raise AssertionError("Call budget exceeded")
        if budget < 3 and partial["status"] != "unknown":
            raise AssertionError("Budget exhaustion became a proof")
    c = Candidate("direct", (0., .05), "wide")
    before = width_key(c)
    old = geometry.HALF_OPENING
    try:
        geometry.HALF_OPENING = 1.2
        if width_key(c) == before or search(allow_relocation=False)["status"] != "carry_witness_found":
            raise AssertionError("Door change retained an obsolete proof")
    finally:
        geometry.HALF_OPENING = old
    if context_key(c) == context_key(Candidate("direct", (1., 1.), "wide")):
        raise AssertionError("Collision context omitted parking")
    if search(max_calls=0)["unresolved_candidates"] != 6:
        raise AssertionError("Unprocessed candidates were lost")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = self_check() if args.check else demo()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
