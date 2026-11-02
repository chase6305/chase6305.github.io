"""Finite TAMP-style outer search around a verified 2D carrying-stage oracle.

Standard library only. Relocating the blocker and picking/placing the target
are abstract primitives: their trajectories, IK and dynamics are NOT verified.
This script illustrates scheduling and scoped proofs, not a full robot planner.
"""
import argparse
from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path

import tamp_constraints as geometry

GRID_STEP = .2
GRID_BOUNDS = (-10, 10, -6, 6)  # Integer indices: x=-2..2 m, y=-1.2..1.2 m.
GRID_WORLD_BOUNDS = (-2., 2., -1.2, 1.2)


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


def grid_layout(step):
    """Represent a fixed physical domain; the task endpoints must lie on the grid."""
    step = geometry.finite_scalar(step, "step")
    if step <= 0:
        raise ValueError("step must be positive")
    quotients = [bound / step for bound in GRID_WORLD_BOUNDS]
    if any(not math.isfinite(value) or abs(value) > 2**52 for value in quotients):
        raise ValueError("step is too small to represent grid indices reliably")
    # Include exact physical boundaries despite division rounding (1.2 / .2).
    bounds = tuple((math.ceil(math.nextafter(q, -math.inf)) if i % 2 == 0
                    else math.floor(math.nextafter(q, math.inf)))
                   for i, q in enumerate(quotients))
    start, goal = (round(-2. / step), 0), (round(2. / step), 0)
    for node, target in ((start, (-2., 0.)), (goal, (2., 0.))):
        if (not (bounds[0] <= node[0] <= bounds[1] and bounds[2] <= node[1] <= bounds[3])
                or math.dist((node[0] * step, node[1] * step), target) > 8 * math.ulp(2.)):
            raise ValueError("step must place the fixed start and goal on grid nodes")
    return step, bounds, start, goal


def grid_motion(parked, radius, max_expansions=1000, step=None):
    """A* on a finite 8-neighbor grid; grid failure never proves continuous failure."""
    parked, radius = geometry.validate_scene(parked, radius)
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError("max_expansions must be a nonnegative integer")
    step, bounds, start, goal = grid_layout(GRID_STEP if step is None else step)
    metadata = {"grid_step_m": step, "grid_bounds_m": GRID_WORLD_BOUNDS,
                "grid_node_count": (bounds[1] - bounds[0] + 1) * (bounds[3] - bounds[2] + 1)}
    point = lambda node: (node[0] * step, node[1] * step)
    if not all(geometry.verify_path([point(n), point(n)], radius, parked)["valid"]
               for n in (start, goal)):
        return {"status": "unknown", "reason": "grid_endpoint_collision", "expanded_nodes": 0, **metadata}
    frontier = [(math.dist(point(start), point(goal)), 0., start)]
    costs, parents, expanded = {start: 0.}, {}, 0
    while frontier:
        _, cost, node = heapq.heappop(frontier)
        if cost != costs[node]:
            continue  # An older, more expensive entry for this node.
        if node == goal:
            nodes = [goal]
            while nodes[-1] != start:
                nodes.append(parents[nodes[-1]])
            path = [point(n) for n in reversed(nodes)]
            checks = geometry.verify_path(path, radius, parked)
            if not checks["valid"]:
                raise RuntimeError("Grid search produced an invalid path")
            return {"status": "feasible", "reason": "verified_grid_path", "path_m": path,
                    "path_length_m": cost, "expanded_nodes": expanded, **checks, **metadata}
        if expanded >= max_expansions:
            return {"status": "unknown", "reason": "grid_budget_exhausted", "expanded_nodes": expanded, **metadata}
        expanded += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nxt = (node[0] + dx, node[1] + dy)
                if not (bounds[0] <= nxt[0] <= bounds[1]
                        and bounds[2] <= nxt[1] <= bounds[3]):
                    continue
                edge = [point(node), point(nxt)]
                if not geometry.verify_path(edge, radius, parked)["valid"]:
                    continue
                new_cost = cost + math.dist(*edge)
                if new_cost < costs.get(nxt, math.inf):
                    costs[nxt], parents[nxt] = new_cost, node
                    estimate = new_cost + math.dist(point(nxt), point(goal))
                    heapq.heappush(frontier, (estimate, new_cost, nxt))
    return {"status": "unknown", "reason": "finite_grid_exhausted", "expanded_nodes": expanded, **metadata}


def grid_resolution_demo():
    """Same geometry and continuous edge checks, five different finite graphs."""
    return {"scope": "Fixed carrying scene; finite-graph paths, not continuous optima or timing benchmarks.",
            "parking_m": (1., 0.), "radius_m": .25, "max_expansions": 1000,
            "results": [grid_motion((1., 0.), .25, step=step)
                        for step in (2., 1., .4, .2, .1)]}


def resolution_summary(result):
    lines = ["Same carrying scene; changing only grid spacing",
             "Step (m)  Nodes  Expanded  Status    Length (m)"]
    for row in result["results"]:
        length = f"{row['path_length_m']:.6f}" if "path_length_m" in row else "--"
        lines.append(f"{row['grid_step_m']:>8g}  {row['grid_node_count']:>5}"
                     f"  {row['expanded_nodes']:>8}  {row['status']:<8}  {length}")
    lines.append("Grid exhaustion remains unknown for continuous motion.")
    return "\n".join(lines) + "\n"


def refine_one(candidate):
    if candidate.attempt == 0:
        return geometry.refine(candidate.parking,
                               geometry.GRASPS[candidate.grasp])
    return grid_motion(candidate.parking, geometry.GRASPS[candidate.grasp])


def search(initial_parking=(0., .05), allow_relocation=True, reuse=True,
           max_attempts=2, max_calls=20):
    initial_parking, _ = geometry.validate_scene(initial_parking, 0.)
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
        if not isinstance(result, dict) or result.get("status") not in {"feasible", "infeasible", "unknown"}:
            raise ValueError("Refinement must return feasible, infeasible or unknown")
        if result["status"] == "infeasible":
            _, certificate = geometry.crossing_intervals(
                geometry.GRASPS[candidate.grasp], candidate.parking)
            if certificate is None or result.get("reason") != certificate:
                raise ValueError("Infeasibility was not supported by this example's geometric certificate")
        trace.append({**row, "status": result["status"], "reason": result["reason"]})
        if result["status"] == "feasible":
            path = result.get("path_m")
            if not isinstance(path, (list, tuple)) or len(path) < 2:
                raise ValueError("A carry witness must be a finite 2D polyline")
            try:
                path = [geometry.finite_point(point) for point in path]
            except ValueError as error:
                raise ValueError("A carry witness must be a finite 2D polyline") from error
            if (math.dist(path[0], (-2., 0.)) > geometry.EPS
                    or math.dist(path[-1], (2., 0.)) > geometry.EPS):
                raise ValueError("A carry witness must connect the specified start and goal")
            checks = geometry.verify_path(path,
                geometry.GRASPS[candidate.grasp], candidate.parking)
            if not checks["valid"]:
                raise RuntimeError("Refinement returned an invalid witness")
            actions = (["relocate blocker"] if candidate.skeleton != "direct" else [])
            actions += ["pick target", "carry target", "place target"]
            return {"status": "carry_witness_found", "refinement_calls": calls,
                    "trace": trace, "skeleton": candidate.skeleton,
                    "grasp": candidate.grasp, "abstract_actions": actions,
                    "carry_path_m": path, "carry_checks": checks,
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


def summary(result):
    """Short console view; the full JSON keeps the complete geometry and trace."""
    lines = ["Constructed carrying-stage example (not a complete robot plan)",
             "Case                         Calls  Result"]
    for name, case in result.items():
        if name != "scope":
            lines.append(f"{name:<29}{case['refinement_calls']:>5}  {case['status']}")
    lines += ["", "Detour revisit trace:"]
    for row in result["detour_with_revisit"]["trace"]:
        lines.append(f"  {row['grasp']:<6} attempt {row['attempt']}: "
                     f"{row['status']:<10} {row['reason']}")
    lines += ["", "Only the carrying path is verified; see JSON for unverified components."]
    return "\n".join(lines) + "\n"


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--resolutions", action="store_true", help="Compare five grid spacings in one scene")
    parser.add_argument("--summary", action="store_true", help="Print a short trace instead of full JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = grid_resolution_demo() if args.resolutions else self_check() if args.check else demo()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    short = resolution_summary if args.resolutions else summary
    print(short(result) if args.summary else text, end="")
