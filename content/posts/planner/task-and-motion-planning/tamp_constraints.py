"""A constructed TAMP refinement subproblem, not a robot motion planner.

Standard library, metres. A carried disk must cross a wall opening after a
blocker has been parked. The two parking choices and two grasp envelopes are
finite; the crossing height is continuous. Blocker relocation and arm IK are
assumed by the skeleton and are NOT verified here. Contact counts as collision.
"""
import argparse
import json
import math
from pathlib import Path
import random

HALF_OPENING = .5
BLOCKER_RADIUS = .4
PARKING = {"near": (0., .05), "far": (1., 1.)}
GRASPS = {"wide": .6, "narrow": .25}
EPS = 1e-9


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist(point, start)
    t = max(0., min(1., ((point[0]-start[0])*dx + (point[1]-start[1])*dy) / length_squared))
    return math.dist(point, (start[0]+t*dx, start[1]+t*dy))


def segment_wall_ray_distance(start, end, sign):
    """Exact segment distance to x=0, sign*y >= HALF_OPENING (two wall rays)."""
    if start[0] != end[0]:
        t = -start[0] / (end[0]-start[0])
        if 0 <= t <= 1 and sign*(start[1]+t*(end[1]-start[1])) >= HALF_OPENING:
            return 0.
    elif start[0] == 0 and max(sign*start[1], sign*end[1]) >= HALF_OPENING:
        return 0.
    def endpoint_distance(point):
        return math.hypot(point[0], max(0., HALF_OPENING-sign*point[1]))
    return min(endpoint_distance(start), endpoint_distance(end),
               point_segment_distance((0., sign*HALF_OPENING), start, end))


def verify_path(path, radius, parked):
    """Check whole straight segments, rather than just their endpoints."""
    if len(path) < 2:
        raise ValueError("A path requires at least two points")
    segments = list(zip(path, path[1:]))
    wall_clearance = min(segment_wall_ray_distance(a, b, sign)
                         for a, b in segments for sign in (-1, 1))
    blocker_clearance = min(point_segment_distance(parked, a, b) for a, b in segments)
    return {"valid": wall_clearance > radius + EPS
                     and blocker_clearance > radius + BLOCKER_RADIUS + EPS,
            "wall_centerline_clearance_m": wall_clearance,
            "blocker_centerline_clearance_m": blocker_clearance}


def crossing_intervals(radius, parked):
    """Necessary crossing heights; empty intervals give a geometric certificate."""
    margin = HALF_OPENING-radius-EPS
    if margin <= 0:
        return [], "door_width_certificate"
    low, high = -margin, margin
    total_radius = radius+BLOCKER_RADIUS+EPS
    if abs(parked[0]) > total_radius:
        return [(low, high)], None
    dy = math.sqrt(total_radius**2-parked[0]**2)
    forbidden_low, forbidden_high = parked[1]-dy, parked[1]+dy
    pieces = []
    # Open interval endpoints already exclude contact. Shifting them again
    # could erase a narrow but nonempty feasible interval and fake a proof.
    for left, right in [(low, min(high, forbidden_low)),
                        (max(low, forbidden_high), high)]:
        if right > left:
            pieces.append((left, right))
    return pieces, None if pieces else "crossing_obstruction_certificate"


def refine(parked, radius):
    """A failed candidate path is unknown; only necessary-condition proofs prune."""
    intervals, certificate = crossing_intervals(radius, parked)
    if certificate:
        return {"status": "infeasible", "reason": certificate, "crossing_intervals_m": []}
    low, high = intervals[0]
    height = (low+high)/2
    path = [(-2., 0.), (-1., height), (1., height), (2., 0.)]
    checks = verify_path(path, radius, parked)
    return {"status": "feasible" if checks["valid"] else "unknown",
            "reason": "verified_segments" if checks["valid"] else "candidate_path_collision",
            "crossing_intervals_m": intervals, "crossing_height_m": height,
            "path_m": path, **checks}


def enumerate_choices(reuse_certificate):
    rows, rejected_grasps = [], set()
    for parking, parked in PARKING.items():
        for grasp, radius in GRASPS.items():
            if reuse_certificate and grasp in rejected_grasps:
                rows.append({"parking": parking, "grasp": grasp, "status": "pruned",
                             "reason": "reused_door_width_certificate"})
                continue
            result = refine(parked, radius)
            rows.append({"parking": parking, "grasp": grasp, **result})
            # This necessary condition depends on the grasp envelope and the
            # unchanged wall, not on parking. Never learn from mere path failure.
            if result["reason"] == "door_width_certificate":
                rejected_grasps.add(grasp)
    return {"choices": rows, "refinement_calls": sum(r["status"] != "pruned" for r in rows),
            "verified_choices": [[r["parking"], r["grasp"]] for r in rows if r["status"] == "feasible"]}


def demo():
    after_door = (1., 0.)
    candidate = refine(after_door, GRASPS["narrow"])
    detour = [(-2., 0.), (-1., 0.), (.3, 0.), (.3, 1.), (2., 1.), (2., 0.)]
    return {"constructed_example_only": True,
            "scope": "Refine a given carrying stage; no blocker relocation, arm IK, dynamics or hardware.",
            "units": "m", "opening_width_m": 2*HALF_OPENING,
            "blocker_radius_m": BLOCKER_RADIUS,
            "parking_m": PARKING, "grasp_envelope_radius_m": GRASPS,
            "without_reuse": enumerate_choices(False), "with_reuse": enumerate_choices(True),
            "failed_candidate_is_not_infeasibility": {
                "parking_m": after_door, "candidate": candidate,
                "alternative_path_m": detour, "alternative_checks": verify_path(detour, .25, after_door)},
            "factorized_state_count": {"domain_sizes": [4, 5, 5, 3, 2],
                                       "cartesian_product": 4*5*5*3*2}}


def self_check():
    result = demo()
    if result["without_reuse"]["refinement_calls"] != 4 or result["with_reuse"]["refinement_calls"] != 3:
        raise AssertionError("Unexpected finite enumeration counts")
    for name in ("without_reuse", "with_reuse"):
        if result[name]["verified_choices"] != [["far", "narrow"]]:
            raise AssertionError("Unexpected feasible choice")
    extra = result["failed_candidate_is_not_infeasibility"]
    if extra["candidate"]["status"] != "unknown" or not extra["alternative_checks"]["valid"]:
        raise AssertionError("A failed straight path was confused with infeasibility")
    if not math.isclose(extra["alternative_checks"]["wall_centerline_clearance_m"], .3):
        raise AssertionError("Wrong whole-segment wall distance")
    # All path endpoints are clear here, yet the segment intersects the blocker.
    if verify_path([(-2., 0.), (2., 0.)], .25, (0., 0.))["valid"]:
        raise AssertionError("Endpoint-only collision check")
    if segment_wall_ray_distance((-1., 1.), (1., 1.), 1) != 0:
        raise AssertionError("A wall intersection was missed")
    if point_segment_distance((1., 0.), (0., 0.), (0., 0.)) != 1:
        raise AssertionError("Degenerate segment failed")
    tiny_gap = (HALF_OPENING-.25-EPS)-(.25+BLOCKER_RADIUS+EPS)-5e-10
    if not crossing_intervals(.25, (0., tiny_gap))[0]:
        raise AssertionError("A nonempty crossing interval was mistaken for infeasibility")
    rng = random.Random(20261029)
    for _ in range(500):
        point, start, end = [tuple(rng.uniform(-2, 2) for _ in range(2)) for _ in range(3)]
        exact = point_segment_distance(point, start, end)
        sampled = min(math.dist(point, (start[0]+i/1000*(end[0]-start[0]),
                                       start[1]+i/1000*(end[1]-start[1]))) for i in range(1001))
        if not (-1e-12 <= sampled-exact <= math.dist(start, end)/1000+1e-12):
            raise AssertionError("Independent dense-sampling distance bound failed")
        for sign in (-1, 1):
            exact_wall = segment_wall_ray_distance(start, end, sign)
            sampled_wall = min(math.hypot(start[0]+i/1000*(end[0]-start[0]),
                max(0., HALF_OPENING-sign*(start[1]+i/1000*(end[1]-start[1]))))
                for i in range(1001))
            if not (-1e-12 <= sampled_wall-exact_wall <= math.dist(start, end)/1000+1e-12):
                raise AssertionError("Independent wall-distance bound failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = self_check() if args.check else demo()
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
