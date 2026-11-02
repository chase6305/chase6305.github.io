"""A shared-grasp constraint example with an analytic planar 3R arm.

Python 3.10+, standard library only. Lengths in metres, angles in radians.
Three unrestricted revolute joints; link lengths (1, 1, 0.25). A pose is
(x, y, yaw) in SE(2). Grasp g maps object coordinates into the hand frame.
Only endpoint kinematics is checked. Collision, support/contact stability,
connecting motions, limits, dynamics and hardware execution are not checked.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

LINKS = (1.0, 1.0, 0.25)
POSITION_TOL = 1e-10
ANGLE_TOL = 1e-10


def numbers(values, count):
    values = tuple(values)
    if len(values) != count or any(type(x) not in (int, float) or not math.isfinite(x)
                                   for x in values):
        raise ValueError(f"Expected {count} finite real numbers")
    return values


def wrap(angle):
    return math.remainder(angle, 2 * math.pi)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        numbers((self.x, self.y, self.yaw), 3)


def compose(parent, child):
    """T_AB T_BC = T_AC: rotate the child translation before adding it."""
    c, s = math.cos(parent.yaw), math.sin(parent.yaw)
    return Pose(parent.x + c * child.x - s * child.y,
                parent.y + s * child.x + c * child.y,
                wrap(parent.yaw + child.yaw))


def inverse(pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose(-c * pose.x - s * pose.y, s * pose.x - c * pose.y, -pose.yaw)


def joint_positions(joints):
    joints = numbers(joints, 3)
    points, x, y, angle = [(0., 0.)], 0., 0., 0.
    for length, joint in zip(LINKS, joints):
        angle = wrap(angle + joint)
        x, y = x + length * math.cos(angle), y + length * math.sin(angle)
        points.append((x, y))
    return points


def forward(joints):
    joints = numbers(joints, 3)
    x, y = joint_positions(joints)[-1]
    return Pose(x, y, wrap(math.fsum(joints)))


def residual(actual, expected):
    return {"position_m": math.hypot(actual.x - expected.x, actual.y - expected.y),
            "angle_rad": abs(wrap(actual.yaw - expected.yaw))}


def hand_ik(target):
    """Return verified IK representatives for an unrestricted planar 3R arm.

    A zero-radius folded wrist has a continuous family of solutions: the two
    representatives below do not enumerate that family. The clear out-of-reach
    examples use the analytic wrist-radius bound, not failed numerical starts.
    """
    l1, l2, l3 = LINKS
    u = target.x - l3 * math.cos(target.yaw)
    v = target.y - l3 * math.sin(target.yaw)
    radius = math.hypot(u, v)
    if radius > l1 + l2 + POSITION_TOL or radius < abs(l1 - l2) - POSITION_TOL:
        return []
    cosine = (u * u + v * v - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    bend = math.acos(max(-1., min(1., cosine)))
    solutions = []
    for q2 in (bend, -bend):
        q1 = math.atan2(v, u) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q = tuple(wrap(x) for x in (q1, q2, target.yaw - q1 - q2))
        error = residual(forward(q), target)
        if error["position_m"] > POSITION_TOL or error["angle_rad"] > ANGLE_TOL:
            raise ArithmeticError("Analytic IK did not pass the independent FK residual check")
        if not any(all(abs(wrap(a - b)) <= ANGLE_TOL for a, b in zip(q, old))
                   for old in solutions):
            solutions.append(q)
    return solutions


def grasp_ik(placement, grasp):
    """Kin(q,p,g): solve FK(q)=p inverse(g), then verify FK(q) g=p."""
    target = compose(placement, inverse(grasp))
    solutions = hand_ik(target)
    errors = [residual(compose(forward(q), grasp), placement) for q in solutions]
    if any(e["position_m"] > POSITION_TOL or e["angle_rad"] > ANGLE_TOL for e in errors):
        raise ArithmeticError("IK witness violated the shared frame convention")
    l3 = LINKS[2]
    wrist_radius = math.hypot(target.x - l3 * math.cos(target.yaw),
                              target.y - l3 * math.sin(target.yaw))
    return {"hand_target": asdict(target), "wrist_radius_m": wrist_radius,
            "solutions_rad": solutions, "object_pose_residuals": errors}


GRASPS = {"front": Pose(0., 0., 0.), "back": Pose(0., 0., math.pi)}
PLACEMENTS = {"left": Pose(-2., 0., 0.), "middle": Pose(0., 1., 0.),
              "right": Pose(2., 0., 0.)}


def demo():
    witnesses = {name: {g: grasp_ik(p, transform) for g, transform in GRASPS.items()}
                 for name, p in PLACEMENTS.items()}

    def shared(source, target):
        return [g for g in GRASPS if witnesses[source][g]["solutions_rad"]
                and witnesses[target][g]["solutions_rad"]]

    stages = []
    for source, target, grasp in (("left", "middle", "back"), ("middle", "right", "front")):
        stages.append({"from": source, "to": target, "shared_grasp": grasp,
                       "pickup_q_rad": witnesses[source][grasp]["solutions_rad"][0],
                       "placement_q_rad": witnesses[target][grasp]["solutions_rad"][0]})
    return {"scope": "Constructed endpoint-kinematics constraints, not a verified robot plan.",
            "lengths_m": LINKS, "position_tolerance_m": POSITION_TOL, "angle_tolerance_rad": ANGLE_TOL,
            "joint_model": "Three unrestricted revolute joints; singular IK families are not enumerated.",
            "grasp_convention": "g = hand_T_object; world_T_object = FK(q) g",
            "grasps": {name: asdict(g) for name, g in GRASPS.items()},
            "placements": {name: asdict(p) for name, p in PLACEMENTS.items()},
            "endpoint_witnesses": witnesses, "direct_shared_grasps": shared("left", "right"),
            "via_middle_shared_grasps": {"left_to_middle": shared("left", "middle"),
                                          "middle_to_right": shared("middle", "right")},
            "regrasp_endpoint_stages": stages,
            "unverified": ["collision", "connecting paths", "support and grasp stability",
                           "approach and retreat", "joint limits", "dynamics", "hardware"]}


def self_check():
    result = demo()
    counts = [[len(result["endpoint_witnesses"][p][g]["solutions_rad"])
               for g in GRASPS] for p in PLACEMENTS]
    if counts != [[0, 2], [2, 2], [2, 0]] or result["direct_shared_grasps"]:
        raise AssertionError("The two endpoints no longer require incompatible grasps")
    if result["via_middle_shared_grasps"] != {"left_to_middle": ["back"], "middle_to_right": ["front"]}:
        raise AssertionError("Intermediate placement did not separate the two grasp constraints")
    rng = random.Random(20261029)
    for _ in range(500):
        q = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
        g = Pose(rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(-math.pi, math.pi))
        p = compose(forward(q), g)
        if not grasp_ik(p, g)["solutions_rad"]:
            raise AssertionError("An FK-constructed target lost all IK witnesses")
    return result


def summary(result):
    lines = ["Planar 3R endpoint constraints (not collision-checked)",
             "Placement  front IK  back IK"]
    for name, row in result["endpoint_witnesses"].items():
        lines.append(f"{name:<11}{len(row['front']['solutions_rad']):>8}{len(row['back']['solutions_rad']):>9}")
    lines += ["Direct shared grasps: none", "Via middle: back, then front",
              "Only endpoint IK is verified; connecting motions and support are unverified."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = self_check() if args.check else demo()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(summary(result) if args.summary else text, end="")
