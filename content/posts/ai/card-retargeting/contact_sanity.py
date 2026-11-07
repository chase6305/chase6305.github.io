"""Constructed contact checks; no CARD solver, simulator, or hardware required."""

import argparse
import math


def force_priorities(active_forces):
    """An empty contact gate and zero force on an active contact are distinct.

    The all-zero convention (unit priorities) is an explicit choice for this
    example, not a claim about CARD's released implementation.
    """
    values = tuple(float(value) for value in active_forces)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Active forces must be finite and nonnegative")
    largest = max(values, default=0.0)
    return tuple(1.0 if largest == 0 else 1.0 + 0.5 * value / largest for value in values)


def rigid_point(local, angle, translation):
    c, s = math.cos(angle), math.sin(angle)
    x, y = local
    return (c * x - s * y + translation[0], s * x + c * y + translation[1])


def distance_squared(left, right):
    return math.fsum((a - b) ** 2 for a, b in zip(left, right))


def check():
    assert force_priorities(iter(())) == ()
    assert force_priorities(iter((0, 0))) == (1.0, 1.0)
    assert force_priorities((0, 2, 4)) == (1.0, 1.25, 1.5)
    assert [v * v for v in force_priorities((0, 2, 4))] == [1.0, 1.5625, 2.25]
    for invalid in ((-1,), (float("nan"),), (float("inf"),)):
        try:
            force_priorities(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid force accepted")

    # A fixed pad material point at (10,0) mm rotates 90 degrees about the origin.
    previous = rigid_point((10, 0), 0.0, (0, 0))
    current = rigid_point((10, 0), math.pi / 2, (0, 0))
    transported = rigid_point(previous, math.pi / 2, (0, 0))
    assert distance_squared(current, transported) == 0.0
    assert math.isclose(distance_squared(current, previous), 200.0)
    migrated = rigid_point((11, 0), math.pi / 2, (0, 0))
    assert math.isclose(distance_squared(migrated, previous), 221.0)
    assert math.isclose(distance_squared(migrated, transported), 1.0)

    # Independently check rigid transport with arbitrary old/new poses. Recover
    # the material point in the old local frame before applying the new pose.
    for old_angle in (0.0, 0.7, -1.1):
        for new_angle in (-0.3, 1.2, 2.4):
            old_t, new_t = (2.0, -3.0), (-4.0, 5.0)
            old_world = rigid_point((10.0, -2.0), old_angle, old_t)
            old_local = rigid_point(
                (old_world[0] - old_t[0], old_world[1] - old_t[1]),
                -old_angle,
                (0, 0),
            )
            transported = rigid_point(old_local, new_angle, new_t)
            expected = rigid_point((10.0, -2.0), new_angle, new_t)
            assert distance_squared(transported, expected) < 1e-25

    # The half-angle identity provides a separate expression for the same scale.
    for degrees in (0, 15, 30, 45, 90, 180):
        angle = math.radians(degrees)
        loss_scale = 10.0 * math.sqrt(2.0 * (1.0 - math.cos(angle)))
        chord_scale = 20.0 * abs(math.sin(angle / 2))
        assert math.isclose(loss_scale, chord_scale, abs_tol=1e-12)
    scale_30 = 20.0 * math.sin(math.pi / 12)
    print("Active force priorities: (1, 1.25, 1.5); squared weights: (1, 1.5625, 2.25)")
    print("Same material point, 90-degree rotation: world difference^2=200 mm^2; transported residual=0")
    print(f"30-degree direction error equivalent length: {scale_30:.6f} mm")
    print("All-zero active forces retain unit priorities; empty gate has no contacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else parser.print_help()
