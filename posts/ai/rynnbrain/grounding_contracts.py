"""CPU-only teaching checks, not a RynnBrain evaluator or robot controller."""

import math
import re
from itertools import product


def finite(values):
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) for v in values):
        raise ValueError("Expected nonempty, finite numerical values")
    return values


def normalized_to_pixels(x, y, width, height):
    """Use pixel-center endpoints, matching the contact-point cookbook."""
    x, y = finite((x, y))
    if any(type(v) is not int or v < 1 for v in (width, height)):
        raise ValueError("Image dimensions must be positive integers")
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Normalized point is outside [0, 1000]")
    return x * (width - 1) / 1000, y * (height - 1) / 1000


def parse_contact(response):
    """Strict final-answer contract: reject extra text and multiple outputs.

    This is intentionally stricter than an illustration notebook's search.
    The returned angle is an undirected image-plane axis, not a robot pose.
    """
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = (rf"\s*<grasp pose>\s*\(\s*({number})\s*,\s*({number})\s*\)"
               rf"\s*,\s*({number})\s*</grasp pose>\s*")
    match = re.fullmatch(pattern, response)
    if not match:
        raise ValueError("Expected exactly one contact-point answer")
    x, y, angle = finite(map(float, match.groups()))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Contact point is outside [0, 1000]")
    return x, y, angle % 180


def backproject(u, v, depth, fx, fy, cx, cy):
    """Pinhole camera, rectified pixels and independently supplied metric depth."""
    u, v, depth, fx, fy, cx, cy = finite((u, v, depth, fx, fy, cx, cy))
    if depth <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("Depth and focal lengths must be positive")
    return finite(((u - cx) * depth / fx, (v - cy) * depth / fy, depth))


def decode_angles(values, *, encoding):
    """Require an explicit codec because the source notebook mixes unit labels.

    Return radians only. This does not choose Euler order or a robot frame.
    """
    values = finite(values)
    if len(values) != 3:
        raise ValueError("Expected three angles")
    if encoding == "normalized_pi":
        if any(abs(v) > 1 for v in values):
            raise ValueError("Normalized angles must lie in [-1, 1]")
        return tuple(v * math.pi for v in values)
    if encoding == "radians":
        if any(abs(v) > math.pi for v in values):
            raise ValueError("This teaching codec expects angles in [-pi, pi]")
        return values
    raise ValueError("Choose radians or normalized_pi explicitly")


def rotation_zyx(angles, *, encoding):
    """Active rotation of column vectors: Rz(z) @ Ry(y) @ Rx(x).

    Angles are supplied as (x, y, z), not (yaw, pitch, roll).
    This is a declared teaching convention, not automatic protocol detection.
    """
    x, y, z = decode_angles(angles, encoding=encoding)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def box_corners(center, dimensions, angles, *, encoding):
    """Eight camera-frame corners; dimensions are FULL local-axis lengths.

    No sorting into drawing edges, no predicted depth validation against reality.
    Negative/zero camera depth is rejected separately by project_point().
    """
    center, dimensions = finite(center), finite(dimensions)
    if len(center) != 3 or len(dimensions) != 3 or any(d <= 0 for d in dimensions):
        raise ValueError("Expected a 3D center and three positive full dimensions")
    rotation = rotation_zyx(angles, encoding=encoding)
    corners = []
    for signs in product((-1, 1), repeat=3):
        local = tuple(sign * length / 2 for sign, length in zip(signs, dimensions))
        corners.append(finite(tuple(c + sum(r * p for r, p in zip(row, local))
                                    for c, row in zip(center, rotation))))
    return tuple(corners)


def project_point(point, fx, fy, cx, cy):
    """Rectified pinhole projection; reject points at/behind the camera plane."""
    point = finite(point)
    fx, fy, cx, cy = finite((fx, fy, cx, cy))
    if len(point) != 3 or point[2] <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("Expected a 3D point with positive depth and focal lengths")
    x, y, z = point
    return finite((fx * x / z + cx, fy * y / z + cy))


def masked_mse(prediction, target, mask):
    """Single-vector teaching loss; real policies also have time/batch masks."""
    prediction, target = finite(prediction), finite(target)
    mask = tuple(mask)
    if not (len(prediction) == len(target) == len(mask)):
        raise ValueError("Prediction, target and mask must have identical lengths")
    if any(type(v) is not bool for v in mask) or not any(mask):
        raise ValueError("Expected a boolean mask with at least one active axis")
    try:
        errors = [(p - t) ** 2 for p, t, active in zip(prediction, target, mask) if active]
    except OverflowError as error:
        raise ValueError("Non-finite loss") from error
    result = sum(errors) / len(errors)
    if not math.isfinite(result):
        raise ValueError("Non-finite loss")
    return result


def self_test():
    assert normalized_to_pixels(500, 250, 1920, 1080) == (959.5, 269.75)
    assert normalized_to_pixels(1000, 1000, 1920, 1080) == (1919, 1079)
    assert normalized_to_pixels(0, 1000, 1, 1) == (0, 0)
    assert parse_contact("<grasp pose> (500, 250), 190 </grasp pose>") == (500, 250, 10)
    assert backproject(1060, 540, 2, 1000, 1000, 960, 540) == (0.2, 0, 2)
    assert decode_angles((0.5, 0, 0), encoding="normalized_pi")[0] == math.pi / 2
    assert decode_angles((0.5, 0, 0), encoding="radians")[0] == 0.5
    assert masked_mse((2, 999), (0, 0), (True, False)) == 4
    corners = box_corners((0, 0, 2), (0.6, 0.2, 0.4), (0, 0, 0), encoding="radians")
    assert len(corners) == 8
    assert corners[0] == (-0.3, -0.1, 1.8)
    assert project_point((0.2, 0, 2), 1000, 1000, 960, 540) == (1060, 540)
    rotation = rotation_zyx((0, 0, 0.5), encoding="normalized_pi")
    assert math.isclose(rotation[1][0], 1) and abs(rotation[0][0]) < 1e-12
    print("12 CPU teaching checks passed; no model weights or hardware used.")


if __name__ == "__main__":
    self_test()
