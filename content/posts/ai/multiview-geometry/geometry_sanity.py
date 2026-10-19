"""Constructed pinhole examples for the multiview geometry article.

Python 3.9+, standard library only. No images, models or measured data.
Coordinate convention: +x right, +y down, +z forward; lengths in metres.
The right camera has the same orientation and centre (baseline, 0, 0).
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Pinhole:
    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, point):
        x, y, z = point
        if not all(math.isfinite(value) for value in point) or z <= 0:
            raise ValueError("Projection requires a finite point with positive Z")
        return self.fx * x / z + self.cx, self.fy * y / z + self.cy

    def unproject(self, pixel, depth_z):
        if not math.isfinite(depth_z) or depth_z <= 0:
            raise ValueError("Z-depth must be finite and positive")
        u, v = pixel
        return (
            (u - self.cx) * depth_z / self.fx,
            (v - self.cy) * depth_z / self.fy,
            depth_z,
        )

    def crop_resize(self, x0, y0, sx, sy):
        # Continuous coordinates, without padding or half-pixel offsets.
        return Pinhole(self.fx * sx, self.fy * sy,
                       (self.cx - x0) * sx, (self.cy - y0) * sy)


def transfer(camera, pixel, depth_z, baseline, wrong_direction=False):
    """Left -> right with parallel axes. Correct world2cam translation is -B."""
    x, y, z = camera.unproject(pixel, depth_z)
    translation_x = baseline if wrong_direction else -baseline
    return camera.project((x + translation_x, y, z))


def pixel_error(predicted, observed):
    return math.dist(predicted, observed)


def calculate_case(
    baseline_m=0.12,
    depth_z_m=2.0,
    reverse_pose=False,
    crop_resize=False,
    stale_intrinsics=False,
    match_offset_px=0.0,
):
    """Recompute a browser-lab input through explicit 3D backprojection."""
    for name, value, low, high in (
        ("baseline_m", baseline_m, 0.01, 0.5),
        ("depth_z_m", depth_z_m, 0.1, 10),
        ("match_offset_px", match_offset_px, -200, 200),
    ):
        if (type(value) not in (int, float) or not low <= value <= high
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    for value in (reverse_pose, crop_resize, stale_intrinsics):
        if type(value) is not bool:
            raise ValueError("Switches must be booleans")
    if stale_intrinsics and not crop_resize:
        raise ValueError("Stale intrinsics require crop_resize=True")
    camera = Pinhole(800.0, 800.0, 320.0, 240.0)
    left = camera.project((0.2, 0.1, 2.0))
    observed = camera.project((0.2 - 0.12, 0.1, 2.0))
    matched = (observed[0] + match_offset_px, observed[1])
    if crop_resize:
        left, observed, matched = [
            ((u - 80) * 0.5, (v - 40) * 0.5)
            for u, v in (left, observed, matched)
        ]
        if not stale_intrinsics:
            camera = camera.crop_resize(80, 40, 0.5, 0.5)
    projected = transfer(camera, left, depth_z_m, baseline_m, reverse_pose)
    width, height = (280, 220) if crop_resize else (640, 480)

    def in_frame(pixel):
        return 0 <= pixel[0] < width and 0 <= pixel[1] < height

    return {
        "input": {
            "baseline_m": baseline_m, "depth_z_m": depth_z_m,
            "reverse_pose": reverse_pose, "crop_resize": crop_resize,
            "stale_intrinsics": stale_intrinsics, "match_offset_px": match_offset_px,
        },
        "grid": {"width": width, "height": height,
                 "scale_from_original": 0.5 if crop_resize else 1},
        "left_uv": list(left), "observed_right_uv": list(observed),
        "matched_right_uv": list(matched), "projected_right_uv": list(projected),
        "reprojection_error_px": pixel_error(projected, observed),
        "match_reprojection_error_px": pixel_error(projected, matched),
        "vertical_epipolar_error_px": abs(matched[1] - left[1]),
        "baseline_scale": baseline_m / 0.12, "depth_scale": depth_z_m / 2,
        "depth_relative_error_percent": (depth_z_m / 2 - 1) * 100,
        "projected_in_frame": in_frame(projected),
        "matched_in_frame": in_frame(matched),
    }


def run_experiment():
    camera = Pinhole(800.0, 800.0, 320.0, 240.0)
    baseline = 0.12
    point_world = (0.2, 0.1, 2.0)  # Left camera is the world reference.
    left = camera.project(point_world)
    right = camera.project((point_world[0] - baseline, *point_world[1:]))
    disparity = left[0] - right[0]
    depth_z = camera.fx * baseline / disparity

    # Case 1: pose direction. These cameras have R=I, so inverting the pose
    # flips its translation. No estimated model outputs are involved.
    correct = transfer(camera, left, depth_z, baseline)
    reversed_pose = transfer(camera, left, depth_z, baseline, wrong_direction=True)

    # Case 2: crop at (80,40), then scale both axes by 0.5. The observed right
    # pixel is transformed independently of the backprojection implementation.
    x0, y0, sx, sy = 80.0, 40.0, 0.5, 0.5
    left_processed = ((left[0] - x0) * sx, (left[1] - y0) * sy)
    right_processed = ((right[0] - x0) * sx, (right[1] - y0) * sy)
    processed_camera = camera.crop_resize(x0, y0, sx, sy)
    correct_processed = transfer(processed_camera, left_processed, depth_z, baseline)
    stale_intrinsics = transfer(camera, left_processed, depth_z, baseline)

    # Case 3: simultaneous 2x baseline and depth is invisible to perspective
    # projection, but doubles reconstructed physical distances.
    scaled = transfer(camera, left, depth_z * 2, baseline * 2)
    wrong_baseline_only = transfer(camera, left, depth_z, baseline * 2)

    # Case 4: all correspondences remain on their epipolar lines when the
    # baseline magnitude is wrong. Even a gross horizontal mismatch passes
    # this rectified vertical-residual check, but not reprojection with depth.
    wrong_horizontal_match = (right[0] + 100.0, right[1])

    cases = [
        ("correct_geometry", right, correct),
        ("pose_direction_reversed", right, reversed_pose),
        ("crop_resize_with_updated_K", right_processed, correct_processed),
        ("crop_resize_with_stale_K", right_processed, stale_intrinsics),
        ("baseline_and_depth_both_x2", right, scaled),
        ("baseline_only_x2", right, wrong_baseline_only),
    ]
    rows = [
        {"case": name, "predicted_right_uv": list(prediction),
         "observed_right_uv": list(observed),
         "reprojection_error_px": pixel_error(prediction, observed)}
        for name, observed, prediction in cases
    ]
    return {
        "data": "constructed, not model measurements",
        "original_K": {"fx": camera.fx, "fy": camera.fy,
                       "cx": camera.cx, "cy": camera.cy},
        "left_uv": list(left), "right_uv": list(right),
        "baseline_m": baseline, "depth_z_m": depth_z,
        "disparity_px": disparity,
        "processed_K": {"fx": processed_camera.fx, "fy": processed_camera.fy,
                        "cx": processed_camera.cx, "cy": processed_camera.cy},
        "cases": rows,
        "epipolar_blind_spot": {
            "wrong_horizontal_match_uv": list(wrong_horizontal_match),
            "vertical_residual_px": abs(wrong_horizontal_match[1] - left[1]),
            "reprojection_error_px": pixel_error(correct, wrong_horizontal_match),
        },
    }


def verify(result):
    """Check hand-computed projections and deliberately broken conditions."""
    for actual, expected in zip(result["left_uv"], (400.0, 280.0)):
        assert math.isclose(actual, expected, abs_tol=1e-9)
    for actual, expected in zip(result["right_uv"], (352.0, 280.0)):
        assert math.isclose(actual, expected, abs_tol=1e-9)
    expected_errors = (0.0, 96.0, 0.0, 24.0, 0.0, 48.0)
    for row, expected in zip(result["cases"], expected_errors):
        assert math.isclose(row["reprojection_error_px"], expected, abs_tol=1e-9), row
    assert result["processed_K"] == {"fx": 400.0, "fy": 400.0, "cx": 120.0, "cy": 100.0}
    assert result["epipolar_blind_spot"]["vertical_residual_px"] == 0.0
    assert math.isclose(result["epipolar_blind_spot"]["reprojection_error_px"], 100.0)

    # At a noncentral pixel, Z-depth differs from Euclidean range. Updating
    # K must preserve the actual camera-space point through crop/resize.
    camera = Pinhole(800.0, 800.0, 320.0, 240.0)
    original = camera.unproject((400.0, 280.0), 2.0)
    recovered = camera.crop_resize(80, 40, 0.5, 0.5).unproject((160.0, 120.0), 2.0)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(original, recovered))
    assert math.sqrt(sum(value * value for value in original)) > 2.0
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        try:
            camera.unproject((400.0, 280.0), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid Z-depth was accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Browser-lab JSON file, or - for standard input")
    args = parser.parse_args()
    if args.input is not None:
        try:
            if args.input == "-":
                payload = json.load(sys.stdin)
            else:
                with open(args.input, encoding="utf-8") as source:
                    payload = json.load(source)
            # Recompute from inputs; ignore any previously exported results.
            result = calculate_case(**payload["input"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(str(error))
    else:
        result = run_experiment()
        verify(result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
