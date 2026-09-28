"""Deterministic 4x11 asymmetric-circle-grid detector check (OpenCV 4.13).

The synthetic image has no lens distortion, motion blur or perspective.
Successful detection verifies the software setup, not a physical camera.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("circle-grid-check"))
    args = parser.parse_args()
    pattern = (4, 11)
    cols, rows = pattern
    image = np.full((600, 520), 255, np.uint8)
    expected = np.array([[100 + 40 * (2 * col + row % 2), 80 + 40 * row]
                         for row in range(rows) for col in range(cols)], np.float32)
    for point in expected:
        cv2.circle(image, tuple(point.astype(int)), 13, 0, -1)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1200
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    found, centers = cv2.findCirclesGrid(
        image, pattern, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=detector,
    )
    if not found:
        raise RuntimeError("Synthetic grid was not found; inspect the OpenCV build and parameters")
    assert centers.shape == (cols * rows, 1, 2)
    # Check the ordering, not just whether the two point sets match.
    errors = np.linalg.norm(centers[:, 0] - expected, axis=1)
    assert float(errors.max()) < 0.1, "Unexpected centre coordinates or point order"
    object_points = np.array([[(2 * col + row % 2) * .01, row * .01, 0]
                              for row in range(rows) for col in range(cols)])
    np.testing.assert_allclose(object_points[1] - object_points[0], [.02, 0, 0])
    np.testing.assert_allclose(object_points[cols] - object_points[0], [.01, .01, 0])

    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(annotated, pattern, centers, found)
    for index in (0, 3, 40, 43):
        x, y = centers[index, 0].astype(int)
        cv2.putText(annotated, str(index), (x - 8, y - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (60, 40, 20), 2, cv2.LINE_AA)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("board.png", image), ("board-detected.png", annotated)]:
        if not cv2.imwrite(str(args.output_dir / name), data):
            raise OSError(f"Cannot write {name}")
    print(json.dumps({"opencv": cv2.__version__, "points": len(centers),
                      "max_center_error_px": float(errors.max()),
                      "ordered_correspondence": "passed",
                      "same_row_spacing_m": .02, "row_spacing_m": .01}, indent=2))


if __name__ == "__main__":
    main()
