"""Small, dependency-free detection geometry examples; no model inference.

The resize calculation covers centered, ratio-preserving LetterBox with optional
stride-minimal padding. It omits stretching, scale-up limits, image interpolation,
mask/keypoint transforms and runtime-specific behavior. Coordinates are continuous
box edges, not inclusive pixel indices. Use each recorded transform for inversion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Iterable


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class Placement:
    source_width: int
    source_height: int
    gain: float
    resized_width: int
    resized_height: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def output_size(self) -> tuple[int, int]:
        """Return output (width, height), unlike array shape (height, width)."""
        return (self.resized_width + self.left + self.right,
                self.resized_height + self.top + self.bottom)

    def forward(self, box: Box) -> Box:
        r = self.gain
        return Box(box.x1 * r + self.left, box.y1 * r + self.top,
                   box.x2 * r + self.left, box.y2 * r + self.top)

    def inverse(self, box: Box) -> Box:
        """Invert this transform without clipping or reconstructing its padding."""
        r = self.gain
        return Box((box.x1 - self.left) / r, (box.y1 - self.top) / r,
                   (box.x2 - self.left) / r, (box.y2 - self.top) / r)


def letterbox(width: int, height: int, target_width: int = 640,
              target_height: int = 640, *, minimal: bool = False,
              stride: int = 32) -> Placement:
    """Compute centered resize and integer padding; do not transform image pixels.

    minimal=True reduces total padding modulo stride. It assumes the requested
    target is stride-aligned, as in the article's 640x640 example.
    """
    for value, name in ((width, "width"), (height, "height"),
                        (target_width, "target_width"), (target_height, "target_height"),
                        (stride, "stride")):
        positive_integer(value, name)
    if not isinstance(minimal, bool):
        raise ValueError("minimal must be boolean")
    if minimal and (target_width % stride or target_height % stride):
        raise ValueError("minimal mode requires stride-aligned targets")
    gain = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    horizontal, vertical = target_width - resized_width, target_height - resized_height
    if minimal:
        horizontal %= stride
        vertical %= stride
    # Odd padding puts the extra pixel at the bottom/right. Integer division
    # expresses the split independently of the source's +/-0.1 rounding idiom.
    left, top = horizontal // 2, vertical // 2
    return Placement(width, height, gain, resized_width, resized_height,
                     left, top, horizontal - left, vertical - top)


def normalized_label(box: Box, width: int, height: int) -> tuple[float, ...]:
    positive_integer(width, "width")
    positive_integer(height, "height")
    if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
        raise ValueError("label box must lie inside the original image")
    return ((box.x1 + box.x2) / (2 * width), (box.y1 + box.y2) / (2 * height),
            (box.x2 - box.x1) / width, (box.y2 - box.y1) / height)


def iou(first: Box, second: Box) -> float:
    intersection = (max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
                    * max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1)))
    return intersection / (first.area + second.area - intersection)


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    class_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise ValueError("box must be a Box")
        object.__setattr__(self, "score", finite_number(self.score, "score"))
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between zero and one")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")


def greedy_nms(detections: Iterable[Detection], threshold: float = 0.5,
               *, class_agnostic: bool = False) -> list[Detection]:
    """Illustrate standard greedy NMS; not a replacement for an inference backend.

    Exact-threshold overlaps survive (suppression uses >). Equal scores retain
    input order in this small example, unlike unspecified device/backend ties.
    All input scores are already assumed to pass the confidence filter.
    """
    threshold = finite_number(threshold, "threshold")
    if not 0 <= threshold <= 1 or not isinstance(class_agnostic, bool):
        raise ValueError("threshold must be in [0,1] and class_agnostic must be boolean")
    pending = sorted(list(detections), key=lambda detection: detection.score, reverse=True)
    selected = []
    while pending:
        best = pending.pop(0)
        selected.append(best)
        pending = [other for other in pending
                   if not ((class_agnostic or other.class_id == best.class_id)
                           and iou(best.box, other.box) > threshold)]
    return selected


def example() -> dict:
    original = Box(320, 180, 960, 540)
    square = letterbox(1280, 720)
    rectangle = letterbox(1280, 720, minimal=True)
    candidates = [Detection(Box(0, 0, 100, 100), .95, 0),
                  Detection(Box(10, 10, 110, 110), .90, 0),
                  Detection(Box(10, 10, 110, 110), .85, 1)]
    return {
        "scope": "constructed geometry, no trained model or speed measurement",
        "square": {"output_wh": square.output_size, "left_top": [square.left, square.top],
                   "gain": square.gain, "box": asdict(square.forward(original))},
        "minimal_rectangle": {"output_wh": rectangle.output_size,
                              "left_top": [rectangle.left, rectangle.top],
                              "gain": rectangle.gain, "box": asdict(rectangle.forward(original))},
        "normalized_label": normalized_label(Box(100, 50, 300, 150), 1000, 500),
        "nms": {"first_second_iou": iou(candidates[0].box, candidates[1].box),
                "class_aware_kept_scores": [d.score for d in greedy_nms(candidates)],
                "class_agnostic_kept_scores": [d.score for d in greedy_nms(candidates, class_agnostic=True)],
                "iou_threshold_0_7_kept_scores": [d.score for d in greedy_nms(candidates, .7)]},
    }


def check() -> None:
    """Check known values and geometric invariants; uses only the standard library."""
    from fractions import Fraction
    assert letterbox(1280, 720).output_size == (640, 640)
    assert letterbox(1280, 720, minimal=True).output_size == (640, 384)
    assert letterbox(1280, 720).top == 140
    assert letterbox(1280, 720, minimal=True).top == 12
    assert normalized_label(Box(100, 50, 300, 150), 1000, 500) == (.2, .2, .2, .2)
    a, b = Box(0, 0, 100, 100), Box(10, 10, 110, 110)
    assert math.isclose(iou(a, b), float(Fraction(81, 119)), abs_tol=1e-15)
    assert iou(a, a) == 1 and iou(a, Box(100, 0, 200, 100)) == 0
    for width, height in ((1280, 720), (721, 1281), (1, 3000), (3000, 1), (1, 1), (641, 639)):
        box = Box(0, 0, width, height)
        for minimal in (False, True):
            transform = letterbox(width, height, minimal=minimal)
            restored = transform.inverse(transform.forward(box))
            assert all(math.isclose(getattr(box, key), getattr(restored, key), abs_tol=1e-10)
                       for key in ("x1", "y1", "x2", "y2"))
            assert transform.resized_width >= 1 and transform.resized_height >= 1
            assert 0 <= transform.right - transform.left <= 1
            assert 0 <= transform.bottom - transform.top <= 1
            assert transform.output_size[0] % 32 == transform.output_size[1] % 32 == 0
    rows = [Detection(a, .95, 0), Detection(b, .9, 0), Detection(b, .85, 1)]
    assert [d.score for d in greedy_nms(iter(rows))] == [.95, .85]
    assert [d.score for d in greedy_nms(rows, class_agnostic=True)] == [.95]
    assert len(greedy_nms(rows, .7)) == 3 and greedy_nms([]) == []
    assert len(greedy_nms(rows, 1)) == 3
    for call in (lambda: letterbox(0, 1), lambda: letterbox(True, 2),
                 lambda: letterbox(1, 2, 639, 640, minimal=True),
                 lambda: Box(0, 0, 0, 1), lambda: Box(0, 0, math.inf, 1),
                 lambda: normalized_label(Box(-1, 0, 1, 1), 2, 2),
                 lambda: Detection(a, math.nan, 0), lambda: Detection(a, .5, -1),
                 lambda: greedy_nms([], -.1), lambda: greedy_nms([], math.nan)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check()
    print(json.dumps(example(), ensure_ascii=False, indent=2, allow_nan=False))
