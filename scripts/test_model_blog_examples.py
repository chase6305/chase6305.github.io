#!/usr/bin/env python3
"""Offline regression tests for two model explainers. No weights or robot I/O."""

import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


grounding = module("grounding", "content/posts/ai/rynnbrain/grounding_contracts.py")
budget = module("budget", "content/posts/ai/internvl-3-5/token_budget.py")


class ModelBlogExamples(unittest.TestCase):
    def test_article_examples(self):
        grounding.self_test()
        budget.self_test()

    def test_point_endpoints(self):
        for width, height in ((1, 1), (1920, 1080), (17, 400), (4096, 256)):
            self.assertEqual(grounding.normalized_to_pixels(0, 0, width, height), (0, 0))
            self.assertEqual(grounding.normalized_to_pixels(1000, 1000, width, height),
                             (width - 1, height - 1))
        for args in ((-1, 0, 10, 10), (0, 1001, 10, 10), (math.nan, 0, 10, 10),
                     (0, math.inf, 10, 10), (0, 0, 0, 10), (0, 0, 10.5, 10),
                     (0, 0, True, 10), (True, 0, 10, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                grounding.normalized_to_pixels(*args)

    def test_contact_contract(self):
        sample = "<grasp pose> (5e2, 250), -170 </grasp pose>"
        self.assertEqual(grounding.parse_contact(sample), (500, 250, 10))
        for text in ("", "<grasp_pose> (1,2),3 </grasp_pose>", sample + sample,
                     "Explanation: " + sample, sample + " execute now",
                     "<grasp pose> (1001, 250), 0 </grasp pose>",
                     "<grasp pose> (nan, 250), 0 </grasp pose>",
                     "<grasp pose> (500, 250), 1e999 </grasp pose>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                grounding.parse_contact(text)

    def test_camera_roundtrip(self):
        rng = random.Random(20260905)
        for _ in range(100):
            x, y, z = rng.uniform(-2, 2), rng.uniform(-1, 1), rng.uniform(0.2, 10)
            fx, fy, cx, cy = 900, 1100, 960, 540
            u, v = fx * x / z + cx, fy * y / z + cy
            recovered = grounding.backproject(u, v, z, fx, fy, cx, cy)
            for actual, expected in zip(recovered, (x, y, z)):
                self.assertAlmostEqual(actual, expected, places=12)
        for depth, fx in ((0, 1000), (-1, 1000), (1, 0), (1, -1), (math.inf, 1000)):
            with self.subTest(depth=depth, fx=fx), self.assertRaises(ValueError):
                grounding.backproject(0, 0, depth, fx, 1000, 0, 0)

    def test_angle_codec_is_explicit(self):
        self.assertNotEqual(grounding.decode_angles((0.5, 0, 0), encoding="radians"),
                            grounding.decode_angles((0.5, 0, 0), encoding="normalized_pi"))
        for angles, encoding in (((0, 0, 0), "auto"), ((2, 0, 0), "normalized_pi"),
                                 ((4, 0, 0), "radians"), ((0, 0), "radians"),
                                 ((math.nan, 0, 0), "radians")):
            with self.subTest(angles=angles), self.assertRaises(ValueError):
                grounding.decode_angles(angles, encoding=encoding)

    def test_mask(self):
        self.assertEqual(grounding.masked_mse((2, -999), (0, 777), (True, False)), 4)
        for p, t, m in (((1,), (0,), (False,)), ((1,), (0,), (1,)),
                        ((1, 2), (0,), (True,)), ((math.nan,), (0,), (True,)),
                        ((1,), (math.inf,), (True,)), ((1e200,), (0,), (True,))):
            with self.subTest(mask=m), self.assertRaises(ValueError):
                grounding.masked_mse(p, t, m)

    def test_rotation_axes_and_orthogonality(self):
        for axis, source, expected in ((0, (0, 1, 0), (0, 0, 1)),
                                       (1, (0, 0, 1), (1, 0, 0)),
                                       (2, (1, 0, 0), (0, 1, 0))):
            angles = [0, 0, 0]
            angles[axis] = 0.5
            rotation = grounding.rotation_zyx(angles, encoding="normalized_pi")
            actual = [sum(r * v for r, v in zip(row, source)) for row in rotation]
            for a, e in zip(actual, expected):
                self.assertAlmostEqual(a, e)
        rng = random.Random(173)
        composed = grounding.rotation_zyx((0.5, 0, 0.5), encoding="normalized_pi")
        for actual, expected in zip((row[1] for row in composed), (0, 0, 1)):
            self.assertAlmostEqual(actual, expected, msg="Rx must act before Rz on column vectors")
        for _ in range(100):
            angles = [rng.uniform(-1, 1) for _ in range(3)]
            rotation = grounding.rotation_zyx(angles, encoding="normalized_pi")
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(sum(a * b for a, b in zip(rotation[i], rotation[j])), int(i == j))
            a, b, c = rotation
            determinant = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
            self.assertAlmostEqual(determinant, 1)

    def test_box_and_projection(self):
        center, dimensions = (0, 0, 2), (0.6, 0.2, 0.4)
        corners = grounding.box_corners(center, dimensions, (0, 0, 0), encoding="radians")
        self.assertEqual(len(set(corners)), 8)
        self.assertEqual(corners[0], (-0.3, -0.1, 1.8))
        rotated = grounding.box_corners(center, dimensions, (0, 0, 0.5), encoding="normalized_pi")
        for original, changed in zip(corners, rotated):
            self.assertAlmostEqual(sum((p - c)**2 for p, c in zip(original, center)),
                                   sum((p - c)**2 for p, c in zip(changed, center)))
        for i, expected in enumerate((0.1, -0.3, 1.8)):
            self.assertAlmostEqual(rotated[0][i], expected)
        for point in corners:
            pixel = grounding.project_point(point, 1000, 1000, 960, 540)
            recovered = grounding.backproject(*pixel, point[2], 1000, 1000, 960, 540)
            for a, e in zip(recovered, point):
                self.assertAlmostEqual(a, e)
        for point in ((0, 0, 0), (0, 0, -1), (0, 0), (math.inf, 0, 1)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                grounding.project_point(point, 1000, 1000, 960, 540)
        for dims in ((0, 1, 1), (-1, 1, 1), (1, 1), (math.nan, 1, 1)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                grounding.box_corners(center, dims, (0, 0, 0), encoding="radians")

    def test_clipped_surrogate(self):
        for ratio, advantage, expected in ((1.5, 1, 1.2), (0.5, -1, -0.8),
                                           (1.5, -1, -1.5), (0.5, 1, 0.5), (1, 0, 0)):
            self.assertAlmostEqual(budget.clipped_surrogate(ratio, advantage), expected)
        for ratio, advantage, epsilon in ((0, 1, .2), (1, 1, -1), (1, 1, 1),
                                          (math.nan, 1, .2), (1, True, .2), (1e300, 1e300, .2)):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                budget.clipped_surrogate(ratio, advantage, epsilon=epsilon)

    def test_token_budget_properties(self):
        for tiles in range(1, 129):
            self.assertEqual(budget.visual_tokens(tiles), 256 * tiles)
            previous = 0
            for high in range(tiles + 1):
                count = budget.visual_tokens(tiles, high)
                self.assertGreater(count, previous)
                self.assertGreaterEqual(count, 64 * tiles)
                self.assertLessEqual(count, 256 * tiles)
                previous = count
        for tiles, high in ((0, 0), (1, -1), (1, 2), (2, 0.5), (True, 1)):
            with self.subTest(tiles=tiles, high=high), self.assertRaises(ValueError):
                budget.visual_tokens(tiles, high)

    def test_training_arithmetic(self):
        self.assertEqual(budget.square_sample_weights((1, 4, 9)), (1 / 6, 2 / 6, 3 / 6))
        for lengths in ((), (0,), (-1,), (1.5,)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                budget.square_sample_weights(lengths)
        old, new = (math.log(0.2), math.log(0.4)), (math.log(0.4), math.log(0.2))
        self.assertAlmostEqual(budget.sequence_ratio(new, old), 1)
        for new_values, old_values in (((), ()), ((0,), (0, 0)), ((0.1,), (0,)),
                                       ((math.nan,), (0,)), ((0,), (-1000,))):
            with self.subTest(new=new_values), self.assertRaises(ValueError):
                budget.sequence_ratio(new_values, old_values)
        self.assertEqual(budget.group_advantages((0.5, 0.5)), (0, 0))
        advantages = budget.group_advantages((0, 0, 1, 2))
        self.assertAlmostEqual(sum(advantages), 0)
        self.assertAlmostEqual(sum(a * a for a in advantages) / len(advantages), 1)
        for rewards in ((), (1,), (math.nan, 1), (math.inf, 0), (1e200, -1e200)):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                budget.group_advantages(rewards)

    def test_inference_cli_and_syntax_without_weights(self):
        for name in ("rynnbrain", "internvl-3-5"):
            post = ROOT / "content/posts/ai" / name
            for path in post.glob("*.py"):
                ast.parse(path.read_text())
            command = [sys.executable, "-B", str(post / "infer_image.py")]
            help_result = subprocess.run(command + ["--help"], capture_output=True, text=True, timeout=10)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            bad = subprocess.run(command + ["--image", str(post / "nonexistent-fixture.png")],
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("existing local file", bad.stderr)
            self.assertIn("--prepare-only", help_result.stdout)
            self.assertIn("--local-files-only", help_result.stdout)
            for invalid in (["--max-new-tokens", "0"], ["--max-input-tokens", "0"],
                            ["--question", "  "], ["--max-new-tokens", "4097"]):
                result = subprocess.run(command + ["--image", str(post / "index.md")] + invalid,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_inventory_checks_and_pinned_revisions(self):
        review = json.loads((ROOT / "docs/blog-editorial-review.json").read_text())["posts"]
        for record in review:
            if record.get("added_on"):
                article = (ROOT / record["path"]).read_text()
                self.assertGreaterEqual(len(record["acceptance_checks"]), 2)
                for check in record["acceptance_checks"]:
                    self.assertIn(check, article)
                # Only model inference explainers require a pinned loading CLI.
                if Path(record["path"]).parent.name in ("rynnbrain", "internvl-3-5"):
                    script = (ROOT / record["path"]).with_name("infer_image.py").read_text()
                    self.assertIn("trust_remote_code=False", script)
                    self.assertRegex(script, r'REVISION = "[0-9a-f]{40}"')


if __name__ == "__main__":
    unittest.main(verbosity=2)
