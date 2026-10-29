#!/usr/bin/env python3
"""Check mapping structure and portable numerical publication records."""
import copy
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'content/posts/ai/morphmimic'
spec = importlib.util.spec_from_file_location('morphmimic_article_contracts', BUNDLE / 'morphmimic_contracts.py')
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


class RecordChecks(unittest.TestCase):
    def test_float_accumulation_rounding(self):
        contracts.compare_record({'center': [.4000000000000001]}, {'center': [.39999999999999997]})

    def test_changed_published_precision(self):
        with self.assertRaisesRegex(ValueError, r'\$\.return: number differs'):
            contracts.compare_record({'return': .9358}, {'return': .9359})

    def test_integer_and_boolean_fields(self):
        for left, right in [(930, 931), (930, 930.), (True, 1), (True, False)]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                contracts.compare_record(left, right)

    def test_nonfinite_values(self):
        for value in [math.nan, math.inf, -math.inf]:
            for left, right in [(value, value), (.1, value), (value, .1)]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    contracts.compare_record(left, right)

    def test_structure(self):
        for left, right in [({'a': 1}, {'b': 1}), ([1], [1, 2]), ([1], {'0': 1})]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                contracts.compare_record(left, right)

    def test_clean_and_corrupt_cli_records(self):
        with tempfile.TemporaryDirectory(prefix='chase-morphmimic-record-') as folder:
            bundle = Path(folder)
            (bundle / 'assets').mkdir()
            shutil.copyfile(BUNDLE / 'morphmimic_contracts.py', bundle / 'morphmimic_contracts.py')
            record = json.loads((BUNDLE / 'assets/contracts-results.json').read_text())
            def check(data):
                (bundle / 'assets/contracts-results.json').write_text(json.dumps(data))
                return subprocess.run([sys.executable, '-B', str(bundle / 'morphmimic_contracts.py'), '--check'],
                                      capture_output=True, text=True, timeout=10)
            good = check(record)
            self.assertEqual(good.returncode, 0, good.stderr)
            corrupted = copy.deepcopy(record)
            corrupted['reported_comparisons'][0]['unified_return'] += .0001
            bad = check(corrupted)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('$.reported_comparisons[0].unified_return', bad.stderr)
            self.assertNotIn('passed', bad.stdout)


class MappingChecks(unittest.TestCase):
    def test_random_roundtrips_and_untracked_defaults(self):
        rng = random.Random(20261029)
        for _ in range(500):
            size = rng.randrange(1, 13)
            native = [rng.uniform(-3, 3) for _ in range(size)]
            default = [rng.uniform(-1, 1) for _ in range(size)]
            indices = rng.sample(range(size), rng.randrange(1, size + 1))
            signs = [rng.choice([-1, 1]) for _ in indices]
            offsets = [rng.uniform(-.2, .2) for _ in indices]
            canonical = contracts.canonical_position(native, indices, signs, offsets)
            restored = contracts.native_position(canonical, indices, signs, offsets, default)
            for index in range(size):
                if index in indices:
                    self.assertAlmostEqual(restored[index], native[index], places=12)
                else:
                    self.assertEqual(restored[index], default[index])

    def test_invalid_mapping_never_truncates_or_overwrites(self):
        invalid = [([2, 0, 1], [-1, 1], [.1, .2, -.3]),
                   ([2, 0, 1], [-1, 1, -1], [.1, .2]),
                   ([0, 0, 1], [1, 1, 1], [0, 0, 0]),
                   ([-1, 0, 1], [1, 1, 1], [0, 0, 0]),
                   ([4, 0, 1], [1, 1, 1], [0, 0, 0]),
                   ([True, 0, 1], [1, 1, 1], [0, 0, 0]),
                   ([2, 0, 1], [0, 1, 1], [0, 0, 0]),
                   ([2, 0, 1], [True, 1, 1], [0, 0, 0]),
                   ([2, 0, 1], [1, 1, 1], [math.nan, 0, 0]),
                   ([], [], []), (None, [1], [0])]
        for indices, signs, offsets in invalid:
            with self.subTest(indices=indices, signs=signs, offsets=offsets):
                with self.assertRaises(ValueError):
                    contracts.canonical_position([.5, -.2, .9, 1.7], indices, signs, offsets)
                with self.assertRaises(ValueError):
                    contracts.native_position([0, 0, 0], indices, signs, offsets, [0, 0, 0, 1.7])

    def test_inverse_requires_every_mapped_joint(self):
        for canonical in ([.2], [.2, .3, .4]):
            with self.assertRaises(ValueError):
                contracts.native_position(canonical, [0, 1], [1, -1], [0, 0], [.5, .6])

    def test_coordinates_are_finite_and_inputs_are_unchanged(self):
        for value in (math.nan, math.inf, -math.inf, True, "0", 1j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.canonical_position([value], [0], [1], [0])
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.native_position([value], [0], [1], [0], [0])
        default = [.5, 1.7]
        self.assertEqual(contracts.native_position([.2], [0], [1], [0], default), [.2, 1.7])
        self.assertEqual(default, [.5, 1.7])


if __name__ == '__main__':
    unittest.main(verbosity=2)
