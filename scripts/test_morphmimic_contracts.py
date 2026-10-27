#!/usr/bin/env python3
"""Keep numerical publication checks portable without accepting changed results."""
import copy
import importlib.util
import json
import math
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
