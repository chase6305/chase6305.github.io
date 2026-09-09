import unittest
from prompt_eval import context_budget, summarize, wilson


class PromptEvalTests(unittest.TestCase):
    def test_window_boundary_and_overflow(self):
        self.assertEqual(context_budget(30, [6, 12], 12, 100)['free_seconds'], 0)
        with self.assertRaises(ValueError):
            context_budget(30, [6, 13], 12, 100)
        with self.assertRaises(ValueError):
            context_budget(30, [float('nan')], 12, 100)

    def test_task_weighting_changes_aggregate(self):
        result = summarize([(9, 10), (10, 20)])
        self.assertAlmostEqual(result['macro'], .7)
        self.assertAlmostEqual(result['micro'], 19/30)

    def test_wilson_boundary_and_count_validation(self):
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0.)
        self.assertAlmostEqual(high, .2775327998628892)
        with self.assertRaises(ValueError):
            wilson(11, 10)
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == '__main__':
    unittest.main()
