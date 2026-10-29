import unittest
from prompt_eval import context_budget, paired_outcomes, summarize, wilson


class PromptEvalTests(unittest.TestCase):
    def test_prompt_iterators_are_not_consumed_before_accounting(self):
        expected = context_budget(30, [6, 8], 12, 100)
        for durations in (iter([6, 8]), (x for x in [6, 8])):
            with self.subTest(kind=type(durations).__name__):
                self.assertEqual(context_budget(30, durations, 12, 100), expected)
        self.assertEqual(context_budget(30, iter([]), 12, 100),
                         dict(used_seconds=12, free_seconds=18,
                              nominal_action_samples=1200))
        with self.assertRaises(ValueError):
            context_budget(30, (x for x in [6, 13]), 12, 100)
        with self.assertRaises(ValueError):
            context_budget(30, (x for x in [6, float('nan')]), 12, 100)

    def test_task_iterators_match_sequences_and_reject_empty_or_invalid(self):
        tasks = [(9, 10), (10, 20)]
        expected = summarize(tasks)
        self.assertEqual(summarize(iter(tasks)), expected)
        self.assertEqual(summarize(pair for pair in tasks), expected)
        self.assertEqual(summarize(iter([(0, 1), (1, 1)])), dict(macro=.5, micro=.5))
        for tasks in (iter([]), iter([(1, 1), (2, 1)]), iter([(1, 1), (0, 0)])):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                summarize(tasks)

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

    def test_paired_outcomes_and_reversal(self):
        left = [True, True, True, True, True, False, False, False]
        right = [True, True, False, False, False, True, False, False]
        result = paired_outcomes(left, right)
        self.assertEqual([result[k] for k in ('both', 'left_only', 'right_only', 'neither')], [2, 3, 1, 2])
        self.assertEqual(result['success_rate_difference'], .25)
        self.assertEqual(paired_outcomes(right, left)['success_rate_difference'], -.25)

    def test_paired_missing_or_nonboolean_trials_rejected(self):
        for left, right in [([], []), ([True], []), ([True], [None]), ([1], [False])]:
            with self.assertRaises(ValueError):
                paired_outcomes(left, right)

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
