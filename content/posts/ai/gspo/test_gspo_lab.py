"""Gradient checks, away from clipping boundaries, for the GSPO article."""
import unittest
import torch
from gspo_lab import group_advantages, inspect, objective


class GspoTests(unittest.TestCase):
    def test_cancellation_and_one_sided_clipping(self):
        self.assertEqual(inspect([.5, 2.], 1., "gspo")["gradient"], [.5, .5])
        torch.testing.assert_close(torch.tensor(inspect([.5, 2.], 1., "grpo")["gradient"]),
                                   torch.tensor([.25, 0.]))
        torch.testing.assert_close(torch.tensor(inspect([.5, 2.], -1., "grpo")["gradient"]),
                                   torch.tensor([0., -1.]))
        for ratio, advantage, active in ((1.3, 1., False), (.7, 1., True),
                                          (1.3, -1., True), (.7, -1., False)):
            gradient = inspect([ratio, ratio], advantage, "gspo")["gradient"]
            expected = [advantage * ratio / 2] * 2 if active else [0., 0.]
            torch.testing.assert_close(torch.tensor(gradient), torch.tensor(expected))

    def test_gspo_token_matches_first_gradient_with_variable_lengths(self):
        old = torch.full((2, 3), -3., dtype=torch.float64, requires_grad=True)
        current = (old.detach() + torch.tensor([[.02, -.01, 0.], [.03, -.01, -.02]])).requires_grad_()
        mask = torch.tensor([[True, True, False], [True, True, True]])
        advantages = torch.tensor([1., -1.], dtype=torch.float64, requires_grad=True)
        a = objective(current, old, mask, advantages, "gspo")
        b = objective(current, old, mask, advantages, "gspo-token")
        ga, = torch.autograd.grad(a, current, retain_graph=True)
        gb, = torch.autograd.grad(b, current)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(ga, gb)
        self.assertEqual(ga[0, 2], 0.)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_old_and_advantage_have_no_gradient_path(self):
        old = torch.full((1, 2), -3., requires_grad=True)
        current = old.detach().clone().requires_grad_()
        advantage = torch.ones(1, requires_grad=True)
        value = objective(current, old, torch.ones_like(old, dtype=torch.bool), advantage)
        _, old_grad, advantage_grad = torch.autograd.grad(value, (current, old, advantage), allow_unused=True)
        self.assertIsNone(old_grad)
        self.assertIsNone(advantage_grad)

    def test_mask_excludes_nan_padding_before_arithmetic(self):
        old = torch.tensor([[-3., float('nan')]], dtype=torch.float64)
        current = torch.tensor([[-3., float('nan')]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, False]])
        for kind in ("grpo", "gspo", "gspo-token"):
            value = objective(current, old, mask, torch.ones(1), kind)
            grad, = torch.autograd.grad(value, current)
            torch.testing.assert_close(grad, torch.tensor([[1., 0.]], dtype=torch.float64))

    def test_zero_variance_and_empty_mask(self):
        torch.testing.assert_close(group_advantages(torch.ones((2, 3))), torch.zeros((2, 3)))
        with self.assertRaises(ValueError):
            objective(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.zeros((1, 2), dtype=torch.bool), torch.ones(1))

    def test_gradient_matches_finite_difference(self):
        old = torch.full((2, 3), -3., dtype=torch.float64)
        current = old + torch.tensor([[.03, -.02, .01], [-.04, .02, 0.]], dtype=torch.float64)
        current.requires_grad_()
        mask = torch.tensor([[True, True, True], [True, True, False]])
        advantages = torch.tensor([1., -1.], dtype=torch.float64)
        analytic, = torch.autograd.grad(objective(current, old, mask, advantages), current)
        numerical = torch.zeros_like(current)
        h = 1e-6
        for row in range(2):
            for col in range(3):
                plus, minus = current.detach().clone(), current.detach().clone()
                plus[row, col] += h
                minus[row, col] -= h
                numerical[row, col] = (objective(plus, old, mask, advantages) -
                                       objective(minus, old, mask, advantages)) / (2*h)
        torch.testing.assert_close(analytic, numerical, rtol=1e-6, atol=1e-9)

    def test_normalized_policy_update_and_clipping_plateau(self):
        from gspo_update import run
        rows = run()['history']
        self.assertEqual(rows[0]['objective'], 0.)
        self.assertGreater(rows[1]['exact_expected_reward'], rows[0]['exact_expected_reward'])
        self.assertAlmostEqual(rows[1]['exact_expected_reward'], .544079442108941)
        self.assertGreater(rows[3]['ratios'][-1], 1.2)
        self.assertEqual(rows[3]['flat_fraction'], .5)
        self.assertEqual(rows[3]['exact_expected_reward'], rows[5]['exact_expected_reward'])

    def test_initial_grpo_and_gspo_gradients_agree(self):
        old = torch.full((1, 3), -3., dtype=torch.float64)
        current = old.clone().requires_grad_()
        mask = torch.ones_like(old, dtype=torch.bool)
        gradients = [torch.autograd.grad(objective(current, old, mask, torch.ones(1), kind), current)[0]
                     for kind in ("grpo", "gspo", "gspo-token")]
        for grad in gradients[1:]:
            torch.testing.assert_close(grad, gradients[0])


if __name__ == "__main__":
    unittest.main()
