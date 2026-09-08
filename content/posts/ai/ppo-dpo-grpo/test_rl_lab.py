"""Run with: python -B -m unittest -v test_rl_lab.py (CPU only)."""
import contextlib
import io
import math
import tempfile
import unittest

import torch

import rl_lab as lab
import ppo_chain as chain
import token_objectives as tokens

D = torch.float64


class AtomicAlgorithms(unittest.TestCase):
    def test_ppo_signs_and_flat_regions(self):
        logp = torch.tensor([1.5, .5, 1.5, .5], dtype=D).log().requires_grad_()
        old = torch.zeros(4, dtype=D, requires_grad=True)
        advantage = torch.tensor([1., -1., -1., 1.], dtype=D, requires_grad=True)
        losses = lab.clipped_policy_loss(logp, old, advantage)
        torch.testing.assert_close(losses, torch.tensor([-1.2, .8, 1.5, -.5], dtype=D))
        losses.sum().backward()
        torch.testing.assert_close(logp.grad, torch.tensor([0., 0., 1.5, -.5], dtype=D))
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)

    def test_dpo_direction_and_frozen_reference(self):
        chosen = torch.tensor([-2.], dtype=D, requires_grad=True)
        rejected = torch.tensor([-2.], dtype=D, requires_grad=True)
        reference = torch.tensor([-2.], dtype=D, requires_grad=True)
        loss = lab.dpo_loss(chosen, rejected, reference, reference, beta=.5).mean()
        self.assertAlmostEqual(loss.item(), math.log(2))
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertIsNone(reference.grad)
        improved = lab.dpo_loss(chosen.detach()+.1, rejected.detach()-.1, reference, reference)
        self.assertLess(improved.item(), loss.item())

    def test_gae_terminal_and_lambda_limits(self):
        rewards = torch.tensor([0., 0., 1.], dtype=D)
        zeros = torch.zeros_like(rewards, requires_grad=True)
        terminal = torch.tensor([False, False, True])
        advantages, returns = lab.gae(rewards, zeros, zeros, terminal, terminal, gamma=1, lam=1)
        torch.testing.assert_close(advantages, torch.ones_like(rewards))
        torch.testing.assert_close(returns, advantages)
        self.assertFalse(advantages.requires_grad)
        local, _ = lab.gae(rewards, zeros, zeros, terminal, terminal, gamma=1, lam=0)
        torch.testing.assert_close(local, rewards)

    def test_gae_time_limit_bootstrap_and_trace_boundary(self):
        rewards = torch.tensor([0., 0., 100.], dtype=D)
        values = torch.tensor([.5, .8, 0.], dtype=D)
        next_values = torch.tensor([.8, 2., 0.], dtype=D)
        boundary = torch.tensor([False, True, True])
        terminal = torch.tensor([False, False, True])
        advantages, returns = lab.gae(
            rewards, values, next_values, terminal, boundary, gamma=1, lam=1,
        )
        # The 100 reward belongs to the reset episode; it must not leak backward.
        torch.testing.assert_close(returns, torch.tensor([2., 2., 100.], dtype=D))
        terminal[1] = True
        _, returns = lab.gae(rewards, values, next_values, terminal, boundary, gamma=1, lam=1)
        torch.testing.assert_close(returns, torch.tensor([0., 0., 100.], dtype=D))

    def test_group_locality_and_zero_variance(self):
        rewards = torch.tensor([[0., 0., 1., 1.], [5., 5., 5., 5.]], dtype=D)
        advantages = lab.group_advantages(rewards)
        torch.testing.assert_close(advantages[0], torch.tensor([-1., -1., 1., 1.], dtype=D))
        torch.testing.assert_close(advantages[1], torch.zeros(4, dtype=D))
        shifted = rewards + torch.tensor([[100.], [-9.]], dtype=D)
        torch.testing.assert_close(lab.group_advantages(shifted), advantages)
        with self.assertRaises(ValueError):
            lab.group_advantages(torch.ones(4, 1))

    def test_exact_kl_and_gradient(self):
        logits = torch.tensor([[.1, -.2, .3]], dtype=D, requires_grad=True)
        ref = torch.zeros_like(logits, requires_grad=True)
        kl = lab.categorical_kl(logits, ref)
        self.assertGreater(kl.item(), 0)
        kl.sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertIsNone(ref.grad)
        self.assertAlmostEqual(lab.categorical_kl(logits.detach(), logits.detach()).item(), 0)

    def test_response_shift_mask_and_padding_invariance(self):
        # IDs: two prompt tokens, one answer token, EOS, then padding.
        logits = torch.arange(40, dtype=D).reshape(2, 5, 4) / 11
        logits.requires_grad_()
        ids = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 1, 3, 0]])
        mask = torch.tensor([[False, False, True, True, False]] * 2)
        logps = lab.response_logps(logits, ids, mask)
        expected = torch.stack([
            logits[b, 1].log_softmax(-1)[ids[b, 2]]
            + logits[b, 2].log_softmax(-1)[ids[b, 3]]
            for b in range(2)
        ])
        torch.testing.assert_close(logps, expected)
        logps.sum().backward()
        self.assertEqual(logits.grad[:, [0, 3, 4]].abs().sum().item(), 0)
        padded_logits = torch.cat([logits.detach(), torch.zeros(2, 2, 4, dtype=D)], dim=1)
        padded_ids = torch.cat([ids, torch.zeros(2, 2, dtype=torch.long)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
        torch.testing.assert_close(lab.response_logps(padded_logits, padded_ids, padded_mask), logps)

    def test_invalid_inputs(self):
        x = torch.ones(2, dtype=D)
        for function in (
            lambda: lab.clipped_policy_loss(x, x[:1], x),
            lambda: lab.dpo_loss(x, x, x, x, beta=0),
            lambda: lab.group_advantages(torch.tensor([[float("nan"), 1.]])),
            lambda: lab.response_logps(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long),
                                      torch.zeros(1, 2, dtype=torch.bool)),
        ):
            with self.assertRaises(ValueError):
                function()

    def test_differentiable_loss_gradcheck(self):
        chosen = torch.tensor([-.6, -1.2], dtype=D, requires_grad=True)
        rejected = torch.tensor([-1.1, -.9], dtype=D, requires_grad=True)
        ref = torch.full_like(chosen, -1.)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: lab.dpo_loss(a, b, ref, ref),
                                                (chosen, rejected)))


class TrainingLoops(unittest.TestCase):
    def test_grpo_fixed_action_budget(self):
        for group_size in (2, 4, 8, 16):
            prompts = 128 // group_size
            rows, settings = lab.train("grpo", steps=8, group_size=group_size, grpo_prompts=prompts)
            self.assertEqual(settings["online_batch_prompts"], prompts)
            self.assertEqual(rows[-1]["sampled_actions"], 8 * 128)
            self.assertTrue(all(0 <= row["zero_group_fraction"] <= 1 for row in rows))
        default = lab.train("grpo", steps=8)
        self.assertEqual(default, lab.train("grpo", steps=8, group_size=8, grpo_prompts=16))
        for count in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                lab.train("grpo", grpo_prompts=count)

    def test_preference_corruption_separates_loss_from_reward(self):
        for seed in (0, 7, 19):
            with self.subTest(seed=seed):
                rows, settings = lab.train("dpo", steps=80, seed=seed, preference_flips=12)
                self.assertLess(rows[-1]["dpo_training_loss"], rows[0]["dpo_training_loss"])
                self.assertGreater(rows[-1]["dpo_clean_loss"], rows[0]["dpo_clean_loss"])
                self.assertLess(rows[-1]["expected_reward"], -.4)
                self.assertEqual(settings["reward_table"], lab.reward_table().tolist())
                for clean, noisy in zip(settings["clean_preference_pairs"], settings["training_preference_pairs"]):
                    self.assertEqual(noisy, [clean[0], clean[2], clean[1]])

    def test_preference_flip_count_and_reproducibility(self):
        first, settings = lab.train("dpo", steps=5, seed=7, preference_flips=3)
        self.assertEqual((first, settings), lab.train("dpo", steps=5, seed=7, preference_flips=3))
        changed = [i for i, (a, b) in enumerate(zip(settings["clean_preference_pairs"],
                                                    settings["training_preference_pairs"])) if a != b]
        self.assertEqual(changed, settings["flipped_pair_indices"])
        self.assertEqual(len(changed), 3)
        self.assertEqual(first[-1]["pair_presentations"], 60)
        clean, _ = lab.train("dpo", steps=5)
        self.assertTrue(all(row["dpo_training_loss"] == row["dpo_clean_loss"] for row in clean))
        for flips in (-1, 13, .5, True):
            with self.assertRaises(ValueError):
                lab.train("dpo", preference_flips=flips)
        with self.assertRaises(ValueError):
            lab.train("ppo", preference_flips=1)

    def test_learning_across_seeds(self):
        for name in ("ppo", "dpo", "grpo"):
            for seed in (0, 7, 19):
                with self.subTest(algorithm=name, seed=seed):
                    rows, settings = lab.train(name, steps=80, seed=seed)
                    self.assertGreater(rows[-1]["best_action_probability"], .9)
                    self.assertGreater(rows[-1]["expected_reward"], rows[0]["expected_reward"]+.6)
                    self.assertTrue(all(
                        math.isfinite(value)
                        for row in rows for value in row.values() if isinstance(value, float)
                    ))
                    self.assertEqual(settings["device"], "cpu")

    def test_reproducibility_and_cli_artifacts(self):
        first, _ = lab.train("grpo", steps=8, seed=3)
        second, _ = lab.train("grpo", steps=8, seed=3)
        self.assertEqual(first, second)
        import json
        import sys
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            args = ["rl_lab.py", "--algorithm", "all", "--steps", "3",
                    "--preference-flips", "3", "--output", tmp]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                lab.main()
            for name in ("ppo", "dpo", "grpo"):
                data = json.loads((Path(tmp)/f"{name}.json").read_text())
                self.assertEqual(data["final"]["step"], 3)
                if name == "dpo":
                    self.assertEqual(data["settings"]["preference_flips"], 3)
                else:
                    self.assertNotIn("preference_flips", data["settings"])
                self.assertEqual(len((Path(tmp)/f"{name}.csv").read_text().splitlines()), 5)


class TokenObjectives(unittest.TestCase):
    def test_equal_policy_length_and_nonzero_grpo_gradient(self):
        report = tokens.run()
        self.assertEqual(report["alignment"]["lengths"], [2, 3])
        self.assertAlmostEqual(report["dpo"]["initial_loss"], math.log(2))
        self.assertAlmostEqual(report["dpo"]["relative_margin_before"], 0)
        self.assertAlmostEqual(report["grpo"]["initial_loss"], 0)
        self.assertAlmostEqual(report["initial_global_token_loss"], .2)
        for name in ("dpo", "grpo"):
            self.assertGreater(report[name]["gradient_norm"], 0)
            self.assertLess(report[name]["final_loss"], report[name]["initial_loss"])
            self.assertGreater(report[name]["relative_margin_after"], 0)

    def test_padding_preserves_losses_and_parameter_gradients(self):
        for name in ("dpo", "grpo"):
            results = []
            for padding in (0, 4):
                actor = (torch.arange(25, dtype=D).reshape(5, 5) / 37).requires_grad_()
                old = torch.zeros_like(actor, requires_grad=True)
                reference = torch.zeros_like(actor, requires_grad=True)
                ids, mask = tokens.fixture(padding)
                loss = tokens.objectives(actor, old, reference, ids, mask)[name]
                loss.backward()
                self.assertIsNone(old.grad)
                self.assertIsNone(reference.grad)
                results.append((loss.detach(), actor.grad))
            torch.testing.assert_close(results[0], results[1])

    def test_eos_shared_with_padding_still_counts(self):
        ids, mask = tokens.fixture()
        logits = torch.zeros(2, 5, 5, dtype=D)
        correct = lab.response_logps(logits, ids, mask)
        wrongly_removed_eos = lab.response_logps(logits, ids, mask & (ids != 4))
        torch.testing.assert_close(wrongly_removed_eos - correct,
                                   torch.full((2,), math.log(5), dtype=D))
        self.assertEqual(mask[0, 3:].tolist(), [True, False])
        self.assertEqual(ids[0, 3:].tolist(), [4, 4])


class MultistepPPO(unittest.TestCase):
    def test_delayed_reward_credit_and_episode_order(self):
        actor = torch.zeros(3, 2, dtype=D, requires_grad=True)
        critic = torch.zeros(3, dtype=D, requires_grad=True)
        batch = chain.rollout(actor, critic, torch.Generator().manual_seed(7))
        rewards = batch["rewards"].reshape(-1, 2)
        self.assertTrue((rewards[:, 0] == 0).all())
        self.assertTrue((rewards[:, 1] == 1).any())
        self.assertTrue((rewards[:, 1] == 0).any())
        states = batch["states"].reshape(-1, 2)
        actions = batch["actions"].reshape(-1, 2)
        torch.testing.assert_close(states[:, 0], torch.zeros(64, dtype=torch.long))
        torch.testing.assert_close(states[:, 1], actions[:, 0] + 1)
        for lam in (0., .95, 1.):
            advantage, returns = lab.gae(
                batch["rewards"], batch["values"], batch["next_values"],
                batch["terminated"], batch["boundary"], gamma=1., lam=lam,
            )
            # Each first decision inherits ONLY its own terminal reward.
            torch.testing.assert_close(advantage[::2], lam * rewards[:, 1], rtol=0, atol=1e-14)
            torch.testing.assert_close(advantage[1::2], rewards[:, 1])
            self.assertFalse(returns.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in batch.values()))

    def test_multistep_learning_and_reproducibility(self):
        for seed in (0, 7, 19):
            with self.subTest(seed=seed):
                rows, report = chain.train(steps=80, seed=seed)
                self.assertAlmostEqual(rows[0]["expected_return"], .3)
                self.assertGreater(rows[-1]["expected_return"], .97)
                self.assertGreater(rows[-1]["left_probability"], .97)
                self.assertEqual(rows[-1]["sampled_transitions"], 80 * 128)
                self.assertTrue(all(math.isfinite(v) for row in rows for v in row.values()))
                self.assertFalse(report["settings"]["reference_kl"])
        self.assertEqual(chain.train(steps=5, seed=7), chain.train(steps=5, seed=7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
