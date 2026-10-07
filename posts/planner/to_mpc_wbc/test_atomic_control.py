"""Regression checks for the teaching solver's input and numerical boundaries."""
import unittest
from unittest.mock import patch

import numpy as np
from atomic_control import InvalidQPInput, QPSolveError, mpc, solve_tiny_qp
from feedback_demo import horizon_trap


class TinyQPTests(unittest.TestCase):
    def test_lists_solve_known_convex_problem(self):
        solution, _ = solve_tiny_qp([[2]], [-2], [[1]], [-2], [2])
        np.testing.assert_allclose(solution, [1.], atol=1e-12, rtol=0)

    def test_invalid_shapes_are_input_errors(self):
        valid = [np.eye(1), np.zeros(1), np.ones((1, 1)), -np.ones(1), np.ones(1)]
        for index, replacement in ((0, np.ones(1)), (1, np.array(0.)),
                                   (1, np.zeros(0)), (1, np.zeros(3)),
                                   (2, np.ones((1, 2))), (3, np.zeros((1, 1)))):
            with self.subTest(index=index, shape=replacement.shape):
                data = valid.copy()
                data[index] = replacement
                with self.assertRaises(InvalidQPInput):
                    solve_tiny_qp(*data)

    def test_nonfinite_and_complex_inputs_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf, 1j):
            with self.subTest(value=value), self.assertRaises(InvalidQPInput):
                solve_tiny_qp([[1]], [value], [[1]], [-1], [1])

    def test_nonconvex_or_asymmetric_hessian_is_rejected(self):
        for h in (np.zeros((2, 2)), np.diag([1., -1.]), np.array([[1., .1], [0., 1.]])):
            with self.subTest(h=h.tolist()), self.assertRaises(InvalidQPInput):
                solve_tiny_qp(h, np.zeros(2), np.eye(2), -np.ones(2), np.ones(2))

    def test_inconsistent_constraints_are_solve_errors(self):
        # Both rows individually have valid bounds, but require x >= 1 and x <= 0.
        with self.assertRaises(QPSolveError):
            solve_tiny_qp([[1]], [0], [[1], [1]], [1, -2], [2, 0])
        with self.assertRaises(QPSolveError):
            solve_tiny_qp([[1]], [0], [[1]], [1], [0])

    def test_nonfinite_linear_solutions_never_become_commands(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                with patch("atomic_control.np.linalg.solve",
                           side_effect=lambda matrix, rhs: np.full(rhs.shape, value)):
                    with self.assertRaises(QPSolveError):
                        solve_tiny_qp([[1]], [0], [[1]], [-1], [1])

    def test_mpc_rejects_invalid_scalar_or_initial_state(self):
        for kwargs in ({"goal": [.3]}, {"goal": [.3, .4]}, {"goal": [[.3], [.3, .4]]}, {"goal": True},
                       {"goal": ".3"}, {"goal": 1j}, {"goal": np.inf},
                       {"goal": .3, "q0": .51}, {"goal": .3, "q0": np.nan},
                       {"goal": .3, "v0": -1.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(InvalidQPInput):
                mpc(**kwargs)

    def test_mpc_allows_unreachable_soft_goal(self):
        plan = mpc(2.)
        np.testing.assert_allclose(plan["velocity"], [.2, .4], atol=1e-9, rtol=0)
        self.assertLessEqual(max(plan["positions"]), .5)
        self.assertGreater(abs(plan["positions"][-1] - plan["goal"]), 1.)

    def test_horizon_trap_does_not_hide_input_errors(self):
        first = {"positions": [.4, .46, .5], "velocity": [.6, .4]}
        with patch("feedback_demo.mpc", side_effect=[first, InvalidQPInput("invalid input")]):
            with self.assertRaises(InvalidQPInput):
                horizon_trap()


if __name__ == "__main__":
    unittest.main()
