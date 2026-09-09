"""Regression checks for the teaching solver's input and numerical boundaries."""
import unittest
from unittest.mock import patch

import numpy as np
from atomic_control import InvalidQPInput, QPSolveError, solve_tiny_qp


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


if __name__ == "__main__":
    unittest.main()
