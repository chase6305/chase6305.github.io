"""Tiny, inspectable control QPs; NumPy only, no WholeBodyX or robot hardware.

Run: python atomic_control.py
Enumerating active sets is only suitable for this two-variable teaching example.
"""
import argparse
import csv
from pathlib import Path
from itertools import combinations
import json

import numpy as np


class InvalidQPInput(ValueError):
    """Data shape, finiteness or convexity is outside this example's contract."""


class QPSolveError(ValueError):
    """No valid command: infeasible constraints or numerically unresolved solve."""


def solve_tiny_qp(h, g, a, lower, upper):
    """Solve strictly convex min .5*x'Hx + g'x, lower <= A*x <= upper.

    Enumerate independent active subsets and require primal feasibility,
    nonnegative multipliers and stationarity (the convex KKT certificate).
    Only intended for finite bounds and at most two decision variables.
    """
    data = (h, g, a, lower, upper)
    try:
        if any(np.iscomplexobj(value) for value in data):
            raise InvalidQPInput("QP data must be real")
        h, g, a, lower, upper = (np.asarray(value, dtype=float) for value in data)
    except (TypeError, ValueError) as exc:
        raise InvalidQPInput("QP data must be real numeric arrays") from exc
    if g.ndim != 1 or not 1 <= g.size <= 2:
        raise InvalidQPInput("g must be a vector with one or two elements")
    n = g.size
    if h.shape != (n, n) or a.ndim != 2 or a.shape[1] != n or a.shape[0] < 1:
        raise InvalidQPInput("H must be n-by-n and A must be m-by-n with m >= 1")
    if lower.shape != (a.shape[0],) or upper.shape != lower.shape:
        raise InvalidQPInput("lower and upper must be vectors with one entry per row of A")
    if not all(np.isfinite(value).all() for value in (h, g, a, lower, upper)):
        raise InvalidQPInput("This example requires finite QP data")
    if not np.allclose(h, h.T, rtol=0, atol=1e-12):
        raise InvalidQPInput("H must be symmetric within absolute tolerance 1e-12")
    h = h / 2 + h.T / 2
    if np.linalg.eigvalsh(h).min() <= 0:
        raise InvalidQPInput("H must be positive definite for this tiny solver")
    if np.any(lower > upper):
        raise QPSolveError("Infeasible bounds: lower exceeds upper")
    c, d = np.vstack((a, -a)), np.concatenate((upper, -lower))
    for count in range(n + 1):
        for active in combinations(range(len(d)), count):
            ca = c[list(active)]
            system = np.block([[h, ca.T], [ca, np.zeros((count, count))]])
            try:
                solution = np.linalg.solve(system, np.r_[-g, d[list(active)]])
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(solution).all():
                continue
            x, multipliers = solution[:n], solution[n:]
            with np.errstate(over="ignore", invalid="ignore"):
                feasibility = c @ x - d
                gradient = h @ x + g + ca.T @ multipliers
            if not np.isfinite(feasibility).all() or not np.isfinite(gradient).all():
                continue
            primal = max(0., float(np.max(feasibility)))
            dual = max(0., float(np.max(-multipliers))) if count else 0.
            stationarity = float(np.max(np.abs(gradient)))
            if max(primal, dual, stationarity) <= 1e-9:
                return x, dict(primal=primal, dual=dual, stationarity=stationarity)
    raise QPSolveError("No KKT candidate found; infeasible or numerically unresolved")


def mpc(goal, q0=0., v0=0.):
    dt = .1
    prediction = dt * np.tril(np.ones((2, 2)))
    weights = np.diag([10., 30.])
    h = prediction.T @ weights @ prediction + .1 * np.eye(2)
    g = prediction.T @ weights @ np.full(2, q0 - goal)
    difference = np.array([[1., 0.], [-1., 1.]])
    a = np.vstack((np.eye(2), prediction, difference))
    lower = np.r_[[-1., -1.], np.full(2, -.5 - q0), [v0 - .2, -.2]]
    upper = np.r_[[1., 1.], np.full(2, .5 - q0), [v0 + .2, .2]]
    velocity, residuals = solve_tiny_qp(h, g, a, lower, upper)
    return dict(goal=goal, velocity=velocity.tolist(),
                positions=np.r_[q0, q0 + prediction @ velocity].tolist(), kkt=residuals)


def wbc():
    # Two conflicting scalar velocity tasks and a small regularizer.
    targets, weights, regularization = np.array([.8, -.4]), np.array([4., 1.]), .1
    h = np.array([[weights.sum() + regularization]])
    g = np.array([-weights @ targets])
    command, residuals = solve_tiny_qp(h, g, np.ones((1, 1)), np.array([-.2]), np.array([.2]))
    unconstrained = float(-g[0] / h[0, 0])
    np.testing.assert_allclose(command, [.2], atol=1e-9, rtol=0)
    return dict(unconstrained=unconstrained, command=float(command[0]),
                task_residuals=(command[0] - targets).tolist(), kkt=residuals)


def coupled_wbc():
    """Two coupled velocity tasks: clipping is feasible but not optimal."""
    jacobian = np.array([[1., 1.], [1., -1.]])
    targets = np.array([1., 0.])
    weights = np.diag([4., 1.])
    regularization = .1
    h = jacobian.T @ weights @ jacobian + regularization * np.eye(2)
    g = -jacobian.T @ weights @ targets
    lower, upper = np.array([-.2, -.8]), np.array([.2, .8])
    unconstrained = np.linalg.solve(h, -g)
    clipped = np.clip(unconstrained, lower, upper)
    command, residuals = solve_tiny_qp(h, g, np.eye(2), lower, upper)

    def objective(u):
        error = jacobian @ u - targets
        return float(.5 * (error @ weights @ error + regularization * (u @ u)))

    np.testing.assert_allclose(command, [.2, 2. / 3.], atol=1e-9, rtol=0)
    if objective(command) >= objective(clipped) - 1e-9:
        raise AssertionError("Coupled QP should improve over elementwise clipping")
    return dict(unconstrained=unconstrained.tolist(), clipped=clipped.tolist(),
                command=command.tolist(), clipped_objective=objective(clipped),
                optimal_objective=objective(command),
                task_residuals=(jacobian @ command - targets).tolist(), kkt=residuals)


def priority_wbc():
    """Weighted tasks vs an analytically reduced two-level hierarchy.

    The high-priority sum target is feasible, so its optimum residual is zero.
    Preserve it exactly by writing u = [t, target - t] in the lower-level QP.
    This is a specific two-task example, not a general hierarchical QP solver.
    """
    target, regularization = .6, .1
    lower, upper = np.array([-.2, -.8]), np.array([.2, .8])
    jacobian = np.array([[1., 1.], [1., -1.]])
    weighted = []
    for weight in (4., 40., 400.):
        weights = np.diag([weight, 1.])
        h = jacobian.T @ weights @ jacobian + regularization * np.eye(2)
        g = -jacobian.T @ weights @ np.array([target, 0.])
        command, _ = solve_tiny_qp(h, g, np.eye(2), lower, upper)
        # Independent scalar derivative with the first joint at its upper bound.
        expected_second = (.4 * weight + .2) / (weight + 1.1)
        np.testing.assert_allclose(command, [.2, expected_second], atol=1e-9, rtol=0)
        weighted.append(dict(weight=weight, command=command.tolist(),
                             high_residual=float(command.sum() - target),
                             low_residual=float(command[0] - command[1])))

    if not lower.sum() <= target <= upper.sum():
        raise AssertionError("This tutorial requires a feasible high-priority target")
    anchor, null = np.array([0., target]), np.array([[1.], [-1.]])
    low_jacobian = jacobian[1:]
    low_h = low_jacobian.T @ low_jacobian + regularization * np.eye(2)
    reduced_h = null.T @ low_h @ null
    reduced_g = (null.T @ low_h @ anchor).reshape(1)
    t_lower = max(lower[0], target - upper[1])
    t_upper = min(upper[0], target - lower[1])
    t, residuals = solve_tiny_qp(reduced_h, reduced_g, np.ones((1, 1)),
                               np.array([t_lower]), np.array([t_upper]))
    command = anchor + (null @ t).reshape(2)
    np.testing.assert_allclose(command, [.2, .4], atol=1e-9, rtol=0)
    if abs(command.sum() - target) > 1e-12:
        raise AssertionError("Lower-priority solve changed the high-priority optimum")
    return dict(weighted=weighted, hierarchical=dict(command=command.tolist(),
                high_residual=float(command.sum() - target),
                low_residual=float(command[0] - command[1]), reduced_kkt=residuals))


def unreachable_priority():
    """Distinguish an unreachable soft task from inconsistent hard constraints.

    Here the optimal sum is at a box vertex, leaving no freedom for level two.
    The analytical interval argument is specific to this scalar sum task.
    """
    lower, upper = np.array([-.2, -.8]), np.array([.2, .8])
    cases = []
    for target in (1.2, -1.2):
        achieved = float(np.clip(target, lower.sum(), upper.sum()))
        t_lower = max(lower[0], achieved - upper[1])
        t_upper = min(upper[0], achieved - lower[1])
        if not np.isclose(t_lower, t_upper, atol=1e-12, rtol=0):
            raise AssertionError("This vertex example should have no remaining freedom")
        command = np.array([t_lower, achieved - t_lower])
        np.testing.assert_allclose(command, np.sign(target) * upper, atol=1e-12, rtol=0)
        np.testing.assert_allclose(abs(command.sum() - target), .2, atol=1e-12, rtol=0)
        # The same target imposed as an exact equality makes the feasible set empty.
        a = np.vstack((np.eye(2), np.ones((1, 2))))
        try:
            solve_tiny_qp(np.eye(2), np.zeros(2), a,
                          np.r_[lower, target], np.r_[upper, target])
        except ValueError:
            rejected = True
        else:
            raise AssertionError("An impossible hard sum target was accepted")
        cases.append(dict(target=target, achieved_sum=achieved, command=command.tolist(),
                          high_residual=float(command.sum() - target),
                          low_residual=float(command[0] - command[1]),
                          remaining_interval=[t_lower, t_upper], hard_target_rejected=rejected))
    return cases


def rollout():
    """Replan from the measured integrator state; apply only the first velocity."""
    q, v, goal, rows = 0., 0., .3, []
    for step in range(100):
        plan = mpc(goal, q, v)
        command = plan["velocity"][0]
        next_q = q + .1 * command
        if abs(command) > 1. + 1e-9 or abs(command - v) > .2 + 1e-9 or abs(next_q) > .5 + 1e-9:
            raise AssertionError("Executed step violates a hard bound")
        rows.append(dict(step=step + 1, time=.1 * (step + 1), position=next_q,
                         velocity=command, absolute_error=abs(goal - next_q)))
        q, v = next_q, command
    if rows[-1]["absolute_error"] > 1e-5:
        raise AssertionError("Receding-horizon example did not converge")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-atomic"))
    args = parser.parse_args()
    # Include a deliberately infeasible box; never turn solver failure into a command.
    try:
        solve_tiny_qp(np.eye(1), np.zeros(1), np.ones((1, 1)), np.ones(1), np.zeros(1))
    except ValueError:
        rejected = True
    else:
        raise AssertionError("Infeasible constraints were accepted")
    cases = [mpc(g) for g in (.3, -.3, 0.)]
    for case in cases:
        expected = np.sign(case["goal"]) * np.array([.2, .4])
        np.testing.assert_allclose(case["velocity"], expected, atol=1e-9, rtol=0)
    rows = rollout()
    report = dict(numpy=np.__version__, mpc=cases, wbc=wbc(), coupled_wbc=coupled_wbc(),
                  priority_wbc=priority_wbc(), unreachable_priority=unreachable_priority(),
                  infeasible_rejected=rejected,
                  rollout=dict(steps=len(rows), final=rows[-1]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rollout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    text = json.dumps(report, indent=2) + "\n"
    (args.output / "report.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
