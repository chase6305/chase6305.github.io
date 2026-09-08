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


def solve_tiny_qp(h, g, a, lower, upper):
    """Solve strictly convex min .5*x'Hx + g'x, lower <= A*x <= upper.

    Enumerate independent active subsets and require primal feasibility,
    nonnegative multipliers and stationarity (the convex KKT certificate).
    Only intended for finite bounds and at most two decision variables.
    """
    n = len(g)
    if n > 2 or not all(np.isfinite(v).all() for v in (h, g, a, lower, upper)):
        raise ValueError("This example requires finite data and at most two variables")
    if not np.allclose(h, h.T) or np.linalg.eigvalsh(h).min() <= 0:
        raise ValueError("H must be symmetric positive definite")
    c, d = np.vstack((a, -a)), np.concatenate((upper, -lower))
    for count in range(n + 1):
        for active in combinations(range(len(d)), count):
            ca = c[list(active)]
            system = np.block([[h, ca.T], [ca, np.zeros((count, count))]])
            try:
                solution = np.linalg.solve(system, np.r_[-g, d[list(active)]])
            except np.linalg.LinAlgError:
                continue
            x, multipliers = solution[:n], solution[n:]
            primal = max(0., float(np.max(c @ x - d)))
            dual = max(0., float(np.max(-multipliers))) if count else 0.
            stationarity = float(np.max(np.abs(h @ x + g + ca.T @ multipliers)))
            if max(primal, dual, stationarity) <= 1e-9:
                return x, dict(primal=primal, dual=dual, stationarity=stationarity)
    raise ValueError("No KKT candidate found; infeasible or numerically unresolved")


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
    report = dict(numpy=np.__version__, mpc=cases, wbc=wbc(), coupled_wbc=coupled_wbc(), infeasible_rejected=rejected,
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
