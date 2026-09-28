"""Audit IPOPT candidates: constrained stationarity, infeasibility, local minima.

Run: python -B solver_audit.py --output-dir results
Requires CasADi 3.7.2, NumPy, Matplotlib. Scalar synthetic problems only.
"""
import argparse
import json
from pathlib import Path

import casadi as ca
import numpy as np


OPTIONS = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
           "ipopt.tol": 1e-10, "ipopt.max_iter": 100,
           "ipopt.bound_relax_factor": 0., "error_on_fail": False}


def bounded_case(lower):
    x = ca.SX.sym("x")
    objective, constraint = (x - 2)**2, x*x
    solver = ca.nlpsol("bounded", "ipopt", {"x": x, "f": objective, "g": constraint}, OPTIONS)
    solution = solver(x0=max(0, lower), lbx=lower, ubx=3, lbg=-ca.inf, ubg=1)
    value, cost, g, lx, lg = [float(solution[key]) for key in ("x", "f", "g", "lam_x", "lam_g")]
    stationarity = 2*(value - 2) + 2*value*lg + lx
    # CasADi signed multipliers: lower-bound <=0, upper-bound >=0.
    lower_multiplier, upper_multiplier = max(-lx, 0), max(lx, 0)
    primal = max(0., lower - value, value - 3, g - 1)
    complementarity = max(abs(lg * (1 - g)),
                           abs(lower_multiplier * (value - lower)),
                           abs(upper_multiplier * (3 - value)))
    accepted = bool(solver.stats()["success"] and np.isfinite(
        [value, cost, g, lx, lg]).all() and primal < 1e-7
        and abs(stationarity) < 1e-7 and lg >= -1e-7 and complementarity < 1e-7)
    return {"variable_lower_bound": lower, "x": value, "objective": cost,
            "solver_success": bool(solver.stats()["success"]),
            "return_status": solver.stats()["return_status"],
            "objective_gradient": 2*(value - 2), "lam_g": lg, "lam_x": lx,
            "primal_violation": primal, "stationarity_abs": abs(stationarity),
            "complementarity_abs": complementarity, "accepted": accepted}


def nonconvex_cases():
    x = ca.SX.sym("x")
    objective = (x*x - 1)**2 + .2*x
    solver = ca.nlpsol("two_wells", "ipopt", {"x": x, "f": objective}, OPTIONS)
    rows = []
    # Independently enumerate the stationary points of this scalar polynomial.
    stationary = np.roots([4., 0., -4., .2])
    candidates = [-2., 2., *[float(r.real) for r in stationary if abs(r.imag) < 1e-12 and -2 < r.real < 2]]
    global_cost = min((v*v - 1)**2 + .2*v for v in candidates)
    for initial in (-1.5, 1.5):
        result = solver(x0=initial, lbx=-2, ubx=2)
        value, cost = float(result["x"]), float(result["f"])
        assert solver.stats()["success"]
        assert abs(4*value*(value*value - 1) + .2 + float(result["lam_x"])) < 1e-7
        assert 12*value*value - 4 > 0  # both are strict local minima
        rows.append({"initial_x": initial, "x": value, "objective": cost,
                     "return_status": solver.stats()["return_status"],
                     "gap_to_enumerated_global_minimum": cost - global_cost})
    assert rows[1]["objective"] - rows[0]["objective"] > .3
    assert abs(rows[0]["gap_to_enumerated_global_minimum"]) < 1e-12
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    feasible, infeasible = bounded_case(-2.), bounded_case(2.)
    assert feasible["accepted"] and abs(feasible["x"] - 1) < 1e-8
    assert abs(feasible["objective_gradient"]) > 1.9
    assert not infeasible["accepted"] and not infeasible["solver_success"]
    assert infeasible["primal_violation"] > 2.9
    local = nonconvex_cases()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    xx = np.linspace(-2, 3, 600)
    axes[0].plot(xx, (xx - 2)**2, color="#397bb2")
    axes[0].axvspan(-1, 1, color="#dcecdf", alpha=.8)
    axes[0].scatter([1], [1], color="#397c47", zorder=4)
    axes[0].annotate("Constrained minimum\nx = 1, gradient = -2", xy=(1, 1),
                     xytext=(-1.8, 6), arrowprops={"arrowstyle": "->"})
    axes[0].set(title="1. A nonzero objective gradient", xlabel="x", ylabel="Objective", ylim=(-.5, 11))
    axes[1].plot(xx, (xx - 2)**2, color="#397bb2")
    axes[1].axvspan(-1, 1, color="#dcecdf", alpha=.8, label="x squared <= 1")
    axes[1].axvspan(2, 3, color="#f7e0c6", alpha=.8, label="x >= 2")
    axes[1].scatter([infeasible["x"]], [infeasible["objective"]], color="#c45548", marker="x", s=70, zorder=4)
    axes[1].set(title="2. Low cost, infeasible candidate", xlabel="x", ylim=(-.5, 11))
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].text(.2, 4, "No overlap", ha="center")
    xx = np.linspace(-1.6, 1.6, 600)
    axes[2].plot(xx, (xx*xx - 1)**2 + .2*xx, color="#7965aa")
    for row, color in zip(local, ("#397c47", "#d87525")):
        axes[2].scatter([row["x"]], [row["objective"]], color=color, zorder=4)
        axes[2].annotate(f"f = {row['objective']:.4f}", (row["x"], row["objective"]),
                         xytext=(0, 15), textcoords="offset points", ha="center", color=color)
    axes[2].set(title="3. Two successful local solves", xlabel="x", ylim=(-.5, 2))
    for ax in axes:
        ax.grid(alpha=.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "solver-candidates.png", dpi=160)
    plt.close(fig)
    report = {"casadi_version": ca.__version__, "feasible": feasible,
              "infeasible": infeasible, "nonconvex": local,
              "scope": "dimensionless scalar examples, no robot or performance benchmark"}
    (args.output_dir / "solver-audit-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
