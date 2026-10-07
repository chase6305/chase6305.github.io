"""Read-only WholeBodyX integration experiment; fixed-base kinematics, no hardware.

Install a local WholeBodyX checkout first. Run with --output results-control.
Both cases use identical posture tasks and bounds; only the reference differs.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import scipy
import osqp
from wholebodyx import JointMPC, KinematicWBC, MPCReferenceManager, RobotState
from wholebodyx.model import PlanarDualArm
from wholebodyx.simulation import JointIntegrator
from wholebodyx.tasks import PostureTask


def run(with_mpc, steps=250):
    model = PlanarDualArm()
    q0 = np.array([.2, .6, -.4, -.2, -.6, .4])
    goal = np.array([.5, .4, -.3, -.5, -.4, .3])
    dt = .02
    plant = JointIntegrator(model.limits, RobotState(q0, np.zeros(6)))
    controller = KinematicWBC(model.limits)
    manager = MPCReferenceManager(JointMPC(model.limits), model.limits) if with_mpc else None
    reasons, rows = Counter(), []
    for step in range(steps):
        state = plant.read()
        qref, vref, reason = goal, np.zeros(6), "direct"
        if manager is not None:
            reference = manager.update(state, goal, dt)
            if not reference.success:
                raise RuntimeError(f"MPC reference rejected: {reference.solver}")
            qref, vref, reason = reference.position, reference.velocity, reference.reason
        reasons[reason] += 1
        task = PostureTask(qref, feedforward=vref).linearize(model, state)
        result = controller.step(state, [task], dt)
        if result.command is None:
            raise RuntimeError(f"WBC failed: {result.solver}")
        plant.write(result.command, dt)
        actual = plant.read()
        lo, hi = model.limits.velocity_bounds(state, dt)
        violation = max(0., float(np.max(lo - actual.v)), float(np.max(actual.v - hi)))
        if violation > 1e-6:
            raise AssertionError(f"Executed velocity violates bounds: {violation}")
        if not np.allclose(actual.q, state.q + dt * actual.v, atol=1e-10, rtol=0):
            raise AssertionError("Executed state violates integrator equation")
        rows.append(dict(step=step + 1, time=actual.stamp, replan_reason=reason,
                         joint_error_rad=float(np.linalg.norm(actual.q - goal)),
                         max_velocity_bound_violation=violation,
                         solver_violation=float(result.solver.violation)))
    if rows[-1]["joint_error_rad"] > 1e-4:
        raise AssertionError("Tracking did not converge within the test horizon")
    return rows, dict(initial_joint_error_rad=float(np.linalg.norm(q0-goal)),
                     final=rows[-1], replan_reasons=dict(reasons),
                     max_velocity_bound_violation=max(r["max_velocity_bound_violation"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-control"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="six-joint fixed-base integrator; posture tasks only; no learned policy",
                  dt=.02, steps=250, numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__)
    for name, enabled in (("wbc", False), ("mpc-wbc", True)):
        rows, report[name] = run(enabled)
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
