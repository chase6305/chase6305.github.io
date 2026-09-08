"""Compare feedback with replay under a synthetic position-state offset.

Keep atomic_control.py beside this file. Requires NumPy only.
The offset is a model experiment, not a physical impulse or contact simulation.
"""
import argparse
import csv
import json
from pathlib import Path

from atomic_control import mpc, rollout


def run(replan, nominal):
    q, v, rows = 0., 0., []
    for step in range(100):
        offset = -.08 if step == 30 else 0.
        q += offset  # Applied before this tick's measurement and planning.
        measured_q = q
        command = mpc(.3, q, v)["velocity"][0] if replan else nominal[step]["velocity"]
        next_q = q + .1 * command
        violation = max(0., abs(command) - 1., abs(command - v) - .2, abs(next_q) - .5)
        if violation > 1e-9:
            raise AssertionError("Executed step violates an integrator bound")
        rows.append(dict(step=step + 1, time=.1 * (step + 1), offset=offset,
                         measured_position=measured_q, position=next_q, velocity=command,
                         absolute_error=abs(.3 - next_q)))
        q, v = next_q, command
    return rows


def horizon_trap():
    first = mpc(.5, q0=.4, v0=.8)
    q, v = first["positions"][1], first["velocity"][0]
    try:
        mpc(.5, q0=q, v0=v)
    except ValueError:
        return dict(first_plan=first, next_position=q, next_velocity=v,
                    next_solve_rejected=True)
    raise AssertionError("The short-horizon trap should have no feasible continuation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-feedback"))
    args = parser.parse_args()
    nominal = rollout()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope="single-joint integrator; position offset; no WBC or physical impulse",
                  offset_time=3., offset_rad=-.08, cases={}, horizon_trap=horizon_trap())
    for name, replan in (("replay", False), ("feedback", True)):
        rows = run(replan, nominal)
        expected = 0. if replan else .08
        if abs(rows[-1]["absolute_error"] - expected) > 1e-5:
            raise AssertionError("Unexpected final tracking error")
        report["cases"][name] = dict(first_after_offset=rows[30], final=rows[-1])
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    text = json.dumps(report, indent=2) + "\n"
    (args.output / "report.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
