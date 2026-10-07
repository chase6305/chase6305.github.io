"""Ruckig 0.19.4: viability, moving terminal states and synchronization.

Python 3.10+, ruckig, NumPy, Matplotlib. Offline only; no waypoints/cloud API.
Run: python -B ruckig_boundary_checks.py --output-dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import ruckig


def parameters(target):
    n = len(target)
    inp = ruckig.InputParameter(n)
    inp.current_position = inp.current_velocity = inp.current_acceleration = [0.] * n
    inp.target_position = list(target)
    inp.target_velocity = inp.target_acceleration = [0.] * n
    inp.max_velocity = inp.max_acceleration = inp.max_jerk = [1.] * n
    return inp


def viability():
    inp = parameters([2.])
    inp.current_velocity = [.99]
    inp.current_acceleration = [.5]
    otg = ruckig.Ruckig(1, .01)
    # Even maximum negative jerk needs a/j seconds to remove positive acceleration.
    unavoidable_peak = .99 + .5**2 / (2 * 1.)
    assert unavoidable_peak > 1.
    try:
        otg.validate_input(inp, True, True)
    except ruckig.RuckigError as error:
        rejected = str(error).strip()
    else:
        raise AssertionError("invalid approaching-limit state was accepted")
    inp.current_acceleration = [-.5]
    assert otg.validate_input(inp, True, True)  # now decelerating: direction matters
    return {"unavoidable_peak_velocity_rad_s": unavoidable_peak,
            "approaching_limit_rejected": rejected, "opposite_acceleration_valid": True}


def moving_terminal(discrete):
    dt = .01
    inp = parameters([1.])
    inp.target_velocity = [.3]
    inp.duration_discretization = (ruckig.DurationDiscretization.Discrete if discrete
                                   else ruckig.DurationDiscretization.Continuous)
    otg, out = ruckig.Ruckig(1, dt), ruckig.OutputParameter(1)
    assert otg.validate_input(inp, True, True)
    for tick in range(1, 10001):
        result = otg.update(inp, out)
        if result not in (ruckig.Result.Working, ruckig.Result.Finished):
            raise RuntimeError(f"trajectory update failed: {result}")
        if result == ruckig.Result.Finished:
            break
        out.pass_to_input(inp)
    else:
        raise AssertionError("update budget exceeded")
    duration = float(out.trajectory.duration)
    exact = np.asarray(out.trajectory.at_time(duration))[:, 0]
    np.testing.assert_allclose(exact, [1., .3, 0.], atol=1e-10)
    late = float(out.time) - duration
    assert -1e-12 <= late <= dt + 1e-10
    np.testing.assert_allclose(out.new_position[0], 1. + .3 * late, atol=1e-10)
    if discrete:
        assert abs(duration / dt - round(duration / dt)) < 1e-10
    return {"duration_s": duration, "first_finished_tick": tick,
            "first_finished_time_s": float(out.time),
            "position_at_first_finished_rad": out.new_position[0],
            "velocity_at_first_finished_rad_s": out.new_velocity[0],
            "exact_terminal_state": exact.tolist()}


def synchronized(sync):
    inp = parameters([1., .4])
    inp.synchronization = sync
    otg = ruckig.Ruckig(2, .01)
    trajectory = ruckig.Trajectory(2)
    assert otg.validate_input(inp, True, True)
    result = otg.calculate(inp, trajectory)
    if result not in (ruckig.Result.Working, ruckig.Result.Finished):
        raise RuntimeError(f"calculation failed: {result}")
    t = np.linspace(0., trajectory.duration, 1001)
    states = np.asarray([trajectory.at_time(float(ti)) for ti in t])
    q, dq, ddq = states[:, 0], states[:, 1], states[:, 2]
    np.testing.assert_allclose(q[[0, -1]], [[0., 0.], [1., .4]], atol=1e-10)
    assert np.max(abs(dq)) <= 1. + 1e-10 and np.max(abs(ddq)) <= 1. + 1e-10
    xy = np.column_stack((np.cos(q[:, 0]) + .6 * np.cos(q.sum(axis=1)),
                          np.sin(q[:, 0]) + .6 * np.sin(q.sum(axis=1))))
    chord = xy[-1] - xy[0]
    difference = xy - xy[0]
    distance = abs(chord[0] * difference[:, 1] - chord[1] * difference[:, 0]) / np.linalg.norm(chord)
    return t, q, xy, {"duration_s": float(trajectory.duration),
                     "max_joint_line_error_rad": float(abs(q[:, 1] - .4 * q[:, 0]).max()),
                     "max_tcp_distance_from_endpoint_line_m": float(distance.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    time = synchronized(ruckig.Synchronization.Time)
    phase = synchronized(ruckig.Synchronization.Phase)
    assert time[3]["max_joint_line_error_rad"] > .01  # counterexample, not all Time profiles
    assert phase[3]["max_joint_line_error_rad"] < 1e-12
    assert phase[3]["max_tcp_distance_from_endpoint_line_m"] > .05
    report = {"ruckig": ruckig.__version__, "viability": viability(),
              "moving_terminal_continuous": moving_terminal(False),
              "moving_terminal_discrete": moving_terminal(True),
              "time_synchronization": time[3], "phase_synchronization": phase[3],
              "scope": "Two planar revolute joints; links 1 and 0.6 m; no collision, torque or hardware test."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for data, label, color, style in [(time, "Time synchronization", "#dc8d42", "-"),
                                       (phase, "Phase synchronization", "#4777b6", "--")]:
        _, q, xy, _ = data
        axes[0].plot(q[:, 0], q[:, 1], style, label=label, color=color, lw=2)
        axes[1].plot(xy[:, 0], xy[:, 1], style, label=label, color=color, lw=2)
    axes[1].plot(phase[2][[0, -1], 0], phase[2][[0, -1], 1], ":", color="#687387", label="TCP endpoint chord")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set(xlabel="Joint 1 [rad]", ylabel="Joint 2 [rad]", title="Planning coordinates")
    axes[1].set(xlabel="TCP x [m]", ylabel="TCP y [m]", title="Same motion after nonlinear FK")
    axes[1].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "ruckig-synchronization.png", dpi=160)
    plt.close(fig)
    (args.output_dir / "ruckig-boundary-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
