"""Deterministic PID experiments: exact ZOH plant, output saturation, integral.

Download with pid_controller.py. Requires NumPy, SciPy and Matplotlib.
Run: python -B pid_experiments.py --output-dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

from pid_controller import PID


def simulate(kind, scenario, dt=0.002):
    if kind not in {"PD", "PID", "PID without anti-windup"}:
        raise ValueError("unknown controller")
    if scenario not in {"constant", "saturation"} or not np.isfinite(dt) or dt <= 0:
        raise ValueError("invalid scenario or sample interval")
    duration = 20.0 if scenario == "constant" else 16.0
    count = round(duration / dt)
    if count < 1 or not np.isclose(count * dt, duration):
        raise ValueError("duration must be an integer number of intervals")
    # m*q'' + b*q' + k*q = u + d, in SI units; m=1, b=2, k=4, d=-1.
    ad, bd, _, _, _ = cont2discrete(
        (np.array([[0., 1.], [-4., -2.]]), np.array([[0.], [1.]]),
         np.eye(2), np.zeros((2, 1))), dt, method="zoh")
    controller = PID(16., 0. if kind == "PD" else 8., 6., 8., tau=.02,
                     anti_windup=kind != "PID without anti-windup")
    t = np.arange(count + 1) * dt
    reference = np.ones_like(t) if scenario == "constant" else np.where(t < 3., 3., .5)
    state = np.zeros(2)
    rows = []
    for i, ti in enumerate(t):
        output = controller.update(float(reference[i]), float(state[0]), dt)
        rows.append([state[0], state[1], output, controller.last_raw, controller.integral])
        if i < count:
            state = ad @ state + bd[:, 0] * (output - 1.)
    values = np.asarray(rows)
    assert np.isfinite(values).all() and np.max(abs(values[:, 2])) <= 8.
    return {"t": t, "reference": reference, "q": values[:, 0], "dq": values[:, 1],
            "u": values[:, 2], "raw": values[:, 3], "integral": values[:, 4]}


def metrics(data, switch_time=None):
    t, error = data["t"], data["reference"] - data["q"]
    result = {"final_position_m": float(data["q"][-1]),
              "final_error_m": float(error[-1]),
              "max_abs_integral_N": float(abs(data["integral"]).max()),
              "fraction_saturated_samples": float(np.mean(abs(data["raw"]) > 8.))}
    if switch_time is not None:
        # First time after the last >2 cm error, measured over this finite record.
        bad = np.flatnonzero((t >= switch_time) & (abs(error) > .02))
        next_index = int(bad[-1] + 1) if len(bad) else int(np.searchsorted(t, switch_time))
        result["recorded_2cm_settling_after_switch_s"] = (
            float(t[next_index] - switch_time) if next_index < len(t) else None)
    return result


def checks():
    controller = PID(2., 1., .1, 3.)
    for _ in range(1000):
        assert controller.update(100., 0., .01) == 3.
    assert controller.integral == 0.
    snapshot = vars(controller).copy()
    for sample in [(1., 0., 0.), (float("nan"), 0., .01), (1e308, -1e308, .01)]:
        try:
            controller.update(*sample)
        except ValueError:
            assert vars(controller) == snapshot
        else:
            raise AssertionError("invalid sample accepted")
    derivative = PID(0., 0., 2., 100.)
    derivative.update(0., .5, .01)
    assert derivative.update(100., .5, .01) == 0.  # no derivative kick from reference
    derivative.reset(measured=.5, integral_output=1.)
    assert derivative.update(.5, .5, .01) == 1.
    return {"held_saturation": "passed", "invalid_sample_state_rollback": "passed",
            "reference_derivative_kick": "absent", "reset": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = {"constant": {kind: simulate(kind, "constant") for kind in ("PD", "PID")},
               "saturation": {kind: simulate(kind, "saturation")
                              for kind in ("PID without anti-windup", "PID")}}
    # Independent equilibrium for the PD case: q=(Kp*r+d)/(k+Kp)=15/20.
    assert abs(results["constant"]["PD"]["q"][-1] - .75) < 1e-6
    assert abs(results["constant"]["PID"]["q"][-1] - 1.) < .001
    refined = simulate("PID", "saturation", dt=.001)
    coarse = results["saturation"]["PID"]
    discretization_difference = float(abs(coarse["q"] - refined["q"][::2]).max())
    assert discretization_difference < .01
    report = {"plant": {"mass_kg": 1., "damping_Ns_per_m": 2., "spring_N_per_m": 4.,
                         "disturbance_N": -1., "force_limit_N": 8.},
              "controller": {"kp_N_per_m": 16., "ki_N_per_m_s": 8.,
                             "kd_Ns_per_m": 6., "derivative_tau_s": .02},
              "sample_interval_s": .002,
              "maximum_position_difference_after_halving_dt_m": discretization_difference,
              "checks": checks(),
              "metrics": {scene: {kind: metrics(data, 3. if scene == "saturation" else None)
                                    for kind, data in kinds.items()}
                          for scene, kinds in results.items()},
              "scope": "Synthetic exact-ZOH plant; no sensor noise, hardware or latency model."}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"PD": "#4777b6", "PID": "#2e9878", "PID without anti-windup": "#dc8d42"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")
    for column, (scene, kinds) in enumerate(results.items()):
        for kind, data in kinds.items():
            for row, key in enumerate(("q", "u", "integral")):
                axes[row, column].plot(data["t"], data[key], label=kind, color=colors[kind], lw=1.7)
        data = next(iter(kinds.values()))
        axes[0, column].plot(data["t"], data["reference"], "k--", label="Reference", lw=1.2)
        for limit in (-8, 8):
            axes[1, column].axhline(limit, color="#8c93a0", ls=":", lw=1)
        axes[0, column].set_title("Constant load: PD vs PID" if scene == "constant"
                                  else "Unreachable reference, then return at 3 s")
        axes[0, column].legend(fontsize=8)
        for row, label in enumerate(("Position [m]", "Applied force [N]", "Integral contribution [N]")):
            axes[row, column].set_ylabel(label)
            axes[row, column].grid(alpha=.22)
        axes[2, column].set_xlabel("Time [s]")
    fig.suptitle("Same plant and gains; output limited to [-8, 8] N", fontsize=14)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "pid-saturation-study.png", dpi=150)
    plt.close(fig)
    (args.output_dir / "pid-saturation-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
