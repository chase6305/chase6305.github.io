"""A stable measurement filter can destabilize a sampled feedback loop.

Python 3.10+, numpy, scipy, matplotlib. Offline linear model, no robot I/O.
Run: python -B feedback_filter.py --output-dir feedback-results
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

DT = 0.005
KP, KD = 80.0, 4.0
MASS, DAMPING, STIFFNESS = 1.0, 2.0, 4.0


def plant():
    a = np.array([[0.0, 1.0], [-STIFFNESS / MASS, -DAMPING / MASS]])
    b = np.array([[0.0], [1.0 / MASS]])
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), DT)
    return ad, bd[:, 0]


def alpha(frequency):
    return 1.0 if frequency is None else -np.expm1(-2 * np.pi * frequency * DT)


def feedback_matrix(frequency):
    """State BEFORE filtering: [q_k, v_k, filtered_q_(k-1)]."""
    ad, bd = plant()
    gain = alpha(frequency)
    filtered = np.array([gain, 0.0, 1.0 - gain])
    derivative = (filtered - np.array([0.0, 0.0, 1.0])) / DT
    command = -KP * filtered - KD * derivative  # reference r = 0
    return np.vstack([np.column_stack([ad, np.zeros(2)])
                      + np.outer(bd, command), filtered])


def reference_matrix(frequency):
    """The same filter on the reference; raw q and its difference are fed back."""
    _, bd = plant()
    result = np.zeros((4, 4))
    result[:3, :3] = feedback_matrix(None)
    result[:2, 3] = bd * KP * (1.0 - alpha(frequency))
    result[3, 3] = 1.0 - alpha(frequency)
    return result


def radius(matrix):
    return float(abs(np.linalg.eigvals(matrix)).max())


def simulate(frequency, duration=4.0):
    ad, bd = plant()
    gain = alpha(frequency)
    # The filter starts at the measured position: there is no startup D kick.
    state = np.array([0.001, 0.0, 0.001])
    matrix = feedback_matrix(frequency)
    times = np.arange(round(duration / DT) + 1) * DT
    position = [state[0]]
    for _ in times[1:]:
        q, _, previous = state
        filtered = previous + gain * (q - previous)
        command = -KP * filtered - KD * (filtered - previous) / DT
        physical = ad @ state[:2] + bd * command
        updated = np.r_[physical, filtered]
        # Compare the direct control/plant loop with the derived augmented system.
        np.testing.assert_allclose(updated, matrix @ state, atol=1e-13, rtol=1e-12)
        state = updated
        position.append(state[0])
    return times, np.asarray(position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("feedback-results"))
    args = parser.parse_args()
    frequencies = [None, 10.0, 2.0, 1.0]
    rows, traces = [], []
    for frequency in frequencies:
        matrix = feedback_matrix(frequency)
        poles = np.linalg.eigvals(matrix)
        gain = alpha(frequency)
        times, position = simulate(frequency)
        rows.append({"measurement_filter_hz": frequency,
                     "alpha": float(gain),
                     "open_loop_white_noise_std_ratio": float(np.sqrt(gain / (2 - gain))),
                     "closed_loop_poles": [[float(p.real), float(p.imag)] for p in poles],
                     "closed_loop_spectral_radius": radius(matrix),
                     "max_abs_position_m_in_4s": float(abs(position).max())})
        traces.append((times, position))
    assert rows[0]["closed_loop_spectral_radius"] < 0.99
    assert rows[1]["closed_loop_spectral_radius"] < 0.99
    assert 0.999 < rows[2]["closed_loop_spectral_radius"] < 1
    assert rows[3]["closed_loop_spectral_radius"] > 1.003
    assert rows[3]["max_abs_position_m_in_4s"] > 0.01
    # Reference filtering gives a block-triangular cascade, not the same loop.
    grid = np.geomspace(0.4, 50.0, 160)
    reference_radii = [radius(reference_matrix(fc)) for fc in grid]
    expected = np.maximum(radius(feedback_matrix(None)), 1 - alpha(grid))
    np.testing.assert_allclose(reference_radii, expected, atol=1e-12)
    assert max(reference_radii) < 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    colors = ["#616e7c", "#3b78ae", "#8164a5", "#cf772c"]
    labels = ["No measurement LPF", "Feedback LPF: 10 Hz",
              "Feedback LPF: 2 Hz", "Feedback LPF: 1 Hz"]
    for (times, position), color, label in zip(traces, colors, labels):
        axes[0].plot(times, 1000 * position, color=color, label=label, linewidth=1.7)
    axes[0].set(xlabel="Time [s]", ylabel="Position [mm]",
                title="Release from 1 mm; zero reference, no noise")
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[1].semilogx(grid, [radius(feedback_matrix(fc)) for fc in grid],
                     color="#cf772c", label="Filter in feedback")
    axes[1].semilogx(grid, reference_radii, color="#3b78ae", linestyle="--",
                     label="Filter on reference")
    axes[1].axhline(1.0, color="#616e7c", linestyle=":", label="Unit-circle boundary")
    axes[1].scatter([10, 2, 1], [row["closed_loop_spectral_radius"] for row in rows[1:]],
                    c=colors[1:], zorder=3)
    axes[1].set(xlabel="LPF frequency parameter [Hz]", ylabel="Largest pole magnitude",
                title="Identical filter; different loop dynamics")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "feedback-filter-stability.png", dpi=160)
    plt.close(fig)
    result = {"versions": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
              "sample_period_s": DT, "mass_kg": MASS, "damping_Ns_m": DAMPING,
              "stiffness_N_m": STIFFNESS, "kp_N_m": KP, "kd_Ns_m": KD,
              "plant_discretization": "zero-order hold, current command applied over next interval",
              "derivative": "backward difference of the current filtered measurement",
              "initial_position_m": 0.001, "initial_velocity_m_s": 0.0,
              "initial_filter_position_m": 0.001,
              "limitations": "linear model; no saturation, extra delay, noise or real robot",
              "reference_filter_1hz_spectral_radius": radius(reference_matrix(1.0)),
              "results": rows}
    (args.output_dir / "feedback-filter-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
