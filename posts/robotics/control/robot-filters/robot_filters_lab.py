"""Reproducible synthetic experiments for the robotics filtering article.

Python 3.10+, NumPy, SciPy, Matplotlib. No robot, network or model required.
Run: python -B robot_filters_lab.py --output-dir assets
The examples illustrate filter behavior; they are not real-time drivers.
"""
from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy import linalg, signal
from scipy.spatial.transform import Rotation


def alpha(cutoff_hz, dt):
    """Backward-Euler low-pass coefficient used throughout the One Euro demo."""
    if not np.isfinite([cutoff_hz, dt]).all() or min(cutoff_hz, dt) <= 0:
        raise ValueError("cutoff_hz and dt must be finite and positive")
    return dt / (dt + 1.0 / (2.0 * np.pi * cutoff_hz))


class WeightedAverage:
    """weights[0] multiplies the newest sample; partial windows renormalize.

    Unlike the cited Unitree example, equal-valued samples are retained.
    """
    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float)
        if (self.w.ndim != 1 or self.w.size == 0
                or not np.isfinite(self.w).all() or (self.w < 0).any()
                or self.w[0] <= 0):
            raise ValueError("finite nonnegative weights, with w[0] > 0, required")
        self.w = self.w / self.w.sum()
        self.q = deque(maxlen=len(self.w))

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("nonfinite value")
        self.q.appendleft(float(value))
        w = self.w[:len(self.q)]
        return float(np.dot(w, self.q) / w.sum())


class OneEuro:
    """Scalar variant using innovation against the previous filtered value.

    Timestamps must increase strictly. Gaps/resets belong to the caller.
    """
    def __init__(self, min_cutoff=1.0, beta=4.0, derivative_cutoff=1.0):
        if (not np.isfinite([min_cutoff, beta, derivative_cutoff]).all()
                or min(min_cutoff, derivative_cutoff) <= 0 or beta < 0):
            raise ValueError("invalid One Euro parameters")
        self.minimum, self.beta, self.dc = min_cutoff, beta, derivative_cutoff
        self.t, self.y, self.d = None, None, 0.0

    def update(self, timestamp, value):
        if not np.isfinite([timestamp, value]).all():
            raise ValueError("timestamp/value must be finite")
        if self.t is None:
            self.t, self.y = float(timestamp), float(value)
            return self.y
        dt = timestamp - self.t
        ad = alpha(self.dc, dt)  # validates time before mutating state
        derivative = (value - self.y) / dt
        next_d = ad * derivative + (1 - ad) * self.d
        a = alpha(self.minimum + self.beta * abs(next_d), dt)
        self.y = a * value + (1 - a) * self.y
        self.t, self.d = float(timestamp), next_d
        return float(self.y)


class RotationOneEuro:
    """A specified pose-filter extension, not a universal One Euro standard.

    Input/output: unit quaternions in xyzw order. Uses the magnitude of the
    innovation relative to the last filtered orientation, in rad/s; a scalar
    low-pass adapts the gain. Geometry is handled by SO(3) Log/Exp.
    """
    def __init__(self, min_cutoff=1.0, beta=0.3, derivative_cutoff=1.0):
        OneEuro(min_cutoff, beta, derivative_cutoff)  # parameter validation
        self.minimum, self.beta, self.dc = min_cutoff, beta, derivative_cutoff
        self.t, self.r, self.speed = None, None, 0.0

    def update(self, timestamp, xyzw):
        q = np.asarray(xyzw, dtype=float)
        if (q.shape != (4,) or not np.isfinite(q).all()
                or not np.isfinite(timestamp) or np.linalg.norm(q) < 1e-12):
            raise ValueError("finite timestamp and nonzero xyzw quaternion required")
        measured = Rotation.from_quat(q / np.linalg.norm(q))
        if self.t is None:
            self.t, self.r = float(timestamp), measured
            return self.r.as_quat()
        dt = timestamp - self.t
        ad = alpha(self.dc, dt)
        innovation = (self.r.inv() * measured).as_rotvec()
        speed = np.linalg.norm(innovation) / dt
        next_speed = ad * speed + (1 - ad) * self.speed
        a = alpha(self.minimum + self.beta * next_speed, dt)
        self.r = self.r * Rotation.from_rotvec(a * innovation)
        self.t, self.speed = float(timestamp), next_speed
        return self.r.as_quat()


class OnlineButterworth:
    def __init__(self, sample_hz, cutoff_hz, order=2):
        if not 0 < cutoff_hz < sample_hz / 2:
            raise ValueError("cutoff must lie between zero and Nyquist")
        self.sos = signal.butter(order, cutoff_hz, fs=sample_hz, output="sos")
        self.state = None

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("nonfinite measurement")
        if self.state is None:
            self.state = signal.sosfilt_zi(self.sos) * value
        y, self.state = signal.sosfilt(self.sos, [value], zi=self.state)
        return float(y[0])


def butterworth2_coefficients(sample_hz, cutoff_hz):
    """Normalized b, a with a[0]=1; feedback uses -a[1], -a[2]."""
    if (not np.isfinite([sample_hz, cutoff_hz]).all()
            or not 0 < cutoff_hz < sample_hz/2):
        raise ValueError("finite positive sample rate and cutoff below Nyquist required")
    c = np.tan(np.pi*cutoff_hz/sample_hz)
    denominator = 1 + np.sqrt(2)*c + c*c
    b0 = c*c/denominator
    return (np.array([b0, 2*b0, b0]),
            np.array([1, 2*(c*c-1)/denominator,
                      (1-np.sqrt(2)*c+c*c)/denominator]))


@lru_cache(maxsize=128)
def msd_matrices(natural_hz, damping_ratio, dt):
    """Exact ZOH matrices for y'' + 2*zeta*wn*y' + wn**2*y = wn**2*u."""
    if (not np.isfinite([natural_hz, damping_ratio, dt]).all()
            or min(natural_hz, damping_ratio, dt) <= 0):
        raise ValueError("positive finite dynamics parameters required")
    wn = 2 * np.pi * natural_hz
    augmented = np.array([[0, 1, 0], [-wn**2, -2*damping_ratio*wn, wn**2],
                          [0, 0, 0]], dtype=float)
    transition = linalg.expm(augmented * dt)
    return transition[:2, :2], transition[:2, 2]


def msd_filter(t, target, natural_hz=5.0, damping_ratio=1.0,
               initial_position=None, initial_velocity=0.0):
    """At t[k], advance with target[k-1] held on [t[k-1], t[k])."""
    t, target = np.asarray(t, dtype=float), np.asarray(target, dtype=float)
    if (t.ndim != 1 or target.shape != t.shape or t.size == 0
            or not np.isfinite(t).all() or not np.isfinite(target).all()
            or (np.diff(t) <= 0).any()):
        raise ValueError("finite, strictly increasing t and matching scalar target required")
    position = target[0] if initial_position is None else initial_position
    if not np.isfinite([position, initial_velocity]).all():
        raise ValueError("invalid initial position or velocity")
    state = np.array([position, initial_velocity], dtype=float)
    y = np.empty_like(target)
    y[0] = position
    for k in range(1, len(t)):
        a, b = msd_matrices(natural_hz, damping_ratio, float(t[k] - t[k-1]))
        state = a @ state + b * target[k-1]
        y[k] = state[0]
    return y


def run_scalar(filter_object, values):
    return np.array([filter_object.update(x) for x in values])


def synthetic_signal():
    fs = 200.0
    t = np.arange(0, 10, 1/fs)
    truth = np.zeros_like(t)
    ramp = (t >= 2) & (t < 4)
    truth[ramp] = .2*(t[ramp]-2)
    truth[(t >= 4) & (t < 6)] = .4
    returning = (t >= 6) & (t < 8)
    truth[returning] = .2*(1 + np.cos(np.pi*(t[returning]-6)/2))
    rng = np.random.default_rng(20260925)
    measured = truth + rng.normal(0, .006, len(t)) + .003*np.sin(2*np.pi*20*t)
    return fs, t, truth, measured


def checks():
    """Independent identities and failure cases, not implementation snapshots."""
    passed = []
    y = run_scalar(WeightedAverage([.4, .3, .2, .1]), [1, 2, 3, 4, 5])
    np.testing.assert_allclose(y[-2:], [3, 4])
    passed.append("WMA newest-weight convention and ramp delay")

    sos = signal.butter(4, 5, fs=200, output="sos")
    x = np.random.default_rng(4).normal(size=200)
    zi = signal.sosfilt_zi(sos) * x[0]
    whole, _ = signal.sosfilt(sos, x, zi=zi)
    pieces = []
    for chunk in np.split(x, [13, 81, 129]):
        out, zi = signal.sosfilt(sos, chunk, zi=zi)
        pieces.append(out)
    np.testing.assert_allclose(whole, np.concatenate(pieces), atol=1e-14)
    passed.append("SOS state continuity across arbitrary chunks")

    for fs in [50., 200., 1000.]:
        for fraction in [.001, .01, .1, .4, .49]:
            cutoff = fs*fraction
            b, a = butterworth2_coefficients(fs, cutoff)
            reference_b, reference_a = signal.butter(2, cutoff, fs=fs)
            np.testing.assert_allclose(b, reference_b, atol=1e-14)
            np.testing.assert_allclose(a, reference_a, atol=1e-14)
            _, response = signal.freqz(b, a, worN=[cutoff], fs=fs)
            np.testing.assert_allclose(abs(response), 1/np.sqrt(2), atol=2e-11)
            assert (abs(np.roots(a)) < 1).all()
    passed.append("Butterworth coefficients, prewarped cutoff and stable poles across sample rates")

    t = np.arange(200) / 100.0
    a, b = OneEuro(beta=4), OneEuro(beta=.004)
    meters = np.sin(t) * .1
    ym = np.array([a.update(ti, xi) for ti, xi in zip(t, meters)])
    ymm = np.array([b.update(ti, xi*1000) for ti, xi in zip(t, meters)])
    np.testing.assert_allclose(ym*1000, ymm, atol=1e-11)
    try:
        a.update(t[-1], 0.0)
    except ValueError:
        passed.append("One Euro unit conversion and duplicate timestamp rejection")
    else:
        raise AssertionError("duplicate timestamp accepted")

    rotation = RotationOneEuro()
    q = Rotation.from_euler("z", 179, degrees=True).as_quat()
    r0 = Rotation.from_quat(rotation.update(0.0, q))
    r1 = Rotation.from_quat(rotation.update(.01, -q))
    assert (r0.inv()*r1).magnitude() < 1e-12
    rotation = RotationOneEuro(min_cutoff=1/(2*np.pi*.01), beta=0)
    rotation.update(0.0, q)
    q2 = Rotation.from_euler("z", -179, degrees=True).as_quat()
    middle = Rotation.from_quat(rotation.update(.01, q2))
    expected = Rotation.from_euler("z", 180, degrees=True)
    assert (middle.inv()*expected).magnitude() < 1e-12
    passed.append("SO(3) quaternion sign invariance and 179/-179 crossing")

    # A fixed change of world/body coordinates must commute with this scalar-
    # gain geometric update. Noncommuting rotations catch multiplication-order
    # errors that a single-axis wrap example cannot expose.
    t = np.linspace(0, 1, 101)
    measured = Rotation.from_euler("xyz", np.column_stack([
        .8*np.sin(3*t), .6*np.cos(2*t), 1.1*t]))
    frame = Rotation.from_rotvec([.4, -.6, .2])

    def filtered_rotations(sequence):
        filt = RotationOneEuro(min_cutoff=1.3, beta=.4)
        return Rotation.from_quat([
            filt.update(ti, qi) for ti, qi in zip(t, sequence.as_quat())])

    original = filtered_rotations(measured)
    world_changed = filtered_rotations(frame * measured)
    body_changed = filtered_rotations(measured * frame)
    assert ((frame*original).inv()*world_changed).magnitude().max() < 1e-12
    assert ((original*frame).inv()*body_changed).magnitude().max() < 1e-12
    passed.append("SO(3) fixed world/body frame equivariance for coupled rotations")

    t = np.arange(301) / 200.0
    y = msd_filter(t, np.ones_like(t), initial_position=0.0)
    wn = 2*np.pi*5
    expected = 1 - (1 + wn*t)*np.exp(-wn*t)
    np.testing.assert_allclose(y, expected, atol=2e-14)
    passed.append("Exact ZOH critical-damping analytic step response")

    target = .01
    velocity = 2.0
    y = msd_filter(t, np.full_like(t, target), initial_position=0.0,
                   initial_velocity=velocity)
    expected = target + (-target + (velocity-wn*target)*t)*np.exp(-wn*t)
    np.testing.assert_allclose(y, expected, atol=2e-14)
    assert y.max() > target + .01
    passed.append("Critical damping does not prohibit overshoot from nonzero velocity")

    weights = signal.savgol_coeffs(7, 2, pos=6, deriv=1, delta=.01, use="dot")
    np.testing.assert_allclose(weights @ (np.arange(7)*.01), 1.0, atol=1e-12)
    passed.append("Causal endpoint SG preserves constant-velocity derivative")

    t = np.arange(100) / 100.0
    np.testing.assert_allclose(np.cos(2*np.pi*70*t), np.cos(2*np.pi*30*t), atol=1e-12)
    passed.append("Aliasing: 70 Hz and 30 Hz cosines at 100 Hz sampling")
    return passed


def parameter_sweep(fs, t, truth, measured):
    """Compare parameters on one fixed signal, not a robot or universal ranking.

    Every row uses the same static and constant-speed metric windows. The grid
    is deliberately explicit; a selected row is only best within this grid.
    """
    still = (t >= .8) & (t < 1.8)
    ramp = (t >= 2.5) & (t < 3.5)
    rows = []

    def add(family, parameter, value, output):
        rows.append({
            "family": family, "parameter": parameter, "value": float(value),
            "stationary_std_mm": float(np.std(output[still])*1000),
            "stationary_bias_mm": float(np.mean(output[still]-truth[still])*1000),
            "ramp_equivalent_lag_ms": float(np.mean(truth[ramp]-output[ramp])/.2*1000),
        })

    for size in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
        weights = np.arange(size, 0, -1, dtype=float)
        add("Weighted moving average", "window_samples", size,
            run_scalar(WeightedAverage(weights), measured))
    for cutoff in np.geomspace(.5, 20, 28):
        sos = signal.butter(2, cutoff, fs=fs, output="sos")
        output, _ = signal.sosfilt(sos, measured, zi=signal.sosfilt_zi(sos)*measured[0])
        add("Butterworth 2nd", "cutoff_hz", cutoff, output)
        for beta in [0.0, 4.0]:
            f = OneEuro(min_cutoff=cutoff, beta=beta, derivative_cutoff=1.0)
            output = np.array([f.update(ti, xi) for ti, xi in zip(t, measured)])
            add(f"One Euro beta={beta:g}", "minimum_cutoff_hz", cutoff, output)
        add("Critical MSD", "natural_hz", cutoff,
            msd_filter(t, measured, natural_hz=cutoff))
    return rows


def make_plots(destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .20, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "lines.linewidth": 1.7})
    colors = ["#377db8", "#9260b5", "#d8882d", "#518b58", "#c65568"]
    styles = ['-', '--', '-.', ':', (0, (5, 2, 1, 2))]
    fs, t, truth, measured = synthetic_signal()
    signal_data = {"sample_hz": fs, "seed": 20260925, "unit": "m",
                   "t": t.tolist(), "truth": truth.tolist(), "measurement": measured.tolist()}
    (destination/"filter-signal.json").write_text(json.dumps(signal_data,separators=(",",":"))+"\n")
    outputs = {
        "WMA [0.4,0.3,0.2,0.1]": run_scalar(WeightedAverage([.4,.3,.2,.1]), measured),
        "Butterworth 2nd, 3 Hz": run_scalar(OnlineButterworth(fs, 3), measured),
        "One Euro (1 Hz, beta=4)": None,
        "Critical MSD, fn=5 Hz": msd_filter(t, measured),
    }
    euro = OneEuro(min_cutoff=1, beta=4, derivative_cutoff=1)
    outputs["One Euro (1 Hz, beta=4)"] = np.array([euro.update(ti, xi) for ti, xi in zip(t, measured)])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    ranges = [(0, 10), (.8, 1.8), (2, 2.65)]
    titles = ["Synthetic position tracking (200 Hz)", "Stationary detail", "Motion onset detail"]
    for ax, (lo, hi), title in zip(axes, ranges, titles):
        mask = (t >= lo) & (t <= hi)
        ax.plot(t[mask], measured[mask]*1000, color="#b0b7be", alpha=.65, lw=.8, label="Measurement")
        ax.plot(t[mask], truth[mask]*1000, color="#263544", lw=2.2, label="Truth")
        for (name, values), color, style in zip(outputs.items(), colors, styles):
            ax.plot(t[mask], values[mask]*1000, color=color, ls=style, label=name)
        ax.set(title=title, xlabel="Time (s)", ylabel="Position (mm)")
    axes[0].legend(ncol=2, fontsize=9, loc="upper right")
    fig.savefig(destination/"tracking-tradeoff.png", dpi=170)
    plt.close(fig)

    static = (t >= .8) & (t < 1.8)
    constant_speed = (t >= 2.5) & (t < 3.5)
    metrics = {}
    for name, values in {"Measurement": measured, **outputs}.items():
        metrics[name] = {
            "stationary_std_mm": float(np.std(values[static])*1000),
            "ramp_equivalent_lag_ms": float(np.mean(truth[constant_speed]-values[constant_speed])/.2*1000),
            "whole_trace_rmse_mm": float(np.sqrt(np.mean((truth-values)**2))*1000),
        }

    families = {
        "Butterworth": signal.butter(4, 5, fs=fs, output="sos"),
        "Bessel (mag norm)": signal.bessel(4, 5, fs=fs, norm="mag", output="sos"),
        "Chebyshev I (1 dB)": signal.cheby1(4, 1, 5, fs=fs, output="sos"),
        "Elliptic (1/40 dB)": signal.ellip(4, 1, 40, 5, fs=fs, output="sos"),
    }
    f = np.geomspace(.1, 80, 2400)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), constrained_layout=True)
    for (name, sos), color, style in zip(families.items(), colors, styles):
        _, h = signal.sosfreqz(sos, worN=f, fs=fs)
        axes[0].semilogx(f, 20*np.log10(np.maximum(abs(h), 1e-10)), color=color, ls=style, label=name)
        group_ms = -np.gradient(np.unwrap(np.angle(h)), 2*np.pi*f)*1000
        inside = f <= 5
        axes[1].plot(f[inside], group_ms[inside], color=color, ls=style)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", ylim=(-65, 3),
                title="4th-order IIR families; design frequency = 5 Hz, fs = 200 Hz")
    axes[0].axvline(5, ls="--", color="#9ba4ad", lw=1)
    axes[0].legend(fontsize=10)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Group delay (ms)",
                title="Passband delay: the same design frequency is not the same bandwidth")
    fig.savefig(destination/"iir-frequency-delay.png", dpi=170)
    plt.close(fig)

    t_step = np.arange(0, .6, 1/fs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    damping = [.5, 1/np.sqrt(2), 1., 2.]
    for zeta, color, style in zip(damping, colors, styles):
        y = msd_filter(t_step, np.ones_like(t_step), damping_ratio=zeta, initial_position=0)
        axes[0].plot(t_step, y, color=color, ls=style, label=f"zeta={zeta:.3g}")
        r = np.linspace(0, 3, 700)
        h = 1/np.sqrt((1-r*r)**2+(2*zeta*r)**2)
        axes[1].plot(r, 20*np.log10(h), color=color, ls=style)
    axes[0].axhline(1, color="#7e8790", ls="--", lw=1)
    axes[0].set(xlabel="Time (s)", ylabel="Normalized position", title="Same natural frequency: fn = 5 Hz")
    axes[0].legend(fontsize=10)
    axes[1].axhline(-3.0103, color="#7e8790", ls="--", lw=1)
    axes[1].set(xlabel="Frequency / natural frequency", ylabel="Magnitude (dB)",
                title="Damping changes cutoff and resonance", ylim=(-35, 4))
    fig.savefig(destination/"spring-damper-response.png", dpi=170)
    plt.close(fig)

    pose_t = np.arange(0, 2, .01)
    continuous = 170 + 20*pose_t/pose_t[-1]
    wrapped = (continuous+180) % 360 - 180
    naive, naive_y = [], wrapped[0]
    rot_filter = RotationOneEuro(min_cutoff=2, beta=0)
    rotations = []
    pose_alpha = alpha(2.0, .01)
    for ti, angle in zip(pose_t, wrapped):
        naive_y = pose_alpha*angle + (1-pose_alpha)*naive_y
        naive.append(naive_y)
        q = Rotation.from_euler("z", angle, degrees=True).as_quat()
        if int(round(ti/.01)) % 2:
            q = -q  # equivalent quaternion, deliberately change encoding sign
        rotations.append(rot_filter.update(ti, q))
    filtered_rot = Rotation.from_quat(rotations)
    displayed = np.rad2deg(np.unwrap(filtered_rot.as_euler("zyx")[:,0]))
    truth_rot = Rotation.from_euler("z", continuous, degrees=True)
    naive_rot = Rotation.from_euler("z", naive, degrees=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(pose_t, continuous, color="#263544", label="True continuous angle")
    axes[0].plot(pose_t, naive, color=colors[4], ls='--', label="Naive wrapped-angle EMA")
    axes[0].plot(pose_t, displayed, color=colors[0], label="SO(3) low-pass")
    axes[0].set(xlabel="Time (s)", ylabel="Yaw (deg)", title="Crossing +180/-180 degrees")
    axes[0].legend(fontsize=10)
    axes[1].plot(pose_t, np.rad2deg((truth_rot.inv()*naive_rot).magnitude()), color=colors[4], ls='--')
    axes[1].plot(pose_t, np.rad2deg((truth_rot.inv()*filtered_rot).magnitude()), color=colors[0])
    axes[1].set(xlabel="Time (s)", ylabel="Geodesic error (deg)",
                title="Orientation error; alternating quaternion signs do not change rotation")
    fig.savefig(destination/"rotation-wrap.png", dpi=170)
    plt.close(fig)

    sweep = parameter_sweep(fs, t, truth, measured)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    families = [("Weighted moving average", colors[0]),
                ("Butterworth 2nd", colors[1]), ("One Euro beta=0", colors[4]),
                ("One Euro beta=4", colors[2]), ("Critical MSD", colors[3])]
    detail = ax.inset_axes([.52, .33, .45, .33])
    for (family, color), style, marker in zip(families, styles, ['o', 's', '^', 'D', 'v']):
        rows = sorted((r for r in sweep if r['family']==family),
                      key=lambda r:r['stationary_std_mm'])
        ax.plot([r['stationary_std_mm'] for r in rows],
                [r['ramp_equivalent_lag_ms'] for r in rows],
                linestyle=style, marker=marker, markersize=3, color=color, label=family)
        detail.plot([r['stationary_std_mm'] for r in rows],
                    [r['ramp_equivalent_lag_ms'] for r in rows],
                    linestyle=style, marker=marker, markersize=3, color=color)
    detail.axvline(1.0, color='#87939f', ls='--', lw=1)
    detail.set(xlim=(.5,1.5), ylim=(0,140), xticks=[.5,1,1.5],
               yticks=[0,50,100], title='Detail near 1 mm',
               xlabel='Standard deviation (mm)', ylabel='Lag (ms)')
    detail.tick_params(labelsize=9)
    detail.title.set_fontsize(10)
    detail.xaxis.label.set_fontsize(9)
    detail.yaxis.label.set_fontsize(9)
    ax.axvline(1.0, color='#87939f', ls='--', lw=1)
    ax.text(1.03, 570, '1 mm static-jitter budget', color='#596777', fontsize=10)
    ax.set(xlabel='Stationary standard deviation (mm)',
           ylabel='Ramp equivalent lag (ms)', xlim=(0, 6.6), ylim=(-12, 630),
           title='Parameter sweep on the same synthetic position trace')
    ax.legend(loc='upper right', fontsize=10)
    fig.savefig(destination/'parameter-tradeoff.png', dpi=170)
    plt.close(fig)
    (destination/'filter-parameter-sweep.json').write_text(
        json.dumps({'experiment':'synthetic; grid exploration, not a universal ranking',
                    'static_window_s':[.8,1.8], 'ramp_window_s':[2.5,3.5],
                    'intervals':'left closed, right open', 'rows':sweep}, indent=2)+'\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = {"checks": checks(), "experiment": "synthetic; not robot measurements"}
    if args.output_dir:
        report["metrics"] = make_plots(args.output_dir)
        (args.output_dir/"filter-lab-results.json").write_text(
            json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
