"""Position/torque CSV -> local-polynomial state estimates -> processed CSV.

NumPy only, offline, bounded revolute coordinates in radians and joint-side N*m.
Requires one cleaned, continuous segment; no unwrap, contact removal, clock-drift
estimation or automatic filter tuning. A known signed torque-channel delay may
be corrected; it is never estimated from training/validation prediction errors.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from identification_common import digest, joint_names, load_raw_csv, write_processed_csv, write_raw_csv


def local_polynomial_states(times, positions, window, degree):
    """Fit each centered sample window using actual, possibly irregular times.

    Scale local time to [-1,1] before solving. Drop both ends rather than
    silently switching to a different one-sided derivative estimator.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    if times.ndim != 1 or positions.ndim != 2 or len(times) != len(positions) or not positions.shape[1]:
        raise ValueError("Expected aligned time and sample-by-joint position arrays")
    if not np.isfinite(times).all() or not np.isfinite(positions).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Positions/times must be finite and times strictly increasing")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 3 or window % 2 != 1:
        raise ValueError("window must be an odd integer of at least 3 samples")
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 2 <= degree < window:
        raise ValueError("degree must be an integer between 2 and window-1")
    if len(times) < window + 1:
        raise ValueError("Need at least window+1 samples to retain two output rows")
    half = window // 2
    indices = np.arange(half, len(times) - half)
    q, v, a, durations, conditions = [], [], [], [], []
    for index in indices:
        local = times[index - half:index + half + 1] - times[index]
        scale = np.max(np.abs(local))
        design = np.vander(local / scale, N=degree + 1, increasing=True)
        coefficients, _, rank, singular = np.linalg.lstsq(
            design, positions[index - half:index + half + 1], rcond=1e-12)
        if rank != degree + 1:
            raise ValueError("A local time window is numerically rank deficient")
        q.append(coefficients[0])
        v.append(coefficients[1] / scale)
        a.append(2 * coefficients[2] / scale**2)
        durations.append(float(local[-1] - local[0]))
        conditions.append(float(singular[0] / singular[-1]))
    return indices, np.asarray(q), np.asarray(v), np.asarray(a), {
        "window_span_min_s": min(durations), "window_span_max_s": max(durations),
        "max_scaled_local_condition": max(conditions), "fit_relative_cutoff": 1e-12,
        "local_fit_summary_scope": "All full position windows before torque-overlap trimming"}


def preprocess_arrays(times, positions, torque, window, degree, torque_delay=0., max_gap=None):
    if isinstance(torque_delay, bool) or not isinstance(torque_delay, Real) or not math.isfinite(torque_delay):
        raise ValueError("torque_delay must be finite seconds")
    times, positions, torque = (np.asarray(values, float) for values in (times, positions, torque))
    if torque.shape != positions.shape or not np.isfinite(torque).all():
        raise ValueError("Torque must be finite and aligned with position samples")
    if max_gap is not None:
        if isinstance(max_gap, bool) or not isinstance(max_gap, Real) or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be finite positive seconds")
        if times.ndim == 1 and np.any(np.diff(times) > max_gap):
            raise ValueError("Input contains a gap exceeding max_gap; process continuous segments separately")
    indices, q, v, a, details = local_polynomial_states(times, positions, window, degree)
    shifted_times = times[indices] + torque_delay
    overlap = (shifted_times >= times[0]) & (shifted_times <= times[-1])
    if np.sum(overlap) < 2:
        raise ValueError("Known delay and window leave fewer than two overlapping samples")
    indices, q, v, a = indices[overlap], q[overlap], v[overlap], a[overlap]
    shifted_times = shifted_times[overlap]
    aligned_torque = np.column_stack([np.interp(shifted_times, times, torque[:, joint])
                                      for joint in range(torque.shape[1])])
    if not all(np.isfinite(values).all() for values in (q, v, a, aligned_torque)):
        raise ValueError("State estimation exceeded floating-point range")
    details.update({"raw_samples": len(times), "retained_samples": len(indices),
        "discarded_before": int(indices[0]), "discarded_after": int(len(times) - indices[-1] - 1),
        "sample_interval_min_s": float(np.min(np.diff(times))),
        "sample_interval_max_s": float(np.max(np.diff(times))),
        "sample_interval_median_s": float(np.median(np.diff(times))),
        "window_samples": int(window), "polynomial_degree": int(degree),
        "torque_delay_s": float(torque_delay), "max_allowed_gap_s": float(max_gap) if max_gap is not None else None,
        "delay_convention": "tau_log(t)=tau_true(t-delay); match tau_log(t+delay) to states at t",
        "torque_interpolation": "linear; no endpoint extrapolation",
        "scope": "Offline state-estimation baseline, not identical filtering of Y and torque; noise covariance and physical accuracy are not certified."})
    return (times[indices], q, v, a, aligned_torque), details


def process_file(source, output, names, window, degree, torque_delay=0., max_gap=None):
    names = joint_names(names)
    output = Path(output)
    metadata = output.with_suffix(".meta.json")
    for path in (output, metadata):
        if path.exists() or path.is_symlink():
            raise ValueError("Preprocessing needs fresh CSV and metadata output paths")
    times, positions, torque = load_raw_csv(source, names)
    values, report = preprocess_arrays(times, positions, torque, window, degree, torque_delay, max_gap)
    report.update({"numpy": np.__version__, "algorithm_sha256": digest(__file__), "joint_names": list(names),
                   "helper_sha256": digest(Path(__file__).with_name("identification_common.py")),
                   "input_sha256": digest(source), "output_csv": output.name})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_processed_csv(output, names, *values)
    report["output_sha256"] = digest(output)
    with metadata.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def make_demo(directory):
    """Create raw-like data; only this optional mode needs Pinocchio."""
    from pinocchio_identification import BUNDLE, build_model, synthetic_data
    import pinocchio as pin
    directory = Path(directory)
    names = ("train-raw.csv", "validation-raw.csv", "demo.json")
    if any((directory / name).exists() or (directory / name).is_symlink() for name in names):
        raise ValueError("Demo generation needs fresh output files")
    model = build_model(BUNDLE / "two_link_id.urdf")
    directory.mkdir(parents=True, exist_ok=True)
    for validation, filename in zip((False, True), names[:2]):
        times, q, _, _, torque, _ = synthetic_data(model, validation)
        noise = np.random.default_rng(20261104 + int(validation)).normal(0., 1e-5, q.shape)
        write_raw_csv(directory / filename, ["joint1", "joint2"], times, q + noise, torque)
    report = {"data_kind": "constructed_position_and_joint_torque", "numpy": np.__version__,
        "pinocchio": pin.__version__, "algorithm_sha256": digest(__file__),
        "helper_sha256": digest(Path(__file__).with_name("identification_common.py")),
        "dynamics_script_sha256": digest(Path(__file__).with_name("pinocchio_identification.py")),
        "urdf_sha256": digest(BUNDLE / "two_link_id.urdf"), "position_noise_std_rad": 1e-5,
        "position_noise_seeds": [20261104, 20261105], "torque_delay_s": 0.,
        "torque_noise_std_nm": .02, "torque_noise_seeds": [20261029, 20261030],
        "input_sha256": {name: digest(directory / name) for name in names[:2]},
        "scope": "Same constructed inertia mismatch and trajectories as the Pinocchio example; no measured robot data."}
    with (directory / "demo.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-demo", type=Path, help="Generate two raw-like fixture files; needs Pinocchio")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--joints", nargs="+")
    parser.add_argument("--window", type=int)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--torque-delay", type=float, default=0.)
    parser.add_argument("--max-gap", type=float)
    args = parser.parse_args()
    try:
        if args.make_demo:
            if any(value is not None for value in (args.input, args.output, args.joints, args.window, args.max_gap)) or args.torque_delay != 0. or args.degree != 3:
                raise ValueError("Demo generation and CSV processing are separate commands")
            result = make_demo(args.make_demo)
        else:
            if any(value is None for value in (args.input, args.output, args.joints, args.window)):
                raise ValueError("Processing needs --input, --output, --joints and --window")
            result = process_file(args.input, args.output, args.joints, args.window, args.degree,
                                  args.torque_delay, args.max_gap)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
