"""Compare five offline position windows on identical constructed timestamps.

Needs NumPy and Pinocchio; no robot data, automatic tuning or hardware.
Analytic derivative and noise-free torque truth are used only for diagnostics.
Every configuration fits the same central training times and is evaluated on
the same separate central validation times. The window settings are fixed.
"""
import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

from identification_common import digest, format_and_save_report, rmse, scaled_svd_fit
from pinocchio_identification import BUNDLE, build_model, regressors, synthetic_data
from preprocess_identification import preprocess_arrays

SETTINGS = ((11, 5), (21, 3), (21, 5), (41, 5), (81, 5))


def run():
    model = build_model(BUNDLE / "two_link_id.urdf")
    datasets = []
    for validation in (False, True):
        times, q, v, a, torque, truth = synthetic_data(model, validation)
        noise = np.random.default_rng(20261104 + int(validation)).normal(0., 1e-5, q.shape)
        datasets.append((times, q + noise, a, torque, truth, q))
    half = max(window // 2 for window, _ in SETTINGS)
    common = [np.arange(half, len(data[0]) - half) for data in datasets]
    rows = []
    for window, degree in SETTINGS:
        prepared = []
        for dataset, indices in zip(datasets, common):
            times, positions, acceleration, torque, truth, true_position = dataset
            (t, q, v, a, force), details = preprocess_arrays(
                times, positions, torque, window, degree, max_gap=.03)
            selected = np.searchsorted(t, times[indices])
            if not np.array_equal(t[selected], times[indices]):
                raise RuntimeError("Window comparison lost a common timestamp")
            matrix = regressors(model, q[selected], v[selected], a[selected])
            (_, _, _, noiseless_a, _), _ = preprocess_arrays(
                times, true_position, torque, window, degree, max_gap=.03)
            prepared.append((matrix, force[selected], truth[indices],
                             rmse(a[selected], acceleration[indices]), details,
                             rmse(noiseless_a[selected], acceleration[indices])))
        training, validation = prepared
        fit = scaled_svd_fit(training[0], training[1].ravel())
        prediction = validation[0] @ fit["parameters"]
        rows.append({"window_samples": window, "polynomial_degree": degree,
            "train_window_span_s": training[4]["window_span_max_s"],
            "validation_window_span_s": validation[4]["window_span_max_s"],
            "retained_rank": fit["rank"],
            "validation_acceleration_rmse_rad_s2": validation[3],
            "validation_acceleration_noiseless_position_rmse_rad_s2": validation[5],
            "validation_measured_torque_rmse_nm": rmse(prediction, validation[1].ravel()),
            "validation_noise_free_torque_rmse_nm": rmse(prediction, validation[2].ravel())})
    support = {}
    for name, dataset, indices in zip(("train", "validation"), datasets, common):
        times = dataset[0][indices]
        support[name] = {"samples": len(indices), "scalar_equations": len(indices) * model.nv,
                         "first_time_s": float(times[0]), "last_time_s": float(times[-1])}
    return {"data_kind": "constructed_same_geometry_inertia_mismatch", "numpy": np.__version__, "pinocchio": pin.__version__,
        "algorithm_sha256": digest(__file__), "helper_sha256": digest(BUNDLE / "identification_common.py"),
        "preprocessor_sha256": digest(BUNDLE / "preprocess_identification.py"),
        "dynamics_script_sha256": digest(BUNDLE / "pinocchio_identification.py"),
        "urdf_sha256": digest(BUNDLE / "two_link_id.urdf"),
        "position_noise_std_rad": 1e-5, "position_noise_seeds": [20261104, 20261105],
        "torque_noise_std_nm": .02, "torque_noise_seeds": [20261029, 20261030],
        "known_torque_delay_s": 0., "common_support": support, "results": rows,
        "scope": "Fixed window comparison on shared times, not automatic selection. Analytic truth is diagnostic only; validation never enters fitting. No measured robot accuracy."}


def summary(result):
    support = result["common_support"]
    lines = [f"Fixed windows; identical {support['train']['samples']}/{support['validation']['samples']} central train/validation times",
             "Window  Degree  Accel RMSE   Torque RMSE   Noise-free RMSE"]
    for row in result["results"]:
        lines.append(f"{row['window_samples']:>6}  {row['polynomial_degree']:>6}"
            f"  {row['validation_acceleration_rmse_rad_s2']:>10.6f}"
            f"  {row['validation_measured_torque_rmse_nm']:>12.6f}"
            f"  {row['validation_noise_free_torque_rmse_nm']:>15.6f}")
    lines.append("Acceleration: rad/s^2; torque: N*m. Constructed data only.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    try:
        result = run()
        protected = [__file__, *(BUNDLE / name for name in (
            "two_link_id.urdf", "identification_common.py", "pinocchio_identification.py", "preprocess_identification.py"))]
        text = format_and_save_report(result, args.output, protected)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(summary(result) if args.summary else text, end="")
