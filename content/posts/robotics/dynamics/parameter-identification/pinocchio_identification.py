"""URDF -> Pinocchio torque regressor -> scaled SVD -> held-out prediction.

No arguments: run a constructed two-link data experiment, not robot data.
Real processed files: --urdf robot.urdf --train train.csv --validation valid.csv
CSV uses time_s,q/<joint>...,dq/<joint>...,ddq/<joint>...,tau/<joint>... .
Only fixed-base models with one bounded revolute coordinate per joint are handled.
No hardware, contact/friction estimation, physical-inertia fitting or URDF writeback.
"""
import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin

from identification_common import digest, format_and_save_report, load_processed_csv, rmse, scaled_svd_fit, unseen_fraction, write_processed_csv

BUNDLE = Path(__file__).resolve().parent
PARAMETER_ORDER = ["m", "mcx", "mcy", "mcz", "Ixx_O", "Ixy_O", "Iyy_O", "Ixz_O", "Iyz_O", "Izz_O"]


def gravity_vector(values):
    try:
        values = tuple(values)
        if len(values) != 3 or any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("Gravity needs three finite components in the model root frame")
        vector = np.asarray(values, dtype=float)
    except (TypeError, OverflowError) as error:
        raise ValueError("Gravity needs three finite components in the model root frame") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Gravity needs three finite components in the model root frame")
    return vector


def build_model(path, gravity=None):
    vector = None if gravity is None else gravity_vector(gravity)
    model = pin.buildModelFromUrdf(str(path))
    if model.nv < 1 or model.nq != model.nv or model.njoints - 1 != model.nv:
        raise ValueError("This CSV example needs a fixed base and one bounded scalar coordinate per joint")
    if any(joint.nq != 1 or joint.nv != 1 for joint in model.joints[1:]):
        raise ValueError("Continuous/multi-coordinate joints need an explicit CSV configuration adapter")
    if any(not joint.shortname().startswith("JointModelR") for joint in model.joints[1:]):
        raise ValueError("This torque-in-N*m example handles bounded revolute joints only")
    if vector is not None:
        model.gravity.linear = vector
    return model


def regressors(model, q, v, a):
    q, v, a = (np.asarray(value, float) for value in (q, v, a))
    if q.ndim != 2 or q.shape != v.shape or q.shape != a.shape or q.shape[1] != model.nv:
        raise ValueError("Expected aligned sample-by-joint q/dq/ddq arrays")
    if not len(q) or not all(np.isfinite(value).all() for value in (q, v, a)):
        raise ValueError("State arrays must be nonempty and finite")
    data = model.createData()
    # Both algorithms reuse data buffers; copy each result before the next call.
    return np.vstack([pin.computeJointTorqueRegressor(model, data, qi, vi, ai).copy()
                      for qi, vi, ai in zip(q, v, a)])


def synthetic_truth_model(model):
    """Same kinematics, different physical inertias; never used by the solver."""
    if model.nv != 2 or list(model.names)[1:] != ["joint1", "joint2"]:
        raise ValueError("Constructed inertias are defined only for the bundled two-link example")
    truth_model = pin.Model(model)
    for index, (mass, center, diagonal) in enumerate((
            (1.38, [.32, .012, -.008], [.0007, .041, .0412]),
            (.72, [.21, -.006, .01], [.0004, .0115, .0117])), 1):
        truth_model.inertias[index] = pin.Inertia(mass, np.array(center), np.diag(diagonal))
    return truth_model


def synthetic_data(model, validation=False):
    if model.nv != 2:
        raise ValueError("The constructed trajectories are defined only for the bundled two-link model")
    times = np.linspace(0., 8. if not validation else 7., 401 if not validation else 301)
    phase = .7 if validation else 0.
    frequencies = ((1.1, 2.3), (1.9, 3.3)) if validation else ((1.2, 2.7), (1.7, 3.1))
    q, v, a = [], [], []
    for index, (w1, w2) in enumerate(frequencies):
        first, second = .45 - .05 * index, .12 + .02 * index
        q.append(first * np.sin(w1 * times + phase) + second * np.sin(w2 * times + .3))
        v.append(first * w1 * np.cos(w1 * times + phase) + second * w2 * np.cos(w2 * times + .3))
        a.append(-first * w1**2 * np.sin(w1 * times + phase) - second * w2**2 * np.sin(w2 * times + .3))
    q, v, a = np.array(q).T, np.array(v).T, np.array(a).T
    truth_model = synthetic_truth_model(model)
    data = truth_model.createData()
    truth = np.vstack([pin.rnea(truth_model, data, qi, vi, ai).copy() for qi, vi, ai in zip(q, v, a)])
    rng = np.random.default_rng(20261029 + int(validation))
    measured = truth + rng.normal(0., .02, truth.shape)
    return times, q, v, a, measured, truth


def derivative_check(model, q, v, a):
    """State derivatives differ from the inertial-parameter regressor Y."""
    data = model.createData()
    derivatives = [value.copy() for value in pin.computeRNEADerivatives(model, data, q, v, a)]
    step, errors = 1e-6, []
    for variable, expected in enumerate(derivatives):
        numerical = np.empty_like(expected)
        for column in range(model.nv):
            plus, minus = [q.copy(), v.copy(), a.copy()], [q.copy(), v.copy(), a.copy()]
            plus[variable][column] += step
            minus[variable][column] -= step
            upper = pin.rnea(model, data, *plus).copy()
            lower = pin.rnea(model, data, *minus).copy()
            numerical[:, column] = (upper - lower) / (2 * step)
        errors.append(float(np.max(np.abs(expected - numerical))))
    return dict(zip(("dtau_dq", "dtau_dv", "dtau_da"), errors))


def run(urdf, train=None, validation=None, csv_dir=None, gravity=None):
    model = build_model(urdf, gravity)
    names = list(model.names)[1:]
    synthetic = train is None and validation is None
    if synthetic:
        if Path(urdf).resolve() != (BUNDLE / "two_link_id.urdf").resolve():
            raise ValueError("Provide independent processed train/validation files for a custom URDF")
        training, held_out = synthetic_data(model), synthetic_data(model, validation=True)
    else:
        if train is None or validation is None:
            raise ValueError("Provide train and validation together")
        if digest(train) == digest(validation):
            raise ValueError("Training and validation files must not be identical")
        training = load_processed_csv(train, names) + (None,)
        held_out = load_processed_csv(validation, names) + (None,)
    _, q, v, a, torque, truth = training
    _, qv, vv, av, torque_v, truth_v = held_out
    matrix, check = regressors(model, q, v, a), regressors(model, qv, vv, av)
    fit = scaled_svd_fit(matrix, torque.ravel())
    candidate = fit["parameters"]
    nominal = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
    # Keep the article convention r = measured torque - predicted torque.
    train_residual = torque - (matrix @ candidate).reshape(torque.shape)
    val_residual = torque_v - (check @ candidate).reshape(torque_v.shape)
    nominal_val_residual = torque_v - (check @ nominal).reshape(torque_v.shape)
    report = {"backend": "Pinocchio", "pinocchio": pin.__version__, "numpy": np.__version__,
        "algorithm_sha256": digest(__file__), "helper_sha256": digest(BUNDLE / "identification_common.py"),
        "data_kind": "synthetic_joint_torque" if synthetic else "user_processed_joint_torque",
        "urdf_sha256": digest(urdf), "joint_names": names,
        "gravity_m_s2": model.gravity.linear.tolist(),
        "regressor_model": "Rigid-body inertial parameters, including gravity",
        "omitted_torque_terms": ["actuator armature", "joint friction", "torque bias", "external contact"],
        "nominal_model_armature_kg_m2": model.armature.tolist(),
        "train_samples": len(q), "validation_samples": len(qv),
        "regressor_shape": list(matrix.shape), "parameter_count": matrix.shape[1],
        "retained_rank": fit["rank"], "unobserved_directions": matrix.shape[1] - fit["rank"],
        "column_scaling": "training column L2 norms; zero columns use scale 1",
        "training_column_l2_norms": fit["raw_column_l2_norms"].tolist(),
        "svd_relative_cutoff": fit["rcond"], "retained_scaled_condition": fit["retained_condition"],
        "scaled_singular_values": fit["singular_values"].tolist(),
        "validation_unseen_scaled_direction_fraction": unseen_fraction(check, fit),
        "train_measured_rmse_nm": rmse(matrix @ candidate, torque.ravel()),
        "validation_measured_rmse_nm": rmse(check @ candidate, torque_v.ravel()),
        "residual_definition": "measured torque minus predicted torque",
        "per_joint_validation_rmse_nm": np.sqrt(np.mean(val_residual**2, axis=0)).tolist(),
        "per_joint_validation_mean_residual_nm": np.mean(val_residual, axis=0).tolist(),
        "per_joint_validation_peak_absolute_residual_nm": np.max(np.abs(val_residual), axis=0).tolist(),
        "per_joint_train_rmse_nm": np.sqrt(np.mean(train_residual**2, axis=0)).tolist(),
        "nominal_urdf_validation_measured_rmse_nm": rmse(check @ nominal, torque_v.ravel()),
        "per_joint_nominal_urdf_validation_rmse_nm": np.sqrt(np.mean(nominal_val_residual**2, axis=0)).tolist(),
        "parameter_order_per_joint": PARAMETER_ORDER,
        "prediction_parameter_vector": candidate.tolist(),
        "physical_parameters_uniquely_identified": False,
        "interpretation": "One minimum-norm prediction vector in scaled coordinates of a data-dependent retained subspace; not a physical URDF update."}
    if synthetic:
        truth_model = synthetic_truth_model(model)
        parameters = np.concatenate([inertia.toDynamicParameters() for inertia in truth_model.inertias[1:]])
        report.update({"regressor_rnea_max_error_nm": float(np.max(np.abs(matrix @ parameters - truth.ravel()))),
            "synthetic_truth_dynamic_parameters": parameters.tolist(),
            "validation_noise_free_rmse_nm": rmse(check @ candidate, truth_v.ravel()),
            "nominal_urdf_validation_noise_free_rmse_nm": rmse(check @ nominal, truth_v.ravel()),
            "state_derivative_finite_difference_max_errors": derivative_check(model, q[11], v[11], a[11]),
            "scope": "Same-geometry synthetic torques with inertias different from the nominal URDF, analytic motion derivatives and IID torque noise; no measured robot accuracy."})
    else:
        report["input_sha256"] = {"train": digest(train), "validation": digest(validation)}
    if csv_dir is not None:
        csv_dir = Path(csv_dir)
        if any((csv_dir / name).exists() for name in ("train.csv", "validation.csv")):
            raise ValueError("CSV export needs fresh train.csv and validation.csv paths")
        csv_dir.mkdir(parents=True, exist_ok=True)
        write_processed_csv(csv_dir / "train.csv", names, *training[:5])
        write_processed_csv(csv_dir / "validation.csv", names, *held_out[:5])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=BUNDLE / "two_link_id.urdf")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--csv-dir", type=Path, help="Export aligned copies of the input data")
    parser.add_argument("--gravity", nargs=3, type=float, metavar=("GX", "GY", "GZ"),
                        help="Gravity in the fixed-base model root frame, m/s^2; default 0 0 -9.81")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.urdf, args.train, args.validation, args.csv_dir, args.gravity)
        protected = [args.urdf, args.train, args.validation]
        if args.csv_dir is not None:
            protected.extend(args.csv_dir / name for name in ("train.csv", "validation.csv"))
        text = format_and_save_report(result, args.output, protected)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(text, end="")
