"""URDF -> symbolic inverse dynamics -> lumped parameters -> held-out fit.

Constructed fixed-base pendulum: q=0 hangs down, joint axis is +Y.
The script only parameterizes the bundled pendulum, not arbitrary URDFs.
Processed real data can be supplied for that same geometry and conventions.
No hardware, physical-inertia optimization or controller/URDF writeback.
"""
import argparse
from importlib.metadata import version
import math
from numbers import Real
from pathlib import Path

import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import MultibodyForces_, SpatialInertia_, UnitInertia_
from pydrake.symbolic import DecomposeLumpedParameters, Expression, Variable

from identification_common import digest, format_and_save_report, load_processed_csv, rmse, scaled_svd_fit, unseen_fraction

BUNDLE = Path(__file__).resolve().parent
URDF = BUNDLE / "pendulum_id.urdf"
GRAVITY = 9.81


def build_plant():
    plant = MultibodyPlant(time_step=0.)
    Parser(plant).AddModels(URDF)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.mutable_gravity_field().set_gravity_vector([0., 0., -GRAVITY])
    plant.Finalize()
    return plant


def remove_mass_zero_terms(expression, mass):
    """On the physical domain mass>0, remove exact 0/mass artifacts.

    Drake 1.58 can retain these after spatial-inertia transformations.
    This is an exact expression-tree substitution, not a numerical test
    of whether a fitted or sampled parameter looks small.
    """
    if not isinstance(expression, Expression):
        return expression
    if expression.EqualTo(Expression(0.) / mass):
        return Expression(0.)
    constructor, arguments = expression.Unapply()
    return constructor(*(remove_mass_zero_terms(value, mass) for value in arguments))


def symbolic_regressor(plant):
    symbolic = plant.ToSymbolic()
    context = symbolic.CreateDefaultContext()
    m, ell, k, damping = [Variable(name) for name in ("m", "ell", "k", "b")]
    q, v, a = [Variable(name) for name in ("q", "v", "a")]
    # k = J_com_yy / m is a CENTRAL unit inertia (m^2), not kg*m^2.
    # Shift it to the body origin before using the SpatialInertia constructor.
    central = UnitInertia_[Expression](k, k, .005)
    about_origin = central.ShiftFromCenterOfMass([0., 0., ell])
    inertia = SpatialInertia_[Expression](m, [0., 0., -ell], about_origin)
    symbolic.GetBodyByName("pendulum").SetSpatialInertiaInBodyFrame(context, inertia)
    symbolic.SetPositions(context, [q])
    symbolic.SetVelocities(context, [v])
    forces = MultibodyForces_[Expression](symbolic)
    # CalcInverseDynamics does not itself add gravity. Include force elements.
    symbolic.CalcForceElementsContribution(context, forces)
    torque = symbolic.CalcInverseDynamics(context, [a], forces)[0] + damping * v
    torque = remove_mass_zero_terms(torque, m).Expand()
    matrix, alpha, offset = DecomposeLumpedParameters([torque], [m, ell, k, damping])
    # The decomposition does not promise an ordering. Display A, b, h in that
    # order, matching [a, v, g*sin(q)] and tau=A*a+b*v+g*h*sin(q).
    desired = [Expression(a), Expression(v), GRAVITY * q.sin()]
    order = []
    for feature in desired:
        matches = [index for index, value in enumerate(matrix[0])
                   if value.Expand().EqualTo(feature.Expand())]
        if len(matches) != 1:
            raise ValueError("Unexpected symbolic basis; review the decomposition for this Drake version")
        order.append(matches[0])
    if len(alpha) != 3 or not offset[0].EqualTo(Expression(0.)):
        raise ValueError("Expected three lumped parameters and zero known offset")
    return {"torque": torque, "matrix": matrix[:, order], "alpha": alpha[order],
            "offset": offset, "state": (q, v, a), "physical": (m, ell, k, damping)}


def regressors(symbols, q, v, a):
    q, v, a = (np.asarray(values, float) for values in (q, v, a))
    if q.ndim != 2 or q.shape != v.shape or q.shape != a.shape or q.shape[1] != 1:
        raise ValueError("Expected aligned sample-by-one-joint state arrays")
    if not len(q) or not all(np.isfinite(values).all() for values in (q, v, a)):
        raise ValueError("State arrays must be nonempty and finite")
    return np.vstack([[value.Evaluate(dict(zip(symbols["state"], state)))
                       for value in symbols["matrix"][0]]
                      for state in zip(q[:, 0], v[:, 0], a[:, 0])])


def numerical_torques(plant, q, v, a, damping=.08):
    context = plant.CreateDefaultContext()
    forces = MultibodyForces_[float](plant)
    result = []
    for qi, vi, ai in zip(q, v, a):
        plant.SetPositions(context, qi)
        plant.SetVelocities(context, vi)
        plant.CalcForceElementsContribution(context, forces)
        result.append(plant.CalcInverseDynamics(context, ai, forces) + damping * vi)
    return np.asarray(result)


def synthetic_data(plant, validation=False):
    times = np.linspace(0., 8. if not validation else 7., 401 if not validation else 301)
    phase = .6 if validation else 0.
    first, second = (1.7, 3.2) if validation else (1.2, 2.7)
    q = (.7 * np.sin(first * times + phase) + .2 * np.sin(second * times + .2))[:, None]
    v = (.7 * first * np.cos(first * times + phase) + .2 * second * np.cos(second * times + .2))[:, None]
    a = (-.7 * first**2 * np.sin(first * times + phase) - .2 * second**2 * np.sin(second * times + .2))[:, None]
    truth = numerical_torques(plant, q, v, a)
    measured = truth + np.random.default_rng(20261031 + int(validation)).normal(0., .02, truth.shape)
    return times, q, v, a, measured, truth


def validate_com_distance(distance):
    if isinstance(distance, bool) or not isinstance(distance, Real) or not math.isfinite(distance) or distance <= 0:
        raise ValueError("Known COM distance must be a finite positive number in meters")
    return float(distance)


def recover_axis_parameters(lumped, known_com_distance):
    """An independent COM-distance measurement removes one ambiguity.

    Only the pendulum's mass, Y-axis central inertia and viscous damping
    are recovered; the full three-dimensional inertia remains unspecified.
    Negative estimates are reported with failed necessary checks, not clipped.
    """
    known_com_distance = validate_com_distance(known_com_distance)
    lumped = np.asarray(lumped, float)
    if lumped.shape != (3,) or not np.isfinite(lumped).all():
        raise ValueError("Expected finite [A, b, h] coefficients")
    total_inertia, damping, first_moment = lumped
    mass = first_moment / known_com_distance
    central_inertia = total_inertia - first_moment * known_com_distance
    if not np.isfinite([mass, central_inertia]).all():
        raise ValueError("Recovered axis parameters exceed floating-point range")
    return {"known_com_distance_m": float(known_com_distance), "mass_kg": float(mass),
        "central_y_inertia_kg_m2": float(central_inertia), "viscous_damping_nm_s_per_rad": float(damping),
        "necessary_axis_checks": {"mass_positive": bool(mass > 0),
            "central_y_inertia_positive": bool(central_inertia > 0), "damping_nonnegative": bool(damping >= 0)},
        "assumption": "COM distance is independently known; no full 3D inertia or uncertainty certification."}


def run(train=None, validation=None, known_com_distance=None):
    if known_com_distance is not None:
        known_com_distance = validate_com_distance(known_com_distance)
    plant = build_plant()
    symbols = symbolic_regressor(plant)
    synthetic = train is None and validation is None
    if synthetic:
        training, held_out = synthetic_data(plant), synthetic_data(plant, validation=True)
    else:
        if train is None or validation is None:
            raise ValueError("Provide train and validation together")
        if digest(train) == digest(validation):
            raise ValueError("Training and validation files must not be identical")
        training = load_processed_csv(train, ["joint1"]) + (None,)
        held_out = load_processed_csv(validation, ["joint1"]) + (None,)
    _, q, v, a, torque, truth = training
    _, qv, vv, av, torque_v, truth_v = held_out
    matrix, check = regressors(symbols, q, v, a), regressors(symbols, qv, vv, av)
    fit = scaled_svd_fit(matrix, torque.ravel())
    candidate = fit["parameters"]
    report = {"backend": "Drake", "drake": version("drake"), "numpy": np.__version__,
        "algorithm_sha256": digest(__file__), "helper_sha256": digest(BUNDLE / "identification_common.py"),
        "data_kind": "synthetic_joint_torque" if synthetic else "user_processed_joint_torque",
        "urdf_sha256": digest(URDF), "joint_names": ["joint1"],
        "gravity_m_s2": [0., 0., -GRAVITY],
        "symbolic_inverse_dynamics": str(symbols["torque"]),
        "lumped_parameter_expressions": [str(value) for value in symbols["alpha"]],
        "lumped_parameter_names": ["A = J_com_yy + m*ell^2", "b", "h = m*ell"],
        "lumped_parameter_units": ["kg*m^2", "N*m*s/rad", "kg*m"],
        "lumped_parameter_estimates": candidate.tolist(),
        "train_samples": len(q), "validation_samples": len(qv),
        "regressor_shape": list(matrix.shape), "retained_rank": fit["rank"],
        "training_column_l2_norms": fit["raw_column_l2_norms"].tolist(),
        "svd_relative_cutoff": fit["rcond"], "retained_scaled_condition": fit["retained_condition"],
        "scaled_singular_values": fit["singular_values"].tolist(),
        "validation_unseen_scaled_direction_fraction": unseen_fraction(check, fit),
        "train_measured_rmse_nm": rmse(matrix @ candidate, torque.ravel()),
        "validation_measured_rmse_nm": rmse(check @ candidate, torque_v.ravel()),
        "physical_parameters_uniquely_identified": False,
        "interpretation": "Three lumped coefficients do not uniquely identify unknown m, ell, J_com_yy and b."}
    if synthetic:
        parameters = {key: value for key, value in zip(symbols["physical"], (2., .5, .085, .08))}
        alpha_true = np.asarray([value.Evaluate(parameters) for value in symbols["alpha"]])
        symbolic_torque = np.asarray([symbols["torque"].Evaluate(
            parameters | dict(zip(symbols["state"], state)))
            for state in zip(q[:, 0], v[:, 0], a[:, 0])])
        report.update({"lumped_parameter_truth": alpha_true.tolist(),
            "symbolic_numeric_inverse_dynamics_max_error_nm": float(np.max(np.abs(symbolic_torque - truth.ravel()))),
            "decomposition_max_error_nm": float(np.max(np.abs(matrix @ alpha_true - truth.ravel()))),
            "validation_noise_free_rmse_nm": rmse(check @ candidate, truth_v.ravel()),
            "scope": "Bundled-URDF numerical inverse dynamics with explicit viscous damping, analytic motion derivatives and IID torque noise; not robot measurements."})
    else:
        report["input_sha256"] = {"train": digest(train), "validation": digest(validation)}
    if known_com_distance is not None:
        report["axis_parameters_given_known_com_distance"] = recover_axis_parameters(candidate, known_com_distance)
        report["axis_model_unique_given_known_com_distance"] = fit["rank"] == 3
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--known-com-distance", type=float, help="Independently known positive COM distance in meters")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.train, args.validation, args.known_com_distance)
        text = format_and_save_report(result, args.output, (URDF, args.train, args.validation))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(text, end="")
