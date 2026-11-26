#!/usr/bin/env python3
"""Replay two pinned NeuralActuator checkpoints on a few NAD files using JAX CPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pickle
import sys
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

SOURCE_REVISION = "541a93aac4368e73883e86a51f31a7add6bde059"
WEIGHTS_REVISION = "cd58d48d465450ffb734ed8ba22e48266ccdb2f2"
SOURCE_SHA = {
    "models/transformer.py": "524a53b9d73ca27c7808ae052c472fbe9295bfdebb725c1e153e616216029490",
    "infer_actuator.py": "629d1f58103ecf047b7a9bf1cc462b43af40190f270629f58c3d006514217b8d",
    "evaluate_actuator.py": "120b1f771751788e4174fa9225fe34117d56263a9fae8fbd9a877e32de9c54de",
    "evaluate_actuator_franka.py": "4e6c10f22e676b2087323c5fe6ae536f3e2cb4a58359941c01991104eedb87f9",
    "configs/force_sensor.yaml": "a09b50080df7d03b6e5a17f5920b720b1a3cb543a24debee0202f9d2119fbc9a",
    "configs/franka_lift_hold_ft2.yaml": "419d751942c888cbf8663ade5d7536fbd8a2dba929b5020cfdf6487c574c8622",
    "data/force_sensor/force_x_plus/test/001.csv": "b6f850350f29f7c5fad35e1acd41ba2e1f826343dd36ca166137f3dc3bdaecf1",
    "data/force_sensor/force_x_plus_ref/test/001.csv": "0adb6e618c7ba1b6f0119d29733e29dee8721b212be58875cdfd6189bacb8bea",
    "data/franka/lift_hold/test/200_006.csv": "86d49d30d60821021a6da1a884937f49be601999bba47cda00cb9d3ff7234cf9",
    "data/franka/lift_hold/test/300_001.csv": "4c64575867126ee1af7e3d5fb84edff1dbe2c6225d704063d3d102358dd38dfc",
    "data/franka/lift_hold/test/400_001.csv": "1fd5790665dfde8c9845c5ecb87e5a85cdcaa4f4a9ca2eefa0e835920665a020",
    "data/franka/lift_hold/test/500_006.csv": "974794597fb05698e2f9e38a66fc7e9d99d374cf9da0e26dea40f76cd1f092ae",
    "data/franka/lift_hold/test/600_003.csv": "46f3fb16a10d5eb885f5c7709a4cdb0b465b9b985fd2f6acb32f02560b2e20f1",
}
WEIGHTS_SHA = {
    "omx_force_sensor.pkl": "edceee654507303227c2247217bf643aa316103e0b8b32b6df45c32c603b3776",
    "franka_lift_hold.pkl": "0894168cac986744b76d7d4e19d1995401cff4579aff63f49c54f7de447d64dd",
}
FRANKA_MJCF = {
    "robot_franka/scene.xml": "59b4ad053a3d354eb5dbbc10137b220ae10312478b95a8fa41f82fa5dd48908f",
    "robot_franka/panda.xml": "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273",
}
FRANKA_HAND_Z_OFFSET_M = 0.10
PAYLOAD_DELTA_KG = 0.10
GRAVITY_M_S2 = 9.81


def verified(folder, digests):
    contents = {}
    for name, digest in digests.items():
        data = (folder / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
        contents[name] = data
    return contents


def load_selected(text, filename, names, namespace):
    """Load only verified CSV/feature/replay functions, without simulator imports."""
    tree = ast.parse(text, filename)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing a replay function in {filename}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return tuple(namespace[name] for name in names)


def franka_static_compensation(model, positions):
    """Compute a static compensation increment at a hypothetical point on the hand."""
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    hand = model.body("hand").id
    arm_joint_ids = [model.joint(f"joint{index}").id for index in range(1, 8)]
    addresses = model.jnt_qposadr[arm_joint_ids]
    dofs = model.jnt_dofadr[arm_joint_ids]
    jacobian = np.empty((3, model.nv))
    projected_force = np.empty(model.nv)
    moment = np.zeros(3)
    delta_force = np.array([0.0, 0.0, -PAYLOAD_DELTA_KG * GRAVITY_M_S2])
    increments = []
    for pose in positions:
        data.qpos[addresses] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        hand_z = data.xmat[hand].reshape(3, 3)[:, 2]
        point = data.xpos[hand] + FRANKA_HAND_Z_OFFSET_M * hand_z
        mujoco.mj_jac(model, data, jacobian, None, point, hand)
        increment = -(jacobian.T @ delta_force)

        projected_force.fill(0)
        mujoco.mj_applyFT(model, data, delta_force, moment, point, hand, projected_force)
        np.testing.assert_allclose(increment, -projected_force, rtol=0, atol=1e-12)
        increments.append(increment[dofs])
    return np.asarray(increments)


def franka_external_torque_projection(jacobian_z, estimated_torque, nominal_force_z):
    """Compare both CSV torque signs using the same vertical point-force model."""
    import numpy as np

    length_squared = np.sum(jacobian_z * jacobian_z, axis=1)
    if np.any(length_squared <= 0):
        raise ValueError("Cannot project at a pose with a zero vertical Jacobian")
    direct_force = np.sum(jacobian_z * estimated_torque, axis=1) / length_squared
    residual = estimated_torque - jacobian_z * direct_force[:, None]
    return {
        "as_logged_forcez_mae_n": float(np.abs(direct_force - nominal_force_z).mean()),
        "reversed_sign_forcez_mae_n": float(np.abs(-direct_force - nominal_force_z).mean()),
        "as_logged_mean_forcez_n": float(direct_force.mean()),
        "mean_joint_torque_residual_norm_nm": float(np.linalg.norm(residual, axis=1).mean()),
        "min_vertical_jacobian_norm_m_per_rad": float(np.sqrt(length_squared.min())),
    }


def replay(source, weights, franka_scene=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    import yaml
    from numpy._core.multiarray import _reconstruct

    jax.config.update("jax_platforms", "cpu")
    transformer = ModuleType("_pinned_neuralactuator_transformer")
    sys.modules[transformer.__name__] = transformer
    model_file = "models/transformer.py"
    exec(compile(source[model_file], model_file, "exec"), transformer.__dict__)

    class ArrayUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = {
                ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
                ("numpy", "ndarray"): np.ndarray,
                ("numpy", "dtype"): np.dtype,
            }
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"Unexpected checkpoint global: {module}.{name}")
            return allowed[module, name]

    infer_file = "infer_actuator.py"
    (build_scan,) = load_selected(source[infer_file], infer_file, ("_build_force_only",),
                                  {"np": np, "jax": jax, "jnp": jnp})
    report = {
        "source_revision": SOURCE_REVISION,
        "weights_revision": WEIGHTS_REVISION,
        "cpu_packages": {name: version(name) for name in ("jax", "jaxlib", "flax", "numpy", "pandas", "pyyaml")},
        "cases": {},
    }
    traces = {}

    def runner(robot, config, filename, joint_count):
        weights_payload = ArrayUnpickler(io.BytesIO(weights[filename])).load()
        params = jax.tree.map(jnp.asarray, weights_payload["params"])
        norm = None
        if "feature_mean" in weights_payload:
            norm = (weights_payload["feature_mean"], weights_payload["feature_std"])
        keys = ("hidden_dim", "latent_dim", "dropout_rate", "num_heads", "num_layers",
                "d_ff", "pool_type", "use_gated_attention")
        model = transformer.TransformerActuator(n_joints=joint_count, **{k: config[k] for k in keys})
        scan, _ = build_scan(model, config, norm, robot)
        history_size = config["history_length"]
        report[f"{robot}_parameter_count"] = sum(array.size for array in jax.tree.leaves(params))
        report[f"{robot}_weight_key"] = "params"

        def run(name, features, data, score_against_log=True):
            # Match the CLI, including its N−1 output rows in force-only mode.
            features = np.asarray(features[:-1], dtype=np.float32)
            history = jnp.zeros((history_size, features.shape[1]), dtype=jnp.float32)
            _, force, gate, _ = map(np.asarray, scan(params, history, jnp.asarray(features)))
            if not np.isfinite(force).all() or not np.isfinite(gate).all():
                raise ValueError(f"Non-finite force or gate in {name}")

            # Rebuild the normalized windows without scan and check its buffer and gate.
            positions = np.array([0, 1, history_size, len(features) // 2, len(features) - 1])
            tokens = np.clip((features - norm[0]) / norm[1], -10, 10) if norm else features
            padded = np.concatenate((np.zeros_like(features[:history_size]), tokens))
            windows = np.stack([padded[t:t + history_size + 1] for t in positions])
            _, expected, raw, expected_gate, _, _ = model.apply(
                params, windows[:, :-1].reshape(len(positions), -1), windows[:, -1], training=False
            )
            np.testing.assert_allclose(force[positions], expected, rtol=1e-5, atol=3e-5)
            np.testing.assert_allclose(gate[positions], np.ravel(expected_gate), rtol=1e-5, atol=3e-5)
            np.testing.assert_allclose(expected, raw * expected_gate, rtol=0, atol=1e-7)

            reference = data["forces"][:-1]
            if score_against_log:
                error = np.abs(force - reference)
                report["cases"][name] = {
                    "frames": len(features),
                    "mae_xyz_n": error.mean(axis=0).tolist(),
                    "component_mae_n": float(error.mean()),
                    "mean_gate": float(gate.mean()),
                }
            traces[f"{name}_force"] = force
            traces[f"{name}_gate"] = gate
            traces[f"{name}_reference"] = reference
            traces[f"{name}_time"] = data["timestamps"][:-1]
            return force, gate

        return run

    omx_file = "evaluate_actuator.py"
    omx_load, omx_features = load_selected(source[omx_file], omx_file,
                                         ("load_csv_data", "build_features"), {"np": np, "pd": pd})
    omx_config = yaml.safe_load(source["configs/force_sensor.yaml"])
    omx_run = runner("omx", omx_config, "omx_force_sensor.pkl", 5)
    alpha = float(omx_config.get("current_lowpass_alpha", 0.0))
    paths = [f"data/force_sensor/{task}/test/001.csv" for task in ("force_x_plus", "force_x_plus_ref")]
    loaded, reference = [omx_load(io.BytesIO(source[path]), alpha) for path in paths]
    np.testing.assert_allclose(loaded["goal_positions"], reference["goal_positions"], rtol=0, atol=1e-12)

    def build_omx(data):
        return [omx_features(data["goal_positions"][i], data["positions"][i],
                              data["aperture"][i] * 1000, data["currents"][i], data["velocities"][i],
                              data["volts"][i], data["temps"][i], data["goal_aperture"][i])
                for i in range(data["n_samples"])]

    omx_run("omx_push", build_omx(loaded), loaded)
    omx_run("omx_reference", build_omx(reference), reference)
    with_swapped_current = {**loaded, "currents": reference["currents"]}
    omx_run("omx_push_reference_current", build_omx(with_swapped_current), loaded)

    franka_file = "evaluate_actuator_franka.py"
    franka_load, franka_features = load_selected(source[franka_file], franka_file,
                                               ("load_csv_data", "build_features"), {"np": np, "pd": pd})
    franka_config = yaml.safe_load(source["configs/franka_lift_hold_ft2.yaml"])
    franka_run = runner("franka", franka_config, "franka_lift_hold.pkl", 7)
    model = None
    if franka_scene is not None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(franka_scene))
        report["franka_torque_probe"] = {
            "mujoco_version": version("mujoco"),
            "mjcf_sha256": FRANKA_MJCF,
            "assumed_hand_local_z_m": FRANKA_HAND_Z_OFFSET_M,
            "payload_delta_kg": PAYLOAD_DELTA_KG,
            "gravity_m_s2": GRAVITY_M_S2,
            "expected_vertical_force_delta_n_if_tracking": -PAYLOAD_DELTA_KG * GRAVITY_M_S2,
            "task_scores": {},
        }

    def build_franka(data, goals, torque):
        return [
            franka_features(goals[i], data["positions"][i], data["gripper_width"][i], torque[i],
                            data["velocities"][i], data["motor_pos"][i], data["motor_vel"][i])
            for i in range(data["n_samples"])
        ]

    for payload, csv_path in franka_config["test_datasets"].items():
        csv_bytes = source[csv_path]
        franka = franka_load(io.BytesIO(csv_bytes), franka_config)
        actual = pd.read_csv(io.BytesIO(csv_bytes))
        start = int(len(actual) * franka_config["trim_start"])
        end = int(len(actual) * (1 - franka_config["trim_end"]))
        logged = actual.iloc[start:end][[f"cmd_pos{axis}" for axis in range(1, 8)]].to_numpy()
        no_lookahead = franka["positions"] * franka_config["lookahead_scale"]
        base_force, base_gate = franka_run(
            f"franka_{payload}_proxy",
            build_franka(franka, franka["cmd_positions"], franka["tau_d"]),
            franka,
        )
        for mode, goals in (("same_frame", no_lookahead), ("logged_goals", logged)):
            features = build_franka(franka, goals, franka["tau_d"])
            franka_run(f"franka_{payload}_{mode}", features, franka)

        if model is not None and payload != "200g":
            increment = franka_static_compensation(model, franka["positions"])
            jacobian_z = increment[:-1] / (PAYLOAD_DELTA_KG * GRAVITY_M_S2)
            estimated_columns = [f"tau_ext{axis}" for axis in range(1, 8)]
            estimated_torque = actual.iloc[start:end - 1][estimated_columns].to_numpy()
            probe = {
                "frames": len(base_force),
                "max_added_joint_torque_abs_nm": np.abs(increment[:-1]).max(axis=0).tolist(),
                "baseline_mean_gate": float(base_gate.mean()),
                "external_torque_sign_check": franka_external_torque_projection(
                    jacobian_z, estimated_torque, franka["forces"][:-1, 2],
                ),
            }
            for mode, sign in (("add_payload", 1), ("remove_payload", -1)):
                features = build_franka(franka, franka["cmd_positions"], franka["tau_d"] + sign * increment)
                force, gate = franka_run(f"franka_{payload}_{mode}", features, franka, score_against_log=False)
                difference = force[:, 2] - base_force[:, 2]
                probe[mode] = {
                    "mean_forcez_delta_n": float(difference.mean()),
                    "median_forcez_delta_n": float(np.median(difference)),
                    "expected_sign_fraction": float(np.mean(sign * difference < 0)),
                    "mean_gate": float(gate.mean()),
                }
            report["franka_torque_probe"]["task_scores"][payload] = probe

            if payload == "400g":
                index = len(base_force) // 2
                stop = index + franka_config["history_length"] + 1
                pulsed_torque = franka["tau_d"].copy()
                pulsed_torque[index] += increment[index]
                features = build_franka(franka, franka["cmd_positions"], pulsed_torque)
                force, gate = franka_run("franka_400g_one_frame", features, franka, score_against_log=False)
                np.testing.assert_array_equal(force[:index], base_force[:index])
                np.testing.assert_array_equal(force[stop:], base_force[stop:])
                np.testing.assert_array_equal(gate[:index], base_gate[:index])
                np.testing.assert_array_equal(gate[stop:], base_gate[stop:])
                report["franka_torque_probe"]["one_frame_400g"] = {
                    "modified_retained_row": index,
                    "last_possible_affected_row": stop - 1,
                    "duration_ms_between_first_and_last": float(
                        (franka["timestamps"][stop - 1] - franka["timestamps"][index]) * 1000
                    ),
                    "forcez_delta_n_within_window": (force[index:stop, 2] - base_force[index:stop, 2]).tolist(),
                }

    return report, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned NeuralActuator directory")
    parser.add_argument("--weights", type=Path, required=True, help="Directory containing the two published weights")
    parser.add_argument("--trace", type=Path, help="Optional new NPZ containing force, gate and timestamps")
    parser.add_argument(
        "--franka-torque-probe", action="store_true",
        help="Check Franka torque input responses and logged torque signs (requires MuJoCo and Panda meshes)",
    )
    args = parser.parse_args()
    try:
        sources = verified(args.source, SOURCE_SHA)
        weights = verified(args.weights, WEIGHTS_SHA)
        scene = None
        if args.franka_torque_probe:
            verified(args.source, FRANKA_MJCF)
            scene = args.source / "robot_franka/scene.xml"
        report, traces = replay(sources, weights, scene)
        if args.trace:
            import numpy as np
            with args.trace.open("xb") as stream:
                np.savez_compressed(stream, **traces)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, pickle.UnpicklingError) as error:
        parser.exit(1, f"Replay failed: {error}\n")


if __name__ == "__main__":
    main()
