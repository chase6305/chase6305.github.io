#!/usr/bin/env python3
"""Replay selected GMR, UMR and OTRetarget behavior against pinned sources."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import pathlib
import pickle
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree


# GMR 25f56db79ac411258b377b6d4ccba0c8ff3372ac
GMR_SOURCE = ("general_motion_retargeting/motion_retarget.py",
              "5ec97cd4ebb6e3ce4fb525731ead381ccc3567eeb425a88a842688bed2f49718")
GMR_G1 = ("assets/unitree_g1/g1_mocap_29dof.xml",
          "85a25128d42ad3cc8dd071c089806a2773a59ee0689edcc646aeb0201ae8f4c8")
GMR_SMPL = ("general_motion_retargeting/utils/smpl.py",
            "4d358d78c667e6c362fe0e2db984041564a8e8e8a6f19ba29c83c5a8e6687b0d")
GMR_CLI = ("scripts/smplx_to_robot.py",
           "1dae07c8a3ff9fc1ef13b4305b9fa27a7ae4265af40433cb8b3824e32240ae29")
# UMR d1cdaedaf67a827198da71c5fa831f48b2e51f91
UMR_PIPELINE = ("scripts/humanoid_retarget_pipeline.py",
                "520fe066345eda0fca5d97bd2f12c32c1cadd09c9b41b68cc0185d60e62077d0")
UMR_CONFIG = ("scripts/humanoid_retarget_config.py",
              "e926b2f48ba6b559ce24144983a19cae62a258ca771a4426fbcc89918a33aa22")
UMR_TRAIN = ("scripts/train_correspondence_template_residual_ae.py",
             "90cb6e79f869fc424dad57252604fe33e8b2317f638a1cab63475d3b280d0df6")
UMR_CHARACTER = ("scripts/character_surface_retarget_common.py",
                 "9a9aa44e22cf8aeb763a2c2d267378696468dd783e9ae85d58a5976b6af0b100")
UMR_CHARACTER_MODEL = ("assets/mimickit/humanoid.xml",
                       "c595f943910c455648e3b1089c8d5be5a2abb7850dbfc8feba952ca68fa67319")
UMR_CHARACTER_MOTION = ("sample_data/humanoid_character/humanoid_spinkick.pkl",
                        "0e2468b1d2812bf625a6847c8aaa53dceb5cb58a0a0ec8e78f0744656dfc77e3")
UMR_CHARACTER_COMPAT = (
    ("scripts/mimickit_compat/kin_char_model.py", "e25613947a970b8f8dee8a21175adf8d9a790a040c9173a61a0889c06e3b9c45"),
    ("scripts/mimickit_compat/mjcf_char_model.py", "081ea5e06172d81545f5355b3ed67ada20c7020acd2fc33fde14174fb311ecfd"),
    ("scripts/mimickit_compat/torch_util.py", "05b9702445026e4aefeaf049b8ab4e94e370c8bcc7a7b92c3367aa7bb16e7c89"),
)
# lucidrains/OTRetarget e6e87b85435439b2fa330f2fa0878e8c9dd81ca8
OT_SOURCE = ("ot_retarget/ot_retarget.py",
             "667daba40ff20813ad47b63c903af712f8dbb42c63bdea3ca18cd5fd79af968a")

LEVER_MODEL = """<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.25">
      <body name="lever">
        <joint name="hinge" axis="0 0 1" limited="true" range="-85 85"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.25 0 0"/>
        <body name="tip" pos="0.25 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="MOTOR_LABEL" joint="hinge"/></actuator>
</mujoco>"""


def verified(root, spec):
    relative, digest = spec
    source = root / relative
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Source hash mismatch: {source}")
    return data


def executable_nodes(data, kind, names):
    nodes = [node for node in ast.parse(data).body if isinstance(node, kind) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Expected declarations are missing")
    return nodes


def run_declarations(nodes, namespace):
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<verified declarations>", "exec"), namespace)


def check_gmr_timing(source):
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation

    namespace = {"np": np, "R": Rotation, "interp1d": interp1d, "JOINT_NAMES": ["root", "tip"]}
    names = {"get_smplx_data_offline_fast", "slerp"}
    run_declarations(executable_nodes(source, ast.FunctionDef, names), namespace)
    align = namespace["get_smplx_data_offline_fast"]

    index = np.arange(60, dtype=float)
    root_rotvec = np.zeros((60, 3))
    root_rotvec[:, 2] = index / 60
    pose = np.zeros((60, 2, 3))
    positions = np.zeros_like(pose)
    positions[:, 0, 0] = index / 100
    positions[:, 1] = positions[:, 0] + [0, 0.2, 0.3]
    model = SimpleNamespace(parents=(-1, 0))
    joints = SimpleNamespace(detach=lambda: SimpleNamespace(numpy=lambda: positions))
    output = SimpleNamespace(global_orient=root_rotvec, full_pose=pose, joints=joints)

    cases = []
    for source_fps, expected_count, expected_fps in [(24, 60, 30), (50, 60, 50), (60, 30, 30), (100, 20, 100 / 3)]:
        frames, fps = align({"mocap_frame_rate": np.array(source_fps), "pose_body": pose}, model, output)
        assert len(frames) == expected_count and math.isclose(fps, expected_fps)
        # Uniform endpoint interpolation has a closed form for this constant-axis, linear input.
        expected_index = np.arange(expected_count) * (59 / (expected_count - 1))
        np.testing.assert_allclose([f["root"][0][0] for f in frames], expected_index / 100, atol=1e-12)
        half_angle = expected_index / 120
        expected_quat = np.column_stack((np.cos(half_angle), np.zeros((expected_count, 2)), np.sin(half_angle)))
        np.testing.assert_allclose([f["tip"][1] for f in frames], expected_quat, atol=1e-6, rtol=0)
        cases.append({"source_fps": source_fps, "aligned_frames": len(frames), "aligned_fps": fps})
    return cases


def check_gmr_export(source, directory):
    import numpy as np

    cli_main = next(node for node in ast.parse(source).body if isinstance(node, ast.If))
    quaternion = np.arange(1, 5) / math.sqrt(30)
    pose_frames = [np.r_[i, 0, 1, quaternion, i / 10, -i / 10] for i in range(4)]
    viewer_frames = []
    viewer_options = []

    def viewer(**options):
        viewer_options.append(options)
        return SimpleNamespace(step=lambda **frame: viewer_frames.append(frame), close=lambda: None)

    def align(*_parameters, tgt_fps):
        assert tgt_fps == 30
        return pose_frames, 50.

    bindings = {
        "__name__": "__main__", "__file__": GMR_CLI[0], "argparse": argparse, "os": os,
        "pathlib": pathlib, "time": time, "np": np, "IK_CONFIG_DICT": {"smplx": {"unitree_g1": None}},
        "load_smplx_file": lambda *_: (None, None, None, 1.8), "get_smplx_data_offline_fast": align,
        "GMR": lambda **_: SimpleNamespace(retarget=lambda pose: pose, scaled_human_data={}),
        "RobotMotionViewer": viewer,
    }
    destination = directory / "export" / "motion.pkl"
    with patch.object(sys, "argv", ["smplx_to_robot.py", "--smplx_file", "fixture.npz", "--save_path", str(destination)]):
        with contextlib.redirect_stdout(io.StringIO()):
            run_declarations([cli_main], bindings)
    with destination.open("rb") as handle:
        motion = pickle.load(handle)  # The verified CLI just wrote this synthetic fixture.
    np.testing.assert_allclose(motion["root_pos"][:, 0], [1, 2, 3])
    np.testing.assert_allclose(motion["root_rot"], np.tile(quaternion[[1, 2, 3, 0]], (3, 1)))
    np.testing.assert_allclose(motion["dof_pos"], [[.1, -.1], [.2, -.2], [.3, -.3]])
    np.testing.assert_allclose(viewer_frames[0]["root_rot"], quaternion)
    assert len(viewer_frames) == 3 and viewer_options[0]["motion_fps"] == motion["fps"] == 50.
    return {"adapted_frames": 4, "serialized_frames": 3, "first_index": 1, "root_rotation_order": "xyzw", "fps": 50.}


def check_gmr(root, directory):
    data = verified(root, GMR_SOURCE)
    timing = check_gmr_timing(verified(root, GMR_SMPL))
    export = check_gmr_export(verified(root, GMR_CLI), directory)
    model = ElementTree.fromstring(verified(root, GMR_G1))
    motors = model.findall("./actuator/motor")
    assert len(motors) == 29 and all(motor.get("name") == motor.get("joint") for motor in motors)

    import mink
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    xml_path = directory / "lever.xml"
    xml_path.write_text(LEVER_MODEL.replace("MOTOR_LABEL", "hinge"), encoding="utf-8")
    config_path = directory / "lever.json"
    root_task = ["pelvis", 0., 1., [0., 0., 0.], [1., 0., 0., 0.]]
    config = {
        "robot_root_name": "pelvis", "human_root_name": "pelvis", "ground_height": 0.,
        "human_height_assumption": 1.8, "use_ik_match_table1": True, "use_ik_match_table2": True,
        "human_scale_table": {"pelvis": 1., "hand": 1.},
        "ik_match_table1": {"pelvis": root_task, "tip": ["hand", 0., 2., [0, 0, 0], [1., 0., 0., 0.]]},
        "ik_match_table2": {"pelvis": root_task, "tip": ["hand", 1., 2., [0.05, 0, 0], [2**-0.5, 0, 0, 2**-0.5]]},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    namespace = {"mink": mink, "mj": mujoco, "np": np, "R": Rotation, "json": json,
                 "ROBOT_XML_DICT": {"test": xml_path}, "IK_CONFIG_DICT": {"test": {"test": config_path}}}
    run_declarations(executable_nodes(data, ast.ClassDef, {"GeneralMotionRetargeting"}), namespace)
    retargeter_class = namespace["GeneralMotionRetargeting"]

    def create(limited):
        with contextlib.redirect_stdout(io.StringIO()):
            return retargeter_class("test", "test", verbose=False, use_velocity_limit=limited)

    def frame(angle):
        return {"pelvis": ([0., 0., 0.25], [1., 0., 0., 0.]),
                "hand": ([.25 * math.cos(angle), .25 * math.sin(angle), .25],
                         [math.cos(angle / 2), 0., 0., math.sin(angle / 2)])}

    retargeter = create(False)
    retargeter.update_targets(frame(.6))
    first = retargeter.human_body_to_task1["hand"].transform_target_to_world
    second = retargeter.human_body_to_task2["hand"].transform_target_to_world
    np.testing.assert_allclose(first.parameters(), second.parameters(), atol=1e-12)
    np.testing.assert_allclose(first.translation(), frame(.6)["hand"][0], atol=1e-12)
    assert not np.allclose(retargeter.pos_offsets1["hand"], retargeter.pos_offsets2["hand"])
    qpos = [float(retargeter.retarget(frame(angle))[0]) for angle in (.6, .8, .95)]
    np.testing.assert_allclose(qpos, [.6, .8, .95], atol=.015)

    limited = create(True)
    assert len(retargeter.ik_limits) == 1 and len(limited.ik_limits) == 2
    limited_delta = float(limited.retarget(frame(1.3))[0])
    per_step = 3 * math.pi * .005
    assert per_step < limited_delta < 1.3 and limited_delta * 30 > 3 * math.pi
    xml_path.write_text(LEVER_MODEL.replace("MOTOR_LABEL", "different_name"), encoding="utf-8")
    try:
        create(True)
    except KeyError as error:
        assert "different_name" in str(error), error
    else:
        raise AssertionError("The actuator label must be a joint name when this velocity limit is enabled")
    return {"qpos_rad": qpos, "step_bound_rad": per_step, "one_call_with_limit_rad": limited_delta,
            "rad_per_second_if_30_fps": limited_delta * 30, "g1_same_name_pairs": len(motors), "timing": timing,
            "export": export}


def check_umr(root, directory):
    data = verified(root, UMR_PIPELINE)
    config_source = verified(root, UMR_CONFIG)
    import numpy as np

    namespace = {"np": np, "Path": Path}
    run_declarations(executable_nodes(config_source, ast.FunctionDef, {"robot_config", "section"}), namespace)
    run_declarations(executable_nodes(data, ast.FunctionDef, {
        "bool_value", "expected_smpl_slot_name", "expected_correspondence_training_mode", "correspondence_slots_compatible",
    }), namespace)
    compatible = namespace["correspondence_slots_compatible"]
    robot = {"name": "g1", "slot_name": "g1-sample", "xml": "unused.xml", "point_cloud_center": "body:pelvis"}
    default = {"robot": robot, "correspondence": {"smpl_name": "human", "train": {}}}
    fixed = {"robot": robot, "correspondence": {"smpl_name": "human", "train": {"fixed_template": True}}}
    assert not compatible(directory / "absent.npz", default, quiet=True)
    fixture = directory / "slots.npz"
    for names, training, cfg, expected in (
        (["human", "g1-sample"], None, default, True),
        (["human", "g1"], None, default, True),
        (["human", "g1"], None, fixed, False),
        (["human", "g1"], "all_samples", fixed, False),
        (["human", "g1"], "robot2smpl_only", fixed, True),
        (["other", "g1"], "all_samples", default, False),
        (["human", "h2"], "all_samples", default, False),
    ):
        fields = {"names": names}
        if training is not None:
            fields["training_mode"] = training
        np.savez(fixture, **fields)
        assert compatible(fixture, cfg, quiet=True) is expected, (names, training, cfg)
    return {"metadata_only_cases": 8, "mode_and_name_mismatches_detected": True}


def check_umr_training(root, directory):
    import numpy as np

    source = directory / "train_correspondence_template_residual_ae.py"
    source.write_bytes(verified(root, UMR_TRAIN))
    vertices = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=np.float32)
    faces = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                      [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], dtype=np.int32)
    robot_vertices = vertices * np.array([.93, .85, 1.22], dtype=np.float32)
    human_points = np.einsum("ncd,c->nd", vertices[faces], [.6, .25, .15])
    robot_points = np.einsum("ncd,c->nd", robot_vertices[faces], [.6, .25, .15])
    data = directory / "training-sample.npz"
    face_ids = np.arange(len(faces))
    np.savez(data, names=["smplx_neutral", "unitree_g1"], points=[human_points, robot_points],
             mesh_vertices_0=vertices, mesh_faces_0=faces, sample_face_ids_0=face_ids,
             mesh_vertices_1=robot_vertices, mesh_faces_1=faces, sample_face_ids_1=face_ids)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    results = []
    for batch, mode, train, fixed in (
        (1, "robot2smpl_only", [1], [0]), (2, "all_samples", [0, 1], []),
    ):
        output = directory / f"training-{batch}"
        args = [sys.executable, "-B", str(source), "--data", str(data), "--out-dir", str(output),
                "--device", "cpu", "--epochs", "8", "--num-points", "12", "--batch-size", str(batch),
                "--latent-dim", "32", "--hidden-dim", "24", "--noise-std", "0", "--edge-graph", "geodesic",
                "--edge-k", "3", "--save-every", "0"]
        subprocess.run(args, cwd=directory, env=env, text=True, capture_output=True, check=True, timeout=45)
        with np.load(output / "correspondence_slots_final.npz", allow_pickle=False) as slots:
            reconstructed = slots["reconstructed_slots"]
            residual = slots["residual_slots"]
            edges = slots["template_edge_index"]
            assert reconstructed.shape == residual.shape == (2, 12, 3) and np.isfinite(reconstructed).all()
            assert edges.shape == (36, 2) and np.all(edges[:, 0] != edges[:, 1])
            assert slots["training_mode"].item() == mode and np.linalg.norm(residual[1]) > 0
            np.testing.assert_array_equal(slots["train_sample_indices"], train)
            np.testing.assert_array_equal(slots["fixed_template_indices"], fixed)
            template_order = np.lexsort((human_points[:, 0], human_points[:, 2], human_points[:, 1]))
            np.testing.assert_array_equal(slots["template_sort_index"], template_order)
            if fixed:
                np.testing.assert_array_equal(residual[0], np.zeros((12, 3)))
                np.testing.assert_allclose(reconstructed[0], human_points[template_order], atol=1e-7)
            results.append({"batch_size": batch, "mode": mode, "slot_shape": list(reconstructed.shape),
                            "trained": train, "fixed": fixed, "geodesic_edges": len(edges)})
    return results


def check_umr_character(root, directory):
    import mujoco
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    xml = verified(root, UMR_CHARACTER_MODEL)
    record = pickle.loads(verified(root, UMR_CHARACTER_MOTION))
    adapter = verified(root, UMR_CHARACTER)
    package = directory / "_motion_probe_mimickit"
    package.mkdir()
    (package / "__init__.py").touch()
    for spec in UMR_CHARACTER_COMPAT:
        (package / Path(spec[0]).name).write_bytes(verified(root, spec))
    with patch.object(sys, "path", [str(directory), *sys.path]):
        character_model = importlib.import_module(package.name + ".mjcf_char_model")
        torch_util = importlib.import_module(package.name + ".torch_util")

    expected_names = {"slice_frames", "load_mimickit_char_model", "motion_body_transforms"}
    namespace = {"np": np, "torch": torch, "mjcf_char_model": character_model, "torch_util": torch_util,
                 "resolve_repo_path": Path}
    run_declarations(executable_nodes(adapter, ast.FunctionDef, expected_names), namespace)
    model_path = directory / "mimickit_humanoid.xml"
    model_path.write_bytes(xml)
    frames = np.asarray(record["frames"], dtype=float)
    assert frames.shape == (78, 34) and record["fps"] == 60
    selected = namespace["slice_frames"](len(frames), 12, -1, 2, 8)
    np.testing.assert_array_equal(selected, [12, 14, 16, 18, 20, 22, 24, 26])
    transforms = namespace["motion_body_transforms"](record, np.arange(len(frames)), model_path)

    def check_model(model):
        data = mujoco.MjData(model)
        bodies = [model.body(name).id for name in transforms["body_names"]]
        positions, rotations = [], []
        for frame in frames:
            data.qpos[:3] = frame[:3]
            data.qpos[3:7] = Rotation.from_rotvec(frame[3:6]).as_quat()[[3, 0, 1, 2]]
            offset = 6
            for joint in range(1, model.njnt):
                start = model.jnt_qposadr[joint]
                if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL:
                    quat = Rotation.from_rotvec(frame[offset:offset + 3]).as_quat()
                    data.qpos[start:start + 4] = quat[[3, 0, 1, 2]]
                    offset += 3
                else:
                    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
                    data.qpos[start] = frame[offset]
                    offset += 1
            assert offset == len(frame)
            mujoco.mj_forward(model, data)
            positions.append(data.xpos[bodies].copy())
            rotations.append(data.xquat[bodies][:, [1, 2, 3, 0]])
        errors = np.linalg.norm(positions - transforms["body_pos"], axis=-1)
        frame, body = np.unravel_index(errors.argmax(), errors.shape)
        decoded = transforms["body_rot_xyzw"].astype(float)
        decoded /= np.linalg.norm(decoded, axis=-1, keepdims=True)
        rotations = np.asarray(rotations)
        quaternion_errors = np.minimum(np.linalg.norm(rotations - decoded, axis=-1),
                                       np.linalg.norm(rotations + decoded, axis=-1))
        return {"max_position_error_m": float(errors[frame, body]), "frame": int(frame),
                "body": transforms["body_names"][body], "max_quaternion_difference": float(quaternion_errors.max())}

    literal = check_model(mujoco.MjModel.from_xml_string(xml.decode("utf-8")))
    skeleton = ElementTree.fromstring(xml)
    skeleton.remove(skeleton.find("actuator"))
    for body in skeleton.findall("worldbody//body"):
        joints = body.findall("joint")
        if len(joints) == 3:
            for joint in joints:
                body.remove(joint)
            ElementTree.SubElement(body, "joint", name=body.attrib["name"] + "_ball", type="ball", limited="false")
    converted = check_model(mujoco.MjModel.from_xml_string(ElementTree.tostring(skeleton, encoding="unicode")))
    assert literal["max_position_error_m"] > 0.50 and literal["body"] == "right_foot" and literal["frame"] == 50
    assert converted["max_position_error_m"] < 1e-6 and converted["max_quaternion_difference"] < 1e-6, converted
    return {"frames": len(frames), "source_fps": record["fps"], "selected_source_frames": selected.tolist(),
            "decimated_fps": record["fps"] / 2, "literal_hinges": literal, "rotation_vector_groups": converted}


def check_ot(root):
    import torch
    from scipy.optimize import minimize

    source = verified(root, OT_SOURCE)
    namespace = {"__name__": "pinned_otretarget"}
    exec(compile(source, OT_SOURCE[0], "exec"), namespace)
    joint_retarget = namespace["joint_retarget"]
    correspondence = namespace["surface_correspondence"]
    sinkhorn = namespace["sinkhorn"]

    human = torch.tensor([[-1, 0, 0], [0, 0, 0.3], [1, 0, 0]], dtype=torch.float64)
    robot = torch.tensor([[-0.8, 0, 0], [0.8, 0, 0]], dtype=human.dtype)
    indices, plan = correspondence(human, robot, return_plan=True)
    assert indices.tolist() == [0, 2] and plan.shape == (3, 2)
    torch.testing.assert_close(plan.sum(dim=-1), plan.new_full((3,), 1 / 3))
    torch.testing.assert_close(plan.sum(dim=-2), plan.new_full((2,), 1 / 2))
    torch.testing.assert_close(correspondence(human * 2 + 5, robot * 3 - 1, return_plan=True)[1], plan)
    turned = robot * robot.new_tensor([-1, -1, 1])
    assert correspondence(human, turned).tolist() == [2, 0]
    costs = human.new_tensor([[0.0, 1.0], [1.0, 0.0]])
    assert math.isclose(sinkhorn(costs).sum().item(), 1.0)
    assert sinkhorn(costs + 1000).count_nonzero().item() == 0

    config = torch.tensor([[1.0]], dtype=torch.float64)
    poses = torch.tensor([[[0.0, 0.0, 1.35, 1.0, 0.0, 0.0, 0.0]]], dtype=config.dtype)

    def style(_config, objects):
        return 0.5 * (objects[..., 2] - 1.35).square().sum()

    def interaction(robot, objects):
        return (robot[..., 0] - objects[..., 2]).square().sum()

    def solve(objective, initial_config, initial_poses, constraints, *, iters):
        def value_gradient(array):
            heights = torch.tensor(array, dtype=config.dtype, requires_grad=True)
            robot = heights[0].reshape_as(initial_config)
            objects = initial_poses.clone()
            objects[..., 2] = heights[1]
            value = objective(robot, objects)
            gradient, = torch.autograd.grad(value, heights)
            return value.detach().item(), gradient.detach().numpy()

        start = [initial_config.item(), initial_poses[..., 2].item()]
        result = minimize(value_gradient, start, jac=True, method="SLSQP", bounds=constraints,
                          options={"maxiter": iters, "ftol": 1e-14})
        assert result.success, result.message
        return result.x

    initial = joint_retarget(config, poses, style, interaction)
    assert math.isclose(initial.item(), 0.1225)
    results = {"transport": {"shape": list(plan.shape), "source_indices": indices.tolist(),
                             "large_additive_shift_underflows": True}}
    for label, bounds, expected_box in (
        ("fixed_box", (1.35, 1.35), 1.35),
        ("movable_box", (1.10, 1.40), 1.25),
        ("box_above_support", (1.30, 1.40), 1.30),
    ):
        hand, box = joint_retarget(config, poses, style, interaction, solver_fn=solve,
                                   constraints=[(0.0, 1.20), bounds], iters=30)
        assert math.isclose(hand, 1.20, abs_tol=1e-8) and math.isclose(box, expected_box, abs_tol=1e-8)
        results[label] = {"hand_m": float(hand), "box_m": float(box), "gap_m": float(box - hand)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr", type=Path, help="Root of the fixed GMR checkout")
    parser.add_argument("--umr", type=Path, help="Root of the fixed UMR checkout")
    parser.add_argument("--ot", type=Path, help="Root of the fixed lucidrains OTRetarget checkout")
    parser.add_argument("--check-umr-training", action="store_true", help="With --umr, run two small CPU fits with PyTorch")
    parser.add_argument("--check-umr-character", action="store_true", help="With --umr, verify the bundled Character input and FK")
    args = parser.parse_args()
    if args.gmr is None and args.umr is None and args.ot is None:
        parser.error("Provide at least one of --gmr, --umr or --ot")
    if (args.check_umr_training or args.check_umr_character) and args.umr is None:
        parser.error("The UMR training and Character checks require --umr")
    result = {}
    try:
        with TemporaryDirectory(prefix="retargeting-probe-") as temporary:
            directory = Path(temporary)
            if args.gmr is not None:
                result["gmr"] = check_gmr(args.gmr, directory)
            if args.umr is not None:
                result["umr"] = check_umr(args.umr, directory)
                if args.check_umr_training:
                    result["umr"]["training"] = check_umr_training(args.umr, directory)
                if args.check_umr_character:
                    result["umr"]["character"] = check_umr_character(args.umr, directory)
            if args.ot is not None:
                result["otretarget"] = check_ot(args.ot)
    except subprocess.CalledProcessError as error:
        parser.error(error.stderr.strip() or str(error))
    except subprocess.TimeoutExpired:
        parser.error("UMR synthetic training did not finish in 45 seconds")
    except (OSError, ImportError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
