#!/usr/bin/env python3
"""Check selected functions from two fixed public source revisions offline."""

from __future__ import annotations

import argparse
import ast
import contextlib
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace


# Holosoma c9882fcefe195d8ec0549bd44c2f2853f8084735
HOLOSOMA = {
    "src/holosoma_retargeting/holosoma_retargeting/src/utils.py":
        "c271bae52aae9b7027b321334ee218450366b6b61a798e7dd1c2abdf9b320538",
    "src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py":
        "aa51b4d99c47f13c4cb11d6c9de9a82e57dfff02acbd3397c59957cc403a67bc",
}
# HOI-Retarget b341dadde3fc745ef5ea606eaacbde9a0cbb916f
HOI = {
    "hoi_retarget/config.py": "53b7930e0d53d3e6b706089d219d43fcff9c44671733586d505ba80f69232b9b",
    "hoi_retarget/optimization/window.py": "513741ce79ad55803db1d8014d8738ea10006be6d95055a8a9d22579ec773dec",
}


def verified_sources(root: Path, expected: dict[str, str]) -> dict[str, ast.Module]:
    result = {}
    for name, digest in expected.items():
        path = root / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {path}")
        result[path.name] = ast.parse(raw, filename=str(path))
    return result


def selected_functions(body, names, bindings):
    functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError("Expected functions are missing")
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<verified source functions>", "exec"), bindings)
    return SimpleNamespace(**{node.name: bindings[node.name] for node in functions})


def check_holosoma(sources):
    import numpy as np
    from scipy.spatial import Delaunay
    import trimesh

    utils = selected_functions(sources["utils.py"].body, {
        "extract_foot_sticking_sequence_velocity", "create_interaction_mesh", "get_adjacency_list",
        "calculate_laplacian_coordinates", "calculate_laplacian_matrix",
        "transform_points_local_to_world", "transform_points_world_to_local",
    }, {"np": np, "Delaunay": Delaunay, "trimesh": trimesh})
    motion = selected_functions(sources["robot_retarget.py"].body, {
        "convert_object_poses_to_mujoco_order",
    }, {"np": np})

    stance = utils.extract_foot_sticking_sequence_velocity(
        np.array([[[0, 0, 0], [0, 0, 0]], [[0.005, 0, 0.20], [0.015, 0, 0]]]),
        ["L_Toe", "R_Toe"], ["L_Toe", "R_Toe"],
    )
    assert stance == [{"L_Toe": False, "R_Toe": False}, {"L_Toe": True, "R_Toe": False}]

    tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vertices, simplices = utils.create_interaction_mesh(tetra)
    adjacency = utils.get_adjacency_list(simplices, len(vertices))
    assert list(map(set, adjacency)) == [set(range(4)) - {i} for i in range(4)]
    # A single tetrahedron is a complete graph, so each residual is 4/3 times its centered point.
    expected = (4 / 3) * (tetra - tetra.mean(axis=0))
    np.testing.assert_allclose(utils.calculate_laplacian_coordinates(vertices, adjacency), expected, atol=1e-12)
    np.testing.assert_allclose(utils.calculate_laplacian_matrix(vertices, adjacency) @ vertices, expected, atol=1e-12)
    np.testing.assert_allclose(utils.calculate_laplacian_coordinates(vertices + [2, -1, 3], adjacency), expected, atol=1e-12)

    half = 2**-0.5
    pose = np.array([[half, 0., 0., half, 1., 0.5, 0.]])
    reordered = motion.convert_object_poses_to_mujoco_order(pose)
    np.testing.assert_array_equal(reordered, [[1., 0.5, 0., half, 0., 0., half]])
    point = np.array([[0.2, 0.1, 0.]])
    rotation, translation = reordered[0, 3:], reordered[0, :3]
    world = utils.transform_points_local_to_world(rotation, translation, point)
    np.testing.assert_allclose(world, [[0.9, 0.7, 0.]], atol=1e-12)
    np.testing.assert_allclose(utils.transform_points_world_to_local(rotation, translation, world), point, atol=1e-12)
    return {
        "stance_two_frames": [{name: bool(active) for name, active in row.items()} for row in stance],
        "tetra_vertices": len(vertices),
        "rotated_anchor": world.round(12).tolist()[0],
    }


def check_hoi(sources, include_solver=False):
    projector = next(node for node in sources["window.py"].body
                     if isinstance(node, ast.ClassDef) and node.name == "PinocchioMotionProjectorWindow")
    unified = selected_functions(projector.body, {"_unified_pass"}, {})._unified_pass
    window_class = next(node for node in sources["config.py"].body
                        if isinstance(node, ast.ClassDef) and node.name == "WindowConfig")
    values = {node.target.id: ast.literal_eval(node.value) for node in window_class.body if isinstance(node, ast.AnnAssign)}
    cfg = SimpleNamespace(window=SimpleNamespace(**values))

    def run(frames, failures):
        calls = []

        def solve(prep, start, end, accumulated):
            pin = 0 if start == 0 else cfg.window.pin_frames
            calls.append((start, end, accumulated["q"][start:start + pin]))
            if len(calls) in failures:
                return None
            return {"q": [-1] * pin + [1000 * len(calls) + i for i in range(start + pin, end)]}

        prep = SimpleNamespace(N=frames, obj_pos=object(), obj_rot=object(), v_ref=object())
        stand_in = SimpleNamespace(cfg=cfg, _build_initial_guess=lambda p: {"q": list(range(p.N))}, _solve_window=solve)
        with contextlib.redirect_stdout(io.StringIO()):
            result = unified(stand_in, prep)
        if result is not None:
            assert result["obj_pos"] is prep.obj_pos and result["obj_rot"] is prep.obj_rot
        return result, calls

    trajectory, tail_calls = run(118, set())
    assert tail_calls == [(0, 60, []), (57, 117, [1057, 1058, 1059]), (58, 118, [1058, 1059, 2060])]
    assert trajectory["q"][57:63] == [1057, 1058, 1059, 2060, 3061, 3062]
    assert trajectory["q"][-1] == 3117

    partial, failed_calls = run(150, {2})
    assert failed_calls == [(0, 60, []), (57, 117, [1057, 1058, 1059]), (114, 150, [114, 115, 116])]
    assert partial["q"][60:117] == list(range(60, 117)) and partial["q"][117] == 3117
    assert run(118, {1, 2, 3})[0] is None

    failed_tail, _ = run(118, {3})
    assert failed_tail["q"][60:117] == [2000 + i for i in range(60, 117)]
    assert failed_tail["q"][117] == 117
    assert run(5, set())[1] == [(0, 5, [])]
    try:
        run(4, set())
    except ValueError as error:
        assert "No window to solve" in str(error)
    else:
        raise AssertionError("A four-frame clip cannot start a window with these defaults")

    residual_node, = [node for node in ast.walk(projector) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "da" for target in node.targets)]
    residual = compile(ast.Expression(residual_node.value), "<verified third difference>", "eval")
    value = [eval(residual, {"q_t": 1, "q_tm1": 0, "q_tm2": 0, "q_tm3": 0, "dt": Fraction(1, rate)})
             for rate in (30, 60)]
    assert value == [900, 3600]
    result = {"short_tail": tail_calls, "middle_failure": failed_calls, "tail_failure_retains_prior_values": True,
              "minimum_clip_frames": 5, "third_difference_30_60": list(map(int, value))}
    if include_solver:
        result["slider_solve"] = check_hoi_solver(projector)
    return result


def check_hoi_solver(projector):
    import time

    import casadi as ca
    import numpy as np

    solve = selected_functions(projector.body, {"_solve_window"}, {"ca": ca, "np": np, "time": time})._solve_window

    def ignore(*_args, **_kwargs):
        pass

    analysis = SimpleNamespace(**{name: ignore for name in (
        "start_pass", "add_variable", "add_constraint", "add_cost", "end_pass",
    )})
    robot = SimpleNamespace(
        base_q_start=0, joint_q_indices=[7], joint_v_indices=[6],
        q_lower=[-np.inf] * 7 + [-1.], q_upper=[np.inf] * 7 + [1.], v_abs=[np.inf] * 6 + [1.2],
        fk_world_pos={"hand"}, world_pos=lambda _link, q: ca.vertcat(q[7], 0, 0), fk_rel_pos={},
    )
    config = SimpleNamespace(
        tracking=SimpleNamespace(link_weights_world={}, link_weights_base={}),
        smoothness=SimpleNamespace(base_velocity_limits=False, velocity_reg=0., first_order_reg=0.),
        object=SimpleNamespace(contact_weight=1000., pin_orient=False),
        window=SimpleNamespace(pin_frames=3, terminal_frames=0, ik_tracking_weight=1., terminal_weight=0.,
                               max_wall_time=10., keep_last_iterate_on_failure=False),
        foot_contact=SimpleNamespace(enabled=False),
    )
    subject = SimpleNamespace(cfg=config, robot=robot, _analysis=analysis, window_status=[])
    frames = 7
    initial = np.zeros((frames, 8))
    initial[:, 6] = 1.
    prep = SimpleNamespace(
        dt=1 / 30, nq=8, body_names=["hand"], root_pos=np.zeros((frames, 3)), dof_pos=np.zeros((frames, 1)),
        dof_names=["slider"], dof_idx_map={"slider": 7}, body_contact_flags=[False] * 6 + [True],
        contact_points_object_dm_per_frame=[{}] * 6 + [{"hand": ca.DM([.02, 0, 0])}],
        obj_pos=np.tile([.1, 0, 0], (frames, 1)), R_world_object_dm=[ca.DM.eye(3)] * frames,
        contact_link_names=["hand"], per_link_flags=None,
    )
    with contextlib.redirect_stdout(io.StringIO()) as log:
        solved = solve(subject, prep, 1, 7, {"q": initial})
    assert solved is not None and subject.window_status[0]["status"] == "solved", log.getvalue()
    positions = solved["q"][:, 7]
    # For terminal x near 0.12 m, earlier least-squares positions are x-0.08 and x-0.04.
    terminal = float((1001 * Fraction(12, 100)) / 1003)
    np.testing.assert_allclose(positions, [0, 0, 0, terminal - .08, terminal - .04, terminal], atol=3e-6, rtol=0)
    return {"casadi": ca.__version__, "positions_m": positions.tolist(), "terminal_gap_mm": float(1000 * (.12 - positions[-1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holosoma", type=Path, help="Root of the exact Holosoma source checkout")
    parser.add_argument("--hoi", type=Path, help="Root of the exact HOI-Retarget source checkout")
    parser.add_argument("--check-hoi-solver", action="store_true", help="With --hoi, use CasADi/IPOPT for a synthetic slider solve")
    args = parser.parse_args()
    if args.holosoma is None and args.hoi is None:
        parser.error("Provide --holosoma and/or --hoi")
    if args.check_hoi_solver and args.hoi is None:
        parser.error("--check-hoi-solver requires --hoi")
    result = {}
    try:
        if args.holosoma is not None:
            result["holosoma"] = check_holosoma(verified_sources(args.holosoma, HOLOSOMA))
        if args.hoi is not None:
            result["hoi"] = check_hoi(verified_sources(args.hoi, HOI), args.check_hoi_solver)
    except (OSError, ImportError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
