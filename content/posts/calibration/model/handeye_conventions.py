"""Synthetic hand-eye conventions and translation observability.

NumPy and OpenCV with calibrateHandEye; no camera, robot or image downloads.
Known transforms produce exact observations; this checks algebra, not accuracy
under camera noise, visibility, time misalignment or robot model error.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def transform(rotvec, translation):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    result[:3, 3] = translation
    return result


def inverse(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def solve(left, right):
    rotation, translation = cv2.calibrateHandEye(
        [p[:3, :3] for p in left], [p[:3, 3] for p in left],
        [p[:3, :3] for p in right], [p[:3, 3] for p in right],
        method=cv2.CALIB_HAND_EYE_PARK)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation.ravel()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-10)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('handeye-results.json'))
    args = parser.parse_args()
    if not hasattr(cv2, 'calibrateHandEye'):
        raise RuntimeError(f'OpenCV {cv2.__version__} does not expose calibrateHandEye')
    rng = np.random.default_rng(20260928)
    robot = [transform(rng.normal(0, .55, 3), rng.uniform(-.3, .3, 3)) for _ in range(24)]
    train = 18
    camera_in_hand = transform([.2, -.3, .1], [.04, -.02, .12])  # g <- c
    board_in_base = transform([-.1, .2, .3], [.5, .1, .8])     # b <- t
    images_in_hand = [inverse(camera_in_hand) @ inverse(g) @ board_in_base for g in robot]
    in_hand = solve(robot[:train], images_in_hand[:train])
    np.testing.assert_allclose(in_hand, camera_in_hand, atol=1e-10)
    in_residual = max(np.abs(g @ in_hand @ c-board_in_base).max()
                      for g, c in zip(robot[train:], images_in_hand[train:]))

    camera_in_base = transform([.15, .4, -.2], [.45, -.25, .7])  # b <- c
    board_in_hand = transform([.1, -.15, .2], [.02, .01, .06])   # g <- t
    images_to_hand = [inverse(camera_in_base) @ g @ board_in_hand for g in robot]
    direct = solve([inverse(g) for g in robot[:train]], images_to_hand[:train])
    board_first = solve(robot[:train], [inverse(c) for c in images_to_hand[:train]])
    via_board = robot[0] @ board_first @ inverse(images_to_hand[0])
    np.testing.assert_allclose(direct, camera_in_base, atol=1e-10)
    np.testing.assert_allclose(board_first, board_in_hand, atol=1e-10)
    np.testing.assert_allclose(via_board, direct, atol=1e-10)
    out_residual = max(np.abs(g @ board_first-direct @ c).max()
                       for g, c in zip(robot[train:], images_to_hand[train:]))

    regimes = []
    translations = rng.uniform(-.3, .3, (18, 3))
    for name in ('translation_only', 'single_z_rotation_axis', 'multiple_rotation_axes'):
        rotations = np.zeros((18, 3))
        if name == 'single_z_rotation_axis': rotations[:, 2] = np.linspace(-.8, .8, 18)
        if name == 'multiple_rotation_axes': rotations = rng.normal(0, .55, (18, 3))
        motions = [transform(r, t) for r, t in zip(rotations, translations)]
        relative = [inverse(motions[0]) @ g for g in motions[1:]]
        design = np.vstack([a[:3, :3]-np.eye(3) for a in relative])
        rank = int(np.linalg.matrix_rank(design, tol=1e-10))
        assert rank == {'translation_only': 0, 'single_z_rotation_axis': 2, 'multiple_rotation_axes': 3}[name]
        # A translation along the shared z axis commutes with the first two regimes.
        gauge = transform([0., 0., 0.], [0., 0., .1])
        wrong_handeye, shifted_board = gauge @ camera_in_hand, gauge @ board_in_base
        observations = [inverse(camera_in_hand) @ inverse(g) @ board_in_base for g in motions]
        closure = max(np.linalg.norm((g @ wrong_handeye @ c)[:3, 3]-shifted_board[:3, 3])
                      for g, c in zip(motions, observations))
        if rank < 3: assert closure < 1e-12
        else: assert closure > .01
        regimes.append({'motion': name, 'translation_design_rank': rank,
                        'singular_values': np.linalg.svd(design, compute_uv=False).tolist(),
                        'wrong_handeye_translation_m': .1,
                        'wrong_candidate_max_closure_error_m': float(closure)})
    report = {'opencv': cv2.__version__, 'numpy': np.__version__, 'seed': 20260928,
              'training_poses': train, 'held_out_poses': len(robot)-train,
              'eye_in_hand_transform_max_error': float(np.abs(in_hand-camera_in_hand).max()),
              'eye_in_hand_held_out_closure_matrix_max_error': float(in_residual),
              'eye_to_hand_direct_transform_max_error': float(np.abs(direct-camera_in_base).max()),
              'eye_to_hand_two_routes_max_difference': float(np.abs(via_board-direct).max()),
              'eye_to_hand_held_out_closure_matrix_max_error': float(out_residual),
              'degeneracy': regimes,
              'scope': 'Exact transforms only; rank describes the translation subsystem after fixing hand-eye rotation, not the entire joint calibration problem. No image visibility or measured-noise model.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
