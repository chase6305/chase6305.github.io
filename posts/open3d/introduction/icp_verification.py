"""Open3D legacy ICP: known transform, direction, empty matches and mutation.

No GUI, downloads or sensor data. Run with NumPy and Open3D 0.19.0.
"""
import copy
import json

import numpy as np
import open3d as o3d


def usable(result, min_fitness, max_rmse, min_correspondences=3):
    return (np.isfinite(result.transformation).all()
            and np.isfinite(result.fitness) and np.isfinite(result.inlier_rmse)
            and len(result.correspondence_set) >= min_correspondences
            and result.fitness >= min_fitness
            and result.inlier_rmse <= max_rmse)


def main():
    rng = np.random.default_rng(42)
    points = rng.uniform([-.6, -.3, -.1], [.8, .5, .3], (600, 3))
    points[:, 2] += .2 * points[:, 0] ** 2
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    truth = np.eye(4)
    truth[:3, :3] = source.get_rotation_matrix_from_xyz((.04, -.03, .05))
    truth[:3, 3] = [.025, -.015, .02]
    target = copy.deepcopy(source).transform(truth)
    registration = o3d.pipelines.registration
    result = registration.registration_icp(
        source, target, .12, np.eye(4),
        registration.TransformationEstimationPointToPoint(),
        registration.ICPConvergenceCriteria(max_iteration=80),
    )
    assert usable(result, min_fitness=.99, max_rmse=1e-8)
    # ICP(source, target) returns source -> target, not its inverse.
    np.testing.assert_allclose(result.transformation, truth, atol=1e-8)
    np.testing.assert_allclose(np.asarray(source.points), points, atol=1e-14)
    aligned = copy.deepcopy(source).transform(result.transformation)
    np.testing.assert_allclose(np.asarray(aligned.points), np.asarray(target.points), atol=1e-8)
    rotation_delta = result.transformation[:3, :3].T @ truth[:3, :3]
    rotation_error = np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1))
    translation_error = np.linalg.norm(result.transformation[:3, 3] - truth[:3, 3])

    # Zero correspondences can report RMSE=0. Check coverage as well as RMSE.
    far = copy.deepcopy(target).translate([10, 0, 0])
    failed = registration.registration_icp(
        source, far, .01, np.eye(4), registration.TransformationEstimationPointToPoint(),
    )
    assert len(failed.correspondence_set) == 0
    assert not usable(failed, min_fitness=.99, max_rmse=1e-8)

    changed = copy.deepcopy(source).transform(truth)
    before_identity = np.asarray(changed.points).copy()
    changed.transform(np.eye(4))
    np.testing.assert_array_equal(np.asarray(changed.points), before_identity)
    assert not np.allclose(np.asarray(changed.points), points)
    criteria = registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=.999)
    assert criteria.confidence == .999
    print(json.dumps({
        "open3d": o3d.__version__, "seed": 42, "points": len(points),
        "fitness": result.fitness, "inlier_rmse_m": result.inlier_rmse,
        "translation_error_m": float(translation_error), "rotation_error_rad": float(rotation_error),
        "empty_match_fitness": failed.fitness, "empty_match_rmse": failed.inlier_rmse,
        "empty_matches_rejected": True, "source_unchanged": True,
        "identity_does_not_undo_transform": True, "ransac_confidence": criteria.confidence,
    }, indent=2))


if __name__ == "__main__":
    main()
