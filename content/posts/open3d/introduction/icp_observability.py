"""Point-to-plane ICP: zero residual can coexist with an unobserved pose.

NumPy and Open3D 0.19.0, CPU only, synthetic exact normals, no GUI/downloads.
The reported local Hessian is built from this residual, not Open3D's generic
get_information_matrix_from_point_clouds helper.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def cloud(points, normals):
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    result.normals = o3d.utility.Vector3dVector(normals)
    return result


def linearized_rows(points, normals, length_scale=1.):
    # Parameters are [translation / length_scale, rotation], residual is r/length_scale.
    return np.column_stack((normals, np.cross(points, normals)/length_scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('icp-observability.json'))
    args = parser.parse_args()
    reg = o3d.pipelines.registration
    grid = np.arange(-12, 13)*.05
    xy = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    plane = np.column_stack((xy, np.zeros(len(xy))))
    normals = np.tile([0., 0., 1.], (len(plane), 1))
    target_grid = np.arange(-20, 21)*.05
    target_xy = np.stack(np.meshgrid(target_grid, target_grid), axis=-1).reshape(-1, 2)
    target_points = np.column_stack((target_xy, np.zeros(len(target_xy))))
    source = cloud(plane, normals)
    target = cloud(target_points, np.tile([0., 0., 1.], (len(target_points), 1)))
    rows = linearized_rows(plane, normals)
    plane_rank = int(np.linalg.matrix_rank(rows))
    assert plane_rank == 3
    np.testing.assert_allclose(rows[:, [0, 1, 5]], 0, atol=1e-14)
    plane_spectrum = np.linalg.eigvalsh(rows.T @ rows/len(rows))
    generic_information = reg.get_information_matrix_from_point_clouds(source, target, .001, np.eye(4))
    generic_rank = int(np.linalg.matrix_rank(generic_information))
    assert generic_rank == 6  # This helper does not use the point-to-plane normals.
    forward = reg.evaluate_registration(source, target, .001, np.eye(4))
    reverse = reg.evaluate_registration(target, source, .001, np.eye(4))
    assert forward.fitness == 1.
    np.testing.assert_allclose(reverse.fitness, len(plane)/len(target_points), atol=1e-14)
    observations = []
    for x_shift in (0., .1):
        initial = np.eye(4); initial[0, 3] = x_shift
        result = reg.registration_icp(source, target, .08, initial,
                                      reg.TransformationEstimationPointToPlane(),
                                      reg.ICPConvergenceCriteria(max_iteration=30))
        transformed = plane @ result.transformation[:3, :3].T + result.transformation[:3, 3]
        signed_distances = transformed[:, 2]
        assert result.fitness > .999
        assert result.inlier_rmse < 1e-10
        np.testing.assert_allclose(result.transformation, initial, atol=1e-12)
        np.testing.assert_allclose(signed_distances, 0, atol=1e-12)
        observations.append({'initial_x_m': x_shift, 'estimated_x_m': float(result.transformation[0, 3]),
                             'fitness': result.fitness, 'inlier_euclidean_rmse_m': result.inlier_rmse,
                             'point_to_plane_rmse_m': float(np.sqrt(np.mean(signed_distances**2)))})

    # Three perpendicular, spatially extended patches; sample away from intersections.
    rng = np.random.default_rng(20260928)
    corner_points, corner_normals = [], []
    for fixed_axis in range(3):
        patch = np.zeros((200, 3))
        patch[:, [j for j in range(3) if j != fixed_axis]] = rng.uniform(.15, .75, (200, 2))
        normal = np.zeros_like(patch); normal[:, fixed_axis] = 1
        corner_points.append(patch); corner_normals.append(normal)
    corner_points, corner_normals = np.vstack(corner_points), np.vstack(corner_normals)
    corner_rows = linearized_rows(corner_points, corner_normals)
    corner_rank = int(np.linalg.matrix_rank(corner_rows))
    assert corner_rank == 6
    spectrum = np.linalg.eigvalsh(corner_rows.T @ corner_rows / len(corner_rows))
    assert spectrum[0] > 1e-4
    corner = cloud(corner_points, corner_normals)
    initial = np.eye(4)
    initial[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([.01, -.02, .015])
    initial[:3, 3] = [.015, -.01, .012]
    recovered = reg.registration_icp(corner, corner, .08, initial,
                                     reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=60))
    np.testing.assert_allclose(recovered.transformation, np.eye(4), atol=1e-10)

    # Independent finite-difference residual test for the translation-first left update.
    perturb = np.array([.2, -.1, .15, .13, -.21, .12])
    h = 1e-6
    def residual(sign):
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(sign*h*perturb[3:])
        moved = corner_points @ rotation.T + sign*h*perturb[:3]
        return np.sum(corner_normals * (moved-corner_points), axis=1)
    finite_difference = (residual(1)-residual(-1))/(2*h)
    np.testing.assert_allclose(finite_difference, corner_rows @ perturb, atol=1e-9)
    report = {'open3d': o3d.__version__, 'numpy': np.__version__,
              'length_scale_m': 1., 'parameter_order': ['tx/ell', 'ty/ell', 'tz/ell', 'rx', 'ry', 'rz'],
              'plane': {'source_points': len(plane), 'target_points': len(target_points),
                        'local_rank': plane_rank, 'normalized_hessian_eigenvalues': plane_spectrum.tolist(),
                        'generic_information_helper_rank': generic_rank,
                        'forward_fitness_at_1mm': forward.fitness,
                        'reverse_fitness_at_1mm': reverse.fitness,
                        'solutions': observations},
              'corner': {'points': len(corner_points), 'local_rank': corner_rank,
                         'normalized_hessian_eigenvalues': spectrum.tolist(),
                         'recovered_transform_max_error': float(np.max(np.abs(recovered.transformation-np.eye(4))))},
              'residual_directional_derivative_max_error': float(np.max(np.abs(finite_difference-corner_rows@perturb))),
              'scope': 'Noiseless planar samples and exact target normals, local linearized residual geometry. Repeated-grid plane permits different perfect nearest-neighbor alignments. Full local rank does not establish global uniqueness or calibrated covariance.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
