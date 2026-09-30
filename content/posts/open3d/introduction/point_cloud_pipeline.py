"""Small Open3D 0.19.0 preprocessing example: metres, CPU, no GUI or downloads.

Run with NumPy and either open3d or open3d-cpu. Temporary PLY files are removed.
The synthetic surface has known isolated outliers; it is not a sensor benchmark.
"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    x, y = np.meshgrid(np.linspace(-.3, .3, 31), np.linspace(-.2, .2, 21))
    surface = np.column_stack((x.ravel(), y.ravel(),
                               (.05*np.sin(3*x) + .1*y*y).ravel()))
    isolated = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    invalid = np.array([[np.nan, 0., 0.], [0., np.inf, 0.]])
    raw = np.vstack((surface, isolated, invalid))
    finite = raw[np.isfinite(raw).all(axis=1)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(finite))

    # PLY stores coordinates; the convention "metres" is supplied by this example.
    with tempfile.TemporaryDirectory(prefix="open3d-pipeline-") as directory:
        filename = Path(directory)/"cloud.ply"
        if not o3d.io.write_point_cloud(str(filename), cloud):
            raise RuntimeError("Could not write point cloud")
        loaded = o3d.io.read_point_cloud(str(filename))
        if loaded.is_empty():
            raise RuntimeError("Read an empty point cloud")
        np.testing.assert_allclose(np.asarray(loaded.points), finite, atol=1e-12)

    down = loaded.voxel_down_sample(voxel_size=.04)
    clean, indices = down.remove_radius_outlier(nb_points=3, radius=.065)
    # These indices refer to `down`, not raw or loaded.
    np.testing.assert_allclose(np.asarray(clean.points),
                               np.asarray(down.points)[indices])
    assert len(indices) > 0
    assert len(down.points)-len(clean.points) == len(isolated)
    assert np.max(np.linalg.norm(np.asarray(clean.points), axis=1)) < .4

    clean.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.12, max_nn=30))
    normals = np.asarray(clean.normals)
    assert normals.shape == np.asarray(clean.points).shape
    assert np.isfinite(normals).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-12)
    # Unit normals need not have consistent signs; Poisson reconstruction needs
    # an additional, suitable orientation step for its own input geometry.

    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0., 0., .2))
    transform[:3, 3] = [.15, -.05, .02]
    original = np.asarray(clean.points).copy()
    moved = copy.deepcopy(clean).transform(transform)
    expected = original @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(np.asarray(moved.points), expected, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(clean.points), original)
    moved.transform(np.linalg.inv(transform))
    np.testing.assert_allclose(np.asarray(moved.points), original, atol=1e-12)

    report = {
        "open3d": o3d.__version__, "units": "m",
        "points": {"raw": len(raw), "finite": len(finite),
                   "voxel": len(down.points), "clean": len(clean.points)},
        "voxel_size_m": .04, "outlier_radius_m": .065,
        "isolated_outliers_removed": len(down.points)-len(clean.points),
        "source_unchanged": True, "transform_round_trip": True,
        "scope": "Known synthetic surface; counts do not measure real-sensor accuracy.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
