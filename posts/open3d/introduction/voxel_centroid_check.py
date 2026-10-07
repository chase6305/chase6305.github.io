"""Compare sample centroids and occupied cell centers on the same fixed grid.

Open3D 0.19.0, NumPy; optional --plot requires Matplotlib. Units are meters.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    points = np.array([[.1,.1,.1], [.2,.1,.1], [.1,.3,.2],
                       [1.1,.2,.1], [1.8,.7,.2]])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    lower, upper, size = np.zeros(3), np.array([3., 2., 1.]), 1.
    grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size=size, min_bound=lower, max_bound=upper)
    down, _, trace = pcd.voxel_down_sample_and_trace(size, lower, upper)
    occupied = sorted(tuple(v.grid_index) for v in grid.get_voxels())
    assert occupied == [(0,0,0), (1,0,0)]
    np.testing.assert_allclose(grid.origin, lower)
    centroids = {}
    for position, indexes in zip(np.asarray(down.points), trace):
        original = points[np.asarray(indexes, dtype=int)]
        np.testing.assert_allclose(position, original.mean(axis=0))
        cells = np.floor((original-lower)/size).astype(int)
        assert np.all(cells == cells[0])
        centroids[tuple(cells[0])] = position
    rows = []
    for cell in occupied:
        center = grid.get_voxel_center_coordinate(np.asarray(cell))
        np.testing.assert_allclose(center, lower+(np.array(cell)+.5)*size)
        rows.append({"grid_index": list(map(int, cell)), "center_m": center.tolist(),
                     "sample_centroid_m": centroids[cell].tolist()})
    report = {"open3d": o3d.__version__, "input_points": len(points),
              "voxel_size_m": size, "origin_m": grid.origin.tolist(),
              "occupied_cells": rows,
              "scope": "Point occupancy only; unobserved cells are not certified free space."}
    print(json.dumps(report, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for cell in occupied:
            ax.add_patch(Rectangle(cell[:2], 1, 1, facecolor="#e6f1fa", edgecolor="#83aacb"))
        centers = np.array([row["center_m"] for row in rows])
        averages = np.array([row["sample_centroid_m"] for row in rows])
        ax.scatter(points[:,0], points[:,1], s=50, color="#315e8a", label="Input points")
        ax.scatter(averages[:,0], averages[:,1], s=140, marker="*", color="#da812c", label="Sample centroid")
        ax.scatter(centers[:,0], centers[:,1], s=80, marker="x", linewidths=2,
                   color="#925ab3", label="Occupied cell center")
        ax.text(2.5, .5, "No observed point\n(not proof of free space)", ha="center", va="center", fontsize=9)
        ax.set(xlim=(-.1,3.1), ylim=(-.1,1.15), aspect="equal", xlabel="x (m)", ylabel="y (m)",
               title="Same 1 m grid: sample centroid and cell center differ\nXY projection; every sample lies in the z = [0, 1) layer")
        ax.set_xticks([0,1,2,3]);ax.set_yticks([0,.5,1]);ax.grid(alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5,-.22), ncol=3)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot,dpi=170);plt.close(fig)


if __name__ == "__main__":
    main()
