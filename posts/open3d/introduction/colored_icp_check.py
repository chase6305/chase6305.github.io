"""Compare geometry and texture on a synthetic plane, Open3D 0.19.0 CPU.

No camera, downloads or GUI. NumPy and Open3D are required; --plot adds Matplotlib.
Exact normals and a noiseless transformed copy isolate the role of texture.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def pose_error(estimate, truth):
    relative = estimate[:3, :3].T @ truth[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))
    return {
        "translation_error_m": float(np.linalg.norm(estimate[:3, 3]-truth[:3, 3])),
        "rotation_error_rad": float(angle),
        "matrix_max_error": float(np.max(np.abs(estimate-truth))),
    }


def colored(source, target):
    reg = o3d.pipelines.registration
    estimate = np.eye(4)
    for voxel in (.04, .02, .01):
        a, b = source.voxel_down_sample(voxel), target.voxel_down_sample(voxel)
        result = reg.registration_colored_icp(
            a, b, max(2*voxel, .025), estimate,
            reg.TransformationEstimationForColoredICP(lambda_geometric=.968),
            reg.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8,
                                       max_iteration=100),
        )
        estimate = result.transformation
    return estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("colored-icp-results.json"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    x, y = np.meshgrid(np.linspace(-.5, .5, 61), np.linspace(-.4, .4, 49))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    colors = np.column_stack((.5+.4*np.sin(5*x.ravel()+2*y.ravel()),
                              .5+.4*np.sin(-2*x.ravel()+6*y.ravel()),
                              .5+.4*np.cos(3*x.ravel()-y.ravel())))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    source.colors = o3d.utility.Vector3dVector(colors)
    source.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(points), 1)))
    truth = np.eye(4)
    truth[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0., 0., .08))
    truth[:3, 3] = [.05, -.03, 0.]
    target = copy.deepcopy(source).transform(truth)
    reg = o3d.pipelines.registration
    geometry = reg.registration_icp(
        source, target, .1, np.eye(4), reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=100),
    ).transformation
    textured = colored(source, target)
    plain_source, plain_target = copy.deepcopy(source), copy.deepcopy(target)
    plain_source.paint_uniform_color([.2, .7, .3])
    plain_target.paint_uniform_color([.2, .7, .3])
    uniform = colored(plain_source, plain_target)

    np.testing.assert_allclose(geometry, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(uniform, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(textured, truth, atol=1e-8)
    np.testing.assert_array_equal(np.asarray(source.points), points)
    results = {name: pose_error(estimate, truth) for name, estimate in
               (("point_to_plane", geometry), ("uniform_colored", uniform),
                ("textured_colored", textured))}
    report = {
        "open3d": o3d.__version__, "points": len(points), "units": "m, rad",
        "truth": truth.tolist(), "initial_transform": np.eye(4).tolist(),
        "voxel_sizes_m": [.04, .02, .01], "lambda_geometric": .968,
        "results": results,
        "scope": "Noiseless plane with exact normals and copied texture; not a real-scan benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.facecolor": "white"})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        target_points = np.asarray(target.points)
        axes[0].scatter(target_points[:, 0], target_points[:, 1], c=colors, s=3)
        axes[0].set(title="Known target texture", xlabel="x [m]", ylabel="y [m]", aspect="equal")
        labels = ["Geometry", "Uniform\ncolor", "Texture"]
        for axis, key, scale, title in [
            (axes[1], "translation_error_m", 1000., "Translation error [mm]"),
            (axes[2], "rotation_error_rad", 180/np.pi, "Rotation error [deg]"),
        ]:
            values = [entry[key]*scale for entry in results.values()]
            axis.bar(labels, values, color=["#88b8e8", "#bfa6dc", "#83c6a2"])
            axis.set(title=title, ylim=(0, max(values)*1.3))
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
            for index, value in enumerate(values):
                label = "< 0.00001" if value < 1e-5 else f"{value:.2f}"
                axis.text(index, value+max(values)*.045, label, ha="center", fontsize=10)
        fig.suptitle("Same plane and initial pose: texture supplies missing in-plane constraints")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
