#!/usr/bin/env python3
"""Inspect a complete HOI-Retarget Dataset Viewer row (schema version 2).

With --urdf, use NumPy and Pinocchio to evaluate labelled link-to-anchor
distances. With --object-mesh, also measure unsigned point-to-triangle
distances. The mesh uses the published object frame. Distances are in metres.
With --partner, compare one reciprocal CoRoleHOI row on the same time grid.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def names(row, key):
    values = row[key]
    if (not isinstance(values, list) or not values
            or any(not isinstance(name, str) or not name for name in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{key} must contain unique nonempty names")
    return values


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def matrix(row, key, frames, width):
    values = row[key]
    if (len(values) != frames
            or any(len(sample) != width or not all(map(finite_number, sample))
                   for sample in values)):
        raise ValueError(f"{key} must be a finite {frames}-by-{width} array")
    return values


def contact_targets(row, frames, links):
    flags = row["per_link_contact_flags"]
    anchors = row["fixed_contact_points_per_frame_in_object_frame"]
    if len(flags) != frames or len(anchors) != frames:
        raise ValueError("contact arrays must have n_frames rows")
    targets = []
    for frame, (mask, points) in enumerate(zip(flags, anchors)):
        if len(mask) != len(links) or len(points) != len(links):
            raise ValueError(f"frame {frame}: contact columns must follow contact_link_names")
        active = {}
        for link, flag, point in zip(links, mask, points):
            if type(flag) is not bool or len(point) != 3:
                raise ValueError(f"frame {frame}: expected a Boolean mask and a 3-vector for {link}")
            if flag:
                if not all(map(finite_number, point)):
                    raise ValueError(f"frame {frame}: active anchor is missing for {link}")
                active[link] = point
            elif any(coord is not None for coord in point):
                raise ValueError(f"frame {frame}: expected null coordinates on inactive {link}")
        targets.append(active)
    return targets


def normalized_wxyz(quaternion):
    norm = math.hypot(*quaternion)
    if not math.isclose(norm, 1.0, rel_tol=1e-3):
        raise ValueError("expected a unit wxyz quaternion")
    return [value / norm for value in quaternion]


def summarize(row, urdf=None, mesh=None):
    if row["schema_version"] != 2:
        raise ValueError("expected schema_version=2")
    frames = row["n_frames"]
    fps = row["fps"]
    if type(frames) is not int or frames < 1 or not finite_number(fps) or fps <= 0:
        raise ValueError("expected positive n_frames and fps")
    if not math.isclose(row["duration_s"], frames / fps, rel_tol=1e-6):
        raise ValueError("duration_s disagrees with n_frames / fps")
    joints = names(row, "dof_names")
    links = names(row, "contact_link_names")
    root_pos = matrix(row, "root_pos", frames, 3)
    object_pos = matrix(row, "object_pos", frames, 3)
    root_rot = list(map(normalized_wxyz, matrix(row, "root_rot", frames, 4)))
    object_rot = list(map(normalized_wxyz, matrix(row, "object_rot", frames, 4)))
    joint_pos = matrix(row, "dof_pos", frames, len(joints))
    targets = contact_targets(row, frames, links)
    active_frames = {link: [t for t, frame in enumerate(targets) if link in frame] for link in links}
    result = {
        "clip_id": row["clip_id"],
        "sequence_id": row["sequence_id"],
        "frames": frames,
        "fps": fps,
        "duration_s_t_over_fps": frames / fps,
        "sample_span_s_t_minus_1_over_fps": (frames - 1) / fps,
        "n_dof": len(joints),
        "object_scale": row["object_scale"],
        "scale_object_mesh": row["scale_object_mesh"],
        "scale_object_trajectory": row["scale_object_trajectory"],
        "qc_pass": row["qc_pass"],
        "qc_flags": row["qc_flags"],
        "labelled_frames_by_link": {link: len(indices) for link, indices in active_frames.items()},
    }
    if urdf is not None:
        palms, anchors = link_samples(urdf, joints, links, targets, root_pos, root_rot, joint_pos, object_pos, object_rot)
        result["labelled_link_to_anchor_m"] = {
            link: distance_summary([math.dist(palm, anchor) for palm, anchor in zip(palms[link], anchors[link])])
            for link in links
        }
        if mesh is not None:
            result["unsigned_mesh_distance_m"] = {
                "link": mesh_distances(mesh, palms, active_frames),
                "anchor": mesh_distances(mesh, anchors, active_frames),
                "watertight": bool(mesh.is_watertight),
            }
    return result


def distance_summary(distances):
    return {"mean": mean(distances), "max": max(distances)} if distances else None


def angle_deg(first, second):
    # For unit quaternions, the shorter chord accounts for q and -q.
    difference = math.dist(first, second)
    sum_norm = math.hypot(*(a + b for a, b in zip(first, second)))
    return math.degrees(4 * math.atan2(min(difference, sum_norm), max(difference, sum_norm)))


def pair_summary(reference, partner):
    for key in ("sequence_id", "robot", "object_model_path", "object_scale", "object_aug_scale",
                "scale_object_mesh", "scale_object_trajectory", "n_frames", "fps"):
        if reference[key] != partner[key]:
            raise ValueError(f"pair rows disagree on {key}")
    if any(row["dataset"] != "corolehoi" or row["n_actors"] != 2 for row in (reference, partner)):
        raise ValueError("both rows must be from a two-actor CoRoleHOI event")
    if (reference["clip_id"] == partner["clip_id"]
            or reference["partner_clip_id"] != partner["clip_id"]
            or partner["partner_clip_id"] != reference["clip_id"]):
        raise ValueError("partner_clip_id must point to the other row in both directions")
    if type(reference["pair_composable"]) is not bool or partner["pair_composable"] is not reference["pair_composable"]:
        raise ValueError("expected the same Boolean pair_composable value on both rows")

    offset_cm = reference["partner_z_offset_cm"]
    reverse_cm = partner["partner_z_offset_cm"]
    if (not finite_number(offset_cm) or not finite_number(reverse_cm)
            or not math.isclose(offset_cm, -reverse_cm, rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("expected opposite, finite partner_z_offset_cm values")

    frames = reference["n_frames"]
    first_pos = matrix(reference, "object_pos", frames, 3)
    second_pos = matrix(partner, "object_pos", frames, 3)
    world_offsets = [[a - b for a, b in zip(first, second)] for first, second in zip(first_pos, second_pos)]
    first_rot = map(normalized_wxyz, matrix(reference, "object_rot", frames, 4))
    second_rot = map(normalized_wxyz, matrix(partner, "object_rot", frames, 4))
    height = [0, 0, offset_cm / 100]
    translation = world_offsets[0]
    return {
        "partner_clip_id": partner["clip_id"],
        "dataset_pair_composable": reference["pair_composable"],
        "published_partner_to_reference_z_m": height[2],
        "first_frame_partner_to_reference_xyz_m": translation,
        "object_position_error_m": {
            "unaligned": distance_summary([math.hypot(*d) for d in world_offsets]),
            "after_published_z_shift": distance_summary([math.dist(d, height) for d in world_offsets]),
            "after_first_frame_xyz_shift": distance_summary([math.dist(d, translation) for d in world_offsets]),
        },
        "object_rotation_error_deg": distance_summary([angle_deg(first, second) for first, second in zip(first_rot, second_rot)]),
    }


def link_samples(urdf, joints, links, targets, root_pos, root_rot, joint_pos, object_pos, object_rot):
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data = model.createData()
    if set(joints) != set(model.names[2:]):
        raise ValueError("dof_names must match the URDF's movable joint names")
    joint_slots = []
    for name in joints:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1:
            raise ValueError(f"expected a single position coordinate for {name}")
        joint_slots.append(joint.idx_q)
    frame_ids = {link: model.getFrameId(link, pin.FrameType.BODY) for link in links}
    if any(index >= model.nframes for index in frame_ids.values()):
        raise ValueError("URDF must contain all measured contact links")

    q = pin.neutral(model)
    palms = {link: [] for link in links}
    anchors = {link: [] for link in links}
    for t, active in enumerate(targets):
        if not active:
            continue
        qw, qx, qy, qz = root_rot[t]
        q[:3] = root_pos[t]
        q[3:7] = [qx, qy, qz, qw]  # Pinocchio free-flyer uses xyzw.
        q[joint_slots] = joint_pos[t]
        pin.framesForwardKinematics(model, data, q)
        object_from_world = pin.Quaternion(*object_rot[t]).matrix().T
        for link, anchor in active.items():
            world_link = data.oMf[frame_ids[link]].translation
            palms[link].append(object_from_world @ (world_link - object_pos[t]))
            anchors[link].append(anchor)
    return palms, anchors


def mesh_distances(mesh, samples, active_frames):
    import trimesh

    result = {}
    for link, points in samples.items():
        if not points:
            result[link] = None
            continue
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
        result[link] = distance_summary(distances.tolist())
        distant = [t for t, distance in zip(active_frames[link], distances) if distance >= .020]
        result[link]["frames_at_or_above_20mm"] = len(distant)
        result[link]["frame_indices_at_or_above_20mm"] = distant
    return result


def read_viewer_row(path):
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    rows = payload["rows"]
    if len(rows) != 1 or rows[0]["truncated_cells"]:
        raise ValueError("download exactly one Dataset Viewer row with all cells present")
    return rows[0]["row"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row", type=Path, help="JSON body from Dataset Viewer /rows with length=1")
    parser.add_argument("--partner", type=Path, help="optional reciprocal CoRoleHOI row with length=1")
    parser.add_argument("--urdf", type=Path, help="matching floating-base robot description")
    parser.add_argument("--object-mesh", type=Path, help="optional triangle mesh in the object's local frame")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="uniform factor to apply to --object-mesh")
    args = parser.parse_args()
    if args.object_mesh and not args.urdf:
        parser.error("--object-mesh requires --urdf")
    if not math.isfinite(args.mesh_scale) or args.mesh_scale <= 0:
        parser.error("--mesh-scale must be positive and finite")
    try:
        mesh = None
        if args.object_mesh is not None:
            import trimesh

            mesh = trimesh.load_mesh(args.object_mesh)
            if mesh.is_empty:
                raise ValueError("object mesh must contain triangles")
            mesh.apply_scale(args.mesh_scale)
        row = read_viewer_row(args.row)
        result = summarize(row, args.urdf, mesh)
        if args.partner is not None:
            partner = read_viewer_row(args.partner)
            summarize(partner)
            result["pair"] = pair_summary(row, partner)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
