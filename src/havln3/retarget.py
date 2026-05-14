from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation

from havln3.gltf_utils import (
    VertexNormalizer,
    deform_skinned_primitives,
    identity_quaternions,
    joint_base_name,
    node_global_matrices,
    write_object_config,
)
from havln3.materials import fix_habitat_materials
from havln3.motion import MotionClip, SMPL22_TO_MIXAMO, resample_motion


SMPL22_PARENTS = {
    0: None,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 9,
    14: 9,
    15: 12,
    16: 13,
    17: 14,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
}

SMPL22_CHILDREN: dict[int, list[int]] = {index: [] for index in SMPL22_TO_MIXAMO}
for _child, _parent in SMPL22_PARENTS.items():
    if _parent is not None:
        SMPL22_CHILDREN[_parent].append(_child)


SMPL_LEG_JOINTS = {
    "Left": {"hip": 1, "knee": 4, "ankle": 7, "toe": 10},
    "Right": {"hip": 2, "knee": 5, "ankle": 8, "toe": 11},
}


@dataclass(frozen=True)
class RetargetOptions:
    frames: int = 120
    fps: int = 24
    target_height: float = 1.72
    ground_y: float = -0.2
    rotation_scale: float = 0.65
    lower_leg_rotation_scale: float = 0.18
    foot_rotation_scale: float = 0.35
    include_root_orientation: bool = True
    preserve_root_motion: bool = False
    root_motion_scale: float = 1.0
    snap_to_ground: bool = True
    stabilize_root_yaw: bool = False
    foot_contact_lock: bool = False
    foot_contact_height: float = 0.12
    foot_lock_blend_frames: int = 4
    alpha_cutoff: float = 0.55
    material_roughness: float = 0.88
    detect_texture_alpha: bool = True
    calibrated_leg_ik: bool = True
    prefer_joint_position_ik: bool = False
    solidify_shell: bool = True
    body_shell_thickness: float = 0.018
    hair_shell_thickness: float = 0.006


@dataclass(frozen=True)
class LegCalibration:
    side: str
    hip: int
    knee: int
    ankle: int
    toe: int | None
    parent_node: int | None
    upper_len: float
    lower_len: float
    rest_pole_parent_local: np.ndarray
    rest_upper_local: np.ndarray
    rest_lower_local: np.ndarray
    rest_pole_hip_local: np.ndarray
    rest_pole_knee_local: np.ndarray
    rest_pole_foot_local: np.ndarray | None
    rest_toe_local: np.ndarray | None


@dataclass
class FrameGeometry:
    vertices_by_primitive: list[np.ndarray]
    joints: np.ndarray | None


def _scaled_quat_from_matrix(matrix: np.ndarray, scale: float) -> np.ndarray:
    rotation = Rotation.from_matrix(matrix)
    if scale != 1.0:
        rotation = Rotation.from_rotvec(rotation.as_rotvec() * scale)
    return rotation.as_quat()


def _local_quats_for_frame(
    frame_rot_mats: np.ndarray,
    joint_by_name: dict[str, int],
    *,
    options: RetargetOptions,
) -> np.ndarray:
    local = identity_quaternions(len(joint_by_name))
    for source_index, mixamo_name in SMPL22_TO_MIXAMO.items():
        if source_index == 0 and not options.include_root_orientation:
            continue
        target_index = joint_by_name.get(mixamo_name)
        if target_index is None:
            continue
        scale = options.rotation_scale
        if mixamo_name in {"LeftLeg", "RightLeg"}:
            scale = options.lower_leg_rotation_scale
        elif mixamo_name in {"LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"}:
            scale = options.foot_rotation_scale
        local[target_index] = _scaled_quat_from_matrix(frame_rot_mats[source_index], scale)
    return local


def _rotation_from_matrix(matrix: np.ndarray) -> Rotation:
    rotation = matrix[:3, :3].astype(np.float64)
    norms = np.linalg.norm(rotation, axis=0)
    rotation = rotation / np.maximum(norms, 1e-9)
    return Rotation.from_matrix(rotation)


def _local_rest_rotation(gltf: GLTF2, node_index: int) -> Rotation:
    node = gltf.nodes[node_index]
    if node.matrix is not None:
        matrix = np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
        return _rotation_from_matrix(matrix)
    return Rotation.from_quat(node.rotation or [0.0, 0.0, 0.0, 1.0])


def _source_to_avatar_axes(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 2, 1]]


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _node_parent_indices(gltf: GLTF2) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(gltf.nodes or []):
        for child_index in node.children or []:
            parents[child_index] = parent_index
    return parents


def _projected_unit(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = vector - normal * float(np.dot(vector, normal))
    return _safe_unit(projected)


def _fallback_pole(direction: np.ndarray) -> np.ndarray:
    candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(candidate, direction))) > 0.92:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    pole = _projected_unit(candidate, direction)
    if pole is None:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return pole


def _limb_pole(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    line = ankle - hip
    line_unit = _safe_unit(line)
    if line_unit is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pole = knee - hip - line_unit * float(np.dot(knee - hip, line_unit))
    pole_unit = _safe_unit(pole)
    return pole_unit if pole_unit is not None else _fallback_pole(line_unit)


def _basis_rotation(
    rest_primary: np.ndarray,
    rest_secondary: np.ndarray,
    desired_primary: np.ndarray,
    desired_secondary: np.ndarray,
) -> Rotation:
    rest_primary_unit = _safe_unit(rest_primary)
    desired_primary_unit = _safe_unit(desired_primary)
    if rest_primary_unit is None or desired_primary_unit is None:
        return Rotation.identity()

    rest_secondary_unit = _projected_unit(rest_secondary, rest_primary_unit)
    desired_secondary_unit = _projected_unit(desired_secondary, desired_primary_unit)
    if rest_secondary_unit is None or desired_secondary_unit is None:
        return Rotation.align_vectors([desired_primary_unit], [rest_primary_unit])[0]

    rest_basis = np.column_stack(
        [
            rest_primary_unit,
            rest_secondary_unit,
            np.cross(rest_primary_unit, rest_secondary_unit),
        ]
    )
    desired_basis = np.column_stack(
        [
            desired_primary_unit,
            desired_secondary_unit,
            np.cross(desired_primary_unit, desired_secondary_unit),
        ]
    )
    return Rotation.from_matrix(desired_basis @ rest_basis.T)


def _leg_calibrations(
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    rest_globals: list[np.ndarray],
) -> list[LegCalibration]:
    skin = gltf.skins[0]
    parents = _node_parent_indices(gltf)
    rest_positions = np.stack([rest_globals[joint][:3, 3] for joint in skin.joints])
    rest_rotations = {
        skin_index: _rotation_from_matrix(rest_globals[node_index])
        for skin_index, node_index in enumerate(skin.joints)
    }

    calibrations: list[LegCalibration] = []
    for side in ("Left", "Right"):
        hip = joint_by_name.get(f"{side}UpLeg")
        knee = joint_by_name.get(f"{side}Leg")
        ankle = joint_by_name.get(f"{side}Foot")
        if hip is None or knee is None or ankle is None:
            continue

        toe = joint_by_name.get(f"{side}ToeBase")
        hip_pos = rest_positions[hip]
        knee_pos = rest_positions[knee]
        ankle_pos = rest_positions[ankle]
        pole = _limb_pole(hip_pos, knee_pos, ankle_pos)
        parent_node = parents.get(skin.joints[hip])
        parent_rot = _rotation_from_matrix(rest_globals[parent_node]) if parent_node is not None else Rotation.identity()

        rest_upper = knee_pos - hip_pos
        rest_lower = ankle_pos - knee_pos
        calibrations.append(
            LegCalibration(
                side=side,
                hip=hip,
                knee=knee,
                ankle=ankle,
                toe=toe,
                parent_node=parent_node,
                upper_len=float(np.linalg.norm(rest_upper)),
                lower_len=float(np.linalg.norm(rest_lower)),
                rest_pole_parent_local=parent_rot.inv().apply(pole),
                rest_upper_local=rest_rotations[hip].inv().apply(rest_upper),
                rest_lower_local=rest_rotations[knee].inv().apply(rest_lower),
                rest_pole_hip_local=rest_rotations[hip].inv().apply(pole),
                rest_pole_knee_local=rest_rotations[knee].inv().apply(pole),
                rest_pole_foot_local=rest_rotations[ankle].inv().apply(pole) if toe is not None else None,
                rest_toe_local=(
                    rest_rotations[ankle].inv().apply(rest_positions[toe] - ankle_pos)
                    if toe is not None
                    else None
                ),
            )
        )
    return calibrations


def _two_bone_knee_position(
    hip: np.ndarray,
    direction: np.ndarray,
    pole: np.ndarray,
    *,
    upper_len: float,
    lower_len: float,
    reach_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    chain_len = upper_len + lower_len
    min_reach = abs(upper_len - lower_len) + 1e-5
    max_reach = max(chain_len - 1e-5, min_reach + 1e-5)
    reach = float(np.clip(reach_ratio * chain_len, min_reach, max_reach))
    ankle = hip + direction * reach
    along = (upper_len**2 - lower_len**2 + reach**2) / (2.0 * reach)
    height = float(np.sqrt(max(upper_len**2 - along**2, 0.0)))
    knee = hip + direction * along + pole * height
    return knee, ankle


def _apply_calibrated_leg_ik(
    *,
    local: np.ndarray,
    source_joints: np.ndarray,
    gltf: GLTF2,
    calibrations: list[LegCalibration],
) -> np.ndarray:
    if not calibrations:
        return local

    skin = gltf.skins[0]
    source = _source_to_avatar_axes(source_joints.astype(np.float64))
    current_globals = node_global_matrices(gltf, local)
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index) for skin_index, node_index in enumerate(skin.joints)
    }

    for calibration in calibrations:
        source_map = SMPL_LEG_JOINTS[calibration.side]
        source_hip = source[source_map["hip"]]
        source_knee = source[source_map["knee"]]
        source_ankle = source[source_map["ankle"]]
        source_direction = _safe_unit(source_ankle - source_hip)
        if source_direction is None:
            continue

        source_upper_len = float(np.linalg.norm(source_knee - source_hip))
        source_lower_len = float(np.linalg.norm(source_ankle - source_knee))
        source_chain_len = source_upper_len + source_lower_len
        if source_chain_len < 1e-8:
            continue
        reach_ratio = float(np.linalg.norm(source_ankle - source_hip) / source_chain_len)

        parent_rot = (
            _rotation_from_matrix(current_globals[calibration.parent_node])
            if calibration.parent_node is not None
            else Rotation.identity()
        )
        pole = parent_rot.apply(calibration.rest_pole_parent_local)
        pole = _projected_unit(pole, source_direction)
        if pole is None:
            source_pole = _limb_pole(source_hip, source_knee, source_ankle)
            pole = _projected_unit(source_pole, source_direction)
        if pole is None:
            pole = _fallback_pole(source_direction)

        hip_pos = current_globals[skin.joints[calibration.hip]][:3, 3]
        knee_pos, ankle_pos = _two_bone_knee_position(
            hip_pos,
            source_direction,
            pole,
            upper_len=calibration.upper_len,
            lower_len=calibration.lower_len,
            reach_ratio=reach_ratio,
        )

        desired_upper = knee_pos - hip_pos
        desired_lower = ankle_pos - knee_pos
        desired_hip_rot = _basis_rotation(
            calibration.rest_upper_local,
            calibration.rest_pole_hip_local,
            desired_upper,
            pole,
        )
        hip_delta = local_rest_rot[calibration.hip].inv() * parent_rot.inv() * desired_hip_rot
        local[calibration.hip] = hip_delta.as_quat()

        hip_global_rot = parent_rot * local_rest_rot[calibration.hip] * hip_delta
        desired_knee_rot = _basis_rotation(
            calibration.rest_lower_local,
            calibration.rest_pole_knee_local,
            desired_lower,
            pole,
        )
        knee_delta = local_rest_rot[calibration.knee].inv() * hip_global_rot.inv() * desired_knee_rot
        local[calibration.knee] = knee_delta.as_quat()

        if (
            calibration.toe is not None
            and calibration.rest_toe_local is not None
            and calibration.rest_pole_foot_local is not None
        ):
            source_toe = source[source_map["toe"]]
            source_toe_direction = _safe_unit(source_toe - source_ankle)
            if source_toe_direction is not None:
                knee_global_rot = hip_global_rot * local_rest_rot[calibration.knee] * knee_delta
                desired_foot_rot = _basis_rotation(
                    calibration.rest_toe_local,
                    calibration.rest_pole_foot_local,
                    source_toe_direction,
                    pole,
                )
                foot_delta = local_rest_rot[calibration.ankle].inv() * knee_global_rot.inv() * desired_foot_rot
                local[calibration.ankle] = foot_delta.as_quat()

    return local


def _ik_local_quats_for_frame(
    source_joints: np.ndarray,
    gltf: GLTF2,
    joint_by_name: dict[str, int],
) -> np.ndarray:
    skin = gltf.skins[0]
    local = identity_quaternions(len(joint_by_name))
    source = _source_to_avatar_axes(source_joints.astype(np.float64))

    rest_globals = node_global_matrices(gltf, local)
    rest_positions = np.stack([rest_globals[joint][:3, 3] for joint in skin.joints])
    rest_global_rot = {
        skin_index: _rotation_from_matrix(rest_globals[node_index])
        for skin_index, node_index in enumerate(skin.joints)
    }
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index) for skin_index, node_index in enumerate(skin.joints)
    }

    global_rot: dict[int, Rotation] = {}
    source_index_by_name = {name: index for index, name in SMPL22_TO_MIXAMO.items()}
    for source_index in range(len(SMPL22_TO_MIXAMO)):
        name = SMPL22_TO_MIXAMO[source_index]
        skin_index = joint_by_name.get(name)
        if skin_index is None:
            continue

        parent_source = SMPL22_PARENTS[source_index]
        if parent_source is not None:
            parent_name = SMPL22_TO_MIXAMO[parent_source]
            parent_skin_index = joint_by_name.get(parent_name)
            parent_rot = global_rot.get(parent_skin_index, Rotation.identity())
        else:
            parent_rot = Rotation.identity()

        rest_vectors = []
        posed_vectors = []
        for child_source in SMPL22_CHILDREN[source_index]:
            child_name = SMPL22_TO_MIXAMO[child_source]
            child_skin_index = joint_by_name.get(child_name)
            if child_skin_index is None or child_name not in source_index_by_name:
                continue
            rest_vector_global = rest_positions[child_skin_index] - rest_positions[skin_index]
            rest_vector_local = rest_global_rot[skin_index].inv().apply(rest_vector_global)
            posed_vector = source[child_source] - source[source_index]
            rest_unit = _safe_unit(rest_vector_local)
            posed_unit = _safe_unit(posed_vector)
            if rest_unit is None or posed_unit is None:
                continue
            rest_vectors.append(rest_unit)
            posed_vectors.append(posed_unit)

        if rest_vectors:
            desired_global, _ = Rotation.align_vectors(np.asarray(posed_vectors), np.asarray(rest_vectors))
            delta = local_rest_rot[skin_index].inv() * parent_rot.inv() * desired_global
            local[skin_index] = delta.as_quat()
            global_rot[skin_index] = desired_global
        else:
            delta = Rotation.identity()
            local[skin_index] = delta.as_quat()
            global_rot[skin_index] = parent_rot * local_rest_rot[skin_index] * delta
    return local


def _source_geometries(avatar_glb: Path) -> list[Any]:
    scene = trimesh.load(avatar_glb, force="scene")
    return list(scene.geometry.values())


def _root_offset(clip: MotionClip, frame_index: int, *, options: RetargetOptions) -> np.ndarray | None:
    if not options.preserve_root_motion or clip.root_positions is None:
        return None
    raw = (clip.root_positions[frame_index] - clip.root_positions[0]) * options.root_motion_scale
    return np.asarray(raw, dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _translate_frame(frame: FrameGeometry, delta: np.ndarray) -> None:
    frame.vertices_by_primitive = [vertices + delta for vertices in frame.vertices_by_primitive]
    if frame.joints is not None:
        frame.joints = frame.joints + delta


def _rotate_y(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_euler("y", angle).as_matrix()
    return (points - pivot) @ rotation.T + pivot


def _rotate_frame_y(frame: FrameGeometry, angle: float, pivot: np.ndarray) -> None:
    frame.vertices_by_primitive = [_rotate_y(vertices, angle, pivot) for vertices in frame.vertices_by_primitive]
    if frame.joints is not None:
        frame.joints = _rotate_y(frame.joints, angle, pivot)


def _joint_index(joint_by_name: dict[str, int], *names: str) -> int | None:
    for name in names:
        index = joint_by_name.get(name)
        if index is not None:
            return index
    return None


def _joint_point(
    joints: np.ndarray,
    joint_by_name: dict[str, int],
    *names: str,
) -> np.ndarray | None:
    index = _joint_index(joint_by_name, *names)
    if index is None:
        return None
    return joints[index]


def _root_yaw_from_joints(joints: np.ndarray, joint_by_name: dict[str, int]) -> float | None:
    left = _joint_point(joints, joint_by_name, "LeftUpLeg", "LeftLeg", "LeftFoot")
    right = _joint_point(joints, joint_by_name, "RightUpLeg", "RightLeg", "RightFoot")
    if left is None or right is None:
        return None
    side = right - left
    side[1] = 0.0
    side_unit = _safe_unit(side)
    if side_unit is None:
        return None
    return float(np.arctan2(side_unit[2], side_unit[0]))


def _frame_pivot(frame: FrameGeometry, joint_by_name: dict[str, int]) -> np.ndarray:
    if frame.joints is not None:
        hips = _joint_point(frame.joints, joint_by_name, "Hips")
        if hips is not None:
            return hips
        return frame.joints.mean(axis=0)
    return np.vstack(frame.vertices_by_primitive).mean(axis=0)


def _apply_root_yaw_stabilization(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
) -> dict[str, Any]:
    reference_yaw: float | None = None
    yaw_deltas: list[float] = []
    for frame in frames:
        if frame.joints is None:
            continue
        yaw = _root_yaw_from_joints(frame.joints.copy(), joint_by_name)
        if yaw is None:
            continue
        reference_yaw = yaw
        break

    if reference_yaw is None:
        return {"enabled": True, "frames_adjusted": 0, "reason": "missing hip/leg joints"}

    for frame in frames:
        if frame.joints is None:
            continue
        yaw = _root_yaw_from_joints(frame.joints.copy(), joint_by_name)
        if yaw is None:
            yaw_deltas.append(0.0)
            continue
        delta = _wrap_angle(yaw - reference_yaw)
        yaw_deltas.append(delta)
        if abs(delta) > 1e-6:
            _rotate_frame_y(frame, delta, _frame_pivot(frame, joint_by_name))

    return {
        "enabled": True,
        "reference_yaw_degrees": float(np.degrees(reference_yaw)),
        "frames_adjusted": int(sum(abs(delta) > 1e-6 for delta in yaw_deltas)),
        "max_yaw_delta_degrees": float(np.degrees(max((abs(delta) for delta in yaw_deltas), default=0.0))),
    }


def _foot_points(
    joints: np.ndarray,
    joint_by_name: dict[str, int],
    side: str,
) -> tuple[np.ndarray | None, float | None]:
    indices = [
        joint_by_name[name]
        for name in (f"{side}Foot", f"{side}ToeBase")
        if name in joint_by_name
    ]
    if not indices:
        return None, None
    points = joints[indices]
    return points.mean(axis=0), float(points[:, 1].min())


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _edge_blend_weight(index: int, start: int, end: int, frame_count: int, blend_frames: int) -> float:
    if blend_frames <= 0:
        return 1.0
    weight = 1.0
    if start > 0:
        weight = min(weight, (index - start + 1) / (blend_frames + 1))
    if end < frame_count - 1:
        weight = min(weight, (end - index + 1) / (blend_frames + 1))
    return float(np.clip(weight, 0.0, 1.0))


def _source_foot_contact_mask(foot_contacts: np.ndarray | None, frame_count: int) -> np.ndarray | None:
    if foot_contacts is None or len(foot_contacts) != frame_count:
        return None
    if foot_contacts.ndim != 2 or foot_contacts.shape[1] < 2:
        return None
    contacts = foot_contacts.astype(bool)
    left = contacts[:, : min(2, contacts.shape[1])].any(axis=1)
    if contacts.shape[1] >= 4:
        right = contacts[:, 2:4].any(axis=1)
    else:
        right = contacts[:, -1]
    return np.column_stack([left, right])


def _select_lock_segments(support_mask: np.ndarray) -> list[tuple[int, int]]:
    frame_count = len(support_mask)
    segments = _contiguous_segments(support_mask)
    selected: list[tuple[int, int]] = []
    start_window_end = max(1, frame_count // 5)
    landing_window_start = max(0, int(frame_count * 0.55))
    for start, end in segments:
        if start <= start_window_end or end >= landing_window_start:
            selected.append((start, end))
    if not any(end >= landing_window_start for _, end in selected):
        lock_frames = max(3, min(8, frame_count // 5))
        selected.append((frame_count - lock_frames, frame_count - 1))
    return selected


def _apply_foot_contact_lock(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    *,
    ground_y: float,
    contact_height: float,
    blend_frames: int,
    source_foot_contacts: np.ndarray | None = None,
) -> dict[str, Any]:
    frame_count = len(frames)
    centers = np.full((frame_count, 2, 3), np.nan, dtype=np.float64)
    lows = np.full((frame_count, 2), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        if frame.joints is None:
            continue
        for side_index, side in enumerate(("Left", "Right")):
            center, low = _foot_points(frame.joints, joint_by_name, side)
            if center is not None and low is not None:
                centers[frame_index, side_index] = center
                lows[frame_index, side_index] = low

    if np.isnan(lows).all():
        return {"enabled": True, "locked_frames": 0, "reason": "missing foot joints"}

    contact_mask = lows <= (ground_y + contact_height)
    source_mask = _source_foot_contact_mask(source_foot_contacts, frame_count)
    if source_mask is not None:
        contact_mask = contact_mask | (source_mask & (lows <= ground_y + contact_height * 2.0))
    contact_mask = np.nan_to_num(contact_mask, nan=False).astype(bool)
    support_mask = contact_mask.any(axis=1)
    segments = _select_lock_segments(support_mask)

    locked_frames: set[int] = set()
    applied_segments: list[dict[str, Any]] = []
    for start, end in segments:
        if start > end:
            continue
        segment_mask = contact_mask[start : end + 1]
        side_indices = [index for index in range(2) if segment_mask[:, index].any()]
        if not side_indices:
            anchor_lows = lows[start]
            if np.isnan(anchor_lows).all():
                continue
            side_indices = [int(np.nanargmin(anchor_lows))]

        anchor_frame = start
        segment_record = {
            "start": int(start),
            "end": int(end),
            "anchor_frame": int(anchor_frame),
            "sides": ["Left" if index == 0 else "Right" for index in side_indices],
        }
        applied_segments.append(segment_record)
        for frame_index in range(start, end + 1):
            deltas = []
            for side_index in side_indices:
                current = centers[frame_index, side_index]
                anchor = centers[anchor_frame, side_index]
                low = lows[frame_index, side_index]
                if np.isnan(current).any() or np.isnan(anchor).any() or np.isnan(low):
                    continue
                delta = np.array(
                    [
                        anchor[0] - current[0],
                        ground_y - low,
                        anchor[2] - current[2],
                    ],
                    dtype=np.float64,
                )
                deltas.append(delta)
            if not deltas:
                continue
            weight = _edge_blend_weight(frame_index, start, end, frame_count, blend_frames)
            delta = np.mean(deltas, axis=0) * weight
            _translate_frame(frames[frame_index], delta)
            locked_frames.add(frame_index)

    ground_lifts = 0
    for frame in frames:
        lowest = min(float(vertices[:, 1].min()) for vertices in frame.vertices_by_primitive)
        if lowest < ground_y:
            _translate_frame(frame, np.array([0.0, ground_y - lowest, 0.0], dtype=np.float64))
            ground_lifts += 1

    return {
        "enabled": True,
        "locked_frames": len(locked_frames),
        "segments": applied_segments,
        "ground_lift_frames": ground_lifts,
        "contact_height": contact_height,
        "blend_frames": blend_frames,
    }


def _postprocess_frames(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    clip: MotionClip,
    options: RetargetOptions,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "root_yaw_stabilization": {"enabled": False},
        "foot_contact_lock": {"enabled": False},
    }
    if options.stabilize_root_yaw:
        report["root_yaw_stabilization"] = _apply_root_yaw_stabilization(frames, joint_by_name)
    if options.foot_contact_lock:
        report["foot_contact_lock"] = _apply_foot_contact_lock(
            frames,
            joint_by_name,
            ground_y=options.ground_y,
            contact_height=options.foot_contact_height,
            blend_frames=options.foot_lock_blend_frames,
            source_foot_contacts=clip.foot_contacts,
        )
    return report


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    sorted_edges = np.sort(edges, axis=1)
    unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _is_hair_primitive(source: Any) -> bool:
    names = [str(source.metadata.get("name", "")) if hasattr(source, "metadata") else ""]
    material = getattr(getattr(source, "visual", None), "material", None)
    if material is not None:
        names.append(str(getattr(material, "name", "") or ""))
    return "hair" in " ".join(names).lower()


def _solidified_visual(source: Any, vertex_count: int) -> Any:
    visual = source.visual.copy()
    if hasattr(source.visual, "uv") and source.visual.uv is not None:
        uv = np.asarray(source.visual.uv)
        if len(uv) == vertex_count:
            return trimesh.visual.TextureVisuals(
                uv=np.vstack([uv, uv]),
                material=source.visual.material,
            )
    return visual


def _solidify_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    source: Any,
    *,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if thickness <= 0:
        return vertices, faces, source.visual.copy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9

    outer = vertices + normals * (thickness * 0.5)
    inner = vertices - normals * (thickness * 0.5)
    solid_vertices = np.vstack([outer, inner])
    inner_offset = len(vertices)

    face_sets = [faces, faces[:, ::-1] + inner_offset]
    side_faces = []
    for a, b in _boundary_edges(faces):
        side_faces.append([a, b, b + inner_offset])
        side_faces.append([a, b + inner_offset, a + inner_offset])
    if side_faces:
        face_sets.append(np.asarray(side_faces, dtype=np.int64))

    solid_faces = np.vstack(face_sets)
    return solid_vertices, solid_faces, _solidified_visual(source, len(vertices))


def _export_frame(
    glb_path: Path,
    vertices_by_primitive: list[np.ndarray],
    source_geometries: list[Any],
    *,
    options: RetargetOptions,
) -> None:
    scene = trimesh.Scene()
    for primitive_index, vertices in enumerate(vertices_by_primitive):
        if primitive_index >= len(source_geometries):
            raise ValueError(
                f"deformed primitive count exceeds source geometry count: {primitive_index + 1}>{len(source_geometries)}"
            )
        source = source_geometries[primitive_index]
        faces = source.faces.copy()
        thickness = (
            options.hair_shell_thickness if _is_hair_primitive(source) else options.body_shell_thickness
        )
        if not options.solidify_shell:
            thickness = 0.0
        solid_vertices, solid_faces, visual = _solidify_vertices(
            vertices,
            faces,
            source,
            thickness=thickness,
        )
        mesh = trimesh.Trimesh(vertices=solid_vertices, faces=solid_faces, process=False)
        mesh.visual = visual
        name = source.metadata.get("name", "primitive") if hasattr(source, "metadata") else "primitive"
        scene.add_geometry(mesh, geom_name=f"{name}_{primitive_index:02d}")
    scene.export(glb_path, include_normals=True)


def retarget_motion_to_avatar(
    *,
    avatar_glb: Path,
    motion: MotionClip,
    output_dir: Path,
    asset_name: str,
    prompt: str,
    identity: str,
    options: RetargetOptions | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or RetargetOptions()
    clip = resample_motion(motion, options.frames)
    output_dir.mkdir(parents=True, exist_ok=True)

    gltf = GLTF2().load(str(avatar_glb))
    if not gltf.skins:
        raise ValueError(f"avatar GLB has no skin: {avatar_glb}")
    skin = gltf.skins[0]
    joint_by_name = {joint_base_name(gltf.nodes[joint].name): index for index, joint in enumerate(skin.joints)}
    source_geometries = _source_geometries(avatar_glb)
    rest_joint_deltas = identity_quaternions(len(skin.joints))
    rest_globals = node_global_matrices(gltf, rest_joint_deltas)
    leg_calibrations = _leg_calibrations(gltf, joint_by_name, rest_globals)
    leg_ik_enabled = options.calibrated_leg_ik and clip.posed_joints is not None and bool(leg_calibrations)

    rest_vertices = deform_skinned_primitives(gltf, rest_joint_deltas)
    normalizer = VertexNormalizer.from_vertices(
        rest_vertices, target_height=options.target_height, ground_y=options.ground_y
    )

    material_changes: dict[str, Any] = {}
    bounds: list[dict[str, list[float]]] = []
    frame_geometries: list[FrameGeometry] = []
    joint_names = [joint_base_name(gltf.nodes[joint].name) for joint in skin.joints]

    for frame_index in range(options.frames):
        if options.prefer_joint_position_ik and clip.posed_joints is not None:
            local_quats = _ik_local_quats_for_frame(clip.posed_joints[frame_index], gltf, joint_by_name)
        else:
            local_quats = _local_quats_for_frame(
                clip.local_rot_mats[frame_index], joint_by_name, options=options
            )
            if leg_ik_enabled:
                local_quats = _apply_calibrated_leg_ik(
                    local=local_quats,
                    source_joints=clip.posed_joints[frame_index],
                    gltf=gltf,
                    calibrations=leg_calibrations,
                )
        deformed = deform_skinned_primitives(gltf, local_quats)
        globals_ = node_global_matrices(gltf, local_quats)
        joint_points = np.stack([globals_[joint][:3, 3] for joint in skin.joints])
        transformed, transformed_joints = normalizer.transform(
            deformed,
            joint_points,
            root_offset=_root_offset(clip, frame_index, options=options),
            snap_to_ground=options.snap_to_ground,
        )
        frame_geometries.append(FrameGeometry(vertices_by_primitive=transformed, joints=transformed_joints))

    postprocess = _postprocess_frames(frame_geometries, joint_by_name, clip, options)
    skeleton_frames = [
        frame.joints.tolist()
        for frame in frame_geometries
        if frame.joints is not None
    ]

    for frame_index, frame in enumerate(frame_geometries):
        glb_path = output_dir / f"frame{frame_index:03d}.glb"
        _export_frame(glb_path, frame.vertices_by_primitive, source_geometries, options=options)
        material_changes[glb_path.name] = fix_habitat_materials(
            glb_path,
            alpha_cutoff=options.alpha_cutoff,
            roughness=options.material_roughness,
            detect_texture_alpha=options.detect_texture_alpha,
        )
        write_object_config(output_dir / f"frame{frame_index:03d}.object_config.json", glb_path)

        combined = np.vstack(frame.vertices_by_primitive)
        bounds.append({"min": combined.min(axis=0).tolist(), "max": combined.max(axis=0).tolist()})

    skeleton = {
        "joint_names": joint_names,
        "frames": skeleton_frames,
        "fps": options.fps,
        "source_skeleton": "SMPL-22/Kimodo local rotations retargeted to avatar skin joints",
        "leg_ik": {
            "enabled": leg_ik_enabled,
            "strategy": "calibrated two-bone IK with target-rig knee pole constraints",
            "legs": [calibration.side for calibration in leg_calibrations],
        },
        "postprocess": postprocess,
    }
    (output_dir / "skeleton.json").write_text(json.dumps(skeleton, indent=2), encoding="utf-8")

    persona = {
        "asset_name": asset_name,
        "identity": identity,
        "prompt": prompt,
        "avatar_glb": str(avatar_glb),
        "motion_source": str(motion.source_path),
        "frames": options.frames,
        "fps": options.fps,
        "appearance_strategy": "preserve original ViCo/Mixamo skinned GLB mesh, UVs, skin weights, and textures",
        "retarget_strategy": (
            "map SMPL-22/GEM/Kimodo local rotations to compatible ViCo/Mixamo skin joints; "
            "when posed joints are available, solve calibrated two-bone leg IK using the target rig knee pole; "
            "optionally stabilize root yaw and lock support feet during contact/landing"
        ),
        "selection": selection or {},
        "material_strategy": {
            "hair_and_alpha_materials": "MASK",
            "double_sided": True,
            "roughness_min": options.material_roughness,
            "solidify_shell": options.solidify_shell,
            "body_shell_thickness_m": options.body_shell_thickness,
            "hair_shell_thickness_m": options.hair_shell_thickness,
        },
        "postprocess": postprocess,
    }
    (output_dir / "persona.json").write_text(json.dumps(persona, indent=2), encoding="utf-8")
    (output_dir / "habitat_material_fix_report.json").write_text(
        json.dumps({"material_changes": material_changes}, indent=2), encoding="utf-8"
    )

    report = {
        "asset_name": asset_name,
        "asset_dir": str(output_dir),
        "avatar_glb": str(avatar_glb),
        "motion_source": str(motion.source_path),
        "prompt": prompt,
        "frames": options.frames,
        "glb_count": len(list(output_dir.glob("frame*.glb"))),
        "object_config_count": len(list(output_dir.glob("frame*.object_config.json"))),
        "skeleton": str(output_dir / "skeleton.json"),
        "bounds": bounds,
        "min_ground_y": min(item["min"][1] for item in bounds),
        "max_top_y": max(item["max"][1] for item in bounds),
        "max_height": max(item["max"][1] - item["min"][1] for item in bounds),
        "leg_ik": skeleton["leg_ik"],
        "postprocess": postprocess,
    }
    return report
