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

SMPL_ARM_JOINTS = {
    "Left": {"shoulder": 16, "elbow": 18, "wrist": 20},
    "Right": {"shoulder": 17, "elbow": 19, "wrist": 21},
}

GLTF_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
SOURCE_TO_AVATAR_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RetargetOptions:
    frames: int = 120
    fps: int = 24
    target_height: float = 1.72
    ground_y: float = -0.2
    rotation_scale: float = 0.65
    arm_rotation_scale: float = 0.65
    forearm_rotation_scale: float = 0.45
    hand_rotation_scale: float = 0.18
    lower_leg_rotation_scale: float = 0.18
    foot_rotation_scale: float = 0.35
    include_root_orientation: bool = True
    preserve_root_motion: bool = False
    root_motion_scale: float = 1.0
    snap_to_ground: bool = True
    stabilize_root_yaw: bool = False
    foot_contact_lock: bool = False
    foot_orientation_lock: bool = True
    foot_contact_height: float = 0.12
    foot_contact_velocity: float = 0.02
    foot_contact_use_source: bool = True
    foot_lock_blend_frames: int = 4
    grounded_foot_ik: bool = True
    foot_support_min_frames: int = 8
    foot_support_max_frames: int = 16
    foot_support_max_air_frames: int = 8
    airborne_leg_stabilization: bool = False
    airborne_leg_stabilization_strength: float = 0.85
    airborne_tuck_reach_ratio: float = 0.52
    alpha_cutoff: float = 0.55
    material_roughness: float = 0.88
    detect_texture_alpha: bool = True
    calibrated_leg_ik: bool = True
    calibrated_arm_ik: bool = True
    body_relative_leg_ik: bool = False
    prefer_joint_position_ik: bool = False
    procedural_running_arm_swing: bool = False
    running_arm_swing_strength: float = 0.35
    running_arm_forward_ratio: float = 0.52
    running_arm_drop_ratio: float = 0.50
    running_arm_side_ratio: float = 0.055
    running_arm_reach_min: float = 0.46
    running_arm_reach_max: float = 0.68
    locomotion_torso_counter_rotation: bool = False
    torso_counter_rotation_degrees: float = 7.0
    torso_counter_rotation_strength: float = 0.45
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
    rest_foot_up_local: np.ndarray | None
    rest_contact_toe_global: np.ndarray | None


@dataclass(frozen=True)
class ArmCalibration:
    side: str
    upper: int
    forearm: int
    hand: int
    parent_node: int | None
    upper_len: float
    lower_len: float
    rest_upper_local: np.ndarray
    rest_lower_local: np.ndarray
    rest_pole_parent_local: np.ndarray
    rest_pole_upper_local: np.ndarray
    rest_pole_forearm_local: np.ndarray


@dataclass
class FrameGeometry:
    vertices_by_primitive: list[np.ndarray]
    joints: np.ndarray | None


@dataclass(frozen=True)
class FootContactAnalysis:
    centers: np.ndarray
    lows: np.ndarray
    speeds: np.ndarray
    contact_mask: np.ndarray
    segments: list[tuple[int, int]]
    source_contacts_used: bool
    source_contact_ratio: list[float]


@dataclass(frozen=True)
class AirborneLegGuide:
    direction_local: np.ndarray
    reach_ratio: float
    skip_foot_ik: bool = True


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
        if mixamo_name in {"LeftArm", "RightArm"}:
            scale = options.arm_rotation_scale
        elif mixamo_name in {"LeftForeArm", "RightForeArm"}:
            scale = options.forearm_rotation_scale
        elif mixamo_name in {"LeftHand", "RightHand"}:
            scale = options.hand_rotation_scale
        elif mixamo_name in {"LeftLeg", "RightLeg"}:
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


def _source_rotation_to_avatar_axes(matrix: np.ndarray) -> Rotation:
    return Rotation.from_matrix(SOURCE_TO_AVATAR_AXES @ matrix @ SOURCE_TO_AVATAR_AXES.T)


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


def _orthonormal_body_basis(side: np.ndarray, up_hint: np.ndarray) -> np.ndarray | None:
    side_unit = _safe_unit(side)
    if side_unit is None:
        return None
    up_unit = _projected_unit(up_hint, side_unit)
    if up_unit is None:
        up_unit = _fallback_pole(side_unit)
    forward_unit = _safe_unit(np.cross(side_unit, up_unit))
    if forward_unit is None:
        return None
    up_unit = _safe_unit(np.cross(forward_unit, side_unit))
    if up_unit is None:
        return None
    return np.column_stack([side_unit, up_unit, forward_unit])


def _source_body_basis(source: np.ndarray) -> np.ndarray | None:
    if len(source) <= 3:
        return None
    hips = source[0]
    side = source[SMPL_LEG_JOINTS["Right"]["hip"]] - source[SMPL_LEG_JOINTS["Left"]["hip"]]
    for spine_index in (9, 6, 3, 15):
        if spine_index < len(source):
            basis = _orthonormal_body_basis(side, source[spine_index] - hips)
            if basis is not None:
                return basis
    return None


def _target_body_basis_from_globals(
    globals_: list[np.ndarray],
    skin: Any,
    joint_by_name: dict[str, int],
) -> np.ndarray | None:
    hips_index = joint_by_name.get("Hips")
    left_index = joint_by_name.get("LeftUpLeg")
    right_index = joint_by_name.get("RightUpLeg")
    if hips_index is None or left_index is None or right_index is None:
        return None
    hips = globals_[skin.joints[hips_index]][:3, 3]
    left = globals_[skin.joints[left_index]][:3, 3]
    right = globals_[skin.joints[right_index]][:3, 3]
    side = right - left
    for spine_name in ("Spine2", "Spine1", "Spine", "Neck", "Head"):
        spine_index = joint_by_name.get(spine_name)
        if spine_index is None:
            continue
        basis = _orthonormal_body_basis(side, globals_[skin.joints[spine_index]][:3, 3] - hips)
        if basis is not None:
            return basis
    return None


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
        rest_toe = rest_positions[toe] - ankle_pos if toe is not None else None
        rest_contact_toe = _projected_unit(rest_toe, GLTF_UP) if rest_toe is not None else None
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
                rest_toe_local=rest_rotations[ankle].inv().apply(rest_toe) if rest_toe is not None else None,
                rest_foot_up_local=rest_rotations[ankle].inv().apply(GLTF_UP) if toe is not None else None,
                rest_contact_toe_global=rest_contact_toe,
            )
        )
    return calibrations


def _arm_calibrations(
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    rest_globals: list[np.ndarray],
) -> list[ArmCalibration]:
    skin = gltf.skins[0]
    parents = _node_parent_indices(gltf)
    rest_positions = np.stack([rest_globals[joint][:3, 3] for joint in skin.joints])
    rest_rotations = {
        skin_index: _rotation_from_matrix(rest_globals[node_index])
        for skin_index, node_index in enumerate(skin.joints)
    }
    body_basis = _target_body_basis_from_globals(rest_globals, skin, joint_by_name)
    rest_side = body_basis[:, 0] if body_basis is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rest_forward = body_basis[:, 2] if body_basis is not None else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    calibrations: list[ArmCalibration] = []
    for side, sign in (("Left", -1.0), ("Right", 1.0)):
        upper = joint_by_name.get(f"{side}Arm")
        forearm = joint_by_name.get(f"{side}ForeArm")
        hand = joint_by_name.get(f"{side}Hand")
        if upper is None or forearm is None or hand is None:
            continue

        upper_pos = rest_positions[upper]
        forearm_pos = rest_positions[forearm]
        hand_pos = rest_positions[hand]
        rest_upper = forearm_pos - upper_pos
        rest_lower = hand_pos - forearm_pos
        upper_len = float(np.linalg.norm(rest_upper))
        lower_len = float(np.linalg.norm(rest_lower))
        if upper_len < 1e-8 or lower_len < 1e-8:
            continue

        parent_node = parents.get(skin.joints[upper])
        parent_rot = _rotation_from_matrix(rest_globals[parent_node]) if parent_node is not None else Rotation.identity()
        pole = _projected_unit(rest_forward, rest_side * sign)
        if pole is None:
            pole = _fallback_pole(rest_side * sign)

        calibrations.append(
            ArmCalibration(
                side=side,
                upper=upper,
                forearm=forearm,
                hand=hand,
                parent_node=parent_node,
                upper_len=upper_len,
                lower_len=lower_len,
                rest_upper_local=rest_rotations[upper].inv().apply(rest_upper),
                rest_lower_local=rest_rotations[forearm].inv().apply(rest_lower),
                rest_pole_parent_local=parent_rot.inv().apply(pole),
                rest_pole_upper_local=rest_rotations[upper].inv().apply(pole),
                rest_pole_forearm_local=rest_rotations[forearm].inv().apply(pole),
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


def _source_body_relative_direction(
    direction: np.ndarray,
    *,
    source_body_basis: np.ndarray | None = None,
    target_body_basis: np.ndarray | None = None,
    source_global_rotations: dict[int, Rotation] | None = None,
    source_parent_index: int | None = None,
    target_parent_rotation: Rotation = Rotation.identity(),
) -> np.ndarray | None:
    direction_unit = _safe_unit(direction)
    if direction_unit is None:
        return None
    if source_body_basis is not None and target_body_basis is not None:
        local_direction = source_body_basis.T @ direction_unit
        mapped = target_body_basis @ local_direction
        mapped_unit = _safe_unit(mapped)
        return mapped_unit if mapped_unit is not None else direction_unit
    if source_global_rotations is None or source_parent_index is None:
        return direction_unit
    source_parent_rotation = source_global_rotations.get(source_parent_index)
    if source_parent_rotation is None:
        return direction_unit
    local_direction = source_parent_rotation.inv().apply(direction_unit)
    mapped = target_parent_rotation.apply(local_direction)
    mapped_unit = _safe_unit(mapped)
    return mapped_unit if mapped_unit is not None else direction_unit


def _apply_calibrated_leg_ik(
    *,
    local: np.ndarray,
    source_joints: np.ndarray,
    source_global_rot_mats: np.ndarray | None,
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibrations: list[LegCalibration],
    body_relative: bool,
    leg_guide: dict[str, AirborneLegGuide] | None = None,
) -> np.ndarray:
    if not calibrations:
        return local

    skin = gltf.skins[0]
    source = _source_to_avatar_axes(source_joints.astype(np.float64))
    source_basis = _source_body_basis(source) if body_relative else None
    source_global_rotations: dict[int, Rotation] | None = None
    if body_relative and source_global_rot_mats is not None:
        source_global_rotations = {
            index: _source_rotation_to_avatar_axes(source_global_rot_mats[index].astype(np.float64))
            for index in range(min(len(source_global_rot_mats), len(SMPL22_TO_MIXAMO)))
        }
    current_globals = node_global_matrices(gltf, local)
    target_basis = _target_body_basis_from_globals(current_globals, skin, joint_by_name) if body_relative else None
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index) for skin_index, node_index in enumerate(skin.joints)
    }

    for calibration in calibrations:
        source_map = SMPL_LEG_JOINTS[calibration.side]
        source_hip = source[source_map["hip"]]
        source_knee = source[source_map["knee"]]
        source_ankle = source[source_map["ankle"]]

        source_upper_len = float(np.linalg.norm(source_knee - source_hip))
        source_lower_len = float(np.linalg.norm(source_ankle - source_knee))
        source_chain_len = source_upper_len + source_lower_len
        if source_chain_len < 1e-8:
            continue
        guide = leg_guide.get(calibration.side) if leg_guide is not None else None
        reach_ratio = (
            guide.reach_ratio
            if guide is not None
            else float(np.linalg.norm(source_ankle - source_hip) / source_chain_len)
        )

        parent_rot = (
            _rotation_from_matrix(current_globals[calibration.parent_node])
            if calibration.parent_node is not None
            else Rotation.identity()
        )
        if guide is not None and target_basis is not None:
            source_direction = _safe_unit(target_basis @ guide.direction_local)
        else:
            source_direction = _source_body_relative_direction(
                source_ankle - source_hip,
                source_body_basis=source_basis,
                target_body_basis=target_basis,
                source_global_rotations=source_global_rotations,
                source_parent_index=SMPL22_PARENTS[source_map["hip"]],
                target_parent_rotation=parent_rot,
            )
        if source_direction is None:
            continue
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
            (guide is None or not guide.skip_foot_ik)
            and calibration.toe is not None
            and calibration.rest_toe_local is not None
            and calibration.rest_pole_foot_local is not None
        ):
            source_toe = source[source_map["toe"]]
            source_toe_direction = _source_body_relative_direction(
                source_toe - source_ankle,
                source_body_basis=source_basis,
                target_body_basis=target_basis,
                source_global_rotations=source_global_rotations,
                source_parent_index=source_map["knee"],
                target_parent_rotation=hip_global_rot * local_rest_rot[calibration.knee] * knee_delta,
            )
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


def _apply_calibrated_arm_ik(
    *,
    local: np.ndarray,
    source_joints: np.ndarray,
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibrations: list[ArmCalibration],
) -> np.ndarray:
    if not calibrations:
        return local

    skin = gltf.skins[0]
    source = _source_to_avatar_axes(source_joints.astype(np.float64))
    current_globals = node_global_matrices(gltf, local)
    source_basis = _source_body_basis(source)
    target_basis = _target_body_basis_from_globals(current_globals, skin, joint_by_name)
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index)
        for skin_index, node_index in enumerate(skin.joints)
    }

    for calibration in calibrations:
        source_map = SMPL_ARM_JOINTS[calibration.side]
        source_shoulder = source[source_map["shoulder"]]
        source_elbow = source[source_map["elbow"]]
        source_wrist = source[source_map["wrist"]]

        source_upper_len = float(np.linalg.norm(source_elbow - source_shoulder))
        source_lower_len = float(np.linalg.norm(source_wrist - source_elbow))
        source_chain_len = source_upper_len + source_lower_len
        if source_chain_len < 1e-8:
            continue

        source_direction = _source_body_relative_direction(
            source_wrist - source_shoulder,
            source_body_basis=source_basis,
            target_body_basis=target_basis,
        )
        if source_direction is None:
            continue
        reach_ratio = float(np.linalg.norm(source_wrist - source_shoulder) / source_chain_len)
        reach_ratio = float(np.clip(reach_ratio, 0.20, 0.985))

        parent_rot = (
            _rotation_from_matrix(current_globals[calibration.parent_node])
            if calibration.parent_node is not None
            else Rotation.identity()
        )
        source_pole = _limb_pole(source_shoulder, source_elbow, source_wrist)
        pole = None
        if source_basis is not None and target_basis is not None:
            pole = _projected_unit(target_basis @ (source_basis.T @ source_pole), source_direction)
        if pole is None:
            pole = _projected_unit(source_pole, source_direction)
        if pole is None:
            pole = parent_rot.apply(calibration.rest_pole_parent_local)
            pole = _projected_unit(pole, source_direction)
        if pole is None:
            pole = _fallback_pole(source_direction)

        shoulder_pos = current_globals[skin.joints[calibration.upper]][:3, 3]
        elbow_pos, wrist_pos = _two_bone_knee_position(
            shoulder_pos,
            source_direction,
            pole,
            upper_len=calibration.upper_len,
            lower_len=calibration.lower_len,
            reach_ratio=reach_ratio,
        )

        desired_upper_rot = _basis_rotation(
            calibration.rest_upper_local,
            calibration.rest_pole_upper_local,
            elbow_pos - shoulder_pos,
            pole,
        )
        upper_delta = local_rest_rot[calibration.upper].inv() * parent_rot.inv() * desired_upper_rot
        local[calibration.upper] = upper_delta.as_quat()

        upper_global_rot = parent_rot * local_rest_rot[calibration.upper] * upper_delta
        desired_forearm_rot = _basis_rotation(
            calibration.rest_lower_local,
            calibration.rest_pole_forearm_local,
            wrist_pos - elbow_pos,
            pole,
        )
        forearm_delta = (
            local_rest_rot[calibration.forearm].inv()
            * upper_global_rot.inv()
            * desired_forearm_rot
        )
        local[calibration.forearm] = forearm_delta.as_quat()
        local[calibration.hand] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

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


def _root_offset(
    clip: MotionClip,
    frame_index: int,
    *,
    options: RetargetOptions,
    normalizer: VertexNormalizer,
) -> np.ndarray | None:
    if not options.preserve_root_motion or clip.root_positions is None:
        return None
    raw = np.asarray(clip.root_positions[frame_index] - clip.root_positions[0], dtype=np.float64)
    return raw * options.root_motion_scale * normalizer.scale


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


def _body_forward_from_globals(
    globals_: list[np.ndarray],
    gltf: GLTF2,
    skin: Any,
    joint_by_name: dict[str, int],
) -> np.ndarray | None:
    for left_name, right_name in (("LeftShoulder", "RightShoulder"), ("LeftUpLeg", "RightUpLeg")):
        left_index = joint_by_name.get(left_name)
        right_index = joint_by_name.get(right_name)
        if left_index is None or right_index is None:
            continue
        left = globals_[skin.joints[left_index]][:3, 3]
        right = globals_[skin.joints[right_index]][:3, 3]
        side = _projected_unit(right - left, GLTF_UP)
        if side is None:
            continue
        forward = _safe_unit(np.cross(GLTF_UP, side))
        if forward is not None:
            return forward
    return None


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


def _blend_rotation(source: Rotation, target: Rotation, weight: float) -> Rotation:
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return source
    if weight >= 1.0:
        return target
    delta = target * source.inv()
    return Rotation.from_rotvec(delta.as_rotvec() * weight) * source


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


def _fill_short_false_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask
    filled = mask.copy()
    for start, end in _contiguous_segments(~filled):
        if start == 0 or end == len(filled) - 1:
            continue
        if end - start + 1 <= max_gap_frames:
            filled[start : end + 1] = True
    return filled


def _drop_short_true_segments(mask: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return mask
    cleaned = mask.copy()
    for start, end in _contiguous_segments(cleaned):
        if end - start + 1 < min_frames:
            cleaned[start : end + 1] = False
    return cleaned


def _stabilize_contact_mask(
    enter_mask: np.ndarray,
    stay_mask: np.ndarray,
    *,
    min_frames: int = 2,
    max_gap_frames: int = 2,
) -> np.ndarray:
    stabilized = np.zeros_like(enter_mask, dtype=bool)
    for side_index in range(enter_mask.shape[1]):
        active = False
        for frame_index in range(len(enter_mask)):
            if active:
                active = bool(stay_mask[frame_index, side_index])
            else:
                active = bool(enter_mask[frame_index, side_index])
            stabilized[frame_index, side_index] = active
        stabilized[:, side_index] = _fill_short_false_gaps(
            stabilized[:, side_index],
            max_gap_frames,
        )
        stabilized[:, side_index] = _drop_short_true_segments(
            stabilized[:, side_index],
            min_frames,
        )
    return stabilized


def _select_lock_segments(support_mask: np.ndarray) -> list[tuple[int, int]]:
    frame_count = len(support_mask)
    segments = _contiguous_segments(support_mask)
    selected: list[tuple[int, int]] = []
    start_window_end = max(1, frame_count // 5)
    landing_window_start = max(0, int(frame_count * 0.55))
    for start, end in segments:
        if start <= start_window_end or end >= landing_window_start:
            selected.append((start, end))
    landing_indices = [index for index, (start, end) in enumerate(selected) if end >= landing_window_start]
    if landing_indices:
        index = landing_indices[-1]
        start, end = selected[index]
        if start >= landing_window_start and end < frame_count - 1:
            selected[index] = (start, frame_count - 1)
    else:
        lock_frames = max(3, min(8, frame_count // 5))
        selected.append((frame_count - lock_frames, frame_count - 1))
    return selected


def _foot_contact_analysis(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    *,
    ground_y: float,
    contact_height: float,
    contact_velocity: float | None = None,
    source_foot_contacts: np.ndarray | None = None,
    use_source_contacts: bool = True,
) -> FootContactAnalysis | None:
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
        return None

    speeds = np.full((frame_count, 2), np.nan, dtype=np.float64)
    for side_index in range(2):
        valid = np.isfinite(centers[:, side_index]).all(axis=1)
        side_speeds = np.full(frame_count, np.nan, dtype=np.float64)
        if frame_count == 1:
            side_speeds[valid] = 0.0
        elif valid.any():
            pair_valid = valid[:-1] & valid[1:]
            diffs = np.full(frame_count - 1, np.nan, dtype=np.float64)
            if pair_valid.any():
                horizontal = centers[:, side_index, [0, 2]]
                diffs[pair_valid] = np.linalg.norm(np.diff(horizontal, axis=0)[pair_valid], axis=1)
            for frame_index in range(frame_count):
                neighbors = []
                if frame_index > 0 and np.isfinite(diffs[frame_index - 1]):
                    neighbors.append(float(diffs[frame_index - 1]))
                if frame_index < frame_count - 1 and np.isfinite(diffs[frame_index]):
                    neighbors.append(float(diffs[frame_index]))
                if neighbors:
                    side_speeds[frame_index] = min(neighbors)
        speeds[:, side_index] = side_speeds

    height_mask = np.isfinite(lows) & (lows <= (ground_y + contact_height))
    stay_height_mask = np.isfinite(lows) & (
        lows <= ground_y + max(contact_height * 1.7, contact_height + 0.025)
    )
    if contact_velocity is not None and contact_velocity > 0.0:
        speed_mask = np.isfinite(speeds) & (speeds <= contact_velocity)
        stay_speed_mask = np.isfinite(speeds) & (
            speeds <= max(contact_velocity * 2.25, contact_velocity + 0.015)
        )
    else:
        speed_mask = np.isfinite(lows)
        stay_speed_mask = np.isfinite(lows)
    enter_mask = height_mask & speed_mask
    stay_mask = stay_height_mask & stay_speed_mask

    source_contacts_used = False
    source_contact_ratio = [0.0, 0.0]
    source_mask = _source_foot_contact_mask(source_foot_contacts, frame_count) if use_source_contacts else None
    if source_mask is not None:
        ratios = source_mask.mean(axis=0)
        source_contact_ratio = [float(ratios[0]), float(ratios[1])]
        usable = (ratios >= 0.03) & (ratios <= 0.75)
        if usable.any():
            source_contacts_used = True
            source_height_mask = np.isfinite(lows) & (lows <= ground_y + contact_height * 1.25)
            if contact_velocity is not None and contact_velocity > 0.0:
                source_speed_mask = np.isfinite(speeds) & (speeds <= contact_velocity * 1.75)
                source_stay_speed_mask = np.isfinite(speeds) & (speeds <= contact_velocity * 2.5)
            else:
                source_speed_mask = np.isfinite(lows)
                source_stay_speed_mask = np.isfinite(lows)
            source_enter = (
                source_mask
                & usable.reshape(1, 2)
                & source_height_mask
                & source_speed_mask
            )
            source_stay = (
                source_mask
                & usable.reshape(1, 2)
                & (np.isfinite(lows) & (lows <= ground_y + contact_height * 1.7))
                & source_stay_speed_mask
            )
            enter_mask = enter_mask | source_enter
            stay_mask = stay_mask | source_stay
    contact_mask = _stabilize_contact_mask(
        np.nan_to_num(enter_mask, nan=False).astype(bool),
        np.nan_to_num(stay_mask, nan=False).astype(bool),
        min_frames=2,
        max_gap_frames=2,
    )
    support_mask = contact_mask.any(axis=1)
    return FootContactAnalysis(
        centers=centers,
        lows=lows,
        speeds=speeds,
        contact_mask=contact_mask,
        segments=[
            (start, end)
            for start, end in _contiguous_segments(support_mask)
            if end - start + 1 >= 2
        ],
        source_contacts_used=source_contacts_used,
        source_contact_ratio=source_contact_ratio,
    )


def _max_airborne_gap(contact_mask: np.ndarray) -> int:
    support = contact_mask.any(axis=1)
    return max((end - start + 1 for start, end in _contiguous_segments(~support)), default=0)


def _split_long_segment(start: int, end: int, max_frames: int) -> list[tuple[int, int]]:
    if max_frames <= 1 or end - start + 1 <= max_frames:
        return [(start, end)]
    segments: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        segment_end = min(end, cursor + max_frames - 1)
        segments.append((cursor, segment_end))
        cursor = segment_end + 1
    return segments


def _expand_support_contact_mask(
    analysis: FootContactAnalysis,
    *,
    fps: int,
    ground_y: float,
    contact_height: float,
    min_segment_frames: int,
    max_air_frames: int,
) -> np.ndarray:
    frame_count = len(analysis.contact_mask)
    if frame_count == 0:
        return analysis.contact_mask.copy()

    min_segment_frames = max(2, int(min_segment_frames))
    pre_frames = max(2, min_segment_frames // 2)
    post_frames = max(pre_frames + 1, min_segment_frames)
    min_gap = max(min_segment_frames, int(round(max(fps, 1) * 0.16)))
    height_ceiling = ground_y + max(contact_height * 2.5, 0.08)

    expanded = np.zeros_like(analysis.contact_mask, dtype=bool)
    for side_index in range(2):
        raw_side = analysis.contact_mask[:, side_index]
        for start, end in _contiguous_segments(raw_side):
            span = end - start + 1
            if span < min_segment_frames:
                pad = min_segment_frames - span
                start = max(0, start - (pad + 1) // 2)
                end = min(frame_count - 1, end + pad // 2)
            expanded[start : end + 1, side_index] = True

        lows = analysis.lows[:, side_index]
        finite = np.isfinite(lows)
        candidates: list[int] = []
        for frame_index in range(frame_count):
            if not finite[frame_index] or lows[frame_index] > height_ceiling:
                continue
            start = max(0, frame_index - 2)
            end = min(frame_count, frame_index + 3)
            window = lows[start:end][finite[start:end]]
            if len(window) and lows[frame_index] <= np.nanmin(window) + 1e-6:
                candidates.append(frame_index)

        selected: list[int] = []
        for frame_index in candidates:
            if not selected or frame_index - selected[-1] >= min_gap:
                selected.append(frame_index)
            elif lows[frame_index] < lows[selected[-1]]:
                selected[-1] = frame_index

        for frame_index in selected:
            start = max(0, frame_index - pre_frames)
            end = min(frame_count - 1, frame_index + post_frames)
            expanded[start : end + 1, side_index] = True

    for frame_index in range(frame_count):
        if expanded[frame_index].sum() <= 1:
            continue
        side_scores = analysis.lows[frame_index].copy()
        finite = np.isfinite(side_scores)
        if not finite.any():
            expanded[frame_index] = False
            continue
        speeds = analysis.speeds[frame_index]
        if np.isfinite(speeds).any():
            speed_penalty = np.nan_to_num(
                speeds,
                nan=float(np.nanmax(speeds[np.isfinite(speeds)])),
            )
            side_scores = side_scores + speed_penalty * 0.35
        side_scores[~finite] = np.inf
        keep = int(np.argmin(side_scores))
        expanded[frame_index] = False
        expanded[frame_index, keep] = True

    max_air_frames = max(0, int(max_air_frames))
    if max_air_frames > 0:
        for _ in range(frame_count):
            gaps = [
                (start, end)
                for start, end in _contiguous_segments(~expanded.any(axis=1))
                if end - start + 1 > max_air_frames
            ]
            if not gaps:
                break
            for start, end in gaps:
                gap_lows = analysis.lows[start : end + 1]
                finite = np.isfinite(gap_lows)
                if not finite.any():
                    continue
                flat_index = int(np.argmin(np.where(finite, gap_lows, np.inf)))
                local_frame, side_index = np.unravel_index(flat_index, gap_lows.shape)
                frame_index = start + int(local_frame)
                support_start = max(start, frame_index - pre_frames)
                support_end = min(end, support_start + min_segment_frames - 1)
                support_start = max(start, support_end - min_segment_frames + 1)
                expanded[support_start : support_end + 1, side_index] = True

    return expanded


def _smooth_unit_vectors(vectors: np.ndarray, valid: np.ndarray, *, radius: int = 2) -> np.ndarray:
    smoothed = vectors.copy()
    for index in range(len(vectors)):
        if not valid[index]:
            continue
        start = max(0, index - radius)
        end = min(len(vectors), index + radius + 1)
        window = vectors[start:end][valid[start:end]]
        if len(window) == 0:
            continue
        average = _safe_unit(window.mean(axis=0))
        if average is not None:
            smoothed[index] = average
    return smoothed


def _canonical_airborne_leg_direction(side: str, tuck: float) -> np.ndarray:
    side_offset = -0.08 if side == "Left" else 0.08
    extended = np.array([side_offset, -0.98, 0.08], dtype=np.float64)
    tucked = np.array([side_offset, -0.20, 0.98], dtype=np.float64)
    blended = extended * (1.0 - tuck) + tucked * tuck
    direction = _safe_unit(blended)
    return direction if direction is not None else tucked


def _airborne_leg_guides(
    *,
    clip: MotionClip,
    rough_frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    options: RetargetOptions,
) -> tuple[list[dict[str, AirborneLegGuide] | None], dict[str, Any]]:
    frame_count = len(rough_frames)
    guides: list[dict[str, AirborneLegGuide] | None] = [None for _ in range(frame_count)]
    if clip.posed_joints is None:
        return guides, {"enabled": True, "guided_frames": 0, "reason": "missing posed joints"}

    analysis = _foot_contact_analysis(
        rough_frames,
        joint_by_name,
        ground_y=options.ground_y,
        contact_height=options.foot_contact_height,
        contact_velocity=options.foot_contact_velocity,
        source_foot_contacts=None,
        use_source_contacts=False,
    )
    if analysis is None:
        return guides, {"enabled": True, "guided_frames": 0, "reason": "missing foot joints"}

    support_mask = analysis.contact_mask.any(axis=1)
    air_segments = [
        (start, end)
        for start, end in _contiguous_segments(~support_mask)
        if end - start + 1 >= 3
    ]
    if not air_segments:
        return guides, {"enabled": True, "guided_frames": 0, "segments": []}

    local_dirs = {side: np.zeros((frame_count, 3), dtype=np.float64) for side in ("Left", "Right")}
    reach_ratios = {side: np.ones(frame_count, dtype=np.float64) for side in ("Left", "Right")}
    valid = {side: np.zeros(frame_count, dtype=bool) for side in ("Left", "Right")}

    for frame_index in range(frame_count):
        source = _source_to_avatar_axes(clip.posed_joints[frame_index].astype(np.float64))
        basis = _source_body_basis(source)
        if basis is None:
            continue
        for side in ("Left", "Right"):
            source_map = SMPL_LEG_JOINTS[side]
            hip = source[source_map["hip"]]
            knee = source[source_map["knee"]]
            ankle = source[source_map["ankle"]]
            direction = _safe_unit(ankle - hip)
            upper = float(np.linalg.norm(knee - hip))
            lower = float(np.linalg.norm(ankle - knee))
            chain = upper + lower
            if direction is None or chain < 1e-8:
                continue
            local_direction = _safe_unit(basis.T @ direction)
            if local_direction is None:
                continue
            local_dirs[side][frame_index] = local_direction
            reach_ratios[side][frame_index] = float(np.linalg.norm(ankle - hip) / chain)
            valid[side][frame_index] = True

    smoothed_dirs = {
        side: _smooth_unit_vectors(local_dirs[side], valid[side], radius=2)
        for side in ("Left", "Right")
    }
    strength = float(np.clip(options.airborne_leg_stabilization_strength, 0.0, 1.0))
    guided_frames: set[int] = set()
    segment_records: list[dict[str, int]] = []

    for start, end in air_segments:
        span = max(1, end - start)
        segment_records.append({"start": int(start), "end": int(end)})
        for frame_index in range(start, end + 1):
            phase = (frame_index - start) / span
            tuck = float(np.sin(np.pi * phase))
            blend = strength * (0.35 + 0.65 * tuck)
            blend *= _edge_blend_weight(
                frame_index,
                start,
                end,
                frame_count,
                options.foot_lock_blend_frames,
            )
            if blend <= 0.0:
                continue
            frame_guides: dict[str, AirborneLegGuide] = {}
            for side in ("Left", "Right"):
                if not valid[side][frame_index]:
                    continue
                canonical_direction = _canonical_airborne_leg_direction(side, tuck)
                direction = _safe_unit(
                    smoothed_dirs[side][frame_index] * (1.0 - blend)
                    + canonical_direction * blend
                )
                if direction is None:
                    continue
                canonical_reach = (1.0 - tuck) * 0.96 + tuck * options.airborne_tuck_reach_ratio
                reach_ratio = float(
                    reach_ratios[side][frame_index] * (1.0 - blend) + canonical_reach * blend
                )
                frame_guides[side] = AirborneLegGuide(
                    direction_local=direction,
                    reach_ratio=reach_ratio,
                )
            if frame_guides:
                guides[frame_index] = frame_guides
                guided_frames.add(frame_index)

    return guides, {
        "enabled": True,
        "guided_frames": len(guided_frames),
        "segments": segment_records,
        "strength": strength,
        "tuck_reach_ratio": options.airborne_tuck_reach_ratio,
        "strategy": (
            "smooth airborne leg directions in source pelvis-local space and blend them toward "
            "a symmetric backflip tuck curve before target-rig two-bone IK"
        ),
    }


def _prompt_requests_running_arm_swing(prompt: str) -> bool:
    text = prompt.lower()
    locomotion_words = (
        "run",
        "running",
        "jog",
        "jogging",
        "walk",
        "walking",
        "circle",
        "绕圈",
        "小圈",
        "跑",
        "慢跑",
        "走",
    )
    return any(word in text for word in locomotion_words)


def _moving_average(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values
    smoothed = values.copy()
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed[index] = float(np.mean(values[start:end]))
    return smoothed


def _source_leg_phase_signal(clip: MotionClip, frame_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    signal = np.zeros(frame_count, dtype=np.float64)
    valid = np.zeros(frame_count, dtype=bool)
    if clip.posed_joints is not None:
        for frame_index in range(min(frame_count, len(clip.posed_joints))):
            source = _source_to_avatar_axes(clip.posed_joints[frame_index].astype(np.float64))
            basis = _source_body_basis(source)
            if basis is None:
                continue
            hips = source[0]
            left_ankle = source[SMPL_LEG_JOINTS["Left"]["ankle"]]
            right_ankle = source[SMPL_LEG_JOINTS["Right"]["ankle"]]
            left_local = basis.T @ (left_ankle - hips)
            right_local = basis.T @ (right_ankle - hips)
            signal[frame_index] = float(left_local[2] - right_local[2])
            valid[frame_index] = True

    if valid.any():
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < frame_count:
            all_indices = np.arange(frame_count)
            signal = np.interp(all_indices, valid_indices, signal[valid_indices])
        signal = signal - float(np.mean(signal))
        amplitude = float(np.nanpercentile(np.abs(signal), 90.0))
        source = "posed_joints"
    else:
        cycles = max(2.0, frame_count / 36.0)
        signal = np.sin(np.linspace(0.0, cycles * 2.0 * np.pi, frame_count, endpoint=False))
        amplitude = 1.0
        source = "synthetic_cycle"

    if amplitude < 1e-6:
        cycles = max(2.0, frame_count / 36.0)
        signal = np.sin(np.linspace(0.0, cycles * 2.0 * np.pi, frame_count, endpoint=False))
        amplitude = 1.0
        source = "synthetic_cycle"

    normalized = np.clip(signal / amplitude, -1.0, 1.0)
    normalized = _moving_average(normalized, radius=2)
    return normalized, {
        "source": source,
        "valid_frames": int(valid.sum()),
        "amplitude": float(amplitude),
    }


def _apply_procedural_running_arm_swing(
    *,
    local_quats_by_frame: list[np.ndarray],
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibrations: list[ArmCalibration],
    clip: MotionClip,
    prompt: str,
    options: RetargetOptions,
) -> dict[str, Any]:
    if not options.procedural_running_arm_swing:
        return {"enabled": False, "reason": "disabled"}
    if not _prompt_requests_running_arm_swing(prompt):
        return {"enabled": False, "reason": "prompt is not locomotion-like"}
    if not calibrations:
        return {"enabled": True, "frames_adjusted": 0, "reason": "missing arm calibration"}

    frame_count = len(local_quats_by_frame)
    if frame_count == 0:
        return {"enabled": True, "frames_adjusted": 0, "reason": "empty clip"}

    skin = gltf.skins[0]
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index)
        for skin_index, node_index in enumerate(skin.joints)
    }
    phase_signal, phase_report = _source_leg_phase_signal(clip, frame_count)
    strength = float(np.clip(options.running_arm_swing_strength, 0.0, 1.0))
    adjusted_frames: set[int] = set()
    adjusted_channels = 0

    for frame_index, local in enumerate(local_quats_by_frame):
        globals_ = node_global_matrices(gltf, local)
        body_basis = _target_body_basis_from_globals(globals_, skin, joint_by_name)
        if body_basis is None:
            continue
        side_axis = body_basis[:, 0]
        up_axis = body_basis[:, 1]
        forward_axis = body_basis[:, 2]

        for calibration in calibrations:
            side_sign = -1.0 if calibration.side == "Left" else 1.0
            # Left arm moves backward when the left leg is forward; right arm mirrors it.
            swing = (-phase_signal[frame_index] if calibration.side == "Left" else phase_signal[frame_index])
            swing = float(np.clip(swing, -1.0, 1.0))
            shoulder = globals_[skin.joints[calibration.upper]][:3, 3]
            chain_len = calibration.upper_len + calibration.lower_len
            forward_ratio = float(max(options.running_arm_forward_ratio, 0.0))
            drop_ratio = float(max(options.running_arm_drop_ratio, 0.0))
            side_ratio = float(max(options.running_arm_side_ratio, 0.0))
            hand_target = (
                shoulder
                + side_axis * side_sign * (side_ratio * chain_len)
                - up_axis * (drop_ratio * chain_len)
                + forward_axis * (forward_ratio * chain_len * swing)
            )
            direction = _safe_unit(hand_target - shoulder)
            if direction is None:
                continue
            pole_hint = side_axis * side_sign * 0.85 - up_axis * 0.15
            pole = _projected_unit(pole_hint, direction)
            if pole is None:
                pole = _fallback_pole(direction)
            reach_ratio = float(
                np.clip(
                    np.linalg.norm(hand_target - shoulder) / max(chain_len, 1e-8),
                    options.running_arm_reach_min,
                    options.running_arm_reach_max,
                )
            )
            elbow, wrist = _two_bone_knee_position(
                shoulder,
                direction,
                pole,
                upper_len=calibration.upper_len,
                lower_len=calibration.lower_len,
                reach_ratio=reach_ratio,
            )

            parent_rot = (
                _rotation_from_matrix(globals_[calibration.parent_node])
                if calibration.parent_node is not None
                else Rotation.identity()
            )
            desired_upper = elbow - shoulder
            desired_upper_rot = _basis_rotation(
                calibration.rest_upper_local,
                calibration.rest_pole_upper_local,
                desired_upper,
                pole,
            )
            target_upper_delta = (
                local_rest_rot[calibration.upper].inv()
                * parent_rot.inv()
                * desired_upper_rot
            )
            current_upper_delta = Rotation.from_quat(local[calibration.upper])
            upper_delta = _blend_rotation(current_upper_delta, target_upper_delta, strength)
            local[calibration.upper] = upper_delta.as_quat()

            upper_global_rot = parent_rot * local_rest_rot[calibration.upper] * upper_delta
            desired_lower = wrist - elbow
            desired_forearm_rot = _basis_rotation(
                calibration.rest_lower_local,
                calibration.rest_pole_forearm_local,
                desired_lower,
                pole,
            )
            target_forearm_delta = (
                local_rest_rot[calibration.forearm].inv()
                * upper_global_rot.inv()
                * desired_forearm_rot
            )
            current_forearm_delta = Rotation.from_quat(local[calibration.forearm])
            local[calibration.forearm] = _blend_rotation(
                current_forearm_delta,
                target_forearm_delta,
                strength,
            ).as_quat()

            current_hand_delta = Rotation.from_quat(local[calibration.hand])
            local[calibration.hand] = _blend_rotation(
                current_hand_delta,
                Rotation.identity(),
                strength * 0.55,
            ).as_quat()
            adjusted_channels += 3
            adjusted_frames.add(frame_index)

    return {
        "enabled": True,
        "frames_adjusted": len(adjusted_frames),
        "channels_adjusted": adjusted_channels,
        "strength": strength,
        "target": {
            "forward_ratio": float(options.running_arm_forward_ratio),
            "drop_ratio": float(options.running_arm_drop_ratio),
            "side_ratio": float(options.running_arm_side_ratio),
            "reach_min": float(options.running_arm_reach_min),
            "reach_max": float(options.running_arm_reach_max),
        },
        "phase": phase_report,
        "strategy": (
            "derive a reciprocal arm phase from Kimodo left/right ankle forward offsets, "
            "then solve target-rig two-bone shoulder-elbow-hand IK with hands close to the torso"
        ),
    }


def _apply_locomotion_torso_counter_rotation(
    *,
    local_quats_by_frame: list[np.ndarray],
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    clip: MotionClip,
    prompt: str,
    options: RetargetOptions,
) -> dict[str, Any]:
    if not options.locomotion_torso_counter_rotation:
        return {"enabled": False, "reason": "disabled"}
    if not _prompt_requests_running_arm_swing(prompt):
        return {"enabled": False, "reason": "prompt is not locomotion-like"}
    frame_count = len(local_quats_by_frame)
    if frame_count == 0:
        return {"enabled": True, "frames_adjusted": 0, "reason": "empty clip"}

    skin = gltf.skins[0]
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index)
        for skin_index, node_index in enumerate(skin.joints)
    }
    phase_signal, phase_report = _source_leg_phase_signal(clip, frame_count)
    max_angle = np.radians(float(max(options.torso_counter_rotation_degrees, 0.0)))
    strength = float(np.clip(options.torso_counter_rotation_strength, 0.0, 1.0))
    channels = (
        ("Spine", -0.20),
        ("Spine1", -0.30),
        ("Spine2", -0.38),
        ("LeftShoulder", -0.16),
        ("RightShoulder", -0.16),
    )
    applied_channels = 0
    adjusted_frames: set[int] = set()

    for frame_index, local in enumerate(local_quats_by_frame):
        phase = float(np.clip(phase_signal[frame_index], -1.0, 1.0))
        if abs(phase) < 1e-5:
            continue
        for name, weight in channels:
            skin_index = joint_by_name.get(name)
            if skin_index is None or skin_index not in local_rest_rot:
                continue
            local_up = local_rest_rot[skin_index].inv().apply(GLTF_UP)
            local_up_unit = _safe_unit(local_up)
            if local_up_unit is None:
                continue
            angle = phase * max_angle * strength * weight
            offset = Rotation.from_rotvec(local_up_unit * angle)
            current = Rotation.from_quat(local[skin_index])
            local[skin_index] = (offset * current).as_quat()
            applied_channels += 1
            adjusted_frames.add(frame_index)

    return {
        "enabled": True,
        "frames_adjusted": len(adjusted_frames),
        "channels_adjusted": applied_channels,
        "strength": strength,
        "max_degrees": float(options.torso_counter_rotation_degrees),
        "phase": phase_report,
        "strategy": (
            "add a small leg-phase-driven counter-yaw to spine/chest/shoulders before arm cleanup, "
            "leaving root, pelvis, and legs untouched"
        ),
    }


def _apply_foot_contact_lock(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    *,
    ground_y: float,
    contact_height: float,
    blend_frames: int,
    contact_velocity: float | None = None,
    source_foot_contacts: np.ndarray | None = None,
    use_source_contacts: bool = True,
) -> dict[str, Any]:
    frame_count = len(frames)
    analysis = _foot_contact_analysis(
        frames,
        joint_by_name,
        ground_y=ground_y,
        contact_height=contact_height,
        contact_velocity=contact_velocity,
        source_foot_contacts=source_foot_contacts,
        use_source_contacts=use_source_contacts,
    )
    if analysis is None:
        return {"enabled": True, "locked_frames": 0, "reason": "missing foot joints"}

    centers = analysis.centers
    lows = analysis.lows
    contact_mask = analysis.contact_mask
    segments = analysis.segments

    locked_frames: set[int] = set()
    applied_segments: list[dict[str, Any]] = []
    frame_deltas: list[list[np.ndarray]] = [[] for _ in range(frame_count)]
    sides = ("Left", "Right")
    for side_index, side in enumerate(sides):
        side_segments = [
            (start, end)
            for start, end in _contiguous_segments(contact_mask[:, side_index])
            if end - start + 1 >= 2
        ]
        for start, end in side_segments:
            side_centers = centers[start : end + 1, side_index]
            side_lows = lows[start : end + 1, side_index]
            side_speeds = analysis.speeds[start : end + 1, side_index]
            valid = np.isfinite(side_centers).all(axis=1) & np.isfinite(side_lows)
            if not valid.any():
                continue
            anchor_xy = np.nanmedian(side_centers[valid][:, [0, 2]], axis=0)
            score = np.abs(side_lows - ground_y)
            if np.isfinite(side_speeds).any():
                score = score + np.nan_to_num(side_speeds, nan=np.nanmax(side_speeds[np.isfinite(side_speeds)])) * 8.0
            score[~valid] = np.inf
            anchor_frame = int(start + np.argmin(score))
            applied_segments.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "anchor_frame": anchor_frame,
                    "side": side,
                }
            )
            for frame_index in range(start, end + 1):
                current = centers[frame_index, side_index]
                low = lows[frame_index, side_index]
                if np.isnan(current).any() or np.isnan(low):
                    continue
                weight = _edge_blend_weight(frame_index, start, end, frame_count, blend_frames)
                if weight <= 0.0:
                    continue
                delta = np.array(
                    [
                        anchor_xy[0] - current[0],
                        ground_y - low,
                        anchor_xy[1] - current[2],
                    ],
                    dtype=np.float64,
                )
                frame_deltas[frame_index].append(delta * weight)

    for frame_index, deltas in enumerate(frame_deltas):
        if not deltas:
            continue
        _translate_frame(frames[frame_index], np.mean(deltas, axis=0))
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
        "contact_velocity": contact_velocity,
        "contact_frames_by_side": {
            "Left": int(contact_mask[:, 0].sum()),
            "Right": int(contact_mask[:, 1].sum()),
        },
        "source_contacts_used": analysis.source_contacts_used,
        "source_contact_ratio": analysis.source_contact_ratio,
        "strategy": "per-foot height+horizontal-speed contact locking",
        "blend_frames": blend_frames,
    }


def _apply_contact_foot_orientation_lock(
    *,
    local_quats_by_frame: list[np.ndarray],
    rough_frames: list[FrameGeometry],
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibrations: list[LegCalibration],
    clip: MotionClip,
    options: RetargetOptions,
) -> dict[str, Any]:
    if not calibrations:
        return {"enabled": True, "locked_frames": 0, "reason": "missing leg calibration"}

    analysis = _foot_contact_analysis(
        rough_frames,
        joint_by_name,
        ground_y=options.ground_y,
        contact_height=options.foot_contact_height,
        contact_velocity=options.foot_contact_velocity,
        source_foot_contacts=clip.foot_contacts,
        use_source_contacts=options.foot_contact_use_source,
    )
    if analysis is None:
        return {"enabled": True, "locked_frames": 0, "reason": "missing foot joints"}

    skin = gltf.skins[0]
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index) for skin_index, node_index in enumerate(skin.joints)
    }
    calibration_by_side = {calibration.side: calibration for calibration in calibrations}
    reference_globals = node_global_matrices(gltf, local_quats_by_frame[0])
    reference_forward = _body_forward_from_globals(reference_globals, gltf, skin, joint_by_name)

    reference_yaw: float | None = None
    if options.stabilize_root_yaw:
        for frame in rough_frames:
            if frame.joints is None:
                continue
            yaw = _root_yaw_from_joints(frame.joints.copy(), joint_by_name)
            if yaw is not None:
                reference_yaw = yaw
                break

    frame_count = len(local_quats_by_frame)
    locked_frames: set[int] = set()
    applied_segments: list[dict[str, Any]] = []

    for start, end in analysis.segments:
        if start > end:
            continue
        segment_mask = analysis.contact_mask[start : end + 1]
        side_indices = [index for index in range(2) if segment_mask[:, index].any()]
        if not side_indices:
            continue

        applied_segments.append(
            {
                "start": int(start),
                "end": int(end),
                "sides": ["Left" if index == 0 else "Right" for index in side_indices],
            }
        )
        for frame_index in range(start, end + 1):
            weight = _edge_blend_weight(frame_index, start, end, frame_count, options.foot_lock_blend_frames)
            if weight <= 0.0:
                continue
            globals_ = node_global_matrices(gltf, local_quats_by_frame[frame_index])
            for side_index in side_indices:
                if not analysis.contact_mask[frame_index, side_index]:
                    continue
                side = "Left" if side_index == 0 else "Right"
                calibration = calibration_by_side.get(side)
                if (
                    calibration is None
                    or calibration.rest_toe_local is None
                    or calibration.rest_foot_up_local is None
                ):
                    continue

                parent_rotation = _rotation_from_matrix(globals_[skin.joints[calibration.knee]])
                desired_toe_direction = reference_forward if reference_forward is not None else calibration.rest_contact_toe_global
                if desired_toe_direction is None:
                    continue
                if reference_yaw is not None and rough_frames[frame_index].joints is not None:
                    yaw = _root_yaw_from_joints(rough_frames[frame_index].joints.copy(), joint_by_name)
                    if yaw is not None:
                        yaw_delta = _wrap_angle(yaw - reference_yaw)
                        desired_toe_direction = Rotation.from_rotvec(GLTF_UP * yaw_delta).apply(
                            desired_toe_direction
                        )
                desired_foot_rotation = _basis_rotation(
                    calibration.rest_toe_local,
                    calibration.rest_foot_up_local,
                    desired_toe_direction,
                    GLTF_UP,
                )
                target_delta = local_rest_rot[calibration.ankle].inv() * parent_rotation.inv() * desired_foot_rotation
                current_delta = Rotation.from_quat(local_quats_by_frame[frame_index][calibration.ankle])
                local_quats_by_frame[frame_index][calibration.ankle] = _blend_rotation(
                    current_delta,
                    target_delta,
                    weight,
                ).as_quat()
                if calibration.toe is not None:
                    current_toe_delta = Rotation.from_quat(local_quats_by_frame[frame_index][calibration.toe])
                    local_quats_by_frame[frame_index][calibration.toe] = _blend_rotation(
                        current_toe_delta,
                        Rotation.identity(),
                        weight,
                    ).as_quat()
                locked_frames.add(frame_index)

    return {
        "enabled": True,
        "locked_frames": len(locked_frames),
        "segments": applied_segments,
        "contact_height": options.foot_contact_height,
        "contact_velocity": options.foot_contact_velocity,
        "source_contacts_used": analysis.source_contacts_used,
        "source_contact_ratio": analysis.source_contact_ratio,
        "strategy": (
            "align contact foot toe vector to the target rig body-facing direction, "
            "align foot up to ground normal, and neutralize toe-base curl"
        ),
        "blend_frames": options.foot_lock_blend_frames,
    }


def _target_point_to_source_axes(
    point: np.ndarray,
    *,
    normalizer: VertexNormalizer,
    root_offset: np.ndarray | None,
) -> np.ndarray:
    target = np.asarray(point, dtype=np.float64).copy()
    if root_offset is not None:
        target -= np.asarray(root_offset, dtype=np.float64)
    return np.array(
        [
            target[0] / normalizer.scale,
            target[2] / normalizer.scale,
            (target[1] - normalizer.target_ground_y) / normalizer.scale
            + normalizer.source_ground,
        ],
        dtype=np.float64,
    )


def _apply_leg_ik_target(
    *,
    local: np.ndarray,
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibration: LegCalibration,
    target_ankle: np.ndarray,
    weight: float,
    desired_toe_direction: np.ndarray | None,
    local_rest_rot: dict[int, Rotation],
) -> bool:
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return False

    skin = gltf.skins[0]
    globals_ = node_global_matrices(gltf, local)
    parent_rot = (
        _rotation_from_matrix(globals_[calibration.parent_node])
        if calibration.parent_node is not None
        else Rotation.identity()
    )
    hip_pos = globals_[skin.joints[calibration.hip]][:3, 3]
    current_ankle = globals_[skin.joints[calibration.ankle]][:3, 3]
    blended_ankle = current_ankle + (target_ankle - current_ankle) * weight
    direction = _safe_unit(blended_ankle - hip_pos)
    if direction is None:
        return False

    reach_ratio = float(
        np.linalg.norm(blended_ankle - hip_pos) / (calibration.upper_len + calibration.lower_len)
    )
    pole = parent_rot.apply(calibration.rest_pole_parent_local)
    pole = _projected_unit(pole, direction)
    if pole is None:
        knee_pos = globals_[skin.joints[calibration.knee]][:3, 3]
        pole = _limb_pole(hip_pos, knee_pos, current_ankle)
        pole = _projected_unit(pole, direction)
    if pole is None:
        pole = _fallback_pole(direction)

    knee_pos, ankle_pos = _two_bone_knee_position(
        hip_pos,
        direction,
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
    target_hip_delta = local_rest_rot[calibration.hip].inv() * parent_rot.inv() * desired_hip_rot
    current_hip_delta = Rotation.from_quat(local[calibration.hip])
    hip_delta = _blend_rotation(current_hip_delta, target_hip_delta, weight)
    local[calibration.hip] = hip_delta.as_quat()

    hip_global_rot = parent_rot * local_rest_rot[calibration.hip] * hip_delta
    desired_knee_rot = _basis_rotation(
        calibration.rest_lower_local,
        calibration.rest_pole_knee_local,
        desired_lower,
        pole,
    )
    target_knee_delta = local_rest_rot[calibration.knee].inv() * hip_global_rot.inv() * desired_knee_rot
    current_knee_delta = Rotation.from_quat(local[calibration.knee])
    knee_delta = _blend_rotation(current_knee_delta, target_knee_delta, weight)
    local[calibration.knee] = knee_delta.as_quat()

    if (
        desired_toe_direction is not None
        and calibration.rest_toe_local is not None
        and calibration.rest_foot_up_local is not None
    ):
        globals_ = node_global_matrices(gltf, local)
        knee_global_rot = _rotation_from_matrix(globals_[skin.joints[calibration.knee]])
        toe_direction = _projected_unit(desired_toe_direction, GLTF_UP)
        if toe_direction is not None:
            desired_foot_rotation = _basis_rotation(
                calibration.rest_toe_local,
                calibration.rest_foot_up_local,
                toe_direction,
                GLTF_UP,
            )
            target_foot_delta = (
                local_rest_rot[calibration.ankle].inv()
                * knee_global_rot.inv()
                * desired_foot_rotation
            )
            current_foot_delta = Rotation.from_quat(local[calibration.ankle])
            local[calibration.ankle] = _blend_rotation(
                current_foot_delta,
                target_foot_delta,
                weight,
            ).as_quat()

        if calibration.toe is not None:
            current_toe_delta = Rotation.from_quat(local[calibration.toe])
            local[calibration.toe] = _blend_rotation(
                current_toe_delta,
                Rotation.identity(),
                weight,
            ).as_quat()

    return True


def _apply_grounded_foot_ik(
    *,
    local_quats_by_frame: list[np.ndarray],
    rough_frames: list[FrameGeometry],
    gltf: GLTF2,
    joint_by_name: dict[str, int],
    calibrations: list[LegCalibration],
    clip: MotionClip,
    normalizer: VertexNormalizer,
    options: RetargetOptions,
) -> dict[str, Any]:
    if not calibrations:
        return {"enabled": True, "ik_frames": 0, "reason": "missing leg calibration"}

    analysis = _foot_contact_analysis(
        rough_frames,
        joint_by_name,
        ground_y=options.ground_y,
        contact_height=options.foot_contact_height,
        contact_velocity=options.foot_contact_velocity,
        source_foot_contacts=clip.foot_contacts,
        use_source_contacts=options.foot_contact_use_source,
    )
    if analysis is None:
        return {"enabled": True, "ik_frames": 0, "reason": "missing foot joints"}

    support_mask = _expand_support_contact_mask(
        analysis,
        fps=options.fps,
        ground_y=options.ground_y,
        contact_height=options.foot_contact_height,
        min_segment_frames=options.foot_support_min_frames,
        max_air_frames=options.foot_support_max_air_frames,
    )
    skin = gltf.skins[0]
    calibration_by_side = {calibration.side: calibration for calibration in calibrations}
    local_rest_rot = {
        skin_index: _local_rest_rotation(gltf, node_index)
        for skin_index, node_index in enumerate(skin.joints)
    }

    sides = ("Left", "Right")
    applied_segments: list[dict[str, Any]] = []
    ik_frames: set[int] = set()
    side_clearance = np.zeros(2, dtype=np.float64)
    side_center_offset = np.zeros(2, dtype=np.float64)
    for side_index in range(2):
        lows = analysis.lows[:, side_index]
        centers = analysis.centers[:, side_index]
        finite_low = np.isfinite(lows)
        if finite_low.any():
            clearance = float(np.nanpercentile(lows[finite_low] - options.ground_y, 5.0))
            side_clearance[side_index] = float(
                np.clip(clearance, 0.0, max(options.foot_contact_height, 0.02))
            )
        valid_center = np.isfinite(centers).all(axis=1) & finite_low
        if valid_center.any():
            side_center_offset[side_index] = float(
                np.nanmedian(centers[valid_center, 1] - lows[valid_center])
            )

    support_segment_count = 0
    internal_split_count = 0
    for side_index, side in enumerate(sides):
        calibration = calibration_by_side.get(side)
        if calibration is None:
            continue
        side_segments: list[tuple[int, int, int, int]] = []
        for support_start, support_end in _contiguous_segments(support_mask[:, side_index]):
            support_segment_count += 1
            split_segments = _split_long_segment(
                support_start,
                support_end,
                max(2, options.foot_support_max_frames),
            )
            internal_split_count += max(0, len(split_segments) - 1)
            side_segments.extend(
                (start, end, support_start, support_end)
                for start, end in split_segments
            )
        for start, end, support_start, support_end in side_segments:
            if end - start + 1 < 2:
                continue
            centers = analysis.centers[start : end + 1, side_index]
            lows = analysis.lows[start : end + 1, side_index]
            speeds = analysis.speeds[start : end + 1, side_index]
            valid = np.isfinite(centers).all(axis=1) & np.isfinite(lows)
            if not valid.any():
                continue

            anchor_xy = np.nanmedian(centers[valid][:, [0, 2]], axis=0)
            target_center_y = (
                options.ground_y
                + side_clearance[side_index]
                + side_center_offset[side_index]
            )
            score = np.abs(lows - (options.ground_y + side_clearance[side_index]))
            if np.isfinite(speeds).any():
                speed_fallback = float(np.nanmax(speeds[np.isfinite(speeds)]))
                score += np.nan_to_num(speeds, nan=speed_fallback) * 4.0
            score[~valid] = np.inf
            anchor_frame = int(start + np.argmin(score))
            anchor_center = np.array([anchor_xy[0], target_center_y, anchor_xy[1]], dtype=np.float64)
            anchor_local = local_quats_by_frame[anchor_frame]
            anchor_globals = node_global_matrices(gltf, anchor_local)
            anchor_ankle = anchor_globals[skin.joints[calibration.ankle]][:3, 3]
            if calibration.toe is not None:
                anchor_toe = anchor_globals[skin.joints[calibration.toe]][:3, 3]
                anchor_current_center = (anchor_ankle + anchor_toe) * 0.5
            else:
                anchor_current_center = anchor_ankle
            anchor_center_to_ankle = anchor_current_center - anchor_ankle
            anchor_forward = _body_forward_from_globals(
                anchor_globals,
                gltf,
                skin,
                joint_by_name,
            )
            if anchor_forward is None:
                anchor_forward = calibration.rest_contact_toe_global
            applied_segments.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "support_start": int(support_start),
                    "support_end": int(support_end),
                    "anchor_frame": anchor_frame,
                    "side": side,
                }
            )

            for frame_index in range(start, end + 1):
                weight = _edge_blend_weight(
                    frame_index,
                    support_start,
                    support_end,
                    len(local_quats_by_frame),
                    options.foot_lock_blend_frames,
                )
                if weight <= 0.0:
                    continue
                local = local_quats_by_frame[frame_index]
                root_offset = _root_offset(
                    clip,
                    frame_index,
                    options=options,
                    normalizer=normalizer,
                )
                target_center = _target_point_to_source_axes(
                    anchor_center,
                    normalizer=normalizer,
                    root_offset=root_offset,
                )
                target_ankle = target_center - anchor_center_to_ankle
                if _apply_leg_ik_target(
                    local=local,
                    gltf=gltf,
                    joint_by_name=joint_by_name,
                    calibration=calibration,
                    target_ankle=target_ankle,
                    weight=weight,
                    desired_toe_direction=anchor_forward,
                    local_rest_rot=local_rest_rot,
                ):
                    ik_frames.add(frame_index)

    return {
        "enabled": True,
        "ik_frames": len(ik_frames),
        "segments": applied_segments,
        "raw_contact_frames_by_side": {
            "Left": int(analysis.contact_mask[:, 0].sum()),
            "Right": int(analysis.contact_mask[:, 1].sum()),
        },
        "support_frames_by_side": {
            "Left": int(support_mask[:, 0].sum()),
            "Right": int(support_mask[:, 1].sum()),
        },
        "max_airborne_gap_before": _max_airborne_gap(analysis.contact_mask),
        "max_airborne_gap_after": _max_airborne_gap(support_mask),
        "support_segment_count": support_segment_count,
        "internal_split_count": internal_split_count,
        "contact_switches_before": int(np.abs(np.diff(analysis.contact_mask.astype(np.int8), axis=0)).sum()),
        "contact_switches_after": int(np.abs(np.diff(support_mask.astype(np.int8), axis=0)).sum()),
        "source_contacts_used": analysis.source_contacts_used,
        "source_contact_ratio": analysis.source_contact_ratio,
        "strategy": (
            "expand low-foot gait phases into alternating support windows, pin each support foot "
            "with target-rig two-bone IK, and align contact foot/toe to body-facing ground plane"
        ),
        "blend_frames": options.foot_lock_blend_frames,
    }


def _postprocess_frames(
    frames: list[FrameGeometry],
    joint_by_name: dict[str, int],
    clip: MotionClip,
    options: RetargetOptions,
    *,
    apply_contact_lock: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "root_yaw_stabilization": {"enabled": False},
        "foot_contact_lock": {"enabled": False},
        "foot_orientation_lock": {"enabled": False},
    }
    if options.stabilize_root_yaw:
        report["root_yaw_stabilization"] = _apply_root_yaw_stabilization(frames, joint_by_name)
    if options.foot_contact_lock and apply_contact_lock:
        report["foot_contact_lock"] = _apply_foot_contact_lock(
            frames,
            joint_by_name,
            ground_y=options.ground_y,
            contact_height=options.foot_contact_height,
            contact_velocity=options.foot_contact_velocity,
            blend_frames=options.foot_lock_blend_frames,
            source_foot_contacts=clip.foot_contacts,
            use_source_contacts=options.foot_contact_use_source,
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


def _build_frame_geometries(
    *,
    gltf: GLTF2,
    skin: Any,
    local_quats_by_frame: list[np.ndarray],
    normalizer: VertexNormalizer,
    clip: MotionClip,
    options: RetargetOptions,
) -> list[FrameGeometry]:
    frames: list[FrameGeometry] = []
    for frame_index, local_quats in enumerate(local_quats_by_frame):
        deformed = deform_skinned_primitives(gltf, local_quats)
        globals_ = node_global_matrices(gltf, local_quats)
        joint_points = np.stack([globals_[joint][:3, 3] for joint in skin.joints])
        transformed, transformed_joints = normalizer.transform(
            deformed,
            joint_points,
            root_offset=_root_offset(clip, frame_index, options=options, normalizer=normalizer),
            snap_to_ground=options.snap_to_ground,
        )
        frames.append(FrameGeometry(vertices_by_primitive=transformed, joints=transformed_joints))
    return frames


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
    arm_calibrations = _arm_calibrations(gltf, joint_by_name, rest_globals)
    leg_ik_enabled = options.calibrated_leg_ik and clip.posed_joints is not None and bool(leg_calibrations)
    arm_ik_enabled = options.calibrated_arm_ik and clip.posed_joints is not None and bool(arm_calibrations)

    rest_vertices = deform_skinned_primitives(gltf, rest_joint_deltas)
    normalizer = VertexNormalizer.from_vertices(
        rest_vertices, target_height=options.target_height, ground_y=options.ground_y
    )

    material_changes: dict[str, Any] = {}
    bounds: list[dict[str, list[float]]] = []
    joint_names = [joint_base_name(gltf.nodes[joint].name) for joint in skin.joints]

    def build_local_quats_by_frame(
        leg_guides: list[dict[str, AirborneLegGuide] | None] | None = None,
    ) -> list[np.ndarray]:
        quats_by_frame: list[np.ndarray] = []
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
                        source_global_rot_mats=clip.global_rot_mats[frame_index]
                        if clip.global_rot_mats is not None
                        else None,
                        gltf=gltf,
                        joint_by_name=joint_by_name,
                        calibrations=leg_calibrations,
                        body_relative=options.body_relative_leg_ik,
                        leg_guide=leg_guides[frame_index] if leg_guides is not None else None,
                    )
                if arm_ik_enabled:
                    local_quats = _apply_calibrated_arm_ik(
                        local=local_quats,
                        source_joints=clip.posed_joints[frame_index],
                        gltf=gltf,
                        joint_by_name=joint_by_name,
                        calibrations=arm_calibrations,
                    )
            quats_by_frame.append(local_quats)
        return quats_by_frame

    local_quats_by_frame = build_local_quats_by_frame()

    pre_postprocess: dict[str, Any] = {
        "airborne_leg_stabilization": {"enabled": False},
        "calibrated_arm_ik": {"enabled": arm_ik_enabled},
        "torso_counter_rotation": {"enabled": False},
        "running_arm_swing": {"enabled": False},
        "foot_orientation_lock": {"enabled": False},
        "grounded_foot_ik": {"enabled": False},
    }
    if options.airborne_leg_stabilization and leg_ik_enabled:
        rough_frames = _build_frame_geometries(
            gltf=gltf,
            skin=skin,
            local_quats_by_frame=local_quats_by_frame,
            normalizer=normalizer,
            clip=clip,
            options=options,
        )
        leg_guides, air_report = _airborne_leg_guides(
            clip=clip,
            rough_frames=rough_frames,
            joint_by_name=joint_by_name,
            options=options,
        )
        pre_postprocess["airborne_leg_stabilization"] = air_report
        if air_report.get("guided_frames", 0) > 0:
            local_quats_by_frame = build_local_quats_by_frame(leg_guides)

    pre_postprocess["torso_counter_rotation"] = _apply_locomotion_torso_counter_rotation(
        local_quats_by_frame=local_quats_by_frame,
        gltf=gltf,
        joint_by_name=joint_by_name,
        clip=clip,
        prompt=prompt,
        options=options,
    )

    pre_postprocess["running_arm_swing"] = _apply_procedural_running_arm_swing(
        local_quats_by_frame=local_quats_by_frame,
        gltf=gltf,
        joint_by_name=joint_by_name,
        calibrations=arm_calibrations,
        clip=clip,
        prompt=prompt,
        options=options,
    )

    grounded_foot_ik_expected = options.grounded_foot_ik and leg_ik_enabled
    if options.foot_contact_lock and options.foot_orientation_lock and grounded_foot_ik_expected:
        pre_postprocess["foot_orientation_lock"] = {
            "enabled": True,
            "locked_frames": 0,
            "reason": "handled inside grounded foot IK support segments",
        }
    elif options.foot_contact_lock and options.foot_orientation_lock:
        rough_frames = _build_frame_geometries(
            gltf=gltf,
            skin=skin,
            local_quats_by_frame=local_quats_by_frame,
            normalizer=normalizer,
            clip=clip,
            options=options,
        )
        pre_postprocess["foot_orientation_lock"] = _apply_contact_foot_orientation_lock(
            local_quats_by_frame=local_quats_by_frame,
            rough_frames=rough_frames,
            gltf=gltf,
            joint_by_name=joint_by_name,
            calibrations=leg_calibrations,
            clip=clip,
            options=options,
        )

    grounded_foot_ik_applied = False
    if options.foot_contact_lock and options.grounded_foot_ik and leg_ik_enabled:
        rough_frames = _build_frame_geometries(
            gltf=gltf,
            skin=skin,
            local_quats_by_frame=local_quats_by_frame,
            normalizer=normalizer,
            clip=clip,
            options=options,
        )
        ground_ik_report = _apply_grounded_foot_ik(
            local_quats_by_frame=local_quats_by_frame,
            rough_frames=rough_frames,
            gltf=gltf,
            joint_by_name=joint_by_name,
            calibrations=leg_calibrations,
            clip=clip,
            normalizer=normalizer,
            options=options,
        )
        grounded_foot_ik_applied = ground_ik_report.get("ik_frames", 0) > 0
        pre_postprocess["grounded_foot_ik"] = ground_ik_report

    frame_geometries = _build_frame_geometries(
        gltf=gltf,
        skin=skin,
        local_quats_by_frame=local_quats_by_frame,
        normalizer=normalizer,
        clip=clip,
        options=options,
    )
    postprocess = _postprocess_frames(
        frame_geometries,
        joint_by_name,
        clip,
        options,
        apply_contact_lock=not grounded_foot_ik_applied,
    )
    postprocess = {**postprocess, **pre_postprocess}
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
            "strategy": (
                "body-relative calibrated two-bone IK with target-rig knee pole constraints"
                if options.body_relative_leg_ik
                else "calibrated two-bone IK with target-rig knee pole constraints"
            ),
            "legs": [calibration.side for calibration in leg_calibrations],
            "body_relative": options.body_relative_leg_ik,
        },
        "arm_ik": {
            "enabled": arm_ik_enabled,
            "strategy": "calibrated two-bone IK from SMPL posed shoulder/elbow/wrist positions",
            "arms": [calibration.side for calibration in arm_calibrations],
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
            "when posed joints are available, solve calibrated two-bone leg and arm IK using target rig poles; "
            "body-relative leg mapping and airborne leg stabilization are experimental opt-in modes; "
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
        "arm_ik": skeleton["arm_ik"],
        "postprocess": postprocess,
    }
    return report
