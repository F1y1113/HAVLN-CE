from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from havln3.retarget import (
    FrameGeometry,
    _apply_foot_contact_lock,
    _apply_root_yaw_stabilization,
    _source_body_relative_direction,
)


JOINTS = {
    "Hips": 0,
    "LeftUpLeg": 1,
    "RightUpLeg": 2,
    "LeftFoot": 3,
    "LeftToeBase": 4,
    "RightFoot": 5,
    "RightToeBase": 6,
}


def _frame(joints: np.ndarray) -> FrameGeometry:
    vertices = joints + np.array([0.0, -0.02, 0.0])
    return FrameGeometry(vertices_by_primitive=[vertices.copy()], joints=joints.copy())


def test_root_yaw_stabilization_keeps_initial_horizontal_facing() -> None:
    frames = [
        _frame(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.2],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.2],
                ]
            )
        ),
        _frame(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, -1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [0.2, 0.0, -1.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.0, 1.0],
                ]
            )
        ),
    ]

    report = _apply_root_yaw_stabilization(frames, JOINTS)
    stabilized_side = frames[1].joints[JOINTS["RightUpLeg"]] - frames[1].joints[JOINTS["LeftUpLeg"]]

    assert report["frames_adjusted"] == 1
    assert stabilized_side[0] > 1.9
    assert abs(stabilized_side[2]) < 1e-6


def test_foot_contact_lock_anchors_landing_foot_position() -> None:
    frames = []
    for foot_x in (0.0, 0.1, 0.2, 0.65):
        joints = np.array(
            [
                [foot_x, 1.0, 0.0],
                [foot_x - 0.2, 0.8, 0.0],
                [foot_x + 0.2, 0.8, 0.0],
                [foot_x - 0.2, 0.02, 0.0],
                [foot_x - 0.2, 0.01, 0.2],
                [foot_x + 0.2, 0.02, 0.0],
                [foot_x + 0.2, 0.01, 0.2],
            ],
            dtype=np.float64,
        )
        frames.append(_frame(joints))

    report = _apply_foot_contact_lock(
        frames,
        JOINTS,
        ground_y=0.0,
        contact_height=0.05,
        blend_frames=0,
    )
    left_toe = frames[-1].joints[JOINTS["LeftToeBase"]]
    left_anchor = frames[0].joints[JOINTS["LeftToeBase"]]

    assert report["locked_frames"] == 4
    assert np.allclose(left_toe[[0, 2]], left_anchor[[0, 2]])
    assert left_toe[1] >= 0.0
    assert abs(frames[-1].vertices_by_primitive[0][:, 1].min() - 0.0) < 1e-6


def test_body_relative_leg_direction_follows_target_parent_frame() -> None:
    source_parent = Rotation.from_euler("y", 90, degrees=True)
    target_parent = Rotation.from_euler("y", -45, degrees=True)
    source_local_leg = np.array([0.0, -1.0, 0.25], dtype=np.float64)
    source_world_leg = source_parent.apply(source_local_leg)

    mapped = _source_body_relative_direction(
        source_world_leg,
        source_global_rotations={0: source_parent},
        source_parent_index=0,
        target_parent_rotation=target_parent,
    )

    expected = target_parent.apply(source_local_leg / np.linalg.norm(source_local_leg))
    assert mapped is not None
    assert np.allclose(mapped, expected)
    assert not np.allclose(mapped, source_world_leg / np.linalg.norm(source_world_leg))
