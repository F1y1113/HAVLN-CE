from __future__ import annotations

from pathlib import Path

import numpy as np

from havln3.motion import MotionClip
from havln3.motion_quality import MotionQualityOptions, score_motion_clip


def _synthetic_clip(angles: np.ndarray, *, fold_midair: bool = False) -> MotionClip:
    frames = len(angles)
    posed = np.zeros((frames, 22, 3), dtype=np.float64)
    for frame_index, angle in enumerate(angles):
        pelvis = np.array([0.0, 0.0, 1.0])
        side = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, np.sin(angle), np.cos(angle)])
        forward = np.array([0.0, np.cos(angle), -np.sin(angle)])
        posed_frame = np.zeros((22, 3), dtype=np.float64)
        posed_frame[0] = pelvis
        posed_frame[3] = pelvis + up * 0.25
        posed_frame[6] = pelvis + up * 0.42
        posed_frame[9] = pelvis + up * 0.58
        posed_frame[12] = pelvis + up * 0.68
        posed_frame[15] = pelvis + up * 0.82
        posed_frame[13] = posed_frame[9] - side * 0.18
        posed_frame[14] = posed_frame[9] + side * 0.18
        posed_frame[16] = posed_frame[13] - side * 0.08 - up * 0.1
        posed_frame[17] = posed_frame[14] + side * 0.08 - up * 0.1
        posed_frame[18] = posed_frame[16] - side * 0.1 - up * 0.18
        posed_frame[19] = posed_frame[17] + side * 0.1 - up * 0.18
        posed_frame[20] = posed_frame[18] - side * 0.08 - up * 0.14
        posed_frame[21] = posed_frame[19] + side * 0.08 - up * 0.14
        posed_frame[1] = pelvis - side * 0.11 - up * 0.05
        posed_frame[2] = pelvis + side * 0.11 - up * 0.05
        tuck = np.sin(np.pi * frame_index / max(frames - 1, 1))
        for side_name, sign, hip_index, knee_index, ankle_index, toe_index in (
            ("Left", -1.0, 1, 4, 7, 10),
            ("Right", 1.0, 2, 5, 8, 11),
        ):
            del side_name
            hip = posed_frame[hip_index]
            if fold_midair and abs(frame_index - frames // 2) <= 1:
                ankle = posed_frame[15] - up * 0.04 + side * sign * 0.02
                knee = (hip + ankle) * 0.5 + forward * 0.02
            else:
                leg_dir = -up * (1.0 - tuck * 0.45) + forward * (tuck * 0.45)
                leg_dir = leg_dir / np.linalg.norm(leg_dir)
                knee = hip + leg_dir * 0.38
                ankle = hip + leg_dir * 0.76
            posed_frame[knee_index] = knee
            posed_frame[ankle_index] = ankle
            posed_frame[toe_index] = ankle + forward * 0.12
            if fold_midair and abs(frame_index - frames // 2) <= 1:
                wrist_index = 20 if sign < 0 else 21
                posed_frame[wrist_index] = ankle + side * sign * 0.01
        posed[frame_index] = posed_frame
    mats = np.tile(np.eye(3, dtype=np.float64), (frames, 22, 1, 1))
    return MotionClip(
        local_rot_mats=mats,
        root_positions=np.zeros((frames, 3), dtype=np.float64),
        posed_joints=posed,
        foot_contacts=None,
        source_path=Path("synthetic.npz"),
    )


def test_motion_quality_prefers_complete_backflip_over_partial_rotation() -> None:
    full = _synthetic_clip(np.linspace(0.0, -2.0 * np.pi, 48))
    partial = _synthetic_clip(np.concatenate([np.linspace(0.0, -1.0 * np.pi, 24), np.linspace(-1.0 * np.pi, 0.0, 24)]))

    options = MotionQualityOptions(frames=48)
    full_report = score_motion_clip(full, prompt="A person performs a clean standing backflip.", options=options)
    partial_report = score_motion_clip(partial, prompt="A person performs a clean standing backflip.", options=options)

    assert full_report.score > partial_report.score
    assert full_report.metrics["backflip_total_degrees"] > 300
    assert "backflip does not complete enough net rotation" in partial_report.reasons


def test_motion_quality_penalizes_midair_body_fold() -> None:
    clean = _synthetic_clip(np.linspace(0.0, -2.0 * np.pi, 48))
    folded = _synthetic_clip(np.linspace(0.0, -2.0 * np.pi, 48), fold_midair=True)

    options = MotionQualityOptions(frames=48)
    clean_report = score_motion_clip(clean, prompt="A person performs a clean standing backflip.", options=options)
    folded_report = score_motion_clip(folded, prompt="A person performs a clean standing backflip.", options=options)

    assert clean_report.score > folded_report.score
    assert folded_report.metrics["foot_head_distance_min_ratio"] < clean_report.metrics["foot_head_distance_min_ratio"]
    assert "foot gets too close to head/face" in folded_report.reasons


def test_motion_quality_penalizes_self_contact_tuck() -> None:
    clean = _synthetic_clip(np.linspace(0.0, -2.0 * np.pi, 48))
    folded = _synthetic_clip(np.linspace(0.0, -2.0 * np.pi, 48), fold_midair=True)

    options = MotionQualityOptions(frames=48)
    clean_report = score_motion_clip(clean, prompt="A person performs a clean standing backflip.", options=options)
    folded_report = score_motion_clip(folded, prompt="A person performs a clean standing backflip.", options=options)

    assert folded_report.metrics["hand_foot_clearance_min_ratio"] < clean_report.metrics["hand_foot_clearance_min_ratio"]
    assert folded_report.metrics["limb_core_clearance_min_ratio"] < clean_report.metrics["limb_core_clearance_min_ratio"]
    assert "hands and feet get too close during tuck" in folded_report.reasons
    assert "limbs get too close to torso/head" in folded_report.reasons
