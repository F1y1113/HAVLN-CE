from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from havln3.motion import MotionClip, load_motion_file, resample_motion


SMPL_LEG_JOINTS = {
    "Left": {"hip": 1, "knee": 4, "ankle": 7, "toe": 10},
    "Right": {"hip": 2, "knee": 5, "ankle": 8, "toe": 11},
}


@dataclass(frozen=True)
class MotionQualityOptions:
    frames: int = 60
    min_score: float = 62.0
    min_backflip_rotation_degrees: float = 280.0
    min_backflip_total_degrees: float = 240.0
    max_leg_direction_acceleration: float = 0.52
    max_leg_direction_velocity: float = 1.65
    min_leg_reach_ratio: float = 0.22
    min_knee_angle_degrees: float = 22.0
    min_foot_head_distance_ratio: float = 0.11
    max_root_drift_ratio: float = 1.15


@dataclass(frozen=True)
class MotionQualityReport:
    path: str
    score: float
    passed: bool
    reasons: list[str]
    metrics: dict[str, float | int | bool | str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "passed": self.passed,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _source_to_quality_axes(points: np.ndarray) -> np.ndarray:
    return points[:, [0, 2, 1]]


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _projected_unit(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    projected = vector - normal * float(np.dot(vector, normal))
    return _safe_unit(projected)


def _orthonormal_body_basis(points: np.ndarray) -> np.ndarray | None:
    if len(points) <= 3:
        return None
    hips = points[0]
    side = _safe_unit(points[SMPL_LEG_JOINTS["Right"]["hip"]] - points[SMPL_LEG_JOINTS["Left"]["hip"]])
    if side is None:
        return None
    for spine_index in (9, 6, 3, 15):
        if spine_index >= len(points):
            continue
        up = _projected_unit(points[spine_index] - hips, side)
        if up is None:
            continue
        forward = _safe_unit(np.cross(side, up))
        if forward is None:
            continue
        up = _safe_unit(np.cross(forward, side))
        if up is None:
            continue
        return np.column_stack([side, up, forward])
    return None


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float | None:
    a_unit = _safe_unit(a)
    b_unit = _safe_unit(b)
    if a_unit is None or b_unit is None:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a_unit, b_unit)), -1.0, 1.0))))


def _is_backflip_prompt(prompt: str | None) -> bool:
    text = (prompt or "").lower()
    return "backflip" in text or "back flip" in text or "后空翻" in text


def _is_stationary_prompt(prompt: str | None) -> bool:
    text = (prompt or "").lower()
    return any(token in text for token in ("standing", "stationary", "in place", "原地"))


def _finite_or_default(value: float | None, default: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def _penalize_above(value: float, limit: float, scale: float) -> float:
    return max(0.0, value - limit) * scale


def _penalize_below(value: float, limit: float, scale: float) -> float:
    return max(0.0, limit - value) * scale


def _motion_points(clip: MotionClip, frames: int) -> np.ndarray | None:
    if clip.posed_joints is None:
        return None
    sampled = resample_motion(clip, frames)
    if sampled.posed_joints is None:
        return None
    return np.asarray([_source_to_quality_axes(frame) for frame in sampled.posed_joints], dtype=np.float64)


def _body_height(points: np.ndarray) -> float:
    extents = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
    heights = np.linalg.norm(extents, axis=1)
    return max(float(np.nanmedian(heights)), 1e-6)


def _backflip_metrics(points: np.ndarray) -> dict[str, float]:
    first_basis = _orthonormal_body_basis(points[0])
    if first_basis is None:
        return {
            "backflip_rotation_span_degrees": 0.0,
            "backflip_total_degrees": 0.0,
            "backflip_direction_consistency": 0.0,
            "upside_down_dot": 1.0,
            "final_upright_dot": 0.0,
        }
    up0 = first_basis[:, 1]
    forward0 = first_basis[:, 2]
    angles: list[float] = []
    up_dots: list[float] = []
    for frame in points:
        up = _safe_unit(frame[15] - frame[0])
        if up is None:
            continue
        up_dots.append(float(np.dot(up, up0)))
        angles.append(float(np.arctan2(np.dot(up, forward0), np.dot(up, up0))))
    if len(angles) < 2:
        return {
            "backflip_rotation_span_degrees": 0.0,
            "backflip_total_degrees": 0.0,
            "backflip_direction_consistency": 0.0,
            "upside_down_dot": 1.0,
            "final_upright_dot": 0.0,
        }
    unwrapped = np.unwrap(np.asarray(angles, dtype=np.float64))
    total = float(np.degrees(unwrapped[-1] - unwrapped[0]))
    span = float(np.degrees(np.nanmax(unwrapped) - np.nanmin(unwrapped)))
    direction = np.sign(total) if abs(total) > 1e-6 else 1.0
    deltas = np.diff(unwrapped) * direction
    consistency = float(np.mean(deltas > -0.08)) if len(deltas) else 0.0
    return {
        "backflip_rotation_span_degrees": span,
        "backflip_total_degrees": abs(total),
        "backflip_direction_consistency": consistency,
        "upside_down_dot": float(np.nanmin(up_dots)) if up_dots else 1.0,
        "final_upright_dot": float(up_dots[-1]) if up_dots else 0.0,
    }


def _leg_metrics(points: np.ndarray, body_height: float) -> dict[str, float]:
    frame_count = len(points)
    local_dirs = np.full((frame_count, 2, 3), np.nan, dtype=np.float64)
    reach_ratios = np.full((frame_count, 2), np.nan, dtype=np.float64)
    knee_angles: list[float] = []
    foot_head_distances: list[float] = []
    asymmetry: list[float] = []

    for frame_index, frame in enumerate(points):
        basis = _orthonormal_body_basis(frame)
        if basis is None:
            continue
        mirrored_dirs: list[np.ndarray] = []
        for side_index, side in enumerate(("Left", "Right")):
            joints = SMPL_LEG_JOINTS[side]
            hip = frame[joints["hip"]]
            knee = frame[joints["knee"]]
            ankle = frame[joints["ankle"]]
            toe = frame[joints["toe"]]
            direction = _safe_unit(ankle - hip)
            upper_len = float(np.linalg.norm(knee - hip))
            lower_len = float(np.linalg.norm(ankle - knee))
            chain_len = upper_len + lower_len
            if direction is not None:
                local_direction = _safe_unit(basis.T @ direction)
                if local_direction is not None:
                    local_dirs[frame_index, side_index] = local_direction
                    mirrored = local_direction.copy()
                    if side == "Right":
                        mirrored[0] *= -1.0
                    mirrored_dirs.append(mirrored)
            if chain_len > 1e-8:
                reach_ratios[frame_index, side_index] = float(np.linalg.norm(ankle - hip) / chain_len)
            knee_angle = _angle_degrees(hip - knee, ankle - knee)
            if knee_angle is not None:
                knee_angles.append(knee_angle)
            head = frame[15]
            foot_head_distances.extend(
                [
                    float(np.linalg.norm(ankle - head) / body_height),
                    float(np.linalg.norm(toe - head) / body_height),
                ]
            )
        if len(mirrored_dirs) == 2:
            asymmetry.append(float(np.linalg.norm(mirrored_dirs[0] - mirrored_dirs[1])))

    valid_dirs = np.nan_to_num(local_dirs, nan=0.0)
    direction_velocity = np.diff(valid_dirs, axis=0).reshape(max(frame_count - 1, 0), -1)
    direction_acceleration = np.diff(valid_dirs, n=2, axis=0).reshape(max(frame_count - 2, 0), -1)
    return {
        "leg_direction_velocity_max": float(np.nanmax(np.linalg.norm(direction_velocity, axis=1)))
        if len(direction_velocity)
        else 0.0,
        "leg_direction_acceleration_mean": float(np.nanmean(np.linalg.norm(direction_acceleration, axis=1)))
        if len(direction_acceleration)
        else 0.0,
        "leg_reach_ratio_min": float(np.nanmin(reach_ratios)) if not np.isnan(reach_ratios).all() else 0.0,
        "knee_angle_min_degrees": float(np.nanmin(knee_angles)) if knee_angles else 0.0,
        "foot_head_distance_min_ratio": float(np.nanmin(foot_head_distances)) if foot_head_distances else 0.0,
        "leg_asymmetry_mean": float(np.nanmean(asymmetry)) if asymmetry else 0.0,
    }


def _shape_jerk(points: np.ndarray, body_height: float) -> float:
    local = points - points[:, :1]
    acceleration = np.diff(local, n=2, axis=0)
    if len(acceleration) == 0:
        return 0.0
    return float(np.nanmean(np.linalg.norm(acceleration.reshape(len(acceleration), -1), axis=1)) / body_height)


def _root_drift_ratio(clip: MotionClip, frames: int, body_height: float) -> float:
    if clip.root_positions is None:
        return 0.0
    root = resample_motion(clip, frames).root_positions
    if root is None or len(root) < 2:
        return 0.0
    horizontal = root[:, [0, 1]]
    drift = float(np.linalg.norm(horizontal[-1] - horizontal[0]))
    return drift / body_height


def score_motion_clip(
    clip: MotionClip,
    *,
    prompt: str | None = None,
    options: MotionQualityOptions | None = None,
) -> MotionQualityReport:
    options = options or MotionQualityOptions()
    points = _motion_points(clip, options.frames)
    reasons: list[str] = []
    if points is None:
        return MotionQualityReport(
            path=str(clip.source_path),
            score=0.0,
            passed=False,
            reasons=["missing posed_joints; cannot evaluate Kimodo skeleton quality"],
            metrics={"has_posed_joints": False},
        )
    finite_ratio = float(np.isfinite(points).mean())
    body_height = _body_height(points)
    metrics: dict[str, float | int | bool | str | None] = {
        "has_posed_joints": True,
        "finite_ratio": finite_ratio,
        "body_height": body_height,
        "frames": int(options.frames),
    }
    metrics.update(_backflip_metrics(points))
    metrics.update(_leg_metrics(points, body_height))
    metrics["shape_jerk"] = _shape_jerk(points, body_height)
    metrics["root_drift_ratio"] = _root_drift_ratio(clip, options.frames, body_height)

    score = 100.0
    if finite_ratio < 0.999:
        score -= (0.999 - finite_ratio) * 250.0
        reasons.append("non-finite joint values")

    score -= _penalize_above(
        float(metrics["shape_jerk"]),
        0.12,
        65.0,
    )
    score -= _penalize_above(
        float(metrics["leg_direction_acceleration_mean"]),
        options.max_leg_direction_acceleration,
        26.0,
    )
    score -= _penalize_above(
        float(metrics["leg_direction_velocity_max"]),
        options.max_leg_direction_velocity,
        16.0,
    )
    score -= _penalize_below(
        float(metrics["leg_reach_ratio_min"]),
        options.min_leg_reach_ratio,
        80.0,
    )
    score -= _penalize_below(
        float(metrics["knee_angle_min_degrees"]),
        options.min_knee_angle_degrees,
        0.85,
    )
    score -= _penalize_below(
        float(metrics["foot_head_distance_min_ratio"]),
        options.min_foot_head_distance_ratio,
        115.0,
    )
    score -= _penalize_above(float(metrics["leg_asymmetry_mean"]), 0.62, 14.0)

    if _is_backflip_prompt(prompt or clip.prompt):
        span = float(metrics["backflip_rotation_span_degrees"])
        total = float(metrics["backflip_total_degrees"])
        consistency = float(metrics["backflip_direction_consistency"])
        upside_down_dot = float(metrics["upside_down_dot"])
        final_upright_dot = float(metrics["final_upright_dot"])
        if span < options.min_backflip_rotation_degrees:
            score -= (options.min_backflip_rotation_degrees - span) * 0.18
            reasons.append("backflip rotation span is too small")
        if total < options.min_backflip_total_degrees:
            score -= (options.min_backflip_total_degrees - total) * 0.14
            reasons.append("backflip does not complete enough net rotation")
        if consistency < 0.58:
            score -= (0.58 - consistency) * 28.0
            reasons.append("backflip direction is inconsistent")
        if upside_down_dot > -0.35:
            score -= (upside_down_dot + 0.35) * 24.0
            reasons.append("motion never reaches a clear inverted torso phase")
        if final_upright_dot < 0.45:
            score -= (0.45 - final_upright_dot) * 20.0
            reasons.append("motion does not return near upright")

    if _is_stationary_prompt(prompt or clip.prompt):
        drift = float(metrics["root_drift_ratio"])
        if drift > options.max_root_drift_ratio:
            score -= (drift - options.max_root_drift_ratio) * 18.0
            reasons.append("root drifts too far for a stationary prompt")

    if float(metrics["leg_reach_ratio_min"]) < options.min_leg_reach_ratio:
        reasons.append("leg chain collapses too tightly")
    if float(metrics["foot_head_distance_min_ratio"]) < options.min_foot_head_distance_ratio:
        reasons.append("foot gets too close to head/face")
    if float(metrics["knee_angle_min_degrees"]) < options.min_knee_angle_degrees:
        reasons.append("knee fold angle is too extreme")
    if float(metrics["leg_direction_acceleration_mean"]) > options.max_leg_direction_acceleration:
        reasons.append("airborne leg direction changes too abruptly")

    score = float(np.clip(score, 0.0, 100.0))
    passed = score >= options.min_score and not any(
        reason
        in {
            "missing posed_joints; cannot evaluate Kimodo skeleton quality",
            "non-finite joint values",
        }
        for reason in reasons
    )
    return MotionQualityReport(
        path=str(clip.source_path),
        score=score,
        passed=passed,
        reasons=reasons,
        metrics={key: _finite_or_default(value) if isinstance(value, float) else value for key, value in metrics.items()},
    )


def score_motion_file(
    path: Path,
    *,
    prompt: str | None = None,
    options: MotionQualityOptions | None = None,
    gem_param_group: str = "body_params_global",
) -> MotionQualityReport:
    clip = load_motion_file(path, gem_param_group=gem_param_group)
    return score_motion_clip(clip, prompt=prompt, options=options)


def rank_motion_files(
    paths: list[Path],
    *,
    prompt: str | None = None,
    options: MotionQualityOptions | None = None,
    gem_param_group: str = "body_params_global",
) -> list[MotionQualityReport]:
    reports = [
        score_motion_file(
            path,
            prompt=prompt,
            options=options,
            gem_param_group=gem_param_group,
        )
        for path in paths
    ]
    return sorted(reports, key=lambda report: report.score, reverse=True)


def write_motion_quality_report(path: Path, reports: list[MotionQualityReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected": reports[0].to_dict() if reports else None,
                "candidates": [report.to_dict() for report in reports],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
