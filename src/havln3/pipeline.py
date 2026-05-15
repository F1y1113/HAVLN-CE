from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from havln3.avatar_assets import select_avatar
from havln3.motion import generate_kimodo_motion_candidates, load_motion_file
from havln3.motion_quality import MotionQualityOptions, rank_motion_files, write_motion_quality_report
from havln3.retarget import RetargetOptions, retarget_motion_to_avatar


def _slugify(text: str, *, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug[:max_length].strip("_") or "avatar_action")


@dataclass(frozen=True)
class AvatarActionRequest:
    prompt: str
    output_root: Path
    avatar_root: Path = Path("vico_assets_probe/models")
    avatar_glb: Path | None = None
    avatar_catalog: Path | None = None
    motion_npz: Path | None = None
    amass_npz: Path | None = None
    gem_smpl_params: Path | None = None
    gem_param_group: str = "body_params_global"
    asset_name: str | None = None
    identity: str | None = None
    generator: str = "existing"
    kimodo_model: str = "Kimodo-SMPLX-RP-v1"
    kimodo_num_samples: int = 1
    motion_quality_select: bool = True
    motion_quality_min_score: float = 62.0
    duration: float = 5.0
    seed: int | None = None
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
    calibrated_leg_ik: bool = True
    body_relative_leg_ik: bool = False
    prefer_joint_position_ik: bool = False
    solidify_shell: bool = True
    body_shell_thickness: float = 0.018
    hair_shell_thickness: float = 0.006


@dataclass(frozen=True)
class AvatarActionResult:
    asset_name: str
    asset_dir: Path
    avatar_glb: Path
    motion_path: Path
    report_path: Path
    skeleton_path: Path


class AvatarActionPipeline:
    def run(self, request: AvatarActionRequest) -> AvatarActionResult:
        avatar_match, top_matches = select_avatar(
            request.prompt,
            request.avatar_root,
            preferred_avatar=request.avatar_glb,
            catalog_path=request.avatar_catalog,
        )
        avatar = avatar_match.asset
        asset_name = request.asset_name or _slugify(f"{avatar.display_name}_{request.prompt}")
        asset_dir = request.output_root / "Data" / "HAPS2_0" / asset_name
        motion_path, amass_path, motion_selection = self._resolve_motion(request, asset_name)
        motion = load_motion_file(
            motion_path,
            amass_path=amass_path,
            gem_param_group=request.gem_param_group,
        )
        identity = request.identity or avatar.display_name
        options = RetargetOptions(
            frames=request.frames,
            fps=request.fps,
            target_height=request.target_height,
            ground_y=request.ground_y,
            rotation_scale=request.rotation_scale,
            arm_rotation_scale=request.arm_rotation_scale,
            forearm_rotation_scale=request.forearm_rotation_scale,
            hand_rotation_scale=request.hand_rotation_scale,
            lower_leg_rotation_scale=request.lower_leg_rotation_scale,
            foot_rotation_scale=request.foot_rotation_scale,
            include_root_orientation=request.include_root_orientation,
            preserve_root_motion=request.preserve_root_motion,
            root_motion_scale=request.root_motion_scale,
            stabilize_root_yaw=request.stabilize_root_yaw,
            foot_contact_lock=request.foot_contact_lock,
            foot_orientation_lock=request.foot_orientation_lock,
            foot_contact_height=request.foot_contact_height,
            foot_contact_velocity=request.foot_contact_velocity,
            foot_contact_use_source=request.foot_contact_use_source,
            foot_lock_blend_frames=request.foot_lock_blend_frames,
            grounded_foot_ik=request.grounded_foot_ik,
            foot_support_min_frames=request.foot_support_min_frames,
            foot_support_max_frames=request.foot_support_max_frames,
            foot_support_max_air_frames=request.foot_support_max_air_frames,
            airborne_leg_stabilization=request.airborne_leg_stabilization,
            airborne_leg_stabilization_strength=request.airborne_leg_stabilization_strength,
            airborne_tuck_reach_ratio=request.airborne_tuck_reach_ratio,
            calibrated_leg_ik=request.calibrated_leg_ik,
            body_relative_leg_ik=request.body_relative_leg_ik,
            prefer_joint_position_ik=request.prefer_joint_position_ik,
            solidify_shell=request.solidify_shell,
            body_shell_thickness=request.body_shell_thickness,
            hair_shell_thickness=request.hair_shell_thickness,
        )
        selection = {
            "selected": {
                "path": str(avatar.path),
                "display_name": avatar.display_name,
                "provider": avatar.provider,
                "score": avatar_match.score,
                "reasons": list(avatar_match.reasons),
                "tags": sorted(avatar.tags),
            },
            "top_candidates": [
                {
                    "path": str(match.asset.path),
                    "display_name": match.asset.display_name,
                    "score": match.score,
                    "reasons": list(match.reasons),
                }
                for match in top_matches
            ],
            "motion": motion_selection,
        }
        report = retarget_motion_to_avatar(
            avatar_glb=avatar.path,
            motion=motion,
            output_dir=asset_dir,
            asset_name=asset_name,
            prompt=request.prompt,
            identity=identity,
            options=options,
            selection=selection,
        )
        report_dir = request.output_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{asset_name}_retarget_report.json"
        report_path.write_text(json.dumps({**report, "selection": selection}, indent=2), encoding="utf-8")
        return AvatarActionResult(
            asset_name=asset_name,
            asset_dir=asset_dir,
            avatar_glb=avatar.path,
            motion_path=motion_path,
            report_path=report_path,
            skeleton_path=asset_dir / "skeleton.json",
        )

    def _resolve_motion(
        self,
        request: AvatarActionRequest,
        asset_name: str,
    ) -> tuple[Path, Path | None, dict[str, object]]:
        if request.motion_npz:
            return request.motion_npz, request.amass_npz, {
                "mode": "provided",
                "selected": str(request.motion_npz),
                "amass": str(request.amass_npz) if request.amass_npz else None,
            }
        if request.gem_smpl_params:
            return request.gem_smpl_params, None, {
                "mode": "provided_gem",
                "selected": str(request.gem_smpl_params),
                "amass": None,
            }
        if request.generator == "kimodo":
            output_stem = request.output_root / "motion_generation" / asset_name / asset_name
            generated = generate_kimodo_motion_candidates(
                request.prompt,
                output_stem,
                model=request.kimodo_model,
                duration=request.duration,
                seed=request.seed,
                num_samples=request.kimodo_num_samples,
            )
            if not generated:
                raise RuntimeError(f"Kimodo did not produce .npz motion files under {output_stem.parent}")
            if len(generated) == 1 or not request.motion_quality_select:
                selected, amass = generated[0]
                return selected, amass, {
                    "mode": "kimodo",
                    "quality_select": False,
                    "selected": str(selected),
                    "amass": str(amass) if amass else None,
                    "candidates": [str(path) for path, _ in generated],
                }

            amass_by_motion = {motion_path: amass_path for motion_path, amass_path in generated}
            quality_options = MotionQualityOptions(min_score=request.motion_quality_min_score)
            quality_reports = rank_motion_files(
                [motion_path for motion_path, _ in generated],
                prompt=request.prompt,
                options=quality_options,
                gem_param_group=request.gem_param_group,
            )
            quality_report_path = output_stem.parent / "motion_quality_report.json"
            write_motion_quality_report(quality_report_path, quality_reports)
            best = quality_reports[0]
            best_path = Path(best.path)
            return best_path, amass_by_motion.get(best_path), {
                "mode": "kimodo",
                "quality_select": True,
                "selected": str(best_path),
                "amass": str(amass_by_motion.get(best_path)) if amass_by_motion.get(best_path) else None,
                "quality_report": str(quality_report_path),
                "best_score": best.score,
                "best_passed": best.passed,
                "best_reasons": best.reasons,
                "candidate_count": len(quality_reports),
            }
        raise ValueError(
            "No motion source was provided. Pass --motion-npz/--gem-smpl-params, "
            "or use --generator kimodo in an environment where kimodo_gen is installed."
        )
