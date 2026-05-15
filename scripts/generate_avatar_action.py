#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from havln3.pipeline import AvatarActionPipeline, AvatarActionRequest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a textured HA-VLN/ViCo human action asset from one sentence."
    )
    parser.add_argument("prompt", help="One-sentence person/action description.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--avatar-root", type=Path, default=Path("vico_assets_probe/models"))
    parser.add_argument("--avatar-glb", type=Path)
    parser.add_argument("--avatar-catalog", type=Path)
    parser.add_argument("--motion-npz", type=Path, help="Existing Kimodo .npz output.")
    parser.add_argument("--amass-npz", type=Path, help="Optional AMASS .npz companion file.")
    parser.add_argument("--gem-smpl-params", type=Path, help="Existing GEM smpl_params.pt file.")
    parser.add_argument("--gem-param-group", default="body_params_global")
    parser.add_argument("--generator", choices=["existing", "kimodo"], default="existing")
    parser.add_argument("--kimodo-model", default="Kimodo-SMPLX-RP-v1")
    parser.add_argument("--kimodo-num-samples", type=int, default=1)
    parser.add_argument(
        "--no-motion-quality-select",
        action="store_true",
        help="Disable automatic quality ranking when Kimodo generates multiple samples.",
    )
    parser.add_argument("--motion-quality-min-score", type=float, default=62.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--asset-name")
    parser.add_argument("--identity")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--target-height", type=float, default=1.72)
    parser.add_argument("--ground-y", type=float, default=-0.2)
    parser.add_argument("--rotation-scale", type=float, default=0.65)
    parser.add_argument("--arm-rotation-scale", type=float, default=0.65)
    parser.add_argument("--forearm-rotation-scale", type=float, default=0.45)
    parser.add_argument("--hand-rotation-scale", type=float, default=0.18)
    parser.add_argument("--lower-leg-rotation-scale", type=float, default=0.18)
    parser.add_argument("--foot-rotation-scale", type=float, default=0.35)
    parser.add_argument("--no-root-orientation", action="store_true")
    parser.add_argument("--preserve-root-motion", action="store_true")
    parser.add_argument("--root-motion-scale", type=float, default=1.0)
    parser.add_argument(
        "--stabilize-root-yaw",
        action="store_true",
        help="Post-process stationary flips/poses so the avatar keeps its initial horizontal facing.",
    )
    parser.add_argument(
        "--foot-contact-lock",
        action="store_true",
        help="Post-process support/landing frames so low feet stay anchored to the floor.",
    )
    parser.add_argument(
        "--no-foot-orientation-lock",
        action="store_true",
        help="Disable contact-foot orientation stabilization when --foot-contact-lock is enabled.",
    )
    parser.add_argument("--foot-contact-height", type=float, default=0.12)
    parser.add_argument(
        "--foot-contact-velocity",
        type=float,
        default=0.02,
        help="Maximum per-frame horizontal foot speed for detected contact frames.",
    )
    parser.add_argument(
        "--ignore-source-foot-contacts",
        action="store_true",
        help="Use only geometric contact detection instead of Kimodo's foot contact labels.",
    )
    parser.add_argument("--foot-lock-blend-frames", type=int, default=4)
    parser.add_argument(
        "--no-grounded-foot-ik",
        action="store_true",
        help="Use legacy whole-frame contact correction instead of per-foot two-bone IK targets.",
    )
    parser.add_argument("--foot-support-min-frames", type=int, default=8)
    parser.add_argument("--foot-support-max-frames", type=int, default=16)
    parser.add_argument("--foot-support-max-air-frames", type=int, default=8)
    parser.add_argument(
        "--airborne-leg-stabilization",
        action="store_true",
        help="Experimental: stabilize airborne acrobatic leg motion in pelvis-local space before leg IK.",
    )
    parser.add_argument("--airborne-leg-stabilization-strength", type=float, default=0.85)
    parser.add_argument("--airborne-tuck-reach-ratio", type=float, default=0.52)
    parser.add_argument("--no-calibrated-leg-ik", action="store_true")
    parser.add_argument(
        "--body-relative-leg-ik",
        action="store_true",
        help="Experimental: map Kimodo leg directions through pelvis-local frames before leg IK.",
    )
    parser.add_argument("--no-solidify-shell", action="store_true")
    parser.add_argument("--body-shell-thickness", type=float, default=0.018)
    parser.add_argument("--hair-shell-thickness", type=float, default=0.006)
    parser.add_argument(
        "--joint-ik",
        action="store_true",
        help="Experimental: retarget from Kimodo posed_joints by bone-direction IK.",
    )
    parser.add_argument(
        "--no-procedural-running-arms",
        action="store_true",
        help="Disable the locomotion-specific target-rig arm swing cleanup layer.",
    )
    parser.add_argument("--running-arm-swing-strength", type=float, default=0.35)
    parser.add_argument("--running-arm-forward-ratio", type=float, default=0.52)
    parser.add_argument("--running-arm-drop-ratio", type=float, default=0.50)
    parser.add_argument("--running-arm-side-ratio", type=float, default=0.055)
    parser.add_argument("--running-arm-reach-min", type=float, default=0.46)
    parser.add_argument("--running-arm-reach-max", type=float, default=0.68)
    parser.add_argument(
        "--no-torso-counter-rotation",
        action="store_true",
        help="Disable the locomotion-specific spine/chest counter-rotation cleanup layer.",
    )
    parser.add_argument("--torso-counter-rotation-degrees", type=float, default=7.0)
    parser.add_argument("--torso-counter-rotation-strength", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = AvatarActionPipeline().run(
        AvatarActionRequest(
            prompt=args.prompt,
            output_root=args.output_root,
            avatar_root=args.avatar_root,
            avatar_glb=args.avatar_glb,
            avatar_catalog=args.avatar_catalog,
            motion_npz=args.motion_npz,
            amass_npz=args.amass_npz,
            gem_smpl_params=args.gem_smpl_params,
            gem_param_group=args.gem_param_group,
            generator=args.generator,
            kimodo_model=args.kimodo_model,
            kimodo_num_samples=args.kimodo_num_samples,
            motion_quality_select=not args.no_motion_quality_select,
            motion_quality_min_score=args.motion_quality_min_score,
            duration=args.duration,
            seed=args.seed,
            asset_name=args.asset_name,
            identity=args.identity,
            frames=args.frames,
            fps=args.fps,
            target_height=args.target_height,
            ground_y=args.ground_y,
            rotation_scale=args.rotation_scale,
            arm_rotation_scale=args.arm_rotation_scale,
            forearm_rotation_scale=args.forearm_rotation_scale,
            hand_rotation_scale=args.hand_rotation_scale,
            lower_leg_rotation_scale=args.lower_leg_rotation_scale,
            foot_rotation_scale=args.foot_rotation_scale,
            include_root_orientation=not args.no_root_orientation,
            preserve_root_motion=args.preserve_root_motion,
            root_motion_scale=args.root_motion_scale,
            stabilize_root_yaw=args.stabilize_root_yaw,
            foot_contact_lock=args.foot_contact_lock,
            foot_orientation_lock=not args.no_foot_orientation_lock,
            foot_contact_height=args.foot_contact_height,
            foot_contact_velocity=args.foot_contact_velocity,
            foot_contact_use_source=not args.ignore_source_foot_contacts,
            foot_lock_blend_frames=args.foot_lock_blend_frames,
            grounded_foot_ik=not args.no_grounded_foot_ik,
            foot_support_min_frames=args.foot_support_min_frames,
            foot_support_max_frames=args.foot_support_max_frames,
            foot_support_max_air_frames=args.foot_support_max_air_frames,
            airborne_leg_stabilization=args.airborne_leg_stabilization,
            airborne_leg_stabilization_strength=args.airborne_leg_stabilization_strength,
            airborne_tuck_reach_ratio=args.airborne_tuck_reach_ratio,
            calibrated_leg_ik=not args.no_calibrated_leg_ik,
            body_relative_leg_ik=args.body_relative_leg_ik,
            prefer_joint_position_ik=args.joint_ik,
            procedural_running_arm_swing=not args.no_procedural_running_arms,
            running_arm_swing_strength=args.running_arm_swing_strength,
            running_arm_forward_ratio=args.running_arm_forward_ratio,
            running_arm_drop_ratio=args.running_arm_drop_ratio,
            running_arm_side_ratio=args.running_arm_side_ratio,
            running_arm_reach_min=args.running_arm_reach_min,
            running_arm_reach_max=args.running_arm_reach_max,
            locomotion_torso_counter_rotation=not args.no_torso_counter_rotation,
            torso_counter_rotation_degrees=args.torso_counter_rotation_degrees,
            torso_counter_rotation_strength=args.torso_counter_rotation_strength,
            solidify_shell=not args.no_solidify_shell,
            body_shell_thickness=args.body_shell_thickness,
            hair_shell_thickness=args.hair_shell_thickness,
        )
    )
    print(
        json.dumps(
            {
                "asset_name": result.asset_name,
                "asset_dir": str(result.asset_dir),
                "avatar_glb": str(result.avatar_glb),
                "motion_path": str(result.motion_path),
                "report_path": str(result.report_path),
                "skeleton_path": str(result.skeleton_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
