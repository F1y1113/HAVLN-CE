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
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--asset-name")
    parser.add_argument("--identity")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--target-height", type=float, default=1.72)
    parser.add_argument("--ground-y", type=float, default=-0.2)
    parser.add_argument("--rotation-scale", type=float, default=0.65)
    parser.add_argument("--lower-leg-rotation-scale", type=float, default=0.18)
    parser.add_argument("--foot-rotation-scale", type=float, default=0.35)
    parser.add_argument("--no-root-orientation", action="store_true")
    parser.add_argument("--preserve-root-motion", action="store_true")
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
    parser.add_argument("--foot-lock-blend-frames", type=int, default=4)
    parser.add_argument("--no-calibrated-leg-ik", action="store_true")
    parser.add_argument("--no-solidify-shell", action="store_true")
    parser.add_argument("--body-shell-thickness", type=float, default=0.018)
    parser.add_argument("--hair-shell-thickness", type=float, default=0.006)
    parser.add_argument(
        "--joint-ik",
        action="store_true",
        help="Experimental: retarget from Kimodo posed_joints by bone-direction IK.",
    )
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
            duration=args.duration,
            seed=args.seed,
            asset_name=args.asset_name,
            identity=args.identity,
            frames=args.frames,
            fps=args.fps,
            target_height=args.target_height,
            ground_y=args.ground_y,
            rotation_scale=args.rotation_scale,
            lower_leg_rotation_scale=args.lower_leg_rotation_scale,
            foot_rotation_scale=args.foot_rotation_scale,
            include_root_orientation=not args.no_root_orientation,
            preserve_root_motion=args.preserve_root_motion,
            stabilize_root_yaw=args.stabilize_root_yaw,
            foot_contact_lock=args.foot_contact_lock,
            foot_orientation_lock=not args.no_foot_orientation_lock,
            foot_contact_height=args.foot_contact_height,
            foot_lock_blend_frames=args.foot_lock_blend_frames,
            calibrated_leg_ik=not args.no_calibrated_leg_ik,
            prefer_joint_position_ik=args.joint_ik,
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
