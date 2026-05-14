from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SMPL22_JOINT_NAMES = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

SMPL22_TO_MIXAMO = {index: name for index, name in enumerate(SMPL22_JOINT_NAMES)}


@dataclass(frozen=True)
class MotionClip:
    local_rot_mats: np.ndarray
    root_positions: np.ndarray | None
    posed_joints: np.ndarray | None
    foot_contacts: np.ndarray | None
    source_path: Path
    prompt: str | None = None
    fps: float | None = None
    global_rot_mats: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.local_rot_mats.shape[0])


def _rotvecs_to_mats(root_orient: np.ndarray, pose_body: np.ndarray) -> np.ndarray:
    frames = int(root_orient.shape[0])
    mats = np.tile(np.eye(3, dtype=np.float64), (frames, 22, 1, 1))
    mats[:, 0] = Rotation.from_rotvec(root_orient.reshape(frames, 3)).as_matrix()
    body = pose_body.reshape(frames, 21, 3)
    mats[:, 1:22] = Rotation.from_rotvec(body.reshape(-1, 3)).as_matrix().reshape(frames, 21, 3, 3)
    return mats


def _load_torch_smpl_params(path: Path, param_group: str = "body_params_global") -> MotionClip:
    import torch

    params = torch.load(path, map_location="cpu")
    source = params[param_group]
    body_pose = source["body_pose"].detach().float().cpu().numpy()
    global_orient = source["global_orient"].detach().float().cpu().numpy()
    root_positions = None
    for key in ("transl", "trans", "translation"):
        if key in source:
            root_positions = source[key].detach().float().cpu().numpy()
            break
    prompt = None
    segment_info = params.get("segment_info")
    if isinstance(segment_info, list) and segment_info:
        prompt = segment_info[0].get("caption")
    mats = _rotvecs_to_mats(global_orient, body_pose)
    return MotionClip(
        local_rot_mats=mats,
        root_positions=root_positions,
        posed_joints=None,
        foot_contacts=None,
        source_path=path,
        prompt=prompt,
    )


def load_motion_file(
    motion_path: Path,
    *,
    amass_path: Path | None = None,
    gem_param_group: str = "body_params_global",
) -> MotionClip:
    if motion_path.suffix == ".pt":
        return _load_torch_smpl_params(motion_path, gem_param_group)

    data = np.load(motion_path, allow_pickle=True)
    if "local_rot_mats" in data:
        fps = float(data["fps"]) if "fps" in data else None
        root = data["smooth_root_pos"] if "smooth_root_pos" in data else data.get("root_positions")
        return MotionClip(
            local_rot_mats=data["local_rot_mats"].astype(np.float64),
            root_positions=None if root is None else root.astype(np.float64),
            posed_joints=data["posed_joints"].astype(np.float64) if "posed_joints" in data else None,
            foot_contacts=data["foot_contacts"].astype(bool) if "foot_contacts" in data else None,
            source_path=motion_path,
            fps=fps,
            global_rot_mats=data["global_rot_mats"].astype(np.float64) if "global_rot_mats" in data else None,
        )

    if "root_orient" in data and "pose_body" in data:
        return MotionClip(
            local_rot_mats=_rotvecs_to_mats(data["root_orient"], data["pose_body"]),
            root_positions=data["trans"].astype(np.float64) if "trans" in data else None,
            posed_joints=None,
            foot_contacts=data["foot_contacts"].astype(bool) if "foot_contacts" in data else None,
            source_path=motion_path,
            fps=float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else None,
            global_rot_mats=None,
        )

    if amass_path:
        return load_motion_file(amass_path, gem_param_group=gem_param_group)
    raise ValueError(f"unsupported motion file format: {motion_path}")


def _resample_array(values: np.ndarray | None, target_frames: int) -> np.ndarray | None:
    if values is None or len(values) == target_frames:
        return values
    src_t = np.linspace(0.0, 1.0, len(values))
    dst_t = np.linspace(0.0, 1.0, target_frames)
    flat = values.reshape(len(values), -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for channel in range(flat.shape[1]):
        out[:, channel] = np.interp(dst_t, src_t, flat[:, channel])
    return out.reshape((target_frames,) + values.shape[1:])


def _resample_rotations(mats: np.ndarray, target_frames: int) -> np.ndarray:
    if len(mats) == target_frames:
        return mats
    src_t = np.linspace(0.0, 1.0, len(mats))
    dst_t = np.linspace(0.0, 1.0, target_frames)
    out = np.empty((target_frames,) + mats.shape[1:], dtype=np.float64)
    for joint_index in range(mats.shape[1]):
        rotations = Rotation.from_matrix(mats[:, joint_index])
        out[:, joint_index] = Slerp(src_t, rotations)(dst_t).as_matrix()
    return out


def _resample_contacts(values: np.ndarray | None, target_frames: int) -> np.ndarray | None:
    if values is None or len(values) == target_frames:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, target_frames)).astype(np.int64)
    return values[indices].astype(bool)


def resample_motion(clip: MotionClip, target_frames: int) -> MotionClip:
    return MotionClip(
        local_rot_mats=_resample_rotations(clip.local_rot_mats, target_frames),
        root_positions=_resample_array(clip.root_positions, target_frames),
        posed_joints=_resample_array(clip.posed_joints, target_frames),
        foot_contacts=_resample_contacts(clip.foot_contacts, target_frames),
        source_path=clip.source_path,
        prompt=clip.prompt,
        fps=clip.fps,
        global_rot_mats=_resample_rotations(clip.global_rot_mats, target_frames)
        if clip.global_rot_mats is not None
        else None,
    )


def generate_kimodo_motion(
    prompt: str,
    output_stem: Path,
    *,
    model: str = "Kimodo-SMPLX-RP-v1",
    duration: float = 5.0,
    kimodo_bin: str = "kimodo_gen",
    seed: int | None = None,
    cfg_type: str | None = None,
    extra_args: list[str] | None = None,
    num_samples: int = 1,
) -> tuple[Path, Path | None]:
    candidates = generate_kimodo_motion_candidates(
        prompt,
        output_stem,
        model=model,
        duration=duration,
        kimodo_bin=kimodo_bin,
        seed=seed,
        cfg_type=cfg_type,
        extra_args=extra_args,
        num_samples=num_samples,
    )
    if candidates:
        return candidates[0]
    motion_path = output_stem.with_suffix(".npz")
    amass_path = output_stem.with_name(output_stem.name + "_amass.npz")
    return motion_path, amass_path if amass_path.exists() else None


def generate_kimodo_motion_candidates(
    prompt: str,
    output_stem: Path,
    *,
    model: str = "Kimodo-SMPLX-RP-v1",
    duration: float = 5.0,
    kimodo_bin: str = "kimodo_gen",
    seed: int | None = None,
    cfg_type: str | None = None,
    extra_args: list[str] | None = None,
    num_samples: int = 1,
) -> list[tuple[Path, Path | None]]:
    executable = shutil.which(kimodo_bin)
    if not executable:
        raise RuntimeError(
            f"{kimodo_bin!r} was not found. Install Kimodo or pass --motion-npz to retarget an existing output."
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        prompt,
        "--model",
        model,
        "--duration",
        str(duration),
        "--output",
        str(output_stem),
        "--num_samples",
        str(max(1, num_samples)),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if cfg_type:
        command.extend(["--cfg_type", cfg_type])
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)

    return _generated_motion_candidates(output_stem)


def _generated_motion_candidates(output_stem: Path) -> list[tuple[Path, Path | None]]:
    candidates: list[Path] = []

    def append_unique(paths: list[Path]) -> None:
        for path in paths:
            if path.exists() and path not in candidates:
                candidates.append(path)

    direct = output_stem.with_suffix(".npz")
    append_unique([direct])
    append_unique(sorted(output_stem.parent.glob(f"{output_stem.name}*.npz")))
    append_unique(sorted((output_stem.parent / output_stem.name).glob(f"{output_stem.name}*.npz")))
    if output_stem.is_dir():
        append_unique(sorted(output_stem.glob(f"{output_stem.name}*.npz")))
    if not candidates:
        return []

    result: list[tuple[Path, Path | None]] = []
    for motion_path in candidates:
        suffix = motion_path.stem.removeprefix(output_stem.name).lstrip("_")
        companion_names = [
            motion_path.with_name(motion_path.stem + "_amass.npz"),
            motion_path.parent / f"amass_{suffix}.npz" if suffix else None,
            output_stem.with_name(output_stem.name + "_amass.npz"),
        ]
        if suffix:
            companion_names.append(output_stem.with_name(f"amass_{suffix}.npz"))
            companion_names.append(output_stem / f"amass_{suffix}.npz")
        companion_names.append(output_stem / f"{output_stem.name}_amass.npz")
        amass_path = next((path for path in companion_names if path and path.exists()), None)
        result.append((motion_path, amass_path))
    return result
