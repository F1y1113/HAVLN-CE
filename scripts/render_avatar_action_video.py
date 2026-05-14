#!/usr/bin/env python3
"""Render one HAPS/HA-VLN animated human asset from a camera path.

The script is intentionally small and simulator-facing: it loads the exported
`frameXXX.object_config.json` files from a HAPS asset directory, places one frame
at a time into Habitat-Sim, and writes an RGB video from a fixed or orbit camera.
It is useful for quickly inspecting generated humans before wiring them into
full HA-R2R annotations.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import habitat_sim
import imageio
import magnum as mn
import numpy as np

try:
    import quaternion  # noqa: F401
except Exception:
    quaternion = None


def frame_sort_key(path):
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else 0


def parse_vec3(value):
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return parts


def yaw_quaternion(yaw):
    if not hasattr(np, "quaternion"):
        raise RuntimeError(
            "numpy-quaternion is required by Habitat-Sim AgentState.rotation; "
            "install it in the HA-VLN simulator environment."
        )
    half = yaw * 0.5
    return np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def look_yaw(camera_position, target_position):
    direction_x = target_position[0] - camera_position[0]
    direction_z = target_position[2] - camera_position[2]
    return math.atan2(direction_x, -direction_z)


def make_configuration(scene_file, width, height, gpu_device, camera_pitch):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.gpu_device_id = gpu_device
    backend_cfg.scene_id = scene_file
    backend_cfg.enable_physics = True

    sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [height, width]
    sensor_spec.position = [0.0, 0.0, 0.0]
    sensor_spec.orientation = [camera_pitch, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def load_frame_template_ids(sim, asset_dir):
    manager = sim.get_object_template_manager()
    config_files = sorted(asset_dir.glob("frame*.object_config.json"), key=frame_sort_key)
    frame_files = config_files or sorted(asset_dir.glob("frame*.glb"), key=frame_sort_key)
    if not frame_files:
        raise FileNotFoundError("no frame*.object_config.json or frame*.glb files found in {}".format(asset_dir))

    template_ids = []
    for frame_file in frame_files:
        loaded = manager.load_configs(str(frame_file))
        if not loaded:
            raise RuntimeError("failed to load object template: {}".format(frame_file))
        template_ids.append(loaded[0])
    return template_ids


def euler_degrees_to_quat(rotation_degrees):
    return (
        mn.Quaternion.rotation(mn.Deg(rotation_degrees[0]), mn.Vector3.x_axis())
        * mn.Quaternion.rotation(mn.Deg(rotation_degrees[1]), mn.Vector3.y_axis())
        * mn.Quaternion.rotation(mn.Deg(rotation_degrees[2]), mn.Vector3.z_axis())
    )


def set_camera(sim, camera_position, target_position):
    state = habitat_sim.AgentState()
    state.position = np.array(camera_position, dtype=np.float32)
    state.rotation = yaw_quaternion(look_yaw(camera_position, target_position))
    sim.initialize_agent(0, state)


def render_video(args):
    repo_root = Path(__file__).resolve().parents[1]
    asset_dir = Path(args.asset_dir)
    if not asset_dir.is_absolute():
        asset_dir = repo_root / asset_dir
    scene_file = args.scene_file
    if scene_file is None and args.scan:
        scene_file = str(repo_root / "Data" / "scene_datasets" / "mp3d" / args.scan / "{}.glb".format(args.scan))
    scene_file = scene_file or "NONE"
    if scene_file != "NONE" and not os.path.exists(scene_file):
        raise FileNotFoundError("scene file not found: {}".format(scene_file))

    cfg = make_configuration(scene_file, args.width, args.height, args.gpu_device, args.camera_pitch)
    sim = habitat_sim.Simulator(cfg)
    template_ids = load_frame_template_ids(sim, asset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    object_rotation = euler_degrees_to_quat(args.rotation)
    current_object_id = None
    frame_count = min(args.frames, len(template_ids))

    try:
        with imageio.get_writer(str(output_path), fps=args.fps) as writer:
            for frame_index in range(frame_count):
                if current_object_id is not None and current_object_id in sim.get_existing_object_ids():
                    sim.remove_object(current_object_id)

                current_object_id = sim.add_object(template_ids[frame_index])
                if current_object_id == -1:
                    raise RuntimeError("failed to add object for frame {}".format(frame_index))
                sim.set_translation(args.translation, current_object_id)
                sim.set_rotation(object_rotation, current_object_id)

                if args.camera_mode == "orbit":
                    angle = args.orbit_start + args.orbit_radians * (frame_index / max(frame_count - 1, 1))
                    camera_position = [
                        args.target[0] + math.sin(angle) * args.camera_radius,
                        args.camera_height,
                        args.target[2] + math.cos(angle) * args.camera_radius,
                    ]
                else:
                    camera_position = args.camera_position

                set_camera(sim, camera_position, args.target)
                sim.step_physics(1.0 / 60.0)
                observation = sim.get_sensor_observations()["color_sensor"]
                writer.append_data(observation[:, :, :3])
    finally:
        if current_object_id is not None and current_object_id in sim.get_existing_object_ids():
            sim.remove_object(current_object_id)
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, help="HAPS asset directory containing frame GLBs/configs.")
    parser.add_argument("--output", required=True, help="Output video path, for example outputs/avatar_camera.mp4.")
    parser.add_argument("--scan", help="Matterport scan id under Data/scene_datasets/mp3d.")
    parser.add_argument("--scene-file", help="Explicit scene .glb path. Use NONE for an empty Habitat scene.")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--translation", type=parse_vec3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--rotation", type=parse_vec3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--target", type=parse_vec3, default=[0.0, 0.85, 0.0])
    parser.add_argument("--camera-mode", choices=["static", "orbit"], default="static")
    parser.add_argument("--camera-position", type=parse_vec3, default=[0.0, 1.25, 3.2])
    parser.add_argument("--camera-height", type=float, default=1.25)
    parser.add_argument("--camera-radius", type=float, default=3.2)
    parser.add_argument("--camera-pitch", type=float, default=-0.08)
    parser.add_argument("--orbit-start", type=float, default=math.pi)
    parser.add_argument("--orbit-radians", type=float, default=math.pi * 0.55)
    args = parser.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()
