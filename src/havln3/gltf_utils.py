from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation


_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def joint_base_name(name: str | None) -> str:
    if not name:
        return ""
    return name.split(":")[-1].split("|")[-1].strip()


def read_accessor(gltf: GLTF2, accessor_index: int | None) -> np.ndarray:
    if accessor_index is None:
        raise ValueError("missing accessor index")
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None:
        raise ValueError(f"accessor {accessor_index} has no bufferView")
    buffer_view = gltf.bufferViews[accessor.bufferView]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor.componentType]).newbyteorder("<")
    components = _TYPE_COUNTS[accessor.type]
    blob = gltf.binary_blob()
    offset = int(buffer_view.byteOffset or 0) + int(accessor.byteOffset or 0)
    count = int(accessor.count)
    stride = int(buffer_view.byteStride or 0)

    if stride and stride != dtype.itemsize * components:
        rows = []
        for row_index in range(count):
            row_offset = offset + row_index * stride
            rows.append(np.frombuffer(blob, dtype=dtype, count=components, offset=row_offset))
        array = np.vstack(rows)
    else:
        array = np.frombuffer(blob, dtype=dtype, count=count * components, offset=offset)
        array = array.reshape(count, components)

    if accessor.normalized and np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.signedinteger):
            max_value = float(np.iinfo(array.dtype).max)
            array = np.maximum(array.astype(np.float64) / max_value, -1.0)
        else:
            max_value = float(np.iinfo(array.dtype).max)
            array = array.astype(np.float64) / max_value
    return array.copy()


def _node_matrix(node: Any, rotation_delta: np.ndarray | None = None) -> np.ndarray:
    if node.matrix is not None:
        matrix = np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
        if rotation_delta is not None:
            delta = np.eye(4, dtype=np.float64)
            delta[:3, :3] = Rotation.from_quat(rotation_delta).as_matrix()
            matrix = matrix @ delta
        return matrix

    translation = np.asarray(node.translation or [0.0, 0.0, 0.0], dtype=np.float64)
    rotation = np.asarray(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if rotation_delta is not None:
        rotation = (Rotation.from_quat(rotation) * Rotation.from_quat(rotation_delta)).as_quat()
    scale = np.asarray(node.scale or [1.0, 1.0, 1.0], dtype=np.float64)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(rotation).as_matrix() @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def node_global_matrices(gltf: GLTF2, local_joint_deltas: np.ndarray | None = None) -> list[np.ndarray]:
    skin = gltf.skins[0] if gltf.skins else None
    joint_to_skin_index = {}
    if skin is not None:
        joint_to_skin_index = {joint: index for index, joint in enumerate(skin.joints or [])}

    globals_: list[np.ndarray | None] = [None] * len(gltf.nodes)

    def local_matrix(node_index: int) -> np.ndarray:
        skin_index = joint_to_skin_index.get(node_index)
        delta = None
        if skin_index is not None and local_joint_deltas is not None:
            delta = local_joint_deltas[skin_index]
        return _node_matrix(gltf.nodes[node_index], delta)

    def visit(node_index: int, parent: np.ndarray) -> None:
        current = parent @ local_matrix(node_index)
        globals_[node_index] = current
        for child in gltf.nodes[node_index].children or []:
            visit(child, current)

    scene_indices = gltf.scene
    scenes = [gltf.scenes[scene_indices]] if scene_indices is not None and gltf.scenes else gltf.scenes or []
    for scene in scenes:
        for root in scene.nodes or []:
            visit(root, np.eye(4, dtype=np.float64))
    for index, matrix in enumerate(globals_):
        if matrix is None:
            visit(index, np.eye(4, dtype=np.float64))
    return [matrix if matrix is not None else np.eye(4, dtype=np.float64) for matrix in globals_]


def deform_skinned_primitives(gltf: GLTF2, local_joint_deltas: np.ndarray) -> list[np.ndarray]:
    if not gltf.skins:
        raise ValueError("avatar GLB has no skin")
    skin = gltf.skins[0]
    inverse_bind = read_accessor(gltf, skin.inverseBindMatrices)
    inverse_bind = inverse_bind.reshape((-1, 4, 4)).transpose(0, 2, 1)
    globals_ = node_global_matrices(gltf, local_joint_deltas)
    joint_mats = np.stack([globals_[joint] @ inverse_bind[index] for index, joint in enumerate(skin.joints)])

    deformed: list[np.ndarray] = []
    for mesh in gltf.meshes or []:
        for primitive in mesh.primitives:
            positions = read_accessor(gltf, primitive.attributes.POSITION).astype(np.float64)
            joints_accessor = getattr(primitive.attributes, "JOINTS_0", None)
            weights_accessor = getattr(primitive.attributes, "WEIGHTS_0", None)
            if joints_accessor is None or weights_accessor is None:
                deformed.append(positions)
                continue

            joints = read_accessor(gltf, joints_accessor).astype(np.int64)
            weights = read_accessor(gltf, weights_accessor).astype(np.float64)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
            homogeneous = np.c_[positions, np.ones(len(positions), dtype=np.float64)]
            out = np.zeros((len(positions), 4), dtype=np.float64)
            for weight_index in range(weights.shape[1]):
                out += weights[:, weight_index : weight_index + 1] * np.einsum(
                    "nij,nj->ni", joint_mats[joints[:, weight_index]], homogeneous
                )
            deformed.append(out[:, :3])
    return deformed


def identity_quaternions(count: int) -> np.ndarray:
    quats = np.zeros((count, 4), dtype=np.float64)
    quats[:, 3] = 1.0
    return quats


@dataclass(frozen=True)
class VertexNormalizer:
    scale: float
    source_ground: float
    target_ground_y: float
    source_up_axis: int = 2

    @property
    def _axis_order(self) -> tuple[int, int, int]:
        if self.source_up_axis == 2:
            return (0, 2, 1)
        if self.source_up_axis == 1:
            return (0, 1, 2)
        return (1, 0, 2)

    @classmethod
    def from_vertices(
        cls,
        vertices_by_primitive: list[np.ndarray],
        *,
        target_height: float,
        ground_y: float,
        source_up_axis: int = 2,
    ) -> "VertexNormalizer":
        combined = np.vstack(vertices_by_primitive)
        source_ground = float(combined[:, source_up_axis].min())
        source_top = float(combined[:, source_up_axis].max())
        height = max(source_top - source_ground, 1e-9)
        return cls(
            scale=float(target_height) / height,
            source_ground=source_ground,
            target_ground_y=ground_y,
            source_up_axis=source_up_axis,
        )

    def _to_target_axes(self, values: np.ndarray) -> np.ndarray:
        out = values[:, self._axis_order].copy()
        out[:, 1] -= self.source_ground
        out *= self.scale
        out[:, 1] += self.target_ground_y
        return out

    def transform(
        self,
        vertices_by_primitive: list[np.ndarray],
        points: np.ndarray | None = None,
        *,
        root_offset: np.ndarray | None = None,
        snap_to_ground: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        transformed = [self._to_target_axes(vertices) for vertices in vertices_by_primitive]

        transformed_points = None
        if points is not None:
            transformed_points = self._to_target_axes(points)

        if root_offset is not None:
            root = np.asarray(root_offset, dtype=np.float64)
            transformed = [vertices + root for vertices in transformed]
            if transformed_points is not None:
                transformed_points = transformed_points + root

        if snap_to_ground:
            lowest = min(float(vertices[:, 1].min()) for vertices in transformed)
            if lowest < self.target_ground_y:
                lift = self.target_ground_y - lowest
                transformed = [vertices + np.array([0.0, lift, 0.0]) for vertices in transformed]
                if transformed_points is not None:
                    transformed_points = transformed_points + np.array([0.0, lift, 0.0])

        return transformed, transformed_points


def write_object_config(path: Path, glb_path: Path) -> None:
    config = {
        "render_asset": glb_path.name,
        "collision_asset": glb_path.name,
        "scale": [1.0, 1.0, 1.0],
        "use_mesh_for_collision": True,
        "requires_lighting": True,
        "mass": 1.0,
        "margin": 0.0,
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
