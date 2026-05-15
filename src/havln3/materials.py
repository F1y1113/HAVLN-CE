from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pygltflib import GLTF2


_ALPHA_KEYWORDS = ("hair", "lash", "eyelash", "brow")


def _image_bytes(gltf: GLTF2, image_index: int, glb_path: Path) -> bytes | None:
    image = gltf.images[image_index]
    if image.bufferView is not None:
        view = gltf.bufferViews[image.bufferView]
        blob = gltf.binary_blob()
        start = int(view.byteOffset or 0)
        end = start + int(view.byteLength or 0)
        return bytes(blob[start:end])
    if image.uri:
        if image.uri.startswith("data:"):
            _, encoded = image.uri.split(",", 1)
            return base64.b64decode(encoded)
        image_path = (glb_path.parent / image.uri).resolve()
        if image_path.exists():
            return image_path.read_bytes()
    return None


def _base_color_has_alpha(gltf: GLTF2, material: Any, glb_path: Path) -> bool:
    pbr = material.pbrMetallicRoughness
    if not pbr or not pbr.baseColorTexture:
        return False
    texture = gltf.textures[pbr.baseColorTexture.index]
    if texture.source is None:
        return False
    raw = _image_bytes(gltf, texture.source, glb_path)
    if not raw:
        return False
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
                return False
            alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
            return float((alpha < 250).mean()) > 0.01
    except Exception:
        return False


def fix_habitat_materials(
    glb_path: Path,
    *,
    alpha_cutoff: float = 0.55,
    roughness: float = 0.88,
    detect_texture_alpha: bool = True,
) -> list[dict[str, object]]:
    """Make exported avatar frames friendlier to Habitat/ViCo rendering.

    The fix preserves original UVs and base-color textures. It changes only
    render-state metadata that commonly causes transparent hair cards to sort
    badly in Habitat. Exported animation frames are always marked double-sided
    because Habitat side/back camera views can otherwise make single-surface
    clothing read as a flat paper cutout.
    """

    gltf = GLTF2().load(str(glb_path))
    changes: list[dict[str, object]] = []
    for index, material in enumerate(gltf.materials or []):
        name = material.name or f"material_{index}"
        lower = name.lower()
        pbr = material.pbrMetallicRoughness
        before = {
            "name": name,
            "alphaMode": material.alphaMode,
            "alphaCutoff": material.alphaCutoff,
            "doubleSided": material.doubleSided,
            "metallicFactor": pbr.metallicFactor if pbr else None,
            "roughnessFactor": pbr.roughnessFactor if pbr else None,
        }

        if pbr is not None:
            pbr.metallicFactor = 0.0
            pbr.roughnessFactor = max(float(pbr.roughnessFactor or 0.0), roughness)
            pbr.metallicRoughnessTexture = None

        alpha_like = any(keyword in lower for keyword in _ALPHA_KEYWORDS)
        if detect_texture_alpha and not alpha_like:
            alpha_like = material.alphaMode == "BLEND" or _base_color_has_alpha(gltf, material, glb_path)
        if alpha_like:
            material.alphaMode = "MASK"
            material.alphaCutoff = alpha_cutoff
        else:
            material.alphaMode = "OPAQUE"
            material.alphaCutoff = None
            if pbr is not None and pbr.baseColorFactor and len(pbr.baseColorFactor) >= 4:
                pbr.baseColorFactor[3] = 1.0
        material.doubleSided = True

        after = {
            "name": name,
            "alphaMode": material.alphaMode,
            "alphaCutoff": material.alphaCutoff,
            "doubleSided": material.doubleSided,
            "metallicFactor": pbr.metallicFactor if pbr else None,
            "roughnessFactor": pbr.roughnessFactor if pbr else None,
        }
        if after != before:
            changes.append({"material_index": index, "before": before, "after": after})

    if changes:
        gltf.save_binary(str(glb_path))
    return changes
