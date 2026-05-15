from __future__ import annotations

import json
import re
import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pygltflib import GLTF2

_FEMALE_NAMES = {
    "clara",
    "elena",
    "elizabeth",
    "emily",
    "erika",
    "eve",
    "freya",
    "ivy",
    "jody",
    "kate",
    "layla",
    "louise",
    "megan",
    "mira",
    "naomi",
    "olivia",
    "scarlett",
    "shannon",
    "sophia",
    "sophie",
    "yara",
    "zara",
}

_MALE_NAMES = {
    "adam",
    "adrian",
    "alex",
    "brian",
    "brycer",
    "chad",
    "dylan",
    "elliot",
    "ethan",
    "felix",
    "james",
    "joe",
    "julian",
    "kenji",
    "leonard",
    "liam",
    "malik",
    "marco",
    "marcus",
    "morten",
    "nico",
    "rafael",
    "roth",
    "steve",
    "tariq",
    "theo",
    "victor",
    "walter",
    "zane",
}

_DEFAULT_TAGS_BY_STEM = {
    "mixamo_joe_anderson": {"male", "office", "business", "suit", "formal"},
    "mixamo_james_thompson": {"male", "office", "business", "formal"},
    "mixamo_steve_johnson": {"male", "casual", "office"},
    "mixamo_brian_carter": {"male", "casual"},
    "mixamo_alex_jefferson": {"male", "casual"},
    "mixamo_roth_miller": {"male", "casual"},
    "mixamo_chad_thompson": {"male", "casual"},
    "police": {"male", "police", "uniform", "officer"},
    "male_1": {"male", "generic"},
    "mixamo_female_1_decimated": {"female", "generic"},
}

_PROMPT_SYNONYMS = {
    "man": {"male"},
    "male": {"male"},
    "boy": {"male"},
    "gentleman": {"male"},
    "男人": {"male"},
    "男性": {"male"},
    "男": {"male"},
    "woman": {"female"},
    "female": {"female"},
    "girl": {"female"},
    "lady": {"female"},
    "女士": {"female"},
    "女人": {"female"},
    "女性": {"female"},
    "女": {"female"},
    "office": {"office", "business"},
    "business": {"business", "office"},
    "suit": {"suit", "formal"},
    "formal": {"formal", "suit"},
    "worker": {"office"},
    "西装": {"suit", "formal"},
    "办公室": {"office", "business"},
    "商务": {"business", "formal"},
    "警察": {"police", "officer", "uniform"},
    "police": {"police", "officer", "uniform"},
    "officer": {"police", "officer", "uniform"},
    "uniform": {"uniform"},
}


def _tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for phrase, tags in _PROMPT_SYNONYMS.items():
        if phrase in text.lower():
            words.update(tags)
    return words


def _display_name_from_stem(stem: str) -> str:
    clean = stem
    for prefix in ("mixamo_", "custom_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
            break
    clean = clean.replace("_merge", "").replace("_decimated", "")
    clean = clean.replace("_", " ").replace(".glb", "")
    return " ".join(part.capitalize() for part in clean.split())


@dataclass(frozen=True)
class AvatarAsset:
    path: Path
    display_name: str
    provider: str
    tags: set[str] = field(default_factory=set)
    max_alpha_fraction: float = 0.0
    alpha_material_count: int = 0

    @property
    def stem_key(self) -> str:
        return self.path.stem.lower()

    @property
    def tokens(self) -> set[str]:
        return _tokenize(f"{self.display_name} {self.path.stem} {' '.join(sorted(self.tags))}")


@dataclass(frozen=True)
class AvatarMatch:
    asset: AvatarAsset
    score: float
    reasons: tuple[str, ...]


def _image_bytes(gltf: GLTF2, image_index: int, glb_path: Path) -> bytes | None:
    image = gltf.images[image_index]
    if image.bufferView is not None:
        view = gltf.bufferViews[image.bufferView]
        blob = gltf.binary_blob()
        start = int(view.byteOffset or 0)
        return bytes(blob[start : start + int(view.byteLength or 0)])
    if image.uri:
        if image.uri.startswith("data:"):
            return base64.b64decode(image.uri.split(",", 1)[1])
        image_path = glb_path.parent / image.uri
        if image_path.exists():
            return image_path.read_bytes()
    return None


def _alpha_profile(path: Path) -> tuple[float, int]:
    try:
        gltf = GLTF2().load(str(path))
    except Exception:
        return 1.0, 1

    fractions: list[float] = []
    for material in gltf.materials or []:
        pbr = material.pbrMetallicRoughness
        if not pbr or not pbr.baseColorTexture:
            continue
        texture = gltf.textures[pbr.baseColorTexture.index]
        if texture.source is None:
            continue
        raw = _image_bytes(gltf, texture.source, path)
        if not raw:
            continue
        try:
            with Image.open(io.BytesIO(raw)) as image:
                alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
                fractions.append(float((alpha < 250).mean()))
        except Exception:
            continue
    return (max(fractions) if fractions else 0.0, sum(fraction > 0.05 for fraction in fractions))


def _asset_from_path(path: Path, catalog: dict[str, Any] | None = None) -> AvatarAsset:
    stem = path.stem.lower()
    provider = "mixamo" if stem.startswith("mixamo_") else "custom" if stem.startswith("custom_") else "local"
    display_name = _display_name_from_stem(path.stem)
    tags = set(_DEFAULT_TAGS_BY_STEM.get(stem, set()))
    tokens = _tokenize(display_name)
    if tokens & _FEMALE_NAMES:
        tags.add("female")
    if tokens & _MALE_NAMES:
        tags.add("male")
    tags.update(tokens - _FEMALE_NAMES - _MALE_NAMES)

    if catalog:
        override = catalog.get(path.name) or catalog.get(path.stem) or catalog.get(str(path))
        if isinstance(override, dict):
            display_name = str(override.get("display_name") or display_name)
            tags.update(str(tag).lower() for tag in override.get("tags", []))
            provider = str(override.get("provider") or provider)

    max_alpha, alpha_count = _alpha_profile(path)
    return AvatarAsset(
        path=path,
        display_name=display_name,
        provider=provider,
        tags=tags,
        max_alpha_fraction=max_alpha,
        alpha_material_count=alpha_count,
    )


def load_avatar_catalog(catalog_path: Path | None) -> dict[str, Any]:
    if not catalog_path:
        return {}
    if not catalog_path.exists():
        raise FileNotFoundError(f"avatar catalog does not exist: {catalog_path}")
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"avatar catalog must be a JSON object: {catalog_path}")
    return data


def build_avatar_index(avatar_root: Path, catalog_path: Path | None = None) -> list[AvatarAsset]:
    catalog = load_avatar_catalog(catalog_path)
    assets = [_asset_from_path(path, catalog) for path in sorted(avatar_root.glob("*.glb"))]
    if not assets:
        raise FileNotFoundError(f"no .glb avatars found under {avatar_root}")
    return assets


def score_avatar(prompt: str, asset: AvatarAsset) -> AvatarMatch:
    prompt_tokens = _tokenize(prompt)
    asset_tokens = asset.tokens
    overlap = prompt_tokens & asset_tokens

    score = float(len(overlap) * 8)
    reasons: list[str] = []
    if overlap:
        reasons.append(f"matched tokens: {', '.join(sorted(overlap)[:8])}")

    display_lower = asset.display_name.lower()
    if display_lower and display_lower in prompt.lower():
        score += 80
        reasons.append(f"explicit name match: {asset.display_name}")

    if "male" in prompt_tokens or "female" in prompt_tokens:
        wanted = "female" if "female" in prompt_tokens else "male"
        if wanted in asset.tags:
            score += 30
            reasons.append(f"gender tag: {wanted}")
        elif {"male", "female"} & asset.tags:
            score -= 25

    for tag in ("police", "office", "business", "suit", "uniform", "formal"):
        if tag in prompt_tokens and tag in asset.tags:
            score += 20
            reasons.append(f"appearance tag: {tag}")

    if asset.provider == "mixamo":
        score += 8
        reasons.append("Mixamo/ViCo skeleton compatibility")
    has_strong_suit_match = bool({"suit", "formal"} & prompt_tokens & asset.tags)
    if asset.max_alpha_fraction > 0.5 and has_strong_suit_match and asset.alpha_material_count <= 1:
        score -= 5
        reasons.append(f"single alpha atlas allowed with solidified shell: {asset.max_alpha_fraction:.2f}")
    elif asset.max_alpha_fraction > 0.5:
        score -= 120
        reasons.append(f"large transparent texture area unsafe for Habitat: {asset.max_alpha_fraction:.2f}")
    elif asset.max_alpha_fraction > 0.1:
        score -= 45
        reasons.append(f"moderate transparent texture area: {asset.max_alpha_fraction:.2f}")
    elif asset.alpha_material_count == 0:
        score += 12
        reasons.append("opaque material profile")
    if "merge" in asset.path.stem.lower():
        score -= 6
    if "decimated" in asset.path.stem.lower():
        score -= 4
    if "generic" in asset.tags:
        score -= 3

    if not reasons:
        reasons.append("fallback best available humanoid GLB")
    return AvatarMatch(asset=asset, score=score, reasons=tuple(reasons))


def select_avatar(
    prompt: str,
    avatar_root: Path,
    *,
    preferred_avatar: Path | None = None,
    catalog_path: Path | None = None,
) -> tuple[AvatarMatch, list[AvatarMatch]]:
    if preferred_avatar:
        asset = _asset_from_path(preferred_avatar, load_avatar_catalog(catalog_path))
        return AvatarMatch(asset=asset, score=float("inf"), reasons=("explicit avatar path",)), []

    matches = [score_avatar(prompt, asset) for asset in build_avatar_index(avatar_root, catalog_path)]
    matches.sort(key=lambda item: (item.score, item.asset.provider == "mixamo"), reverse=True)
    return matches[0], matches[:5]
