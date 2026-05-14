from __future__ import annotations

import json
from pathlib import Path

import pytest
from pygltflib import GLTF2

from havln3.avatar_assets import select_avatar
from havln3.pipeline import AvatarActionPipeline, AvatarActionRequest


REPO_ROOT = Path(__file__).resolve().parents[2]
AVATAR_ROOT = REPO_ROOT / "vico_assets_probe" / "models"
KIMODO_SAMPLE = (
    REPO_ROOT
    / "local_experiments"
    / "kimodo_motion_backflip_wave"
    / "office_worker_backflip_then_wave_00.npz"
)

pytestmark = pytest.mark.skipif(
    not AVATAR_ROOT.exists() or not KIMODO_SAMPLE.exists(),
    reason="ViCo avatar probes and Kimodo sample motion are local asset fixtures",
)


def test_select_avatar_allows_strong_suit_match_with_solidified_alpha_atlas() -> None:
    match, top = select_avatar(
        "An office worker in a suit does a backflip and waves.",
        AVATAR_ROOT,
    )

    assert match.asset.path.name == "mixamo_Joe_Anderson.glb"
    assert top
    assert {"office", "suit", "formal"} <= match.asset.tags
    assert match.asset.alpha_material_count <= 1


def test_pipeline_exports_textured_frames_and_skeleton(tmp_path: Path) -> None:
    result = AvatarActionPipeline().run(
        AvatarActionRequest(
            prompt="An office worker in a suit does a backflip and waves.",
            output_root=tmp_path,
            avatar_root=AVATAR_ROOT,
            motion_npz=KIMODO_SAMPLE,
            asset_name="test_office_backflip_wave",
            frames=2,
        )
    )

    assert result.asset_dir.exists()
    assert len(list(result.asset_dir.glob("frame*.glb"))) == 2
    assert len(list(result.asset_dir.glob("frame*.object_config.json"))) == 2
    assert result.skeleton_path.exists()
    assert result.report_path.exists()

    skeleton = json.loads(result.skeleton_path.read_text(encoding="utf-8"))
    assert len(skeleton["joint_names"]) >= 60
    assert len(skeleton["frames"]) == 2
    assert len(skeleton["frames"][0]) == len(skeleton["joint_names"])
    assert skeleton["leg_ik"]["enabled"] is True
    assert skeleton["leg_ik"]["strategy"] == "calibrated two-bone IK with target-rig knee pole constraints"

    frame = GLTF2().load(str(result.asset_dir / "frame000.glb"))
    for material in frame.materials or []:
        assert material.alphaMode != "BLEND"
        assert material.doubleSided is True

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["selection"]["selected"]["path"].endswith("mixamo_Joe_Anderson.glb")
