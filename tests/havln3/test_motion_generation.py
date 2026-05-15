from __future__ import annotations

from havln3.motion import _generated_motion_candidates


def test_generated_motion_candidates_find_kimodo_output_directory(tmp_path) -> None:
    output_stem = tmp_path / "clean_standing_backflip"
    output_stem.mkdir()
    motion_path = output_stem / "clean_standing_backflip_00.npz"
    amass_path = output_stem / "amass_00.npz"
    motion_path.write_bytes(b"")
    amass_path.write_bytes(b"")

    assert _generated_motion_candidates(output_stem) == [(motion_path, amass_path)]
