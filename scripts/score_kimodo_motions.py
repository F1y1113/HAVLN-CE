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

from havln3.motion_quality import MotionQualityOptions, rank_motion_files  # noqa: E402


def is_motion_candidate(path: Path) -> bool:
    return path.suffix in {".npz", ".pt"} and not path.stem.startswith("amass_") and not path.stem.endswith("_amass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Kimodo/GEM motion samples before retargeting.")
    parser.add_argument("paths", nargs="+", type=Path, help="Motion .npz/.pt files or directories.")
    parser.add_argument("--prompt", default="", help="Action prompt used for semantic checks such as backflip.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--min-score", type=float, default=62.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--gem-param-group", default="body_params_global")
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.npz")))
            expanded.extend(sorted(path.glob("*.pt")))
            expanded.extend(sorted(path.glob("*/*.npz")))
            expanded.extend(sorted(path.glob("*/*.pt")))
        else:
            expanded.extend(sorted(path.parent.glob(path.name)) if any(ch in path.name for ch in "*?[]") else [path])
    return [path for path in expanded if path.exists() and is_motion_candidate(path)]


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.paths)
    if not paths:
        raise SystemExit("No existing motion files matched.")
    reports = rank_motion_files(
        paths,
        prompt=args.prompt,
        options=MotionQualityOptions(frames=args.frames, min_score=args.min_score),
        gem_param_group=args.gem_param_group,
    )
    payload = {
        "selected": reports[0].to_dict(),
        "candidates": [report.to_dict() for report in reports],
    }
    text = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
