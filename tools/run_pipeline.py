#!/usr/bin/env python3
"""Run the complete example pipeline with the active Python environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEO_DIR,
    ROOT,
    write_pipeline_info,
)


STAGES = ("depth", "tracking", "camerahmr", "match", "bbox")


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--from-stage", choices=STAGES, default="depth")
    parser.add_argument("--to-stage", choices=STAGES, default="bbox")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace, python: str) -> dict[str, list[str]]:
    """Build stage commands separately so their ordering contract is testable."""
    common = ["--video-dir", str(args.video_dir), "--results-dir", str(args.results_dir)]
    commands = {
        "depth": [python, "tools/run_pi3_depth.py", *common],
        "tracking": [python, "tools/run_tracking.py", *common],
        "camerahmr": [python, "tools/run_camerahmr.py", *common],
        "match": [
            python,
            "tools/run_gemini_captions.py",
            *common,
            "--stage",
            "match",
        ],
        "bbox": [python, "tools/run_bbox_video.py", *common],
    }
    if args.save_vis:
        for stage in ("tracking", "camerahmr", "bbox"):
            commands[stage].append("--save-vis")
    if args.overwrite:
        for stage in ("depth", "tracking", "camerahmr"):
            commands[stage].append("--overwrite")
    return commands


def main() -> None:
    args = parse_args()
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise ValueError("--from-stage must not come after --to-stage")
    selected_stages = STAGES[start : end + 1]
    python = str(Path(sys.executable).resolve())

    write_pipeline_info(args.video_dir, args.results_dir)
    commands = build_commands(args, python)
    for stage in selected_stages:
        run(commands[stage], ROOT)


if __name__ == "__main__":
    main()
