#!/usr/bin/env python3
"""Estimate per-frame SMPL parameters from the saved tracking results."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from pipeline_common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEO_DIR,
    ROOT,
    discover_shots,
    extract_frames,
    require_files,
    shot_paths,
)


CAMERAHMR_REPO = ROOT / "3rdparty" / "CameraHMR"


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-type", choices=("smpl", "smplx"), default="smpl")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save one OBJ mesh and rendered JPG per detected subject/frame",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CameraHMR inference requires CUDA")
    require_files(
        [CAMERAHMR_REPO / "mesh_estimator.py"], "CameraHMR source checkout"
    )
    sys.path.insert(0, str(CAMERAHMR_REPO))
    with working_directory(CAMERAHMR_REPO):
        from camerahmr_batch import BatchHumanMeshEstimator

        print("Loading CameraHMR models")
        estimator = BatchHumanMeshEstimator(model_type=args.model_type)
        for shot_id, video_path in discover_shots(args.video_dir):
            paths = shot_paths(args.results_dir, shot_id)
            tracking_path = paths.tracks
            require_files([tracking_path], f"shot_{shot_id} tracking result")
            images = extract_frames(video_path, paths.images)
            hmr_output = paths.camerahmr
            json_output = paths.camerahmr_json
            if json_output.exists() and len(list(json_output.glob("*.json"))) == len(images):
                if not args.overwrite:
                    print(f"[shot_{shot_id}] CameraHMR JSON already complete; skipping")
                    continue
            tracks = np.load(tracking_path, allow_pickle=True).item()
            print(f"[shot_{shot_id}] CameraHMR on {len(images)} frames, {len(tracks)} tracks")
            estimator.run_on_images(str(paths.images), str(hmr_output), args.save_vis, tracks)


if __name__ == "__main__":
    main()
