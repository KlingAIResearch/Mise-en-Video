#!/usr/bin/env python3
"""Detect people in each shot and track their masks with ViTDet + SAM2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.config import LazyConfig
from pycocotools import mask as mask_utils
from sam2.build_sam import build_sam2_video_predictor

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mev_matplotlib_cache")

from identity import appearance_name
from pipeline_common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEO_DIR,
    ROOT,
    discover_shots,
    extract_frames,
    require_files,
    shot_paths,
    write_json,
)


ASSETS = ROOT / "3rdparty" / "tracking_assets"
CAMERAHMR_DATA = ROOT / "3rdparty" / "CameraHMR" / "data"

import tracking_backend as backend


def fill_missing_frames(raw_tracks: object, frame_count: int) -> np.ndarray:
    """Normalize the object-array format and retain every track on every frame."""
    if raw_tracks is None:
        return np.array({}, dtype=object)
    tracks = raw_tracks.item() if isinstance(raw_tracks, np.ndarray) else raw_tracks
    normalized = {}
    for human_id, per_frame in tracks.items():
        normalized[int(human_id)] = dict(per_frame)
        for frame_id in range(frame_count):
            normalized[int(human_id)].setdefault(
                frame_id,
                {
                    "id": int(human_id),
                    "frame": frame_id,
                    "rle": None,
                    "det": False,
                    "det_box": None,
                    "seg_box": None,
                },
            )
    return np.array(normalized, dtype=object)


def build_models(detector_config: Path, detector_checkpoint: Path, sam_checkpoint: Path):
    config = LazyConfig.load(str(detector_config))
    config.train.init_checkpoint = str(detector_checkpoint)
    for predictor in config.model.roi_heads.box_predictors:
        predictor.test_score_thresh = 0.25
    detector = backend.DefaultPredictor_Lazy(config)
    video_predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(sam_checkpoint)
    )
    return detector, video_predictor


def save_reference_images(
    shot_id: int,
    image_paths: list[Path],
    tracks: object,
    reference_dir: Path,
) -> dict[str, dict[str, object]]:
    """Save the largest valid masked crop for every shot-local track ID."""
    if isinstance(tracks, np.ndarray):
        tracks = tracks.item()
    reference_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, object]] = {}

    for human_id, per_frame in tracks.items():
        best = None
        for frame_id, record in per_frame.items():
            if not record.get("det") or record.get("rle") is None:
                continue
            frame_id = int(frame_id)
            if frame_id < 0 or frame_id >= len(image_paths):
                continue
            box = record.get("seg_box")
            if box is None or len(box) != 4:
                continue
            mask = np.asarray(mask_utils.decode(record["rle"])).squeeze().astype(bool)
            pixel_count = int(mask.sum())
            if pixel_count == 0:
                continue
            if best is None or pixel_count > best[0]:
                best = (pixel_count, frame_id, np.asarray(box).astype(int), mask)

        if best is None:
            print(f"[shot_{shot_id}] no valid reference frame for local ID {human_id}")
            continue

        pixel_count, frame_id, box, mask = best
        frame = cv2.imread(str(image_paths[frame_id]))
        if frame is None or mask.shape != frame.shape[:2]:
            print(f"[shot_{shot_id}] invalid image/mask for local ID {human_id}")
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = sorted((int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))))
        y1, y2 = sorted((int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))))
        if x2 <= x1 or y2 <= y1:
            continue

        masked = frame.copy()
        masked[~mask] = 255
        crop = masked[y1:y2, x1:x2]
        appearance_id = appearance_name(shot_id, human_id)
        output_path = reference_dir / f"{appearance_id}.jpg"
        if not cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Failed to save reference image: {output_path}")
        box_area = max(1, (x2 - x1) * (y2 - y1))
        metadata[appearance_id] = {
            "shot_idx": shot_id,
            "frame_idx": frame_id,
            "score": pixel_count / box_area,
            "pixel_count": pixel_count,
            "seg_box": [x1, y1, x2, y2],
        }
        print(f"[shot_{shot_id}] saved reference image {output_path}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=ASSETS / "pretrain" / "cascade_mask_rcnn_vitdet_h_75ep.py",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=CAMERAHMR_DATA / "pretrained-models" / "model_final_f05665.pkl",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=ASSETS / "sam2" / "sam2.1_hiera_large.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--save-vis", action="store_true", help="Save SAM mask overlays")
    parser.add_argument(
        "--refs-only",
        action="store_true",
        help="Generate ref_imgs from existing tracks.npy without loading tracking models",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = None
    video_predictor = None
    reference_metadata: dict[str, dict[str, object]] = {}
    reference_dir = args.results_dir.resolve() / "ref_imgs"
    if reference_dir.is_symlink():
        target = reference_dir.resolve(strict=False)
        reference_dir.unlink()
        print(f"Removed legacy reference-image symlink {reference_dir} -> {target}")

    for shot_id, video_path in discover_shots(args.video_dir):
        paths = shot_paths(args.results_dir, shot_id)
        tracking_path = paths.tracks
        image_dir = paths.images
        if args.overwrite and not args.refs_only and image_dir.is_symlink():
            target = image_dir.resolve(strict=False)
            image_dir.unlink()
            print(f"[shot_{shot_id}] removed legacy image symlink {image_dir} -> {target}")
        images = extract_frames(
            video_path,
            image_dir,
            overwrite=args.overwrite and not args.refs_only,
        )
        if args.refs_only:
            require_files([tracking_path], f"shot_{shot_id} tracking result")
            print(f"[shot_{shot_id}] using existing {tracking_path}")
            tracks = np.load(tracking_path, allow_pickle=True)
        elif tracking_path.exists() and not args.overwrite:
            print(f"[shot_{shot_id}] using existing {tracking_path}")
            tracks = np.load(tracking_path, allow_pickle=True)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Tracking inference requires CUDA")
            if detector is None or video_predictor is None:
                require_files(
                    [args.detector_config, args.detector_checkpoint, args.sam_checkpoint],
                    "tracking model assets",
                )
                detector, video_predictor = build_models(
                    args.detector_config,
                    args.detector_checkpoint,
                    args.sam_checkpoint,
                )
            print(f"[shot_{shot_id}] tracking {len(images)} frames")
            raw_tracks = backend.detect_sam2(
                detector,
                video_predictor,
                [str(path) for path in images],
                thresh=args.threshold,
                min_size=args.min_size,
                save_vos=args.save_vis,
            )
            tracks = fill_missing_frames(raw_tracks, len(images))
            np.save(tracking_path, tracks)
            print(f"[shot_{shot_id}] saved {tracking_path}")

        reference_metadata.update(
            save_reference_images(
                shot_id,
                images,
                tracks,
                reference_dir,
            )
        )

    reference_info = reference_dir / "face_id_info.json"
    write_json(reference_info, reference_metadata)
    print(f"Saved {len(reference_metadata)} reference-image records to {reference_info}")


if __name__ == "__main__":
    main()
