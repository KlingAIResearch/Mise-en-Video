#!/usr/bin/env python3
"""Run Pi3 per shot and save the single geometry artifact used by the pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pipeline_common import DEFAULT_RESULTS_DIR, DEFAULT_VIDEO_DIR, ROOT, discover_shots, shot_dir


PI3_REPO = ROOT / "3rdparty" / "Pi3"
sys.path.insert(0, str(PI3_REPO))

from pi3.models.pi3 import Pi3  # noqa: E402
from pi3.utils.basic import load_images_as_tensor, write_ply  # noqa: E402
from pi3.utils.geometry import depth_edge  # noqa: E402


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def load_model(checkpoint: Path | None, device: torch.device) -> Pi3:
    if checkpoint is None:
        return Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    model = Pi3().to(device).eval()
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(checkpoint))
    else:
        weights = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(weights)
    return model


def process_shot(
    model: Pi3,
    video_path: Path,
    output_dir: Path,
    device: torch.device,
    interval: int,
    save_ply: bool,
    export_depth: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_images_as_tensor(str(video_path), interval=interval).to(device)
    if images.numel() == 0:
        raise RuntimeError(f"Pi3 received no frames from {video_path}")

    if device.type != "cuda":
        raise RuntimeError("Pi3 inference requires CUDA for this pipeline")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        result = model(images[None])

    # ``depth`` is exactly local_points[..., 2] and confidence is sigmoid(conf), so
    # storing a second depth.npz duplicates tens of MB per shot. Camera poses retain
    # float32 precision; dense maps use float16 to control the main artifact size.
    raw = {
        "points": to_numpy(result["points"]).astype(np.float16),
        "local_points": to_numpy(result["local_points"]).astype(np.float16),
        "conf": to_numpy(result["conf"]).astype(np.float16),
        "camera_poses": to_numpy(result["camera_poses"]).astype(np.float32),
        "source_video": np.array(str(video_path)),
        "frame_interval": np.array(interval, dtype=np.int32),
    }
    raw_path = output_dir / "pi3_all_results.npz"
    np.savez(raw_path, **raw)

    valid_mask = None
    if save_ply or export_depth:
        confidence = torch.sigmoid(result["conf"][..., 0])
        non_edge = ~depth_edge(result["local_points"][..., 2], rtol=0.03)
        valid_mask = torch.logical_and(confidence > 0.1, non_edge)[0]
    if export_depth:
        export_depth_product(raw_path, output_dir / "depth.npz", video_path, interval)

    if save_ply:
        sample = slice(None, None, 15)
        points = result["points"][0][sample]
        mask = valid_mask[sample]
        colors = images.permute(0, 2, 3, 1)[sample]
        write_ply(points[mask], colors[mask], output_dir / "result.ply")


def export_depth_product(raw_path: Path, depth_path: Path, video_path: Path, interval: int) -> None:
    """Optionally export the legacy derived depth artifact for external consumers."""
    with np.load(raw_path) as raw:
        local_depth = raw["local_points"][0, ..., 2].astype(np.float32)
        conf_logits = raw["conf"][0, ..., 0].astype(np.float32)
        camera_poses = raw["camera_poses"][0].astype(np.float32)
    confidence = 1.0 / (1.0 + np.exp(-conf_logits))
    edge = depth_edge(torch.from_numpy(local_depth), rtol=0.03).numpy()
    valid_mask = np.logical_and(confidence > 0.1, ~edge)
    np.savez_compressed(
        depth_path,
        depth=local_depth.astype(np.float16),
        confidence=confidence.astype(np.float16),
        valid_mask=valid_mask,
        camera_poses=camera_poses,
        source_video=np.array(str(video_path)),
        frame_interval=np.array(interval, dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--save-ply", action="store_true", help="Also save a sampled PLY point cloud")
    parser.add_argument(
        "--export-depth",
        action="store_true",
        help="Also write legacy depth.npz (redundant and unused by this pipeline)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = None
    for shot_id, video_path in discover_shots(args.video_dir):
        output = shot_dir(args.results_dir, shot_id)
        raw_result = output / "pi3_all_results.npz"
        depth_result = output / "depth.npz"
        if not args.overwrite and raw_result.exists():
            if args.export_depth and not depth_result.exists():
                print(f"[shot_{shot_id}] exporting depth.npz from existing Pi3 geometry")
                export_depth_product(raw_result, depth_result, video_path, args.interval)
            else:
                print(f"[shot_{shot_id}] Pi3 geometry already exists; skipping")
            continue
        if model is None:
            model = load_model(args.checkpoint, device)
        print(f"[shot_{shot_id}] Pi3: {video_path}")
        process_shot(
            model, video_path, output, device, args.interval,
            args.save_ply, args.export_depth,
        )


if __name__ == "__main__":
    main()
