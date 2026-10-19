#!/usr/bin/env python3
"""Align CameraHMR meshes to Pi3 geometry, smooth OABBs, and render MP4s."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from identity import build_fallback_matches, invert_matches
from pipeline_common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEO_DIR,
    ROOT,
    discover_shots,
    require_files,
    shot_paths,
    write_json,
    write_pipeline_info,
)
from video_rendering import (
    build_complete_bbox_file,
    concatenate_videos,
    render_shot_nocs,
    render_shot_wireframe,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mev_matplotlib_cache")


def ensure_match_file(results_dir: Path, shot_ids: list[int]) -> Path:
    """Create deterministic singleton identities until Gemini matching is available."""
    path = results_dir / "matched_id.json"
    if path.exists():
        return path
    tracks_by_shot = {}
    for shot_id in shot_ids:
        tracking_path = shot_paths(results_dir, shot_id).tracks
        require_files([tracking_path], f"shot_{shot_id} tracks")
        tracks_by_shot[shot_id] = np.load(
            tracking_path, allow_pickle=True
        ).item().keys()
    write_json(path, build_fallback_matches(tracks_by_shot))
    print(f"Wrote fallback within-shot identities to {path}")
    return path


def load_new_id_map(results_dir: Path) -> dict[int, dict[str, str]]:
    path = results_dir / "matched_id.json"
    require_files([path], "cross-shot identity mapping")
    matches = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(matches, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return invert_matches(matches)


def import_alignment_tools():
    """Delay heavy CUDA imports until input validation has completed."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bbox_backend import align_human_to_scene
    from bbox_smoothing import smooth_shot_bboxes

    return align_human_to_scene, smooth_shot_bboxes


def detach_legacy_bbox_outputs(
    results_dir: Path, shot_ids: list[int], smoothed_name: str
) -> None:
    """Avoid writing regenerated results through links to an old result tree."""
    for shot_id in shot_ids:
        paths = shot_paths(results_dir, shot_id)
        for path in (paths.bbox_raw, paths.root / smoothed_name):
            if path.is_symlink():
                target = path.resolve(strict=False)
                path.unlink()
                print(f"[shot_{shot_id}] removed legacy bbox symlink {path} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render existing bbox NPZ files without alignment or smoothing",
    )
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--keyframe-interval", type=int, default=4)
    parser.add_argument("--bbox-file", default="human_bbox_smoothed.npz")
    parser.add_argument("--save-vis", action="store_true", help="Save alignment debug products")
    parser.add_argument("--save-frames", action="store_true", help="Save rendered image frames")
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--render-mode", choices=("nocs", "wireframe"), default="nocs")
    parser.add_argument("--render-batch-size", type=int, default=8)
    parser.add_argument("--overlay-alpha", type=float, default=0.72)
    return parser.parse_args()


def generate_bboxes(args: argparse.Namespace, shot_ids: list[int]) -> None:
    required: list[Path] = []
    for shot_id in shot_ids:
        paths = shot_paths(args.results_dir, shot_id)
        required.extend((paths.pi3, paths.tracks, paths.camerahmr_json))
    require_files(required, "Pi3/tracking/CameraHMR inputs")
    detach_legacy_bbox_outputs(args.results_dir, shot_ids, args.bbox_file)

    align_human_to_scene, smooth_shot_bboxes = import_alignment_tools()
    completed = align_human_to_scene(
        str(args.results_dir),
        save_vis=args.save_vis,
        keyframe_interval=args.keyframe_interval,
    )
    if completed is False:
        raise RuntimeError("Mesh-to-scene alignment did not complete")
    if not args.no_smooth:
        completed = smooth_shot_bboxes(
            str(args.results_dir), sigma=args.sigma, save_name=args.bbox_file
        )
        if completed is False:
            raise RuntimeError("BBox smoothing did not complete")


def render_videos(
    args: argparse.Namespace,
    shots: list[tuple[int, Path]],
    new_id_maps: dict[int, dict[str, str]],
) -> None:
    bbox_name = "human_bbox.npz" if args.no_smooth else args.bbox_file
    rendered: list[Path] = []
    overlays: list[Path] = []
    for shot_id, video_path in shots:
        paths = shot_paths(args.results_dir, shot_id)
        bbox_path = paths.root / bbox_name
        require_files([bbox_path, paths.bbox_raw], f"shot_{shot_id} bbox result")
        restored_count, restored_ids = build_complete_bbox_file(
            bbox_path, paths.bbox_raw, paths.bbox_render
        )
        if restored_count:
            print(
                f"[shot_{shot_id}] restored {restored_count} records for local IDs "
                f"{restored_ids}; no identity-count reduction is applied"
            )

        if args.render_mode == "nocs":
            output = paths.root / "bbox_nocs.mp4"
            overlay = paths.root / "bbox_overlay.mp4"
            count = render_shot_nocs(
                video_path, paths.bbox_render, output, overlay,
                new_id_maps.get(shot_id, {}), args.save_frames,
                args.line_width, args.render_batch_size, args.overlay_alpha,
            )
            overlays.append(overlay)
            print(f"[shot_{shot_id}] rendered {count} frames to {output} and {overlay}")
        else:
            output = paths.root / "bbox_wireframe.mp4"
            count = render_shot_wireframe(
                video_path, paths.bbox_render, output,
                args.save_frames, args.line_width,
            )
            print(f"[shot_{shot_id}] rendered {count} wireframe frames to {output}")
        rendered.append(output)

    concatenate_videos(rendered, args.results_dir / "bbox.mp4")
    if overlays:
        concatenate_videos(overlays, args.results_dir / "bbox_overlay.mp4")
    print(f"Saved merged bbox video to {args.results_dir / 'bbox.mp4'}")


def main() -> None:
    args = parse_args()
    args.video_dir = args.video_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be positive")
    if args.sigma < 0:
        raise ValueError("--sigma must be non-negative")

    shots = discover_shots(args.video_dir)
    shot_ids = [shot_id for shot_id, _ in shots]
    write_pipeline_info(args.video_dir, args.results_dir)
    ensure_match_file(args.results_dir, shot_ids)
    identity_maps = load_new_id_map(args.results_dir)
    if not args.render_only:
        generate_bboxes(args, shot_ids)
    render_videos(args, shots, identity_maps)


if __name__ == "__main__":
    main()
