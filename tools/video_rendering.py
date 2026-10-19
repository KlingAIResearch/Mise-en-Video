"""Render aligned OABBs and combine per-shot videos.

The pure bbox output never contains identity text. Only the source-video overlay
uses the validated cross-shot ``new_id_*`` labels.
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

from pipeline_common import ROOT


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _load_bbox(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        if "shot_human_bbox" not in payload:
            raise KeyError(f"Missing 'shot_human_bbox' in {path}")
        return payload["shot_human_bbox"].item()


def build_complete_bbox_file(
    preferred_path: Path,
    raw_path: Path,
    output_path: Path,
) -> tuple[int, list[str]]:
    """Keep smoothed records and restore identities omitted during smoothing."""
    preferred = _load_bbox(preferred_path)
    raw = _load_bbox(raw_path)
    restored_records = 0
    restored_ids: set[str] = set()
    for frame_id, raw_records in raw.items():
        output_records = preferred.setdefault(frame_id, {})
        existing_ids = {str(human_id) for human_id in output_records}
        for human_id, record in raw_records.items():
            if str(human_id) in existing_ids:
                continue
            output_records[human_id] = record
            existing_ids.add(str(human_id))
            restored_records += 1
            restored_ids.add(str(human_id))
    np.savez(output_path, shot_human_bbox=preferred)
    return restored_records, sorted(restored_ids, key=int)


def _color_for_id(global_id: str) -> tuple[int, int, int]:
    numeric_id = int(global_id.rsplit("_", 1)[-1])
    hue = (numeric_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def _as_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _project_vertices(vertices: object, intrinsics: object) -> np.ndarray | None:
    xyz = _as_numpy(vertices).reshape(-1, 3).astype(np.float64)
    matrix = _as_numpy(intrinsics).reshape(3, 3).astype(np.float64)
    if xyz.shape[0] != 8 or np.any(xyz[:, 2] <= 1e-6):
        return None
    pixels = (matrix @ xyz.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    if not np.all(np.isfinite(pixels)):
        return None
    return np.rint(pixels).astype(np.int32)


def _display_id(identity_map: dict[str, str], local_id: object) -> str:
    key = str(local_id)
    if key not in identity_map:
        raise KeyError(f"No new_id mapping for local tracking id {key}")
    return identity_map[key]


def _draw_identity_labels(
    frame: np.ndarray,
    records: dict,
    identity_map: dict[str, str],
    line_width: int,
) -> None:
    height, width = frame.shape[:2]
    for local_id, record in records.items():
        oabb = record.get("oabb", {})
        vertices = oabb.get("vertices") if isinstance(oabb, dict) else None
        if vertices is None:
            continue
        pixels = _project_vertices(vertices, record["scene_intr"])
        if pixels is None:
            continue
        label = _display_id(identity_map, local_id)
        center_x = int(np.mean(pixels[:, 0]))
        top_y = int(np.min(pixels[:, 1])) - 8
        anchor = (
            int(np.clip(center_x, 8, max(8, width - 160))),
            int(np.clip(top_y, 28, max(28, height - 8))),
        )
        cv2.putText(
            frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 0), line_width + 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            _color_for_id(label), line_width, cv2.LINE_AA,
        )


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {path}")
    return capture


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise OSError(f"Cannot create video: {path}")
    return writer


def render_shot_wireframe(
    video_path: Path,
    bbox_path: Path,
    output_path: Path,
    save_frames: bool,
    line_width: int,
) -> int:
    """Draw local-ID-colored wireframes without rendering any text labels."""
    bbox = _load_bbox(bbox_path)
    capture = _open_video(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    writer = _open_writer(output_path, fps, size)
    frame_output = output_path.parent / "bbox_frames"
    if save_frames:
        frame_output.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for local_id, record in bbox.get(frame_id, {}).items():
                oabb = record.get("oabb", {})
                vertices = oabb.get("vertices") if isinstance(oabb, dict) else None
                if vertices is None:
                    continue
                pixels = _project_vertices(vertices, record["scene_intr"])
                if pixels is None:
                    continue
                color = _color_for_id(f"new_id_{int(local_id)}")
                for start, end in BOX_EDGES:
                    cv2.line(
                        frame, tuple(pixels[start]), tuple(pixels[end]),
                        color, line_width, cv2.LINE_AA,
                    )
            writer.write(frame)
            if save_frames:
                cv2.imwrite(str(frame_output / f"{frame_id:04d}.jpg"), frame)
            frame_id += 1
    finally:
        capture.release()
        writer.release()
    return frame_id


def _nocs_renderer():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mev_matplotlib_cache")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bbox_backend import render_oabb_colormap_camera_coord_batched

    return render_oabb_colormap_camera_coord_batched


def render_shot_nocs(
    video_path: Path,
    bbox_path: Path,
    nocs_output: Path,
    overlay_output: Path,
    identity_map: dict[str, str],
    save_frames: bool,
    line_width: int,
    batch_size: int,
    overlay_alpha: float,
) -> int:
    """Rasterize canonical-coordinate box colors in bounded frame batches."""
    if batch_size < 1:
        raise ValueError("render batch size must be positive")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay alpha must be between 0 and 1")

    bbox = _load_bbox(bbox_path)
    render_nocs = _nocs_renderer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capture = _open_video(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nocs_writer = _open_writer(nocs_output, fps, (width, height))
    try:
        overlay_writer = _open_writer(overlay_output, fps, (width, height))
    except Exception:
        capture.release()
        nocs_writer.release()
        raise

    frame_output = nocs_output.parent / "bbox_nocs_frames"
    if save_frames:
        frame_output.mkdir(parents=True, exist_ok=True)

    frame_id = 0
    try:
        while True:
            chunk: list[tuple[int, np.ndarray]] = []
            for _ in range(batch_size):
                ok, frame = capture.read()
                if not ok:
                    break
                chunk.append((frame_id, frame))
                frame_id += 1
            if not chunk:
                break

            render_ids: list[int] = []
            oabb_batches: list[torch.Tensor] = []
            intrinsics: list[torch.Tensor] = []
            for current_id, _ in chunk:
                boxes: list[torch.Tensor] = []
                matrix = None
                for record in bbox.get(current_id, {}).values():
                    oabb = record.get("oabb", {})
                    vertices = oabb.get("vertices") if isinstance(oabb, dict) else None
                    if vertices is None:
                        continue
                    boxes.append(torch.as_tensor(vertices, dtype=torch.float32).reshape(8, 3))
                    if matrix is None:
                        matrix = torch.as_tensor(record["scene_intr"], dtype=torch.float32)
                if boxes:
                    render_ids.append(current_id)
                    oabb_batches.append(torch.stack(boxes))
                    intrinsics.append(matrix)

            rendered_by_id: dict[int, np.ndarray] = {}
            if render_ids:
                with torch.inference_mode():
                    rendered = render_nocs(
                        oabb_list=oabb_batches,
                        K_org_list=intrinsics,
                        img_size=(height, width),
                        device=device,
                    ).detach().cpu().numpy()
                rendered_by_id = dict(zip(render_ids, rendered))

            for current_id, source_frame in chunk:
                rgba = rendered_by_id.get(current_id)
                nocs_frame = np.zeros_like(source_frame)
                overlay_frame = source_frame.copy()
                if rgba is not None:
                    # PyTorch3D emits canonical RGB=(X,Y,Z). OpenCV interprets the
                    # bytes as BGR, displaying the required RGB=(Z,Y,X) convention.
                    bgr = np.clip(rgba[..., :3] * 255.0, 0, 255).astype(np.uint8)
                    alpha = np.clip(rgba[..., 3:4], 0.0, 1.0).astype(np.float32)
                    nocs_frame = np.where(alpha > 0, bgr, nocs_frame).astype(np.uint8)
                    blend = alpha * overlay_alpha
                    overlay_frame = np.clip(
                        source_frame.astype(np.float32) * (1.0 - blend)
                        + bgr.astype(np.float32) * blend,
                        0,
                        255,
                    ).astype(np.uint8)
                _draw_identity_labels(
                    overlay_frame, bbox.get(current_id, {}), identity_map, line_width
                )
                nocs_writer.write(nocs_frame)
                overlay_writer.write(overlay_frame)
                if save_frames:
                    cv2.imwrite(str(frame_output / f"{current_id:04d}.png"), nocs_frame)
    finally:
        capture.release()
        nocs_writer.release()
        overlay_writer.release()
    return frame_id


def concatenate_videos(inputs: list[Path], output: Path) -> None:
    """Concatenate decoded frames, normalizing shot resolution when necessary."""
    if not inputs:
        raise ValueError("No shot videos to concatenate")
    first = _open_video(inputs[0])
    fps = first.get(cv2.CAP_PROP_FPS) or 25.0
    size = (
        int(first.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    first.release()
    writer = _open_writer(output, fps, size)
    try:
        for path in inputs:
            capture = _open_video(path)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                capture.release()
    finally:
        writer.release()
