"""Lightweight filesystem helpers shared by all pipeline stages.

Keep this module free of model-framework imports so orchestration and validation can
run before a stage loads CUDA, OpenCV, or a third-party repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "example" / "video"
DEFAULT_RESULTS_DIR = ROOT / "example" / "results"


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class ShotPaths:
    """Canonical artifact locations for one shot."""

    root: Path

    @property
    def pi3(self) -> Path:
        return self.root / "pi3_all_results.npz"

    @property
    def tracks(self) -> Path:
        return self.root / "tracks.npy"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def camerahmr(self) -> Path:
        return self.root / "camerahmr"

    @property
    def camerahmr_json(self) -> Path:
        return self.camerahmr / "camerahmr_json"

    @property
    def bbox_raw(self) -> Path:
        return self.root / "human_bbox.npz"

    @property
    def bbox_smoothed(self) -> Path:
        return self.root / "human_bbox_smoothed.npz"

    @property
    def bbox_render(self) -> Path:
        return self.root / "human_bbox_render.npz"


def video_info(path: Path) -> VideoInfo:
    import cv2

    path = path.resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    info = VideoInfo(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def discover_shots(video_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(shot_id, path)`` pairs sorted by numeric shot id."""
    video_dir = video_dir.resolve()
    pairs: list[tuple[int, Path]] = []
    pattern = re.compile(r"-shot(\d+)\.mp4$", re.IGNORECASE)
    for path in video_dir.glob("*.mp4"):
        match = pattern.search(path.name)
        if match:
            pairs.append((int(match.group(1)), path.resolve()))
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        raise FileNotFoundError(f"No '*-shotN.mp4' videos found under {video_dir}")
    expected = list(range(len(pairs)))
    actual = [shot_id for shot_id, _ in pairs]
    if actual != expected:
        raise ValueError(f"Shot ids must be contiguous from 0; got {actual}")
    return pairs


def discover_global_video(video_dir: Path) -> Path:
    candidates = [
        path.resolve()
        for path in video_dir.glob("*.mp4")
        if not re.search(r"-shot\d+\.mp4$", path.name, re.IGNORECASE)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one global mp4 under {video_dir}, got {len(candidates)}"
        )
    return candidates[0]


def shot_dir(results_dir: Path, shot_id: int) -> Path:
    result = results_dir.resolve() / f"shot_{shot_id}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def shot_paths(results_dir: Path, shot_id: int, *, create: bool = True) -> ShotPaths:
    """Return canonical paths for ``shot_id`` and optionally create its root."""
    root = results_dir.resolve() / f"shot_{shot_id}"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return ShotPaths(root)


def extract_frames(video: Path, output_dir: Path, overwrite: bool = False) -> list[Path]:
    """Extract every frame as a zero-padded JPEG and return the sorted paths."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    expected = video_info(video).frame_count
    if not overwrite and expected > 0 and len(existing) == expected:
        return existing

    if overwrite:
        for path in existing:
            path.unlink()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        output_path = output_dir / f"{frame_id:04d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            cap.release()
            raise OSError(f"Failed to write frame: {output_path}")
        frame_id += 1
    cap.release()
    if frame_id == 0:
        raise RuntimeError(f"No frames decoded from {video}")
    return sorted(output_dir.glob("*.jpg"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON with a consistent missing-file error."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_pipeline_info(video_dir: Path, results_dir: Path) -> Path:
    global_video = discover_global_video(video_dir)
    shots = discover_shots(video_dir)
    shot_infos = [video_info(path) for _, path in shots]
    frame_ranges: list[list[int]] = []
    cursor = 0
    for info in shot_infos:
        frame_ranges.append([cursor, cursor + info.frame_count])
        cursor += info.frame_count

    payload = {
        "shot_num_from_caption": len(shots),
        "video_path": str(global_video),
        "frame_number": frame_ranges,
        "json_info": {
            "video_path": str(global_video),
            **{f"shot_{shot_id}": str(path) for shot_id, path in shots},
        },
        "videos": {
            "global": asdict(video_info(global_video)),
            "shots": [asdict(info) for info in shot_infos],
        },
    }
    output = results_dir.resolve() / "info.json"
    write_json(output, payload)
    return output


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing {label}:\n  - {joined}")
