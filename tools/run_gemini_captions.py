#!/usr/bin/env python3
"""Run the four PDF-specified Gemini caption and subject-identity stages."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from identity import appearance_sort_key, normalize_subject_matches
from pipeline_common import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEO_DIR,
    discover_global_video,
    discover_shots,
    read_json,
    video_info,
    write_json,
)


def parse_json_response(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1])
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Gemini response must be a JSON object")
    return parsed


class GeminiCaptioner:
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.types = types

    def upload(self, path: Path):
        """Create an inline media part for a self-contained Gemini request."""
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            raise ValueError(f"Cannot determine MIME type for {path}")
        return self.types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)

    def generate(self, contents: list[Any]) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return parse_json_response(response.text)


# Transcribed from the four prompt specifications under ``pdf/``. The repository-level
# requirement to retain every tracking ID overrides the PDF's optional top-four pruning.
GLOBAL_PROMPT = """System Instruction:
You are a multi-shot video understanding expert that only outputs video captions.

Task Overview:
Analyze the entire multi-shot video and generate one detailed global caption describing
the overall event, main subjects, scene, and visual style.

Task Requirements:
1. Briefly summarize the overall event of the video.
2. Identify up to four important subjects and describe each subject's stable appearance,
   including age, gender, hair, clothing, and distinctive accessories when visible.
3. Describe the same subject only once even if they appear in multiple shots. Ignore
   subjects that are unimportant, too small, or only briefly visible.
4. Describe the overall environment and important scene elements, such as furniture,
   objects, decorations, and background structures.
5. Describe the overall visual style, lighting, and color characteristics when clearly
   observable.
6. Focus on information that is stable or meaningful across the whole video. Avoid
   detailed shot-specific positions, movements, camera motion, shot scale, or camera angle.
7. Describe only visually observable information and do not infer names, relationships,
   occupations, dialogue, or intentions.

Expected Output Format:
{"global_caption": "FPS-{fps}. The video plays at normal speed. [Overall event.] There
are [number] main subjects in the video. [Detailed descriptions of the main subjects.]
[Detailed description of the overall scene and environment.] [Overall visual style,
lighting, and color characteristics.]"}

Input Multi-Shot Video:
The attached file is the entire multi-shot video. Its FPS is {fps}."""

SHOT_PROMPT = """System Instruction:
You are a video understanding expert that only outputs video captions.

Global Caption:
{global_caption}

Task Overview:
Describe the provided shot, focusing on what happens within this shot.

Task Requirements:
1. Use the Global Caption to consistently identify the subjects.
2. Describe the subjects' actions, movement, interactions, and spatial positions.
3. Describe important subject orientations, such as front, back, or side view.
4. Describe the shot scale, camera angle, and camera movement when visible.
5. Avoid repeating detailed subject appearances or scene descriptions from the Global
   Caption.
6. Describe only visually observable information.

Expected Output Format:
{"shot_caption": "[Detailed natural-language description of this shot.]"}

Input Shot:
The attached file is the video for this shot."""

SUBJECT_MATCH_PROMPT = """System Instruction:
You are a subject identity matching expert that only outputs valid JSON.

Task Overview:
Your task is to match subject identities across different shots. There are {frame_num}
subject images. Their names are: {appearance_names}. The images are provided in the same
order as their names. Group images that represent the same physical subject across
different shots into global identities named new_id_0, new_id_1, etc.

Task Requirements:
1. Merge images only when they are likely to show the same subject, not merely subjects
   with similar appearance.
2. Match identities using face, hairstyle, clothing, accessories, body shape, skin tone,
   and apparent age. Use facial features as the strongest cue when clearly visible.
3. The same subject may appear from different viewpoints, scales, poses, or lighting.
4. Different IDs from the same shot should normally be treated as different identities
   unless there is clear evidence of duplicate detection.
5. Do not force uncertain matches. A clear subject with no reliable match forms its own
   group.
6. Retain every supplied image; do not discard an ID or apply a top-k identity limit.
7. Each supplied image must belong to exactly one group.
8. Use the original image names exactly as provided.
9. Return only valid JSON without any explanation.

Expected Output:
{"new_id_0": ["shot_0-id_0", "shot_1-id_2"],
 "new_id_1": ["shot_0-id_1", "shot_1-id_1"]}

Input Subject Images:
The attached files are the subject images in the order listed above."""

SUBJECT_CAPTION_PROMPT = """System Instruction:
You are a subject image matching expert that only outputs valid JSON.

Task Overview:
Match each subject image to the corresponding subject description in the provided global
caption.

Global Caption:
{global_caption}

The subject identifiers are: {subject_names}
Each image corresponds to one subject identifier in the same order.

Task Requirements:
1. Identify the person using visual appearance such as face, hairstyle, clothing,
   accessories, age, and other distinctive features.
2. Match each subject to the most appropriate subject description from the Global Caption.
3. Return only that subject's description, not actions, scene descriptions, camera
   information, or unrelated content.
4. Preserve the wording of the Global Caption as much as possible. Minor edits are allowed
   only to make the description self-contained and fluent.
5. Do not assign different identifiers to the same person description unless visual
   evidence clearly indicates they are the same subject.
6. If no reliable matching description exists, return an empty string.
7. Return only valid JSON with no explanation.

Expected Output:
{"new_id_0": "[Subject description.]", "new_id_1": "[Subject description.]"}

Input Subject Images:
The attached files are the representative subject images in identifier order."""


def require_key(result: dict[str, Any], key: str, stage: str) -> str:
    value = result.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Gemini {stage} response must contain string key {key!r}")
    return value


def representative_images(
    matches: dict[str, list[str]], reference_dir: Path
) -> list[tuple[str, Path]]:
    metadata_path = reference_dir / "face_id_info.json"
    metadata = (
        read_json(metadata_path)
        if metadata_path.exists()
        else {}
    )
    selected = []
    for subject_id, appearances in matches.items():
        candidates = [
            (reference_dir / f"{name}.jpg", metadata.get(name, {}).get("pixel_count", 0))
            for name in appearances
        ]
        candidates = [(path, score) for path, score in candidates if path.exists()]
        if not candidates:
            raise FileNotFoundError(f"No reference image found for {subject_id}: {appearances}")
        path, _ = max(candidates, key=lambda item: float(item[1]))
        selected.append((subject_id, path))
    return selected


def api_key_from_environment() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) before calling the Gemini API"
        )
    return key


def generate_global_caption(
    captioner: GeminiCaptioner, args: argparse.Namespace, stage_dir: Path
) -> str:
    global_video = discover_global_video(args.video_dir)
    fps = video_info(global_video).fps
    global_prompt = GLOBAL_PROMPT.replace("{fps}", f"{fps:g}")
    global_result = captioner.generate([captioner.upload(global_video), global_prompt])
    global_caption = require_key(global_result, "global_caption", "global caption")
    write_json(stage_dir / "global.json", global_result)
    return global_caption


def load_global_caption(stage_dir: Path) -> str:
    path = stage_dir / "global.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run --stage global before this stage"
        )
    value = read_json(path)
    return require_key(value, "global_caption", "saved global caption")


def generate_shot_captions(
    captioner: GeminiCaptioner,
    args: argparse.Namespace,
    stage_dir: Path,
    global_caption: str,
) -> dict[str, str]:
    shot_results = {}
    for shot_id, path in discover_shots(args.video_dir):
        shot_prompt = SHOT_PROMPT.replace("{global_caption}", global_caption)
        result = captioner.generate([captioner.upload(path), shot_prompt])
        shot_results[f"shot_{shot_id}"] = require_key(
            result, "shot_caption", f"shot_{shot_id} caption"
        )
        write_json(stage_dir / f"shot_{shot_id}.json", result)
    return shot_results


def generate_subject_matches(
    captioner: GeminiCaptioner, args: argparse.Namespace, stage_dir: Path
) -> dict[str, list[str]]:
    reference_dir = args.results_dir / "ref_imgs"
    if not reference_dir.is_dir():
        raise FileNotFoundError(
            f"Missing subject reference directory: {reference_dir}. "
            "Run tools/run_tracking.py --refs-only first."
        )
    reference_paths = sorted(reference_dir.glob("shot_*-id_*.jpg"), key=appearance_sort_key)
    if not reference_paths:
        raise FileNotFoundError(f"No shot_N-id_M.jpg reference images under {reference_dir}")
    appearance_names = [path.stem for path in reference_paths]
    match_prompt = (
        SUBJECT_MATCH_PROMPT.replace("{frame_num}", str(len(reference_paths)))
        .replace("{appearance_names}", json.dumps(appearance_names, ensure_ascii=False))
    )
    raw_matches = captioner.generate(
        [captioner.upload(path) for path in reference_paths] + [match_prompt]
    )
    matches = normalize_subject_matches(raw_matches, appearance_names)
    write_json(stage_dir / "subject_matching.json", matches)
    write_json(args.results_dir / "matched_id.json", matches)
    return matches


def load_subject_matches(results_dir: Path) -> dict[str, list[str]]:
    path = results_dir / "matched_id.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run --stage match before subject")
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def generate_subject_captions(
    captioner: GeminiCaptioner,
    args: argparse.Namespace,
    stage_dir: Path,
    global_caption: str,
    matches: dict[str, list[str]],
) -> dict[str, str]:
    reference_dir = args.results_dir / "ref_imgs"
    representatives = representative_images(matches, reference_dir)
    subject_names = [subject_id for subject_id, _ in representatives]
    subject_prompt = (
        SUBJECT_CAPTION_PROMPT.replace("{global_caption}", global_caption)
        .replace("{subject_names}", json.dumps(subject_names, ensure_ascii=False))
    )
    raw_subjects = captioner.generate(
        [captioner.upload(path) for _, path in representatives] + [subject_prompt]
    )
    unknown_subjects = set(raw_subjects) - set(subject_names)
    if unknown_subjects:
        raise ValueError(f"Gemini returned unknown subject IDs: {sorted(unknown_subjects)}")
    subject_results = {}
    for subject_id in subject_names:
        value = raw_subjects.get(subject_id, "")
        if not isinstance(value, str):
            raise ValueError(f"Subject caption for {subject_id} must be a string")
        subject_results[subject_id] = value
    write_json(stage_dir / "subjects.json", subject_results)
    return subject_results


def run_api(args: argparse.Namespace) -> Path:
    captioner = GeminiCaptioner(api_key_from_environment(), args.model)
    stage_dir = args.results_dir / "captions"
    stage_dir.mkdir(parents=True, exist_ok=True)

    global_caption = None
    matches = None
    if args.stage in ("global", "caption", "all"):
        global_caption = generate_global_caption(captioner, args, stage_dir)

    if args.stage in ("shot", "caption", "all"):
        global_caption = global_caption or load_global_caption(stage_dir)
        generate_shot_captions(captioner, args, stage_dir, global_caption)

    if args.stage in ("match", "merge", "all"):
        matches = generate_subject_matches(captioner, args, stage_dir)

    if args.stage in ("subject", "caption", "all"):
        global_caption = global_caption or load_global_caption(stage_dir)
        matches = matches or load_subject_matches(args.results_dir)
        generate_subject_captions(
            captioner,
            args,
            stage_dir,
            global_caption,
            matches,
        )
    return stage_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument(
        "--stage",
        choices=("global", "shot", "match", "merge", "subject", "caption", "all"),
        default="all",
        help=(
            "Run one Gemini stage; caption runs global, shot, and subject without "
            "changing matched_id.json, while merge is an alias for match"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.video_dir = args.video_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    output_dir = run_api(args)
    print(f"Saved {args.stage} caption stage output under {output_dir}")


if __name__ == "__main__":
    main()
