"""Validation and conversion helpers for cross-shot subject identities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping


APPEARANCE_PATTERN = re.compile(r"^shot_(\d+)-id_(.+)$")
NEW_ID_PATTERN = re.compile(r"^new_id_(\d+)$")


def appearance_name(shot_id: int, local_id: object) -> str:
    return f"shot_{shot_id}-id_{local_id}"


def parse_appearance(value: object) -> tuple[int, str]:
    match = APPEARANCE_PATTERN.fullmatch(str(value))
    if not match:
        raise ValueError(f"Invalid subject appearance identifier: {value!r}")
    return int(match.group(1)), match.group(2)


def appearance_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort numeric local IDs naturally, with non-numeric IDs kept deterministic."""
    try:
        shot_id, local_id = parse_appearance(path.stem)
        return shot_id, int(local_id), path.name
    except (ValueError, TypeError):
        return 10**9, 10**9, path.name


def new_id_sort_key(value: str) -> int:
    match = NEW_ID_PATTERN.fullmatch(str(value))
    if not match:
        raise ValueError(f"Invalid global subject identifier: {value!r}")
    return int(match.group(1))


def normalize_subject_matches(
    raw: Mapping[str, Any], appearances: Iterable[str]
) -> dict[str, list[str]]:
    """Validate matching and restore every omitted appearance as a singleton."""
    ordered_appearances = list(appearances)
    valid = set(ordered_appearances)
    if len(valid) != len(ordered_appearances):
        raise ValueError("Subject appearance identifiers must be unique")

    used: set[str] = set()
    groups: dict[str, list[str]] = {}
    for new_id, members in raw.items():
        new_id_sort_key(str(new_id))
        if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
            raise ValueError(f"Subject identity {new_id!r} must contain a list of names")
        for member in members:
            if member not in valid:
                raise ValueError(f"Unknown subject appearance: {member}")
            if member in used:
                raise ValueError(f"Subject appearance assigned more than once: {member}")
            used.add(member)
        if members:
            groups[str(new_id)] = list(members)

    next_id = max((new_id_sort_key(key) for key in groups), default=-1) + 1
    for missing in ordered_appearances:
        if missing not in used:
            groups[f"new_id_{next_id}"] = [missing]
            next_id += 1
    return dict(sorted(groups.items(), key=lambda item: new_id_sort_key(item[0])))


def build_fallback_matches(track_ids_by_shot: Mapping[int, Iterable[object]]) -> dict[str, list[str]]:
    """Assign a unique global ID to every local track when matching is unavailable."""
    matches: dict[str, list[str]] = {}
    next_id = 0
    for shot_id in sorted(track_ids_by_shot):
        for local_id in sorted(track_ids_by_shot[shot_id], key=lambda value: int(value)):
            matches[f"new_id_{next_id}"] = [appearance_name(shot_id, local_id)]
            next_id += 1
    return matches


def invert_matches(matches: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    """Convert ``new_id -> appearances`` into ``shot -> local_id -> new_id``."""
    result: dict[int, dict[str, str]] = {}
    for new_id, appearances in matches.items():
        new_id_sort_key(str(new_id))
        if not isinstance(appearances, list):
            raise ValueError(f"Subject identity {new_id!r} must contain a list")
        for appearance in appearances:
            shot_id, local_id = parse_appearance(appearance)
            shot_map = result.setdefault(shot_id, {})
            previous = shot_map.setdefault(local_id, str(new_id))
            if previous != str(new_id):
                raise ValueError(
                    f"{appearance} maps to both {previous!r} and {new_id!r}"
                )
    return result
