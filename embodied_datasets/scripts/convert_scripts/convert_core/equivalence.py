"""Semantic equivalence checks for serial and parallel LeRobot outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from convert_core.errors import ConversionError


@dataclass(frozen=True)
class EquivalenceReport:
    schema: bool
    indices_and_values: bool
    episode_and_task_boundaries: bool
    video_frames: bool
    manifest: bool
    total_frames: int
    total_episodes: int
    video_frames_compared: int

    def as_dict(self) -> dict[str, bool | int]:
        return asdict(self)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON for equivalence check: {path}: {exc}") from exc


def _concat_parquet(root: Path, relative: str) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise ConversionError(f"no Parquet files under {root / relative}")
    tables = [pq.read_table(path) for path in paths]
    try:
        return pa.concat_tables(tables).combine_chunks()
    except pa.ArrowInvalid as exc:
        raise ConversionError(f"incompatible Parquet schemas under {root / relative}: {exc}") from exc


def _assert_tables_equal(label: str, reference: Any, candidate: Any) -> None:
    if reference.schema.remove_metadata() != candidate.schema.remove_metadata():
        raise ConversionError(
            f"{label} schema differs:\nreference={reference.schema}\ncandidate={candidate.schema}"
        )
    if not reference.equals(candidate, check_metadata=False):
        if reference.num_rows != candidate.num_rows:
            detail = f"row count {reference.num_rows} != {candidate.num_rows}"
        else:
            differing = [
                name
                for name in reference.column_names
                if not reference[name].equals(candidate[name])
            ]
            detail = f"column values differ: {differing}"
        raise ConversionError(f"{label} differs: {detail}")


def _numbers_close(reference: Any, candidate: Any) -> bool:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return set(reference) == set(candidate) and all(
            _numbers_close(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list) and isinstance(candidate, list):
        return len(reference) == len(candidate) and all(
            _numbers_close(left, right) for left, right in zip(reference, candidate, strict=True)
        )
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        return bool(np.isclose(reference, candidate, rtol=1e-6, atol=1e-8, equal_nan=True))
    return reference == candidate


def _semantic_manifest(path: Path) -> dict[str, Any]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ConversionError(f"conversion manifest is not an object: {path}")
    normalized = dict(value)
    normalized.pop("dataset_uid", None)
    for runtime_key in ("resume", "parallel", "video_validation"):
        normalized.pop(runtime_key, None)
    return normalized


def _decoded_frames(paths: list[Path]) -> Iterator[np.ndarray]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyAV is required for video equivalence checks") from exc
    for path in paths:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ConversionError(f"video has no stream: {path}")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")


def _compare_videos(reference: Path, candidate: Path, video_keys: list[str]) -> int:
    compared = 0
    sentinel = object()
    for key in video_keys:
        reference_paths = sorted((reference / "videos" / key).rglob("*.mp4"))
        candidate_paths = sorted((candidate / "videos" / key).rglob("*.mp4"))
        if not reference_paths or not candidate_paths:
            raise ConversionError(f"missing video files for {key!r}")
        for index, (left, right) in enumerate(
            zip_longest(
                _decoded_frames(reference_paths),
                _decoded_frames(candidate_paths),
                fillvalue=sentinel,
            )
        ):
            if left is sentinel or right is sentinel:
                raise ConversionError(f"video frame count differs for {key!r}")
            if not np.array_equal(left, right):
                raise ConversionError(f"decoded video differs for {key!r} at frame {index}")
            compared += 1
    return compared


def verify_lerobot_equivalence(
    reference: Path,
    candidate: Path,
    *,
    compare_video_frames: bool = True,
) -> EquivalenceReport:
    """Assert schema, values, boundaries, frames, and semantic manifest equality."""

    reference_info = _json(reference / "meta" / "info.json")
    candidate_info = _json(candidate / "meta" / "info.json")
    if reference_info != candidate_info:
        raise ConversionError("meta/info.json differs between serial and parallel outputs")

    reference_data = _concat_parquet(reference, "data")
    candidate_data = _concat_parquet(candidate, "data")
    _assert_tables_equal("frame data", reference_data, candidate_data)

    reference_episodes = _concat_parquet(reference, "meta/episodes")
    candidate_episodes = _concat_parquet(candidate, "meta/episodes")
    _assert_tables_equal("episode metadata", reference_episodes, candidate_episodes)

    # Compare the dedicated task tables directly to avoid file-layout dependence.
    import pyarrow.parquet as pq

    reference_tasks = pq.read_table(reference / "meta" / "tasks.parquet")
    candidate_tasks = pq.read_table(candidate / "meta" / "tasks.parquet")
    _assert_tables_equal("task metadata", reference_tasks, candidate_tasks)

    reference_stats = _json(reference / "meta" / "stats.json")
    candidate_stats = _json(candidate / "meta" / "stats.json")
    if not _numbers_close(reference_stats, candidate_stats):
        differing = [
            key
            for key in sorted(set(reference_stats) | set(candidate_stats))
            if not _numbers_close(reference_stats.get(key), candidate_stats.get(key))
        ]
        raise ConversionError(
            f"meta/stats.json differs between serial and parallel outputs: {differing}"
        )

    reference_manifest = _semantic_manifest(reference / "conversion_manifest.json")
    candidate_manifest = _semantic_manifest(candidate / "conversion_manifest.json")
    if not _numbers_close(reference_manifest, candidate_manifest):
        raise ConversionError("semantic conversion manifest differs")

    video_keys = sorted(
        key
        for key, feature in reference_info["features"].items()
        if feature.get("dtype") == "video"
    )
    compared = _compare_videos(reference, candidate, video_keys) if compare_video_frames else 0
    return EquivalenceReport(
        schema=True,
        indices_and_values=True,
        episode_and_task_boundaries=True,
        video_frames=True,
        manifest=True,
        total_frames=int(reference_info["total_frames"]),
        total_episodes=int(reference_info["total_episodes"]),
        video_frames_compared=compared,
    )
