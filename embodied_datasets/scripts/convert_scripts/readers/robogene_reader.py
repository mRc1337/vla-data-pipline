"""Metadata-only RoboGene v2.1 inspection.

RoboGene is distributed as one legacy LeRobot dataset per task.  This reader
does not decode frames or concatenate files: it freezes the source inventory,
schema, episode lengths, and field semantics that the bounded converter uses
to copy each episode into a v3 work unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError


_EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")
_VIDEO_RE = re.compile(r"episode_(\d+)\.mp4$")


@dataclass(frozen=True)
class RobogeneEpisodeSource:
    source_id: str
    task_root: Path
    split: str
    task_name: str
    source_episode_index: int
    instruction: str
    stats: dict[str, Any]
    length: int
    data_path: Path
    video_paths: tuple[tuple[str, Path], ...]
    data_bytes: int
    video_bytes: int

    @property
    def estimated_peak_bytes(self) -> int:
        # Data is copied and rewritten once in the local unit.  Keep the
        # original plus the rewrite and all video bytes in the reservation.
        return 2 * self.data_bytes + self.video_bytes + 64 * 1024 * 1024


@dataclass(frozen=True)
class RobogenePartition:
    name: str
    split: str
    schema_fingerprint: str
    plan: DatasetConversionPlan
    source_info: dict[str, Any]
    episodes: tuple[RobogeneEpisodeSource, ...]
    source_files: tuple[dict[str, Any], ...]
    empty_tasks: tuple[str, ...]
    sample_evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RobogeneCatalog:
    partitions: tuple[RobogenePartition, ...]
    fingerprint_payload: dict[str, Any]
    mapping_table: tuple[dict[str, Any], ...]


def catalog_to_payload(catalog: RobogeneCatalog) -> dict[str, Any]:
    """Serialize the metadata-only preflight so resume never rescans source dirs."""

    def source_payload(source: RobogeneEpisodeSource) -> dict[str, Any]:
        return {
            "source_id": source.source_id,
            "task_root": str(source.task_root),
            "split": source.split,
            "task_name": source.task_name,
            "source_episode_index": source.source_episode_index,
            "instruction": source.instruction,
            "stats": source.stats,
            "length": source.length,
            "data_path": str(source.data_path),
            "video_paths": [[key, str(path)] for key, path in source.video_paths],
            "data_bytes": source.data_bytes,
            "video_bytes": source.video_bytes,
        }

    def plan_payload(plan: DatasetConversionPlan) -> dict[str, Any]:
        return {
            "dataset_uid": plan.dataset_uid,
            "output_path": str(plan.output_path),
            "fps": plan.fps,
            "measured_fps": plan.measured_fps,
            "robot_type": plan.robot_type,
            "vectors": [
                {"feature_key": item.feature_key, "dim": item.dim, "names": item.names, "dtype": item.dtype, "shape": item.shape}
                for item in plan.vector_features
            ],
            "cameras": [
                {"feature_key": item.feature_key, "height": item.height, "width": item.width}
                for item in plan.camera_features
            ],
            "episodes": [
                {"episode_uid": item.episode_uid, "source_relative_path": item.source_relative_path, "instruction": item.instruction, "num_frames": item.num_frames, "extra": item.extra}
                for item in plan.episodes
            ],
            "extra": plan.extra,
        }

    return {
        "schema_version": 1,
        "fingerprint_payload": catalog.fingerprint_payload,
        "mapping_table": list(catalog.mapping_table),
        "partitions": [
            {
                "name": item.name,
                "split": item.split,
                "schema_fingerprint": item.schema_fingerprint,
                "plan": plan_payload(item.plan),
                "source_info": item.source_info,
                "episodes": [source_payload(source) for source in item.episodes],
                "source_files": list(item.source_files),
                "empty_tasks": list(item.empty_tasks),
                "sample_evidence": list(item.sample_evidence),
            }
            for item in catalog.partitions
        ],
    }


def catalog_from_payload(payload: dict[str, Any]) -> RobogeneCatalog:
    """Restore a catalog written by :func:`catalog_to_payload`."""

    if payload.get("schema_version") != 1 or not isinstance(payload.get("partitions"), list):
        raise ConversionError("unsupported RoboGene preflight resume catalog")
    partitions: list[RobogenePartition] = []
    try:
        for raw in payload["partitions"]:
            plan_raw = raw["plan"]
            vectors = tuple(
                VectorFeatureSpec(
                    str(item["feature_key"]), int(item["dim"]),
                    names=tuple(item["names"]) if item.get("names") is not None else None,
                    dtype=str(item["dtype"]), shape=tuple(item["shape"]),
                )
                for item in plan_raw["vectors"]
            )
            cameras = tuple(
                CameraFeatureSpec(str(item["feature_key"]), int(item["height"]), int(item["width"]))
                for item in plan_raw["cameras"]
            )
            plan = DatasetConversionPlan(
                dataset_uid=str(plan_raw["dataset_uid"]), output_path=Path(str(plan_raw["output_path"])),
                fps=int(plan_raw["fps"]), measured_fps=float(plan_raw["measured_fps"]),
                robot_type=str(plan_raw["robot_type"]), vector_features=vectors, camera_features=cameras,
                episodes=tuple(EpisodePlan(str(item["episode_uid"]), str(item["source_relative_path"]), str(item["instruction"]), int(item["num_frames"]), dict(item["extra"])) for item in plan_raw["episodes"]),
                extra=dict(plan_raw["extra"]),
            )
            sources = tuple(
                RobogeneEpisodeSource(
                    source_id=str(item["source_id"]), task_root=Path(str(item["task_root"])), split=str(item["split"]),
                    task_name=str(item["task_name"]), source_episode_index=int(item["source_episode_index"]),
                    instruction=str(item["instruction"]), stats=dict(item["stats"]), length=int(item["length"]),
                    data_path=Path(str(item["data_path"])),
                    video_paths=tuple((str(key), Path(str(path))) for key, path in item["video_paths"]),
                    data_bytes=int(item["data_bytes"]), video_bytes=int(item["video_bytes"]),
                )
                for item in raw["episodes"]
            )
            partitions.append(RobogenePartition(
                str(raw["name"]), str(raw["split"]), str(raw["schema_fingerprint"]), plan,
                dict(raw["source_info"]), sources, tuple(dict(item) for item in raw["source_files"]),
                tuple(str(item) for item in raw["empty_tasks"]), tuple(dict(item) for item in raw["sample_evidence"]),
            ))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"invalid RoboGene preflight resume catalog: {exc}") from exc
    fingerprint_payload = payload.get("fingerprint_payload")
    mapping_table = payload.get("mapping_table")
    if not isinstance(fingerprint_payload, dict) or not isinstance(mapping_table, list):
        raise ConversionError("invalid RoboGene preflight resume catalog metadata")
    return RobogeneCatalog(tuple(partitions), fingerprint_payload, tuple(dict(item) for item in mapping_table))


def validate_catalog_source_files(catalog: RobogeneCatalog, raw_root: Path) -> None:
    """Re-stat the frozen source inventory without traversing the raw tree."""

    approved_root = raw_root.resolve(strict=False)

    def validate_path(path: Path, label: str) -> None:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(approved_root):
            raise ConversionError(f"RoboGene resume {label} escapes the raw root: {path}")
        if not path.is_file() and not path.is_dir():
            raise ConversionError(f"RoboGene resume {label} is unavailable: {path}")

    for partition in catalog.partitions:
        for record in partition.source_files:
            relative = record.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ConversionError("invalid source path in RoboGene resume catalog")
            path = raw_root / relative
            validate_path(path, "source file")
            try:
                stat = path.stat()
            except OSError as exc:
                raise ConversionError(f"RoboGene resume source file is unavailable: {path}: {exc}") from exc
            if stat.st_size != int(record.get("size", -1)) or stat.st_mtime_ns != int(record.get("mtime_ns", -1)):
                raise ConversionError(f"RoboGene resume source fingerprint changed: {path}")
        for source in partition.episodes:
            validate_path(source.task_root, "task root")
            validate_path(source.data_path, "data file")
            for _key, video_path in source.video_paths:
                validate_path(video_path, "video file")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read RoboGene metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"RoboGene metadata must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConversionError(f"cannot read RoboGene metadata {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ConversionError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _file_inventory(root: Path) -> tuple[tuple[dict[str, Any], ...], tuple[Path, ...]]:
    records: list[dict[str, Any]] = []
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    for path in files:
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return tuple(records), files


def _schema_text(path: Path) -> str:
    import pyarrow.parquet as pq

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception as exc:
        raise ConversionError(f"cannot inspect Parquet footer {path}: {exc}") from exc
    return schema.to_string(show_field_metadata=True)


def _episode_index(path: Path, pattern: re.Pattern[str], label: str) -> int:
    match = pattern.search(path.name)
    if match is None:
        raise ConversionError(f"invalid {label} filename: {path}")
    return int(match.group(1))


def _feature_specs(info: dict[str, Any]) -> tuple[tuple[VectorFeatureSpec, ...], tuple[CameraFeatureSpec, ...]]:
    features = info.get("features")
    if not isinstance(features, dict) or not features:
        raise ConversionError("RoboGene info.json has no features")
    vectors: list[VectorFeatureSpec] = []
    cameras: list[CameraFeatureSpec] = []
    for key, raw in features.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("shape"), list):
            raise ConversionError(f"invalid RoboGene feature definition: {key}")
        shape = tuple(int(value) for value in raw["shape"])
        dtype = raw.get("dtype")
        names = raw.get("names")
        name_tuple = tuple(str(value) for value in names) if isinstance(names, list) else None
        if dtype == "video":
            video_info = raw.get("info")
            if not isinstance(video_info, dict) or len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
                raise ConversionError(f"video feature lacks info: {key}")
            cameras.append(CameraFeatureSpec(key, int(shape[0]), int(shape[1])))
        elif dtype in {"image", "float32", "float64", "int64", "int32", "uint8", "bool"}:
            vectors.append(
                VectorFeatureSpec(
                    key,
                    int(shape[0]) if shape else 1,
                    names=name_tuple,
                    dtype=str(dtype),
                    shape=shape,
                )
            )
        else:
            raise ConversionError(f"unsupported RoboGene dtype {dtype!r} for {key}")
    return tuple(vectors), tuple(cameras)


def _normalise_info(info: dict[str, Any], plan: DatasetConversionPlan) -> dict[str, Any]:
    result = json.loads(json.dumps(info))
    result["codebase_version"] = "v3.0"
    result.pop("total_chunks", None)
    result.pop("total_videos", None)
    result["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    result["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    result["fps"] = int(plan.fps)
    result.setdefault("data_files_size_in_mb", 100)
    result.setdefault("video_files_size_in_mb", 500)
    result["total_episodes"] = len(plan.episodes)
    result["total_frames"] = plan.num_frames
    result["total_tasks"] = len(dict.fromkeys(ep.instruction for ep in plan.episodes))
    result["splits"] = {"train": f"0:{len(plan.episodes)}"}
    for key, feature in result["features"].items():
        if feature.get("dtype") == "video":
            feature["names"] = ["height", "width", "channel"]
        elif feature.get("dtype") != "image":
            feature.setdefault("names", None)
    return result


def _task_rows(task_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    info = _read_json(task_root / "meta" / "info.json")
    if int(info.get("total_episodes", -1)) == 0:
        # The release has a small number of declared-empty task directories
        # that intentionally contain only info.json.  They contribute no
        # episodes and are recorded in the final provenance rather than being
        # treated as malformed non-empty datasets.
        return info, {}, {}
    task_rows = _read_jsonl(task_root / "meta" / "tasks.jsonl")
    episode_rows = _read_jsonl(task_root / "meta" / "episodes.jsonl")
    stats_rows = _read_jsonl(task_root / "meta" / "episodes_stats.jsonl")
    tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    episodes = {int(row["episode_index"]): row for row in episode_rows}
    stats = {int(row["episode_index"]): row for row in stats_rows}
    if set(episodes) != set(stats):
        raise ConversionError(f"episode/stats index mismatch in {task_root}")
    expected = int(info.get("total_episodes", -1))
    if expected != len(episodes):
        raise ConversionError(f"metadata episode count mismatch in {task_root}: {expected} != {len(episodes)}")
    for index, row in episodes.items():
        tasks_for_episode = row.get("tasks")
        if not isinstance(tasks_for_episode, list) or len(tasks_for_episode) != 1:
            raise ConversionError(f"RoboGene episode {task_root}:{index} does not have exactly one task")
        task = str(tasks_for_episode[0])
        if task not in tasks.values():
            raise ConversionError(f"episode task is absent from tasks.jsonl in {task_root}:{index}")
        if int(row.get("length", -1)) <= 0:
            raise ConversionError(f"invalid RoboGene episode length in {task_root}:{index}")
    return info, episodes, stats


def _inspect_payload_samples(
    sources: tuple[RobogeneEpisodeSource, ...],
    cameras: tuple[CameraFeatureSpec, ...],
    expected_fps: float,
) -> tuple[dict[str, Any], ...]:
    """Read only generated columns and video headers for first/middle/last."""

    import pyarrow.parquet as pq
    from convert_core.lerobot_writer import _video_frame_count

    if not sources:
        return ()
    selected: list[RobogeneEpisodeSource] = []
    for source in (sources[0], sources[len(sources) // 2], sources[-1]):
        if source.source_id not in {item.source_id for item in selected}:
            selected.append(source)
    evidence: list[dict[str, Any]] = []
    for source in selected:
        table = pq.read_table(
            source.data_path,
            columns=["episode_index", "frame_index", "index", "task_index", "timestamp"],
        )
        if table.num_rows != source.length:
            raise ConversionError(f"payload row count changed for {source.data_path}")
        values = {name: table[name].to_pylist() for name in table.column_names}
        if int(values["frame_index"][0]) != 0 or int(values["frame_index"][-1]) != source.length - 1:
            raise ConversionError(f"frame_index is not contiguous in {source.data_path}")
        videos: dict[str, Any] = {}
        camera_by_key = {camera.feature_key: camera for camera in cameras}
        for key, path in source.video_paths:
            frames, height, width, actual_fps, codec, pix_fmt = _video_frame_count(path)
            videos[key] = {
                "frames": frames,
                "height": height,
                "width": width,
                "fps": actual_fps,
                "codec": codec,
                "pixel_format": pix_fmt,
            }
            camera = camera_by_key[key]
            if frames != source.length or (height, width) != (camera.height, camera.width) or actual_fps is None or abs(actual_fps - expected_fps) > 1e-6:
                raise ConversionError(f"video header mismatch for {path}")
        evidence.append({"source_id": source.source_id, "rows": source.length, "videos": videos})
    return tuple(evidence)


def inspect_robogene(
    raw_root: Path,
    *,
    limit_tasks: int | None = None,
    limit_episodes: int | None = None,
    limit_shards: int | None = None,
    task_names: set[str] | None = None,
) -> RobogeneCatalog:
    """Inspect all selected metadata and freeze a deterministic catalog."""

    if not raw_root.is_dir():
        raise ConversionError(f"RoboGene raw root is missing: {raw_root}")
    if task_names is not None:
        for task_name in task_names:
            task_path = Path(task_name)
            if not task_name or task_path.is_absolute() or task_path.name != task_name or ".." in task_path.parts:
                raise ConversionError(f"invalid exact RoboGene task name: {task_name!r}")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    empty_tasks: dict[str, list[str]] = {}
    task_count = 0
    selected_episode_count = 0
    selected_shard_count = 0
    exhausted_shards = False
    for split_root in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        split = split_root.name
        if task_names is None:
            task_roots = sorted(path for path in split_root.iterdir() if path.is_dir())
        else:
            # Smoke and benchmark selections must not enumerate the complete
            # TB-scale task directory.  Exact task names can be resolved
            # directly under each declared split.
            task_roots = [
                split_root / task_name
                for task_name in sorted(task_names)
                if (split_root / task_name).is_dir()
            ]
        for task_root in task_roots:
            if exhausted_shards:
                break
            if limit_tasks is not None and task_count >= limit_tasks:
                break
            task_count += 1
            info, episode_rows, stats_rows = _task_rows(task_root)
            if not episode_rows:
                empty_tasks.setdefault(split, []).append(task_root.name)
                continue
            vectors, cameras = _feature_specs(info)
            try:
                task_fps = float(info.get("fps", 0))
            except (TypeError, ValueError) as exc:
                raise ConversionError(f"invalid RoboGene FPS in {task_root}") from exc
            if not math.isfinite(task_fps) or task_fps <= 0:
                raise ConversionError(f"invalid RoboGene FPS in {task_root}: {task_fps!r}")
            if not task_fps.is_integer():
                raise ConversionError(f"RoboGene fractional FPS is not representable by the shared LeRobot plan: {task_fps!r} in {task_root}")
            inventory, task_files = _file_inventory(task_root)
            data_paths = sorted(
                path for path in task_files
                if path.relative_to(task_root).parts[0] == "data" and path.suffix == ".parquet"
            )
            data_by_episode: dict[int, Path] = {}
            for path in data_paths:
                episode_index = _episode_index(path, _EPISODE_RE, "data")
                if episode_index in data_by_episode:
                    raise ConversionError(f"duplicate data episode {episode_index} in {task_root}")
                data_by_episode[episode_index] = path
            video_keys = tuple(camera.feature_key for camera in cameras)
            videos_by_key: dict[str, dict[int, Path]] = {key: {} for key in video_keys}
            for path in sorted(
                path for path in task_files
                if path.relative_to(task_root).parts[0] == "videos" and path.suffix == ".mp4"
            ):
                relative_video = path.relative_to(task_root / "videos")
                # v2.1 releases have used both ``videos/<key>/chunk-*`` and
                # ``videos/chunk-*/<key>``.  The feature name is the
                # non-chunk component immediately above the episode file.
                if len(relative_video.parts) < 2:
                    raise ConversionError(f"invalid RoboGene video layout: {path}")
                key = (
                    relative_video.parts[1]
                    if relative_video.parts[0].startswith("chunk-")
                    else relative_video.parts[0]
                )
                if key not in videos_by_key:
                    raise ConversionError(f"video stream {key!r} is not declared in {task_root / 'meta' / 'info.json'}")
                episode_index = _episode_index(path, _VIDEO_RE, "video")
                if episode_index in videos_by_key[key]:
                    raise ConversionError(f"duplicate video episode {episode_index} for {key} in {task_root}")
                videos_by_key[key][episode_index] = path
            if set(data_by_episode) != set(episode_rows):
                raise ConversionError(f"data episode inventory mismatch in {task_root}")
            if any(set(paths) != set(episode_rows) for paths in videos_by_key.values()):
                raise ConversionError(f"video episode inventory mismatch in {task_root}")
            # A source shard is a task-local ``data/chunk-*`` directory.  In
            # this release most tasks have one such shard, but treating it as
            # a real selection unit keeps --limit-shards meaningful if a
            # later release packs more episodes per task.
            shard_roots = sorted({path.parent for path in data_paths})
            if limit_shards is not None:
                remaining = limit_shards - selected_shard_count
                if remaining <= 0:
                    exhausted_shards = True
                    break
                selected_roots = set(shard_roots[:remaining])
                selected_shard_count += len(selected_roots)
                if len(selected_roots) < len(shard_roots):
                    exhausted_shards = True
            else:
                selected_roots = set(shard_roots)
            selected_data_by_episode = {
                index: path
                for index, path in data_by_episode.items()
                if path.parent in selected_roots
            }
            selected_indices = set(selected_data_by_episode)
            if not selected_indices:
                continue
            for episode_index in sorted(selected_indices):
                if limit_episodes is not None and selected_episode_count >= limit_episodes:
                    break
                episode = episode_rows[episode_index]
                instruction = str(episode["tasks"][0])
                data_path = selected_data_by_episode[episode_index]
                schema_payload = {
                    "robot_type": info.get("robot_type"),
                    "fps": info.get("fps"),
                    "features": info.get("features"),
                    "parquet_schema": _schema_text(data_path),
                }
                schema_fingerprint = hashlib.sha256(
                    json.dumps(schema_payload, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                group = groups.setdefault(
                    (split, schema_fingerprint),
                    {"info": info, "vectors": vectors, "cameras": cameras, "sources": [], "files": [], "tasks": [], "sample_candidates": [], "inventory_tasks": set()},
                )
                if group["vectors"] != vectors or group["cameras"] != cameras:
                    raise ConversionError(f"schema fingerprint collision in {task_root}")
                task_key = f"{split}/{task_root.name}"
                if task_key not in group["inventory_tasks"]:
                    group["files"].extend({"path": f"{split}/{task_root.name}/{row['path']}", "size": row["size"], "mtime_ns": row["mtime_ns"]} for row in inventory)
                    group["tasks"].append(task_root.name)
                    group["inventory_tasks"].add(task_key)
                video_paths = tuple((key, videos_by_key[key][episode_index]) for key in video_keys)
                group["sources"].append(
                    RobogeneEpisodeSource(
                        source_id=f"{split}/{task_root.name}/episode-{episode_index:06d}",
                        task_root=task_root,
                        split=split,
                        task_name=task_root.name,
                        source_episode_index=episode_index,
                        instruction=instruction,
                        stats=stats_rows[episode_index],
                        length=int(episode["length"]),
                        data_path=data_path,
                        video_paths=video_paths,
                        data_bytes=data_path.stat().st_size,
                        video_bytes=sum(path.stat().st_size for _, path in video_paths),
                    )
                )
                group["sample_candidates"].append(group["sources"][-1])
                selected_episode_count += 1
            # A limit is a smoke/benchmark selection, not a full preflight.
            if limit_episodes is not None and selected_episode_count >= limit_episodes:
                break
            if exhausted_shards:
                break
        if limit_tasks is not None and task_count >= limit_tasks:
            break
        if limit_episodes is not None and selected_episode_count >= limit_episodes:
            break
    partitions: list[RobogenePartition] = []
    mapping_table: list[dict[str, Any]] = []
    variants_per_split: dict[str, int] = {}
    for split, _schema in groups:
        variants_per_split[split] = variants_per_split.get(split, 0) + 1
    for partition_number, ((split, schema_fingerprint), group) in enumerate(sorted(groups.items())):
        sources = tuple(group["sources"])
        if not sources:
            continue
        tasks = list(dict.fromkeys(source.instruction for source in sources))
        episodes = tuple(
            EpisodePlan(
                source.source_id,
                source.source_id,
                source.instruction,
                source.length,
                {
                    "robogene_source_id": source.source_id,
                    "checkpoint_unit": source.task_name,
                    "source_split": split,
                    "source_task": source.task_name,
                },
            )
            for source in sources
        )
        partition_mapping = []
        for key, feature in group["info"]["features"].items():
            dtype = feature["dtype"]
            partition_mapping.append(
                {
                    "source_field": key,
                    "shape_dtype": f"{feature['shape']}/{dtype}",
                    "semantics": "RoboGene v2.1 native LeRobot field; preserved",
                    "lerobot_field": key,
                    "conversion": (
                        "copy" if key not in {"episode_index", "index", "task_index"}
                        else "rebase to coordinator-assigned global index"
                    ),
                    "evidence": "meta/info.json and Parquet footer",
                    "lossy": False,
                }
            )
        mapping_table.extend(partition_mapping)
        existing_mapping_fields = {row["source_field"] for row in mapping_table}
        for key, shape_dtype, semantics, conversion in (
            ("timestamp", "scalar/float", "source frame timestamp; preserved", "copy"),
            ("frame_index", "scalar/int64", "episode-local frame ordinal; preserved", "copy"),
            ("episode_index", "scalar/int64", "generated episode identifier", "rebase to coordinator-assigned global index"),
            ("index", "scalar/int64", "generated global frame identifier", "rebase to coordinator-assigned global index"),
            ("task_index", "scalar/int64", "generated task identifier", "map instruction to coordinator task index"),
        ):
            if key in existing_mapping_fields:
                continue
            mapping_table.append(
                {
                    "source_field": key,
                    "shape_dtype": shape_dtype,
                    "semantics": semantics,
                    "lerobot_field": key,
                    "conversion": conversion,
                    "evidence": "Parquet schema and generated-index validation",
                    "lossy": False,
                }
            )
            existing_mapping_fields.add(key)
        variant_suffix = "" if variants_per_split[split] == 1 else f"--schema-{schema_fingerprint[:12]}"
        partition_name = f"{split}{variant_suffix}"
        plan = DatasetConversionPlan(
            dataset_uid=f"robogene-{partition_name}",
            output_path=Path("robogene") / partition_name,
            fps=int(group["info"].get("fps", 0)),
            measured_fps=float(group["info"].get("fps", 0)),
            robot_type=str(group["info"].get("robot_type", "")),
            vector_features=group["vectors"],
            camera_features=group["cameras"],
            episodes=episodes,
            extra={
                "source_dataset": "RoboGene",
                "source_root": str(raw_root),
                "source_splits": [split],
                "source_files": group["files"],
                "field_mapping": partition_mapping,
                "video_encoding": {"source_codec": "h264", "target_codec": "h264", "target_pix_fmt": "yuv420p", "mode": "copy"},
                "partition_rules": ["top-level split", "Parquet schema fingerprint"],
                "schema_fingerprint": schema_fingerprint,
            },
        )
        samples = _inspect_payload_samples(sources, group["cameras"], float(group["info"].get("fps", 0)))
        partitions.append(RobogenePartition(partition_name, split, schema_fingerprint, plan, group["info"], sources, tuple(group["files"]), tuple(empty_tasks.get(split, ())), samples))
    fingerprint_payload = {
        "source_root": str(raw_root),
        "partitions": [
            {
                "name": p.name,
                "split": p.split,
                "schema": p.schema_fingerprint,
                "files": list(p.source_files),
                "episodes": len(p.episodes),
                "frames": p.plan.num_frames,
                "fps": p.plan.fps,
                "robot_type": p.plan.robot_type,
                "features": p.plan.feature_schema(),
                "tasks": [source.instruction for source in p.episodes],
                "empty_tasks": list(p.empty_tasks),
            }
            for p in partitions
        ],
        "field_mapping": list(mapping_table),
    }
    return RobogeneCatalog(tuple(partitions), fingerprint_payload, tuple(mapping_table))
