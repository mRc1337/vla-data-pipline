"""Archive-aware, task-scoped reader for AgiBot World 2026.

The release stores independent LeRobot v2.1 datasets either in one ``.tar.gz``
per real-world shard or in split ``meta/data/videos.tar.gz.NNN`` streams for
simulation variants.  The catalog deliberately records directory names and
file stat tuples only.  Archive traversal is deferred to the selected task's
preflight and the resulting immutable plan is sufficient for resume.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint
from convert_core.episode_spec import CameraFeatureSpec, DatasetConversionPlan, EpisodePlan, VectorFeatureSpec
from convert_core.errors import ConversionError


REQUIRED_META = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
OPTIONAL_META = ("annotations.json",)
_EPISODE_RE = re.compile(r"episode_(\d+)\.(?:parquet|mp4)$")


@dataclass(frozen=True)
class ArchiveParts:
    kind: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class SourceShard:
    shard_id: str
    metadata: ArchiveParts
    data: ArchiveParts
    videos: ArchiveParts
    member_prefix: str
    source_files: tuple[dict[str, Any], ...]
    data_from_sibling_lite: bool = False
    sibling_metadata: ArchiveParts | None = None


@dataclass(frozen=True)
class CatalogTask:
    task_key: str
    task_index: int
    partition_name: str
    family: str
    shards: tuple[SourceShard, ...]
    source_bytes: int
    unavailable_files: tuple[str, ...]


@dataclass(frozen=True)
class AgibotCatalog:
    source_root: Path
    tasks: tuple[CatalogTask, ...]


@dataclass(frozen=True)
class EpisodeSource:
    source_id: str
    shard_id: str
    source_episode_index: int
    instruction: str
    length: int
    stats: dict[str, Any]
    data_member: str
    video_members: tuple[tuple[str, str], ...]
    data_bytes: int
    video_bytes: int
    member_sizes: dict[str, int]
    global_episode_index: int
    global_frame_start: int
    partition_task_index: int


class _JoinedReader(io.RawIOBase):
    """Forward-only reader over byte-split archive parts."""

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        self._paths = tuple(paths)
        self._index = -1
        self._stream: BinaryIO | None = None
        self._open_next()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def _open_next(self) -> bool:
        if self._stream is not None:
            self._stream.close()
        self._index += 1
        if self._index >= len(self._paths):
            self._stream = None
            return False
        self._stream = self._paths[self._index].open("rb", buffering=8 * 1024 * 1024)
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view) and self._stream is not None:
            count = self._stream.readinto(view[total:])
            if count:
                total += count
            elif not self._open_next():
                break
        return total

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def _normal_member_name(value: str) -> str:
    value = value.removeprefix("./")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConversionError(f"unsafe AgiBot archive member: {value!r}")
    return path.as_posix()


@contextmanager
def open_archive(parts: ArchiveParts) -> Iterator[tarfile.TarFile]:
    if not parts.paths:
        raise ConversionError(f"empty AgiBot {parts.kind} archive part list")
    joined = _JoinedReader(parts.paths)
    buffered = io.BufferedReader(joined, buffer_size=8 * 1024 * 1024)
    try:
        with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
            yield archive
    except (tarfile.TarError, OSError) as exc:
        raise ConversionError(f"cannot read AgiBot {parts.kind} archive {parts.paths}: {exc}") from exc
    finally:
        buffered.close()


def _stat_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _part_number(path: Path) -> int:
    try:
        return int(path.suffix.removeprefix("."))
    except ValueError as exc:
        raise ConversionError(f"invalid split archive suffix: {path}") from exc


def _split_parts(root: Path, stem: str) -> tuple[Path, ...]:
    paths = tuple(sorted(root.glob(f"{stem}.tar.gz.*"), key=_part_number))
    if paths and [_part_number(path) for path in paths] != list(range(len(paths))):
        raise ConversionError(f"non-contiguous AgiBot split archive parts under {root}: {stem}")
    return paths


def build_lightweight_catalog(source_root: Path) -> AgibotCatalog:
    """List only stable task directories and archive stat tuples."""

    if not source_root.is_dir():
        raise ConversionError(f"AgiBot source root is missing: {source_root}")
    pending: list[tuple[str, str, str, tuple[SourceShard, ...]]] = []
    for family in ("ImitationLearning", "RichInteraction"):
        family_root = source_root / family
        if not family_root.is_dir():
            continue
        for scene in sorted(path for path in family_root.iterdir() if path.is_dir()):
            for task_root in sorted(path for path in scene.iterdir() if path.is_dir() and path.name.startswith("task_")):
                archives = tuple(sorted(task_root.glob("*.tar.gz")))
                if not archives:
                    continue
                shards = []
                for archive in archives:
                    record = _stat_record(archive, source_root)
                    parts = ArchiveParts("combined", (archive,))
                    shards.append(SourceShard(archive.name.removesuffix(".tar.gz"), parts, parts, parts, "data/", (record,)))
                key = f"real/{family}/{scene.name}/{task_root.name}"
                pending.append((key, f"real-{family.lower()}", "real", tuple(shards)))

    simulation = source_root / "simulation"
    if simulation.is_dir():
        for scenario in sorted(path for path in simulation.iterdir() if path.is_dir()):
            for robot in sorted(path for path in scenario.iterdir() if path.is_dir()):
                for variant in sorted(path for path in robot.iterdir() if path.is_dir()):
                    meta = _split_parts(variant, "meta")
                    videos = _split_parts(variant, "videos")
                    data = _split_parts(variant, "data")
                    sibling = False
                    sibling_meta: tuple[Path, ...] = ()
                    if not data and variant.name == "lite_depth_patch":
                        data = _split_parts(variant.parent / "lite", "data")
                        sibling_meta = _split_parts(variant.parent / "lite", "meta")
                        sibling = True
                    if not meta and not data and not videos:
                        continue
                    paths = tuple(dict.fromkeys((*meta, *data, *videos, *sibling_meta)))
                    records = tuple(_stat_record(path, source_root) for path in paths)
                    shard = SourceShard(
                        f"{scenario.name}-{variant.name}",
                        ArchiveParts("metadata", meta),
                        ArchiveParts("data", data),
                        ArchiveParts("videos", videos),
                        "",
                        records,
                        sibling,
                        ArchiveParts("sibling_metadata", sibling_meta) if sibling_meta else None,
                    )
                    key = f"simulation/{scenario.name}/{robot.name}/{variant.name}"
                    partition = "simulation-lite-depth-patch" if variant.name == "lite_depth_patch" else "simulation-lite"
                    pending.append((key, partition, "simulation", (shard,)))

    tasks: list[CatalogTask] = []
    for task_index, (key, partition, family, shards) in enumerate(sorted(pending)):
        records = [record for shard in shards for record in shard.source_files]
        unavailable = tuple(str(record["path"]) for record in records if int(record["size"]) <= 0)
        tasks.append(CatalogTask(key, task_index, partition, family, shards, sum(int(item["size"]) for item in records), unavailable))
    if not tasks:
        raise ConversionError(f"AgiBot source contains no task archives: {source_root}")
    return AgibotCatalog(source_root, tuple(tasks))


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return _safe(vars(value))
    return value


def catalog_payload(catalog: AgibotCatalog, *, output_uid: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "reader_format": "agibot_world_lerobot_v21_archives",
        "source_root": str(catalog.source_root),
        "output_dataset_uid": output_uid,
        "selection": dict(selection),
        "tasks": [_safe(task) for task in catalog.tasks],
    }
    payload["catalog_fingerprint"] = canonical_fingerprint(payload)
    return payload


def catalog_from_payload(payload: Mapping[str, Any]) -> AgibotCatalog:
    if payload.get("schema_version") != 2 or not isinstance(payload.get("tasks"), list):
        raise ConversionError("unsupported AgiBot task catalog")
    try:
        tasks = []
        for raw in payload["tasks"]:
            shards = []
            for item in raw["shards"]:
                def parts(name: str) -> ArchiveParts:
                    value = item[name]
                    return ArchiveParts(str(value["kind"]), tuple(Path(str(path)) for path in value["paths"]))
                shards.append(SourceShard(
                    str(item["shard_id"]), parts("metadata"), parts("data"), parts("videos"),
                    str(item["member_prefix"]), tuple(dict(row) for row in item["source_files"]),
                    bool(item.get("data_from_sibling_lite", False)),
                    parts("sibling_metadata") if item.get("sibling_metadata") else None,
                ))
            tasks.append(CatalogTask(
                str(raw["task_key"]), int(raw["task_index"]), str(raw["partition_name"]),
                str(raw["family"]), tuple(shards), int(raw["source_bytes"]),
                tuple(str(path) for path in raw.get("unavailable_files", [])),
            ))
        return AgibotCatalog(Path(str(payload["source_root"])), tuple(tasks))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"invalid AgiBot task catalog: {exc}") from exc


def validate_source_records(task: CatalogTask, source_root: Path) -> None:
    validate_source_file_records(
        [record for shard in task.shards for record in shard.source_files],
        source_root,
    )


def validate_source_file_records(records: Sequence[Mapping[str, Any]], source_root: Path) -> None:
    approved = source_root.resolve(strict=False)
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ConversionError("invalid AgiBot source path in catalog")
        path = source_root / relative
        if not path.resolve(strict=False).is_relative_to(approved):
            raise ConversionError(f"AgiBot source path escapes raw root: {path}")
        try:
            stat = path.stat()
        except OSError as exc:
            raise ConversionError(f"AgiBot source file is unavailable: {path}") from exc
        if stat.st_size <= 0:
            raise ConversionError(f"AgiBot source archive is empty/incomplete: {path}")
        if stat.st_size != int(record["size"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ConversionError(f"AgiBot source archive changed since catalog: {path}")


def _json_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid AgiBot {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"AgiBot {description} must be a JSON object")
    return value


def _json_lines(payload: bytes, description: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"invalid AgiBot {description}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ConversionError(f"AgiBot {description}:{line_number} is not an object")
        rows.append(value)
    return rows


def extract_metadata(shard: SourceShard, cache_root: Path) -> dict[str, bytes]:
    """Extract only the contiguous legacy metadata block from one archive."""

    prefix = f"{shard.member_prefix}meta/"
    found: dict[str, bytes] = {}
    seen_meta = False
    with open_archive(shard.metadata) as archive:
        for member in archive:
            name = _normal_member_name(member.name)
            if name.startswith(prefix):
                seen_meta = True
                if not member.isfile():
                    continue
                basename = PurePosixPath(name).name
                if basename not in (*REQUIRED_META, *OPTIONAL_META):
                    continue
                if member.size > 256 * 1024 * 1024:
                    raise ConversionError(f"AgiBot metadata member is unexpectedly large: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ConversionError(f"cannot extract AgiBot metadata member: {name}")
                found[basename] = stream.read()
            elif seen_meta:
                break
    missing = sorted(set(REQUIRED_META) - set(found))
    if missing:
        raise ConversionError(f"AgiBot shard {shard.shard_id} is missing metadata: {missing}")
    cache_root.mkdir(parents=True, exist_ok=True)
    for name, payload in found.items():
        path = cache_root / name
        path.write_bytes(payload)
    atomic_write_json(
        cache_root / "inventory.json",
        {"files": [{"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in sorted(found.items())]},
    )
    return found


def _feature_specs(info: Mapping[str, Any]) -> tuple[tuple[VectorFeatureSpec, ...], tuple[CameraFeatureSpec, ...]]:
    features = info.get("features")
    if not isinstance(features, Mapping) or not features:
        raise ConversionError("AgiBot info.json has no feature schema")
    vectors: list[VectorFeatureSpec] = []
    cameras: list[CameraFeatureSpec] = []
    for key, raw in features.items():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("shape"), list):
            raise ConversionError(f"invalid AgiBot feature definition: {key}")
        shape = tuple(int(value) for value in raw["shape"])
        dtype = str(raw.get("dtype"))
        names = raw.get("names")
        if dtype == "video":
            if len(shape) != 3 or min(shape[:2]) <= 0:
                raise ConversionError(f"invalid AgiBot video feature shape: {key}: {shape}")
            if shape[2] not in {1, 3}:
                raise ConversionError(f"unsupported AgiBot video channel count: {key}: {shape}")
            cameras.append(CameraFeatureSpec(str(key), shape[0], shape[1]))
        elif dtype in {"float32", "float64", "int64", "int32", "uint64", "uint32", "uint8", "bool", "string"}:
            vectors.append(VectorFeatureSpec(str(key), math.prod(shape) if shape else 1, tuple(str(x) for x in names) if isinstance(names, list) else None, dtype, shape))
        else:
            raise ConversionError(f"unsupported AgiBot feature dtype {dtype!r}: {key}")
    return tuple(vectors), tuple(cameras)


def _format_path(template: str, *, episode_index: int, video_key: str | None = None) -> str:
    values = {"episode_chunk": episode_index // 1000, "chunk_index": episode_index // 1000, "episode_index": episode_index, "video_key": video_key}
    try:
        return _normal_member_name(template.format(**values))
    except (KeyError, ValueError) as exc:
        raise ConversionError(f"unsupported AgiBot legacy path template {template!r}: {exc}") from exc


def _mapping(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, raw in info["features"].items():
        generated = key in {"episode_index", "index", "task_index"}
        semantics = "source-generated identifier" if generated else "official AgiBot v2.1 field; native unit preserved (unit is not declared where absent)"
        rows.append({
            "source_field": key,
            "shape_dtype": f"{raw.get('shape')}/{raw.get('dtype')}",
            "semantics": semantics,
            "lerobot_field": key,
            "conversion": "deterministic global reindex" if generated else ("container copy; no decode or re-encode" if raw.get("dtype") == "video" else "identity"),
            "evidence": "official README + source meta/info.json field_descriptions + Parquet/video probe",
            "lossy": False,
        })
    return rows


def _metadata_rows(metadata: Mapping[str, bytes], shard: SourceShard) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    info = _json_object(metadata["info.json"], f"{shard.shard_id}/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ConversionError(f"AgiBot shard {shard.shard_id} is not LeRobot v2.1")
    tasks = _json_lines(metadata["tasks.jsonl"], f"{shard.shard_id}/tasks.jsonl")
    episodes = {int(row["episode_index"]): row for row in _json_lines(metadata["episodes.jsonl"], f"{shard.shard_id}/episodes.jsonl")}
    stats = {int(row["episode_index"]): row for row in _json_lines(metadata["episodes_stats.jsonl"], f"{shard.shard_id}/episodes_stats.jsonl")}
    if set(episodes) != set(stats) or len(episodes) != int(info.get("total_episodes", -1)):
        raise ConversionError(f"AgiBot episode/stat totals disagree in shard {shard.shard_id}")
    task_names = {str(row["task"]) for row in tasks}
    if len(task_names) != 1:
        raise ConversionError(f"AgiBot task shard {shard.shard_id} has {len(task_names)} task labels; refusing to guess a boundary")
    for index, row in episodes.items():
        if row.get("tasks") != [next(iter(task_names))] or int(row.get("length", -1)) <= 0:
            raise ConversionError(f"invalid AgiBot episode metadata {shard.shard_id}:{index}")
    fps = float(info.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0 or not fps.is_integer():
        raise ConversionError(f"unsupported AgiBot FPS in {shard.shard_id}: {fps!r}")
    _feature_specs(info)
    return info, tasks, episodes, stats


def extract_members(
    parts: ArchiveParts,
    wanted: Mapping[str, Path | None],
    *,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, int]:
    """Extract exact regular files in one forward pass and reject omissions."""

    remaining = {_normal_member_name(name): path for name, path in wanted.items()}
    sizes: dict[str, int] = {}
    if not remaining:
        return sizes
    with open_archive(parts) as archive:
        for member in archive:
            name = _normal_member_name(member.name)
            if name not in remaining:
                continue
            destination = remaining[name]
            if not member.isfile() or member.issym() or member.islnk():
                raise ConversionError(f"AgiBot payload member is not a regular file: {name}")
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ConversionError(f"cannot extract AgiBot payload member: {name}")
                with destination.open("wb") as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                        if progress_check is not None:
                            progress_check()
                if destination.stat().st_size != member.size:
                    raise ConversionError(f"short AgiBot payload extraction: {name}")
            sizes[name] = member.size
            remaining.pop(name)
            if not remaining:
                break
    if remaining:
        raise ConversionError(f"AgiBot archive is missing payload members: {sorted(remaining)[:8]}")
    return sizes


def _video_probe(path: Path, *, expected_frames: int, expected_shape: tuple[int, int], expected_fps: float) -> dict[str, Any]:
    import av
    with av.open(str(path), "r") as container:
        if not container.streams.video:
            raise ConversionError(f"AgiBot video has no stream: {path}")
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        rate = stream.average_rate or stream.base_rate
        if frames <= 0:
            raise ConversionError(f"AgiBot video does not declare frame count; refusing full decode: {path}")
        actual = (int(stream.height), int(stream.width))
        actual_fps = float(rate) if rate is not None else None
        if frames != expected_frames or actual != expected_shape or actual_fps is None or abs(actual_fps - expected_fps) > 1e-6:
            raise ConversionError(f"AgiBot video header mismatch: {path}: frames={frames}, shape={actual}, fps={actual_fps}")
        hashes = []
        for frame_index in sorted({0, frames // 2, frames - 1}):
            with av.open(str(path), "r") as sample_container:
                sample_stream = sample_container.streams.video[0]
                target_seconds = frame_index / expected_fps
                target_pts = int(target_seconds / float(sample_stream.time_base))
                sample_container.seek(max(0, target_pts), stream=sample_stream, backward=True)
                chosen = None
                for frame in sample_container.decode(sample_stream):
                    when = float(frame.time) if frame.time is not None else None
                    chosen = frame
                    if when is not None and when + 0.5 / expected_fps >= target_seconds:
                        break
                if chosen is None:
                    raise ConversionError(f"cannot decode sampled AgiBot frame {frame_index}: {path}")
                array = chosen.to_ndarray()
                hashes.append({"frame_index": frame_index, "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        return {"frames": frames, "height": actual[0], "width": actual[1], "fps": actual_fps, "codec": stream.codec.canonical_name, "pix_fmt": stream.codec_context.format.name if stream.codec_context.format else None, "sample_frames": hashes}


def _normalized_video_info(feature: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    is_depth = bool((feature.get("video_info") or {}).get("video.is_depth_map", False))
    normalized = {
        "video.height": int(evidence["height"]),
        "video.width": int(evidence["width"]),
        "video.fps": float(evidence["fps"]),
        "video.channels": 1 if str(evidence["pix_fmt"]).startswith("gray") else 3,
        "has_audio": False,
        "is_depth_map": is_depth,
    }
    if is_depth and evidence["codec"] == "png":
        # LeRobot 0.6's DatasetReader reconstructs an *encoder* config while
        # opening existing videos, and its encoder allowlist omits FFmpeg's
        # lossless PNG codec. Keep the true stream identity in explicit source
        # keys; PyAV still discovers it from the copied MP4 during decoding.
        normalized["source_video.codec"] = evidence["codec"]
        normalized["source_video.pix_fmt"] = evidence["pix_fmt"]
    else:
        normalized["video.codec"] = evidence["codec"]
        normalized["video.pix_fmt"] = evidence["pix_fmt"]
    return normalized


def _parquet_probe(
    path: Path,
    *,
    expected_length: int,
    expected_fps: float,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_length:
        raise ConversionError(f"AgiBot Parquet rows changed: {path}")
    expected = set(info["features"]) - {key for key, value in info["features"].items() if value.get("dtype") == "video"}
    actual = set(parquet.schema_arrow.names)
    if expected != actual:
        raise ConversionError(f"AgiBot Parquet schema fields differ: expected={sorted(expected)}, actual={sorted(actual)}")
    table = parquet.read()
    frame_index = [int(value) for value in table["frame_index"].to_pylist()]
    if frame_index != list(range(expected_length)):
        raise ConversionError(f"AgiBot frame_index is not contiguous: {path}")
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    expected_timestamps = np.arange(expected_length, dtype=np.float64) / expected_fps
    if (
        timestamps.shape != (expected_length,)
        or not np.isfinite(timestamps).all()
        or not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-4)
    ):
        raise ConversionError(f"AgiBot timestamps do not match frame_index/FPS: {path}")
    samples = []
    for index in sorted({0, expected_length // 2, expected_length - 1}):
        row = {name: table[name][index].as_py() for name in table.column_names}
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
        samples.append({"frame_index": index, "sha256": hashlib.sha256(encoded).hexdigest()})
    schema_text = parquet.schema_arrow.to_string(show_field_metadata=True)
    return {
        "rows": expected_length,
        "schema": schema_text,
        "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
        "sample_rows": samples,
        "timestamp_range": [float(timestamps[0]), float(timestamps[-1])],
        "timestamp_step_seconds": 1.0 / expected_fps,
    }


def _archive_groups(shard: SourceShard) -> tuple[ArchiveParts, ...]:
    groups: list[ArchiveParts] = []
    for parts in (shard.data, shard.videos):
        if parts.paths and tuple(path.resolve(strict=False) for path in parts.paths) not in {tuple(path.resolve(strict=False) for path in item.paths) for item in groups}:
            groups.append(parts)
    return tuple(groups)


def preflight_task(
    task: CatalogTask,
    *,
    source_root: Path,
    task_cache_root: Path,
    output_uid: str,
    max_shards: int | None,
    max_episodes: int | None,
    episode_start: int,
    frame_start: int,
    unit_start: int,
    partition_task_index: int,
    encoder_threads: int,
    workers: int,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build an immutable task plan from metadata and bounded payload samples."""

    shards = task.shards[:max_shards] if max_shards is not None else task.shards
    if not shards:
        raise ConversionError(f"AgiBot task has no selected shards: {task.task_key}")
    validate_source_file_records(
        [record for shard in shards for record in shard.source_files],
        source_root,
    )
    raw_episodes: list[tuple[SourceShard, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    schema_payload: dict[str, Any] | None = None
    instruction: str | None = None
    mapping: list[dict[str, Any]] | None = None
    metadata_records: list[dict[str, Any]] = []
    for shard in shards:
        metadata_dir = task_cache_root / "source_metadata" / shard.shard_id
        metadata = extract_metadata(shard, metadata_dir)
        info, _tasks, episodes, stats = _metadata_rows(metadata, shard)
        if shard.data_from_sibling_lite:
            if shard.sibling_metadata is None or not shard.sibling_metadata.paths:
                raise ConversionError(f"AgiBot depth patch has no sibling lite metadata: {shard.shard_id}")
            sibling_shard = SourceShard(
                f"{shard.shard_id}--sibling-lite",
                shard.sibling_metadata,
                shard.data,
                ArchiveParts("videos", ()),
                "",
                shard.source_files,
            )
            sibling_dir = task_cache_root / "source_metadata" / sibling_shard.shard_id
            sibling_metadata = extract_metadata(sibling_shard, sibling_dir)
            sibling_info, sibling_tasks, sibling_episodes, _sibling_stats = _metadata_rows(
                sibling_metadata, sibling_shard
            )
            if _tasks != sibling_tasks or episodes != sibling_episodes:
                raise ConversionError(
                    f"AgiBot depth patch task/episode metadata differs from sibling lite: {shard.shard_id}"
                )
            nonvideo = lambda value: {
                key: feature
                for key, feature in value["features"].items()
                if feature.get("dtype") != "video"
            }
            if (
                info.get("robot_type") != sibling_info.get("robot_type")
                or info.get("fps") != sibling_info.get("fps")
                or nonvideo(info) != nonvideo(sibling_info)
            ):
                raise ConversionError(
                    f"AgiBot depth patch numeric schema differs from sibling lite: {shard.shard_id}"
                )
            sibling_inventory = json.loads((sibling_dir / "inventory.json").read_text(encoding="utf-8"))
            metadata_records.append(
                {
                    "shard_id": sibling_shard.shard_id,
                    "cache_path": str(sibling_dir),
                    "files": sibling_inventory["files"],
                    "role": "sibling-lite-data-metadata",
                }
            )
        current_schema = {"robot_type": info.get("robot_type"), "fps": info.get("fps"), "features": info.get("features")}
        if schema_payload is None:
            schema_payload = current_schema
            mapping = _mapping(info)
        elif current_schema != schema_payload:
            raise ConversionError(f"AgiBot schema changed within task {task.task_key} at shard {shard.shard_id}")
        shard_instruction = str(next(iter({row["tasks"][0] for row in episodes.values()})))
        if instruction is None:
            instruction = shard_instruction
        elif instruction != shard_instruction:
            raise ConversionError(f"AgiBot instruction changed within task {task.task_key}")
        for index in sorted(episodes):
            raw_episodes.append((shard, info, episodes[index], stats[index]))
        inventory = json.loads((metadata_dir / "inventory.json").read_text(encoding="utf-8"))
        metadata_records.append({"shard_id": shard.shard_id, "cache_path": str(metadata_dir), "files": inventory["files"], "data_from_sibling_lite": shard.data_from_sibling_lite})
    if max_episodes is not None:
        raw_episodes = raw_episodes[:max_episodes]
    if not raw_episodes or schema_payload is None or instruction is None or mapping is None:
        raise ConversionError(f"AgiBot task contains no selected episodes: {task.task_key}")
    info = raw_episodes[0][1]
    cameras = tuple(key for key, value in info["features"].items() if value.get("dtype") == "video")
    selected: list[dict[str, Any]] = []
    frame_cursor = frame_start
    for ordinal, (shard, shard_info, episode, stats) in enumerate(raw_episodes):
        index = int(episode["episode_index"])
        chunk = index // int(shard_info.get("chunks_size", 1000))
        data_member = shard.member_prefix + _format_path(str(shard_info["data_path"]), episode_index=index)
        video_members = tuple((key, shard.member_prefix + _format_path(str(shard_info["video_path"]), episode_index=index, video_key=key)) for key in cameras)
        selected.append({
            "source_id": f"{task.task_key}/{shard.shard_id}/episode-{index:06d}", "shard_id": shard.shard_id,
            "source_episode_index": index, "instruction": str(episode["tasks"][0]), "length": int(episode["length"]),
            "stats": stats, "data_member": data_member, "video_members": [list(item) for item in video_members],
            "data_bytes": 0, "video_bytes": 0, "member_sizes": {}, "global_episode_index": episode_start + ordinal,
            "global_frame_start": frame_cursor, "partition_task_index": partition_task_index, "source_chunk": chunk,
        })
        frame_cursor += int(episode["length"])

    sample_positions = sorted({0, len(selected) // 2, len(selected) - 1})
    sample_ids = {selected[index]["source_id"] for index in sample_positions}
    sample_root = task_cache_root / "preflight_samples"
    if sample_root.exists():
        import shutil
        shutil.rmtree(sample_root)
    for shard in shards:
        shard_rows = [row for row in selected if row["shard_id"] == shard.shard_id]
        wanted_by_group: dict[tuple[str, ...], dict[str, Path]] = {}
        for row in shard_rows:
            members = [row["data_member"], *(value[1] for value in row["video_members"])]
            for name in members:
                group = shard.data if name == row["data_member"] else shard.videos
                key = tuple(str(path) for path in group.paths)
                destination = sample_root / hashlib.sha256(name.encode()).hexdigest() if row["source_id"] in sample_ids else None
                wanted_by_group.setdefault(key, {})[name] = destination
        for group_paths, wanted in wanted_by_group.items():
            group = shard.data if tuple(str(path) for path in shard.data.paths) == group_paths else shard.videos
            sizes = extract_members(group, wanted, progress_check=progress_check)
            for row in shard_rows:
                if row["data_member"] in sizes:
                    row["data_bytes"] = sizes[row["data_member"]]
                row["video_bytes"] += sum(sizes.get(member, 0) for _key, member in row["video_members"])
                row["member_sizes"].update({name: size for name, size in sizes.items() if name == row["data_member"] or name in {member for _key, member in row["video_members"]}})

    samples: list[dict[str, Any]] = []
    video_info: dict[str, dict[str, Any]] = {}
    for row in selected:
        if row["source_id"] not in sample_ids:
            continue
        data_path = sample_root / hashlib.sha256(row["data_member"].encode()).hexdigest()
        parquet = _parquet_probe(
            data_path,
            expected_length=int(row["length"]),
            expected_fps=float(info["fps"]),
            info=info,
        )
        videos = {}
        for key, member in row["video_members"]:
            path = sample_root / hashlib.sha256(member.encode()).hexdigest()
            feature = info["features"][key]
            evidence = _video_probe(path, expected_frames=int(row["length"]), expected_shape=tuple(feature["shape"][:2]), expected_fps=float(info["fps"]))
            videos[key] = evidence
            normalized = _normalized_video_info(feature, evidence)
            previous = video_info.setdefault(key, normalized)
            if previous != normalized:
                raise ConversionError(f"AgiBot video schema changed within task {task.task_key}: {key}")
        samples.append({"source_id": row["source_id"], "parquet": parquet, "videos": videos})
    schema_hashes = {sample["parquet"]["schema_sha256"] for sample in samples}
    if len(schema_hashes) != 1:
        raise ConversionError(f"AgiBot Parquet schema changed within task {task.task_key}")
    import shutil
    shutil.rmtree(sample_root, ignore_errors=True)

    features = json.loads(json.dumps(info["features"]))
    for key, value in features.items():
        if value.get("dtype") == "video":
            value["info"] = video_info[key]
            value.pop("video_info", None)
        else:
            value.setdefault("names", None)
    schema_fingerprint = canonical_fingerprint({"robot_type": info.get("robot_type"), "fps": info.get("fps"), "features": features, "parquet_schema_sha256": samples[0]["parquet"]["schema_sha256"]})
    task_bytes = sum(int(row["data_bytes"]) + int(row["video_bytes"]) for row in selected)
    active_unit_bytes = sum(
        sorted(
            (
                2 * int(row["data_bytes"])
                + int(row["video_bytes"])
                + 64 * 1024 * 1024
                for row in selected
            ),
            reverse=True,
        )[:workers]
    )
    largest_unit_bytes = max(
        2 * int(row["data_bytes"])
        + int(row["video_bytes"])
        + 64 * 1024 * 1024
        for row in selected
    )
    largest_shard_materialization = max(
        sum(int(row["data_bytes"]) + int(row["video_bytes"]) for row in selected if row["shard_id"] == shard.shard_id)
        for shard in shards
    )
    task_plan: dict[str, Any] = {
        "schema_version": 1, "reader_format": "agibot_world_lerobot_v21_archives", "task_key": task.task_key,
        "catalog_task_index": task.task_index, "partition_name": task.partition_name, "partition_task_index": partition_task_index,
        "instruction": instruction, "output_dataset_uid": output_uid, "fps": int(info["fps"]), "robot_type": str(info.get("robot_type", "")),
        "schema_fingerprint": schema_fingerprint, "features": features, "mapping_table": mapping,
        "encoding": {"mode": "container copy", "decode": False, "reencode": False, "encoder_threads_per_worker": encoder_threads},
        "transaction": {"unit_commit": "copy-validate-marker-delete-local", "task_commit": "all-unit-markers", "dataset_commit": "success-last"},
        "source_root": str(source_root), "source_files": [record for shard in shards for record in shard.source_files],
        "shards": [_safe(shard) for shard in shards], "source_metadata": metadata_records, "episodes": selected,
        "episode_start": episode_start, "episode_end": episode_start + len(selected), "frame_start": frame_start, "frame_end": frame_cursor,
        "unit_start": unit_start, "unit_end": unit_start + len(selected), "estimated_output_bytes": task_bytes,
        "estimated_local_peak_bytes": active_unit_bytes + largest_shard_materialization,
        "estimated_local_peak_with_remote_materialization_bytes": active_unit_bytes,
        "largest_unit_local_peak_bytes": largest_unit_bytes,
        "preflight_samples": samples,
        "planning_workers": workers,
        "task_source_basis": "official task_* directory" if task.family == "real" else "official simulation scenario/robot/variant leaf",
    }
    task_plan["fingerprint"] = canonical_fingerprint(task_plan)
    return task_plan


def episode_source(row: Mapping[str, Any]) -> EpisodeSource:
    return EpisodeSource(
        str(row["source_id"]), str(row["shard_id"]), int(row["source_episode_index"]), str(row["instruction"]),
        int(row["length"]), dict(row["stats"]), str(row["data_member"]),
        tuple((str(key), str(path)) for key, path in row["video_members"]), int(row["data_bytes"]),
        int(row["video_bytes"]), {str(key): int(value) for key, value in row["member_sizes"].items()},
        int(row["global_episode_index"]), int(row["global_frame_start"]), int(row["partition_task_index"]),
    )


def shard_from_payload(item: Mapping[str, Any]) -> SourceShard:
    def parts(name: str) -> ArchiveParts:
        value = item[name]
        return ArchiveParts(str(value["kind"]), tuple(Path(str(path)) for path in value["paths"]))

    return SourceShard(
        str(item["shard_id"]), parts("metadata"), parts("data"), parts("videos"),
        str(item["member_prefix"]), tuple(dict(row) for row in item["source_files"]),
        bool(item.get("data_from_sibling_lite", False)),
        parts("sibling_metadata") if item.get("sibling_metadata") else None,
    )


def dataset_plan(task_plan: Mapping[str, Any], episodes: Sequence[EpisodeSource] | None = None) -> DatasetConversionPlan:
    selected = tuple(episodes if episodes is not None else (episode_source(row) for row in task_plan["episodes"]))
    vectors, cameras = _feature_specs({"features": task_plan["features"]})
    return DatasetConversionPlan(
        str(task_plan["output_dataset_uid"]), Path(str(task_plan["output_dataset_uid"])) / str(task_plan["partition_name"]),
        int(task_plan["fps"]), float(task_plan["fps"]), str(task_plan["robot_type"]), vectors, cameras,
        tuple(EpisodePlan(source.source_id, source.source_id, source.instruction, source.length, {"source_task": task_plan["task_key"], "source_episode_id": source.source_episode_index, "checkpoint_unit": source.shard_id, "manifest_provenance": {"source_shard": source.shard_id, "source_episode_index": source.source_episode_index}}) for source in selected),
        {"source_dataset": "agibot-world/AgiBotWorld2026", "field_mapping": list(task_plan["mapping_table"]), "video_encoding": {"mode": "copy", "reencoded": False}, "partition_rules": ["official source family/variant", "identical source feature schema"]},
    )
