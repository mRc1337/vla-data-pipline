"""Bounded local-to-final commit for parallel LeRobot work units.

Workers still use the verified generic LeRobot writer, but each unit is given a
globally preassigned chunk.  Only the small generated index columns are
rewritten; bulk files are copied into deterministic final names and validated.
Metadata is streamed and batched after all bulk units are verified, avoiding
LeRobot's full-dataset data/video concatenation pass.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from convert_core.checkpoint import atomic_write_json, read_json_object
from convert_core.episode_spec import DatasetConversionPlan
from convert_core.errors import ConversionError
from convert_core.parallel import (
    ParallelWorkUnit,
    read_verified_unit_marker,
    validate_verified_unit_marker,
    validate_work_units,
)
from convert_core.staging import sha256_file


DIRECT_COMMIT_SCHEMA_VERSION = 2
DEFAULT_METADATA_BATCH_BYTES = 128 * 1024 * 1024
DEFAULT_METADATA_BATCH_EPISODES = 1000
DEFAULT_COPY_BLOCK_BYTES = 64 * 1024 * 1024
DEFAULT_SAMPLE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DirectCommitPreparation:
    committed: tuple[ParallelWorkUnit, ...]
    uncommitted: tuple[ParallelWorkUnit, ...]
    discarded_corrupt: tuple[str, ...]


class DirectCommitUploader:
    """Bounded background local-to-OSS commit queue.

    Conversion workers only enqueue already validated local units.  Upload
    threads own remote writes and delete local bulk only through
    :func:`commit_verified_unit` after remote validation succeeds.
    """

    def __init__(
        self,
        *,
        partition_name: str,
        partition_root: Path,
        resume_root: Path,
        workers: int,
        max_queue_units: int,
        retain_local_after_commit: bool = False,
    ) -> None:
        if workers <= 0:
            raise ValueError("upload workers must be positive")
        if max_queue_units <= 0:
            raise ValueError("upload queue size must be positive")
        self.partition_name = partition_name
        self.partition_root = partition_root
        self.resume_root = resume_root
        self.workers = workers
        self.max_queue_units = max_queue_units
        self.retain_local_after_commit = retain_local_after_commit
        self._queue: queue.Queue[tuple[ParallelWorkUnit, bool] | None] = queue.Queue(
            maxsize=max_queue_units
        )
        self._lock = threading.Lock()
        self._failure: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._elapsed_seconds = 0.0
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"direct-commit-uploader-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    def _record_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = exc

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                unit, trusted = item
                with self._lock:
                    failed = self._failure is not None
                if failed:
                    continue
                try:
                    started = time.monotonic()
                    commit_verified_unit(
                        unit,
                        partition_name=self.partition_name,
                        partition_root=self.partition_root,
                        resume_root=self.resume_root,
                        trust_verified_marker=trusted,
                        retain_local_after_commit=self.retain_local_after_commit,
                    )
                except BaseException as exc:
                    self._record_failure(exc)
                else:
                    with self._lock:
                        self._completed += 1
                        self._elapsed_seconds += time.monotonic() - started
            finally:
                self._queue.task_done()

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise ConversionError(
                f"background upload failed: {type(failure).__name__}: {failure}"
            ) from failure

    def submit(
        self, unit: ParallelWorkUnit, *, trust_verified_marker: bool
    ) -> None:
        if self._closed:
            raise RuntimeError("upload queue is closed")
        while True:
            self.raise_if_failed()
            try:
                self._queue.put(
                    (unit, trust_verified_marker), timeout=0.5
                )
            except queue.Full:
                continue
            with self._lock:
                self._submitted += 1
            return

    def close(self, *, raise_on_failure: bool = True) -> dict[str, int]:
        if not self._closed:
            self._closed = True
            self._queue.join()
            for _thread in self._threads:
                self._queue.put(None)
            for thread in self._threads:
                thread.join()
        if raise_on_failure:
            self.raise_if_failed()
        with self._lock:
            return {
                "workers": self.workers,
                "max_queue_units": self.max_queue_units,
                "submitted_units": self._submitted,
                "completed_units": self._completed,
                "elapsed_seconds": self._elapsed_seconds,
            }


def committed_marker_path(
    resume_root: Path, partition_name: str, unit: ParallelWorkUnit
) -> Path:
    return resume_root / "committed" / partition_name / f"unit-{unit.index:06d}.json"


def unit_metadata_root(
    resume_root: Path, partition_name: str, unit: ParallelWorkUnit
) -> Path:
    return resume_root / "unit_metadata" / partition_name / f"unit-{unit.index:06d}"


def _marker_metadata_root(
    marker: Mapping[str, Any], *, resume_root: Path
) -> Path:
    value = marker.get("metadata_root")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ConversionError("invalid local metadata root in direct commit marker")
    root = (resume_root / value).resolve(strict=False)
    approved = resume_root.resolve(strict=False)
    if not root.is_relative_to(approved):
        raise ConversionError("direct commit metadata root escapes resume directory")
    return root


def _header(unit: ParallelWorkUnit, partition_name: str) -> dict[str, Any]:
    return {
        "direct_commit_schema_version": DIRECT_COMMIT_SCHEMA_VERSION,
        "partition": partition_name,
        "unit_index": unit.index,
        "unit_key": unit.key,
        "fingerprint": unit.fingerprint,
        "episode_start": unit.episode_start,
        "episode_end": unit.episode_end,
        "frame_start": unit.frame_start,
        "frame_end": unit.frame_end,
        "task_indices": list(unit.task_indices),
    }


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def _marker_relative_path(value: Any, description: str) -> Path:
    """Validate a path read from a resumable marker before joining it."""

    if not isinstance(value, str) or not value:
        raise ConversionError(f"invalid {description} path in direct commit marker")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConversionError(f"invalid {description} path in direct commit marker")
    return path


def _validate_record(root: Path, record: Mapping[str, Any], description: str) -> None:
    path = root / _marker_relative_path(record.get("relative_path"), description)
    if not path.is_file():
        raise ConversionError(f"missing {description}: {path}")
    if path.stat().st_size != int(record.get("size", -1)):
        raise ConversionError(f"changed {description} size: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise ConversionError(f"changed {description} digest: {path}")


def _sample_ranges(size: int, sample_bytes: int) -> tuple[tuple[int, int], ...]:
    if size < 0 or sample_bytes <= 0:
        raise ValueError("file size and sample size must be valid")
    length = min(size, sample_bytes)
    if length == 0:
        return ((0, 0),)
    return tuple(
        (offset, length)
        for offset in sorted({0, max(0, (size - length) // 2), max(0, size - length)})
    )


def _sample_evidence(path: Path, *, size: int, sample_bytes: int) -> list[dict[str, Any]]:
    import hashlib

    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for offset, length in _sample_ranges(size, sample_bytes):
            stream.seek(offset)
            payload = stream.read(length)
            if len(payload) != length:
                raise ConversionError(
                    f"short range read from {path}: {len(payload)} != {length} at {offset}"
                )
            rows.append(
                {
                    "offset": offset,
                    "length": length,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return rows


def _parquet_evidence(path: Path) -> dict[str, Any]:
    import hashlib
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        # ``ParquetSchema.__str__`` prepends an object repr containing a
        # process-specific memory address; exclude that first line.
        physical_schema = "\n".join(str(parquet.schema).splitlines()[1:])
    except Exception as exc:
        raise ConversionError(f"cannot reopen Parquet file {path}: {exc}") from exc
    sample_columns = [
        name
        for name in ("episode_index", "frame_index", "index", "task_index", "timestamp")
        if name in parquet.schema_arrow.names
    ]
    sample_rows: list[dict[str, Any]] = []
    if sample_columns and metadata.num_rows:
        try:
            sampled = parquet.read(columns=sample_columns, use_threads=False)
            row_indices = sorted({0, metadata.num_rows // 2, metadata.num_rows - 1})
            for row_index in row_indices:
                sample_rows.append(
                    {
                        name: sampled[name][row_index].as_py()
                        for name in sample_columns
                    }
                )
        except Exception as exc:
            raise ConversionError(f"cannot sample Parquet generated columns {path}: {exc}") from exc
    return {
        "kind": "parquet",
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "num_columns": metadata.num_columns,
        # Arrow extension registration changes ``schema_arrow.serialize()``
        # despite identical Parquet bytes.  The physical footer schema is
        # process-independent and is the format contract we need to verify.
        "physical_schema_sha256": hashlib.sha256(
            physical_schema.encode("utf-8")
        ).hexdigest(),
        "sample_columns": sample_columns,
        "sample_rows": sample_rows,
    }


def _video_evidence(path: Path) -> dict[str, Any]:
    try:
        import av

        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ConversionError(f"video has no video stream: {path}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            declared_frames = int(stream.frames or 0)
            exact_frames = declared_frames
            if exact_frames <= 0:
                exact_frames = sum(1 for _ in container.decode(stream))
            evidence = {
                "kind": "video",
                "declared_frames": declared_frames,
                "frame_count": exact_frames,
                "height": int(stream.height),
                "width": int(stream.width),
                "fps": float(rate) if rate is not None else None,
                "codec": stream.codec.canonical_name,
                "pixel_format": (
                    stream.codec_context.format.name
                    if stream.codec_context.format
                    else None
                ),
            }
    except Exception as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError(f"cannot reopen video file {path}: {exc}") from exc
    return evidence


def _format_evidence(path: Path) -> dict[str, Any]:
    if path.suffix == ".parquet":
        return _parquet_evidence(path)
    if path.suffix == ".mp4":
        return _video_evidence(path)
    raise ConversionError(f"unsupported direct-commit bulk file: {path}")


def _bulk_record(path: Path, *, relative_to: Path, sample_bytes: int) -> dict[str, Any]:
    record = _record(path, relative_to=relative_to)
    record["samples"] = _sample_evidence(
        path, size=int(record["size"]), sample_bytes=sample_bytes
    )
    record["format"] = _format_evidence(path)
    return record


def _validate_bulk_destination(
    path: Path,
    record: Mapping[str, Any],
    *,
    verify_remote_sha256: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise ConversionError(f"missing committed bulk file: {path}")
    size = path.stat().st_size
    if size != int(record.get("size", -1)):
        raise ConversionError(f"changed committed bulk file size: {path}")
    expected_samples = record.get("samples")
    if not isinstance(expected_samples, list) or not expected_samples:
        raise ConversionError(f"missing sample evidence for committed bulk file: {path}")
    actual_samples: list[dict[str, Any]] = []
    import hashlib

    with path.open("rb") as stream:
        for expected in expected_samples:
            if not isinstance(expected, Mapping):
                raise ConversionError(f"invalid sample evidence for {path}")
            offset = int(expected.get("offset", -1))
            length = int(expected.get("length", -1))
            if offset < 0 or length < 0 or offset + length > size:
                raise ConversionError(f"invalid sample range for {path}")
            stream.seek(offset)
            payload = stream.read(length)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != length or digest != expected.get("sha256"):
                raise ConversionError(f"changed committed bulk sample at {offset}: {path}")
            actual_samples.append(
                {"offset": offset, "length": length, "sha256": digest}
            )
    actual_format = _format_evidence(path)
    if actual_format != record.get("format"):
        raise ConversionError(
            f"changed committed bulk format evidence: {path}; "
            f"expected={record.get('format')!r}, actual={actual_format!r}"
        )
    evidence: dict[str, Any] = {
        "size_verified": size,
        "samples": actual_samples,
        "format": actual_format,
        "remote_sha256_verified": False,
    }
    if verify_remote_sha256:
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise ConversionError(f"changed committed bulk digest: {path}")
        evidence.update(remote_sha256_verified=True, remote_sha256=digest)
    return evidence


def _validate_bulk_destination_with_retries(
    path: Path,
    record: Mapping[str, Any],
    *,
    verify_remote_sha256: bool,
    attempts: int = 4,
) -> dict[str, Any]:
    """Tolerate short OSSFS close-to-open metadata visibility delays."""

    if attempts <= 0:
        raise ValueError("remote validation attempts must be positive")
    for attempt in range(attempts):
        try:
            return _validate_bulk_destination(
                path, record, verify_remote_sha256=verify_remote_sha256
            )
        except (ConversionError, OSError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError("unreachable")


def _copy_file(source: Path, destination: Path, *, block_bytes: int) -> None:
    if block_bytes <= 0:
        raise ValueError("copy block size must be positive")
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while payload := incoming.read(block_bytes):
            outgoing.write(payload)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _parse_chunk_file(relative_path: str) -> tuple[int, int]:
    path = Path(relative_path)
    try:
        chunk = int(path.parent.name.removeprefix("chunk-"))
        file_index = int(path.stem.removeprefix("file-"))
    except ValueError as exc:
        raise ConversionError(f"invalid LeRobot chunk path {relative_path!r}") from exc
    return chunk, file_index


def _bulk_source_paths(unit: ParallelWorkUnit) -> list[Path]:
    root = Path(unit.target_path)
    paths: list[Path] = []
    for directory in (root / "data", root / "videos"):
        if directory.is_dir():
            paths.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    if not paths:
        raise ConversionError(f"work unit has no bulk files: {root}")
    return paths


def _destination_for_source(
    unit: ParallelWorkUnit,
    source: Path,
    *,
    data_ordinal: int,
    video_ordinals: dict[str, int],
) -> tuple[Path, int, dict[str, int]]:
    root = Path(unit.target_path)
    relative = source.relative_to(root)
    if relative.parts[0] == "data":
        destination = Path("data") / f"chunk-{unit.index:03d}" / f"file-{data_ordinal:03d}.parquet"
        return destination, data_ordinal + 1, video_ordinals
    if len(relative.parts) < 4 or relative.parts[0] != "videos":
        raise ConversionError(f"unexpected work unit bulk path: {relative}")
    key = relative.parts[1]
    ordinal = video_ordinals.get(key, 0)
    destination = (
        Path("videos")
        / key
        / f"chunk-{unit.index:03d}"
        / f"file-{ordinal:03d}.mp4"
    )
    updated = dict(video_ordinals)
    updated[key] = ordinal + 1
    return destination, data_ordinal, updated


def _set_column(table: Any, name: str, values: Any) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        raise ConversionError(f"Parquet table is missing generated column {name!r}")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _globalize_data_file(path: Path, unit: ParallelWorkUnit) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    local_episode = table["episode_index"].combine_chunks()
    if table.num_rows != unit.weight and len(list((Path(unit.target_path) / "data").rglob("*.parquet"))) == 1:
        raise ConversionError(
            f"unit {unit.key!r} data rows changed: {table.num_rows} != {unit.weight}"
        )
    local_min = int(pc.min(local_episode).as_py())
    local_max = int(pc.max(local_episode).as_py())
    local_count = unit.episode_end - unit.episode_start
    if local_min < 0 or local_max >= local_count:
        raise ConversionError(f"unit {unit.key!r} has invalid local episode indices")
    global_episode = pc.add(local_episode, pa.scalar(unit.episode_start, type=local_episode.type))
    global_index = pc.add(
        table["index"].combine_chunks(),
        pa.scalar(unit.frame_start, type=table.schema.field("index").type),
    )
    task_lookup = pa.array(unit.task_indices, type=table.schema.field("task_index").type)
    global_task = pc.take(task_lookup, local_episode)
    table = _set_column(table, "episode_index", global_episode)
    table = _set_column(table, "index", global_index)
    table = _set_column(table, "task_index", global_task)
    temporary = path.with_name(f".{path.name}.global-{uuid.uuid4().hex}")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        # Both paths are in the same local unit directory. OSS bulk commits
        # never use rename/replace and go through _copy_file below.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def globalize_unit_data_files(unit: ParallelWorkUnit) -> None:
    """Patch generated global indices before the unit's one final inventory."""

    root = Path(unit.target_path)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ConversionError(f"work unit has no Parquet data files: {root}")
    for path in paths:
        _globalize_data_file(path, unit)


def _inventory_map(marker: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for value in marker.get("inventory", []):
        if not isinstance(value, Mapping):
            raise ConversionError("verified unit inventory contains an invalid record")
        relative = value.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in result:
            raise ConversionError("verified unit inventory contains an invalid path")
        result[relative] = value
    if not result:
        raise ConversionError("verified unit inventory is empty")
    return result


def _record_from_inventory(
    path: Path,
    *,
    relative_to: Path,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relative = path.relative_to(relative_to).as_posix()
    value = inventory.get(relative)
    if value is None:
        raise ConversionError(f"verified inventory is missing {relative}")
    size = int(value.get("size", -1))
    digest = value.get("sha256")
    if path.stat().st_size != size or not isinstance(digest, str):
        raise ConversionError(f"verified inventory identity changed for {path}")
    return {"relative_path": relative, "size": size, "sha256": digest}


def _metadata_inventory(
    unit: ParallelWorkUnit,
    inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(unit.target_path)
    paths = []
    for relative in (Path("meta"), Path("conversion_manifest.json")):
        path = root / relative
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(item for item in path.rglob("*") if item.is_file()))
    if not any(record.relative_to(root).as_posix() == "meta/info.json" for record in paths):
        raise ConversionError(f"work unit metadata is incomplete: {root}")
    return [
        _record_from_inventory(path, relative_to=root, inventory=inventory)
        for path in paths
    ]


def _make_commit_intent(
    unit: ParallelWorkUnit,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
    *,
    sample_bytes: int,
    verified_marker: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(unit.target_path)
    inventory = _inventory_map(verified_marker)
    data_ordinal = 0
    video_ordinals: dict[str, int] = {}
    bulk: list[dict[str, Any]] = []
    for source in _bulk_source_paths(unit):
        destination, data_ordinal, video_ordinals = _destination_for_source(
            unit,
            source,
            data_ordinal=data_ordinal,
            video_ordinals=video_ordinals,
        )
        if data_ordinal > 1000 or any(value > 1000 for value in video_ordinals.values()):
            raise ConversionError(
                f"unit {unit.key!r} exceeds LeRobot's 1000-files-per-chunk limit"
            )
        record = _record_from_inventory(
            source, relative_to=root, inventory=inventory
        )
        record["samples"] = _sample_evidence(
            source, size=int(record["size"]), sample_bytes=sample_bytes
        )
        record["format"] = _format_evidence(source)
        record["destination"] = destination.as_posix()
        bulk.append(record)
    destinations = [record["destination"] for record in bulk]
    if len(destinations) != len(set(destinations)):
        raise ConversionError(f"unit {unit.key!r} generated duplicate final paths")
    return {
        **_header(unit, partition_name),
        "status": "committing",
        "partition_root": str(partition_root),
        "metadata_root": unit_metadata_root(
            resume_root, partition_name, unit
        ).relative_to(resume_root).as_posix(),
        "bulk": bulk,
        "work_metadata": _metadata_inventory(unit, inventory),
    }


def _finish_intent_once(
    unit: ParallelWorkUnit,
    marker_path: Path,
    marker: dict[str, Any],
    partition_root: Path,
    resume_root: Path,
    *,
    copy_block_bytes: int,
    verify_remote_sha256: bool,
    trust_local_source: bool = False,
    retain_local_after_commit: bool = False,
) -> dict[str, Any]:
    root = Path(unit.target_path)
    local_bulk: list[Path] = []
    for record in marker.get("bulk", []):
        source = root / _marker_relative_path(record.get("relative_path"), "bulk source")
        local_bulk.append(source)
        destination = partition_root / _marker_relative_path(
            record.get("destination"), "bulk destination"
        )
        evidence: dict[str, Any] | None = None
        if destination.exists():
            try:
                evidence = _validate_bulk_destination_with_retries(
                    destination,
                    record,
                    verify_remote_sha256=verify_remote_sha256,
                )
            except (ConversionError, OSError, ValueError):
                destination.unlink(missing_ok=True)
                if not source.is_file():
                    raise
        if evidence is None:
            if not source.is_file():
                raise ConversionError(
                    f"neither valid local nor remote bulk file exists: {source}, {destination}"
                )
            if not trust_local_source:
                _validate_record(root, record, "work unit bulk file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                _copy_file(source, destination, block_bytes=copy_block_bytes)
                evidence = _validate_bulk_destination_with_retries(
                    destination,
                    record,
                    verify_remote_sha256=verify_remote_sha256,
                )
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        record["remote_validation"] = evidence
    # Retain the complete local unit until every bulk object has copied and
    # passed remote validation.  A mid-upload failure can therefore resume
    # without reconstructing any source payload.
    if not retain_local_after_commit:
        for source in local_bulk:
            source.unlink(missing_ok=True)
    metadata_root = _marker_metadata_root(marker, resume_root=resume_root)
    if root.exists():
        for record in marker.get("work_metadata", []):
            relative = _marker_relative_path(record.get("relative_path"), "work metadata")
            source = root / relative
            destination = metadata_root / relative
            if source.exists():
                _validate_record(root, record, "work unit metadata")
                destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and destination.exists():
                _validate_record(metadata_root, record, "resumed unit metadata")
                if not retain_local_after_commit:
                    source.unlink(missing_ok=True)
            elif source.exists():
                if retain_local_after_commit:
                    _copy_file(source, destination, block_bytes=copy_block_bytes)
                    _validate_record(metadata_root, record, "copied unit metadata")
                else:
                    # Work and resume are both under the mandated local staging
                    # root, so this is a same-filesystem metadata-only move.
                    os.replace(source, destination)
            elif destination.exists():
                _validate_record(metadata_root, record, "resumed unit metadata")
            else:
                raise ConversionError(
                    f"missing local metadata during direct commit: {relative}"
                )
        if not retain_local_after_commit:
            shutil.rmtree(root)
            from convert_core.parallel import verified_marker_path

            verified_marker_path(unit).unlink(missing_ok=True)
    for record in marker.get("work_metadata", []):
        _validate_record(metadata_root, record, "committed unit metadata")
    marker = {
        **marker,
        "status": "verified",
        "bulk": marker.get("bulk", []),
    }
    return marker


def _finish_intent(
    unit: ParallelWorkUnit,
    marker_path: Path,
    marker: dict[str, Any],
    partition_root: Path,
    resume_root: Path,
    *,
    copy_block_bytes: int,
    verify_remote_sha256: bool,
    trust_local_source: bool = False,
    retain_local_after_commit: bool = False,
) -> dict[str, Any]:
    """Finish one upload attempt and durably accumulate its elapsed time."""

    started = time.monotonic()
    previous_elapsed = float(marker.get("upload_elapsed_seconds", 0.0))
    try:
        completed = _finish_intent_once(
            unit,
            marker_path,
            marker,
            partition_root,
            resume_root,
            copy_block_bytes=copy_block_bytes,
            verify_remote_sha256=verify_remote_sha256,
            trust_local_source=trust_local_source,
            retain_local_after_commit=retain_local_after_commit,
        )
    except BaseException:
        # Keep the intent resumable while retaining timing from failed upload
        # attempts.  Never replace the original upload exception if persisting
        # diagnostic timing itself fails.
        marker["upload_elapsed_seconds"] = previous_elapsed + (
            time.monotonic() - started
        )
        try:
            atomic_write_json(marker_path, marker)
        except BaseException:
            pass
        raise
    completed["upload_elapsed_seconds"] = previous_elapsed + (
        time.monotonic() - started
    )
    atomic_write_json(marker_path, completed)
    return completed


def commit_verified_unit(
    unit: ParallelWorkUnit,
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
    copy_block_bytes: int = DEFAULT_COPY_BLOCK_BYTES,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    verify_remote_sha256: bool = False,
    trust_verified_marker: bool = False,
    retain_local_after_commit: bool = False,
) -> dict[str, Any]:
    """Patch indices, copy bulk files to final paths, validate, then delete local bulk."""

    marker_path = committed_marker_path(resume_root, partition_name, unit)
    if marker_path.exists():
        marker = read_json_object(marker_path, "direct commit marker")
        for key, value in _header(unit, partition_name).items():
            if marker.get(key) != value:
                raise ConversionError(f"direct commit marker identity changed at {marker_path}: {key}")
        if marker.get("status") == "committing":
            return _finish_intent(
                unit,
                marker_path,
                marker,
                partition_root,
                resume_root,
                copy_block_bytes=copy_block_bytes,
                verify_remote_sha256=verify_remote_sha256,
                trust_local_source=False,
                retain_local_after_commit=retain_local_after_commit,
            )
        validate_committed_unit(
            unit,
            partition_name=partition_name,
            partition_root=partition_root,
            resume_root=resume_root,
        )
        return marker
    verified_marker = (
        read_verified_unit_marker(unit)
        if trust_verified_marker
        else validate_verified_unit_marker(unit)
    )
    marker = _make_commit_intent(
        unit,
        partition_name,
        partition_root,
        resume_root,
        sample_bytes=sample_bytes,
        verified_marker=verified_marker,
    )
    atomic_write_json(marker_path, marker)
    return _finish_intent(
        unit,
        marker_path,
        marker,
        partition_root,
        resume_root,
        copy_block_bytes=copy_block_bytes,
        verify_remote_sha256=verify_remote_sha256,
        trust_local_source=trust_verified_marker,
        retain_local_after_commit=retain_local_after_commit,
    )


def validate_committed_unit(
    unit: ParallelWorkUnit,
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
) -> dict[str, Any]:
    marker_path = committed_marker_path(resume_root, partition_name, unit)
    marker = read_json_object(marker_path, "direct commit marker")
    for key, value in _header(unit, partition_name).items():
        if marker.get(key) != value:
            raise ConversionError(f"direct commit marker is stale at {marker_path}: {key}")
    if marker.get("status") != "verified":
        raise ConversionError(f"direct commit is unfinished at {marker_path}")
    for record in marker.get("bulk", []):
        _validate_bulk_destination_with_retries(
            partition_root / _marker_relative_path(record.get("destination"), "bulk destination"),
            record,
            verify_remote_sha256=bool(
                isinstance(record.get("remote_validation"), Mapping)
                and record["remote_validation"].get("remote_sha256_verified")
            ),
        )
    root = _marker_metadata_root(marker, resume_root=resume_root)
    for record in marker.get("work_metadata", []):
        _validate_record(root, record, "committed work metadata")
    return marker


def read_committed_unit_marker(
    unit: ParallelWorkUnit,
    *,
    partition_name: str,
    resume_root: Path,
) -> dict[str, Any]:
    """Read a same-run verified marker without a second remote object pass."""

    marker_path = committed_marker_path(resume_root, partition_name, unit)
    marker = read_json_object(marker_path, "direct commit marker")
    for key, value in _header(unit, partition_name).items():
        if marker.get(key) != value:
            raise ConversionError(
                f"direct commit marker is stale at {marker_path}: {key}"
            )
    if marker.get("status") != "verified":
        raise ConversionError(f"direct commit is unfinished at {marker_path}")
    return marker


def _discard_corrupt_commit(
    unit: ParallelWorkUnit,
    marker: Mapping[str, Any],
    *,
    partition_root: Path,
    marker_path: Path,
    resume_root: Path,
) -> None:
    for record in marker.get("bulk", []):
        destination = record.get("destination")
        with suppress(ConversionError):
            (partition_root / _marker_relative_path(destination, "bulk destination")).unlink(
                missing_ok=True
            )
    target = Path(unit.target_path)
    if target.exists():
        shutil.rmtree(target)
    with suppress(ConversionError, ValueError):
        metadata_root = _marker_metadata_root(marker, resume_root=resume_root)
        if metadata_root.exists():
            shutil.rmtree(metadata_root)
    marker_path.unlink(missing_ok=True)


def _discard_preassigned_chunk(unit: ParallelWorkUnit, partition_root: Path) -> None:
    data_chunk = partition_root / "data" / f"chunk-{unit.index:03d}"
    if data_chunk.exists():
        shutil.rmtree(data_chunk)
    videos = partition_root / "videos"
    if videos.is_dir():
        for key_root in videos.iterdir():
            chunk = key_root / f"chunk-{unit.index:03d}"
            if chunk.exists():
                shutil.rmtree(chunk)


def prepare_direct_commits(
    units: Sequence[ParallelWorkUnit],
    *,
    partition_name: str,
    partition_root: Path,
    resume_root: Path,
) -> DirectCommitPreparation:
    """Revalidate every committed chunk and isolate only corrupt units for rebuilding."""

    validate_work_units(units)
    committed: list[ParallelWorkUnit] = []
    uncommitted: list[ParallelWorkUnit] = []
    discarded: list[str] = []
    for unit in units:
        marker_path = committed_marker_path(resume_root, partition_name, unit)
        if not marker_path.exists():
            # An absent marker cannot authorize reuse. Deterministic per-unit
            # chunk assignment lets us remove any orphan left by external
            # marker deletion without touching another unit.
            _discard_preassigned_chunk(unit, partition_root)
            stale_metadata = unit_metadata_root(resume_root, partition_name, unit)
            if stale_metadata.exists():
                shutil.rmtree(stale_metadata)
            uncommitted.append(unit)
            continue
        try:
            marker = read_json_object(marker_path, "direct commit marker")
        except ConversionError:
            _discard_preassigned_chunk(unit, partition_root)
            stale_metadata = unit_metadata_root(resume_root, partition_name, unit)
            if stale_metadata.exists():
                shutil.rmtree(stale_metadata)
            target = Path(unit.target_path)
            if target.exists():
                shutil.rmtree(target)
            marker_path.unlink(missing_ok=True)
            discarded.append(unit.key)
            uncommitted.append(unit)
            continue
        for key, value in _header(unit, partition_name).items():
            if marker.get(key) != value:
                raise ConversionError(f"direct commit identity changed at {marker_path}: {key}")
        try:
            if marker.get("status") == "committing":
                _finish_intent(
                    unit,
                    marker_path,
                    marker,
                    partition_root,
                    resume_root,
                    copy_block_bytes=DEFAULT_COPY_BLOCK_BYTES,
                    verify_remote_sha256=False,
                )
            validate_committed_unit(
                unit,
                partition_name=partition_name,
                partition_root=partition_root,
                resume_root=resume_root,
            )
        except (ConversionError, OSError, ValueError):
            root = Path(unit.target_path)
            retained_sources = [
                root / str(record.get("relative_path"))
                for record in marker.get("bulk", [])
                if isinstance(record, Mapping)
            ]
            if any(path.is_file() for path in retained_sources):
                # A transient upload failure must be resumable without rebuilding
                # or deleting the only verified local copy.
                raise
            _discard_corrupt_commit(
                unit,
                marker,
                partition_root=partition_root,
                marker_path=marker_path,
                resume_root=resume_root,
            )
            discarded.append(unit.key)
            uncommitted.append(unit)
        else:
            committed.append(unit)
    return DirectCommitPreparation(
        tuple(committed), tuple(uncommitted), tuple(discarded)
    )


def _replace_episode_stat(
    values: dict[str, list[Any]], row: int, feature: str, data: np.ndarray
) -> None:
    from lerobot.datasets.compute_stats import get_feature_stats
    from convert_core.lerobot_writer import normalize_generated_index_stats

    stats = normalize_generated_index_stats(
        {feature: get_feature_stats(data, axis=0, keepdims=True)}
    )[feature]
    for stat, result in stats.items():
        column = f"stats/{feature}/{stat}"
        # v2.1 sources legitimately provide only min/max/mean/std/count,
        # while current writers additionally store quantiles.  Preserve the
        # source metadata contract instead of inventing a wider schema during
        # a metadata-only v2.1 -> v3 logical conversion.
        if column in values:
            values[column][row] = np.asarray(result).tolist()


def _mapping_by_local_path(marker: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(record["relative_path"]): str(record["destination"])
        for record in marker.get("bulk", [])
    }


def _patch_episode_table(
    table: Any,
    unit: ParallelWorkUnit,
    marker: Mapping[str, Any],
) -> Any:
    import pyarrow as pa

    values = {name: table[name].to_pylist() for name in table.column_names}
    mapping = _mapping_by_local_path(marker)
    camera_keys = sorted(
        name.removeprefix("videos/").removesuffix("/chunk_index")
        for name in table.column_names
        if name.startswith("videos/") and name.endswith("/chunk_index")
    )
    for row in range(table.num_rows):
        local_episode = int(values["episode_index"][row])
        if local_episode < 0 or local_episode >= len(unit.task_indices):
            raise ConversionError(f"invalid local episode metadata in {unit.key!r}")
        length = int(values["length"][row])
        global_episode = unit.episode_start + local_episode
        global_from = unit.frame_start + int(values["dataset_from_index"][row])
        global_to = unit.frame_start + int(values["dataset_to_index"][row])
        local_data = (
            f"data/chunk-{int(values['data/chunk_index'][row]):03d}/"
            f"file-{int(values['data/file_index'][row]):03d}.parquet"
        )
        if local_data not in mapping:
            raise ConversionError(f"missing data mapping for {local_data}")
        data_chunk, data_file = _parse_chunk_file(mapping[local_data])
        values["episode_index"][row] = global_episode
        values["dataset_from_index"][row] = global_from
        values["dataset_to_index"][row] = global_to
        values["data/chunk_index"][row] = data_chunk
        values["data/file_index"][row] = data_file
        for key in camera_keys:
            local_video = (
                f"videos/{key}/chunk-"
                f"{int(values[f'videos/{key}/chunk_index'][row]):03d}/"
                f"file-{int(values[f'videos/{key}/file_index'][row]):03d}.mp4"
            )
            if local_video not in mapping:
                raise ConversionError(f"missing video mapping for {local_video}")
            video_chunk, video_file = _parse_chunk_file(mapping[local_video])
            values[f"videos/{key}/chunk_index"][row] = video_chunk
            values[f"videos/{key}/file_index"][row] = video_file
        _replace_episode_stat(
            values,
            row,
            "episode_index",
            np.full(length, global_episode, dtype=np.int64),
        )
        _replace_episode_stat(
            values,
            row,
            "index",
            np.arange(global_from, global_to, dtype=np.int64),
        )
        _replace_episode_stat(
            values,
            row,
            "task_index",
            np.full(length, unit.task_indices[local_episode], dtype=np.int64),
        )
    arrays = [pa.array(values[field.name], type=field.type) for field in table.schema]
    return pa.Table.from_arrays(arrays, schema=table.schema)


def _episode_stats(table: Any) -> Iterable[dict[str, dict[str, np.ndarray]]]:
    columns = [name for name in table.column_names if name.startswith("stats/")]
    values = {name: table[name].to_pylist() for name in columns}
    for row in range(table.num_rows):
        stats: dict[str, dict[str, np.ndarray]] = {}
        for column in columns:
            feature, stat = column[len("stats/") :].rsplit("/", 1)
            stats.setdefault(feature, {})[stat] = np.asarray(values[column][row])
        yield stats


def _merge_stats_preserving_empty(
    left: dict[str, dict[str, np.ndarray]],
    right: dict[str, dict[str, np.ndarray]],
    aggregate_stats: Any,
) -> dict[str, dict[str, np.ndarray]]:
    """Merge episode stats without turning official zero-count entries into NaN."""

    merged: dict[str, dict[str, np.ndarray]] = {}
    for feature in left.keys() | right.keys():
        if feature not in left:
            merged[feature] = right[feature]
            continue
        if feature not in right:
            merged[feature] = left[feature]
            continue
        left_count = int(np.asarray(left[feature].get("count", 0)).sum())
        right_count = int(np.asarray(right[feature].get("count", 0)).sum())
        if left_count == 0:
            merged[feature] = right[feature] if right_count else left[feature]
        elif right_count == 0:
            merged[feature] = left[feature]
        else:
            merged[feature] = aggregate_stats(
                [{feature: left[feature]}, {feature: right[feature]}]
            )[feature]
    return merged


def _set_metadata_location(table: Any, chunk_index: int, file_index: int) -> Any:
    return _set_column(
        _set_column(
            table,
            "meta/episodes/chunk_index",
            np.full(table.num_rows, chunk_index, dtype=np.int64),
        ),
        "meta/episodes/file_index",
        np.full(table.num_rows, file_index, dtype=np.int64),
    )


def _validate_final_metadata(plan: DatasetConversionPlan, partition_root: Path) -> None:
    """Validate only compact v3 metadata after logical aggregation."""

    import pyarrow.parquet as pq

    info_path = partition_root / "meta" / "info.json"
    tasks_path = partition_root / "meta" / "tasks.parquet"
    episodes_root = partition_root / "meta" / "episodes"
    stats_path = partition_root / "meta" / "stats.json"
    if not info_path.is_file() or not tasks_path.is_file() or not stats_path.is_file() or not episodes_root.is_dir():
        raise ConversionError(f"final LeRobot metadata is incomplete: {partition_root}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"final LeRobot metadata is not readable: {partition_root}: {exc}") from exc
    if not isinstance(info, dict) or not isinstance(stats, dict):
        raise ConversionError(f"final LeRobot metadata has invalid JSON objects: {partition_root}")
    if info.get("codebase_version") != "v3.0" or info.get("total_episodes") != len(plan.episodes) or info.get("total_frames") != plan.num_frames:
        raise ConversionError(f"final LeRobot info totals/version are inconsistent: {partition_root}")
    if pq.ParquetFile(tasks_path).metadata.num_rows != len(dict.fromkeys(episode.instruction for episode in plan.episodes)):
        raise ConversionError(f"final LeRobot task metadata count is inconsistent: {tasks_path}")
    episode_paths = sorted(episodes_root.rglob("*.parquet"))
    if sum(pq.ParquetFile(path).metadata.num_rows for path in episode_paths) != len(plan.episodes):
        raise ConversionError(f"final LeRobot episode metadata count is inconsistent: {episodes_root}")


def finalize_direct_partition(
    plan: DatasetConversionPlan,
    units: Sequence[ParallelWorkUnit],
    partition_root: Path,
    *,
    resume_root: Path,
    reader_format: str,
    parallel_evidence: Mapping[str, Any],
    metadata_batch_bytes: int = DEFAULT_METADATA_BATCH_BYTES,
    metadata_batch_episodes: int = DEFAULT_METADATA_BATCH_EPISODES,
    revalidate_remote: bool = False,
) -> Path:
    """Stream final metadata/stats after all bulk chunks are committed."""

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    from convert_core.lerobot_writer import (
        LEROBOT_CODEBASE_VERSION,
        LEROBOT_DATA_PATH,
        LEROBOT_VIDEO_PATH,
        build_manifest,
    )

    validate_work_units(units)
    if metadata_batch_bytes <= 0 or metadata_batch_episodes <= 0:
        raise ValueError("metadata batch limits must be positive")
    markers = []
    for unit in units:
        if revalidate_remote:
            marker = validate_committed_unit(
                unit,
                partition_name=plan.output_path.name,
                partition_root=partition_root,
                resume_root=resume_root,
            )
        else:
            marker = read_committed_unit_marker(
                unit,
                partition_name=plan.output_path.name,
                resume_root=resume_root,
            )
        markers.append(marker)
    meta_root = partition_root / "meta"
    if meta_root.exists():
        shutil.rmtree(meta_root)
    meta_root.mkdir(parents=True)

    metadata_roots = [
        _marker_metadata_root(marker, resume_root=resume_root) for marker in markers
    ]
    first_info = json.loads(
        (metadata_roots[0] / "meta" / "info.json").read_text(encoding="utf-8")
    )
    tasks = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    first_info.update(
        codebase_version=LEROBOT_CODEBASE_VERSION,
        data_path=LEROBOT_DATA_PATH,
        video_path=LEROBOT_VIDEO_PATH,
        total_episodes=len(plan.episodes),
        total_frames=plan.num_frames,
        total_tasks=len(tasks),
        splits={"train": f"0:{len(plan.episodes)}"},
    )
    # Legacy v2.1 sources can carry informational counters that v3 does not
    # recognise.  They must not survive in the final v3 metadata.
    first_info.pop("total_chunks", None)
    first_info.pop("total_videos", None)
    atomic_write_json(meta_root / "info.json", first_info)
    task_frame = pd.DataFrame(
        {"task_index": range(len(tasks))},
        index=pd.Index(tasks, name="task"),
    )
    task_frame.to_parquet(meta_root / "tasks.parquet")

    chunks_size = int(first_info["chunks_size"])
    buffered: list[Any] = []
    buffered_rows = 0
    buffered_bytes = 0
    metadata_ordinal = 0
    global_stats: dict[str, dict[str, np.ndarray]] | None = None

    def flush() -> None:
        nonlocal buffered, buffered_rows, buffered_bytes, metadata_ordinal
        if not buffered:
            return
        chunk_index, file_index = divmod(metadata_ordinal, chunks_size)
        table = pa.concat_tables(buffered) if len(buffered) > 1 else buffered[0]
        table = _set_metadata_location(table, chunk_index, file_index)
        path = (
            meta_root
            / "episodes"
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)
        metadata_ordinal += 1
        buffered = []
        buffered_rows = 0
        buffered_bytes = 0

    for unit, marker, metadata_root in zip(
        units, markers, metadata_roots, strict=True
    ):
        episode_paths = sorted((metadata_root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_paths:
            raise ConversionError(f"unit metadata has no episode parquet: {metadata_root}")
        unit_tables = [_patch_episode_table(pq.read_table(path), unit, marker) for path in episode_paths]
        unit_table = pa.concat_tables(unit_tables) if len(unit_tables) > 1 else unit_tables[0]
        if unit_table.num_rows != unit.episode_end - unit.episode_start:
            raise ConversionError(f"unit {unit.key!r} episode metadata count changed")
        for stats in _episode_stats(unit_table):
            global_stats = (
                stats
                if global_stats is None
                else _merge_stats_preserving_empty(global_stats, stats, aggregate_stats)
            )
        if buffered and (
            buffered_rows + unit_table.num_rows > metadata_batch_episodes
            or buffered_bytes + unit_table.nbytes > metadata_batch_bytes
        ):
            flush()
        buffered.append(unit_table)
        buffered_rows += unit_table.num_rows
        buffered_bytes += unit_table.nbytes
    flush()
    if global_stats is None:
        raise ConversionError("direct commit produced no episode statistics")
    write_stats(global_stats, partition_root)
    _validate_final_metadata(plan, partition_root)

    video_evidence: dict[str, list[dict[str, Any]]] = {}
    for marker in markers:
        for record in marker.get("bulk", []):
            destination = str(record.get("destination", ""))
            if not destination.endswith(".mp4"):
                continue
            parts = Path(destination).parts
            if len(parts) < 2 or parts[0] != "videos":
                raise ConversionError(
                    f"committed video has invalid destination: {destination}"
                )
            video_evidence.setdefault(parts[1], []).append(
                {
                    "path": destination,
                    **dict(record.get("remote_validation", {}).get("format", {})),
                    "validation": "size+sample-ranges+container-metadata",
                }
            )
    manifest = build_manifest(plan, reader_format=reader_format)
    manifest.update(
        {
            "num_video_files": sum(len(rows) for rows in video_evidence.values()),
            "video_validation": video_evidence,
            "parallel": {
                "schema_version": DIRECT_COMMIT_SCHEMA_VERSION,
                "direct_final_chunks": True,
                "bulk_aggregation_copy": False,
                "bulk_commit_validation": [
                    {
                        "unit_key": unit.key,
                        "files": [
                            {
                                key: record[key]
                                for key in (
                                    "destination",
                                    "size",
                                    "sha256",
                                    "samples",
                                    "format",
                                    "remote_validation",
                                )
                            }
                            for record in marker.get("bulk", [])
                        ],
                    }
                    for unit, marker in zip(units, markers, strict=True)
                ],
                "aggregation_order": [unit.key for unit in units],
                "checkpoint_granularity": "reader-defined work unit",
                "metadata_batch_files": metadata_ordinal,
                **dict(parallel_evidence),
            },
        }
    )
    atomic_write_json(partition_root / "conversion_manifest.json", manifest)
    return partition_root
