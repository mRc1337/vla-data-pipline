"""Reliable, task-scoped RoboOmni -> LeRobot v3.0 conversion.

The source release does not declare a capture FPS. ``--fps`` therefore names
the output timebase explicitly; it is not an observed source fact. The
expensive TFRecord wire scan is cached as a task catalog and immutable
per-task plans. Only the current task is preflighted on resume.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from convert_core.checkpoint import atomic_write_json, canonical_fingerprint, read_json_object
from convert_core.direct_commit import (
    DirectCommitUploader,
    committed_marker_path,
    commit_verified_unit,
    finalize_direct_partition,
    read_committed_unit_marker,
)
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import validate_video_files, validate_written_dataset, write_dataset
from convert_core.parallel import (
    ParallelWorkUnit,
    PreparedUnits,
    inflight_estimate,
    prepare_work_units,
    run_parallel_work_units,
    validate_verified_unit_marker,
    verified_marker_path,
    write_verified_unit_marker,
)
from convert_core.staging import (
    DEFAULT_MIN_LOCAL_FREE_BYTES,
    StagingCapacityGuard,
    configure_runtime_environment,
    create_incomplete_output,
    make_staging_layout,
    publish_success,
    exclusive_staging_lock,
    validate_source_and_output_roots,
)
from readers.roboomni_reader import (
    RoboOmniCatalog,
    RoboOmniField,
    RoboOmniTask,
    RoboOmniTaskMember,
    _read_record,
    build_catalog,
    build_tasks,
    parse_example,
    task_catalog_payload,
    task_from_payload,
    task_plan_payload,
)


TASK_MARKER_SCHEMA_VERSION = 1
TASK_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UnitPayload:
    plan: DatasetConversionPlan
    fields: tuple[RoboOmniField, ...]
    encoder_threads: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_audio_assets(catalog: Any, final_root: Path) -> dict[str, Any]:
    """Preserve the legacy optional audio sidecar helper for unit tests/tools.

    The task orchestrator does not use this collection-wide copy during task
    conversion; speech references remain in the source fields.  If a caller
    explicitly asks for this helper, it is copy-then-verify and records missing
    references instead of silently dropping them.
    """
    manifest_path = final_root / "audio_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous_payload = read_json_object(manifest_path, "RoboOmni audio manifest")
        previous = {
            str(row["reference"]): row for row in previous_payload.get("assets", [])
            if isinstance(row, dict) and isinstance(row.get("reference"), str)
        }
    assets: list[dict[str, Any]] = []
    for item in catalog.audio_inventory:
        reference = str(item["reference"])
        relative = item.get("relative_path")
        row = {"reference": reference, "source_path": str(item.get("source_path", "")), "relative_path": relative, "exists_at_preflight": bool(item.get("exists")), "status": "missing"}
        if not item.get("exists"):
            assets.append(row)
            continue
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ConversionError(f"invalid RoboOmni audio relative path: {relative!r}")
        source = Path(str(item["source_path"]))
        destination = final_root / "audio" / relative
        old = previous.get(reference, {})
        if destination.is_file() and destination.stat().st_size == int(item["size"]) and old.get("sha256") == _sha256_file(destination):
            row.update(status="copied", destination=str(destination.relative_to(final_root)), size=int(item["size"]), sha256=old["sha256"])
            assets.append(row)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # This compatibility helper may target a different filesystem; use
        # copy-then-verify rather than Path.replace across filesystems.
        shutil.copyfile(source, destination)
        digest = _sha256_file(destination)
        if destination.stat().st_size != int(item["size"]):
            raise ConversionError(f"RoboOmni audio copy size mismatch: {source}")
        row.update(status="copied", destination=str(destination.relative_to(final_root)), size=int(item["size"]), sha256=digest)
        assets.append(row)
    summary = {"referenced": len(assets), "copied": sum(row["status"] == "copied" for row in assets), "missing": sum(row["status"] == "missing" for row in assets)}
    atomic_write_json(manifest_path, {"format": "roboomni_external_audio_v1", "assets": assets, "summary": summary})
    return summary


def _write_immutable(path: Path, payload: Mapping[str, Any], description: str) -> None:
    value = dict(payload)
    if path.is_file():
        if read_json_object(path, description) != value:
            raise ConversionError(f"immutable {description} changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _fingerprint_without(payload: Mapping[str, Any], key: str = "fingerprint") -> str:
    value = dict(payload)
    value.pop(key, None)
    return canonical_fingerprint(value)


def _task_plan_with_fingerprint(
    catalog: RoboOmniCatalog,
    task: RoboOmniTask,
    *,
    fps: int,
    output_uid: str,
    encoder_threads: int,
) -> dict[str, Any]:
    plan = task_plan_payload(
        catalog,
        task,
        fps=fps,
        output_dataset_uid=output_uid,
        encoding={
            "video_encoder": "LeRobot default CPU encoder",
            "fps": fps,
            "encoder_threads_per_worker": encoder_threads,
            "streaming_encoding": True,
            "blocking_streaming_encoding": True,
            "deferred_video_concatenation": True,
            "image_conversion": "decode source image and encode RGB video",
        },
    )
    plan["fingerprint"] = _fingerprint_without(plan)
    return plan


def _task_catalog_entry(task: RoboOmniTask, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_key": task.task_key,
        "instruction": task.instruction,
        "task_index": task.task_index,
        "plan_path": f"tasks/{task.task_key}/task_plan.json",
        "fingerprint": str(plan["fingerprint"]),
        "episode_count": task.episode_count,
        "frame_count": task.frame_count,
        "source_bytes": task.source_bytes,
        "members": [
            {
                "partition_name": member.partition_name,
                "episode_start": member.episode_start,
                "episode_end": member.episode_end,
                "frame_start": member.frame_start,
                "frame_end": member.frame_end,
                "unit_indices": [record.global_episode_index for record in member.records],
            }
            for member in task.members
        ],
    }


def _catalog_fingerprint(payload: Mapping[str, Any]) -> str:
    return _fingerprint_without(payload, "catalog_fingerprint")


def _task_marker_path(resume_root: Path, task_key: str) -> Path:
    return resume_root / "tasks" / task_key / "commit.json"


def _read_task_marker(path: Path, *, task_key: str, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    marker = read_json_object(path, "RoboOmni task commit marker")
    if marker.get("schema_version") != TASK_MARKER_SCHEMA_VERSION or marker.get("status") != "committed":
        raise ConversionError(f"invalid RoboOmni task marker: {path}")
    if marker.get("task_key") != task_key or marker.get("fingerprint") != fingerprint:
        raise ConversionError(f"RoboOmni task marker identity changed: {path}")
    return marker


def _task_sources(task: RoboOmniTask) -> tuple[dict[str, Any], ...]:
    seen: dict[str, dict[str, Any]] = {}
    for member in task.members:
        for item in member.source_files:
            seen[str(item["path"])] = dict(item)
    return tuple(seen.values())


def _validate_task_sources(task: RoboOmniTask) -> None:
    for item in _task_sources(task):
        path = Path(str(item["path"]))
        try:
            stat = path.stat()
        except OSError as exc:
            raise ConversionError(f"RoboOmni task source is unavailable: {path}") from exc
        if stat.st_size != int(item["size"]) or stat.st_mtime_ns != int(item["mtime_ns"]):
            raise ConversionError(f"RoboOmni task source changed since catalog: {path}")


def _record_counts(values: Mapping[str, tuple[str, list[Any]]], fields: Sequence[RoboOmniField], source: str) -> None:
    import numpy as np
    counts: list[int] = []
    for field in fields:
        kind, raw = values.get("steps/" + field.source_key, ("", []))
        if not raw or not kind:
            raise ConversionError(f"task preflight missing steps/{field.source_key}: {source}")
        width = int(np.prod(field.shape)) if field.kind == "numeric" else 1
        if len(raw) % width:
            raise ConversionError(f"task preflight malformed steps/{field.source_key}: {source}")
        counts.append(len(raw) // width)
    if len(set(counts)) != 1 or not counts or counts[0] <= 0:
        raise ConversionError(f"task preflight frame count mismatch: {source}: {counts}")


def _preflight_task(task: RoboOmniTask) -> dict[str, Any]:
    """Read this task's source stats and first/middle/last payload samples."""
    _validate_task_sources(task)
    sampled: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for member in task.members:
        for index in sorted({0, len(member.records) // 2, len(member.records) - 1}):
            record = member.records[index]
            identity = (record.source_file, record.offset)
            if identity in seen:
                continue
            seen.add(identity)
            length, payload = _read_record(Path(record.source_file), record.offset)
            if length != record.payload_length:
                raise ConversionError(f"RoboOmni task record length changed: {record.source_file}@{record.offset}")
            values = parse_example(payload)
            _record_counts(values, member.fields, f"{record.source_file}@{record.offset}")
            sampled.append({"source_file": record.source_file, "offset": record.offset, "payload_length": length})
    return {
        "mode": "task-scoped metadata/index plus first-middle-last payload samples",
        "source_files": len(_task_sources(task)),
        "sampled_records": sampled,
        "episodes": task.episode_count,
        "frames": task.frame_count,
        "estimated_source_bytes": task.source_bytes,
    }


def _feature_specs(fields: tuple[RoboOmniField, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from convert_core.episode_spec import CameraFeatureSpec, VectorFeatureSpec
    import numpy as np
    vectors, cameras = [], []
    for field in fields:
        if field.kind == "image":
            cameras.append(CameraFeatureSpec(_camera_key(field), field.shape[0], field.shape[1]))
        else:
            vectors.append(VectorFeatureSpec(field.output_key, int(np.prod(field.shape)), dtype=field.dtype, shape=field.shape))
    return tuple(vectors), tuple(cameras)


def _camera_key(field: RoboOmniField) -> str:
    names = {"observation/image": "primary", "observation/image_wrist": "wrist", "first_frame_image": "first_frame"}
    return "observation.images." + names.get(field.source_key, field.source_key.replace("/", "."))


def _mapping(fields: Sequence[RoboOmniField]) -> tuple[dict[str, Any], ...]:
    return tuple({
        "source": "steps/" + field.source_key,
        "shape": list(field.shape),
        "dtype": field.dtype,
        "target": _camera_key(field) if field.kind == "image" else field.output_key,
        "conversion": "RGB video encode" if field.kind == "image" else "identity",
        "lossy": field.kind == "image",
        "semantic_unit": "not declared by source" if field.kind == "numeric" else None,
        "evidence": "features.json sidecar",
    } for field in fields)


def _member_plan(task: RoboOmniTask, member: RoboOmniTaskMember, *, output_uid: str, output_path: Path, fps: int, records: Sequence[Any] | None = None) -> DatasetConversionPlan:
    values = tuple(records if records is not None else member.records)
    vectors, cameras = _feature_specs(member.fields)
    episodes = tuple(EpisodePlan(
        f"episode_{record.global_episode_index:08d}",
        f"{record.component}:{Path(record.source_file).name}@{record.offset}",
        record.instruction,
        record.frame_count,
        {"record": record.__dict__, "checkpoint_unit": record.source_file},
    ) for record in values)
    return DatasetConversionPlan(
        output_uid, output_path, fps, float(fps), "unknown", vectors, cameras, episodes,
        {"source_dataset": "fnlp/OmniAction", "source_splits": ["train"],
         "field_mapping": list(_mapping(member.fields)),
         "partition_rules": ["one partition per exact features.json signature"],
         "timebase": {"source_fps": None, "source_timestamps": "absent", "output_fps": fps}},
    )


def _build_units(task: RoboOmniTask, member: RoboOmniTaskMember, *, partition_task_index: int, task_fingerprint: str, output_uid: str, work_root: Path, fps: int, encoder_threads: int) -> tuple[ParallelWorkUnit, ...]:
    units: list[ParallelWorkUnit] = []
    for record in member.records:
        target = work_root / member.partition_name / f"unit-{record.global_episode_index:06d}"
        plan = _member_plan(task, member, output_uid=output_uid, output_path=target, fps=fps, records=(record,))
        source_bytes = max(1, record.payload_length)
        units.append(ParallelWorkUnit(
            index=record.global_episode_index,
            key=f"{task.task_key}/{member.partition_name}/{record.global_episode_index:08d}",
            dataset_uid=output_uid,
            target_path=str(target),
            episode_start=record.global_episode_index,
            episode_end=record.global_episode_index + 1,
            frame_start=record.global_frame_start,
            frame_end=record.global_frame_start + record.frame_count,
            task_indices=(partition_task_index,),
            weight=record.frame_count,
            estimated_memory_bytes=min(2_000_000_000, source_bytes * 2 + 256 * 1024 * 1024),
            estimated_temp_bytes=source_bytes * 3 + 512 * 1024 * 1024,
            fingerprint=task_fingerprint,
            payload=UnitPayload(plan, member.fields, encoder_threads),
        ))
    return tuple(units)


def _unit_worker(unit: ParallelWorkUnit) -> dict[str, Any]:
    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid RoboOmni unit payload: {unit.key}")
    target = Path(unit.target_path)
    if target.exists():
        shutil.rmtree(target)

    def frames(episode: EpisodePlan):
        from readers.roboomni_reader import RoboOmniRecord, decode_record
        yield from decode_record(RoboOmniRecord(**episode.extra["record"]), payload.fields)[0]

    write_dataset(payload.plan, frames, target, streaming_encoding=True, blocking_streaming_encoding=True,
                  encoder_threads=payload.encoder_threads, encoder_temp_root=target / "encoder",
                  batch_metadata_writes=True, deferred_video_concatenation=True)
    validate_written_dataset(payload.plan, target)
    validate_video_files(payload.plan, target, expected_frames=unit.weight)
    from convert_core.direct_commit import globalize_unit_data_files
    globalize_unit_data_files(unit)
    write_verified_unit_marker(unit)
    return {"unit": unit.key, "frames": unit.weight}


def _validate_unit(unit: ParallelWorkUnit) -> None:
    payload = unit.payload
    if not isinstance(payload, UnitPayload):
        raise ConversionError(f"invalid RoboOmni unit payload: {unit.key}")
    if verified_marker_path(unit).is_file():
        validate_verified_unit_marker(unit)
        return
    validate_written_dataset(payload.plan, Path(unit.target_path))
    validate_video_files(payload.plan, Path(unit.target_path), expected_frames=unit.weight)
    from convert_core.direct_commit import globalize_unit_data_files
    globalize_unit_data_files(unit)


def _warmup_first_task(
    task: RoboOmniTask,
    *,
    task_fingerprint: str,
    output_uid: str,
    work_root: Path,
    resume_root: Path,
    fps: int,
    encoder_threads: int,
) -> dict[str, Any]:
    """Encode one short source sample once, then remove every warmup file."""
    member = task.members[0]
    record = member.records[0]
    frame_count = min(60, record.frame_count)
    plan = _member_plan(
        task,
        member,
        output_uid=output_uid + "_warmup",
        output_path=work_root,
        fps=fps,
        records=(record,),
    )
    episode = replace(plan.episodes[0], num_frames=frame_count)
    plan = replace(plan, episodes=(episode,))

    def frames(_episode: EpisodePlan):
        from readers.roboomni_reader import RoboOmniRecord, decode_record
        decoded = decode_record(RoboOmniRecord(**_episode.extra["record"]), member.fields)[0]
        yield from itertools.islice(decoded, frame_count)

    try:
        write_dataset(
            plan,
            frames,
            work_root,
            streaming_encoding=True,
            blocking_streaming_encoding=True,
            encoder_threads=encoder_threads,
            encoder_temp_root=work_root / "encoder",
            batch_metadata_writes=True,
            deferred_video_concatenation=True,
        )
        validate_written_dataset(plan, work_root)
        validate_video_files(plan, work_root, expected_frames=frame_count)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    marker = {
        "schema_version": 1,
        "status": "completed",
        "task_key": task.task_key,
        "task_fingerprint": task_fingerprint,
        "source_file": record.source_file,
        "source_offset": record.offset,
        "source_payload_length": record.payload_length,
        "frames": frame_count,
        "fps": fps,
        "encoder_threads_per_worker": encoder_threads,
    }
    _write_immutable(resume_root / "warmup.json", marker, "RoboOmni warmup marker")
    return marker


def _convert_member(task: RoboOmniTask, member: RoboOmniTaskMember, *, partition_task_index: int, task_fingerprint: str, output_uid: str, work_root: Path, final_root: Path, resume_root: Path, workers: int, upload_workers: int, encoder_threads: int, fps: int, capacity: StagingCapacityGuard) -> dict[str, Any]:
    units = _build_units(task, member, partition_task_index=partition_task_index, task_fingerprint=task_fingerprint, output_uid=output_uid, work_root=work_root, fps=fps, encoder_threads=encoder_threads)
    if not units:
        raise ConversionError(f"empty RoboOmni task member: {task.task_key}/{member.partition_name}")
    estimate = inflight_estimate(units, workers)
    capacity.check(f"preflight task {task.task_key}/{member.partition_name}", required_additional_bytes=estimate.temp_bytes)

    committed, candidates = [], []
    for unit in units:
        marker_path = committed_marker_path(resume_root, member.partition_name, unit)
        if marker_path.is_file():
            commit_verified_unit(unit, partition_name=member.partition_name, partition_root=final_root / member.partition_name, resume_root=resume_root)
            committed.append(unit)
        else:
            candidates.append(unit)
    prepared: PreparedUnits = prepare_work_units(candidates, _validate_unit, require_complete_plan=False) if candidates else PreparedUnits((), (), (), ())
    uploader = DirectCommitUploader(partition_name=member.partition_name, partition_root=final_root / member.partition_name, resume_root=resume_root, workers=upload_workers, max_queue_units=max(1, workers))
    by_index = {unit.index: unit for unit in prepared.pending}
    completion: tuple[str, ...] = ()
    try:
        for unit in prepared.reusable:
            uploader.submit(unit, trust_verified_marker=False)
        def on_result(result: Any) -> None:
            uploader.submit(by_index[result.index], trust_verified_marker=True)
        def before_dispatch(unit: ParallelWorkUnit, active: tuple[ParallelWorkUnit, ...]) -> None:
            capacity.check(f"dispatch task {task.task_key}/{member.partition_name}/{unit.index}", required_additional_bytes=unit.estimated_temp_bytes)
        def health_check() -> None:
            uploader.raise_if_failed()
            capacity.periodic_check(f"running task {task.task_key}/{member.partition_name}")
        if prepared.pending:
            result = run_parallel_work_units(prepared.pending, _unit_worker, workers=workers, on_result=on_result, before_dispatch=before_dispatch, health_check=health_check, health_check_interval_seconds=2.0)
            completion = result.completion_order
        upload_stats = uploader.close()
    except BaseException:
        uploader.close(raise_on_failure=False)
        raise
    markers = [read_committed_unit_marker(unit, partition_name=member.partition_name, resume_root=resume_root) for unit in units]
    return {"partition_name": member.partition_name, "units": len(units), "frames": sum(unit.weight for unit in units), "upload": upload_stats, "reused_units": [unit.key for unit in committed] + [unit.key for unit in prepared.reusable], "completion_order": list(completion), "inflight_estimate": estimate.__dict__, "unit_markers": markers}


def _commit_task(task: RoboOmniTask, plan: Mapping[str, Any], *, preflight: Mapping[str, Any], member_results: Sequence[Mapping[str, Any]], resume_root: Path) -> dict[str, Any]:
    marker = {
        "schema_version": TASK_MARKER_SCHEMA_VERSION,
        "status": "committed",
        "task_key": task.task_key,
        "task_index": task.task_index,
        "instruction": task.instruction,
        "fingerprint": plan["fingerprint"],
        "source": {"source_root": plan["source_root"], "source_files": _task_sources(task), "episode_count": task.episode_count, "frame_count": task.frame_count, "source_bytes": task.source_bytes},
        "index_ranges": [{"partition_name": member.partition_name, "episode_start": member.episode_start, "episode_end": member.episode_end, "frame_start": member.frame_start, "frame_end": member.frame_end} for member in task.members],
        "preflight": dict(preflight),
        "members": list(member_results),
        "task_plan": dict(plan),
        "committed_unix": time.time(),
    }
    path = _task_marker_path(resume_root, task.task_key)
    _write_immutable(path, marker, "RoboOmni task commit marker")
    return marker


def _task_from_marker(marker: Mapping[str, Any]) -> RoboOmniTask:
    try:
        return task_from_payload(marker["task_plan"])
    except (KeyError, TypeError) as exc:
        raise ConversionError("RoboOmni task marker has no usable task plan") from exc


def _finalize_collection(task_markers: Sequence[Mapping[str, Any]], *, output_uid: str, final_root: Path, resume_root: Path, work_root: Path, fps: int, encoder_threads: int) -> list[dict[str, Any]]:
    by_partition: dict[str, list[tuple[RoboOmniTask, RoboOmniTaskMember, ParallelWorkUnit]]] = {}
    local_task_indices: dict[str, dict[str, int]] = {}
    for marker in sorted(task_markers, key=lambda item: int(item["task_index"])):
        task = _task_from_marker(marker)
        for member in task.members:
            mapping = local_task_indices.setdefault(member.partition_name, {})
            local_index = mapping.setdefault(task.task_key, len(mapping))
            units = _build_units(task, member, partition_task_index=local_index, task_fingerprint=str(marker["fingerprint"]), output_uid=output_uid, work_root=work_root / "finalize" / task.task_key, fps=fps, encoder_threads=encoder_threads)
            by_partition.setdefault(member.partition_name, []).extend((task, member, unit) for unit in units)
    results: list[dict[str, Any]] = []
    for partition_name, rows in sorted(by_partition.items()):
        rows.sort(key=lambda item: item[2].index)
        unique_members: dict[tuple[str, str], RoboOmniTaskMember] = {}
        for marker in task_markers:
            task = _task_from_marker(marker)
            for member in task.members:
                if member.partition_name == partition_name:
                    unique_members[(task.task_key, partition_name)] = member
        first_member = next(iter(unique_members.values()))
        records = sorted((record for member in unique_members.values() for record in member.records), key=lambda record: record.global_episode_index)
        vectors, cameras = _feature_specs(first_member.fields)
        plan = DatasetConversionPlan(output_uid, Path(output_uid) / partition_name, fps, float(fps), "unknown", vectors, cameras, tuple(EpisodePlan(f"episode_{record.global_episode_index:08d}", f"{record.component}:{Path(record.source_file).name}@{record.offset}", record.instruction, record.frame_count, {"record": record.__dict__, "checkpoint_unit": record.source_file}) for record in records), {"source_dataset": "fnlp/OmniAction", "source_splits": ["train"], "field_mapping": list(_mapping(first_member.fields)), "timebase": {"source_fps": None, "source_timestamps": "absent", "output_fps": fps}})
        finalize_direct_partition(plan, [row[2] for row in rows], final_root / partition_name, resume_root=resume_root, reader_format="roboomni", parallel_evidence={"task_scoped": True, "task_order": [row[0].task_key for row in rows], "encoder_threads_per_worker": encoder_threads, "remote_revalidation": False})
        results.append({"partition_name": partition_name, "episodes": len(records), "frames": plan.num_frames})
    return results


def _output_parent(requested: Path, output_uid: str = "roboomni") -> Path:
    return requested.parent if requested.name in {output_uid, "roboomni", "robomni"} else requested


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data/embodied_datasets/public_datasets_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/roboomni"))
    parser.add_argument("--local-work-root", type=Path, default=Path.home() / "roboomni_staging")
    parser.add_argument("--output-dataset-uid", default="roboomni")
    parser.add_argument("--fps", type=int, required=True, help="explicit output FPS; source FPS is not declared")
    parser.add_argument("--inspect-only", "--dry-run", action="store_true", dest="inspect_only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--encoder-threads-per-worker", type=int, default=8)
    parser.add_argument("--upload-workers", type=int, default=1)
    parser.add_argument("--min-local-free-bytes", type=int, default=DEFAULT_MIN_LOCAL_FREE_BYTES)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-episodes", type=int, help="maximum episodes per selected task")
    parser.add_argument("--task", action="append", default=[], help="task key or exact source instruction")
    parser.add_argument("--start-task", help="stable task key or zero-based task index")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-id")
    return parser


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    return {"max_shards": args.max_shards, "max_episodes": args.max_episodes, "tasks": sorted(args.task), "start_task": args.start_task, "max_tasks": args.max_tasks}


def _select_tasks(catalog: RoboOmniCatalog, args: argparse.Namespace) -> tuple[RoboOmniTask, ...]:
    tasks = list(catalog.tasks or build_tasks(catalog))
    if args.task:
        requested = set(args.task)
        known = {task.task_key for task in tasks} | {task.instruction for task in tasks}
        missing = sorted(requested - known)
        if missing:
            raise ConversionError(f"unknown RoboOmni task(s): {missing}")
        tasks = [task for task in tasks if task.task_key in requested or task.instruction in requested]
    if args.start_task is not None:
        if args.start_task.isdigit():
            tasks = [task for task in tasks if task.task_index >= int(args.start_task)]
        else:
            positions = [index for index, task in enumerate(tasks) if task.task_key == args.start_task]
            if not positions:
                raise ConversionError(f"unknown --start-task: {args.start_task}")
            tasks = tasks[positions[0]:]
    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            raise SystemExit("--max-tasks must be positive")
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise ConversionError("RoboOmni selection contains no tasks")
    return tuple(tasks)


def _load_or_create_catalog(args: argparse.Namespace, *, layout: Any, raw_root: Path, output_uid: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], RoboOmniCatalog | None]:
    catalog_path = layout.resume / "task_catalog.json"
    selection = _selection(args)
    if args.resume:
        if not catalog_path.is_file():
            raise ConversionError(f"--resume requires existing task catalog: {catalog_path}")
        payload = read_json_object(catalog_path, "RoboOmni task catalog")
        if payload.get("schema_version") != TASK_CATALOG_SCHEMA_VERSION or payload.get("output_dataset_uid") != output_uid or payload.get("fps") != args.fps or payload.get("selection") != selection:
            raise ConversionError("RoboOmni task catalog identity changed; use original arguments")
        if payload.get("catalog_fingerprint") != _catalog_fingerprint(payload):
            raise ConversionError("RoboOmni task catalog fingerprint is corrupt")
        return payload, tuple(dict(item) for item in payload.get("tasks", [])), None
    if catalog_path.exists():
        raise ConversionError(f"task catalog already exists; use --resume: {catalog_path}")
    catalog = build_catalog(raw_root, fps=args.fps, max_shards=args.max_shards)
    selected = _select_tasks(catalog, args)
    selected_keys = {task.task_key for task in selected}
    selected_tasks = tuple(task for task in build_tasks(catalog, max_episodes_per_task=args.max_episodes, selected_task_keys=selected_keys) if task.task_key in selected_keys)
    plans, entries = [], []
    for task in selected_tasks:
        plan = _task_plan_with_fingerprint(catalog, task, fps=args.fps, output_uid=output_uid, encoder_threads=args.encoder_threads_per_worker)
        _write_immutable(layout.resume / "tasks" / task.task_key / "task_plan.json", plan, "RoboOmni task plan")
        plans.append(plan)
        entries.append(_task_catalog_entry(task, plan))
    payload = task_catalog_payload(catalog, output_dataset_uid=output_uid, fps=args.fps, task_plans=entries, selection=selection)
    payload.update({"schema_version": TASK_CATALOG_SCHEMA_VERSION, "mapping_table": list(catalog.mapping_table), "sidecars": list(catalog.sidecar_inventory)})
    payload["catalog_fingerprint"] = _catalog_fingerprint(payload)
    _write_immutable(catalog_path, payload, "RoboOmni task catalog")
    return payload, tuple(entries), catalog


def _load_task_plan(layout: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    path = layout.resume / str(entry["plan_path"])
    plan = read_json_object(path, "RoboOmni task plan")
    if plan.get("fingerprint") != entry.get("fingerprint") or plan.get("task_key") != entry.get("task_key") or plan.get("fingerprint") != _fingerprint_without(plan):
        raise ConversionError(f"RoboOmni task plan does not match catalog: {path}")
    return plan


def _write_state(layout: Any, catalog: Mapping[str, Any], completed: Sequence[str], current: str | None) -> None:
    atomic_write_json(layout.resume / "global_state.json", {"schema_version": 1, "catalog_fingerprint": catalog["catalog_fingerprint"], "completed_task_keys": list(completed), "current_task": current, "updated_unix": time.time()})


def _run(args: argparse.Namespace, *, raw_root: Path, layout: Any, run_id: str) -> int:
    configure_runtime_environment(layout, create=not args.inspect_only)
    catalog_value, entries, catalog = _load_or_create_catalog(args, layout=layout, raw_root=raw_root, output_uid=args.output_dataset_uid)
    state_path = layout.resume / "global_state.json"
    if args.resume and state_path.is_file():
        state = read_json_object(state_path, "RoboOmni global state")
        if state.get("catalog_fingerprint") != catalog_value.get("catalog_fingerprint"):
            raise ConversionError("RoboOmni global state does not match task catalog")
    if args.inspect_only:
        if catalog is None:
            print(json.dumps(catalog_value, ensure_ascii=False, indent=2))
        else:
            selected_plans = [task_from_payload(_load_task_plan(layout, entry), fps=args.fps) for entry in entries[:2]]
            print(json.dumps({"catalog": catalog_value, "task_preflight": [_preflight_task(task) for task in selected_plans]}, ensure_ascii=False, indent=2, default=str))
        return 0
    layout.create_runtime_directories()
    if (layout.final / "_SUCCESS").is_file():
        raise FileExistsError(f"valid RoboOmni output already exists: {layout.final}")
    create_incomplete_output(layout.final, fingerprint=str(catalog_value["catalog_fingerprint"]), run_id=run_id)
    capacity = StagingCapacityGuard(layout.local_root, max_staging_bytes=None, min_free_bytes=args.min_local_free_bytes, interval_seconds=2.0, usage_roots=(layout.work, layout.resume, layout.logs, layout.cache_root))
    completed, markers = [], []
    warmup_path = layout.resume / "warmup.json"
    if warmup_path.is_file():
        warmup = read_json_object(warmup_path, "RoboOmni warmup marker")
        if warmup.get("fps") != args.fps or warmup.get("encoder_threads_per_worker") != args.encoder_threads_per_worker:
            raise ConversionError("RoboOmni warmup parameters changed")
    for entry in entries:
        task_key, fingerprint = str(entry["task_key"]), str(entry["fingerprint"])
        marker = _read_task_marker(_task_marker_path(layout.resume, task_key), task_key=task_key, fingerprint=fingerprint)
        if marker is not None:
            if entry["task_key"] == entries[0]["task_key"] and not warmup_path.is_file():
                raise ConversionError("first task is committed but warmup marker is missing")
            markers.append(marker)
            completed.append(task_key)
            shutil.rmtree(layout.work / "tasks" / task_key, ignore_errors=True)
            _write_state(layout, catalog_value, completed, None)
            continue
        plan_value = _load_task_plan(layout, entry)
        task = task_from_payload(plan_value, fps=args.fps)
        _write_state(layout, catalog_value, completed, task.task_key)
        try:
            preflight = _preflight_task(task)
            if entry["task_key"] == entries[0]["task_key"] and not warmup_path.is_file():
                _warmup_first_task(
                    task,
                    task_fingerprint=fingerprint,
                    output_uid=args.output_dataset_uid,
                    work_root=layout.work / "warmup",
                    resume_root=layout.resume,
                    fps=args.fps,
                    encoder_threads=args.encoder_threads_per_worker,
                )
            member_results = [_convert_member(task, member, partition_task_index=sum(1 for prior in entries if prior["task_index"] < task.task_index and any(item["partition_name"] == member.partition_name for item in prior.get("members", []))), task_fingerprint=fingerprint, output_uid=args.output_dataset_uid, work_root=layout.work / "tasks" / task.task_key, final_root=layout.final, resume_root=layout.resume, workers=args.workers, upload_workers=args.upload_workers, encoder_threads=args.encoder_threads_per_worker, fps=args.fps, capacity=capacity) for member in task.members]
            marker = _commit_task(task, plan_value, preflight=preflight, member_results=member_results, resume_root=layout.resume)
            markers.append(marker)
            completed.append(task.task_key)
            shutil.rmtree(layout.work / "tasks" / task.task_key, ignore_errors=True)
            _write_state(layout, catalog_value, completed, None)
        except BaseException:
            _write_state(layout, catalog_value, completed, task.task_key)
            raise
    if len(markers) != len(entries):
        raise ConversionError("not all selected RoboOmni tasks have commit markers")
    partition_results = _finalize_collection(markers, output_uid=args.output_dataset_uid, final_root=layout.final, resume_root=layout.resume, work_root=layout.work, fps=args.fps, encoder_threads=args.encoder_threads_per_worker)
    manifest = {"format": "roboomni_le_robot_v3_collection", "schema_version": 1, "output_dataset_uid": args.output_dataset_uid, "catalog_fingerprint": catalog_value["catalog_fingerprint"], "source_root": catalog_value["source_root"], "source_fps": None, "source_timestamps": "absent from inspected RLDS features", "output_fps": args.fps, "task_order": [entry["task_key"] for entry in entries], "tasks": [{"task_key": entry["task_key"], "task_index": entry["task_index"], "instruction": entry["instruction"], "episodes": entry["episode_count"], "frames": entry["frame_count"], "fingerprint": entry["fingerprint"]} for entry in entries], "partitions": partition_results, "mapping_table": catalog_value.get("mapping_table", []), "timebase": catalog_value.get("timebase", {}), "resume": {"task_catalog": "resume/task_catalog.json", "task_commit_markers": [f"resume/tasks/{entry['task_key']}/commit.json" for entry in entries], "committed_tasks_skip_source_preflight": True, "remote_bulk_revalidation_on_committed_task": False}, "local_capacity": {"max_local_temp_bytes": None, "min_local_free_bytes": args.min_local_free_bytes, "peak_staging_bytes": capacity.peak_staging_bytes}}
    atomic_write_json(layout.final / "collection_manifest.json", manifest)
    publish_success(layout.final, fingerprint=str(catalog_value["catalog_fingerprint"]), evidence={"tasks": len(entries), "partitions": partition_results, "output_fps": args.fps})
    shutil.rmtree(layout.work, ignore_errors=True)
    print(json.dumps({"published": str(layout.final), "tasks": len(entries), "partitions": partition_results}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("fps", "workers", "encoder_threads_per_worker", "upload_workers", "min_local_free_bytes"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.max_shards is not None and args.max_shards <= 0:
        raise SystemExit("--max-shards must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise SystemExit("--max-episodes must be positive")
    raw_root = args.raw_root.parent if args.raw_root.name in {"roboomni", "robomni"} else args.raw_root
    output_root = _output_parent(args.output_root, args.output_dataset_uid)
    validate_source_and_output_roots(raw_root, output_root)
    if (args.max_shards or args.max_episodes or args.task or args.start_task or args.max_tasks) and args.output_dataset_uid == "roboomni":
        raise ConversionError("subset/task smoke conversion requires an independent --output-dataset-uid")
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    layout = make_staging_layout(output_root=output_root, local_work_root=args.local_work_root, dataset_uid=args.output_dataset_uid, run_id=run_id, work_dir=args.local_work_root / ".conversion_work" / args.output_dataset_uid, resume_dir=args.local_work_root / ".conversion_resume" / args.output_dataset_uid, logs_dir=args.local_work_root / ".conversion_logs" / args.output_dataset_uid, temp_dir=args.local_work_root / ".conversion_work" / args.output_dataset_uid / "tmp")
    with exclusive_staging_lock(layout.lock):
        return _run(args, raw_root=raw_root, layout=layout, run_id=run_id)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConversionError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print("error:", exc, file=sys.stderr)
        raise SystemExit(1)
