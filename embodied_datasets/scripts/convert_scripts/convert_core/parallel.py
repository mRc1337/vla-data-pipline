"""Deterministic scheduling and aggregation for isolated conversion units."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import multiprocessing
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Callable, Sequence
import uuid

import numpy as np

from convert_core.checkpoint import atomic_write_json, read_json_object
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError


PARALLEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlanUnitSlice:
    """A contiguous reader-defined unit with coordinator-assigned indices."""

    index: int
    key: str
    episode_start: int
    episode_end: int
    frame_start: int
    frame_end: int
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParallelWorkUnit:
    """One isolated writer job with a coordinator-assigned target and order."""

    index: int
    key: str
    dataset_uid: str
    target_path: str
    episode_start: int
    episode_end: int
    frame_start: int
    frame_end: int
    task_indices: tuple[int, ...]
    weight: int
    estimated_memory_bytes: int
    estimated_temp_bytes: int
    fingerprint: str
    payload: Any


@dataclass(frozen=True)
class ParallelWorkResult:
    index: int
    key: str
    value: Any


@dataclass(frozen=True)
class ParallelRunResult:
    """Results in plan order plus observed worker completion order."""

    results: tuple[ParallelWorkResult, ...]
    completion_order: tuple[str, ...]


@dataclass(frozen=True)
class InflightEstimate:
    workers: int
    memory_bytes: int
    temp_bytes: int


@dataclass(frozen=True)
class PreparedUnits:
    reusable: tuple[ParallelWorkUnit, ...]
    pending: tuple[ParallelWorkUnit, ...]
    repaired_markers: tuple[str, ...]
    discarded_corrupt: tuple[str, ...]


class ParallelWorkError(ConversionError):
    """An isolated work unit failed; its key remains available to resume logic."""

    def __init__(self, unit: ParallelWorkUnit, *, detail: str | None = None):
        self.unit = unit
        self.detail = detail
        message = f"parallel work unit {unit.key!r} failed"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def _checkpoint_unit(episode: EpisodePlan) -> str:
    return str(episode.extra.get("checkpoint_unit", episode.episode_uid))


def split_plan_into_units(
    plan: DatasetConversionPlan,
    *,
    max_episodes_per_unit: int | None = None,
    max_frames_per_unit: int | None = None,
) -> tuple[PlanUnitSlice, ...]:
    """Freeze contiguous checkpoint units and all global indices before dispatch."""

    if not plan.episodes:
        raise ConversionError("parallel conversion cannot split an empty plan")
    if max_episodes_per_unit is not None and max_episodes_per_unit <= 0:
        raise ConversionError("max_episodes_per_unit must be positive")
    if max_frames_per_unit is not None and max_frames_per_unit <= 0:
        raise ConversionError("max_frames_per_unit must be positive")
    tasks = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    task_index = {task: index for index, task in enumerate(tasks)}
    slices: list[PlanUnitSlice] = []
    episode_start = 0
    frame_start = 0
    closed: set[str] = set()
    while episode_start < len(plan.episodes):
        source_key = _checkpoint_unit(plan.episodes[episode_start])
        if source_key in closed:
            raise ConversionError(f"checkpoint unit {source_key!r} is not contiguous")
        source_end = episode_start + 1
        while (
            source_end < len(plan.episodes)
            and _checkpoint_unit(plan.episodes[source_end]) == source_key
        ):
            source_end += 1
        part_start = episode_start
        part_index = 0
        while part_start < source_end:
            part_end = part_start
            part_frames = 0
            while part_end < source_end:
                episode = plan.episodes[part_end]
                if part_end > part_start and (
                    max_episodes_per_unit is not None
                    and part_end - part_start >= max_episodes_per_unit
                    or max_frames_per_unit is not None
                    and part_frames + episode.num_frames > max_frames_per_unit
                ):
                    break
                part_frames += episode.num_frames
                part_end += 1
            split_source = part_start != episode_start or part_end != source_end
            key = (
                f"{source_key}/part_{part_index:05d}" if split_source else source_key
            )
            frame_end = frame_start + part_frames
            episodes = plan.episodes[part_start:part_end]
            slices.append(
                PlanUnitSlice(
                    index=len(slices),
                    key=key,
                    episode_start=part_start,
                    episode_end=part_end,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    task_indices=tuple(
                        task_index[episode.instruction] for episode in episodes
                    ),
                )
            )
            part_start = part_end
            part_index += 1
            frame_start = frame_end
        closed.add(source_key)
        episode_start = source_end
    return tuple(slices)


def isolated_unit_plan(
    plan: DatasetConversionPlan,
    unit: PlanUnitSlice,
    *,
    dataset_uid: str,
    target_path: Path,
) -> DatasetConversionPlan:
    """Create a mini-dataset plan without changing episode field semantics."""

    return replace(
        plan,
        dataset_uid=dataset_uid,
        output_path=target_path,
        episodes=plan.episodes[unit.episode_start : unit.episode_end],
        extra={
            **plan.extra,
            "parallel_unit": {
                "schema_version": PARALLEL_SCHEMA_VERSION,
                "index": unit.index,
                "key": unit.key,
                "global_episode_start": unit.episode_start,
                "global_episode_end": unit.episode_end,
                "global_frame_start": unit.frame_start,
                "global_frame_end": unit.frame_end,
                "global_task_indices": list(unit.task_indices),
            },
        },
    )


def validate_work_units(
    units: Sequence[ParallelWorkUnit], *, require_complete_plan: bool = True
) -> None:
    if not units:
        raise ConversionError("parallel conversion requires at least one work unit")
    actual_indices = [unit.index for unit in units]
    if actual_indices != sorted(actual_indices) or len(actual_indices) != len(set(actual_indices)):
        raise ConversionError("work unit indices must be unique and in plan order")
    expected_indices = list(range(len(units)))
    if require_complete_plan and actual_indices != expected_indices:
        raise ConversionError(
            f"work unit indices are {actual_indices}, expected {expected_indices}"
        )
    keys = [unit.key for unit in units]
    if len(keys) != len(set(keys)):
        raise ConversionError("parallel work unit keys must be unique")
    targets = [unit.target_path for unit in units]
    if len(targets) != len(set(targets)):
        raise ConversionError("parallel work unit target paths must be unique")
    episode_cursor = 0 if require_complete_plan else units[0].episode_start
    frame_cursor = 0 if require_complete_plan else units[0].frame_start
    for unit in units:
        if (
            require_complete_plan
            and unit.episode_start != episode_cursor
            or unit.episode_end <= unit.episode_start
        ):
            raise ConversionError(f"work unit {unit.key!r} does not cover contiguous episodes")
        if (
            require_complete_plan
            and unit.frame_start != frame_cursor
            or unit.frame_end <= unit.frame_start
        ):
            raise ConversionError(f"work unit {unit.key!r} does not cover contiguous frames")
        if len(unit.task_indices) != unit.episode_end - unit.episode_start:
            raise ConversionError(f"work unit {unit.key!r} has invalid task assignments")
        if unit.weight != unit.frame_end - unit.frame_start or unit.weight <= 0:
            raise ConversionError(f"work unit {unit.key!r} has invalid weight")
        if unit.estimated_memory_bytes < 0 or unit.estimated_temp_bytes < 0:
            raise ConversionError(f"work unit {unit.key!r} has a negative resource estimate")
        if not unit.dataset_uid or not unit.fingerprint:
            raise ConversionError(f"work unit {unit.key!r} is missing identity metadata")
        episode_cursor = unit.episode_end
        frame_cursor = unit.frame_end


def inflight_estimate(
    units: Sequence[ParallelWorkUnit], workers: int
) -> InflightEstimate:
    """Return a conservative bound for simultaneously active work units."""

    validate_work_units(units, require_complete_plan=False)
    if workers <= 0:
        raise ConversionError("workers must be positive")
    active = min(workers, len(units))
    memory = sum(
        sorted((unit.estimated_memory_bytes for unit in units), reverse=True)[:active]
    )
    temporary = sum(
        sorted((unit.estimated_temp_bytes for unit in units), reverse=True)[:active]
    )
    return InflightEstimate(active, memory, temporary)


def validate_inflight_budget(
    units: Sequence[ParallelWorkUnit],
    workers: int,
    *,
    memory_budget_bytes: int,
    temp_budget_bytes: int,
    allow_empty: bool = False,
) -> InflightEstimate:
    if not units and allow_empty:
        if workers <= 0:
            raise ConversionError("workers must be positive")
        return InflightEstimate(0, 0, 0)
    estimate = inflight_estimate(units, workers)
    if estimate.memory_bytes > memory_budget_bytes:
        raise ConversionError(
            "parallel worker memory estimate exceeds the inflight budget: "
            f"{estimate.memory_bytes} > {memory_budget_bytes} bytes"
        )
    if estimate.temp_bytes > temp_budget_bytes:
        raise ConversionError(
            "parallel worker temporary-space estimate exceeds the inflight budget: "
            f"{estimate.temp_bytes} > {temp_budget_bytes} bytes"
        )
    return estimate


def _dispatch_order(units: Sequence[ParallelWorkUnit]) -> list[ParallelWorkUnit]:
    # Largest-first is deterministic and avoids a long tail. Aggregation still
    # uses ``index`` and therefore never depends on completion order.
    return sorted(units, key=lambda unit: (-unit.weight, unit.index, unit.key))


def _persistent_worker_loop(
    connection: Any,
    slot: int,
    worker: Callable[[ParallelWorkUnit], Any],
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Serve units over a Pipe without Queue/SemLock `/dev/shm` artifacts."""

    try:
        if initializer is not None:
            initializer(slot, *initargs)
        connection.send(("ready", slot, None))
    except BaseException as exc:
        detail = "".join(traceback.format_exception(exc))
        connection.send(("init_error", slot, detail))
        connection.close()
        return
    try:
        while True:
            unit = connection.recv()
            if unit is None:
                return
            # A Pipe send only proves that the coordinator handed bytes to the
            # kernel; it does not prove this worker has accepted the unit.  The
            # explicit acknowledgement makes the bounded initial frontier
            # deterministic and prevents a fast failure in one slot from
            # terminating another slot before its already-dispatched unit has
            # actually started.
            connection.send(("started", unit.index, None))
            try:
                value = worker(unit)
            except BaseException as exc:
                detail = "".join(traceback.format_exception(exc))
                connection.send(("error", unit.index, detail))
                return
            connection.send(("ok", unit.index, value))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def run_parallel_work_units(
    units: Sequence[ParallelWorkUnit],
    worker: Callable[[ParallelWorkUnit], Any],
    *,
    workers: int,
    on_result: Callable[[ParallelWorkResult], None] | None = None,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    health_check: Callable[[], None] | None = None,
    health_check_interval_seconds: float = 10.0,
    allow_empty: bool = False,
    before_dispatch: Callable[
        [ParallelWorkUnit, tuple[ParallelWorkUnit, ...]], None
    ]
    | None = None,
) -> ParallelRunResult:
    """Run a bounded frontier and stop dispatching new work after a failure."""

    if not units and allow_empty:
        return ParallelRunResult((), ())
    validate_work_units(units, require_complete_plan=False)
    if workers <= 0:
        raise ConversionError("workers must be positive")
    if health_check_interval_seconds <= 0:
        raise ConversionError("health check interval must be positive")
    dispatch = _dispatch_order(units)
    if workers == 1 and initializer is None:
        ordered: dict[int, ParallelWorkResult] = {}
        completion: list[str] = []
        for unit in dispatch:
            if before_dispatch is not None:
                before_dispatch(unit, ())
            try:
                value = worker(unit)
            except BaseException as exc:
                raise ParallelWorkError(
                    unit, detail="".join(traceback.format_exception(exc))
                ) from exc
            result = ParallelWorkResult(unit.index, unit.key, value)
            ordered[unit.index] = result
            completion.append(unit.key)
            if on_result is not None:
                on_result(result)
            if health_check is not None:
                health_check()
        return ParallelRunResult(
            tuple(ordered[index] for index in sorted(ordered)), tuple(completion)
        )

    context = multiprocessing.get_context("spawn")
    max_workers = min(workers, len(units))
    processes: list[Any] = []
    connections: list[Any] = []
    for slot in range(max_workers):
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_persistent_worker_loop,
            args=(child, slot, worker, initializer, initargs),
            name=f"conversion-worker-{slot}",
        )
        process.start()
        child.close()
        processes.append(process)
        connections.append(parent)

    # Do not dispatch until every process has initialized successfully. This
    # makes GPU binding failures coordinator-visible before any output starts.
    waiting_ready = set(connections)
    while waiting_ready:
        readable = wait_connections(tuple(waiting_ready))
        for connection in readable:
            try:
                status, _slot, detail = connection.recv()
            except EOFError as exc:
                status, detail = "init_error", f"worker exited during initialization: {exc}"
            if status != "ready":
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                for item in connections:
                    item.close()
                raise ConversionError(str(detail))
            waiting_ready.remove(connection)

    iterator = iter(dispatch)
    active: dict[Any, ParallelWorkUnit] = {}
    ordered: dict[int, ParallelWorkResult] = {}
    completion: list[str] = []
    failed: tuple[ParallelWorkUnit, BaseException] | None = None
    coordinator_failure: BaseException | None = None
    last_health_check = time.monotonic()

    def submit_next(connection: Any) -> bool:
        nonlocal coordinator_failure
        try:
            unit = next(iterator)
        except StopIteration:
            return False
        try:
            if before_dispatch is not None:
                before_dispatch(unit, tuple(active.values()))
            connection.send(unit)
            status, unit_index, detail = connection.recv()
            if status != "started" or unit_index != unit.index:
                raise ConversionError(
                    f"worker did not acknowledge unit {unit.index}: "
                    f"status={status!r}, index={unit_index!r}, detail={detail!r}"
                )
        except BaseException as exc:
            coordinator_failure = exc
            return False
        active[connection] = unit
        return True

    try:
        for connection in connections:
            submit_next(connection)
            if coordinator_failure is not None:
                break
        while active and failed is None and coordinator_failure is None:
            timeout = None
            if health_check is not None:
                timeout = max(
                    0.0,
                    health_check_interval_seconds
                    - (time.monotonic() - last_health_check),
                )
            readable = wait_connections(tuple(active), timeout=timeout)
            if health_check is not None and (
                not readable
                or time.monotonic() - last_health_check
                >= health_check_interval_seconds
            ):
                try:
                    health_check()
                except BaseException as exc:
                    coordinator_failure = exc
                last_health_check = time.monotonic()
                if coordinator_failure is not None:
                    break
            successful: list[tuple[ParallelWorkUnit, Any]] = []
            freed: list[Any] = []
            for connection in readable:
                unit = active.pop(connection)
                freed.append(connection)
                try:
                    status, unit_index, value = connection.recv()
                except EOFError as exc:
                    failed = (unit, exc)
                    continue
                if unit_index != unit.index:
                    failed = (
                        unit,
                        ConversionError(
                            f"worker returned unit index {unit_index}, expected {unit.index}"
                        ),
                    )
                elif status == "ok":
                    successful.append((unit, value))
                else:
                    failed = (unit, RuntimeError(str(value)))
            for unit, value in sorted(successful, key=lambda item: item[0].index):
                result = ParallelWorkResult(unit.index, unit.key, value)
                ordered[unit.index] = result
                completion.append(unit.key)
                if on_result is not None:
                    try:
                        on_result(result)
                    except BaseException as exc:
                        coordinator_failure = exc
                        break
            if failed is None and coordinator_failure is None:
                for connection in freed:
                    submit_next(connection)
                    if coordinator_failure is not None:
                        break
    finally:
        abort = failed is not None or coordinator_failure is not None
        for process in processes:
            if abort and process.is_alive():
                process.terminate()
        if not abort:
            for connection in connections:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
        for process in processes:
            process.join()
        for connection in connections:
            connection.close()

    if failed is not None:
        unit, exc = failed
        detail = str(exc)
        if isinstance(exc, EOFError):
            exitcodes = [process.exitcode for process in processes]
            detail = f"worker pipe closed; worker_exitcodes={exitcodes}"
        raise ParallelWorkError(unit, detail=detail) from exc
    if coordinator_failure is not None:
        raise coordinator_failure
    return ParallelRunResult(
        tuple(ordered[index] for index in sorted(ordered)), tuple(completion)
    )


def verified_marker_path(unit: ParallelWorkUnit) -> Path:
    target = Path(unit.target_path)
    return target.with_name(f"{target.name}.verified.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _marker_header(unit: ParallelWorkUnit) -> dict[str, Any]:
    return {
        "parallel_schema_version": PARALLEL_SCHEMA_VERSION,
        "fingerprint": unit.fingerprint,
        "unit_index": unit.index,
        "unit_key": unit.key,
        "dataset_uid": unit.dataset_uid,
        "episode_start": unit.episode_start,
        "episode_end": unit.episode_end,
        "frame_start": unit.frame_start,
        "frame_end": unit.frame_end,
        "task_indices": list(unit.task_indices),
    }


def write_verified_unit_marker(unit: ParallelWorkUnit) -> Path:
    target = Path(unit.target_path)
    if not target.is_dir():
        raise ConversionError(f"cannot verify missing work unit output: {target}")
    marker = {**_marker_header(unit), "inventory": _inventory(target)}
    path = verified_marker_path(unit)
    atomic_write_json(path, marker)
    return path


def read_verified_unit_marker(unit: ParallelWorkUnit) -> dict[str, Any]:
    """Read marker identity without re-reading bulk files.

    This is safe only for a freshly returned worker result in the same run.
    Resume paths must call :func:`validate_verified_unit_marker`.
    """

    path = verified_marker_path(unit)
    marker = read_json_object(path, "parallel work unit marker")
    for key, value in _marker_header(unit).items():
        if marker.get(key) != value:
            raise ConversionError(f"parallel marker is stale or corrupt at {path}: {key}")
    inventory = marker.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ConversionError(f"parallel marker has no inventory at {path}")
    return marker


def validate_verified_unit_marker(unit: ParallelWorkUnit) -> dict[str, Any]:
    marker = read_verified_unit_marker(unit)
    target = Path(unit.target_path)
    if marker.get("inventory") != _inventory(target):
        raise ConversionError(
            f"parallel marker inventory is stale or corrupt at {verified_marker_path(unit)}"
        )
    return marker


def prepare_work_units(
    units: Sequence[ParallelWorkUnit],
    validate_output: Callable[[ParallelWorkUnit], None],
    *,
    require_complete_plan: bool = True,
) -> PreparedUnits:
    """Reuse only validated units; repair marker-only damage without rebuilding."""

    validate_work_units(units, require_complete_plan=require_complete_plan)
    reusable: list[ParallelWorkUnit] = []
    pending: list[ParallelWorkUnit] = []
    repaired: list[str] = []
    discarded: list[str] = []
    for unit in units:
        target = Path(unit.target_path)
        marker = verified_marker_path(unit)
        if not target.exists():
            marker.unlink(missing_ok=True)
            pending.append(unit)
            continue
        try:
            validate_output(unit)
        except BaseException:
            shutil.rmtree(target)
            marker.unlink(missing_ok=True)
            discarded.append(unit.key)
            pending.append(unit)
            continue
        if not marker.exists():
            write_verified_unit_marker(unit)
            repaired.append(unit.key)
            reusable.append(unit)
            continue
        try:
            validate_verified_unit_marker(unit)
        except BaseException:
            # A present but invalid marker means that either the verified files
            # or the conversion identity changed. Rebuild only this unit.
            shutil.rmtree(target)
            marker.unlink(missing_ok=True)
            discarded.append(unit.key)
            pending.append(unit)
            continue
        reusable.append(unit)
    return PreparedUnits(
        tuple(reusable), tuple(pending), tuple(repaired), tuple(discarded)
    )


def _replace_episode_stat(
    values: dict[str, list[Any]],
    row_index: int,
    feature: str,
    data: np.ndarray,
) -> None:
    from lerobot.datasets.compute_stats import get_feature_stats

    prefix = f"stats/{feature}/"
    stats = get_feature_stats(data, axis=0, keepdims=True)
    for stat, result in stats.items():
        column = f"{prefix}{stat}"
        if column not in values:
            raise ConversionError(f"aggregated episode metadata is missing {column}")
        values[column][row_index] = np.asarray(result).tolist()


def _repair_aggregated_index_stats(
    plan: DatasetConversionPlan,
    root: Path,
) -> None:
    """Repair a LeRobot 0.6 aggregation bug in generated-index statistics."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise ConversionError(f"aggregated dataset has no episode metadata: {root}")
    task_names = list(dict.fromkeys(episode.instruction for episode in plan.episodes))
    task_indices = {task: index for index, task in enumerate(task_names)}
    frame_starts: list[int] = []
    cursor = 0
    for episode in plan.episodes:
        frame_starts.append(cursor)
        cursor += episode.num_frames

    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    episode_cursor = 0
    for path in paths:
        table = pq.read_table(path)
        values = {name: table[name].to_pylist() for name in table.column_names}
        for local_index in range(table.num_rows):
            if episode_cursor >= len(plan.episodes):
                raise ConversionError("aggregated metadata has too many episodes")
            if int(values["episode_index"][local_index]) != episode_cursor:
                raise ConversionError(
                    f"aggregated episode order changed at {episode_cursor}: "
                    f"got {values['episode_index'][local_index]}"
                )
            episode = plan.episodes[episode_cursor]
            length = episode.num_frames
            _replace_episode_stat(
                values,
                local_index,
                "episode_index",
                np.full(length, episode_cursor, dtype=np.int64),
            )
            _replace_episode_stat(
                values,
                local_index,
                "index",
                np.arange(
                    frame_starts[episode_cursor],
                    frame_starts[episode_cursor] + length,
                    dtype=np.int64,
                ),
            )
            _replace_episode_stat(
                values,
                local_index,
                "task_index",
                np.full(
                    length,
                    task_indices[episode.instruction],
                    dtype=np.int64,
                ),
            )
            stats: dict[str, dict[str, np.ndarray]] = {}
            for column in table.column_names:
                if column.startswith("stats/"):
                    feature, stat = column[len("stats/") :].rsplit("/", 1)
                    stats.setdefault(feature, {})[stat] = np.asarray(
                        values[column][local_index]
                    )
            all_stats.append(stats)
            episode_cursor += 1
        temporary = path.with_name(f".{path.name}.repair-{uuid.uuid4().hex}")
        arrays = [
            pa.array(values[field.name], type=field.type) for field in table.schema
        ]
        repaired = pa.Table.from_arrays(arrays, schema=table.schema)
        pq.write_table(repaired, temporary)
        temporary.replace(path)
    if episode_cursor != len(plan.episodes):
        raise ConversionError(
            f"aggregated metadata has {episode_cursor} episodes, expected {len(plan.episodes)}"
        )
    write_stats(aggregate_stats(all_stats), root)


def _repair_aggregated_data_schema(plan: DatasetConversionPlan, root: Path) -> None:
    """Restore fixed vector types lost by LeRobot 0.6's pandas aggregation."""

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.feature_utils import get_hf_features_from_features

    hf_schema = get_hf_features_from_features(plan.feature_schema()).arrow_schema
    by_name = {feature.feature_key: feature for feature in plan.vector_features}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        changed = False
        for key, feature in by_name.items():
            if key not in table.column_names:
                raise ConversionError(f"aggregated data is missing planned column {key!r}")
            shape = feature.resolved_shape
            if shape == (1,):
                expected_type = pa.from_numpy_dtype(np.dtype(feature.dtype))
            else:
                expected_type = hf_schema.field(key).type
            column_index = table.schema.get_field_index(key)
            if table.schema.field(column_index).type == expected_type:
                continue
            values = table[key].to_pylist()
            array = pa.array(values, type=expected_type)
            field = pa.field(
                key,
                expected_type,
                nullable=table.schema.field(column_index).nullable,
                metadata=table.schema.field(column_index).metadata,
            )
            table = table.set_column(column_index, field, array)
            changed = True
        if changed:
            temporary = path.with_name(f".{path.name}.schema-{uuid.uuid4().hex}")
            pq.write_table(table, temporary)
            temporary.replace(path)


def aggregate_lerobot_work_units(
    plan: DatasetConversionPlan,
    units: Sequence[ParallelWorkUnit],
    output_path: Path,
    *,
    reader_format: str,
    parallel_evidence: dict[str, Any],
) -> Path:
    """Merge verified unit datasets strictly in coordinator plan order."""

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from convert_core.lerobot_writer import (
        _local_datasets_cache,
        build_manifest,
        publish_temporary_output,
        validate_video_files,
        validate_written_dataset,
    )

    validate_work_units(units)
    for unit in units:
        validate_verified_unit_marker(unit)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.aggregate-incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        with _local_datasets_cache(temporary):
            datasets = [
                LeRobotDataset(repo_id=unit.dataset_uid, root=Path(unit.target_path))
                for unit in units
            ]
            merge_datasets(
                datasets,
                output_repo_id=plan.dataset_uid,
                output_dir=temporary,
                concatenate_videos=True,
                concatenate_data=True,
            )
        _repair_aggregated_data_schema(plan, temporary)
        _repair_aggregated_index_stats(plan, temporary)
        validate_written_dataset(plan, temporary)
        video_evidence = validate_video_files(
            plan, temporary, expected_frames=plan.num_frames
        )
        manifest = build_manifest(plan, reader_format=reader_format)
        manifest.update(
            {
                "num_video_files": sum(len(rows) for rows in video_evidence.values()),
                "video_validation": video_evidence,
                "parallel": {
                    "schema_version": PARALLEL_SCHEMA_VERSION,
                    "aggregation_order": [unit.key for unit in units],
                    "units": [
                        {
                            "index": unit.index,
                            "key": unit.key,
                            "episode_start": unit.episode_start,
                            "episode_end": unit.episode_end,
                            "frame_start": unit.frame_start,
                            "frame_end": unit.frame_end,
                            "task_indices": list(unit.task_indices),
                        }
                        for unit in units
                    ],
                    **parallel_evidence,
                },
            }
        )
        atomic_write_json(temporary / "conversion_manifest.json", manifest)
        publish_temporary_output(temporary, output_path, overwrite=False)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_path
