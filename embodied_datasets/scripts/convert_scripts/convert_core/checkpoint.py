"""Format-agnostic durable checkpoint primitives.

Dataset converters decide what a completed unit means (episode, shard, or
partition).  This module owns the safety mechanics shared by Mobile ALOHA,
GR00T, and config-driven converters: deterministic sibling paths, canonical
fingerprints, atomic JSON, and a non-blocking process lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterator
import uuid

from convert_core.errors import ConversionError


RESUME_SCHEMA_VERSION = 1


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"{description} must contain a JSON object: {path}")
    return value


def resume_paths(output: Path) -> tuple[Path, Path, Path]:
    """Return deterministic data/state/lock siblings for an output path."""

    return (
        output.with_name(f".{output.name}.resume"),
        output.with_name(f".{output.name}.resume-state"),
        output.with_name(f".{output.name}.resume.lock"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _path_records_from_plan(plan: Any) -> list[dict[str, Any]]:
    declared = plan.extra.get("source_files")
    if declared is not None:
        return _json_safe(declared)
    paths: set[Path] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Path):
            if value.is_file():
                paths.add(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for episode in plan.episodes:
        visit(episode.extra)
    records = []
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        records.append(
            {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return records


def decoder_records_from_plan(plan: Any) -> list[dict[str, Any]]:
    """Return durable file/repository identities for configured decoders."""

    records: list[dict[str, Any]] = []
    decoder = plan.extra.get("decoder", {})
    if not isinstance(decoder, dict):
        return records
    for key, value in sorted(decoder.items()):
        if not isinstance(value, (str, Path)) or not value:
            continue
        path = Path(value)
        if path.is_file():
            stat = path.stat()
            records.append(
                {
                    "key": key,
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        elif path.is_dir():
            head_path = path / ".git" / "HEAD"
            head = None
            if head_path.is_file():
                head = head_path.read_text(encoding="utf-8").strip()
                if head.startswith("ref: "):
                    ref_path = path / ".git" / head.removeprefix("ref: ")
                    if ref_path.is_file():
                        head = ref_path.read_text(encoding="utf-8").strip()
            records.append({"key": key, "path": str(path), "git_head": head})
    return records


def build_resume_payload(
    plan: Any,
    *,
    reader_format: str,
    conversion_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the format-independent conversion identity used by ``--resume``."""

    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "reader_format": reader_format,
        "output_dataset_uid": plan.dataset_uid,
        "output_path": str(plan.output_path),
        "source_root": str(plan.extra.get("source_root", "")),
        "source_dataset": plan.extra.get("source_dataset"),
        "source_revision": plan.extra.get("source_revision"),
        "source_files": _path_records_from_plan(plan),
        "source_splits": _json_safe(plan.extra.get("source_splits", [])),
        "fps": plan.fps,
        "measured_fps": plan.measured_fps,
        "robot_type": plan.robot_type,
        "features": _json_safe(plan.feature_schema()),
        "field_mapping": _json_safe(plan.extra.get("field_mapping", [])),
        "task_mapping": sorted({episode.instruction for episode in plan.episodes}),
        "episodes": [
            {
                "episode_uid": episode.episode_uid,
                "source_relative_path": episode.source_relative_path,
                "source_split": episode.extra.get("source_split"),
                "source_segment_id": episode.extra.get("source_segment_id"),
                "source_spans": _json_safe(episode.extra.get("source_spans", [])),
                "num_frames": episode.num_frames,
                "instruction": episode.instruction,
                "checkpoint_unit": episode.extra.get(
                    "checkpoint_unit", episode.episode_uid
                ),
            }
            for episode in plan.episodes
        ],
        "partition_rules": _json_safe(plan.extra.get("partition_rules", [])),
        "decoder": _json_safe(plan.extra.get("decoder", {})),
        "decoder_records": decoder_records_from_plan(plan),
        "conversion_options": _json_safe(conversion_options or {}),
    }


def _changed_categories(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    keys = sorted(set(previous) | set(current))
    return [key for key in keys if previous.get(key) != current.get(key)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(
    root: Path, previous: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    prior = {
        str(record.get("relative_path")): record
        for record in (previous or [])
        if isinstance(record, dict)
    }
    records: list[dict[str, Any]] = []
    for directory in (root / "data", root / "videos", root / "meta"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
                old = prior.get(relative, {})
                digest = None
                if (
                    old.get("size") == stat.st_size
                    and old.get("mtime_ns") == stat.st_mtime_ns
                    and isinstance(old.get("sha256"), str)
                ):
                    digest = old["sha256"]
                records.append(
                    {
                        "relative_path": relative,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": digest or _sha256_file(path),
                    }
                )
    return records


def _remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass


@dataclass(frozen=True)
class ResumePosition:
    completed_episodes: int
    completed_frames: int
    checkpoint_unit: str | None
    reused_units: int


class CheckpointManager:
    """Durable unit checkpoints backed by metadata snapshots.

    A writer is finalized at each unit boundary before :meth:`commit` is
    called.  The snapshot makes recovery independent of any parquet/video file
    that was still open when a later unit was interrupted.
    """

    def __init__(
        self,
        output: Path,
        payload: dict[str, Any],
        *,
        data_root: Path | None = None,
        state_root: Path | None = None,
        lock_path: Path | None = None,
        allow_corrupt_rebuild: bool = False,
    ) -> None:
        default_data, default_state, default_lock = resume_paths(output)
        self.data_root = data_root if data_root is not None else default_data
        self.state_root = state_root if state_root is not None else default_state
        self.lock_path = lock_path if lock_path is not None else default_lock
        self.allow_corrupt_rebuild = allow_corrupt_rebuild
        self.payload = payload
        self.fingerprint = canonical_fingerprint(payload)
        self.state_path = self.state_root / "state.json"
        self.markers_root = self.state_root / "markers"
        self.snapshots_root = self.state_root / "snapshots"
        self._state: dict[str, Any] | None = None

    @property
    def prior_inventory(self) -> list[dict[str, Any]]:
        if self._state is None:
            return []
        return list(self._state.get("inventory", []))

    def prepare(self) -> ResumePosition:
        checkpoint_exists = self.data_root.exists()
        if self.state_path.exists():
            state = read_json_object(self.state_path, "resume state")
            previous_payload = state.get("configuration")
            if not isinstance(previous_payload, dict):
                raise ConversionError(f"resume state is missing configuration: {self.state_path}")
            if state.get("resume_schema_version") != RESUME_SCHEMA_VERSION:
                raise ConversionError(
                    f"resume schema changed in {self.state_path}; move the old checkpoint aside"
                )
            if state.get("fingerprint") != self.fingerprint:
                changed = _changed_categories(previous_payload, self.payload)
                raise ConversionError(
                    "resume checkpoint fingerprint does not match this conversion; changed "
                    f"categories: {', '.join(changed) or 'unknown'}. Use the original "
                    f"arguments or move {self.data_root} and {self.state_root} aside"
                )
            self._state = state
        else:
            if checkpoint_exists:
                raise ConversionError(
                    f"resume data exists without its state: {self.data_root}; move it aside explicitly"
                )
            self.state_root.mkdir(parents=True, exist_ok=True)
            self._state = {
                "resume_schema_version": RESUME_SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "configuration": self.payload,
                "completed_episodes": 0,
                "completed_frames": 0,
                "checkpoint_unit": None,
                "completed_units": 0,
                "snapshot": None,
                "marker": None,
                "inventory": [],
                "history": [],
            }
            atomic_write_json(self.state_path, self._state)

        assert self._state is not None
        completed = int(self._state.get("completed_episodes", 0))
        frames = int(self._state.get("completed_frames", 0))
        unit = self._state.get("checkpoint_unit")
        if completed == 0:
            if self.data_root.exists():
                shutil.rmtree(self.data_root)
            return ResumePosition(0, 0, None, 0)
        if not checkpoint_exists:
            raise ConversionError(
                f"resume state records {completed} episodes but data is missing: {self.data_root}"
            )
        try:
            self._restore_snapshot()
        except ConversionError:
            if not self.allow_corrupt_rebuild:
                raise
            return self.rollback_latest()
        return ResumePosition(
            completed,
            frames,
            str(unit),
            int(self._state.get("completed_units", 0)),
        )

    def _restore_snapshot(self) -> None:
        assert self._state is not None
        snapshot_value = self._state.get("snapshot")
        marker_value = self._state.get("marker")
        if not isinstance(snapshot_value, str) or not isinstance(marker_value, str):
            raise ConversionError(f"resume state has no committed snapshot: {self.state_path}")
        snapshot = self.state_root / snapshot_value
        marker_path = self.state_root / marker_value
        marker = read_json_object(marker_path, "checkpoint marker")
        expected = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "checkpoint_unit": self._state["checkpoint_unit"],
            "completed_episodes": self._state["completed_episodes"],
            "completed_frames": self._state["completed_frames"],
            "inventory": self._state["inventory"],
        }
        if marker != expected:
            raise ConversionError(f"checkpoint marker is corrupt or stale: {marker_path}")
        snapshot_meta = snapshot / "meta"
        if not snapshot_meta.is_dir():
            raise ConversionError(f"checkpoint metadata snapshot is missing: {snapshot_meta}")

        live_meta = self.data_root / "meta"
        live_meta.mkdir(parents=True, exist_ok=True)
        # Nested episode metadata parquet files are immutable after a finalized
        # checkpoint part. Snapshot only mutable root files and use the
        # inventory below to prune files created by an interrupted later part.
        # This avoids recursively copying a growing metadata tree every time.
        for path in snapshot_meta.iterdir():
            if path.is_file():
                shutil.copy2(path, live_meta / path.name)
        allowed = {
            str(record["relative_path"]): record
            for record in self._state["inventory"]
        }
        for directory_name in ("data", "videos", "meta"):
            directory = self.data_root / directory_name
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and path.relative_to(self.data_root).as_posix() not in allowed:
                    path.unlink()
            _remove_empty_directories(directory)
        for relative, record in allowed.items():
            expected_size = int(record["size"])
            path = self.data_root / relative
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ConversionError(
                    f"verified checkpoint file is missing or changed: {path} "
                    f"(expected {expected_size} bytes)"
                )
            expected_sha256 = record.get("sha256")
            # Schema-v1 checkpoints created before content hashes remain
            # readable; the next commit upgrades every legacy record.
            if isinstance(expected_sha256, str) and _sha256_file(path) != expected_sha256:
                raise ConversionError(
                    f"verified checkpoint file checksum changed: {path}"
                )
        images = self.data_root / "images"
        if images.exists():
            shutil.rmtree(images)
        for path in self.data_root.iterdir():
            if path.is_dir() and path.name.startswith("tmp"):
                shutil.rmtree(path)

    def _reset(self) -> ResumePosition:
        if self.data_root.exists():
            shutil.rmtree(self.data_root)
        self._state = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "configuration": self.payload,
            "completed_episodes": 0,
            "completed_frames": 0,
            "checkpoint_unit": None,
            "completed_units": 0,
            "snapshot": None,
            "marker": None,
            "inventory": [],
            "history": [],
        }
        atomic_write_json(self.state_path, self._state)
        return ResumePosition(0, 0, None, 0)

    def rollback_latest(self) -> ResumePosition:
        """Discard the latest checkpoint and restore the newest valid prefix."""

        if self._state is None:
            raise RuntimeError("prepare must be called before rollback_latest")
        history = self._state.get("history", [])
        if not isinstance(history, list):
            return self._reset()
        for history_index in range(len(history) - 2, -1, -1):
            entry = history[history_index]
            if not isinstance(entry, dict):
                continue
            candidate = {
                "resume_schema_version": RESUME_SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "configuration": self.payload,
                **entry,
                "history": history[: history_index + 1],
            }
            self._state = candidate
            try:
                self._restore_snapshot()
            except ConversionError:
                continue
            atomic_write_json(self.state_path, candidate)
            return ResumePosition(
                int(candidate["completed_episodes"]),
                int(candidate["completed_frames"]),
                str(candidate["checkpoint_unit"]),
                int(candidate["completed_units"]),
            )
        return self._reset()

    def commit(
        self,
        *,
        checkpoint_unit: str,
        completed_episodes: int,
        completed_frames: int,
    ) -> None:
        if not (self.data_root / "meta" / "info.json").is_file():
            raise ConversionError("cannot checkpoint before meta/info.json is durable")
        inventory = _inventory(self.data_root, self.prior_inventory)
        identifier = hashlib.sha256(
            f"{checkpoint_unit}\0{completed_episodes}\0{completed_frames}".encode("utf-8")
        ).hexdigest()[:20]
        snapshot = self.snapshots_root / f"{completed_episodes:08d}-{identifier}"
        temporary_snapshot = self.snapshots_root / f".tmp-{uuid.uuid4().hex}"
        temporary_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        snapshot_meta = temporary_snapshot / "meta"
        snapshot_meta.mkdir(parents=True)
        for path in (self.data_root / "meta").iterdir():
            if path.is_file():
                shutil.copy2(path, snapshot_meta / path.name)
        temporary_snapshot.rename(snapshot)

        marker = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "checkpoint_unit": checkpoint_unit,
            "completed_episodes": completed_episodes,
            "completed_frames": completed_frames,
            "inventory": inventory,
        }
        marker_rel = f"markers/{completed_episodes:08d}-{identifier}.json"
        marker_path = self.state_root / marker_rel
        atomic_write_json(marker_path, marker)
        completed_units = int((self._state or {}).get("completed_units", 0)) + 1
        entry = {
            "completed_episodes": completed_episodes,
            "completed_frames": completed_frames,
            "checkpoint_unit": checkpoint_unit,
            "completed_units": completed_units,
            "snapshot": snapshot.relative_to(self.state_root).as_posix(),
            "marker": marker_rel,
            "inventory": inventory,
        }
        previous_history = (self._state or {}).get("history", [])
        if not isinstance(previous_history, list):
            previous_history = []
        self._state = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "configuration": self.payload,
            **entry,
            "history": [*previous_history, entry],
        }
        atomic_write_json(self.state_path, self._state)

    def cleanup_state(self) -> None:
        if self.state_root.exists():
            shutil.rmtree(self.state_root)


@contextmanager
def exclusive_resume_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an advisory lock without waiting for another converter."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(f"another resume process is using {lock_path}") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
