"""Split durable OSS output from POSIX-local encoder runtime paths.

LeRobot's streaming encoder needs seekable POSIX temporary files, while final
datasets, checkpoints, and logs may live on OSSFS.  This module resolves every
writable path before the first write, keeps durable metadata under the staging
root, redirects temporary/cache writes under a separately validated work tree,
and provides cheap coordinator capacity checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping
import uuid

from convert_core.errors import ConversionError
from convert_core.storage import filesystem_snapshot


RUNTIME_ENVIRONMENT_KEYS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "MPLCONFIGDIR",
    "VLA_DATASETS_CACHE_ROOT",
    "PYTHONPYCACHEPREFIX",
)

NONLOCAL_RUNTIME_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "gcsfuse",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
        "sshfs",
    }
)


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().absolute().resolve(strict=False)
    except OSError as exc:
        raise ConversionError(f"cannot resolve writable path {path}: {exc}") from exc


def _require_below(
    path: Path,
    root: Path,
    label: str,
    *,
    root_label: str = "staging root",
) -> Path:
    candidate = _resolved(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ConversionError(
            f"{label} must be inside {root_label} {root}: {path} resolved to {candidate}"
        ) from exc
    if not relative.parts:
        raise ConversionError(f"{label} cannot be the {root_label} itself: {candidate}")
    return candidate


def _unescape_mount_path(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


def _filesystem_type(path: Path) -> str:
    """Return the longest-prefix mount type reported by Linux mountinfo."""

    candidate = _resolved(path)
    matches: list[tuple[int, str]] = []
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConversionError(f"cannot inspect filesystem for {candidate}: {exc}") from exc
    for line in lines:
        try:
            left, right = line.split(" - ", 1)
            mount_point = Path(_unescape_mount_path(left.split()[4]))
            filesystem = right.split()[0]
            candidate.relative_to(mount_point)
        except (IndexError, ValueError):
            continue
        matches.append((len(mount_point.parts), filesystem))
    if not matches:
        raise ConversionError(f"cannot determine filesystem type for {candidate}")
    return max(matches)[1]


def validate_streaming_runtime_filesystems(layout: "ConversionRuntimeLayout") -> None:
    """Reject work/temp filesystems that cannot support streaming encoders."""

    for label, path in (("--work-dir", layout.work_dir), ("--temp-dir", layout.temp_dir)):
        filesystem = _filesystem_type(path)
        if filesystem.startswith("fuse") or filesystem in NONLOCAL_RUNTIME_FILESYSTEMS:
            raise ConversionError(
                f"{label} must use a local POSIX filesystem for streaming video encoding; "
                f"{path} is on {filesystem}. Final output, checkpoints, and logs may remain "
                "on the staging filesystem."
            )


def make_run_id() -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class ConversionRuntimeLayout:
    output_root: Path
    output_path: Path
    work_dir: Path
    resume_dir: Path
    logs_dir: Path
    temp_dir: Path
    lock_path: Path
    run_id: str

    def writable_paths(self) -> tuple[Path, ...]:
        return (
            self.output_path,
            self.work_dir,
            self.resume_dir,
            self.logs_dir,
            self.temp_dir,
            self.lock_path,
        )


def build_runtime_layout(
    *,
    output_root: Path,
    dataset_uid: str,
    work_dir: Path | None = None,
    resume_dir: Path | None = None,
    logs_dir: Path | None = None,
    temp_dir: Path | None = None,
    run_id: str | None = None,
    required_output_root: Path | None = None,
    lock_name: str = "mimicgen.lock",
) -> ConversionRuntimeLayout:
    """Resolve durable paths under staging and runtime paths under local work."""

    root = _resolved(output_root)
    if required_output_root is not None:
        required = _resolved(required_output_root)
        if root != required:
            raise ConversionError(
                f"--output-root must be the approved staging root {required}, got {root}"
            )
    if not dataset_uid or dataset_uid in {".", ".."} or "/" in dataset_uid:
        raise ConversionError(f"invalid dataset uid for staging layout: {dataset_uid!r}")
    identifier = run_id or make_run_id()
    output = _require_below(root / dataset_uid, root, "final output")
    work = _resolved(work_dir or root / ".conversion_work" / "mimicgen" / identifier)
    resume = _require_below(
        resume_dir or root / ".conversion_resume" / "mimicgen",
        root,
        "resume directory",
    )
    logs = _require_below(
        logs_dir or root / ".conversion_logs" / "mimicgen",
        root,
        "logs directory",
    )
    temporary = _require_below(
        temp_dir or work / "temp",
        work,
        "temporary directory",
        root_label="work directory",
    )
    lock = _require_below(
        root / ".conversion_locks" / lock_name,
        root,
        "lock path",
    )
    directories = (output, work, resume, logs, temporary)
    if len(set(directories)) != len(directories):
        raise ConversionError("output, work, resume, logs, and temp paths must be distinct")
    return ConversionRuntimeLayout(
        output_root=root,
        output_path=output,
        work_dir=work,
        resume_dir=resume,
        logs_dir=logs,
        temp_dir=temporary,
        lock_path=lock,
        run_id=identifier,
    )


def runtime_environment(layout: ConversionRuntimeLayout) -> dict[str, str]:
    cache = layout.work_dir / "cache"
    return {
        "TMPDIR": str(layout.temp_dir),
        "TMP": str(layout.temp_dir),
        "TEMP": str(layout.temp_dir),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
        "TORCH_HOME": str(cache / "torch"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "VLA_DATASETS_CACHE_ROOT": str(cache / "huggingface" / "datasets"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
    }


@contextmanager
def redirected_runtime_environment(
    layout: ConversionRuntimeLayout,
) -> Iterator[dict[str, str]]:
    """Redirect all common temporary/cache writes beneath the local work tree."""

    values = runtime_environment(layout)
    root = layout.work_dir
    for key, value in values.items():
        _require_below(Path(value), root, f"environment variable {key}")
    for path in sorted({Path(value) for value in values.values()}):
        path.mkdir(parents=True, exist_ok=True)
    previous = {key: os.environ.get(key) for key in values}
    previous_tempdir = tempfile.tempdir
    os.environ.update(values)
    tempfile.tempdir = values["TMPDIR"]
    try:
        yield values
    finally:
        tempfile.tempdir = previous_tempdir
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class StagingCapacityCheck:
    stage: str
    current_staging_bytes: int
    inflight_bytes: int
    inflight_units: int
    predicted_staging_bytes: int
    max_staging_bytes: int | None
    max_inflight_bytes: int | None
    max_inflight_units: int | None
    filesystem_available_bytes: int
    captured_unix: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StagingCapacityGuard:
    """Enforce explicit staging/inflight limits without a full-tree rescan."""

    def __init__(
        self,
        output_root: Path,
        *,
        max_staging_bytes: int | None,
        max_inflight_bytes: int | None,
        max_inflight_units: int | None,
        interval_seconds: float,
    ) -> None:
        for label, value in (
            ("max_staging_bytes", max_staging_bytes),
            ("max_inflight_bytes", max_inflight_bytes),
            ("max_inflight_units", max_inflight_units),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive")
        if interval_seconds <= 0:
            raise ValueError("storage check interval must be positive")
        self.output_root = output_root
        self.max_staging_bytes = max_staging_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_inflight_units = max_inflight_units
        self.interval_seconds = interval_seconds
        self._last_check_monotonic = 0.0
        self.last_check: StagingCapacityCheck | None = None

    def check(
        self,
        stage: str,
        *,
        current_staging_bytes: int,
        inflight_bytes: int,
        inflight_units: int,
    ) -> StagingCapacityCheck:
        if min(current_staging_bytes, inflight_bytes, inflight_units) < 0:
            raise ValueError("capacity quantities cannot be negative")
        predicted = current_staging_bytes + inflight_bytes
        if self.max_inflight_units is not None and inflight_units > self.max_inflight_units:
            raise ConversionError(
                f"capacity guard stopped before {stage}: inflight units {inflight_units} "
                f"exceed --max-inflight-units={self.max_inflight_units}"
            )
        if self.max_inflight_bytes is not None and inflight_bytes > self.max_inflight_bytes:
            raise ConversionError(
                f"capacity guard stopped before {stage}: inflight bytes {inflight_bytes} "
                f"exceed --max-inflight-bytes={self.max_inflight_bytes}"
            )
        if self.max_staging_bytes is not None and predicted > self.max_staging_bytes:
            raise ConversionError(
                f"capacity guard stopped before {stage}: predicted staging bytes "
                f"{predicted} exceed --max-staging-bytes={self.max_staging_bytes}"
            )
        snapshot = filesystem_snapshot(self.output_root)
        if inflight_bytes > snapshot.available_bytes:
            raise ConversionError(
                f"capacity guard stopped before {stage}: filesystem available bytes "
                f"{snapshot.available_bytes} are below inflight requirement {inflight_bytes}"
            )
        result = StagingCapacityCheck(
            stage=stage,
            current_staging_bytes=current_staging_bytes,
            inflight_bytes=inflight_bytes,
            inflight_units=inflight_units,
            predicted_staging_bytes=predicted,
            max_staging_bytes=self.max_staging_bytes,
            max_inflight_bytes=self.max_inflight_bytes,
            max_inflight_units=self.max_inflight_units,
            filesystem_available_bytes=snapshot.available_bytes,
            captured_unix=time.time(),
        )
        self.last_check = result
        self._last_check_monotonic = time.monotonic()
        return result

    def periodic_check(
        self,
        stage: str,
        *,
        current_staging_bytes: int,
        inflight_bytes: int,
        inflight_units: int,
    ) -> StagingCapacityCheck | None:
        if time.monotonic() - self._last_check_monotonic < self.interval_seconds:
            return None
        return self.check(
            stage,
            current_staging_bytes=current_staging_bytes,
            inflight_bytes=inflight_bytes,
            inflight_units=inflight_units,
        )


class StructuredRunLog:
    """A low-frequency JSONL log; callers must not emit per-frame records."""

    def __init__(self, layout: ConversionRuntimeLayout) -> None:
        self.path = layout.logs_dir / f"{layout.run_id}.jsonl"

    def write(self, event: str, values: Mapping[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time_unix": time.time(),
            "event": event,
            **dict(values or {}),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
