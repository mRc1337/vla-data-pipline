"""OSSFS-safe runtime layout, path isolation, and publication primitives.

The 1X converter is unusual among the small dataset converters: its durable
output is several terabytes and the target filesystem has expensive small-file
operations.  This module therefore keeps all runtime paths under one declared
staging root, rejects symlink escapes before creating anything, redirects
third-party caches explicitly, and uses marker-based publication instead of a
whole-directory rename.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator, Mapping, Sequence

from convert_core.checkpoint import atomic_write_json
from convert_core.errors import ConversionError


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0"
)
DEFAULT_RAW_DATASET_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_raw/1x_world_model_dataset"
)
FORBIDDEN_PUBLISH_ROOT = Path("/mnt/data/embodied_datasets/public_datasets")
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
    "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "NUMBA_CACHE_DIR",
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_existing_symlink_components(path: Path) -> None:
    """Reject any existing symlink on the route to ``path``.

    ``Path.resolve`` alone is not sufficient: it can turn a lexical child of
    the staging root into an external path without explaining which component
    escaped.  Walking existing components also prevents a later write through
    an already-present symlink.
    """

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ConversionError(f"runtime path contains a symbolic link: {current}")
        if not current.exists():
            break


def validate_contained_path(path: Path, root: Path, *, label: str) -> Path:
    """Return a normalized path only if both lexical and resolved forms are contained."""

    normalized_root = _absolute(root)
    normalized = _absolute(path)
    _reject_existing_symlink_components(normalized_root)
    _reject_existing_symlink_components(normalized)
    if not _relative_to(normalized, normalized_root):
        raise ConversionError(
            f"{label} path must be inside staging root {normalized_root}: {normalized}"
        )
    try:
        resolved_root = normalized_root.resolve(strict=False)
        resolved = normalized.resolve(strict=False)
    except OSError as exc:
        raise ConversionError(f"cannot resolve {label} path {normalized}: {exc}") from exc
    if not _relative_to(resolved, resolved_root):
        raise ConversionError(
            f"{label} path escapes staging root after resolution: {normalized} -> {resolved}"
        )
    return normalized


def validate_source_and_output_roots(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Require disjoint read-only source and writable staging trees."""

    source = _absolute(source_root)
    output = _absolute(output_root)
    _reject_existing_symlink_components(source)
    _reject_existing_symlink_components(output)
    source_resolved = source.resolve(strict=False)
    output_resolved = output.resolve(strict=False)
    if _relative_to(output_resolved, source_resolved) or _relative_to(
        source_resolved, output_resolved
    ):
        raise ConversionError(
            f"raw source and staging output must be disjoint: {source} vs {output}"
        )
    forbidden = FORBIDDEN_PUBLISH_ROOT.resolve(strict=False)
    if _relative_to(output_resolved, forbidden):
        raise ConversionError(
            f"staging output may not write public_datasets: {output_resolved}"
        )
    return source, output


@dataclass(frozen=True)
class StagingLayout:
    root: Path
    dataset_uid: str
    run_id: str
    final: Path
    work: Path
    resume: Path
    logs: Path
    temp: Path
    lock: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    @property
    def cache_root(self) -> Path:
        return self.work / "cache"

    @property
    def environment(self) -> dict[str, str]:
        return {
            "TMPDIR": str(self.temp),
            "TMP": str(self.temp),
            "TEMP": str(self.temp),
            "XDG_CACHE_HOME": str(self.cache_root / "xdg"),
            "HF_HOME": str(self.cache_root / "huggingface"),
            "HF_DATASETS_CACHE": str(self.cache_root / "huggingface" / "datasets"),
            "TORCH_HOME": str(self.cache_root / "torch"),
            "MPLCONFIGDIR": str(self.cache_root / "matplotlib"),
            "VLA_DATASETS_CACHE_ROOT": str(self.cache_root / "datasets"),
            "CUDA_CACHE_PATH": str(self.cache_root / "cuda"),
            "TORCH_EXTENSIONS_DIR": str(self.cache_root / "torch-extensions"),
            "NUMBA_CACHE_DIR": str(self.cache_root / "numba"),
        }

    def create_runtime_directories(self) -> None:
        for path in (self.work, self.resume, self.logs, self.temp, self.lock.parent):
            path.mkdir(parents=True, exist_ok=True)
        for value in self.environment.values():
            Path(value).mkdir(parents=True, exist_ok=True)


def make_staging_layout(
    *,
    output_root: Path,
    dataset_uid: str,
    run_id: str,
    work_dir: Path | None = None,
    resume_dir: Path | None = None,
    logs_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> StagingLayout:
    root = _absolute(output_root)
    if Path(dataset_uid).name != dataset_uid or dataset_uid in {"", ".", ".."}:
        raise ConversionError(f"invalid dataset uid {dataset_uid!r}")
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ConversionError(f"invalid run id {run_id!r}")
    values = {
        "final": root / dataset_uid,
        "work": work_dir
        or root / ".conversion_work" / dataset_uid / run_id,
        "resume": resume_dir or root / ".conversion_resume" / dataset_uid,
        "logs": logs_dir or root / ".conversion_logs" / dataset_uid,
        "temp": temp_dir
        or (work_dir or root / ".conversion_work" / dataset_uid / run_id) / "tmp",
        "lock": root / ".conversion_locks" / f"{dataset_uid}.lock",
    }
    checked = {
        label: validate_contained_path(Path(path), root, label=label)
        for label, path in values.items()
    }
    unique = [checked[key] for key in ("final", "work", "resume", "logs", "temp", "lock")]
    for index, first in enumerate(unique):
        for second in unique[index + 1 :]:
            if first == second:
                raise ConversionError(f"runtime paths must be distinct: {first}")
    return StagingLayout(root=root, dataset_uid=dataset_uid, run_id=run_id, **checked)


def configure_runtime_environment(layout: StagingLayout, *, create: bool) -> Mapping[str, str]:
    """Redirect every known temporary/cache environment variable into work storage."""

    if create:
        layout.create_runtime_directories()
    for key, value in layout.environment.items():
        os.environ[key] = value
    # ``tempfile`` caches its first selected directory.  Resetting is required
    # when a test or an imported library called gettempdir() before main().
    tempfile.tempdir = str(layout.temp)
    return layout.environment


def directory_logical_size(root: Path) -> int:
    """Account regular-file bytes without following symlinks."""

    if not root.exists():
        return 0
    if root.is_symlink():
        raise ConversionError(f"capacity accounting refuses symlink root: {root}")
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ConversionError(f"cannot account staging path {directory}: {exc}") from exc
        for entry in entries:
            if entry.is_symlink():
                raise ConversionError(f"capacity accounting refuses symlink: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
    return total


@dataclass(frozen=True)
class CapacitySnapshot:
    stage: str
    staging_bytes: int
    max_staging_bytes: int
    filesystem_available_bytes: int
    required_additional_bytes: int
    captured_unix: float


class StagingCapacityGuard:
    """Periodic logical-usage and free-space checks for the shared staging root."""

    def __init__(
        self,
        root: Path,
        *,
        max_staging_bytes: int,
        interval_seconds: float,
        usage_roots: Sequence[Path] | None = None,
    ) -> None:
        if max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive")
        if interval_seconds <= 0:
            raise ValueError("storage check interval must be positive")
        self.root = root
        self.usage_roots = tuple(usage_roots or (root,))
        for path in self.usage_roots:
            validate_contained_path(path, root, label="capacity-accounting")
        self.max_staging_bytes = max_staging_bytes
        self.interval_seconds = interval_seconds
        self._last_check = 0.0
        self.last_snapshot: CapacitySnapshot | None = None

    def check(self, stage: str, *, required_additional_bytes: int = 0) -> CapacitySnapshot:
        if required_additional_bytes < 0:
            raise ValueError("required_additional_bytes cannot be negative")
        usage = sum(directory_logical_size(path) for path in self.usage_roots)
        filesystem = os.statvfs(self.root)
        available = filesystem.f_bavail * filesystem.f_frsize
        projected = usage + required_additional_bytes
        if projected > self.max_staging_bytes:
            raise ConversionError(
                f"staging capacity limit exceeded before {stage}: current={usage}, "
                f"required_additional={required_additional_bytes}, "
                f"--max-staging-bytes={self.max_staging_bytes}; verified checkpoints retained"
            )
        if required_additional_bytes > available:
            raise ConversionError(
                f"staging filesystem has insufficient free space before {stage}: "
                f"available={available}, required_additional={required_additional_bytes}"
            )
        snapshot = CapacitySnapshot(
            stage=stage,
            staging_bytes=usage,
            max_staging_bytes=self.max_staging_bytes,
            filesystem_available_bytes=available,
            required_additional_bytes=required_additional_bytes,
            captured_unix=time.time(),
        )
        self.last_snapshot = snapshot
        self._last_check = time.monotonic()
        return snapshot

    def periodic_check(
        self, stage: str, *, required_additional_bytes: int = 0
    ) -> CapacitySnapshot | None:
        if time.monotonic() - self._last_check < self.interval_seconds:
            return None
        return self.check(stage, required_additional_bytes=required_additional_bytes)


@contextmanager
def exclusive_staging_lock(lock_path: Path) -> Iterator[None]:
    """Acquire the collection lock without waiting and record owner evidence."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(f"another conversion process holds {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def create_incomplete_output(final: Path, *, fingerprint: str, run_id: str) -> None:
    """Create or validate marker state without relying on a directory rename."""

    success = final / "_SUCCESS"
    incomplete = final / "_INCOMPLETE"
    if success.exists():
        raise FileExistsError(f"valid published output already exists: {final}")
    final.mkdir(parents=True, exist_ok=True)
    if incomplete.exists():
        existing = incomplete.read_text(encoding="utf-8").strip().split("\n", 1)[0]
        if existing != fingerprint:
            raise ConversionError(
                f"incomplete output fingerprint changed at {incomplete}; "
                "use the original arguments or an independent output dataset uid"
            )
        return
    incomplete.write_text(f"{fingerprint}\nrun_id={run_id}\n", encoding="utf-8")


def publish_success(
    final: Path,
    *,
    fingerprint: str,
    evidence: Mapping[str, object],
) -> None:
    """Publish `_SUCCESS` durably, then remove `_INCOMPLETE`."""

    incomplete = final / "_INCOMPLETE"
    if not incomplete.is_file():
        raise ConversionError(f"cannot publish without _INCOMPLETE marker: {final}")
    recorded = incomplete.read_text(encoding="utf-8").splitlines()[0]
    if recorded != fingerprint:
        raise ConversionError(f"_INCOMPLETE fingerprint mismatch at {incomplete}")
    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "published_unix": time.time(),
        **dict(evidence),
    }
    atomic_write_json(final / "_SUCCESS", payload)
    incomplete.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_no_runtime_paths_outside_root(
    layout: StagingLayout,
    *,
    extra_paths: Sequence[Path] = (),
) -> None:
    for label, path in (
        ("final", layout.final),
        ("work", layout.work),
        ("resume", layout.resume),
        ("logs", layout.logs),
        ("temp", layout.temp),
        ("lock", layout.lock),
        *(("extra", item) for item in extra_paths),
    ):
        validate_contained_path(path, layout.root, label=label)
    for key in RUNTIME_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if value is None:
            raise ConversionError(f"runtime environment {key} is not configured")
        validate_contained_path(Path(value), layout.root, label=key)
