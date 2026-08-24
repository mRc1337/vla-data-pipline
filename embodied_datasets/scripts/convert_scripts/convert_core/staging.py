"""Local-work/OSS-final layout, capacity control, and publication primitives.

The 1X converter is unusual among the small dataset converters: its durable
output is several terabytes and the target filesystem has expensive small-file
operations.  Bulk conversion state therefore lives on bounded local storage;
only final Parquet/MP4 objects and publication markers live on OSSFS.  Both
roots are independently contained and protected against symlink escapes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import errno
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Iterator, Mapping, Sequence

from convert_core.checkpoint import atomic_write_json
from convert_core.errors import ConversionError


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0"
)
DEFAULT_LOCAL_WORK_ROOT = Path("/home/pai/zxw/1x_world_model_dataset_staging")
DEFAULT_MAX_LOCAL_TEMP_BYTES = 100_000_000_000
DEFAULT_MIN_LOCAL_FREE_BYTES = 200_000_000_000
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
TRANSIENT_ACCOUNTING_ERRNOS = frozenset(
    {errno.ENOENT, getattr(errno, "ESTALE", 116)}
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
    # ``root`` remains the public output root for compatibility with existing
    # run records.  All runtime paths are contained by ``local_root``.
    root: Path
    local_root: Path
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
        # Decoder/model caches are immutable and expensive to download.  Keep
        # one persistent local cache shared by benchmark and production UIDs.
        return self.local_root / ".conversion_cache"

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
    local_work_root: Path = DEFAULT_LOCAL_WORK_ROOT,
    dataset_uid: str,
    run_id: str,
    work_dir: Path | None = None,
    resume_dir: Path | None = None,
    logs_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> StagingLayout:
    root = _absolute(output_root)
    local_root = _absolute(local_work_root)
    if Path(dataset_uid).name != dataset_uid or dataset_uid in {"", ".", ".."}:
        raise ConversionError(f"invalid dataset uid {dataset_uid!r}")
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ConversionError(f"invalid run id {run_id!r}")
    _reject_existing_symlink_components(local_root)
    root_resolved = root.resolve(strict=False)
    local_resolved = local_root.resolve(strict=False)
    if _relative_to(root_resolved, local_resolved) or _relative_to(
        local_resolved, root_resolved
    ):
        raise ConversionError(
            f"local work root and OSS output root must be disjoint: {local_root} vs {root}"
        )
    remote_values = {
        "final": root / dataset_uid,
        "lock": root / ".conversion_locks" / f"{dataset_uid}.lock",
    }
    local_values = {
        "work": work_dir
        or local_root / ".conversion_work" / dataset_uid / run_id,
        "resume": resume_dir or local_root / ".conversion_resume" / dataset_uid,
        "logs": logs_dir or local_root / ".conversion_logs" / dataset_uid,
        "temp": temp_dir
        or (work_dir or local_root / ".conversion_work" / dataset_uid / run_id) / "tmp",
    }
    checked_remote = {
        label: validate_contained_path(Path(path), root, label=label)
        for label, path in remote_values.items()
    }
    checked_local = {
        label: validate_contained_path(Path(path), local_root, label=label)
        for label, path in local_values.items()
    }
    checked = {**checked_remote, **checked_local}
    unique = [checked[key] for key in ("final", "work", "resume", "logs", "temp", "lock")]
    for index, first in enumerate(unique):
        for second in unique[index + 1 :]:
            if first == second:
                raise ConversionError(f"runtime paths must be distinct: {first}")
    return StagingLayout(
        root=root,
        local_root=local_root,
        dataset_uid=dataset_uid,
        run_id=run_id,
        **checked,
    )


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

    try:
        if not root.exists():
            return 0
    except OSError as exc:
        if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
            return 0
        raise ConversionError(f"cannot account staging path {root}: {exc}") from exc
    if root.is_symlink():
        raise ConversionError(f"capacity accounting refuses symlink root: {root}")
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
                continue
            raise ConversionError(f"cannot account staging path {directory}: {exc}") from exc
        for entry in entries:
            if entry.is_symlink():
                raise ConversionError(f"capacity accounting refuses symlink: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                try:
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError as exc:
                    if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
                        continue
                    raise ConversionError(
                        f"cannot account staging path {entry.path}: {exc}"
                    ) from exc
    return total


@dataclass(frozen=True)
class CapacitySnapshot:
    stage: str
    staging_bytes: int
    max_staging_bytes: int | None
    filesystem_available_bytes: int
    min_filesystem_free_bytes: int
    required_additional_bytes: int
    captured_unix: float


class StagingCapacityGuard:
    """Bound and backpressure all growing state on one local filesystem."""

    def __init__(
        self,
        root: Path,
        *,
        max_staging_bytes: int | None = None,
        min_free_bytes: int = 0,
        interval_seconds: float,
        usage_roots: Sequence[Path] | None = None,
    ) -> None:
        if max_staging_bytes is not None and max_staging_bytes <= 0:
            raise ValueError("max_staging_bytes must be positive")
        if interval_seconds <= 0:
            raise ValueError("storage check interval must be positive")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        self.root = root
        self.usage_roots = tuple(usage_roots or (root,))
        for path in self.usage_roots:
            validate_contained_path(path, root, label="capacity-accounting")
        self.max_staging_bytes = max_staging_bytes
        self.min_free_bytes = min_free_bytes
        self.interval_seconds = interval_seconds
        self._last_check = 0.0
        self.last_snapshot: CapacitySnapshot | None = None
        self.peak_staging_bytes = 0

    def _capture(
        self, stage: str, *, required_additional_bytes: int
    ) -> CapacitySnapshot:
        if required_additional_bytes < 0:
            raise ValueError("required_additional_bytes cannot be negative")
        usage = sum(directory_logical_size(path) for path in self.usage_roots)
        self.peak_staging_bytes = max(self.peak_staging_bytes, usage)
        filesystem = os.statvfs(self.root)
        available = filesystem.f_bavail * filesystem.f_frsize
        return CapacitySnapshot(
            stage=stage,
            staging_bytes=usage,
            max_staging_bytes=self.max_staging_bytes,
            filesystem_available_bytes=available,
            min_filesystem_free_bytes=self.min_free_bytes,
            required_additional_bytes=required_additional_bytes,
            captured_unix=time.time(),
        )

    def _violation(self, snapshot: CapacitySnapshot) -> str | None:
        projected = snapshot.staging_bytes + snapshot.required_additional_bytes
        if (
            snapshot.max_staging_bytes is not None
            and projected > snapshot.max_staging_bytes
        ):
            return (
                f"local capacity limit reached before {snapshot.stage}: "
                f"current={snapshot.staging_bytes}, "
                f"required_additional={snapshot.required_additional_bytes}, "
                f"--max-local-temp-bytes={snapshot.max_staging_bytes}"
            )
        remaining = (
            snapshot.filesystem_available_bytes
            - snapshot.required_additional_bytes
        )
        if remaining < snapshot.min_filesystem_free_bytes:
            return (
                f"local free-space reserve reached before {snapshot.stage}: "
                f"available={snapshot.filesystem_available_bytes}, "
                f"required_additional={snapshot.required_additional_bytes}, "
                f"--min-local-free-bytes={snapshot.min_filesystem_free_bytes}"
            )
        return None

    def check(self, stage: str, *, required_additional_bytes: int = 0) -> CapacitySnapshot:
        snapshot = self._capture(
            stage, required_additional_bytes=required_additional_bytes
        )
        violation = self._violation(snapshot)
        if violation is not None:
            raise ConversionError(
                f"{violation}; verified local checkpoints retained"
            )
        self.last_snapshot = snapshot
        self._last_check = time.monotonic()
        return snapshot

    def wait_for_capacity(
        self,
        stage: str,
        *,
        required_additional_bytes: int = 0,
        poll_seconds: float = 1.0,
        abort_check: Callable[[], None] | None = None,
    ) -> CapacitySnapshot:
        """Pause dispatch until upload/deletion releases enough local space."""

        if poll_seconds <= 0:
            raise ValueError("capacity poll interval must be positive")
        while True:
            if abort_check is not None:
                abort_check()
            snapshot = self._capture(
                stage, required_additional_bytes=required_additional_bytes
            )
            if self._violation(snapshot) is None:
                self.last_snapshot = snapshot
                self._last_check = time.monotonic()
                return snapshot
            time.sleep(min(poll_seconds, 5.0))

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
        ("lock", layout.lock),
    ):
        validate_contained_path(path, layout.root, label=label)
    for label, path in (
        ("work", layout.work),
        ("resume", layout.resume),
        ("logs", layout.logs),
        ("temp", layout.temp),
        *(("extra", item) for item in extra_paths),
    ):
        validate_contained_path(path, layout.local_root, label=label)
    for key in RUNTIME_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if value is None:
            raise ConversionError(f"runtime environment {key} is not configured")
        validate_contained_path(Path(value), layout.local_root, label=key)
