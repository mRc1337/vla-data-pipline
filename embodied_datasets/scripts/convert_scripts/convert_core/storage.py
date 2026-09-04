"""Filesystem-aware storage planning and runtime disk guards.

The raw public datasets live on an OSSFS/FUSE mount whose reported capacity
is not useful for local staging.  This module therefore always measures the
filesystem containing the *local target* and never infers capacity from the
source path.  It is intentionally dataset agnostic: readers provide byte
estimates, while this module owns path validation, reserve policy, periodic
checks, and human-readable evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import math
import os
from pathlib import Path
import time
from typing import Iterable, Sequence

from convert_core.errors import ConversionError


GIB = 1024**3
MINIMUM_SAFETY_RESERVE_BYTES = 200 * GIB
DEFAULT_SAFETY_RESERVE_FRACTION = 0.15
TRANSIENT_ACCOUNTING_ERRNOS = frozenset(
    {errno.ENOENT, getattr(errno, "ESTALE", 116)}
)


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ConversionError(f"cannot resolve an existing ancestor for {path}")
        candidate = parent
    return candidate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class FilesystemSnapshot:
    path: str
    device: int
    fragment_size: int
    total_bytes: int
    used_bytes: int
    free_bytes: int
    available_bytes: int
    captured_unix: float

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def filesystem_snapshot(path: Path) -> FilesystemSnapshot:
    ancestor = _existing_ancestor(path)
    try:
        values = os.statvfs(ancestor)
        device = ancestor.stat().st_dev
    except OSError as exc:
        raise ConversionError(f"cannot inspect target filesystem at {ancestor}: {exc}") from exc
    fragment = int(values.f_frsize or values.f_bsize)
    total = int(values.f_blocks) * fragment
    free = int(values.f_bfree) * fragment
    available = int(values.f_bavail) * fragment
    return FilesystemSnapshot(
        path=str(ancestor),
        device=int(device),
        fragment_size=fragment,
        total_bytes=total,
        used_bytes=total - free,
        free_bytes=free,
        available_bytes=available,
        captured_unix=time.time(),
    )


def required_safety_reserve(
    snapshot: FilesystemSnapshot,
    *,
    requested_bytes: int | None = None,
    reserve_fraction: float = DEFAULT_SAFETY_RESERVE_FRACTION,
) -> int:
    if not 0 < reserve_fraction < 1:
        raise ValueError("reserve_fraction must be between zero and one")
    policy_floor = max(
        MINIMUM_SAFETY_RESERVE_BYTES,
        math.ceil(snapshot.total_bytes * reserve_fraction),
    )
    if requested_bytes is None:
        return policy_floor
    if requested_bytes < policy_floor:
        raise ConversionError(
            f"--min-free-bytes={requested_bytes} is below the mandatory reserve "
            f"{policy_floor} bytes for filesystem {snapshot.path}"
        )
    return requested_bytes


def directory_size(path: Path) -> int:
    """Return logical bytes for regular files without following symlinks."""

    try:
        if not path.exists():
            return 0
        if path.is_symlink():
            raise ConversionError(f"storage accounting refuses symbolic-link root: {path}")
        if path.is_file():
            return path.stat().st_size
    except OSError as exc:
        if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
            return 0
        raise ConversionError(f"cannot account storage root {path}: {exc}") from exc
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            # A worker may remove its bounded cache after the coordinator has
            # queued that directory for scanning. OSSFS may report the race as
            # either ENOENT or ESTALE; in both cases that directory contributes
            # no current bytes, just like a vanished entry below.
            if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
                continue
            raise ConversionError(f"cannot account storage under {directory}: {exc}") from exc
        for entry in entries:
            try:
                if entry.is_symlink():
                    raise ConversionError(
                        f"storage accounting refuses symbolic link: {entry.path}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                else:
                    raise ConversionError(
                        f"storage accounting found non-regular entry: {entry.path}"
                    )
            except OSError as exc:
                # Runtime caches and writers publish files by same-directory
                # rename. An entry may legitimately disappear or become stale
                # between ``scandir`` and ``stat`` while accounting runs
                # concurrently; it contributes no current bytes and must not
                # stop dispatch.
                if exc.errno in TRANSIENT_ACCOUNTING_ERRNOS:
                    continue
                raise ConversionError(f"cannot account storage entry {entry.path}: {exc}") from exc
    return total


def validate_local_layout(
    output_path: Path,
    temp_dir: Path,
    *,
    forbidden_roots: Sequence[Path] = (
        Path("/mnt/data"),
        Path("/dev/shm"),
        Path("/tmp"),
    ),
) -> FilesystemSnapshot:
    """Require output/temp on one approved local filesystem.

    Existing ancestors are compared so this remains valid before either path
    is created.  The broad roots forbidden by the conversion specification are
    rejected after lexical normalization and before any write.
    """

    output = output_path.expanduser().absolute()
    temporary = temp_dir.expanduser().absolute()
    for candidate, label in ((output, "output"), (temporary, "temporary")):
        try:
            resolved_candidate = candidate.resolve(strict=False)
        except OSError as exc:
            raise ConversionError(f"cannot resolve {label} path {candidate}: {exc}") from exc
        for root in forbidden_roots:
            normalized_root = root.expanduser().absolute().resolve(strict=False)
            if _is_relative_to(candidate, normalized_root) or _is_relative_to(
                resolved_candidate, normalized_root
            ):
                raise ConversionError(
                    f"{label} path must use confirmed local staging storage, not "
                    f"{normalized_root}: {candidate} (resolved {resolved_candidate})"
                )
    output_snapshot = filesystem_snapshot(output)
    temp_snapshot = filesystem_snapshot(temporary)
    if output_snapshot.device != temp_snapshot.device:
        raise ConversionError(
            "--temp-dir and staging output must be on the same filesystem for "
            f"bounded temporary accounting and atomic publication: {temporary} vs {output}"
        )
    return output_snapshot


def validate_paths_within_root(
    root: Path,
    paths: dict[str, Path],
    *,
    required_root: Path | None = None,
) -> dict[str, Path]:
    """Resolve write paths and require every one to remain below one root.

    ``resolve(strict=False)`` closes lexical ``..`` and symlink escapes while
    still permitting not-yet-created run directories.  Dataset coordinators
    can additionally pin the accepted root to an operator-approved mount.
    """

    resolved_root = root.expanduser().absolute().resolve(strict=False)
    if required_root is not None:
        approved = required_root.expanduser().absolute().resolve(strict=False)
        if resolved_root != approved:
            raise ConversionError(
                f"--output-root must be the approved staging root {approved}, got {resolved_root}"
            )
    result: dict[str, Path] = {}
    for label, value in paths.items():
        resolved = value.expanduser().absolute().resolve(strict=False)
        if not _is_relative_to(resolved, resolved_root):
            raise ConversionError(
                f"{label} must be inside staging root {resolved_root}, got {value} "
                f"(resolved {resolved})"
            )
        result[label] = resolved
    return result


def _nonoverlapping_roots(paths: Iterable[Path]) -> tuple[Path, ...]:
    ordered = sorted(
        {path.expanduser().absolute().resolve(strict=False) for path in paths},
        key=lambda value: (len(value.parts), value.as_posix()),
    )
    result: list[Path] = []
    for path in ordered:
        if not any(_is_relative_to(path, existing) for existing in result):
            result.append(path)
    return tuple(result)


def scoped_size(paths: Iterable[Path]) -> int:
    """Account multiple roots once, dropping descendants of another root."""

    return sum(directory_size(path) for path in _nonoverlapping_roots(paths))


@dataclass(frozen=True)
class StagingQuotaCheck:
    stage: str
    staging_usage_bytes: int
    staging_limit_bytes: int
    inflight_usage_bytes: int
    inflight_limit_bytes: int
    required_staging_bytes: int
    required_inflight_bytes: int
    inflight_units: int
    maximum_inflight_units: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


class StagingQuotaGuard:
    """Quota guard for object-backed staging whose statvfs is not meaningful."""

    def __init__(
        self,
        *,
        staging_roots: Iterable[Path],
        inflight_roots: Iterable[Path],
        max_staging_bytes: int,
        max_inflight_bytes: int,
        max_inflight_units: int,
        interval_seconds: float,
    ) -> None:
        if min(
            max_staging_bytes,
            max_inflight_bytes,
            max_inflight_units,
        ) <= 0:
            raise ValueError("staging and inflight limits must be positive")
        if interval_seconds <= 0:
            raise ValueError("storage check interval must be positive")
        self.staging_roots = _nonoverlapping_roots(staging_roots)
        self.inflight_roots = _nonoverlapping_roots(inflight_roots)
        if not self.staging_roots or not self.inflight_roots:
            raise ValueError("quota guard requires staging and inflight roots")
        self.max_staging_bytes = max_staging_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_inflight_units = max_inflight_units
        self.interval_seconds = interval_seconds
        self._last_check_monotonic = 0.0
        self.last_check: StagingQuotaCheck | None = None

    def check(
        self,
        stage: str,
        *,
        required_staging_bytes: int = 0,
        required_inflight_bytes: int = 0,
        inflight_units: int = 0,
    ) -> StagingQuotaCheck:
        if min(required_staging_bytes, required_inflight_bytes, inflight_units) < 0:
            raise ValueError("quota check quantities cannot be negative")
        staging_usage = scoped_size(self.staging_roots)
        inflight_usage = scoped_size(self.inflight_roots)
        if staging_usage + required_staging_bytes > self.max_staging_bytes:
            raise ConversionError(
                f"staging quota stopped before {stage}: usage={staging_usage}, "
                f"required={required_staging_bytes}, --max-staging-bytes="
                f"{self.max_staging_bytes}; no new unit was started"
            )
        if inflight_usage + required_inflight_bytes > self.max_inflight_bytes:
            raise ConversionError(
                f"inflight quota stopped before {stage}: usage={inflight_usage}, "
                f"required={required_inflight_bytes}, --max-inflight-bytes="
                f"{self.max_inflight_bytes}; verified parts were retained"
            )
        if inflight_units > self.max_inflight_units:
            raise ConversionError(
                f"inflight unit limit stopped before {stage}: {inflight_units} > "
                f"{self.max_inflight_units}"
            )
        result = StagingQuotaCheck(
            stage=stage,
            staging_usage_bytes=staging_usage,
            staging_limit_bytes=self.max_staging_bytes,
            inflight_usage_bytes=inflight_usage,
            inflight_limit_bytes=self.max_inflight_bytes,
            required_staging_bytes=required_staging_bytes,
            required_inflight_bytes=required_inflight_bytes,
            inflight_units=inflight_units,
            maximum_inflight_units=self.max_inflight_units,
        )
        self.last_check = result
        self._last_check_monotonic = time.monotonic()
        return result

    def periodic_check(self, stage: str, **kwargs: int) -> StagingQuotaCheck | None:
        if time.monotonic() - self._last_check_monotonic < self.interval_seconds:
            return None
        return self.check(stage, **kwargs)


def format_staging_quota_check(check: StagingQuotaCheck) -> str:
    return (
        f"storage stage={check.stage} staging={check.staging_usage_bytes}/"
        f"{check.staging_limit_bytes} inflight={check.inflight_usage_bytes}/"
        f"{check.inflight_limit_bytes} required_staging={check.required_staging_bytes} "
        f"required_inflight={check.required_inflight_bytes} units={check.inflight_units}/"
        f"{check.maximum_inflight_units}"
    )


@dataclass(frozen=True)
class StorageEstimate:
    data_parquet_expected_bytes: int
    data_parquet_lower_bytes: int
    data_parquet_upper_bytes: int
    video_expected_bytes: int
    video_lower_bytes: int
    video_upper_bytes: int
    metadata_stats_manifest_sidecar_upper_bytes: int
    checkpoint_state_upper_bytes: int
    maximum_inflight_units_upper_bytes: int
    encoder_and_temp_upper_bytes: int
    final_output_conservative_upper_bytes: int
    existing_output_retained_bytes: int
    normal_peak_excluding_reserve_bytes: int
    overwrite_peak_excluding_reserve_bytes: int
    safety_reserve_bytes: int
    filesystem_available_bytes: int
    normal_capacity_sufficient: bool
    overwrite_capacity_sufficient: bool
    method: str

    @property
    def available_after_reserve_bytes(self) -> int:
        return max(0, self.filesystem_available_bytes - self.safety_reserve_bytes)

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            **asdict(self),
            "available_after_reserve_bytes": self.available_after_reserve_bytes,
        }


def make_storage_estimate(
    *,
    data_expected_bytes: int,
    data_range_bytes: tuple[int, int],
    video_expected_bytes: int,
    video_range_bytes: tuple[int, int],
    metadata_upper_bytes: int,
    checkpoint_state_upper_bytes: int,
    inflight_upper_bytes: int,
    encoder_temp_upper_bytes: int,
    final_upper_bytes: int,
    existing_output_bytes: int,
    snapshot: FilesystemSnapshot,
    safety_reserve_bytes: int,
    method: str,
) -> StorageEstimate:
    values = (
        data_expected_bytes,
        *data_range_bytes,
        video_expected_bytes,
        *video_range_bytes,
        metadata_upper_bytes,
        checkpoint_state_upper_bytes,
        inflight_upper_bytes,
        encoder_temp_upper_bytes,
        final_upper_bytes,
        existing_output_bytes,
        safety_reserve_bytes,
    )
    if any(value < 0 for value in values):
        raise ValueError("storage estimates cannot be negative")
    if data_range_bytes[0] > data_range_bytes[1]:
        raise ValueError("invalid data size range")
    if video_range_bytes[0] > video_range_bytes[1]:
        raise ValueError("invalid video size range")

    # Checkpoint data is the growing final-format output and is not counted a
    # second time.  Only compact state/snapshots and the maximum active units
    # are additional to the final conservative upper bound.
    normal_peak = (
        final_upper_bytes
        + checkpoint_state_upper_bytes
        + inflight_upper_bytes
        + encoder_temp_upper_bytes
    )
    overwrite_peak = existing_output_bytes + normal_peak
    usable = max(0, snapshot.available_bytes - safety_reserve_bytes)
    return StorageEstimate(
        data_parquet_expected_bytes=data_expected_bytes,
        data_parquet_lower_bytes=data_range_bytes[0],
        data_parquet_upper_bytes=data_range_bytes[1],
        video_expected_bytes=video_expected_bytes,
        video_lower_bytes=video_range_bytes[0],
        video_upper_bytes=video_range_bytes[1],
        metadata_stats_manifest_sidecar_upper_bytes=metadata_upper_bytes,
        checkpoint_state_upper_bytes=checkpoint_state_upper_bytes,
        maximum_inflight_units_upper_bytes=inflight_upper_bytes,
        encoder_and_temp_upper_bytes=encoder_temp_upper_bytes,
        final_output_conservative_upper_bytes=final_upper_bytes,
        existing_output_retained_bytes=existing_output_bytes,
        normal_peak_excluding_reserve_bytes=normal_peak,
        overwrite_peak_excluding_reserve_bytes=overwrite_peak,
        safety_reserve_bytes=safety_reserve_bytes,
        filesystem_available_bytes=snapshot.available_bytes,
        normal_capacity_sufficient=normal_peak <= usable,
        overwrite_capacity_sufficient=overwrite_peak <= usable,
        method=method,
    )


@dataclass(frozen=True)
class DiskCheck:
    stage: str
    filesystem_available_bytes: int
    safety_reserve_bytes: int
    current_task_usage_bytes: int
    maximum_task_usage_bytes: int | None
    required_additional_bytes: int
    inflight_units: int
    estimated_remaining_output_bytes: int
    predicted_peak_usage_bytes: int

    def as_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class DiskGuard:
    """Rate-limited runtime checks shared by coordinator and writer hooks."""

    def __init__(
        self,
        target_path: Path,
        *,
        usage_roots: Iterable[Path],
        min_free_bytes: int | None,
        max_local_bytes: int | None,
        interval_seconds: float,
        baseline_usage_bytes: int | None = None,
        enforce_policy_floor: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("disk check interval must be positive")
        if max_local_bytes is not None and max_local_bytes <= 0:
            raise ValueError("max_local_bytes must be positive")
        self.target_path = target_path
        self.usage_roots = tuple(dict.fromkeys(Path(path) for path in usage_roots))
        if not self.usage_roots:
            raise ValueError("DiskGuard requires at least one usage root")
        snapshot = filesystem_snapshot(target_path)
        self.device = snapshot.device
        if enforce_policy_floor:
            self.min_free_bytes = required_safety_reserve(
                snapshot, requested_bytes=min_free_bytes
            )
        else:
            if min_free_bytes is None or min_free_bytes <= 0:
                raise ValueError("min_free_bytes must be positive")
            self.min_free_bytes = min_free_bytes
        self.max_local_bytes = max_local_bytes
        self.interval_seconds = interval_seconds
        if baseline_usage_bytes is not None and baseline_usage_bytes < 0:
            raise ValueError("baseline_usage_bytes cannot be negative")
        self._baseline_usage = (
            self._usage() if baseline_usage_bytes is None else baseline_usage_bytes
        )
        self._last_check_monotonic = 0.0
        self.last_check: DiskCheck | None = None

    def _usage(self) -> int:
        return sum(directory_size(path) for path in self.usage_roots)

    def check(
        self,
        stage: str,
        *,
        required_additional_bytes: int = 0,
        inflight_units: int = 0,
        estimated_remaining_output_bytes: int = 0,
    ) -> DiskCheck:
        if (
            required_additional_bytes < 0
            or inflight_units < 0
            or estimated_remaining_output_bytes < 0
        ):
            raise ValueError("disk check quantities cannot be negative")
        snapshot = filesystem_snapshot(self.target_path)
        if snapshot.device != self.device:
            raise ConversionError(
                f"target filesystem changed during conversion at {self.target_path}"
            )
        absolute_usage = self._usage()
        task_usage = max(0, absolute_usage - self._baseline_usage)
        if snapshot.available_bytes - required_additional_bytes < self.min_free_bytes:
            raise ConversionError(
                f"disk guard stopped before {stage}: available={snapshot.available_bytes}, "
                f"required_additional={required_additional_bytes}, mandatory_free="
                f"{self.min_free_bytes} bytes. No new unit was started; rerun the same "
                "command with --resume after adding local capacity"
            )
        if self.max_local_bytes is not None and (
            task_usage + required_additional_bytes > self.max_local_bytes
        ):
            raise ConversionError(
                f"disk guard stopped before {stage}: task_usage={task_usage}, "
                f"required_additional={required_additional_bytes}, --max-local-bytes="
                f"{self.max_local_bytes}. Verified checkpoints were retained"
            )
        result = DiskCheck(
            stage=stage,
            filesystem_available_bytes=snapshot.available_bytes,
            safety_reserve_bytes=self.min_free_bytes,
            current_task_usage_bytes=task_usage,
            maximum_task_usage_bytes=self.max_local_bytes,
            required_additional_bytes=required_additional_bytes,
            inflight_units=inflight_units,
            estimated_remaining_output_bytes=estimated_remaining_output_bytes,
            predicted_peak_usage_bytes=task_usage + required_additional_bytes,
        )
        self.last_check = result
        self._last_check_monotonic = time.monotonic()
        return result

    def periodic_check(
        self,
        stage: str,
        *,
        required_additional_bytes: int = 0,
        inflight_units: int = 0,
        estimated_remaining_output_bytes: int = 0,
    ) -> DiskCheck | None:
        if time.monotonic() - self._last_check_monotonic < self.interval_seconds:
            return None
        return self.check(
            stage,
            required_additional_bytes=required_additional_bytes,
            inflight_units=inflight_units,
            estimated_remaining_output_bytes=estimated_remaining_output_bytes,
        )


def format_disk_check(check: DiskCheck) -> str:
    maximum = (
        "unbounded" if check.maximum_task_usage_bytes is None else str(check.maximum_task_usage_bytes)
    )
    return (
        f"disk stage={check.stage} available={check.filesystem_available_bytes} "
        f"reserve={check.safety_reserve_bytes} task_usage={check.current_task_usage_bytes}/"
        f"{maximum} required_additional={check.required_additional_bytes} "
        f"estimated_remaining_output={check.estimated_remaining_output_bytes} "
        f"predicted_peak_usage={check.predicted_peak_usage_bytes} "
        f"inflight={check.inflight_units}"
    )
