from pathlib import Path
import errno

import pytest

from convert_core.errors import ConversionError
from convert_core.storage import (
    GIB,
    FilesystemSnapshot,
    DiskGuard,
    StagingQuotaGuard,
    directory_size,
    make_storage_estimate,
    required_safety_reserve,
    scoped_size,
    validate_local_layout,
    validate_paths_within_root,
)


def _snapshot(*, total: int = 1000 * GIB, available: int = 500 * GIB):
    return FilesystemSnapshot(
        path="/local",
        device=7,
        fragment_size=4096,
        total_bytes=total,
        used_bytes=total - available,
        free_bytes=available,
        available_bytes=available,
        captured_unix=1.0,
    )


def test_reserve_is_maximum_of_200_gib_and_fifteen_percent():
    assert required_safety_reserve(_snapshot(total=1000 * GIB)) == 200 * GIB
    assert required_safety_reserve(_snapshot(total=2000 * GIB)) == 300 * GIB
    with pytest.raises(ConversionError, match="below the mandatory reserve"):
        required_safety_reserve(_snapshot(), requested_bytes=199 * GIB)


def test_storage_estimate_counts_old_output_only_for_overwrite():
    estimate = make_storage_estimate(
        data_expected_bytes=10,
        data_range_bytes=(8, 12),
        video_expected_bytes=20,
        video_range_bytes=(10, 40),
        metadata_upper_bytes=2,
        checkpoint_state_upper_bytes=3,
        inflight_upper_bytes=4,
        encoder_temp_upper_bytes=5,
        final_upper_bytes=54,
        existing_output_bytes=100,
        snapshot=_snapshot(available=300 * GIB),
        safety_reserve_bytes=200 * GIB,
        method="fixture",
    )
    assert estimate.normal_peak_excluding_reserve_bytes == 66
    assert estimate.overwrite_peak_excluding_reserve_bytes == 166


def test_layout_rejects_forbidden_paths_and_requires_same_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    with pytest.raises(ConversionError, match="confirmed local staging storage"):
        validate_local_layout(
            forbidden / "out",
            forbidden / "temp",
            forbidden_roots=(forbidden,),
        )

    first = _snapshot()
    second = FilesystemSnapshot(**{**first.as_dict(), "device": 8})
    calls = iter((first, second))
    monkeypatch.setattr("convert_core.storage.filesystem_snapshot", lambda _path: next(calls))
    with pytest.raises(ConversionError, match="same filesystem"):
        validate_local_layout(
            tmp_path / "out", tmp_path / "temp", forbidden_roots=()
        )


def test_directory_accounting_refuses_symlinks(tmp_path: Path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "data.bin").write_bytes(b"12345")
    assert directory_size(payload) == 5
    (payload / "link").symlink_to(payload / "data.bin")
    with pytest.raises(ConversionError, match="symbolic link"):
        directory_size(payload)


@pytest.mark.parametrize(
    "error_number", [errno.ENOENT, getattr(errno, "ESTALE", 116)]
)
def test_directory_accounting_tolerates_concurrent_cache_removal_or_stale_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int
):
    cache = tmp_path / "worker-cache"
    cache.mkdir()

    def vanished(_path: Path):
        raise OSError(error_number, "cache changed concurrently", str(cache))

    monkeypatch.setattr("convert_core.storage.os.scandir", vanished)
    assert directory_size(cache) == 0


def test_directory_accounting_rejects_nontransient_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "worker-cache"
    cache.mkdir()

    def failed(_path: Path):
        raise OSError(errno.EIO, "backend I/O failure", str(cache))

    monkeypatch.setattr("convert_core.storage.os.scandir", failed)
    with pytest.raises(ConversionError, match="backend I/O failure"):
        directory_size(cache)


def test_layout_cannot_bypass_forbidden_root_through_symlink(tmp_path: Path):
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    link = tmp_path / "local-looking"
    link.symlink_to(forbidden, target_is_directory=True)
    with pytest.raises(ConversionError, match="resolved"):
        validate_local_layout(
            link / "output",
            link / "temp",
            forbidden_roots=(forbidden,),
        )


def test_runtime_guard_stops_before_low_free_or_local_budget_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = _snapshot(available=250 * GIB)
    monkeypatch.setattr("convert_core.storage.filesystem_snapshot", lambda _path: snapshot)
    usage = {"bytes": 0}
    monkeypatch.setattr(
        "convert_core.storage.directory_size", lambda _path: usage["bytes"]
    )
    guard = DiskGuard(
        tmp_path,
        usage_roots=(tmp_path / "task",),
        min_free_bytes=200 * GIB,
        max_local_bytes=10 * GIB,
        interval_seconds=1,
        baseline_usage_bytes=0,
    )
    with pytest.raises(ConversionError, match="mandatory_free"):
        guard.check("new episode", required_additional_bytes=51 * GIB)

    usage["bytes"] = 9 * GIB
    with pytest.raises(ConversionError, match="--max-local-bytes"):
        guard.check("new episode", required_additional_bytes=2 * GIB)

    usage["bytes"] = 1 * GIB
    check = guard.check(
        "periodic",
        required_additional_bytes=2 * GIB,
        estimated_remaining_output_bytes=1 * GIB,
        inflight_units=1,
    )
    assert check.predicted_peak_usage_bytes == 3 * GIB
    assert check.estimated_remaining_output_bytes == 1 * GIB


def test_staging_paths_reject_escape_and_symlink_escape(tmp_path: Path):
    root = tmp_path / "staging"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    with pytest.raises(ConversionError, match="inside staging root"):
        validate_paths_within_root(root, {"work": outside / "work"})

    link = root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConversionError, match="inside staging root"):
        validate_paths_within_root(root, {"work": link / "work"})

    resolved = validate_paths_within_root(
        root,
        {"work": root / ".conversion_work" / "dataset"},
        required_root=root,
    )
    assert resolved["work"].is_relative_to(root)


def test_scoped_size_deduplicates_nested_roots(tmp_path: Path):
    root = tmp_path / "root"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "a.bin").write_bytes(b"123")
    (child / "b.bin").write_bytes(b"4567")
    assert scoped_size((root, child)) == 7


def test_staging_quota_guard_enforces_total_inflight_and_unit_limits(tmp_path: Path):
    staging = tmp_path / "staging"
    work = staging / "work"
    work.mkdir(parents=True)
    (staging / "final.bin").write_bytes(b"12345")
    (work / "active.bin").write_bytes(b"123")
    guard = StagingQuotaGuard(
        staging_roots=(staging,),
        inflight_roots=(work,),
        max_staging_bytes=10,
        max_inflight_bytes=5,
        max_inflight_units=2,
        interval_seconds=1,
    )

    with pytest.raises(ConversionError, match="staging quota"):
        guard.check("next part", required_staging_bytes=3)
    with pytest.raises(ConversionError, match="inflight quota"):
        guard.check("next part", required_inflight_bytes=3)
    with pytest.raises(ConversionError, match="inflight unit limit"):
        guard.check("dispatch", inflight_units=3)
    check = guard.check("safe", inflight_units=2)
    assert check.staging_usage_bytes == 8
    assert check.inflight_usage_bytes == 3
