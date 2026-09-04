from __future__ import annotations

import os
from pathlib import Path

import pytest

from convert_core.errors import ConversionError
from convert_core.runtime_layout import (
    RUNTIME_ENVIRONMENT_KEYS,
    StagingCapacityGuard,
    build_runtime_layout,
    redirected_runtime_environment,
    validate_streaming_runtime_filesystems,
)


def test_runtime_layout_splits_local_runtime_from_durable_staging(tmp_path: Path):
    root = tmp_path / "staging" / "lerobot_v3_0"
    work = tmp_path / "local" / "mimicgen"
    layout = build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        work_dir=work,
        run_id="test-run",
        required_output_root=root,
    )

    for path in (layout.output_path, layout.resume_dir, layout.logs_dir, layout.lock_path):
        path.resolve(strict=False).relative_to(root.resolve())
    layout.work_dir.resolve(strict=False).relative_to(work.resolve())
    layout.temp_dir.resolve(strict=False).relative_to(work.resolve())
    assert layout.output_path == root / "mimicgen"
    assert layout.work_dir == work
    assert layout.temp_dir == work / "temp"
    assert layout.resume_dir == root / ".conversion_resume" / "mimicgen"
    assert layout.lock_path == root / ".conversion_locks" / "mimicgen.lock"


def test_runtime_layout_rejects_unapproved_output_root(tmp_path: Path):
    approved = tmp_path / "approved" / "lerobot_v3_0"
    with pytest.raises(ConversionError, match="must be the approved staging root"):
        build_runtime_layout(
            output_root=tmp_path / "other" / "lerobot_v3_0",
            dataset_uid="mimicgen",
            required_output_root=approved,
        )


@pytest.mark.parametrize("field", ["resume_dir", "logs_dir"])
def test_runtime_layout_rejects_path_escape(tmp_path: Path, field: str):
    root = tmp_path / "staging"
    with pytest.raises(ConversionError, match="must be inside staging root"):
        build_runtime_layout(
            output_root=root,
            dataset_uid="mimicgen",
            run_id="test-run",
            **{field: tmp_path / "outside" / field},
        )


def test_runtime_layout_rejects_temp_escape_from_local_work(tmp_path: Path):
    root = tmp_path / "staging"
    work = tmp_path / "local" / "work"
    with pytest.raises(ConversionError, match="must be inside work directory"):
        build_runtime_layout(
            output_root=root,
            dataset_uid="mimicgen",
            work_dir=work,
            temp_dir=tmp_path / "other-temp",
            run_id="test-run",
        )


def test_runtime_layout_rejects_temp_symlink_escape(tmp_path: Path):
    root = tmp_path / "staging"
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (work / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConversionError, match="must be inside work directory"):
        build_runtime_layout(
            output_root=root,
            dataset_uid="mimicgen",
            work_dir=work,
            temp_dir=work / "linked" / "temp",
            run_id="test-run",
        )


def test_runtime_environment_is_redirected_and_restored(tmp_path: Path):
    root = tmp_path / "staging"
    work = tmp_path / "local" / "work"
    layout = build_runtime_layout(
        output_root=root,
        dataset_uid="mimicgen",
        work_dir=work,
        run_id="test-run",
    )
    previous = {key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS}

    with redirected_runtime_environment(layout) as values:
        for key, value in values.items():
            assert os.environ[key] == value
            Path(value).resolve().relative_to(work.resolve())
            assert Path(value).is_dir()

    assert {key: os.environ.get(key) for key in RUNTIME_ENVIRONMENT_KEYS} == previous


def test_streaming_runtime_rejects_fuse_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    layout = build_runtime_layout(
        output_root=tmp_path / "staging",
        dataset_uid="mimicgen",
        work_dir=tmp_path / "work",
    )
    monkeypatch.setattr("convert_core.runtime_layout._filesystem_type", lambda _path: "fuse.ossfs2")

    with pytest.raises(ConversionError, match="local POSIX filesystem"):
        validate_streaming_runtime_filesystems(layout)


def test_capacity_guard_enforces_all_limits_and_periodic_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    guard = StagingCapacityGuard(
        tmp_path,
        max_staging_bytes=100,
        max_inflight_bytes=40,
        max_inflight_units=2,
        interval_seconds=10,
    )
    now = [100.0]
    monkeypatch.setattr("convert_core.runtime_layout.time.monotonic", lambda: now[0])

    first = guard.check(
        "initial", current_staging_bytes=50, inflight_bytes=40, inflight_units=2
    )
    assert first.predicted_staging_bytes == 90
    assert guard.periodic_check(
        "too-soon", current_staging_bytes=50, inflight_bytes=0, inflight_units=0
    ) is None
    now[0] += 10
    assert guard.periodic_check(
        "periodic", current_staging_bytes=50, inflight_bytes=0, inflight_units=0
    ) is not None

    with pytest.raises(ConversionError, match="inflight units"):
        guard.check("units", current_staging_bytes=0, inflight_bytes=0, inflight_units=3)
    with pytest.raises(ConversionError, match="inflight bytes"):
        guard.check("bytes", current_staging_bytes=0, inflight_bytes=41, inflight_units=1)
    with pytest.raises(ConversionError, match="predicted staging bytes"):
        guard.check("staging", current_staging_bytes=70, inflight_bytes=31, inflight_units=1)
