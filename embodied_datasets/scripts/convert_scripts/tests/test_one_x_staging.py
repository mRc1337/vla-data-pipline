from pathlib import Path

import pytest

from convert_core.errors import ConversionError
from convert_core.staging import (
    RUNTIME_ENVIRONMENT_KEYS,
    StagingCapacityGuard,
    configure_runtime_environment,
    create_incomplete_output,
    exclusive_staging_lock,
    make_staging_layout,
    publish_success,
    validate_contained_path,
    validate_no_runtime_paths_outside_root,
    validate_source_and_output_roots,
)


def test_layout_uses_required_dataset_scoped_directories(tmp_path: Path):
    root = tmp_path / "lerobot_v3_0"
    local = tmp_path / "local"
    layout = make_staging_layout(
        output_root=root,
        local_work_root=local,
        dataset_uid="1x_world_model_dataset",
        run_id="run-1",
    )
    assert layout.final == root / "1x_world_model_dataset"
    assert layout.local_root == local
    assert layout.work == local / ".conversion_work/1x_world_model_dataset/run-1"
    assert layout.resume == local / ".conversion_resume/1x_world_model_dataset"
    assert layout.logs == local / ".conversion_logs/1x_world_model_dataset"
    assert layout.lock == root / ".conversion_locks/1x_world_model_dataset.lock"


def test_layout_rejects_lexical_and_symlink_escape(tmp_path: Path):
    root = tmp_path / "root"
    local = tmp_path / "local"
    root.mkdir()
    with pytest.raises(ConversionError, match="inside staging root"):
        make_staging_layout(
            output_root=root,
            local_work_root=local,
            dataset_uid="dataset",
            run_id="run",
            work_dir=tmp_path / "outside",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    local.mkdir()
    (local / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConversionError, match="symbolic link"):
        validate_contained_path(local / "link" / "child", local, label="work")


def test_source_and_output_must_be_disjoint(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    with pytest.raises(ConversionError, match="disjoint"):
        validate_source_and_output_roots(source, source / "staging")


def test_environment_is_redirected_and_auditable(tmp_path: Path, monkeypatch):
    layout = make_staging_layout(
        output_root=tmp_path / "root",
        local_work_root=tmp_path / "local",
        dataset_uid="dataset",
        run_id="run",
    )
    for key in RUNTIME_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    configured = configure_runtime_environment(layout, create=True)
    assert set(configured) == set(RUNTIME_ENVIRONMENT_KEYS)
    assert all(Path(value).is_dir() for value in configured.values())
    validate_no_runtime_paths_outside_root(layout)


def test_capacity_guard_counts_required_additional_bytes(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "payload").write_bytes(b"12345")
    guard = StagingCapacityGuard(
        root,
        max_staging_bytes=10,
        interval_seconds=1,
    )
    guard.check("small", required_additional_bytes=5)
    with pytest.raises(ConversionError, match="capacity limit"):
        guard.check("too large", required_additional_bytes=6)


def test_capacity_guard_waits_until_upload_deletes_local_bulk(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"123456")
    guard = StagingCapacityGuard(
        root,
        max_staging_bytes=5,
        interval_seconds=1,
    )
    polls = 0

    def emulate_upload_completion() -> None:
        nonlocal polls
        polls += 1
        if polls == 2:
            payload.unlink()

    snapshot = guard.wait_for_capacity(
        "next unit",
        poll_seconds=0.001,
        abort_check=emulate_upload_completion,
    )
    assert polls == 2
    assert snapshot.staging_bytes == 0


def test_marker_publication_and_existing_success_refusal(tmp_path: Path):
    final = tmp_path / "root" / "dataset"
    create_incomplete_output(final, fingerprint="abc", run_id="run")
    assert (final / "_INCOMPLETE").is_file()
    publish_success(final, fingerprint="abc", evidence={"verified": True})
    assert (final / "_SUCCESS").is_file()
    assert not (final / "_INCOMPLETE").exists()
    with pytest.raises(FileExistsError, match="published output"):
        create_incomplete_output(final, fingerprint="abc", run_id="run")


def test_staging_lock_is_nonblocking(tmp_path: Path):
    lock = tmp_path / "root" / ".conversion_locks/dataset.lock"
    with exclusive_staging_lock(lock):
        with pytest.raises(ConversionError, match="another conversion"):
            with exclusive_staging_lock(lock):
                pass
