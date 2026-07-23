import pytest

pytest.importorskip("lerobot")

from pathlib import Path


class _TempRegistryEntry:
    """Mirrors run_convert.py's test helper of the same name. Unlike that
    one, this test file cannot do a bare `from common.io import ...` at
    all -- verify_scripts/common/ only has format_checkers.py (no
    io.py/schema.py/paths.py), so every registry/config read-write here
    goes through the aliased `registry_common` module instead.
    """

    def __init__(self, registry_entry, dataset_config):
        import run_verify

        self.dataset_id = registry_entry.id
        self.registry_common = run_verify._load_registry_common()
        self.registry_path = Path(run_verify.__file__).resolve().parents[1] / "datasets_registry.yaml"
        self.dataset_config_path = (
            Path(run_verify.__file__).resolve().parents[1] / "convert_scripts" / "configs" / f"{self.dataset_id}.yaml"
        )
        entries = self.registry_common.io.load_registry(self.registry_path)
        entries.append(registry_entry)
        self.registry_common.io.save_registry(entries, self.registry_path)
        self.registry_common.io.save_dataset_config(dataset_config, self.dataset_config_path)

    def reload_entry(self):
        entries = self.registry_common.io.load_registry(self.registry_path)
        return next(e for e in entries if e.id == self.dataset_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.dataset_config_path.unlink(missing_ok=True)
        entries = self.registry_common.io.load_registry(self.registry_path)
        entries = [e for e in entries if e.id != self.dataset_id]
        self.registry_common.io.save_registry(entries, self.registry_path)


def test_main_verifies_hdf5_dataset_and_updates_registry(tmp_path):
    import uuid

    import h5py
    import run_verify

    dataset_id = f"verify_hdf5_{uuid.uuid4().hex[:8]}"
    registry_common = run_verify._load_registry_common()
    entry = registry_common.schema.RegistryEntry(
        id=dataset_id, name="Verify HDF5 Test", download_status=registry_common.schema.DownloadStatus.COMPLETED
    )
    config = registry_common.schema.DatasetConfig(
        id=dataset_id, name="Verify HDF5 Test", raw_format=registry_common.schema.RawFormat.HDF5
    )

    with _TempRegistryEntry(entry, config) as cfg:
        raw_path = tmp_path / "raw" / dataset_id
        raw_path.mkdir(parents=True)
        with h5py.File(raw_path / "demo.hdf5", "w") as f:
            demo = f.create_group("data").create_group("demo_0")
            demo.create_group("obs")
            demo.create_dataset("actions", data=[[0.0]])

        exit_code = run_verify.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

        assert exit_code == 0
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.integrity_status == registry_common.schema.IntegrityStatus.VERIFIED

    log_path = run_verify.LOGS_DIR / f"{dataset_id}.log"
    assert log_path.exists()
    log_path.unlink()


def test_main_marks_skipped_no_checker_for_unregistered_custom_dataset(tmp_path):
    import uuid

    import run_verify

    dataset_id = f"verify_custom_{uuid.uuid4().hex[:8]}"
    registry_common = run_verify._load_registry_common()
    entry = registry_common.schema.RegistryEntry(
        id=dataset_id, name="Verify Custom Test", download_status=registry_common.schema.DownloadStatus.COMPLETED
    )
    config = registry_common.schema.DatasetConfig(
        id=dataset_id, name="Verify Custom Test", raw_format=registry_common.schema.RawFormat.CUSTOM
    )

    with _TempRegistryEntry(entry, config) as cfg:
        raw_path = tmp_path / "raw" / dataset_id
        raw_path.mkdir(parents=True)
        (raw_path / "some_file.bin").write_bytes(b"data")

        exit_code = run_verify.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

        assert exit_code == 1
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.integrity_status == registry_common.schema.IntegrityStatus.SKIPPED_NO_CHECKER

    log_path = run_verify.LOGS_DIR / f"{dataset_id}.log"
    assert log_path.exists()
    log_path.unlink()


def test_main_refuses_when_dataset_not_eligible(tmp_path):
    import uuid

    import run_verify

    dataset_id = f"verify_not_eligible_{uuid.uuid4().hex[:8]}"
    registry_common = run_verify._load_registry_common()
    entry = registry_common.schema.RegistryEntry(
        id=dataset_id, name="Not Eligible Test", download_status=registry_common.schema.DownloadStatus.NOT_DOWNLOADED
    )
    config = registry_common.schema.DatasetConfig(id=dataset_id, name="Not Eligible Test")

    with _TempRegistryEntry(entry, config):
        exit_code = run_verify.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

    assert exit_code == 1
