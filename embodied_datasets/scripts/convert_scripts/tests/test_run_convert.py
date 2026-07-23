import pytest

pytest.importorskip("lerobot")

from common.io import load_registry, save_dataset_config, save_registry


class _TempRegistryEntry:
    """main() resolves `datasets_registry.yaml` and
    `convert_scripts/configs/<id>.yaml` relative to run_convert.py's own
    location, never relative to `--data-root` -- both always live in the
    repo. Exercising main() end-to-end means appending a throwaway uuid'd
    RegistryEntry to the real registry (and writing a real per-dataset
    onboarding config file), then removing both, scoped to that uuid'd id
    so nothing collides with real onboarded datasets.
    """

    def __init__(self, registry_entry, dataset_config):
        import run_convert

        self.dataset_id = registry_entry.id
        self.registry_path = run_convert.REGISTRY_PATH
        self.dataset_config_path = run_convert.CONFIGS_DIR / f"{self.dataset_id}.yaml"
        entries = load_registry(self.registry_path)
        entries.append(registry_entry)
        save_registry(entries, self.registry_path)
        save_dataset_config(dataset_config, self.dataset_config_path)

    def reload_entry(self):
        entries = load_registry(self.registry_path)
        return next(e for e in entries if e.id == self.dataset_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.dataset_config_path.unlink(missing_ok=True)
        entries = load_registry(self.registry_path)
        entries = [e for e in entries if e.id != self.dataset_id]
        save_registry(entries, self.registry_path)


def test_main_refuses_when_integrity_not_verified(tmp_path):
    import uuid

    from common.schema import DatasetConfig, IntegrityStatus, RegistryEntry
    import run_convert

    dataset_id = f"not_verified_{uuid.uuid4().hex[:8]}"
    entry = RegistryEntry(id=dataset_id, name="Not Verified Test", integrity_status=IntegrityStatus.NOT_VERIFIED)
    config = DatasetConfig(id=dataset_id, name="Not Verified Test")

    with _TempRegistryEntry(entry, config):
        exit_code = run_convert.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

    assert exit_code == 1


def test_main_converts_and_updates_registry_on_success(tmp_path, monkeypatch):
    import sys
    import types
    import uuid

    from common.schema import DatasetConfig, IntegrityStatus, RegistryEntry
    from common_convert.report import ConversionReport
    import run_convert

    dataset_id = f"convert_success_{uuid.uuid4().hex[:8]}"
    entry = RegistryEntry(id=dataset_id, name="Convert Success Test", integrity_status=IntegrityStatus.VERIFIED)
    config = DatasetConfig(id=dataset_id, name="Convert Success Test", state_dim=9, action_dim=4)

    fake_module = types.ModuleType(dataset_id)

    def fake_convert(raw_path, output_path, dataset_config):
        import numpy as np

        from shared.episode import Episode
        from shared.lerobot_io import write_lerobot_episodes

        episodes = [
            Episode(
                episode_index=0,
                timestamps=np.arange(5, dtype=np.float64) / 10.0,
                state=np.zeros((5, 9), dtype=np.float32),
                action=np.zeros((5, 4), dtype=np.float32),
            )
        ]
        write_lerobot_episodes(episodes, output_path, fps=10.0, robot_type=dataset_config.id)
        return ConversionReport(num_episodes=1, num_frames=5, warnings=["synthetic fixture, not real data"])

    fake_module.convert = fake_convert
    monkeypatch.setitem(sys.modules, dataset_id, fake_module)

    with _TempRegistryEntry(entry, config) as cfg:
        exit_code = run_convert.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

        assert exit_code == 0
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.convert_status.value == "converted"
        assert reloaded_entry.num_episodes == 1
        assert reloaded_entry.num_frames == 5
        assert reloaded_entry.storage_size_gb is not None

    log_path = run_convert.LOGS_DIR / f"{dataset_id}.log"
    assert log_path.exists()
    assert "synthetic fixture" in log_path.read_text()
    log_path.unlink()


def test_main_marks_failed_when_self_check_fails(tmp_path, monkeypatch):
    import sys
    import types
    import uuid

    from common.schema import DatasetConfig, IntegrityStatus, RegistryEntry
    from common_convert.report import ConversionReport
    import run_convert

    dataset_id = f"convert_fails_selfcheck_{uuid.uuid4().hex[:8]}"
    entry = RegistryEntry(id=dataset_id, name="Self-Check Fail Test", integrity_status=IntegrityStatus.VERIFIED)
    # state_dim=9 declared, but the fake convert() below writes 4-wide
    # state -- check_shape must catch the mismatch and fail the whole
    # self-check.
    config = DatasetConfig(id=dataset_id, name="Self-Check Fail Test", state_dim=9)

    fake_module = types.ModuleType(dataset_id)

    def fake_convert(raw_path, output_path, dataset_config):
        import numpy as np

        from shared.episode import Episode
        from shared.lerobot_io import write_lerobot_episodes

        episodes = [
            Episode(
                episode_index=0,
                timestamps=np.arange(3, dtype=np.float64) / 10.0,
                state=np.zeros((3, 4), dtype=np.float32),
                action=np.zeros((3, 4), dtype=np.float32),
            )
        ]
        write_lerobot_episodes(episodes, output_path, fps=10.0, robot_type=dataset_config.id)
        return ConversionReport(num_episodes=1, num_frames=3)

    fake_module.convert = fake_convert
    monkeypatch.setitem(sys.modules, dataset_id, fake_module)

    with _TempRegistryEntry(entry, config) as cfg:
        exit_code = run_convert.main(["--dataset-id", dataset_id, "--data-root", str(tmp_path)])

        assert exit_code == 1
        reloaded_entry = cfg.reload_entry()
        assert reloaded_entry.convert_status.value == "failed"

    log_path = run_convert.LOGS_DIR / f"{dataset_id}.log"
    assert log_path.exists()
    assert "state width 4" in log_path.read_text()
    log_path.unlink()
