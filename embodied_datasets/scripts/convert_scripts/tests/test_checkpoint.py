import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from convert_core.checkpoint import (
    CheckpointManager,
    build_resume_payload,
    canonical_fingerprint,
    exclusive_resume_lock,
)
from convert_core.errors import ConversionError


def _payload(**updates):
    payload = {
        "resume_schema_version": 1,
        "source_files": [{"relative_path": "source.bin", "size": 8, "mtime_ns": 10}],
        "features": {"observation.state": {"dtype": "float32", "shape": [2]}},
        "conversion_options": {"codec": "h264", "preset": "p4"},
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"source_files": [{"relative_path": "source.bin", "size": 9, "mtime_ns": 10}]}, "source_files"),
        ({"features": {"observation.state": {"dtype": "float64", "shape": [2]}}}, "features"),
        ({"conversion_options": {"codec": "hevc", "preset": "p4"}}, "conversion_options"),
    ],
)
def test_resume_rejects_source_schema_and_codec_fingerprint_changes(
    tmp_path: Path, updates: dict, category: str
):
    output = tmp_path / "dataset"
    CheckpointManager(output, _payload()).prepare()

    with pytest.raises(ConversionError, match=rf"changed categories:.*{category}"):
        CheckpointManager(output, _payload(**updates)).prepare()


def _committed_checkpoint(tmp_path: Path) -> CheckpointManager:
    manager = CheckpointManager(tmp_path / "dataset", _payload())
    manager.prepare()
    (manager.data_root / "meta").mkdir(parents=True)
    (manager.data_root / "meta" / "info.json").write_text(
        json.dumps({"checkpoint": "valid"}), encoding="utf-8"
    )
    (manager.data_root / "data").mkdir()
    (manager.data_root / "data" / "part.parquet").write_bytes(b"committed-data")
    manager.commit(checkpoint_unit="unit-0", completed_episodes=1, completed_frames=2)
    return manager


def test_resume_rejects_corrupt_checkpoint_marker(tmp_path: Path):
    manager = _committed_checkpoint(tmp_path)
    state = json.loads(manager.state_path.read_text(encoding="utf-8"))
    (manager.state_root / state["marker"]).write_text("not-json", encoding="utf-8")

    with pytest.raises(ConversionError, match="cannot read checkpoint marker"):
        CheckpointManager(tmp_path / "dataset", _payload()).prepare()


def test_resume_lock_is_non_blocking(tmp_path: Path):
    lock = tmp_path / ".dataset.resume.lock"
    with exclusive_resume_lock(lock):
        with pytest.raises(ConversionError, match="another resume process"):
            with exclusive_resume_lock(lock):
                pass


def test_resume_discards_files_from_uncommitted_active_unit(tmp_path: Path):
    manager = _committed_checkpoint(tmp_path)
    (manager.data_root / "data" / "partial.parquet").write_bytes(b"partial")
    (manager.data_root / "videos" / "camera").mkdir(parents=True)
    (manager.data_root / "videos" / "camera" / "partial.mp4").write_bytes(b"partial")
    (manager.data_root / "meta" / "info.json").write_text("partial", encoding="utf-8")
    (manager.data_root / "images").mkdir()
    (manager.data_root / "images" / "partial.png").write_bytes(b"partial")
    (manager.data_root / "tmp-active-unit").mkdir()

    position = CheckpointManager(tmp_path / "dataset", _payload()).prepare()

    assert position.completed_episodes == 1
    assert (manager.data_root / "data" / "part.parquet").read_bytes() == b"committed-data"
    assert not (manager.data_root / "data" / "partial.parquet").exists()
    assert not (manager.data_root / "videos" / "camera" / "partial.mp4").exists()
    assert not (manager.data_root / "images").exists()
    assert not (manager.data_root / "tmp-active-unit").exists()
    restored = json.loads((manager.data_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert restored == {"checkpoint": "valid"}


def test_resume_payload_fingerprints_decoder_repository_head_and_weight_file(
    tmp_path: Path,
):
    repository = tmp_path / "decoder-repository"
    ref = repository / ".git" / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    (repository / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="utf-8"
    )
    ref.write_text("a" * 40 + "\n", encoding="utf-8")
    weights = tmp_path / "decoder.jit"
    weights.write_bytes(b"first decoder")
    plan = SimpleNamespace(
        dataset_uid="decoder-test",
        output_path=tmp_path / "output",
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        episodes=(),
        extra={
            "source_root": tmp_path / "source",
            "source_dataset": "fixture",
            "source_revision": "revision",
            "source_files": [],
            "source_splits": [],
            "field_mapping": [],
            "partition_rules": {"version": "fixture"},
            "decoder": {
                "v1_decoder_repo": str(repository),
                "v2_decoder_path": str(weights),
            },
        },
        feature_schema=lambda: {},
    )

    first = build_resume_payload(plan, reader_format="fixture")
    ref.write_text("b" * 40 + "\n", encoding="utf-8")
    weights.write_bytes(b"second decoder with a different size")
    second = build_resume_payload(plan, reader_format="fixture")

    assert first["decoder_records"] != second["decoder_records"]
    assert canonical_fingerprint(first) != canonical_fingerprint(second)
    assert first["decoder_records"][0]["git_head"] == "a" * 40
    assert second["decoder_records"][0]["git_head"] == "b" * 40
