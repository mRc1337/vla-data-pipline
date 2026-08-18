from pathlib import Path

import numpy as np
import pytest

import convert_1x_world_model_dataset as converter
import evaluate_1x_world_model_conversion as evaluator
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError


class _Reader:
    def preflight_decoder(self, _plan) -> None:
        return None


def _plan(workspace: Path, version: str) -> DatasetConversionPlan:
    suffix = version.replace(".", "_")
    return DatasetConversionPlan(
        dataset_uid=f"collection_fixture_{suffix}",
        output_path=workspace / suffix,
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(),
        camera_features=(),
        episodes=(
            EpisodePlan(
                episode_uid=f"{version}:0",
                source_relative_path=f"train_{version}/segment:0",
                instruction="Unspecified task; the source dataset provides no instruction.",
                num_frames=2,
                extra={
                    "source_split": f"train_{version}",
                    "source_segment_id": 0,
                    "checkpoint_unit": f"train_{version}/unit_0",
                },
            ),
        ),
        extra={
            "source_root": workspace.parent / "source",
            "source_dataset": "1x-technologies/worldmodel",
            "source_revision": "revision",
            "source_version": version,
            "source_splits": [f"train_{version}"],
            "source_files": [],
            "field_mapping": [],
            "partition_rules": {"source_version": version},
            "decoder": {},
            "video_encoding": {"target_codec": "h264", "video_reencoded": True},
            "unsupported_source_components": {"test_v2.0": "unaligned"},
        },
    )


def test_collection_resume_reuses_completed_partition_and_cleans_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    calls: list[str] = []
    interrupt_v2 = True

    def plans(_args, workspace):
        return _Reader(), [_plan(workspace, "v1.1"), _plan(workspace, "v2.0")]

    def fake_convert(plan, _iter_frames, **_kwargs):
        nonlocal interrupt_v2
        calls.append(plan.dataset_uid)
        if plan.dataset_uid.endswith("v2_0") and interrupt_v2:
            interrupt_v2 = False
            raise RuntimeError("synthetic partition interruption")
        plan.output_path.mkdir(parents=True)
        (plan.output_path / "verified").write_text("ok", encoding="utf-8")
        return plan.output_path

    def fake_validate(_plan, root):
        assert (root / "verified").read_text(encoding="utf-8") == "ok"

    monkeypatch.setattr(converter, "_plans", plans)
    monkeypatch.setattr(converter, "_rgb_encoder", lambda *_args: object())
    monkeypatch.setattr(converter, "_preflight_encoder", lambda *_args: None)
    monkeypatch.setattr(converter, "convert_dataset", fake_convert)
    monkeypatch.setattr(converter, "validate_written_dataset", fake_validate)
    monkeypatch.setattr(converter, "validate_video_files", lambda *_args, **_kwargs: {})

    arguments = [
        "--staging-root",
        str(tmp_path / "stage"),
        "--output-dataset-uid",
        "collection_fixture",
        "--resume",
    ]
    assert converter.main(arguments) == 1
    final = tmp_path / "stage" / "lerobot_v3_0" / "collection_fixture"
    resume_data, resume_state, resume_lock = converter.resume_paths(final)
    assert (resume_data / "v1_1" / "verified").is_file()
    assert resume_state.is_dir()

    assert converter.main(arguments) == 0

    assert (final / "v1_1" / "verified").is_file()
    assert (final / "v2_0" / "verified").is_file()
    assert (final / "collection_manifest.json").is_file()
    assert not resume_data.exists()
    assert not resume_state.exists()
    assert not resume_lock.exists()
    assert calls == ["collection_fixture_v1_1", "collection_fixture_v2_0", "collection_fixture_v2_0"]
    assert "reused completed collection partition" in capsys.readouterr().out


def test_collection_resume_rejects_changed_fingerprint(tmp_path: Path):
    data = tmp_path / ".collection.resume"
    state = tmp_path / ".collection.resume-state"
    converter._prepare_collection_resume(data, state, {"codec": "h264"})

    with pytest.raises(ConversionError, match="collection resume fingerprint changed"):
        converter._prepare_collection_resume(data, state, {"codec": "hevc"})


def test_evaluator_requires_declared_vector_shape_dtype_and_values():
    schema = {"dtype": "float32", "shape": [1]}
    source = np.asarray([0.25], dtype=np.float32)

    valid = evaluator._vector_result(source.copy(), source, schema)
    assert valid["exact"] is True
    assert valid["shape_matches"] is True
    assert valid["dtype_matches"] is True

    scalar = evaluator._vector_result(np.float32(0.25), source, schema)
    assert scalar["exact"] is False
    assert scalar["shape_matches"] is False

    widened = evaluator._vector_result(source.astype(np.float64), source, schema)
    assert widened["exact"] is False
    assert widened["dtype_matches"] is False
