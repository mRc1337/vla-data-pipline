from pathlib import Path
import hashlib
import json
import os
import tempfile
from types import SimpleNamespace
import zlib

import pytest

import convert_dexmimicgen_to_lerobot as converter
import evaluate_dexmimicgen_conversion as evaluator
from convert_core.checkpoint import atomic_write_json
from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.lerobot_writer import build_manifest
from readers.dexmimicgen_hdf5_reader import DexMimicGenPartitionInfo, ModelReference


def test_formal_single_worker_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit, match="2"):
        converter.main(["--workers", "1"])
    assert "at least two deterministic partition workers" in capsys.readouterr().err


@pytest.mark.parametrize("codec", ["h264_nvenc", "hevc_nvenc", "av1_nvenc"])
def test_nvenc_and_unbenchmarked_codecs_are_rejected(
    codec: str, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(SystemExit, match="2"):
        converter.main(["--video-codec", codec, "--estimate-storage"])
    assert "only CPU h264 is approved" in capsys.readouterr().err


def test_encoder_budget_is_capped(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit, match="2"):
        converter.main(["--workers", "4", "--encoder-threads-per-worker", "9"])
    assert "32-thread limit" in capsys.readouterr().err


def test_resume_fingerprint_rejects_configuration_change(tmp_path: Path):
    workspace = tmp_path / "data"
    state = tmp_path / "state"
    converter._prepare_collection_resume(workspace, state, {"codec": "h264"})
    with pytest.raises(ConversionError, match="fingerprint changed"):
        converter._prepare_collection_resume(workspace, state, {"codec": "hevc"})


def test_evaluator_parses_exact_source_episode_locator():
    assert evaluator._source_location("generated/two_arm_threading.hdf5::data/demo_7") == (
        "generated/two_arm_threading.hdf5",
        "demo_7",
    )
    with pytest.raises(ValueError, match="invalid DexMimicGen episode source"):
        evaluator._source_location("generated/two_arm_threading.hdf5")


def test_evaluator_requires_complete_manifest_field_coverage(tmp_path: Path):
    episode = EpisodePlan(
        "episode-0",
        "generated/source.hdf5::data/demo_0",
        "fixture task",
        2,
        {
            "source_episode_id": "demo_0",
            "source_task": "FixtureTask",
            "checkpoint_unit": "fixture/part-00000",
        },
    )
    plan = DatasetConversionPlan(
        "fixture",
        tmp_path / "output",
        20,
        20.0,
        "fixture-robot",
        (VectorFeatureSpec("observation.state", 2),),
        (CameraFeatureSpec("observation.images.camera", 8, 8),),
        (episode,),
        {
            "source_dataset": "fixture/source",
            "source_revision": "revision",
            "source_relative_path": "generated/source.hdf5",
            "source_env_name": "FixtureTask",
            "field_mapping": [
                {
                    "source_key": "data/<demo>/obs/state",
                    "lerobot_key": "observation.state",
                    "lossy": False,
                },
                {
                    "source_key": "data/<demo>/obs/camera",
                    "lerobot_key": "observation.images.camera",
                    "lossy": True,
                },
            ],
        },
    )
    manifest = json.loads(
        json.dumps(build_manifest(plan, reader_format="fixture"))
    )

    assert evaluator._partition_manifest_evidence(manifest, plan)["passed"]
    manifest["field_mapping"] = manifest["field_mapping"][:-1]
    evidence = evaluator._partition_manifest_evidence(manifest, plan)
    assert not evidence["checks"]["field_mapping"]
    assert not evidence["checks"]["field_mapping_coverage"]
    assert not evidence["passed"]


def test_evaluator_validates_success_publication_protocol(tmp_path: Path):
    manifest = tmp_path / "collection_manifest.json"
    manifest.write_text('{"dataset": "fixture"}\n', encoding="utf-8")
    atomic_write_json(
        tmp_path / converter.SUCCESS_MARKER,
        {
            "status": "success",
            "collection_manifest_sha256": converter._sha256_file(manifest),
        },
    )

    assert evaluator._validate_publication(tmp_path, manifest)["passed"]
    atomic_write_json(
        tmp_path / converter.INCOMPLETE_MARKER,
        {"status": "incomplete"},
    )
    assert not evaluator._validate_publication(tmp_path, manifest)["passed"]


def test_model_sidecars_are_content_addressed_deduplicated_and_verified(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    model = '<mujoco><worldbody><body name="fixture"/></worldbody></mujoco>'
    raw = model.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as output:
        data = output.create_group("data")
        for name in ("demo_0", "demo_1"):
            demo = data.create_group(name)
            demo.attrs["model_file"] = model
    references = tuple(
        ModelReference(name, digest, len(raw)) for name in ("demo_0", "demo_1")
    )
    info = SimpleNamespace(source_path=source, model_references=references, partition_name="fixture")

    records = converter._write_model_sidecars([info], tmp_path / "collection")

    assert len(records) == 1
    path = tmp_path / "collection" / records[0]["relative_path"]
    assert zlib.decompress(path.read_bytes()) == raw
    assert records[0]["sha256"] == digest


def _runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "lerobot_v3_0"
    root.mkdir()
    monkeypatch.setattr(converter, "DEFAULT_OUTPUT_ROOT", root)
    args = converter._parser().parse_args(["--output-root", str(root)])
    return converter._runtime_paths(args)


def test_runtime_layout_and_environment_are_confined_to_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _runtime_paths(tmp_path, monkeypatch)
    for value in paths.__dict__.values():
        assert Path(value).is_relative_to(paths.output_root)
    assert paths.final_output == paths.output_root / "dexmimicgen"
    assert paths.work_dir == (
        paths.output_root / ".conversion_work" / "dexmimicgen" / "dexmimicgen"
    )
    assert paths.resume_dir == paths.output_root / ".conversion_resume" / "dexmimicgen"
    assert paths.logs_dir == paths.output_root / ".conversion_logs" / "dexmimicgen"
    assert paths.lock_path == paths.output_root / ".conversion_locks" / "dexmimicgen.lock"

    variables = (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "MPLCONFIGDIR",
        "VLA_DATASETS_CACHE_ROOT",
    )
    for key in variables:
        monkeypatch.setenv(key, "before-test")
    monkeypatch.setattr(tempfile, "tempdir", None)
    rendered = converter._configure_runtime_environment(paths, create=True)
    assert set(rendered) == set(variables)
    assert all(
        Path(value).is_relative_to(paths.output_root)
        for value in rendered.values()
    )
    assert all(os.environ[key] == rendered[key] for key in variables)
    assert Path(tempfile.gettempdir()).is_relative_to(paths.output_root)


def test_runtime_layout_can_use_explicit_local_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lerobot_v3_0"
    local = tmp_path / "local_runtime"
    root.mkdir()
    monkeypatch.setattr(converter, "DEFAULT_OUTPUT_ROOT", root)
    args = converter._parser().parse_args(
        ["--output-root", str(root), "--local-runtime-root", str(local)]
    )

    paths = converter._runtime_paths(args)

    assert paths.work_dir == local / "work"
    assert paths.temp_dir == local / "work" / "temp"
    assert paths.cache_dir == local / "work" / "cache"
    assert paths.resume_dir == root / ".conversion_resume" / "dexmimicgen"
    assert paths.logs_dir == root / ".conversion_logs" / "dexmimicgen"


@pytest.mark.parametrize(
    "flag",
    ["--output", "--work-dir", "--resume-dir", "--logs-dir", "--temp-dir"],
)
def test_runtime_layout_rejects_every_write_path_escape(
    flag: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lerobot_v3_0"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(converter, "DEFAULT_OUTPUT_ROOT", root)
    parser = converter._parser()

    args = parser.parse_args(["--output-root", str(root), flag, str(outside / "path")])
    with pytest.raises(ConversionError, match="inside staging root"):
        converter._runtime_paths(args)


def test_runtime_layout_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lerobot_v3_0"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(converter, "DEFAULT_OUTPUT_ROOT", root)
    parser = converter._parser()

    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    args = parser.parse_args(
        ["--output-root", str(root), "--work-dir", str(link / "work")]
    )
    with pytest.raises(ConversionError, match="inside staging root"):
        converter._runtime_paths(args)


def test_checkpoint_units_group_contiguous_episodes_by_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    episodes = tuple(
        EpisodePlan(f"episode-{index}", f"source::{index}", "task", 1)
        for index in range(5)
    )
    plan = DatasetConversionPlan(
        "fixture",
        tmp_path / "output" / "threading",
        20,
        20.0,
        "Panda+Panda",
        (),
        (),
        episodes,
    )
    info = DexMimicGenPartitionInfo(
        source_path=tmp_path / "source.hdf5",
        source_relative_path="generated/two_arm_threading.hdf5",
        partition_name="threading",
        env_args={},
        source_schema=(),
        plan=plan,
        all_episode_count=5,
        all_frame_count=5,
        selected_numeric_logical_bytes=5,
        selected_image_logical_bytes=0,
        selected_camera_frames=0,
        model_references=(),
        unique_model_count=0,
        schema_fingerprint="fixture",
        episode_length_summary={"min": 1, "max": 1},
    )
    monkeypatch.setattr(converter, "inspect_partition", lambda *_args, **_kwargs: info)
    infos = converter._build_infos(
        tmp_path,
        tmp_path / "output",
        partitions=("threading",),
        max_episodes=None,
        episodes_per_part=2,
    )

    units = [episode.extra["checkpoint_unit"] for episode in infos[0].plan.episodes]
    assert units == [
        "threading/part-00000",
        "threading/part-00000",
        "threading/part-00001",
        "threading/part-00001",
        "threading/part-00002",
    ]
    assert infos[0].plan.extra["checkpointing"] == {
        "granularity": "contiguous episode part",
        "episodes_per_part": 2,
        "per_frame_markers": False,
        "per_episode_markers": False,
    }


def test_stale_incomplete_marker_is_removed_after_valid_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _runtime_paths(tmp_path, monkeypatch)
    paths.final_output.mkdir(parents=True)
    paths.resume_dir.mkdir(parents=True)
    paths.work_dir.mkdir(parents=True)
    manifest = paths.final_output / "collection_manifest.json"
    manifest.write_text(json.dumps({"dataset": "fixture"}), encoding="utf-8")
    atomic_write_json(
        paths.final_output / converter.SUCCESS_MARKER,
        {
            "status": "success",
            "collection_manifest_sha256": converter._sha256_file(manifest),
        },
    )
    atomic_write_json(
        paths.final_output / converter.INCOMPLETE_MARKER,
        {"status": "incomplete"},
    )

    converter._recover_stale_success(paths)

    assert not (paths.final_output / converter.INCOMPLETE_MARKER).exists()
    assert (paths.final_output / converter.SUCCESS_MARKER).is_file()
    assert not paths.resume_dir.exists()
    assert not paths.work_dir.exists()
    assert paths.lock_path.is_file()
    converter._validate_success_marker(paths.final_output)
