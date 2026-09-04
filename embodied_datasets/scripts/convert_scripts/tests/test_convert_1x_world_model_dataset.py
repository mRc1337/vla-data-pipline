from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import convert_1x_world_model_dataset as converter
import evaluate_1x_world_model_conversion as evaluator
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan
from convert_core.errors import ConversionError


class _Reader:
    def preflight_decoder(self, _plan) -> None:
        return None


def test_local_pipeline_defaults_are_bounded():
    args = converter._build_parser().parse_args([])
    assert args.local_work_root == Path.home() / "1x_world_model_dataset_staging"
    assert args.max_local_temp_bytes == 100_000_000_000
    assert args.min_local_free_bytes == 200_000_000_000
    assert args.upload_workers == 1
    assert args.max_upload_queue_units == 2


def test_v1_decoder_optimization_options_are_explicit():
    args = converter._build_parser().parse_args(
        [
            "--v1-postprocess-device",
            "gpu",
            "--decoder-cpu-threads",
            "16",
            "--decode-batch-size",
            "32",
        ]
    )
    assert args.v1_postprocess_device == "gpu"
    assert args.decoder_cpu_threads == 16
    assert args.decode_batch_size == 32


def test_encoder_preflight_uses_a_bounded_short_sample(monkeypatch: pytest.MonkeyPatch):
    encoded_frames = 0

    class FakeStream:
        vcodec = "h264"

        def encode(self, frame=None):
            nonlocal encoded_frames
            if frame is not None:
                encoded_frames += 1
            return ()

    class FakeContainer:
        def add_stream(self, *_args, **_kwargs):
            return FakeStream()

        def mux(self, _packet):
            return None

        def close(self):
            return None

    class FakeVideoFrame:
        @staticmethod
        def from_ndarray(_array, format):
            assert format == "rgb24"
            return object()

    fake_av = SimpleNamespace(
        open=lambda *_args, **_kwargs: FakeContainer(), VideoFrame=FakeVideoFrame
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    plan = SimpleNamespace(
        camera_features=(SimpleNamespace(height=256, width=256),), fps=30
    )
    encoder = SimpleNamespace(
        vcodec="h264", pix_fmt="yuv420p", get_codec_options=lambda **_: {}
    )

    converter._preflight_encoder(plan, encoder)

    assert encoded_frames == converter.PREFLIGHT_FRAME_COUNT == 60


def test_fragmented_streaming_mp4_is_seek_free_and_scoped(
    monkeypatch: pytest.MonkeyPatch,
):
    from lerobot import datasets

    calls: list[tuple[object, str, tuple[object, ...], dict[str, object]]] = []

    def fake_open(file, mode="r", *args, **kwargs):
        calls.append((file, mode, args, kwargs))
        return object()

    video_utils = SimpleNamespace(av=SimpleNamespace(open=fake_open))
    monkeypatch.setattr(datasets, "video_utils", video_utils)

    converter._enable_fragmented_streaming_mp4()
    wrapped = video_utils.av.open
    converter._enable_fragmented_streaming_mp4()
    assert video_utils.av.open is wrapped

    wrapped("episode_streaming.mp4", "w", options={"custom": "yes"})
    wrapped("ordinary.mp4", "w")
    wrapped("episode_streaming.mp4", "r")

    assert calls[0][3]["options"] == {
        "custom": "yes",
        "movflags": (
            "frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets"
        ),
    }
    assert calls[1][3] == {}
    assert calls[2][3] == {}


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


def test_checkpoint_sample_selects_first_episode_from_each_unit(tmp_path: Path):
    base = _plan(tmp_path, "v2.0")
    episodes = tuple(
        replace(
            base.episodes[0],
            episode_uid=f"episode-{index}",
            source_relative_path=f"source/{index}",
            num_frames=index + 1,
            extra={"checkpoint_unit": unit},
        )
        for index, unit in enumerate(("shard-0", "shard-0", "shard-1", "shard-2"))
    )
    plan = replace(base, episodes=episodes)

    selected = converter._select(
        plan,
        max_episodes=None,
        max_units=2,
        one_episode_per_unit=True,
    )

    assert [episode.episode_uid for episode in selected.episodes] == [
        "episode-0",
        "episode-2",
    ]
    assert [episode.extra["checkpoint_unit"] for episode in selected.episodes] == [
        "shard-0",
        "shard-1",
    ]


def test_worker_benchmark_samples_only_dataset_scoped_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sampled_roots: list[tuple[Path, ...]] = []
    commands: list[list[str]] = []

    class FakeSampler:
        def __init__(self, roots):
            sampled_roots.append(tuple(roots))

        def start(self):
            return None

        def stop(self):
            return SimpleNamespace(
                wall_seconds=2.0,
                cpu_seconds=1.0,
                average_cpu_cores=0.5,
                peak_rss_bytes=1024,
                read_bytes=0,
                write_bytes=0,
                read_chars=0,
                write_chars=0,
                peak_temp_bytes=2048,
                io_counters_available=True,
            )

    args = SimpleNamespace(
        benchmark_workers=[1],
        output_dataset_uid="bounded_benchmark",
        output_root=tmp_path / "lerobot_v3_0",
        local_work_root=tmp_path / "local-benchmarks",
        max_episodes=1,
        max_checkpoint_units=None,
        encoder_threads_per_worker=2,
        encoder_threads=None,
        version=["v1.1"],
        video_codec="h264",
        video_quality=18,
        video_preset="medium",
        benchmark_report=tmp_path / "workers.json",
    )

    def fake_run(command):
        commands.append(list(command))
        uid = command[command.index("--output-dataset-uid") + 1]
        final = args.output_root / uid
        final.mkdir(parents=True)
        converter.atomic_write_json(
            final / "collection_manifest.json",
            {"dataset_uid": uid, "partitions": [{"frames": 10}]},
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(converter, "ProcessTreeSampler", FakeSampler)
    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    result = converter._run_worker_benchmarks(
        [
            "--benchmark-workers",
            "1",
            "--max-episodes",
            "1",
            "--output-dataset-uid",
            args.output_dataset_uid,
            "--work-dir",
            str(tmp_path / "unsafe-shared-work"),
        ],
        args,
    )

    assert result == 0
    uid = f"{args.output_dataset_uid}_benchmark_w1"
    local_root = args.local_work_root / "benchmarks" / uid
    assert sampled_roots == [
        (
            args.output_root / uid,
            local_root / ".conversion_work" / uid / "benchmark",
            local_root / ".conversion_resume" / uid,
            local_root / ".conversion_logs" / uid,
        )
    ]
    command = commands[0]
    assert str(tmp_path / "unsafe-shared-work") not in command
    assert command[command.index("--temp-dir") + 1] == str(
        local_root / ".conversion_work" / uid / "benchmark" / "tmp"
    )


def test_formal_multi_worker_conversion_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(converter, "_visible_cuda_devices", lambda: ("0", "1", "2", "3"))

    with pytest.raises(SystemExit, match="2"):
        converter.main(
            [
                "--workers",
                "2",
                "--max-episodes",
                "1",
                "--output-dataset-uid",
                "diagnostic",
            ]
        )

    assert "exact equivalence failed" in capsys.readouterr().err


@pytest.mark.parametrize("codec", ["h264_nvenc", "hevc_nvenc"])
def test_nvenc_is_always_rejected(codec: str, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit, match="2"):
        converter.main(["--video-codec", codec, "--dry-run"])

    assert "NVENC is unsupported" in capsys.readouterr().err


def test_total_parallel_encoder_threads_are_capped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(converter, "_visible_cuda_devices", lambda: ("0", "1", "2", "3"))

    with pytest.raises(SystemExit, match="2"):
        converter.main(
            [
                "--benchmark-workers",
                "1",
                "2",
                "4",
                "--encoder-threads-per-worker",
                "9",
                "--max-episodes",
                "1",
                "--output-dataset-uid",
                "diagnostic",
            ]
        )

    assert "32-thread limit" in capsys.readouterr().err


def test_evaluator_requires_declared_vector_shape_dtype_and_values():
    schema = {"dtype": "float32", "shape": [1]}
    source = np.asarray([0.25], dtype=np.float32)

    valid = evaluator._vector_result(source.copy(), source, schema)
    assert valid["exact"] is True
    assert valid["shape_matches"] is True
    assert valid["dtype_matches"] is True

    scalar = evaluator._vector_result(np.float32(0.25), source, schema)
    assert scalar["exact"] is True
    assert scalar["shape_matches"] is True
    assert scalar["output_storage_shape"] == []
    assert scalar["scalar_storage_normalized"] is True

    widened = evaluator._vector_result(source.astype(np.float64), source, schema)
    assert widened["exact"] is False
    assert widened["dtype_matches"] is False


def test_scalable_workflow_uses_scoped_layout_and_marker_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "lerobot_v3_0"
    local = tmp_path / "local"
    observed_environment: dict[str, str] = {}
    previous_environment = {
        key: converter.os.environ.get(key)
        for key in converter.RUNTIME_ENVIRONMENT_KEYS
    }
    previous_tempdir = converter.tempfile.tempdir

    def plans(_args, final):
        return _Reader(), [_plan(final, "v1.1")]

    def fake_parallel(plan, _work, _resume, _args, _options, _devices, _capacity):
        observed_environment.update(
            {
                key: converter.os.environ[key]
                for key in converter.RUNTIME_ENVIRONMENT_KEYS
            }
        )
        plan.output_path.mkdir(parents=True)
        (plan.output_path / "verified").write_text("ok", encoding="utf-8")

    def fake_validate(_plan, output):
        assert (output / "verified").read_text(encoding="utf-8") == "ok"

    monkeypatch.setattr(converter, "_plans", plans)
    monkeypatch.setattr(converter, "_visible_cuda_devices", lambda: ("0",))
    monkeypatch.setattr(converter, "_convert_parallel_partition", fake_parallel)
    monkeypatch.setattr(converter, "validate_written_dataset", fake_validate)
    monkeypatch.setattr(converter, "validate_video_files", lambda *_args, **_kwargs: {})

    arguments = [
        "--output-root",
        str(root),
        "--raw-root",
        str(tmp_path / "raw"),
        "--local-work-root",
        str(local),
        "--output-dataset-uid",
        "collection_fixture",
        "--workers",
        "1",
    ]
    assert converter.main(arguments) == 0
    final = root / "collection_fixture"
    assert (final / "_SUCCESS").is_file()
    assert not (final / "_INCOMPLETE").exists()
    assert not (root / ".conversion_work/collection_fixture").exists()
    assert not (root / ".conversion_resume/collection_fixture").exists()
    assert not (root / ".conversion_locks/collection_fixture.lock").exists()
    assert not (local / ".conversion_work/collection_fixture").exists()
    assert (local / ".conversion_resume/collection_fixture/completed.json").is_file()
    assert list((local / ".conversion_logs/collection_fixture").glob("*.json"))
    assert all(Path(value).is_relative_to(local) for value in observed_environment.values())
    assert {
        key: converter.os.environ.get(key)
        for key in converter.RUNTIME_ENVIRONMENT_KEYS
    } == previous_environment
    assert converter.tempfile.tempdir == previous_tempdir


def test_scalable_workflow_retains_verified_state_and_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    root = tmp_path / "lerobot_v3_0"
    local = tmp_path / "local"
    fail = True

    def plans(_args, final):
        return _Reader(), [_plan(final, "v1.1")]

    def fake_parallel(plan, _work, _resume, _args, _options, _devices, _capacity):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("worker interrupted")
        plan.output_path.mkdir(parents=True, exist_ok=True)
        (plan.output_path / "verified").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(converter, "_plans", plans)
    monkeypatch.setattr(converter, "_visible_cuda_devices", lambda: ("0",))
    monkeypatch.setattr(converter, "_convert_parallel_partition", fake_parallel)
    monkeypatch.setattr(
        converter,
        "validate_written_dataset",
        lambda _plan, output: (output / "verified").read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(converter, "validate_video_files", lambda *_args, **_kwargs: {})
    arguments = [
        "--output-root",
        str(root),
        "--raw-root",
        str(tmp_path / "raw"),
        "--local-work-root",
        str(local),
        "--output-dataset-uid",
        "resume_fixture",
        "--workers",
        "1",
    ]
    assert converter.main(arguments) == 1
    final = root / "resume_fixture"
    assert (final / "_INCOMPLETE").is_file()
    assert (local / ".conversion_resume/resume_fixture/collection.json").is_file()
    assert converter.main(arguments) == 1
    assert "rerun with --resume" in capsys.readouterr().err
    assert converter.main([*arguments, "--resume"]) == 0
    assert (final / "_SUCCESS").is_file()


def test_scalable_workflow_rejects_runtime_path_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(converter, "_visible_cuda_devices", lambda: ("0",))
    result = converter.main(
        [
            "--output-root",
            str(tmp_path / "root"),
            "--local-work-root",
            str(tmp_path / "local"),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-dataset-uid",
            "path_fixture",
            "--workers",
            "1",
            "--work-dir",
            str(tmp_path / "outside"),
        ]
    )
    assert result == 1
    assert "inside staging root" in capsys.readouterr().err
