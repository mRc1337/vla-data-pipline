import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
import convert_core.lerobot_writer as writer
from convert_core.lerobot_writer import (
    build_manifest,
    validate_info_json,
    validate_parquet_feature_schema,
    validate_video_files,
    write_dataset,
)


def _plan(root: Path) -> DatasetConversionPlan:
    return DatasetConversionPlan(
        dataset_uid="metadata_test",
        output_path=root,
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(
            VectorFeatureSpec(
                feature_key="observation.state",
                dim=2,
                names=("first", "second"),
            ),
        ),
        camera_features=(
            CameraFeatureSpec(
                feature_key="observation.images.head",
                height=256,
                width=256,
            ),
        ),
        episodes=(
            EpisodePlan(
                episode_uid="source:7",
                source_relative_path="split/segment:7",
                instruction="second task encountered first",
                num_frames=2,
                extra={"source_split": "split", "source_segment_id": 7},
            ),
            EpisodePlan(
                episode_uid="source:8",
                source_relative_path="split/segment:8",
                instruction="alphabetically first",
                num_frames=3,
                extra={"source_split": "split", "source_segment_id": 8},
            ),
        ),
    )


def _info(plan: DatasetConversionPlan) -> dict:
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": ["first", "second"],
            },
            "observation.images.head": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 256,
                    "video.width": 256,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": 30,
                    "video.channels": 3,
                    "has_audio": False,
                    "is_depth_map": False,
                },
            },
        },
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": "eve",
        "splits": {"train": "0:2"},
    }


def _write_info(root: Path, value: dict) -> None:
    path = root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_info_json_validator_covers_counts_templates_names_and_video_metadata(
    tmp_path: Path,
):
    plan = _plan(tmp_path)
    valid = _info(plan)
    _write_info(tmp_path, valid)

    assert validate_info_json(plan, tmp_path, 2) == valid

    corruptions = [
        ("codebase_version", lambda value: value.__setitem__("codebase_version", "v2.1")),
        ("total_frames", lambda value: value.__setitem__("total_frames", 4)),
        ("data_path", lambda value: value.__setitem__("data_path", "wrong/{index}")),
        (
            "names",
            lambda value: value["features"]["observation.state"].__setitem__(
                "names", ["second", "first"]
            ),
        ),
        (
            "video.width",
            lambda value: value["features"]["observation.images.head"]["info"].__setitem__(
                "video.width", 128
            ),
        ),
    ]
    for expected_error, corrupt in corruptions:
        value = copy.deepcopy(valid)
        corrupt(value)
        _write_info(tmp_path, value)
        with pytest.raises(ConversionError, match=expected_error):
            validate_info_json(plan, tmp_path, 2)


def test_manifest_records_explicit_output_indices_and_source_task(tmp_path: Path):
    manifest = build_manifest(_plan(tmp_path), reader_format="fixture")

    assert manifest["task_index_mapping"] == {
        "0": "second task encountered first",
        "1": "alphabetically first",
    }
    assert manifest["episodes"][0]["lerobot_episode_index"] == 0
    assert manifest["episodes"][0]["lerobot_task_index"] == 0
    assert manifest["episodes"][0]["source_task"] is None
    assert manifest["episodes"][1]["lerobot_episode_index"] == 1
    assert manifest["episodes"][1]["lerobot_task_index"] == 1


def test_parquet_validator_accepts_lerobot_scalar_encoding_for_length_one_vector(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    plan = replace(
        _plan(tmp_path),
        vector_features=(
            VectorFeatureSpec(
                feature_key="action.hand_closure",
                dim=1,
                names=("hand_closure",),
            ),
        ),
        camera_features=(),
    )
    parquet = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"action.hand_closure": pa.array([0.0, 1.0], type=pa.float32())}),
        parquet,
    )

    validate_parquet_feature_schema(plan, tmp_path)

    pq.write_table(
        pa.table(
            {
                "action.hand_closure": pa.array(
                    [[0.0], [1.0]], type=pa.list_(pa.float32(), 1)
                )
            }
        ),
        parquet,
    )
    with pytest.raises(ConversionError, match="physical type fixed_size_list"):
        validate_parquet_feature_schema(plan, tmp_path)


def test_real_writer_accepts_length_one_ndarray_and_writes_scalar_parquet(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = tmp_path / "output"
    episode = EpisodePlan(
        episode_uid="source:0",
        source_relative_path="split/segment:0",
        instruction="fixture task",
        num_frames=1,
    )
    plan = DatasetConversionPlan(
        dataset_uid="singleton_vector_test",
        output_path=output,
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(
            VectorFeatureSpec(
                feature_key="action.hand_closure",
                dim=1,
                names=("hand_closure",),
            ),
        ),
        camera_features=(),
        episodes=(episode,),
    )

    write_dataset(
        plan,
        lambda _episode: iter(
            (
                {
                    "action.hand_closure": np.array([0.25], dtype=np.float32),
                    "task": "fixture task",
                },
            )
        ),
        output,
    )

    parquet = next((output / "data").rglob("*.parquet"))
    assert pq.read_schema(parquet).field("action.hand_closure").type == pa.float32()
    validate_parquet_feature_schema(plan, output)


def test_generated_index_stats_have_one_schema_for_single_and_multi_frame_episodes(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes = (
        EpisodePlan("source:0", "split/segment:0", "fixture task", 2),
        EpisodePlan("source:1", "split/segment:1", "fixture task", 1),
    )
    plan = DatasetConversionPlan(
        dataset_uid="stable_generated_index_stats",
        output_path=tmp_path / "output",
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(VectorFeatureSpec("observation.state", 1),),
        camera_features=(),
        episodes=episodes,
    )

    def frames(episode: EpisodePlan):
        for frame_index in range(episode.num_frames):
            yield {
                "observation.state": np.asarray([frame_index], dtype=np.float32),
                "task": episode.instruction,
            }

    write_dataset(plan, frames, plan.output_path)

    schema = pq.read_schema(
        next((plan.output_path / "meta" / "episodes").rglob("*.parquet"))
    )
    for feature in ("frame_index", "episode_index", "index", "task_index"):
        for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            assert schema.field(f"stats/{feature}/{stat}").type == pa.list_(pa.float64())
        assert schema.field(f"stats/{feature}/count").type == pa.list_(pa.int64())


def test_streaming_image_stats_have_one_schema_for_multi_and_single_frame_episodes(
    tmp_path: Path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.configs.video import RGBEncoderConfig

    episodes = (
        EpisodePlan(
            "source:0",
            "split/segment:0",
            "fixture task",
            2,
            {"checkpoint_unit": "fixture-unit"},
        ),
        EpisodePlan(
            "source:1",
            "split/segment:1",
            "fixture task",
            1,
            {"checkpoint_unit": "fixture-unit"},
        ),
    )
    output = tmp_path / "output"
    plan = DatasetConversionPlan(
        dataset_uid="stable_streaming_image_stats",
        output_path=output,
        fps=10,
        measured_fps=10.0,
        robot_type="fixture",
        vector_features=(VectorFeatureSpec("observation.state", 1),),
        camera_features=(CameraFeatureSpec("observation.images.head", 16, 16),),
        episodes=episodes,
    )

    def frames(episode: EpisodePlan):
        for frame_index in range(episode.num_frames):
            yield {
                "observation.state": np.asarray([frame_index], dtype=np.float32),
                "observation.images.head": np.full(
                    (16, 16, 3), 32 + frame_index, dtype=np.uint8
                ),
                "task": episode.instruction,
            }

    writer.convert_dataset(
        plan,
        frames,
        reader_format="synthetic",
        resume=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="medium"),
        streaming_encoding=True,
        blocking_streaming_encoding=True,
        encoder_queue_maxsize=1,
        encoder_threads=1,
        fragmented_mp4_writes=True,
    )

    episodes_file = next((output / "meta" / "episodes").rglob("*.parquet"))
    table = pq.read_table(episodes_file)
    schema = table.schema
    for stat in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        assert schema.field(f"stats/observation.images.head/{stat}").type == pa.list_(
            pa.list_(pa.list_(pa.float64()))
        )
    count_key = "stats/observation.images.head/count"
    assert schema.field(count_key).type == pa.list_(pa.int64())
    assert table[count_key].to_pylist() == [[512], [256]]


def test_real_writer_preserves_multidimensional_array_shape(tmp_path: Path):
    output = tmp_path / "output"
    episode = EpisodePlan(
        episode_uid="source:0",
        source_relative_path="split/segment:0",
        instruction="fixture task",
        num_frames=1,
    )
    plan = DatasetConversionPlan(
        dataset_uid="multidimensional_vector_test",
        output_path=output,
        fps=30,
        measured_fps=30.0,
        robot_type="eve",
        vector_features=(
            VectorFeatureSpec(
                feature_key="observation.state.joints",
                dim=2,
                names=("joint_a", "joint_b"),
                dtype="float64",
                shape=(2, 1),
            ),
        ),
        camera_features=(),
        episodes=(episode,),
    )

    write_dataset(
        plan,
        lambda _episode: iter(
            (
                {
                    "observation.state.joints": np.array(
                        [[0.25], [0.5]], dtype=np.float64
                    ),
                    "task": "fixture task",
                },
            )
        ),
        output,
    )

    validate_parquet_feature_schema(plan, output)


def test_video_validator_rejects_codec_or_pixel_format_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = replace(
        _plan(tmp_path),
        extra={
            "video_encoding": {
                "target_codec": "h264",
                "target_pix_fmt": "yuv420p",
            }
        },
    )
    video = (
        tmp_path
        / "videos"
        / "observation.images.head"
        / "chunk-000"
        / "file-000.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")

    monkeypatch.setattr(
        writer,
        "_video_frame_count",
        lambda _path: (5, 256, 256, 30.0, "hevc", "yuv420p"),
    )
    with pytest.raises(ConversionError, match="video codec is 'hevc'"):
        validate_video_files(plan, tmp_path, expected_frames=5)

    monkeypatch.setattr(
        writer,
        "_video_frame_count",
        lambda _path: (5, 256, 256, 30.0, "h264", "yuv444p"),
    )
    with pytest.raises(ConversionError, match="video pixel format is 'yuv444p'"):
        validate_video_files(plan, tmp_path, expected_frames=5)


def test_deferred_metadata_batches_info_stats_and_writer_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import lerobot.datasets.dataset_metadata as metadata_module
    import lerobot.datasets.dataset_writer as writer_module

    calls = {"info": 0, "stats": 0}

    def write_info(_value, _root):
        calls["info"] += 1

    def write_stats(_value, _root):
        calls["stats"] += 1

    monkeypatch.setattr(metadata_module, "write_info", write_info)
    monkeypatch.setattr(metadata_module, "write_stats", write_stats)
    monkeypatch.setattr(writer_module, "write_info", write_info)
    dataset = type(
        "Dataset",
        (),
        {"meta": type("Meta", (), {"info": {}, "stats": {}, "root": tmp_path})()},
    )()

    with writer._deferred_info_stats_writes(dataset, True):
        for _ in range(3):
            metadata_module.write_info({}, tmp_path)
            metadata_module.write_stats({}, tmp_path)
            writer_module.write_info({}, tmp_path)

    assert calls == {"info": 1, "stats": 1}


def test_fragmented_mp4_context_forces_sequential_write_movflags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import lerobot.datasets.video_utils as video_utils

    calls = []

    class FakeContainer:
        def close(self):
            return None

    def fake_open(file, mode=None, *args, **kwargs):
        calls.append((file, mode, args, kwargs))
        return FakeContainer()

    monkeypatch.setattr(video_utils.av, "open", fake_open)
    with writer._fragmented_mp4_writes(True):
        container = video_utils.av.open(
            tmp_path / "fixture.mp4", mode="w", options={"movflags": "faststart"}
        )
        container.close()
        video_utils.av.open("fixture.mp4", mode="r")

    assert calls[0][3]["options"]["movflags"] == (
        "frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets"
    )
    assert "options" not in calls[1][3]


def test_deferred_video_concatenation_flushes_one_time_per_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import types
    import lerobot.datasets.dataset_writer as dataset_writer
    import lerobot.datasets.dataset_metadata as metadata_module

    concat_calls: list[tuple[list[Path], Path]] = []

    def fake_concat(inputs, output, *args, **kwargs):
        del args, kwargs
        paths = [Path(path) for path in inputs]
        concat_calls.append((paths, Path(output)))
        Path(output).write_bytes(b"".join(path.read_bytes() for path in paths))

    monkeypatch.setattr(dataset_writer, "get_file_size_in_mb", lambda _path: 1.0)
    monkeypatch.setattr(dataset_writer, "get_video_duration_in_s", lambda _path: 0.5)
    monkeypatch.setattr(dataset_writer, "concatenate_video_files", fake_concat)
    monkeypatch.setattr(dataset_writer.DatasetWriter, "flush_pending_videos", lambda _self: None)
    monkeypatch.setattr(metadata_module, "write_info", lambda *_args, **_kwargs: None)

    info_calls = []
    meta = types.SimpleNamespace(
        video_files_size_in_mb=100,
        chunks_size=1000,
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        video_keys=["observation.images.agentview"],
        depth_keys=set(),
        info={},
        root=tmp_path,
        update_video_info=lambda *args, **kwargs: info_calls.append((args, kwargs)),
    )
    dataset = types.SimpleNamespace(
        _root=tmp_path,
        _meta=meta,
        _rgb_encoder=None,
        _depth_encoder=None,
    )

    with writer._deferred_video_concatenation(True):
        for index in range(3):
            episode_dir = tmp_path / f"episode-{index}"
            episode_dir.mkdir()
            episode_path = episode_dir / "episode.mp4"
            episode_path.write_bytes(bytes([index + 1]))
            metadata = dataset_writer.DatasetWriter._save_episode_video(
                dataset, "observation.images.agentview", index, temp_path=episode_path
            )
            assert metadata["videos/observation.images.agentview/to_timestamp"] == (index + 1) * 0.5
        assert not list((tmp_path / "videos").rglob("*.mp4"))
        dataset_writer.DatasetWriter.flush_pending_videos(dataset)

    assert len(concat_calls) == 1
    assert len(concat_calls[0][0]) == 3
    assert concat_calls[0][1].is_file()
    assert concat_calls[0][1].read_bytes() == b"\x01\x02\x03"
    assert info_calls


def test_fragmented_mp4_context_retries_transient_ossfs_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import errno
    import lerobot.datasets.video_utils as video_utils

    attempts = 0
    sleeps: list[float] = []

    class FakeContainer:
        def close(self):
            return None

    def flaky_open(file, mode=None, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EINVAL, "fresh OSSFS directory is not visible", file)
        return FakeContainer()

    monkeypatch.setattr(video_utils.av, "open", flaky_open)
    monkeypatch.setattr(writer.time, "sleep", sleeps.append)

    with writer._fragmented_mp4_writes(True):
        container = video_utils.av.open(tmp_path / "fixture.mp4", mode="w")
        container.close()

    assert attempts == 2
    assert sleeps == [0.05]


def test_fragmented_mp4_context_retries_pyav_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import av
    import errno
    import lerobot.datasets.video_utils as video_utils

    attempts = 0
    sleeps: list[float] = []

    class FakeContainer:
        def close(self):
            return None

    def flaky_open(file, mode=None, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise av.error.ValueError(errno.EINVAL, "fresh OSSFS file is not visible", file)
        return FakeContainer()

    monkeypatch.setattr(video_utils.av, "open", flaky_open)
    monkeypatch.setattr(writer.time, "sleep", sleeps.append)

    with writer._fragmented_mp4_writes(True):
        container = video_utils.av.open(tmp_path / "fixture.mp4", mode="w")
        container.close()

    assert attempts == 2
    assert sleeps == [0.05]


def test_fragmented_mp4_context_does_not_retry_unrelated_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import lerobot.datasets.video_utils as video_utils

    attempts = 0

    def invalid_open(file, mode=None, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid codec configuration")

    monkeypatch.setattr(video_utils.av, "open", invalid_open)

    with writer._fragmented_mp4_writes(True), pytest.raises(
        ValueError, match="invalid codec configuration"
    ):
        video_utils.av.open(tmp_path / "fixture.mp4", mode="w")

    assert attempts == 1


def test_in_place_part_resume_reuses_verified_prefix_without_directory_publish(
    tmp_path: Path,
):
    output = tmp_path / "final-compatible"
    state = tmp_path / "resume-state"
    lock = tmp_path / "resume.lock"
    episodes = tuple(
        EpisodePlan(
            episode_uid=f"source:{index}",
            source_relative_path=f"source/{index}",
            instruction="fixture task",
            num_frames=1,
            extra={"checkpoint_unit": f"part-{index // 2}"},
        )
        for index in range(4)
    )
    plan = DatasetConversionPlan(
        dataset_uid="in_place_resume",
        output_path=output,
        fps=30,
        measured_fps=30.0,
        robot_type="fixture",
        vector_features=(VectorFeatureSpec("observation.state", 1),),
        camera_features=(),
        episodes=episodes,
    )
    first_calls: list[str] = []

    def interrupted(episode: EpisodePlan):
        first_calls.append(episode.episode_uid)
        if episode.episode_uid == "source:2":
            raise RuntimeError("synthetic interruption")
        yield {
            "observation.state": np.array([0.25], dtype=np.float32),
            "task": episode.instruction,
        }

    options = {
        "reader_format": "fixture",
        "resume": True,
        "metadata_buffer_size": 2,
        "resume_data_root": output,
        "resume_state_root": state,
        "resume_lock_path": lock,
        "publish_on_complete": False,
        "cleanup_resume_state": False,
        "rebuild_corrupt_checkpoint": True,
        "batch_metadata_writes": True,
    }
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        writer.convert_dataset(plan, interrupted, **options)

    assert first_calls == ["source:0", "source:1", "source:2"]
    assert len(list((state / "markers").glob("*.json"))) == 1
    resumed_calls: list[str] = []

    def resumed(episode: EpisodePlan):
        resumed_calls.append(episode.episode_uid)
        yield {
            "observation.state": np.array([0.25], dtype=np.float32),
            "task": episode.instruction,
        }

    result = writer.convert_dataset(plan, resumed, **options)

    assert result == output
    assert resumed_calls == ["source:2", "source:3"]
    assert (output / "conversion_manifest.json").is_file()
    assert len(list((state / "markers").glob("*.json"))) == 2
    assert not list(tmp_path.glob(".final-compatible.incomplete-*"))

    # Preserve the recorded file size so structural inventory validation
    # passes; the Parquet reopen must still detect the damaged latest part and
    # roll back exactly one checkpoint unit.
    latest_data = sorted((output / "data").rglob("*.parquet"))[-1]
    damaged = bytearray(latest_data.read_bytes())
    damaged[:4] = b"FAIL"
    latest_data.write_bytes(damaged)
    rebuilt_calls: list[str] = []

    def rebuilt(episode: EpisodePlan):
        rebuilt_calls.append(episode.episode_uid)
        yield {
            "observation.state": np.array([0.25], dtype=np.float32),
            "task": episode.instruction,
        }

    writer.convert_dataset(plan, rebuilt, **options)
    assert rebuilt_calls == ["source:2", "source:3"]
    writer.validate_written_dataset(plan, output)
