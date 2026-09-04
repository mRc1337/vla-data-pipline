from pathlib import Path
import io
import struct

import numpy as np
import pytest
from PIL import Image

from convert_core.errors import ConversionError
from readers.roboomni_reader import (
    RoboOmniField,
    RoboOmniCatalog,
    RoboOmniRecord,
    _EXAMPLE,
    catalog_from_payload,
    catalog_payload,
    decode_record,
    parse_example,
    _speech_references,
    validate_catalog_sources,
    build_tasks,
    task_from_payload,
    task_plan_payload,
)


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_flat_tf_example_parse_and_decode_preserves_shapes(tmp_path: Path):
    example = _EXAMPLE()
    feature = example.features.feature
    feature["steps/observation/state"].float_list.value.extend([1, 2, 3, 4])
    feature["steps/conversation"].bytes_list.value.extend([b"hello", b"world"])
    feature["steps/observation/natural_language_instruction"].bytes_list.value.extend([b"", b"pick"])
    feature["steps/observation/image"].bytes_list.value.extend([_png((1, 2, 3)), _png((4, 5, 6))])
    payload = example.SerializeToString()
    path = tmp_path / "sample.tfrecord"
    path.write_bytes(struct.pack("<Q", len(payload)) + b"head" + payload + b"tail")
    values = parse_example(payload)
    assert values["steps/observation/state"][0] == "float32"

    fields = (
        RoboOmniField("observation/state", "numeric", (2,), "float32"),
        RoboOmniField("conversation", "string", (1,), "string"),
        RoboOmniField("observation/natural_language_instruction", "string", (1,), "string"),
        RoboOmniField("observation/image", "image", (2, 3, 3), "uint8"),
    )
    record = RoboOmniRecord("sample", str(path), 0, len(payload), 2, "pick", {})
    frames, _ = decode_record(record, fields)
    assert len(frames) == 2
    assert frames[0]["source.observation.state"].tolist() == [1.0, 2.0]
    assert frames[1]["source.conversation"] == "world"
    assert frames[0]["observation.images.primary"].shape == (2, 3, 3)


def test_catalog_payload_roundtrip_preserves_record_offsets():
    from readers.roboomni_reader import RoboOmniCatalog, RoboOmniPartition, _feature_specs
    from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan

    fields = (RoboOmniField("observation/state", "numeric", (2,), "float32"),)
    vectors, cameras = _feature_specs(fields)
    record = RoboOmniRecord("sample", "/raw/sample", 17, 42, 2, "pick", {}, 0, 0, 0)
    episode = EpisodePlan("episode_0", "sample@17", "pick", 2, {"record": record.__dict__, "checkpoint_unit": "/raw/sample"})
    plan = DatasetConversionPlan("roboomni", Path("roboomni/schema"), 10, 10.0, "unknown", vectors, cameras, (episode,), {})
    catalog = RoboOmniCatalog(Path("/raw"), (RoboOmniPartition("schema", ("sample",), fields, (record,), plan, ({"path": "/raw/sample", "size": 42, "mtime_ns": 1},)),), (), {}, ())
    restored = catalog_from_payload(catalog_payload(catalog), fps=10)
    assert restored.partitions[0].records[0].offset == 17
    assert restored.partitions[0].records[0].frame_count == 2


def test_speech_conv_references_split_external_audio_paths():
    assert _speech_references(("bytes", [b"./semantic/dialogue_1/user_0.wav[UNK]./semantic/dialogue_1/user_1.wav"])) == (
        "semantic/dialogue_1/user_0.wav",
        "semantic/dialogue_1/user_1.wav",
    )
    assert _speech_references(("bytes", [b""])) == ()


def test_audio_inventory_roundtrip_copies_present_and_records_missing(tmp_path: Path):
    from convert_roboomni_to_lerobot import _materialize_audio_assets

    source_root = tmp_path / "raw"
    source = source_root / "speech" / "semantic" / "dialogue_1" / "user_0.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"RIFF synthetic wav")
    stat = source.stat()
    catalog = RoboOmniCatalog(
        source_root, (), (), {}, (), (
            {"reference": "semantic/dialogue_1/user_0.wav", "relative_path": "semantic/dialogue_1/user_0.wav", "source_path": str(source), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            {"reference": "semantic/dialogue_1/user_1.wav", "relative_path": "semantic/dialogue_1/user_1.wav", "source_path": str(source.parent / "user_1.wav"), "exists": False},
        ),
    )
    output = tmp_path / "output"
    assert _materialize_audio_assets(catalog, output) == {"referenced": 2, "copied": 1, "missing": 1}
    assert (output / "audio" / "semantic/dialogue_1/user_0.wav").read_bytes() == source.read_bytes()
    assert _materialize_audio_assets(catalog, output) == {"referenced": 2, "copied": 1, "missing": 1}


def test_catalog_source_validation_catches_changed_audio_stat(tmp_path: Path):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"audio")
    stat = source.stat()
    from readers.roboomni_reader import RoboOmniCatalog

    catalog = RoboOmniCatalog(
        tmp_path, (), (), {}, (),
        ({"reference": "speech.wav", "source_path": str(source), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},),
    )
    validate_catalog_sources(catalog)
    source.write_bytes(b"changed")
    with pytest.raises(ConversionError, match="audio source changed"):
        validate_catalog_sources(catalog)


def test_output_root_accepts_exact_collection_path():
    from convert_roboomni_to_lerobot import _output_parent

    assert _output_parent(Path("/staging/roboomni")) == Path("/staging")
    assert _output_parent(Path("/staging")) == Path("/staging")


def test_tasks_are_stably_ordered_and_grouped_across_schema_partitions():
    from readers.roboomni_reader import RoboOmniPartition, _feature_specs
    from convert_core.episode_spec import DatasetConversionPlan

    fields = (RoboOmniField("observation/state", "numeric", (1,), "float32"),)
    vectors, cameras = _feature_specs(fields)

    def partition(name, component, records):
        plan = DatasetConversionPlan(
            "roboomni", Path("roboomni") / name, 10, 10.0, "unknown",
            vectors, cameras, (), {},
        )
        return RoboOmniPartition(
            name, (component,), fields, tuple(records), plan,
            ({"path": f"/raw/{component}.tfrecord", "size": 100, "mtime_ns": 1},),
        )

    catalog = RoboOmniCatalog(
        Path("/raw"),
        (
            partition("schema_a", "a", (
                RoboOmniRecord("a", "/raw/a.tfrecord", 0, 10, 2, "z-task", {}, 0, 0, 0),
                RoboOmniRecord("a", "/raw/a.tfrecord", 20, 10, 3, "a-task", {}, 1, 2, 0),
            )),
            partition("schema_b", "b", (
                RoboOmniRecord("b", "/raw/b.tfrecord", 0, 10, 4, "a-task", {}, 0, 0, 0),
            )),
        ),
        (), {}, (),
    )
    tasks = build_tasks(catalog)
    assert [task.instruction for task in tasks] == ["a-task", "z-task"]
    assert [task.task_index for task in tasks] == [0, 1]
    assert tasks[0].episode_count == 2
    assert [member.partition_name for member in tasks[0].members] == ["schema_a", "schema_b"]
    assert tasks[0].members[0].records[0].global_episode_index == 0
    assert tasks[0].members[1].records[0].global_episode_index == 0
    assert tasks[1].members[0].records[0].global_episode_index == 1


def test_task_plan_roundtrip_preserves_exact_source_references():
    from readers.roboomni_reader import RoboOmniPartition, _feature_specs
    from convert_core.episode_spec import DatasetConversionPlan

    fields = (RoboOmniField("observation/state", "numeric", (1,), "float32"),)
    vectors, cameras = _feature_specs(fields)
    record = RoboOmniRecord("component", "/raw/shard.tfrecord", 123, 456, 7, "pick", {}, 0, 0, 0)
    plan = DatasetConversionPlan("roboomni", Path("roboomni/schema"), 10, 10.0, "unknown", vectors, cameras, (), {})
    catalog = RoboOmniCatalog(
        Path("/raw"),
        (
            RoboOmniPartition(
                "schema", ("component",), fields, (record,), plan,
                ({"path": "/raw/shard.tfrecord", "size": 456, "mtime_ns": 1},),
            ),
        ),
        (), {}, (),
    )
    task = build_tasks(catalog)[0]
    payload = task_plan_payload(catalog, task, fps=10, output_dataset_uid="smoke", encoding={"encoder": "cpu"})
    restored = task_from_payload(payload, fps=10)
    restored_record = restored.members[0].records[0]
    assert restored.task_key == task.task_key
    assert (restored_record.source_file, restored_record.offset, restored_record.payload_length) == (record.source_file, record.offset, record.payload_length)
