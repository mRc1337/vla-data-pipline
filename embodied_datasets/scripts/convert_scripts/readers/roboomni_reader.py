"""Indexed reader for the RoboOmni/OmniAction RLDS release.

Each TFRecord contains one complete trajectory as a flat tf.train.Example:
steps/<field> stores one value per frame. The index records exact record
offsets and frame counts without decoding image payloads. Heterogeneous
source schemas remain separate LeRobot collection partitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from convert_core.episode_spec import CameraFeatureSpec, DatasetConversionPlan, EpisodePlan, VectorFeatureSpec
from convert_core.errors import ConversionError


@dataclass(frozen=True)
class RoboOmniField:
    source_key: str
    kind: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def output_key(self) -> str:
        return "source." + self.source_key.replace("/", ".")


@dataclass(frozen=True)
class RoboOmniRecord:
    component: str
    source_file: str
    offset: int
    payload_length: int
    frame_count: int
    instruction: str
    context: dict[str, Any]
    global_episode_index: int = 0
    global_frame_start: int = 0
    task_index: int = 0


@dataclass(frozen=True)
class RoboOmniPartition:
    name: str
    components: tuple[str, ...]
    fields: tuple[RoboOmniField, ...]
    records: tuple[RoboOmniRecord, ...]
    plan: DatasetConversionPlan
    source_files: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RoboOmniTaskMember:
    """One schema-partition slice belonging to a stable source task."""

    partition_name: str
    components: tuple[str, ...]
    fields: tuple[RoboOmniField, ...]
    records: tuple[RoboOmniRecord, ...]
    source_files: tuple[dict[str, Any], ...]
    episode_start: int
    episode_end: int
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class RoboOmniTask:
    """Stable task boundary used by the resumable orchestrator."""

    task_key: str
    instruction: str
    task_index: int
    members: tuple[RoboOmniTaskMember, ...]
    episode_count: int
    frame_count: int
    source_bytes: int


@dataclass(frozen=True)
class RoboOmniCatalog:
    source_root: Path
    partitions: tuple[RoboOmniPartition, ...]
    sidecar_inventory: tuple[dict[str, Any], ...]
    payload_scan_coverage: dict[str, Any]
    mapping_table: tuple[dict[str, Any], ...]
    audio_inventory: tuple[dict[str, Any], ...] = ()
    tasks: tuple[RoboOmniTask, ...] = ()


def _example_class() -> Any:
    T = descriptor_pb2.FieldDescriptorProto
    L, O = T.LABEL_REPEATED, T.LABEL_OPTIONAL
    proto = descriptor_pb2.FileDescriptorProto(
        name="roboomni_tf_example.proto", package="tf.train", syntax="proto3"
    )

    def add(name: str, fields: Iterable[tuple[str, int, int, int, str | None]]) -> Any:
        message = proto.message_type.add(name=name)
        for field_name, number, label, field_type, type_name in fields:
            field = message.field.add(
                name=field_name, number=number, label=label, type=field_type
            )
            if type_name:
                field.type_name = type_name
        return message

    add("BytesList", [("value", 1, L, T.TYPE_BYTES, None)])
    add("FloatList", [("value", 1, L, T.TYPE_FLOAT, None)])
    add("Int64List", [("value", 1, L, T.TYPE_INT64, None)])
    add("Feature", [
        ("bytes_list", 1, O, T.TYPE_MESSAGE, ".tf.train.BytesList"),
        ("float_list", 2, O, T.TYPE_MESSAGE, ".tf.train.FloatList"),
        ("int64_list", 3, O, T.TYPE_MESSAGE, ".tf.train.Int64List"),
    ])
    features = add("Features", [])
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, label=O, type=T.TYPE_STRING)
    entry.field.add(
        name="value", number=2, label=O, type=T.TYPE_MESSAGE,
        type_name=".tf.train.Feature"
    )
    features.field.add(
        name="feature", number=1, label=L, type=T.TYPE_MESSAGE,
        type_name=".tf.train.Features.FeatureEntry"
    )
    add("Example", [("features", 1, O, T.TYPE_MESSAGE, ".tf.train.Features")])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tf.train.Example")
    )


_EXAMPLE = _example_class()


def _feature_values(feature: Any) -> tuple[str, list[Any]]:
    if feature.HasField("bytes_list"):
        return "bytes", list(feature.bytes_list.value)
    if feature.HasField("float_list"):
        return "float32", list(feature.float_list.value)
    if feature.HasField("int64_list"):
        return "int64", list(feature.int64_list.value)
    raise ConversionError("TFRecord feature has no value list")


def parse_example(payload: bytes) -> dict[str, tuple[str, list[Any]]]:
    try:
        example = _EXAMPLE.FromString(payload)
    except Exception as exc:
        raise ConversionError(f"cannot parse TFRecord Example: {exc}") from exc
    return {key: _feature_values(value) for key, value in example.features.feature.items()}


def _read_record(path: Path, offset: int) -> tuple[int, bytes]:
    with path.open("rb") as stream:
        stream.seek(offset)
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ConversionError(f"short TFRecord length at {path}:{offset}")
        length = struct.unpack("<Q", raw_length)[0]
        if len(stream.read(4)) != 4:
            raise ConversionError(f"short TFRecord header at {path}:{offset}")
        payload = stream.read(length)
        if len(payload) != length or len(stream.read(4)) != 4:
            raise ConversionError(f"short TFRecord record at {path}:{offset}")
    return int(length), payload


def _dtype(dtype: str) -> str:
    return {"float32": "float32", "int32": "int32", "uint8": "uint8", "bool": "bool"}.get(dtype, dtype)


def _fields_from_sidecar(version_root: Path) -> tuple[RoboOmniField, ...]:
    path = version_root / "features.json"
    if not path.is_file():
        if version_root.parent.name != "bridge_one":
            raise ConversionError(f"missing features.json: {path}")
        reference = version_root.parent.parent / "bridge_identity" / "1.2.0" / "features.json"
        if not reference.is_file():
            raise ConversionError(f"bridge_one has no usable reference features.json: {path}")
        path = reference
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    fields: list[RoboOmniField] = []

    def walk(node: Mapping[str, Any], prefix: str = "") -> None:
        nested = node.get("featuresDict", {}).get("features")
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                walk(value, f"{prefix}/{key}" if prefix else str(key))
            return
        if "sequence" in node:
            walk(node["sequence"].get("feature", {}), f"{prefix}/*")
            return
        if not prefix.startswith("steps/*/"):
            return
        key = prefix.removeprefix("steps/*/")
        if "image" in node:
            image = node["image"]
            shape = tuple(int(x) for x in image.get("shape", {}).get("dimensions", []))
            fields.append(RoboOmniField(key, "image", shape, str(image["dtype"])))
        elif "text" in node:
            fields.append(RoboOmniField(key, "string", (1,), "string"))
        elif "tensor" in node:
            tensor = node["tensor"]
            shape = tuple(int(x) for x in tensor.get("shape", {}).get("dimensions", []))
            fields.append(RoboOmniField(key, "numeric", shape or (1,), _dtype(str(tensor["dtype"]))))
        elif "scalar" in node:
            scalar = node["scalar"].get("tensor", {})
            fields.append(RoboOmniField(key, "numeric", (1,), _dtype(str(scalar.get("dtype", "float32")))))

    walk(tree)
    if not fields:
        raise ConversionError(f"no per-step fields in {path}")
    return tuple(fields)


def _signature(fields: tuple[RoboOmniField, ...]) -> tuple[Any, ...]:
    return tuple((field.source_key, field.kind, field.shape, field.dtype) for field in fields)


def _sidecar_files(version_root: Path, *, max_shards: int | None = None) -> tuple[dict[str, Any], ...]:
    files = []
    paths = sorted(version_root.glob("*-train.tfrecord-*"))
    if max_shards is not None:
        paths = paths[:max_shards]
    for path in paths:
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not files:
        raise ConversionError(f"no train TFRecords below {version_root}")
    return tuple(files)


def _context_json(value: tuple[str, list[Any]]) -> Any:
    kind, values = value
    return [item.decode("utf-8") for item in values] if kind == "bytes" else values


def _speech_references(value: tuple[str, list[Any]] | None) -> tuple[str, ...]:
    """Return the external audio paths encoded by ``speech_conv``.

    The release uses ``[UNK]`` as a separator between one or more relative
    paths.  Keep the original field in the LeRobot data and normalize only
    the path used for the sidecar payload inventory.
    """

    if value is None:
        return ()
    kind, values = value
    if kind != "bytes":
        return ()
    references: list[str] = []
    for item in values:
        text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for reference in text.split("[UNK]"):
            reference = reference.strip()
            if reference.startswith("./"):
                reference = reference[2:]
            if reference and reference not in references:
                references.append(reference)
    return tuple(references)


def _wire_scan(path: Path, fields: tuple[RoboOmniField, ...]) -> Iterator[tuple[int, int, int, str, dict[str, Any]]]:
    offset = 0
    with path.open("rb") as stream:
        while True:
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8 or len(stream.read(4)) != 4:
                raise ConversionError(f"truncated TFRecord header: {path}:{offset}")
            length = struct.unpack("<Q", header)[0]
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ConversionError(f"truncated TFRecord payload: {path}:{offset}")
            values = parse_example(payload)
            counts = []
            for field in fields:
                kind, raw = values.get("steps/" + field.source_key, ("", []))
                if not raw:
                    raise ConversionError(f"record {path}:{offset} missing steps/{field.source_key}")
                width = int(np.prod(field.shape)) if field.kind == "numeric" else 1
                if len(raw) % width:
                    raise ConversionError(f"malformed steps/{field.source_key} at {path}:{offset}")
                counts.append(len(raw) // width)
            if len(set(counts)) != 1 or not counts[0]:
                raise ConversionError(f"heterogeneous or empty frame counts at {path}:{offset}: {counts}")
            instruction_value = values.get("steps/observation/natural_language_instruction") or values.get("steps/language_instruction")
            instruction = ""
            if instruction_value:
                for candidate in instruction_value[1]:
                    decoded = candidate.decode("utf-8")
                    if decoded:
                        instruction = decoded
                        break
            context = {key: _context_json(value) for key, value in values.items() if not key.startswith("steps/")}
            speech_references = _speech_references(values.get("steps/speech_conv"))
            if speech_references:
                context["audio_refs"] = list(speech_references)
            yield offset, int(length), counts[0], instruction, context
            offset += 8 + 4 + length + 4


def _feature_specs(fields: tuple[RoboOmniField, ...]) -> tuple[tuple[VectorFeatureSpec, ...], tuple[CameraFeatureSpec, ...]]:
    vectors, cameras = [], []
    for field in fields:
        if field.kind == "image":
            cameras.append(CameraFeatureSpec(_camera_key(field), field.shape[0], field.shape[1]))
        else:
            vectors.append(VectorFeatureSpec(field.output_key, int(np.prod(field.shape)), dtype=field.dtype, shape=field.shape))
    return tuple(vectors), tuple(cameras)


_PARTITION_RULES = ("one partition per exact features.json signature",)
_TIMEBASE = {
    "source_fps": None,
    "source_timestamps": "absent from inspected RLDS features",
    "output_timestamps": "derived by LeRobot writer from required CLI fps",
}


def _mapping(fields: tuple[RoboOmniField, ...]) -> tuple[dict[str, Any], ...]:
    return tuple({
        "source": "steps/" + field.source_key, "shape": list(field.shape), "dtype": field.dtype,
        "target": _camera_key(field) if field.kind == "image" else field.output_key,
        "conversion": "RGB video encode" if field.kind == "image" else "identity",
        "lossy": field.kind == "image",
    } for field in fields)


def _camera_key(field: RoboOmniField) -> str:
    names = {"observation/image": "primary", "observation/image_wrist": "wrist", "first_frame_image": "first_frame"}
    return "observation.images." + names.get(field.source_key, field.source_key.replace("/", "."))


def _audio_inventory(source_root: Path, records: Iterable[RoboOmniRecord]) -> tuple[dict[str, Any], ...]:
    """Inventory referenced speech files without making raw-data mutations."""

    speech_root = source_root / "speech"
    references = sorted({
        str(reference)
        for record in records
        for reference in record.context.get("audio_refs", [])
        if isinstance(reference, str)
    })
    inventory: list[dict[str, Any]] = []
    for reference in references:
        relative = Path(reference)
        valid = not relative.is_absolute() and ".." not in relative.parts
        path = speech_root / relative if valid else speech_root / "__invalid_reference__"
        row: dict[str, Any] = {
            "reference": reference,
            "relative_path": relative.as_posix() if valid else None,
            "source_path": str(path),
            "exists": False,
        }
        if valid and path.is_file():
            stat = path.stat()
            row.update(exists=True, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        inventory.append(row)
    return tuple(inventory)


def build_catalog(raw_root: Path, *, fps: int, max_shards: int | None = None, max_episodes: int | None = None, selected_tasks: set[str] | None = None) -> RoboOmniCatalog:
    source_root = next((raw_root / name for name in ("roboomni", "robomni") if (raw_root / name).is_dir()), None)
    if source_root is None:
        raise ConversionError(f"RoboOmni source directory is missing below {raw_root}")
    sidecars, grouped = [], {}
    total_records = selected_records = 0
    for version_root in sorted(source_root.glob("*/*")):
        if not version_root.is_dir():
            continue
        info_path = version_root / "dataset_info.json"
        if not info_path.is_file():
            continue
        fields = _fields_from_sidecar(version_root)
        source_files = _sidecar_files(version_root, max_shards=max_shards)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        sidecars.append({"component": version_root.parent.name, "version": version_root.name, "info": info, "fields": [field.__dict__ for field in fields], "files": list(source_files)})
        grouped.setdefault(_signature(fields), []).append((version_root.parent.name, fields, source_files))

    partitions = []
    for partition_index, group in enumerate(grouped.values()):
        if max_episodes is not None and selected_records >= max_episodes:
            break
        components, fields = tuple(item[0] for item in group), group[0][1]
        records, source_files = [], []
        shard_count = 0
        stop = False
        for component, group_fields, files in group:
            if _signature(group_fields) != _signature(fields):
                raise ConversionError(f"schema changed within partition {components}")
            for file_record in files:
                if max_shards is not None and shard_count >= max_shards:
                    stop = True
                    break
                shard_count += 1
                source_files.append(file_record)
                for offset, length, frames, instruction, context in _wire_scan(Path(file_record["path"]), fields):
                    if max_episodes is not None and selected_records >= max_episodes:
                        stop = True
                        break
                    if selected_tasks and instruction not in selected_tasks:
                        total_records += 1
                        continue
                    records.append(RoboOmniRecord(component, file_record["path"], offset, length, frames, instruction, context))
                    selected_records += 1
                    if max_episodes is not None and selected_records >= max_episodes:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if not records:
            continue
        tasks = list(dict.fromkeys(record.instruction for record in records))
        task_indices = {task: index for index, task in enumerate(tasks)}
        indexed, frame_cursor = [], 0
        for episode_index, record in enumerate(records):
            indexed.append(RoboOmniRecord(**{**record.__dict__, "global_episode_index": episode_index, "global_frame_start": frame_cursor, "task_index": task_indices[record.instruction]}))
            frame_cursor += record.frame_count
        vectors, cameras = _feature_specs(fields)
        name = "schema_%02d_%s" % (partition_index, "_".join(sorted(components)))
        episodes = tuple(EpisodePlan("episode_%08d" % index, "%s:%s@%d" % (record.component, Path(record.source_file).name, record.offset), record.instruction, record.frame_count, {"record": record.__dict__, "checkpoint_unit": record.source_file}) for index, record in enumerate(indexed))
        plan = DatasetConversionPlan("roboomni", Path("roboomni") / name, fps, float(fps), "unknown", vectors, cameras, episodes, {"source_root": str(source_root), "source_dataset": "fnlp/OmniAction", "source_splits": ["train"], "field_mapping": list(_mapping(fields)), "partition_rules": list(_PARTITION_RULES), "timebase": dict(_TIMEBASE), "payload_scan_coverage": {"mode": "TFRecord wire index; no image decode"}})
        partitions.append(RoboOmniPartition(name, components, fields, tuple(indexed), plan, tuple(source_files)))
    if not partitions:
        raise ConversionError("RoboOmni preflight selected zero episodes")
    selected = tuple(record for partition in partitions for record in partition.records)
    catalog = RoboOmniCatalog(source_root, tuple(partitions), tuple(sidecars), {"mode": "TFRecord wire index; no image decode", "records": total_records, "selected_records": selected_records}, tuple(row for partition in partitions for row in _mapping(partition.fields)), _audio_inventory(source_root, selected))
    return RoboOmniCatalog(**{**catalog.__dict__, "tasks": build_tasks(catalog)})


def task_key_for_instruction(instruction: str, task_index: int) -> str:
    """Return a filesystem-safe key whose ordering is independent of scan order."""

    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]
    return f"task-{task_index:06d}-{digest}"


def build_tasks(
    catalog: RoboOmniCatalog,
    *,
    max_episodes_per_task: int | None = None,
    selected_task_keys: set[str] | None = None,
    selected_instructions: set[str] | None = None,
) -> tuple[RoboOmniTask, ...]:
    """Group indexed records by the source instruction without another wire scan.

    The catalog is the one expensive wire/index pass.  This function only
    rearranges its immutable record references and assigns deterministic
    partition-local output ranges.  Records are ordered by task key and then
    by their original source position, so every task occupies whole chunks.
    """

    if max_episodes_per_task is not None and max_episodes_per_task <= 0:
        raise ConversionError("max_episodes_per_task must be positive")
    by_instruction: dict[str, list[tuple[int, RoboOmniPartition, RoboOmniRecord]]] = {}
    for partition_index, partition in enumerate(catalog.partitions):
        for record in partition.records:
            by_instruction.setdefault(record.instruction, []).append(
                (partition_index, partition, record)
            )

    instructions = sorted(by_instruction)
    selected: dict[str, list[tuple[int, RoboOmniPartition, RoboOmniRecord]]] = {}
    for task_index, instruction in enumerate(instructions):
        task_key = task_key_for_instruction(instruction, task_index)
        if selected_task_keys and task_key not in selected_task_keys and instruction not in (selected_instructions or set()):
            continue
        values = sorted(
            by_instruction[instruction],
            key=lambda item: (item[0], item[2].global_episode_index, item[2].source_file, item[2].offset),
        )
        if max_episodes_per_task is not None:
            values = values[:max_episodes_per_task]
        if values:
            selected[instruction] = values

    # Reassign output ranges after task selection.  The source records retain
    # their exact TFRecord offsets; only LeRobot's deterministic global indices
    # are changed for the selected collection ordering.
    by_partition: dict[int, list[tuple[str, RoboOmniPartition, RoboOmniRecord]]] = {}
    for instruction in instructions:
        task_index = instructions.index(instruction)
        for partition_index, partition, record in selected.get(instruction, ()):
            by_partition.setdefault(partition_index, []).append((instruction, partition, record))

    indexed_members: dict[tuple[str, int], list[RoboOmniRecord]] = {}
    partition_cursors: dict[int, tuple[int, int]] = {}
    for partition_index, values in by_partition.items():
        episode_cursor = frame_cursor = 0
        for instruction, _partition, record in values:
            task_index = instructions.index(instruction)
            updated = RoboOmniRecord(
                **{
                    **record.__dict__,
                    "global_episode_index": episode_cursor,
                    "global_frame_start": frame_cursor,
                    "task_index": task_index,
                }
            )
            indexed_members.setdefault((instruction, partition_index), []).append(updated)
            episode_cursor += 1
            frame_cursor += updated.frame_count
        partition_cursors[partition_index] = (episode_cursor, frame_cursor)

    tasks: list[RoboOmniTask] = []
    for task_index, instruction in enumerate(instructions):
        values = selected.get(instruction)
        if not values:
            continue
        members: list[RoboOmniTaskMember] = []
        for partition_index, partition in enumerate(catalog.partitions):
            records = tuple(indexed_members.get((instruction, partition_index), ()))
            if not records:
                continue
            source_paths = {record.source_file for record in records}
            source_files = tuple(
                item for item in partition.source_files if str(item.get("path")) in source_paths
            )
            members.append(
                RoboOmniTaskMember(
                    partition.name,
                    partition.components,
                    partition.fields,
                    records,
                    source_files,
                    records[0].global_episode_index,
                    records[-1].global_episode_index + 1,
                    records[0].global_frame_start,
                    records[-1].global_frame_start + records[-1].frame_count,
                )
            )
        tasks.append(
            RoboOmniTask(
                task_key_for_instruction(instruction, task_index),
                instruction,
                task_index,
                tuple(members),
                sum(len(member.records) for member in members),
                sum(member.frame_end - member.frame_start for member in members),
                sum(record.payload_length for member in members for record in member.records),
            )
        )
    return tuple(tasks)


def task_plan_payload(
    catalog: RoboOmniCatalog,
    task: RoboOmniTask,
    *,
    fps: int,
    output_dataset_uid: str,
    encoding: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize one self-contained immutable task plan."""

    def safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [safe(item) for item in value]
        if isinstance(value, list):
            return [safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if hasattr(value, "__dict__"):
            return safe(vars(value))
        return value

    return safe(
        {
            "schema_version": 1,
            "reader_format": "roboomni",
            "source_root": catalog.source_root,
            "task_key": task.task_key,
            "instruction": task.instruction,
            "task_index": task.task_index,
            "episode_count": task.episode_count,
            "frame_count": task.frame_count,
            "source_bytes": task.source_bytes,
            "fps": fps,
            "output_dataset_uid": output_dataset_uid,
            "encoding": dict(encoding),
            "timebase": dict(_TIMEBASE),
            "mapping_table": list(catalog.mapping_table),
            "members": [
                {
                    "partition_name": member.partition_name,
                    "components": member.components,
                    "fields": member.fields,
                    "records": member.records,
                    "source_files": member.source_files,
                    "episode_start": member.episode_start,
                    "episode_end": member.episode_end,
                    "frame_start": member.frame_start,
                    "frame_end": member.frame_end,
                }
                for member in task.members
            ],
        }
    )


def task_from_payload(payload: Mapping[str, Any], *, fps: int | None = None) -> RoboOmniTask:
    """Restore a task plan without touching any source TFRecord."""

    if payload.get("schema_version") != 1 or not isinstance(payload.get("members"), list):
        raise ConversionError("unsupported RoboOmni task plan")
    if fps is not None and payload.get("fps") != fps:
        raise ConversionError("RoboOmni task plan FPS changed")
    members: list[RoboOmniTaskMember] = []
    try:
        for raw in payload["members"]:
            fields = tuple(
                RoboOmniField(str(item["source_key"]), str(item["kind"]), tuple(int(x) for x in item["shape"]), str(item["dtype"]))
                for item in raw["fields"]
            )
            records = tuple(
                RoboOmniRecord(
                    str(item["component"]), str(item["source_file"]), int(item["offset"]),
                    int(item["payload_length"]), int(item["frame_count"]), str(item["instruction"]),
                    dict(item.get("context", {})), int(item.get("global_episode_index", 0)),
                    int(item.get("global_frame_start", 0)), int(item.get("task_index", 0)),
                )
                for item in raw["records"]
            )
            members.append(
                RoboOmniTaskMember(
                    str(raw["partition_name"]), tuple(str(x) for x in raw["components"]), fields,
                    records, tuple(dict(item) for item in raw["source_files"]),
                    int(raw["episode_start"]), int(raw["episode_end"]),
                    int(raw["frame_start"]), int(raw["frame_end"]),
                )
            )
        return RoboOmniTask(
            str(payload["task_key"]), str(payload["instruction"]), int(payload["task_index"]),
            tuple(members), int(payload["episode_count"]), int(payload["frame_count"]),
            int(payload["source_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"invalid RoboOmni task plan: {exc}") from exc


def task_catalog_payload(
    catalog: RoboOmniCatalog,
    *,
    output_dataset_uid: str,
    fps: int,
    task_plans: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Small startup catalog; full records live only in task_plan.json files."""

    return {
        "schema_version": 1,
        "reader_format": "roboomni",
        "source_root": str(catalog.source_root),
        "output_dataset_uid": output_dataset_uid,
        "fps": fps,
        "selection": dict(selection),
        "timebase": dict(_TIMEBASE),
        "partition_rules": list(_PARTITION_RULES),
        "tasks": [dict(item) for item in task_plans],
    }


def catalog_payload(catalog: RoboOmniCatalog) -> dict[str, Any]:
    def safe(value: Any) -> Any:
        if isinstance(value, Path): return str(value)
        if isinstance(value, tuple): return [safe(item) for item in value]
        if isinstance(value, list): return [safe(item) for item in value]
        if isinstance(value, dict): return {str(key): safe(item) for key, item in value.items()}
        if hasattr(value, "__dict__"): return safe(vars(value))
        return value
    return safe({"schema_version": 1, "source_root": catalog.source_root, "reader_format": "roboomni", "partition_rules": _PARTITION_RULES, "timebase": _TIMEBASE, "sidecars": catalog.sidecar_inventory, "payload_scan_coverage": catalog.payload_scan_coverage, "mapping_table": catalog.mapping_table, "audio_inventory": catalog.audio_inventory, "partitions": [{"name": p.name, "components": p.components, "fields": p.fields, "records": p.records, "source_files": p.source_files} for p in catalog.partitions]})


def decode_record(record: RoboOmniRecord, fields: tuple[RoboOmniField, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _length, payload = _read_record(Path(record.source_file), record.offset)
    values = parse_example(payload)
    result = []
    for index in range(record.frame_count):
        frame = {}
        for field in fields:
            kind, raw = values.get("steps/" + field.source_key, ("", []))
            if kind == "" or not raw:
                raise ConversionError(f"record missing steps/{field.source_key}")
            if field.kind == "image":
                try:
                    with Image.open(io.BytesIO(raw[index])) as image:
                        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
                except Exception as exc:
                    raise ConversionError(f"cannot decode image steps/{field.source_key}: {exc}") from exc
                if tuple(value.shape) != field.shape:
                    raise ConversionError(f"image steps/{field.source_key} has shape {value.shape}, expected {field.shape}")
                frame[_camera_key(field)] = value
            elif field.kind == "string":
                frame[field.output_key] = raw[index].decode("utf-8")
            else:
                width = int(np.prod(field.shape))
                frame[field.output_key] = np.asarray(raw[index * width:(index + 1) * width], dtype=field.dtype).reshape(field.shape)
        frame["task"] = record.instruction
        result.append(frame)
    return result, {key: value for key, value in values.items() if not key.startswith("steps/")}


def iter_record_frames(record: RoboOmniRecord, fields: tuple[RoboOmniField, ...]) -> Iterator[dict[str, Any]]:
    yield from decode_record(record, fields)[0]


def catalog_from_payload(payload: dict[str, Any], *, fps: int) -> RoboOmniCatalog:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("partitions"), list):
        raise ConversionError("unsupported RoboOmni preflight catalog")
    partitions = []
    try:
        source_root = Path(str(payload["source_root"]))
        for raw in payload["partitions"]:
            fields = tuple(RoboOmniField(str(item["source_key"]), str(item["kind"]), tuple(int(x) for x in item["shape"]), str(item["dtype"])) for item in raw["fields"])
            records = tuple(RoboOmniRecord(str(item["component"]), str(item["source_file"]), int(item["offset"]), int(item["payload_length"]), int(item["frame_count"]), str(item["instruction"]), dict(item.get("context", {})), int(item.get("global_episode_index", 0)), int(item.get("global_frame_start", 0)), int(item.get("task_index", 0))) for item in raw["records"])
            vectors, cameras = _feature_specs(fields)
            episodes = tuple(EpisodePlan("episode_%08d" % i, "%s:%s@%d" % (r.component, Path(r.source_file).name, r.offset), r.instruction, r.frame_count, {"record": r.__dict__, "checkpoint_unit": r.source_file}) for i, r in enumerate(records))
            name = str(raw["name"])
            plan = DatasetConversionPlan("roboomni", Path("roboomni") / name, fps, float(fps), "unknown", vectors, cameras, episodes, {"source_root": str(source_root), "source_dataset": "fnlp/OmniAction", "source_splits": ["train"], "field_mapping": list(_mapping(fields)), "partition_rules": list(payload.get("partition_rules", _PARTITION_RULES)), "timebase": dict(payload.get("timebase", _TIMEBASE)), "payload_scan_coverage": payload.get("payload_scan_coverage", {})})
            partitions.append(RoboOmniPartition(name, tuple(str(x) for x in raw["components"]), fields, records, plan, tuple(dict(x) for x in raw["source_files"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"invalid RoboOmni preflight catalog: {exc}") from exc
    catalog = RoboOmniCatalog(source_root, tuple(partitions), tuple(dict(x) for x in payload.get("sidecars", [])), dict(payload.get("payload_scan_coverage", {})), tuple(dict(x) for x in payload.get("mapping_table", [])), tuple(dict(x) for x in payload.get("audio_inventory", [])))
    return RoboOmniCatalog(**{**catalog.__dict__, "tasks": build_tasks(catalog)})


def validate_catalog_sources(catalog: RoboOmniCatalog) -> None:
    for partition in catalog.partitions:
        for record in partition.source_files:
            path = Path(str(record["path"]))
            stat = path.stat()
            if stat.st_size != int(record["size"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
                raise ConversionError(f"RoboOmni source changed since preflight: {path}")
    for record in catalog.audio_inventory:
        if not record.get("exists"):
            continue
        path = Path(str(record["source_path"]))
        stat = path.stat()
        if stat.st_size != int(record["size"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ConversionError(f"RoboOmni audio source changed since preflight: {path}")
