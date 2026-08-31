"""Inspection and streaming reader for the Functional Manipulation Benchmark.

The public FMB release is not an RLDS dataset: each trajectory is a compressed
NumPy object array inside a ZIP archive, or the same member as an extracted
``.npy`` file.  The object array contains a dict of complete per-frame arrays.
This reader therefore uses a lazy pickle
unpickler during preflight.  It consumes compressed bytes to reach each
pickle's array metadata, but never materialises the array payload; only the
first, middle, and last selected trajectories are fully decoded for payload
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil
import struct
import time
from typing import Any, Callable, Iterator
import zipfile

import numpy as np

from convert_core.episode_spec import (
    CameraFeatureSpec,
    DatasetConversionPlan,
    EpisodePlan,
    VectorFeatureSpec,
)
from convert_core.errors import ConversionError
from convert_core.dataset_config import DatasetConversionConfig
from readers.base import DatasetReader


FMB_FPS = 10
FMB_HEIGHT = 256
FMB_WIDTH = 256
_MULTI_RE = re.compile(r"trajectory_(?P<object>\d+)_(?P<trajectory>\d+)\.npy$")

_RGB_KEYS = (
    ("obs/side_1", "observation.images.side_1"),
    ("obs/side_2", "observation.images.side_2"),
    ("obs/wrist_1", "observation.images.wrist_1"),
    ("obs/wrist_2", "observation.images.wrist_2"),
)
_DEPTH_KEYS = (
    ("obs/side_1_depth", "observation.depth.side_1"),
    ("obs/side_2_depth", "observation.depth.side_2"),
    ("obs/wrist_1_depth", "observation.depth.wrist_1"),
    ("obs/wrist_2_depth", "observation.depth.wrist_2"),
)
_VECTOR_KEYS = (
    ("obs/tcp_pose", "observation.tcp_pose"),
    ("obs/tcp_vel", "observation.tcp_vel"),
    ("obs/tcp_force", "observation.tcp_force"),
    ("obs/tcp_torque", "observation.tcp_torque"),
    ("obs/q", "observation.q"),
    ("obs/dq", "observation.dq"),
    ("obs/jacobian", "observation.jacobian"),
    ("obs/gripper_pose", "observation.gripper_pose"),
    ("action", "action"),
)
_RGB_SOURCE_KEYS = frozenset(key for key, _feature_key in _RGB_KEYS)
_OBJECT_INFO_KEY = "object_info"
_OBJECT_NAMES = {
    1: "rectangle",
    2: "round",
    3: "oval",
    4: "hexagon",
    5: "arch",
    6: "square-circle",
    7: "double-square",
    8: "3 prong",
    9: "star",
}
_BOARD_OBJECT_NAMES = {
    1: {1: "purple", 2: "blue", 3: "yellow", 4: "green"},
    2: {1: "green", 2: "brown", 3: "blue", 4: "red"},
    3: {1: "green", 2: "blue", 3: "purple", 4: "red"},
}

# The extracted release keeps the archive member paths below a directory
# named after each original shard.  Keep the archive names as the logical
# provenance/checkpoint unit while resolving them to either a ZIP file or an
# extracted directory at read time.
_EXTRACTED_DIRS = {
    "multi_object_manipulation_assembly_1.zip": "assembly_1",
    "multi_object_manipulation_assembly_2.zip": "assembly_2",
    "multi_object_manipulation_assembly_3.zip": "assembly_3",
    "single_object_manipulation.zip": "single_object",
}


def validate_extracted_fmb_root(raw_root: Path) -> Path:
    """Require the complete extracted FMB layout for the production CLI."""

    if not raw_root.is_dir():
        raise ConversionError(f"FMB extracted raw root is missing: {raw_root}")
    unexpected_archives = sorted(path.name for path in raw_root.glob("*.zip"))
    if unexpected_archives:
        raise ConversionError(
            "production FMB conversion accepts extracted shards only; remove or "
            f"relocate ZIP files from {raw_root}: {unexpected_archives}"
        )
    missing: list[str] = []
    for shard_name in _EXTRACTED_DIRS.values():
        shard = raw_root / shard_name
        if not shard.is_dir() or not (shard / ".EXTRACTED_OK").is_file():
            missing.append(f"{shard} (including .EXTRACTED_OK)")
    if missing:
        raise ConversionError(
            "incomplete FMB extracted raw root; missing completed shards: "
            + ", ".join(missing)
        )
    return raw_root


@dataclass(frozen=True)
class _DirectoryMember:
    filename: str
    file_size: int
    compress_size: int = 0
    CRC: int = 0


def _source_for_archive(raw_root: Path, archive_name: str) -> tuple[Path, str]:
    """Return the physical ZIP or extracted directory for a logical shard."""

    archive = raw_root / archive_name
    if archive.is_file():
        return archive, "zip"
    extracted_name = _EXTRACTED_DIRS.get(archive_name)
    if extracted_name is not None:
        extracted = raw_root / extracted_name
        if extracted.is_dir():
            return extracted, "extracted_directory"
    raise ConversionError(
        f"FMB source shard is missing: expected {archive} or "
        f"{raw_root / _EXTRACTED_DIRS.get(archive_name, '<extracted-dir>')}"
    )


def _copy_source_stream(
    source: Any,
    destination: Any,
    *,
    expected_bytes: int,
    progress: Callable[[str], None] | None,
    label: str,
) -> int:
    """Copy one source member to POSIX storage with bounded memory and progress."""

    started = time.monotonic()
    copied = 0
    last_report = started
    while chunk := source.read(16 * 1024 * 1024):
        destination.write(chunk)
        copied += len(chunk)
        now = time.monotonic()
        if progress is not None and (now - last_report >= 5.0 or copied == expected_bytes):
            rate = copied / max(now - started, 1e-9)
            progress(
                f"{label}: {copied}/{expected_bytes} bytes "
                f"({rate / 1024 / 1024:.1f} MiB/s)"
            )
            last_report = now
    destination.flush()
    if copied != expected_bytes:
        raise ConversionError(
            f"staged FMB source size differs for {label}: {copied} != {expected_bytes}"
        )
    return copied


def _stage_source_member(
    source_path: Path,
    source_mode: str,
    member: str,
    destination: Path,
    *,
    expected_bytes: int,
    progress: Callable[[str], None] | None = None,
    label: str,
) -> Path:
    """Copy one ZIP member or extracted NPY to a local POSIX file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as outgoing:
            if source_mode == "zip":
                with zipfile.ZipFile(source_path) as source:
                    with source.open(member) as incoming:
                        _copy_source_stream(
                            incoming,
                            outgoing,
                            expected_bytes=expected_bytes,
                            progress=progress,
                            label=label,
                        )
            else:
                member_path = (source_path / member).resolve()
                if not member_path.is_relative_to(source_path.resolve()) or not member_path.is_file():
                    raise FileNotFoundError(member_path)
                with member_path.open("rb") as incoming:
                    _copy_source_stream(
                        incoming,
                        outgoing,
                        expected_bytes=expected_bytes,
                        progress=progress,
                        label=label,
                    )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def stage_fmb_unit_sources(
    plan: DatasetConversionPlan,
    raw_root: Path,
    destination_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Copy only a unit's NPY members to a local POSIX source tree.

    The conversion worker must not repeatedly decode large NPY payloads over
    OSSFS.  The staged tree deliberately uses the extracted-directory layout,
    even when the source is still a ZIP, so the existing reader can consume it
    without changing the logical episode provenance or run fingerprint.
    """

    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    episodes = plan.episodes
    seen: set[tuple[str, str]] = set()
    for index, episode in enumerate(episodes, start=1):
        archive = str(episode.extra["archive"])
        member = str(episode.extra["member"])
        identity = (archive, member)
        if identity in seen:
            continue
        seen.add(identity)
        source_path, source_mode = _source_for_archive(raw_root, archive)
        shard_name = _EXTRACTED_DIRS.get(archive, Path(archive).stem)
        target = destination_root / shard_name / member
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_bytes = int(episode.extra["source_uncompressed_bytes"])
        label = f"{index}/{len(episodes)} {archive}:{member}"
        if progress is not None:
            progress(f"staging {label}")
        try:
            _stage_source_member(
                source_path,
                source_mode,
                member,
                target,
                expected_bytes=expected_bytes,
                progress=progress,
                label=label,
            )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        if progress is not None:
            progress(f"staged {label} -> {target}")
    return destination_root


def _source_shards(raw_root: Path) -> list[tuple[str, Path, str]]:
    """Discover the four known FMB shards in either supported layout."""

    shards: list[tuple[str, Path, str]] = []
    for archive_name in sorted(_EXTRACTED_DIRS, key=str.casefold):
        try:
            source, mode = _source_for_archive(raw_root, archive_name)
        except ConversionError:
            continue
        shards.append((archive_name, source, mode))
    known_names = {name for name, _source, _mode in shards}
    for archive in sorted(raw_root.glob("*.zip"), key=lambda path: path.name.casefold()):
        if archive.name not in known_names:
            shards.append((archive.name, archive, "zip"))
    return shards


def _directory_members(source: Path) -> list[tuple[_DirectoryMember, Path]]:
    members: list[tuple[_DirectoryMember, Path]] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.name == ".EXTRACTED_OK":
            continue
        member = path.relative_to(source).as_posix()
        if not member.endswith(".npy"):
            raise ConversionError(f"unexpected non-NPY FMB extracted member: {path}")
        stat = path.stat()
        members.append((_DirectoryMember(member, stat.st_size), path))
    return members


def _normalize_object_info(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError(f"{source}: object_info must be a dictionary")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.generic):
            item = item.item()
        if not isinstance(key, str) or not isinstance(item, (str, int, float, bool, type(None))):
            raise ConversionError(f"{source}: object_info must contain string keys and scalar values")
        normalized[key] = item
    return normalized


@dataclass(frozen=True)
class FmbEntry:
    archive: str
    member: str
    kind: str
    board: str | None
    object_id: int | None
    trajectory_id: int | None
    size: int
    compressed_size: int
    crc: int
    num_frames: int
    fields: dict[str, dict[str, Any]]
    instruction: str
    object_info: dict[str, Any] | None = None
    source_mode: str = "zip"
    source_mtime_ns: int = 0


@dataclass(frozen=True)
class FmbPartition:
    name: str
    plan: DatasetConversionPlan
    entries: tuple[FmbEntry, ...]
    schema_fingerprint: str


@dataclass(frozen=True)
class FmbCatalog:
    partitions: tuple[FmbPartition, ...]
    fingerprint_payload: dict[str, Any]
    mapping_table: tuple[dict[str, Any], ...]
    sample_evidence: tuple[dict[str, Any], ...]


class _LazyArray:
    """Pickle target retaining ndarray shape/dtype but discarding raw bytes."""

    def __init__(self) -> None:
        self.state: tuple[Any, ...] | None = None

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.state = state

    @property
    def shape(self) -> tuple[int, ...]:
        if self.state is None:
            raise ConversionError("FMB pickle array has no state")
        return tuple(int(value) for value in self.state[1])

    @property
    def dtype(self) -> np.dtype[Any]:
        if self.state is None:
            raise ConversionError("FMB pickle array has no state")
        return np.dtype(self.state[2])

    def item(self) -> Any:
        if self.state is None or not self.shape == ():
            raise ConversionError("FMB top-level object is not a scalar object array")
        values = self.state[4]
        if not isinstance(values, list) or len(values) != 1:
            raise ConversionError("FMB top-level object array has an invalid payload")
        return values[0]


class _MetadataUnpickler(pickle._Unpickler):
    """Unpickle ndarray metadata while seeking past BINBYTES payloads."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def find_class(self, module: str, name: str) -> Any:
        if module.endswith(".multiarray") and name == "_reconstruct":
            return lambda *_args: _LazyArray()
        return super().find_class(module, name)

    def _skip_bytes(self, size: int, *, bytearray_value: bool = False) -> None:
        if size < 0:
            raise ConversionError("negative byte payload in FMB pickle")
        if size <= 1024 * 1024:
            value = self.read(size)
            if len(value) != size:
                raise ConversionError("truncated byte payload in FMB pickle")
            self.stack.append(bytearray(value) if bytearray_value else value)
            return
        remaining = size
        while remaining:
            chunk = self.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ConversionError("truncated byte payload in FMB pickle")
            remaining -= len(chunk)
        self.stack.append(b"")

    def load_binbytes_metadata(self) -> None:
        self._skip_bytes(struct.unpack("<I", self.read(4))[0])

    def load_short_binbytes_metadata(self) -> None:
        self._skip_bytes(self.read(1)[0])

    def load_binbytes8_metadata(self) -> None:
        self._skip_bytes(struct.unpack("<Q", self.read(8))[0])

    def load_bytearray8_metadata(self) -> None:
        self._skip_bytes(struct.unpack("<Q", self.read(8))[0], bytearray_value=True)


_MetadataUnpickler.dispatch[ord(pickle.BINBYTES)] = _MetadataUnpickler.load_binbytes_metadata
_MetadataUnpickler.dispatch[ord(pickle.SHORT_BINBYTES)] = _MetadataUnpickler.load_short_binbytes_metadata
_MetadataUnpickler.dispatch[ord(pickle.BINBYTES8)] = _MetadataUnpickler.load_binbytes8_metadata
if hasattr(pickle, "BYTEARRAY8"):
    _MetadataUnpickler.dispatch[ord(pickle.BYTEARRAY8)] = _MetadataUnpickler.load_bytearray8_metadata


def _read_lazy_metadata(stream: Any) -> dict[str, Any]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        np.lib.format.read_array_header_3_0(stream)
    else:
        raise ConversionError(f"unsupported FMB NPY version: {version}")
    value = _MetadataUnpickler(stream).load().item()
    if not isinstance(value, dict):
        raise ConversionError("FMB trajectory payload is not a dictionary")
    result: dict[str, Any] = {}
    for key, array in value.items():
        if not isinstance(key, str):
            raise ConversionError("FMB trajectory dictionary contains an invalid field")
        if key == _OBJECT_INFO_KEY:
            result[key] = {
                "kind": "object_info",
                "value": _normalize_object_info(array, "FMB object_info"),
            }
        elif isinstance(array, _LazyArray):
            result[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        else:
            raise ConversionError("FMB trajectory dictionary contains an invalid field")
    return _canonicalize_action_field(result)


def _canonicalize_action_field(fields: dict[str, Any]) -> dict[str, Any]:
    """Use the official singular ``action`` name while accepting old exports."""

    if "action" in fields and "actions" in fields:
        raise ConversionError("FMB trajectory contains both action and actions fields")
    if "actions" not in fields:
        return fields
    canonical = dict(fields)
    canonical["action"] = canonical.pop("actions")
    return canonical


def _load_payload(
    entry: FmbEntry,
    raw_root: Path,
    *,
    local_member_path: Path | None = None,
) -> dict[str, np.ndarray]:
    source_path, source_mode = _source_for_archive(raw_root, entry.archive) if local_member_path is None else (local_member_path, "local")
    try:
        if local_member_path is not None:
            with local_member_path.open("rb") as stream:
                value = np.load(stream, allow_pickle=True).item()
        elif source_mode == "zip":
            with zipfile.ZipFile(source_path) as source, source.open(entry.member) as stream:
                value = np.load(stream, allow_pickle=True).item()
        else:
            member_path = (source_path / entry.member).resolve()
            if not member_path.is_relative_to(source_path.resolve()) or not member_path.is_file():
                raise FileNotFoundError(member_path)
            with member_path.open("rb") as stream:
                value = np.load(stream, allow_pickle=True).item()
    except (OSError, ValueError, zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ConversionError(f"cannot decode FMB trajectory {source_path}:{entry.member}: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ConversionError(f"FMB trajectory is not a string-keyed dictionary: {entry.member}")
    canonical = _canonicalize_action_field(value)
    arrays: dict[str, np.ndarray] = {}
    for key, array in canonical.items():
        if key == _OBJECT_INFO_KEY:
            _normalize_object_info(array, f"FMB object_info in {entry.member}")
            continue
        if not isinstance(array, np.ndarray):
            raise ConversionError(f"FMB field {key} is not a NumPy array: {entry.member}")
        arrays[key] = array
    return arrays


def _entry_identity(info: zipfile.ZipInfo, archive: Path, raw_root: Path) -> str:
    return f"{archive.relative_to(raw_root).as_posix()}:{info.filename}:{info.file_size}:{info.CRC}"


def _object_info_from_fields(fields: dict[str, Any]) -> dict[str, Any] | None:
    metadata = fields.get(_OBJECT_INFO_KEY)
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("kind") != "object_info" or not isinstance(metadata.get("value"), dict):
        raise ConversionError("FMB object_info metadata has an invalid representation")
    return dict(metadata["value"])


def _single_object_instruction(member: str, stem: str, object_info: dict[str, Any] | None) -> str:
    tokens = stem.split("_")
    insert_only = stem.startswith("insert_only_") or "insert_only" in Path(member).parts
    shape_value: Any = object_info.get("shape") if object_info else None
    if shape_value is None:
        shape_index = 2 if insert_only and len(tokens) > 2 else 0
        if len(tokens) <= shape_index:
            raise ConversionError(f"cannot parse single-object shape from FMB member: {member}")
        shape_value = tokens[shape_index]
    try:
        shape_id = int(shape_value)
        object_name = _OBJECT_NAMES[shape_id]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"unknown single-object shape {shape_value!r} in FMB member: {member}") from exc
    if insert_only:
        return f"Insert the {object_name} object."
    return f"Pick up the {object_name} object and insert it."


def _multi_object_instruction(member: str, board: str, object_id: int) -> str:
    try:
        board_id = int(board.removeprefix("board_"))
        object_name = _BOARD_OBJECT_NAMES[board_id][object_id]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(
            f"unknown multi-object board/object mapping board={board!r}, object={object_id!r} in {member}"
        ) from exc
    return f"Pick up the {object_name} object and insert it."


def _parse_entry(
    info: zipfile.ZipInfo | _DirectoryMember,
    archive: Path,
    raw_root: Path,
    fields: dict[str, dict[str, Any]],
    *,
    source_mode: str,
    source_mtime_ns: int,
) -> FmbEntry:
    member = info.filename
    if not member.endswith(".npy"):
        raise ConversionError(f"unexpected non-NPY FMB archive member: {archive}:{member}")
    parts = Path(member).parts
    if "single_object_manipulation" in parts:
        kind = "single_object"
        board = None
        match = None
        stem = Path(member).stem
        trajectory_id = int(stem.rsplit("_", 1)[1]) if stem.rsplit("_", 1)[-1].isdigit() else None
        object_id = None
        tokens = stem.split("_")
        if tokens and tokens[0] == "insert" and len(tokens) >= 8 and tokens[1] == "only":
            tokens = tokens[2:]
        object_info = (
            {key: tokens[index] for index, key in enumerate(("shape", "size", "length", "color", "angle", "distractor"))}
            if len(tokens) >= 7 and tokens[-1].isdigit()
            else None
        )
        object_info = _object_info_from_fields(fields) or object_info
        instruction = _single_object_instruction(member, stem, object_info)
    elif "multi_object_manipulation" in parts:
        kind = "multi_object"
        board = next((part for part in parts if part.startswith("board_")), None)
        match = _MULTI_RE.search(Path(member).name)
        if match is None or board is None:
            raise ConversionError(f"cannot parse FMB multi-object member name: {member}")
        object_id = int(match.group("object"))
        trajectory_id = int(match.group("trajectory"))
        instruction = _multi_object_instruction(member, board, object_id)
        object_info = None
    else:
        raise ConversionError(f"cannot identify FMB dataset kind from member: {member}")
    return FmbEntry(
        archive=archive.relative_to(raw_root).as_posix(),
        member=member,
        kind=kind,
        board=board,
        object_id=object_id,
        trajectory_id=trajectory_id,
        size=info.file_size,
        compressed_size=info.compress_size,
        crc=info.CRC,
        num_frames=int(next(spec for key, spec in fields.items() if key != _OBJECT_INFO_KEY)["shape"][0]),
        fields=fields,
        instruction=instruction,
        object_info=object_info,
        source_mode=source_mode,
        source_mtime_ns=source_mtime_ns,
    )


def _validate_fields(
    fields: dict[str, dict[str, Any]],
    *,
    source: str,
    require_object_id: bool = False,
) -> None:
    required = {key for key, _ in _RGB_KEYS + _DEPTH_KEYS + _VECTOR_KEYS} | {"primitive"}
    missing = required - set(fields)
    if missing:
        raise ConversionError(f"{source}: missing required FMB fields: {sorted(missing)}")
    known = required | {"object_id", _OBJECT_INFO_KEY}
    unknown = set(fields) - known
    if unknown:
        raise ConversionError(f"{source}: unsupported FMB fields would be dropped: {sorted(unknown)}")
    array_specs = [spec for key, spec in fields.items() if key != _OBJECT_INFO_KEY]
    if not array_specs:
        raise ConversionError(f"{source}: trajectory has no array fields")
    first = array_specs[0]["shape"][0]
    if first <= 0:
        raise ConversionError(f"{source}: trajectory has no frames")
    for key, spec in fields.items():
        if key == _OBJECT_INFO_KEY:
            if spec.get("kind") != "object_info" or not isinstance(spec.get("value"), dict):
                raise ConversionError(f"{source}: invalid object_info metadata")
            continue
        shape = tuple(spec["shape"])
        if not shape or shape[0] != first:
            raise ConversionError(f"{source}: field {key} has inconsistent leading dimension {shape}")
    for key, _feature_key in _RGB_KEYS:
        if tuple(fields[key]["shape"]) != (first, 256, 256, 3) or fields[key]["dtype"] != "uint8":
            raise ConversionError(
                f"{source}: field {key} schema is {fields[key]}, "
                f"expected {(first, 256, 256, 3)}/uint8"
            )
    for key, _feature_key in _DEPTH_KEYS:
        if tuple(fields[key]["shape"]) != (first, 256, 256) or fields[key]["dtype"] != "uint16":
            raise ConversionError(
                f"{source}: field {key} schema is {fields[key]}, "
                f"expected {(first, 256, 256)}/uint16"
            )
    expected_vectors = {
        "obs/tcp_pose": (7,),
        "obs/tcp_vel": (6,),
        "obs/tcp_force": (3,),
        "obs/tcp_torque": (3,),
        "obs/q": (7,),
        "obs/dq": (7,),
        "obs/jacobian": (6, 7),
        "action": (7,),
    }
    for key, suffix_shape in expected_vectors.items():
        expected_shape = (first, *suffix_shape)
        if tuple(fields[key]["shape"]) != expected_shape or fields[key]["dtype"] != "float64":
            raise ConversionError(
                f"{source}: field {key} schema is {fields[key]}, "
                f"expected {expected_shape}/float64"
            )
    if tuple(fields["obs/gripper_pose"]["shape"]) != (first,) or fields["obs/gripper_pose"]["dtype"] != "int64":
        raise ConversionError(
            f"{source}: field obs/gripper_pose schema is {fields['obs/gripper_pose']}, "
            f"expected {(first,)}/int64"
        )
    if tuple(fields["primitive"]["shape"]) != (first,) or not str(fields["primitive"]["dtype"]).startswith("<U"):
        raise ConversionError(
            f"{source}: field primitive schema is {fields['primitive']}, "
            f"expected {(first,)}/Unicode string"
        )
    if require_object_id:
        if "object_id" not in fields:
            raise ConversionError(f"{source}: multi-object trajectory is missing object_id")
        if tuple(fields["object_id"]["shape"]) != (first,) or fields["object_id"]["dtype"] != "int64":
            raise ConversionError(
                f"{source}: field object_id schema is {fields['object_id']}, "
                f"expected {(first,)}/int64"
            )
    elif "object_id" in fields:
        raise ConversionError(f"{source}: single-object trajectory unexpectedly contains object_id")
    if require_object_id and _OBJECT_INFO_KEY in fields:
        raise ConversionError(f"{source}: multi-object trajectory unexpectedly contains object_info")


def _schema_fingerprint(fields: dict[str, dict[str, Any]]) -> str:
    # NumPy stores Unicode arrays with a fixed-width dtype (for example,
    # ``<U3`` and ``<U17``).  That width is an implementation detail of each
    # trajectory, not a change to the logical FMB string feature.  Keep the
    # original dtype in the catalog for strict payload validation, but avoid
    # creating one LeRobot partition per observed maximum label length.
    normalized = {}
    for key, spec in fields.items():
        if key == _OBJECT_INFO_KEY:
            normalized[key] = {"kind": "object_info"}
            continue
        shape = list(spec["shape"])
        # The first dimension is the episode length, not a feature-schema
        # dimension.  A fingerprint must therefore compare it symbolically.
        shape[0] = "N"
        normalized[key] = {
            **spec,
            "shape": shape,
            "dtype": "string" if key == "primitive" else spec["dtype"],
        }
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()


def _plan_for_partition(
    config: DatasetConversionConfig,
    raw_root: Path,
    partition_name: str,
    entries: tuple[FmbEntry, ...],
    schema_fingerprint: str,
) -> DatasetConversionPlan:
    sample = entries[0]
    vectors: list[VectorFeatureSpec] = []
    for source_key, feature_key in _DEPTH_KEYS + _VECTOR_KEYS:
        spec = sample.fields[source_key]
        shape = tuple(int(value) for value in spec["shape"][1:])
        vectors.append(VectorFeatureSpec(feature_key, int(np.prod(shape)), dtype=spec["dtype"], shape=shape or (1,)))
    primitive = sample.fields["primitive"]
    vectors.append(VectorFeatureSpec("observation.primitive", 1, dtype="string", shape=(1,)))
    if sample.kind == "multi_object":
        vectors.append(VectorFeatureSpec("observation.object_id", 1, dtype="int64", shape=(1,)))
    cameras = tuple(CameraFeatureSpec(feature_key, FMB_HEIGHT, FMB_WIDTH) for _source, feature_key in _RGB_KEYS)
    episodes = tuple(
        EpisodePlan(
            episode_uid=f"{entry.kind}/{entry.board or 'single'}/{entry.trajectory_id or index}/{index}",
            source_relative_path=f"{entry.archive}:{entry.member}",
            instruction=entry.instruction,
            num_frames=entry.num_frames,
            extra={
                "archive": entry.archive,
                "member": entry.member,
                "kind": entry.kind,
                "board": entry.board,
                "object_id": entry.object_id,
                "trajectory_id": entry.trajectory_id,
                "object_info": entry.object_info,
                "source_uncompressed_bytes": entry.size,
                "source_compressed_bytes": entry.compressed_size,
                "source_crc32": entry.crc,
                "source_mode": entry.source_mode,
                "source_mtime_ns": entry.source_mtime_ns,
                "checkpoint_unit": entry.archive,
                "manifest_provenance": {
                    "archive": entry.archive,
                    "member": entry.member,
                    "kind": entry.kind,
                    "board": entry.board,
                    "object_id": entry.object_id,
                    "trajectory_id": entry.trajectory_id,
                    "object_info": entry.object_info,
                },
            },
        )
        for index, entry in enumerate(entries)
    )
    mapping: list[dict[str, Any]] = []
    for source_key, feature_key in _RGB_KEYS + _DEPTH_KEYS + _VECTOR_KEYS:
        spec = sample.fields[source_key]
        mapping.append({
            "source_field": source_key,
            "shape_dtype": f"{spec['shape']}/{spec['dtype']}",
            "semantics": "official FMB dataset-card field; command/state arrays are preserved by name",
            "lerobot_field": feature_key,
            "conversion": (
                "BGR channel reversal followed by H.264 video encoding"
                if source_key in {key for key, _feature_key in _RGB_KEYS}
                else "no numeric conversion"
            ),
            "evidence": "official dataset card and lazy NPY payload schema scan",
            "lossy": source_key in {key for key, _feature_key in _RGB_KEYS},
        })
    mapping.extend([
        {"source_field": "primitive", "shape_dtype": "[N]/<U*", "semantics": "per-frame primitive label", "lerobot_field": "observation.primitive", "conversion": "UTF-8 string preserved as a LeRobot string feature", "evidence": "payload schema", "lossy": False},
        {"source_field": "object_info", "shape_dtype": "filename metadata/object", "semantics": "single-object shape/size/length/color/angle/distractor attributes", "lerobot_field": "conversion_manifest episode provenance", "conversion": "parse filename tokens without changing values", "evidence": "official dataset card filename convention", "lossy": False},
    ])
    if sample.kind == "multi_object":
        mapping.append({"source_field": "object_id", "shape_dtype": "[N]/int64", "semantics": "object ID selected at each timestep", "lerobot_field": "observation.object_id", "conversion": "none", "evidence": "payload schema", "lossy": False})
    return DatasetConversionPlan(
        dataset_uid=partition_name,
        output_path=raw_root / "__output_placeholder__" / partition_name,
        fps=FMB_FPS,
        measured_fps=float(FMB_FPS),
        robot_type=config.robot_type,
        vector_features=tuple(vectors),
        camera_features=cameras,
        episodes=episodes,
        extra={
            "source_dataset": "Functional Manipulation Benchmark (FMB)",
            "source_root": str(raw_root),
            "source_kind": sample.kind,
            "source_schema_fingerprint": schema_fingerprint,
            "source_fps_evidence": "paper section 2.1: SpaceMouse commands at 10 Hz; no per-frame timestamps are stored",
            "timestamp_provenance": "FMB stores no per-frame timestamps; 10 Hz is taken from the official paper",
            "field_mapping": mapping,
            "video_encoding": {"source": "uint8 BGR ndarray", "target_codec": "h264", "target_pix_fmt": "yuv420p", "mode": "reencode", "lossy": True, "reason": "LeRobot v3 video container uses RGB video encoding; numerical fields remain exact"},
            "partition_rules": ["single_object vs multi_object source schema", "schema fingerprint"],
            "source_files": [],
        },
    )


def inspect_fmb(
    config: DatasetConversionConfig,
    raw_root: Path,
    *,
    max_episodes: int | None = None,
    max_shards: int | None = None,
    local_preflight_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> FmbCatalog:
    """Perform the one archive-index/schema scan used by conversion and resume."""
    if not raw_root.is_dir():
        raise ConversionError(f"FMB raw root is missing: {raw_root}")
    shards = _source_shards(raw_root)
    if not shards:
        raise ConversionError(f"FMB raw root contains no FMB ZIP archives or extracted shards: {raw_root}")
    if max_shards is not None:
        if max_shards <= 0:
            raise ConversionError("max_shards must be positive")
        shards = shards[:max_shards]
    entries: list[FmbEntry] = []
    source_files: list[dict[str, Any]] = []
    preflight_root = local_preflight_root.resolve() if local_preflight_root is not None else None
    if preflight_root is not None:
        preflight_root.mkdir(parents=True, exist_ok=True)
    processed_members = 0
    last_progress = time.monotonic()
    try:
        for archive_name, source_path, source_mode in shards:
            stat = source_path.stat()
            source_files.append({
                "path": archive_name if source_mode == "zip" else _EXTRACTED_DIRS.get(archive_name, source_path.name),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": source_mode,
            })
            # Use the logical archive path in provenance even when its physical
            # source is an extracted directory.  This keeps manifests and unit
            # boundaries stable across the two layouts.
            logical_archive = raw_root / archive_name
            try:
                if source_mode == "zip":
                    with zipfile.ZipFile(source_path) as source:
                        members = [item for item in source.infolist() if not item.is_dir()]
                else:
                    members = _directory_members(source_path)
                for raw_info in members:
                    info = raw_info if source_mode == "zip" else raw_info[0]
                    member_path = raw_info[1] if source_mode != "zip" else None
                    if not info.filename.endswith(".npy"):
                        raise ConversionError(f"unexpected FMB member {source_path}:{info.filename}")
                    if preflight_root is None:
                        if source_mode == "zip":
                            with zipfile.ZipFile(source_path) as source, source.open(info) as stream:
                                fields = _read_lazy_metadata(stream)
                        else:
                            assert member_path is not None
                            with member_path.open("rb") as stream:
                                fields = _read_lazy_metadata(stream)
                    else:
                        staged = preflight_root / "current.npy"
                        staged.unlink(missing_ok=True)
                        _stage_source_member(
                            source_path,
                            source_mode,
                            info.filename,
                            staged,
                            expected_bytes=int(info.file_size),
                            progress=progress,
                            label=f"preflight {archive_name}:{info.filename}",
                        )
                        try:
                            with staged.open("rb") as stream:
                                fields = _read_lazy_metadata(stream)
                        finally:
                            staged.unlink(missing_ok=True)
                    entry = _parse_entry(
                        info,
                        logical_archive,
                        raw_root,
                        fields,
                        source_mode=source_mode,
                        source_mtime_ns=stat.st_mtime_ns if source_mode == "zip" else member_path.stat().st_mtime_ns,
                    )
                    _validate_fields(
                        fields,
                        source=f"{archive_name}:{info.filename}",
                        require_object_id=entry.kind == "multi_object",
                    )
                    entries.append(entry)
                    processed_members += 1
                    now = time.monotonic()
                    if progress is not None and (processed_members == 1 or now - last_progress >= 5.0):
                        progress(f"preflight scanned {processed_members} NPY members")
                        last_progress = now
            except zipfile.BadZipFile as exc:
                raise ConversionError(f"FMB archive is incomplete or invalid: {source_path}: {exc}") from exc
    finally:
        if preflight_root is not None:
            (preflight_root / "current.npy").unlink(missing_ok=True)
    if not entries:
        raise ConversionError("FMB archive inventory contains no trajectories")
    # Keep every archive contiguous inside a partition: the archive is the
    # checkpoint unit and must never be split by board/name ordering.
    entries.sort(key=lambda item: (item.kind, item.archive, item.board or "", item.member))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ConversionError("max_episodes must be positive")
        entries = entries[:max_episodes]
    groups: dict[tuple[str, str], list[FmbEntry]] = {}
    for entry in entries:
        groups.setdefault((entry.kind, _schema_fingerprint(entry.fields)), []).append(entry)
    partitions: list[FmbPartition] = []
    for (kind, schema), group in sorted(groups.items()):
        suffix = "single_object" if kind == "single_object" else "multi_object"
        if len(groups) > 2 or any(key[0] == kind and key[1] != schema for key in groups):
            suffix += f"--schema-{schema[:12]}"
        name = f"functional_manipulation_benchmark_fmb--{suffix}"
        plan = _plan_for_partition(config, raw_root, name, tuple(group), schema)
        plan = DatasetConversionPlan(**{**plan.__dict__, "extra": {**plan.extra, "source_files": source_files}, "output_path": Path(name)})
        partitions.append(FmbPartition(name, plan, tuple(group), schema))
    selected = [entry for partition in partitions for entry in partition.entries]
    sample_entries: list[FmbEntry] = []
    candidates = [selected[0], selected[len(selected) // 2], selected[-1]]
    candidates.extend(partition.entries[0] for partition in partitions)
    for candidate in candidates:
        if not any(item.archive == candidate.archive and item.member == candidate.member for item in sample_entries):
            sample_entries.append(candidate)
    sample_evidence: list[dict[str, Any]] = []
    for entry in sample_entries:
        if preflight_root is None:
            payload = _load_payload(entry, raw_root)
        else:
            source_path, source_mode = _source_for_archive(raw_root, entry.archive)
            staged = preflight_root / "sample.npy"
            staged.unlink(missing_ok=True)
            _stage_source_member(
                source_path,
                source_mode,
                entry.member,
                staged,
                expected_bytes=entry.size,
                progress=progress,
                label=f"preflight sample {entry.archive}:{entry.member}",
            )
            try:
                payload = _load_payload(entry, raw_root, local_member_path=staged)
            finally:
                staged.unlink(missing_ok=True)
        _validate_fields(
            {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in payload.items()},
            source=f"sample {entry.member}",
            require_object_id=entry.kind == "multi_object",
        )
        sample_evidence.append({"source": f"{entry.archive}:{entry.member}", "frames": entry.num_frames, "object_info": entry.object_info, "first_last_shapes": {key: list(value.shape) for key, value in payload.items()}, "first_frame": {"action": payload["action"][0].tolist(), "primitive": str(payload["primitive"][0])}, "last_frame": {"action": payload["action"][-1].tolist(), "primitive": str(payload["primitive"][-1])}})
    fingerprint_payload = {
        "source_root": str(raw_root),
        "source_files": source_files,
        "fps": FMB_FPS,
        "partitions": [{"name": p.name, "schema": p.schema_fingerprint, "episodes": len(p.entries), "frames": p.plan.num_frames} for p in partitions],
        "task_mapping": [
            {
                "partition": partition.name,
                "source": f"{entry.archive}:{entry.member}",
                "task": entry.instruction,
                "frames": entry.num_frames,
                "bytes": entry.size,
                "source_mode": entry.source_mode,
                "source_mtime_ns": entry.source_mtime_ns,
            }
            for partition in partitions
            for entry in partition.entries
        ],
        "partition_rules": ["kind", "schema fingerprint", "archive/member lexical order"],
    }
    mapping = tuple(row for p in partitions for row in p.plan.extra["field_mapping"])
    return FmbCatalog(tuple(partitions), fingerprint_payload, mapping, tuple(sample_evidence))


def catalog_to_payload(catalog: FmbCatalog) -> dict[str, Any]:
    def entry_payload(entry: FmbEntry) -> dict[str, Any]:
        return {"archive": entry.archive, "member": entry.member, "kind": entry.kind, "board": entry.board, "object_id": entry.object_id, "trajectory_id": entry.trajectory_id, "size": entry.size, "compressed_size": entry.compressed_size, "crc": entry.crc, "num_frames": entry.num_frames, "fields": entry.fields, "instruction": entry.instruction, "object_info": entry.object_info, "source_mode": entry.source_mode, "source_mtime_ns": entry.source_mtime_ns}
    return {"schema_version": 1, "fingerprint_payload": catalog.fingerprint_payload, "mapping_table": list(catalog.mapping_table), "sample_evidence": list(catalog.sample_evidence), "partitions": [{"name": p.name, "schema_fingerprint": p.schema_fingerprint, "entries": [entry_payload(e) for e in p.entries]} for p in catalog.partitions]}


def catalog_from_payload(payload: dict[str, Any], config: DatasetConversionConfig, raw_root: Path) -> FmbCatalog:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("partitions"), list):
        raise ConversionError("unsupported FMB preflight catalog")
    partitions: list[FmbPartition] = []
    for raw_partition in payload["partitions"]:
        try:
            entries = tuple(FmbEntry(**dict(raw_entry)) for raw_entry in raw_partition["entries"])
            for entry in entries:
                _validate_fields(
                    entry.fields,
                    source=f"{entry.archive}:{entry.member}",
                    require_object_id=entry.kind == "multi_object",
                )
            schema = str(raw_partition["schema_fingerprint"])
            plan = _plan_for_partition(config, raw_root, str(raw_partition["name"]), entries, schema)
            plan = DatasetConversionPlan(**{**plan.__dict__, "extra": {**plan.extra, "source_files": payload["fingerprint_payload"]["source_files"]}, "output_path": Path(str(raw_partition["name"]))})
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"invalid FMB preflight catalog: {exc}") from exc
        partitions.append(FmbPartition(str(raw_partition["name"]), plan, entries, schema))
    fingerprint = payload.get("fingerprint_payload")
    mapping = payload.get("mapping_table")
    samples = payload.get("sample_evidence")
    if not isinstance(fingerprint, dict) or not isinstance(mapping, list) or not isinstance(samples, list):
        raise ConversionError("FMB preflight catalog metadata is incomplete")
    return FmbCatalog(tuple(partitions), fingerprint, tuple(dict(row) for row in mapping), tuple(dict(row) for row in samples))


def remap_catalog_to_extracted_root(
    catalog: FmbCatalog,
    config: DatasetConversionConfig,
    raw_root: Path,
) -> FmbCatalog | None:
    """Reuse a complete ZIP preflight after the same shards were extracted.

    The old catalog contains all NPY metadata and archive member names.  The
    migration only stats each extracted member and changes the physical
    source mode; it never trusts a missing or differently sized file.  Return
    ``None`` when *raw_root* is not the supported extracted layout.
    """

    shards = _source_shards(raw_root)
    if not shards or not any(mode == "extracted_directory" for _name, _path, mode in shards):
        return None
    if {name for name, _path, _mode in shards} != set(_EXTRACTED_DIRS):
        raise ConversionError(f"incomplete FMB extracted root; expected shards {sorted(_EXTRACTED_DIRS)}")
    source_files = [
        {
            "path": _EXTRACTED_DIRS[archive_name],
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "mode": mode,
        }
        for archive_name, source, mode in shards
    ]
    partitions: list[FmbPartition] = []
    for partition in catalog.partitions:
        entries: list[FmbEntry] = []
        for entry in partition.entries:
            source, mode = _source_for_archive(raw_root, entry.archive)
            if mode != "extracted_directory":
                raise ConversionError(f"FMB catalog shard is not extracted: {entry.archive}")
            member_path = (source / entry.member).resolve()
            if not member_path.is_relative_to(source.resolve()) or not member_path.is_file():
                raise ConversionError(f"FMB extracted member is missing: {member_path}")
            stat = member_path.stat()
            if stat.st_size != entry.size:
                raise ConversionError(
                    f"FMB extracted member size differs from ZIP preflight for {member_path}: "
                    f"{stat.st_size} != {entry.size}"
                )
            entries.append(replace(entry, source_mode=mode, source_mtime_ns=stat.st_mtime_ns))
        entry_tuple = tuple(entries)
        plan = _plan_for_partition(config, raw_root, partition.name, entry_tuple, partition.schema_fingerprint)
        plan = DatasetConversionPlan(
            **{
                **plan.__dict__,
                "extra": {**plan.extra, "source_files": source_files},
                "output_path": Path(partition.name),
            }
        )
        partitions.append(FmbPartition(partition.name, plan, entry_tuple, partition.schema_fingerprint))
    fingerprint_payload = dict(catalog.fingerprint_payload)
    fingerprint_payload["source_root"] = str(raw_root)
    fingerprint_payload["source_files"] = source_files
    fingerprint_payload["task_mapping"] = [
        {
            "partition": partition.name,
            "source": f"{entry.archive}:{entry.member}",
            "task": entry.instruction,
            "frames": entry.num_frames,
            "bytes": entry.size,
            "source_mode": entry.source_mode,
            "source_mtime_ns": entry.source_mtime_ns,
        }
        for partition in partitions
        for entry in partition.entries
    ]
    fingerprint_payload["partition_rules"] = [
        "kind",
        "schema fingerprint",
        "archive/member lexical order",
        "ZIP or extracted-directory source mode",
    ]
    return FmbCatalog(
        tuple(partitions),
        fingerprint_payload,
        catalog.mapping_table,
        catalog.sample_evidence,
    )


def validate_catalog_sources(catalog: FmbCatalog, raw_root: Path) -> None:
    for record in catalog.fingerprint_payload.get("source_files", []):
        path = raw_root / str(record["path"])
        stat = path.stat()
        # OSSFS may refresh a directory inode mtime while listing or reading
        # its children.  For extracted sources, the per-NPY checks below are
        # the authoritative content validation; directory mtime is not.
        if str(record.get("mode", "zip")) == "extracted_directory":
            if not path.is_dir():
                raise ConversionError(f"FMB extracted source directory is missing: {path}")
            continue
        if (
            stat.st_size != int(record["size"])
            or stat.st_mtime_ns != int(record["mtime_ns"])
            or str(record.get("mode", "zip")) != ("extracted_directory" if path.is_dir() else "zip")
        ):
            raise ConversionError(f"FMB source shard changed since preflight: {path}")
    # A directory's own mtime does not change when a file is modified in
    # place.  Validate extracted members individually so a cached preflight
    # cannot silently feed a changed trajectory to a resumed conversion.
    for partition in catalog.partitions:
        for entry in partition.entries:
            if entry.source_mode != "extracted_directory":
                continue
            source_path, _mode = _source_for_archive(raw_root, entry.archive)
            member_path = (source_path / entry.member).resolve()
            if not member_path.is_relative_to(source_path.resolve()) or not member_path.is_file():
                raise ConversionError(f"FMB extracted member is missing since preflight: {member_path}")
            stat = member_path.stat()
            if stat.st_size != entry.size or stat.st_mtime_ns != entry.source_mtime_ns:
                raise ConversionError(f"FMB extracted member changed since preflight: {member_path}")


def iter_fmb_frames(plan: DatasetConversionPlan, episode: EpisodePlan, raw_root: Path) -> Iterator[dict[str, Any]]:
    entry = FmbEntry(
        episode.extra["archive"],
        episode.extra["member"],
        episode.extra["kind"],
        episode.extra.get("board"),
        episode.extra.get("object_id"),
        episode.extra.get("trajectory_id"),
        0,
        0,
        0,
        episode.num_frames,
        {},
        episode.instruction,
    )
    fields = _load_payload(entry, raw_root)
    actual_specs = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in fields.items()
    }
    _validate_fields(
        actual_specs,
        source=episode.source_relative_path,
        require_object_id=episode.extra["kind"] == "multi_object",
    )
    actual_frames = next(iter(fields.values())).shape[0]
    if actual_frames != episode.num_frames and not (
        episode.extra.get("allow_frame_prefix") and actual_frames >= episode.num_frames
    ):
        raise ConversionError(
            f"{episode.source_relative_path}: payload has {actual_frames} frames, "
            f"preflight recorded {episode.num_frames}"
        )
    for index in range(episode.num_frames):
        frame: dict[str, Any] = {"task": episode.instruction}
        for source_key, feature_key in _RGB_KEYS:
            image = np.asarray(fields[source_key][index])
            frame[feature_key] = np.ascontiguousarray(image[..., ::-1])
        for source_key, feature_key in _DEPTH_KEYS + _VECTOR_KEYS:
            value = fields[source_key][index]
            if source_key == "obs/gripper_pose":
                frame[feature_key] = np.asarray([value], dtype=np.int64)
            else:
                frame[feature_key] = value
        frame["observation.primitive"] = str(fields["primitive"][index])
        if episode.extra["kind"] == "multi_object":
            frame["observation.object_id"] = np.asarray([fields["object_id"][index]], dtype=np.int64)
        yield frame


class FmbNpyReader(DatasetReader):
    def build_plan(self, config: DatasetConversionConfig, raw_root: Path, staging_root: Path) -> DatasetConversionPlan:
        catalog = inspect_fmb(config, raw_root)
        if len(catalog.partitions) != 1:
            raise ConversionError("FMB has multiple incompatible partitions; use convert_fmb_to_lerobot.py")
        plan = catalog.partitions[0].plan
        return DatasetConversionPlan(**{**plan.__dict__, "output_path": staging_root / "lerobot_v3_0" / plan.dataset_uid})

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        return iter_fmb_frames(plan, episode, Path(plan.extra["source_root"]))
