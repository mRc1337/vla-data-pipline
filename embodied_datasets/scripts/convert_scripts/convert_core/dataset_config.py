"""Pydantic config for convert_scripts/configs/<dataset-uid>.yaml.

One file per dataset UID drives ``convert_dataset.py``: which source format's
reader to dispatch to, and the per-dataset facts that differ between
otherwise-identical-format datasets (HDF5 key names, RLDS step field paths,
raw-folder metadata field names, vector-feature widths, ...). This mirrors
``process_scripts/common/schema.py``'s ``ProcessConfig`` pattern (a single
``extra="forbid"`` pydantic model loaded from YAML) so a typo'd field name
fails loudly instead of being silently ignored.

Not every field applies to every ``format``; each reader only reads the
fields relevant to it (see the per-field comments below) and ignores the
rest. This is deliberately one flat model instead of a tagged union of
per-format models: it keeps ``configs/*.yaml`` easy to hand-edit and diff,
at the cost of a config technically being able to declare fields its chosen
format ignores. Readers do not warn about unused fields today -- worth
revisiting if that silence ever hides a typo in practice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
import yaml


class VectorFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_key: str
    """LeRobot feature name, e.g. "observation.state" or "action"."""
    source_key: str
    """Reader-specific locator: an HDF5 dataset path, a slash-separated RLDS
    step field path (e.g. "observation/state"), or a dotted JSON field path
    for raw_image_json (e.g. "state")."""
    dim: int
    names: Optional[list[str]] = None


class CameraFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_key: str
    """LeRobot feature name, e.g. "observation.images.front"."""
    source_key: str
    """RLDS step field path (e.g. "observation/image"), or a raw_image_json
    per-frame field naming the image file. Ignored by the hdf5 reader, which
    auto-discovers cameras under ``images_key`` instead -- only set this for
    hdf5 if you need to force one specific key rather than auto-discovering."""


class DatasetConversionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_uid: str
    format: Literal[
        "hdf5",
        "arcap_hdf5",
        "one_x_world_model",
        "dexmimicgen_hdf5",
        "robomimic_hdf5",
        "rlds",
        "raw_image_json",
    ]
    robot_type: str
    fps: Optional[float] = None
    """Fallback FPS, used only when the reader cannot measure it from the
    data itself (camera timestamps/attrs/sidecar for hdf5; declared per
    Open-X convention for rlds; a "fps" metadata field for raw_image_json)."""

    vector_fields: list[VectorFieldConfig] = Field(default_factory=list)
    cameras: list[CameraFieldConfig] = Field(default_factory=list)

    # --- hdf5 reader only ---
    episode_glob: str = "*/*.h5"
    """Glob relative to <raw-root>/<dataset_uid>, matching one file per
    episode. Mobile ALOHA's own convention is "*/*.hdf5"; both suffixes are
    tried regardless of which one is written here."""
    images_key: str = "/observations/images"
    """HDF5 group auto-discovery walks for RGB datasets under this key."""
    instruction_source: Literal["path_parent", "field", "constant"] = "path_parent"
    """path_parent: the episode file's parent directory name (relative to
    the dataset root) is the instruction, Mobile ALOHA's convention.
    field: read ``instruction_field`` (an HDF5 key) instead.
    constant: every episode gets ``instruction_constant``."""
    instruction_field: Optional[str] = None
    instruction_constant: Optional[str] = None
    uncompressed_color_order: Literal["bgr", "rgb"] = "bgr"

    # --- robomimic_hdf5 reader only ---
    source_directory: Optional[str] = None
    """Physical directory below raw_root. Useful when a mounted dataset was
    published under a misspelled directory name."""
    source_file: Optional[str] = None
    """One multi-episode robomimic HDF5 container relative to source_directory."""

    # --- 1X World Model reader only ---
    one_x_version: Literal["v1.1", "v2.0"] = "v2.0"
    """Token/state schema to inspect.  The collection converter invokes the
    reader once per version because the two public releases are heterogeneous."""
    one_x_splits: list[str] = Field(default_factory=list)
    """Exact source split directory names, e.g. train_v2.0 and val_v2.0."""
    one_x_include_test: bool = False
    """Request inspection of test_v2.0.  Conversion is deliberately rejected:
    the challenge samples do not contain one video frame per state frame or
    segment boundaries."""
    one_x_v1_decoder_repo: Optional[str] = None
    """Checkout of the official 1x-technologies/1Xgpt repository."""
    one_x_cosmos_decoder_path: Optional[str] = None
    """Cosmos-Tokenizer-DV8x8x8 decoder.jit used for v2 token decoding."""
    one_x_decode_batch_size: int = Field(default=8, ge=1)
    one_x_v1_checkpoint_segments: int = Field(default=128, ge=1)

    # --- rlds reader only ---
    split: str = "all"
    """A TFDS split name, or "all" to concatenate every split the builder
    reports."""

    # --- raw_image_json reader only ---
    metadata_filename: str = "metadata.json"
    """Per-episode metadata file, expected directly inside each episode
    directory alongside its image files."""
    image_glob: str = "*.jpg"
    """Used only when a camera's ``source_key`` is the "$glob" sentinel:
    naturally-sorted files matching this glob become that camera's frames,
    in order, instead of being looked up per-frame in the metadata JSON."""


def load_dataset_config(path: Path) -> DatasetConversionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DatasetConversionConfig(**raw)
