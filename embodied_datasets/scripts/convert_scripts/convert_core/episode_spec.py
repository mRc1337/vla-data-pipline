"""Format-agnostic plan objects shared by every convert_scripts reader/writer.

A reader's job (see ``readers/``) is to turn one raw dataset directory into a
:class:`DatasetConversionPlan` (validate-only, no writes) and then, given that
plan, stream ready-to-write frame dicts per episode. ``convert_core.lerobot_writer``
never looks at any format-specific source path/key -- it only consumes this
module's dataclasses, exactly the way the original Mobile ALOHA converter's
``ConversionPlan``/``EpisodeSpec`` already separated "what to write" from "how
to read it", just generalized to more than one source format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VectorFeatureSpec:
    """One non-image per-frame feature (state/action/velocity/...)."""

    feature_key: str
    dim: int
    names: tuple[str, ...] | None = None
    dtype: str = "float32"
    shape: tuple[int, ...] | None = None

    @property
    def resolved_shape(self) -> tuple[int, ...]:
        return self.shape if self.shape is not None else (self.dim,)


@dataclass(frozen=True)
class CameraFeatureSpec:
    """One per-frame video feature."""

    feature_key: str
    height: int
    width: int


@dataclass(frozen=True)
class EpisodePlan:
    """One source episode, already validated, ready to stream frames from.

    ``extra`` is reader-private bookkeeping (e.g. an HDF5 file path, or a
    materialized list of decoded RLDS steps) that ``convert_core.lerobot_writer``
    never reads -- only the reader that produced this plan knows how to use it
    in its own ``iter_frames``.
    """

    episode_uid: str
    source_relative_path: str
    instruction: str
    num_frames: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConversionPlan:
    dataset_uid: str
    output_path: Path
    fps: int
    measured_fps: float
    robot_type: str
    vector_features: tuple[VectorFeatureSpec, ...]
    camera_features: tuple[CameraFeatureSpec, ...]
    episodes: tuple[EpisodePlan, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return sum(episode.num_frames for episode in self.episodes)

    def feature_schema(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {}
        for vector in self.vector_features:
            features[vector.feature_key] = {
                "dtype": vector.dtype,
                "shape": vector.resolved_shape,
                "names": list(vector.names) if vector.names else None,
            }
        for camera in self.camera_features:
            features[camera.feature_key] = {
                "dtype": "video",
                "shape": (camera.height, camera.width, 3),
                "names": ["height", "width", "channel"],
            }
        return features
