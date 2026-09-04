"""The reader contract every source format implements.

Mirrors the two-phase shape ``convert_mobile_aloha_to_lerobot.py`` already
used (``inspect_dataset()`` builds a plan, then writing streams frames per
episode) -- generalized so ``convert_dataset.py`` can dispatch to any of them
by ``format`` name without knowing which one it got.

``build_plan`` must not decode video/image pixel payloads -- only shapes,
dtypes, counts, and small metadata sidecars, same discipline as
``dump_dataset_schema.py``. The one documented exception is RLDS: TFRecord
sequences don't expose a per-episode frame count without decoding each
episode's step sequence, so :class:`readers.rlds_reader.RldsReader` decodes
every episode once during ``build_plan`` and caches the result in
``EpisodePlan.extra`` for ``iter_frames`` to reuse -- see that module's
docstring for the memory-scaling caveat this implies for very large datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol

from convert_core.dataset_config import DatasetConversionConfig
from convert_core.episode_spec import DatasetConversionPlan, EpisodePlan


class DatasetReader(Protocol):
    def build_plan(
        self,
        config: DatasetConversionConfig,
        raw_root: Path,
        staging_root: Path,
    ) -> DatasetConversionPlan:
        """Validate the raw dataset and return a plan: feature schema plus
        per-episode metadata. Raises ``convert_core.errors.ConversionError``
        (or a format-specific ``RuntimeError`` for a missing optional
        dependency) when the data cannot be converted without guessing."""

    def iter_frames(self, plan: DatasetConversionPlan, episode: EpisodePlan) -> Iterator[dict[str, Any]]:
        """Yield one dict per frame, ready for ``LeRobotDataset.add_frame()``
        -- already containing every declared vector/camera feature plus
        ``"task"``. Called once per episode, in plan order, exactly once per
        conversion (not re-entrant across episodes)."""
