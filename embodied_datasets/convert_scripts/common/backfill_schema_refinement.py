"""Table-driven corrections applied to the 59 already-onboarded dataset
configs during the 2026-07-10 registry schema refinement (see
docs/superpowers/specs/2026-07-10-registry-schema-refinement-design.md
section 4). Every value below is already justified in the corresponding
dataset's field_sources/suggested_new_enum_values from its onboarding
report -- this module applies known corrections, it does not research
new ones.
"""
from __future__ import annotations

from typing import Any, Dict

from .schema import DatasetConfig

# Design doc section 4, table A: specific field-value corrections.
FIELD_CORRECTIONS: Dict[str, Dict[str, Any]] = {
    "functional_manipulation_benchmark_fmb": {
        "camera_views": ["side", "wrist"],
    },
    "furniturebench": {
        "camera_views": ["front", "wrist"],
    },
    "the_colosseum": {
        "camera_views": ["front", "top", "third_person", "wrist"],
    },
    "gensim2": {
        "camera_views": ["front", "side", "wrist"],
    },
    "fastumi": {
        "camera_views": ["wrist"],
    },
    "mv_umi": {
        "camera_views": ["third_person", "wrist"],
        "gripper_type": "three_jaw",
        "action_frame": "relative_trajectory",
    },
    "omniumi": {
        "camera_views": ["wrist"],
        "action_frame": "mixed_delta_absolute",
    },
    "dexcap": {
        "camera_views": ["body_worn"],
    },
    "aloha_unleashed": {
        "camera_views": ["top", "left_wrist", "right_wrist", "other", "worms_eye"],
    },
    "galaxea_open_world_dataset": {
        "license": "CC-BY-NC-SA-4.0",
    },
    "grutopia": {
        "license": "CC-BY-NC-SA-4.0",
        "collection_method": "scene_asset_curation",
    },
    "airexo_2": {
        "license": "CC-BY-NC-SA-4.0",
        "robot_platform": "flexiv_rizon4",
    },
    "yubi": {
        "license": "CC-BY-NC-SA-4.0",
        "robot_platform": "toyota_eley",
        "is_multi_embodiment": True,
    },
    "egodex": {
        "license": "CC-BY-NC-ND-4.0",
    },
    "bigym": {
        "robot_platform": "unitree_h1",
    },
    "humanoidbench": {
        "robot_platform": "unitree_h1",
        "is_multi_embodiment": True,
    },
    "arcap": {
        "collection_method": "ar_haptic_guided_synthesis",
    },
    "hot3d": {
        "raw_format": "VRS",
    },
    "assembly101": {
        "has_synchronized_multiview_rig": True,
    },
    "robogen": {
        "is_multi_embodiment": True,
    },
}

# Design doc section 4, table B: release_type classification. Any dataset
# id not listed here defaults to "fixed_episode_dataset" -- see
# apply_corrections().
RELEASE_TYPE_OVERRIDES: Dict[str, str] = {
    "robogen": "generation_framework",
    "gensim2": "generation_framework",
    "grutopia": "scene_platform",
    "humanoidbench": "rl_benchmark_env",
}


def apply_corrections(config: DatasetConfig) -> DatasetConfig:
    """Return a new DatasetConfig with this dataset's table-driven
    corrections applied. Fields not mentioned in FIELD_CORRECTIONS for
    this dataset id are left untouched; release_type is always set
    (falling back to "fixed_episode_dataset")."""
    data = config.model_dump(mode="json", exclude_none=True)
    data.update(FIELD_CORRECTIONS.get(config.id, {}))
    data["release_type"] = RELEASE_TYPE_OVERRIDES.get(config.id, "fixed_episode_dataset")
    return DatasetConfig(**data)
