import numpy as np

from episode import Episode, StageResult
from metadata_io import (
    load_metadata,
    metadata_exists,
    save_metadata,
    stage_result_to_record,
)


def _dummy_episode():
    return Episode(episode_index=0, timestamps=np.zeros(1), state=np.zeros((1, 1)), action=np.zeros((1, 1)))


def test_stage_result_to_record_excludes_big_arrays_and_converts_numpy():
    result = StageResult(
        episode=_dummy_episode(),
        skip_reason="trend_misaligned",
        rejected=True,
        dropped_frame_indices=[1, 2],
        stats={
            "lag": np.int64(3),
            "canonical_state": np.zeros((4, 128)),
            "canonical_mask": np.zeros(128, dtype=bool),
            "canonical_dim": 128,
            "median_offset": np.array([0.1, 0.2, 0.3]),
        },
    )
    record = stage_result_to_record("stage2_trend_alignment", result)
    assert record == {
        "stage": "stage2_trend_alignment",
        "skip_reason": "trend_misaligned",
        "rejected": True,
        "dropped_frame_indices": [1, 2],
        "stats": {"lag": 3, "median_offset": [0.1, 0.2, 0.3]},
    }


def test_stage_result_to_record_defaults():
    result = StageResult(episode=_dummy_episode())
    record = stage_result_to_record("stage1_sudden_change", result)
    assert record == {
        "stage": "stage1_sudden_change",
        "skip_reason": None,
        "rejected": False,
        "dropped_frame_indices": [],
        "stats": {},
    }


def test_stage_result_to_record_preserves_nested_frame_reasons():
    # check3_video_quality.py's stats[view]["frame_reasons"] nests a plain
    # {int: str} dict inside stats -- must survive untouched (already
    # JSON-safe, no numpy types inside it).
    result = StageResult(
        episode=_dummy_episode(),
        dropped_frame_indices=[2],
        stats={"head": {"num_black": 1, "num_blurry": 0, "num_still": 0, "frame_reasons": {2: "black"}}},
    )
    record = stage_result_to_record("check3_video_quality", result)
    assert record["stats"] == {"head": {"num_black": 1, "num_blurry": 0, "num_still": 0, "frame_reasons": {2: "black"}}}


def test_save_and_load_metadata_round_trip(tmp_path):
    metadata = {"episodes": [{"episode_index": 0, "stages": []}]}
    save_metadata(metadata, tmp_path)
    assert metadata_exists(tmp_path)
    assert load_metadata(tmp_path) == metadata


def test_load_metadata_returns_none_when_missing(tmp_path):
    assert not metadata_exists(tmp_path)
    assert load_metadata(tmp_path) is None
