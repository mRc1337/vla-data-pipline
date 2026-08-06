import json

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")


def _make_episode(episode_index, num_frames, state_dim, action_dim, frames=None):
    from episode import Episode

    return Episode(
        episode_index=episode_index,
        timestamps=np.arange(num_frames, dtype=np.float64) / 10.0,
        state=np.zeros((num_frames, state_dim), dtype=np.float32),
        action=np.zeros((num_frames, action_dim), dtype=np.float32),
        frames=frames or {},
    )


class _FakeConfig:
    num_arms = 1


def test_build_video_urls_returns_none_final_when_final_root_is_none(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from player_payload import build_video_urls

    dataset_root = tmp_path / "ds_final_none"
    make_synthetic_dataset(
        dataset_root, repo_id="test/payload_final_none", num_episodes=1, num_frames=4, include_video=True,
    )

    urls = build_video_urls(
        raw_root=dataset_root, final_root=None, raw_port=9000, final_port=None,
        episode_index=0, view_keys=["observation.image"],
    )
    assert urls["observation.image"]["final"] is None
    assert urls["observation.image"]["raw"] is not None


def test_build_video_urls_builds_url_and_element_id_for_real_clip(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from player_payload import build_video_urls

    dataset_root = tmp_path / "ds"
    make_synthetic_dataset(dataset_root, repo_id="test/payload_urls", num_episodes=1, num_frames=4, include_video=True)

    urls = build_video_urls(
        raw_root=dataset_root, final_root=None, raw_port=1234, final_port=None,
        episode_index=0, view_keys=["observation.image"],
    )
    raw_clip = urls["observation.image"]["raw"]
    assert raw_clip is not None
    assert raw_clip["url"].startswith("http://127.0.0.1:1234/")
    assert raw_clip["element_id"] == "video-observation.image-raw"
    assert urls["observation.image"]["final"] is None


def test_build_payload_shapes_charts_and_marks_final_frame_count_zero_without_final_episode():
    from player_payload import build_payload

    raw_episode = _make_episode(0, num_frames=5, state_dim=14, action_dim=8)
    payload = build_payload(
        raw_episode, None, _FakeConfig(), {"state": None, "action": None}, video_urls={}, fps=10.0,
    )

    assert payload["fps"] == 10.0
    assert payload["raw_frame_count"] == 5
    assert payload["final_frame_count"] == 0
    assert len(payload["charts"]["state"]) > 0
    for band in payload["charts"]["state"]:
        assert band["populated"] is True
        assert "data" in band["figure"] and "layout" in band["figure"]
    # The whole payload must be JSON-serializable as-is -- this is what gets
    # embedded directly into the HTML component in Task 6.
    json.dumps(payload)


def test_build_payload_reflects_final_episode_when_present():
    from player_payload import build_payload

    raw_episode = _make_episode(0, num_frames=5, state_dim=14, action_dim=8)
    final_episode = _make_episode(0, num_frames=4, state_dim=14, action_dim=8)
    payload = build_payload(
        raw_episode, final_episode, _FakeConfig(), {"state": None, "action": None}, video_urls={}, fps=10.0,
    )

    assert payload["final_frame_count"] == 4
    for band in payload["charts"]["state"]:
        assert "figure" in band
