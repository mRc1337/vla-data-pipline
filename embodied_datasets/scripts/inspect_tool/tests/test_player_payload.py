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


def test_build_video_urls_uses_different_episode_indices_for_raw_and_final(tmp_path):
    from tests.fixtures import make_synthetic_dataset
    from player_payload import build_video_urls

    # Create a dataset with 2 episodes. Each episode gets its own time window in
    # the packed video file (e.g., episode 0: 0.0-0.4s, episode 1: 0.4-0.8s).
    dataset_root = tmp_path / "ds_two_episodes"
    make_synthetic_dataset(
        dataset_root, repo_id="test/payload_final_index", num_episodes=2, num_frames=4, include_video=True,
    )

    # Call build_video_urls with episode_index=0 (raw) and final_episode_index=1 (final).
    # This should fetch video info from different episodes, so the timestamps differ.
    urls = build_video_urls(
        raw_root=dataset_root, final_root=dataset_root, raw_port=1234, final_port=5678,
        episode_index=0, final_episode_index=1, view_keys=["observation.image"],
    )

    raw_clip = urls["observation.image"]["raw"]
    final_clip = urls["observation.image"]["final"]
    assert raw_clip is not None
    assert final_clip is not None

    # Verify that raw and final clips reference different episodes by checking
    # their timestamps are different. With 4-frame episodes at 10fps, episode 0
    # should be at 0.0-0.4s and episode 1 at 0.4-0.8s (they're packed sequentially).
    assert raw_clip["from_timestamp"] != final_clip["from_timestamp"], \
        f"raw and final clips should have different timestamps when using different episode indices"
    assert raw_clip["to_timestamp"] != final_clip["to_timestamp"]
