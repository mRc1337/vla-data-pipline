import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from episode import Episode
from common.schema import ProcessConfig
from stage3_extreme_value import apply, compute_bounds


def _make_episode(index, state_col0):
    n = len(state_col0)
    state = np.stack([state_col0, np.zeros(n)], axis=1)
    return Episode(
        episode_index=index,
        timestamps=np.arange(n, dtype=np.float64),
        state=state,
        action=np.zeros((n, 1)),
    )


def test_compute_bounds_excludes_gripper_dim():
    episodes = [_make_episode(0, np.linspace(0, 100, 50))]
    config = ProcessConfig(id="x", quantile_low=0.1, quantile_high=0.9, gripper_dims_state=[1])
    bounds = compute_bounds(episodes, config)
    assert 1 not in bounds["state"]
    assert 0 in bounds["state"]


def test_apply_drops_out_of_bound_frames():
    normal = np.concatenate([np.linspace(0, 10, 48), [500.0, -500.0]])
    episode = _make_episode(0, normal)
    config = ProcessConfig(id="x", quantile_low=0.02, quantile_high=0.98)
    bounds = compute_bounds([episode], config)
    config.extreme_value_bounds = bounds
    result = apply(episode, config)
    assert 48 in result.dropped_frame_indices
    assert 49 in result.dropped_frame_indices
    assert result.episode.state.shape[0] == 48


def test_apply_with_bounds_none_is_skipped_gracefully():
    episode = _make_episode(0, np.linspace(0, 10, 10))
    config = ProcessConfig(id="x")
    assert config.extreme_value_bounds is None
    result = apply(episode, config)  # must not raise
    assert result.skip_reason == "extreme_value_bounds_not_computed"
    assert result.episode.state.shape[0] == 10


def test_apply_with_empty_bounds_dict_is_skipped_gracefully():
    episode = _make_episode(0, np.linspace(0, 10, 10))
    config = ProcessConfig(id="x")
    config.extreme_value_bounds = {}
    result = apply(episode, config)  # must not raise
    assert result.skip_reason == "extreme_value_bounds_not_computed"


def test_apply_with_empty_inner_bounds_dict_keeps_all_frames():
    episode = _make_episode(0, np.linspace(0, 10, 10))
    config = ProcessConfig(id="x")
    config.extreme_value_bounds = {"state": {}, "action": {}}
    result = apply(episode, config)  # must not raise
    assert result.skip_reason is None
    assert result.dropped_frame_indices == []
    assert result.episode.state.shape[0] == 10


def test_compute_bounds_empty_episode_list_raises_value_error():
    config = ProcessConfig(id="x")
    with pytest.raises(ValueError):
        compute_bounds([], config)


def test_compute_bounds_zero_width_state_and_action_does_not_crash():
    n = 10
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(n, dtype=np.float64),
        state=np.zeros((n, 0)),
        action=np.zeros((n, 0)),
    )
    config = ProcessConfig(id="x")
    bounds = compute_bounds([episode], config)  # must not raise
    assert bounds["state"] == {}
    assert bounds["action"] == {}


def test_apply_all_frames_dropped_yields_zero_length_episode_without_crash():
    episode = _make_episode(0, np.array([500.0, -500.0, 500.0, -500.0]))
    config = ProcessConfig(id="x", quantile_low=0.0, quantile_high=1.0)
    # Force bounds tighter than any of the extreme values so every frame is
    # out-of-bounds, regardless of what compute_bounds would have produced.
    config.extreme_value_bounds = {"state": {0: [-1.0, 1.0]}, "action": {}}
    result = apply(episode, config)  # must not raise
    assert result.episode.state.shape[0] == 0
    assert result.episode.action.shape[0] == 0
    assert result.episode.timestamps.shape[0] == 0
    assert sorted(result.dropped_frame_indices) == [0, 1, 2, 3]


def test_compute_bounds_out_of_range_gripper_dim_index_is_ignored():
    # gripper_dims_state references a column index that does not exist on
    # this dataset's state array (state only has columns 0 and 1). The
    # exemption check is a plain membership test against range(num_dims),
    # so an out-of-range exempt index simply never matches -- it must not
    # raise and must not accidentally exempt an in-range dim.
    episodes = [_make_episode(0, np.linspace(0, 100, 50))]
    config = ProcessConfig(id="x", quantile_low=0.1, quantile_high=0.9, gripper_dims_state=[5])
    bounds = compute_bounds(episodes, config)  # must not raise
    assert 0 in bounds["state"]
    assert 1 in bounds["state"]


def test_inverted_quantiles_does_not_crash_and_yields_widened_bounds():
    # quantile_low > quantile_high is nonsensical but not forbidden by the
    # schema; np.quantile still returns two values (low_q >= high_q in the
    # underlying distribution sense), and apply() must not crash even if
    # the resulting "bounds" end up inverted (low > high).
    #
    # With low > high, the "keep" condition (low <= v <= high) is
    # unsatisfiable for any v, since low > high. So the out-of-bounds
    # mask (v < low) | (v > high) is true for every value: for any v,
    # either v < low, or v >= low > high so v > high. Observed: every
    # frame ends up dropped.
    episode = _make_episode(0, np.linspace(0, 100, 50))
    config = ProcessConfig(id="x", quantile_low=0.9, quantile_high=0.1)
    bounds = compute_bounds([episode], config)  # must not raise
    config.extreme_value_bounds = bounds
    result = apply(episode, config)  # must not raise
    assert sorted(result.dropped_frame_indices) == list(range(50))
    assert result.episode.state.shape[0] == 0


def test_apply_filters_per_view_frames_arrays_alongside_state():
    # _make_episode() never populates episode.frames, so the
    # `frames={view: frames[keep_mask] ...}` line in apply() was never
    # exercised against a real per-view frame array by any prior test.
    # Build an episode with a populated frames dict directly (bypassing
    # _make_episode) and confirm the kept/dropped split is mirrored there.
    n = 4
    state = np.stack([np.array([500.0, -500.0, 0.5, 0.9]), np.zeros(n)], axis=1)
    frames = np.zeros((n, 4, 4, 3))
    frames[0] = 1.0  # out-of-bounds frame, should be dropped
    frames[1] = 2.0  # out-of-bounds frame, should be dropped
    frames[2] = 3.0  # in-bounds frame, should be kept
    frames[3] = 4.0  # in-bounds frame, should be kept
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(n, dtype=np.float64),
        state=state,
        action=np.zeros((n, 1)),
        frames={"cam0": frames},
    )
    config = ProcessConfig(id="x")
    config.extreme_value_bounds = {"state": {0: [-1.0, 1.0]}, "action": {}}
    result = apply(episode, config)  # must not raise
    assert sorted(result.dropped_frame_indices) == [0, 1]
    assert result.episode.frames["cam0"].shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(result.episode.frames["cam0"][0], np.full((4, 4, 3), 3.0))
    np.testing.assert_array_equal(result.episode.frames["cam0"][1], np.full((4, 4, 3), 4.0))


def test_apply_bound_dim_index_beyond_array_width_is_skipped_not_indexed():
    # Regression for a legally-reachable degenerate input: a malformed
    # episode whose state/action arrays have fewer columns than the bound
    # dims referenced in config.extreme_value_bounds (e.g. a corrupted
    # episode with a different column count than the rest of its
    # dataset). Previously _out_of_bounds_mask indexed values[:, dim]
    # unconditionally and raised IndexError. Now it must skip any dim
    # that is out of range for values.shape[1] rather than checking it --
    # so with zero-width state/action, no frames are dropped for that
    # dim (there is nothing to check it against).
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(5, dtype=np.float64),
        state=np.zeros((5, 0)),
        action=np.zeros((5, 0)),
    )
    config = ProcessConfig(id="x")
    config.extreme_value_bounds = {"state": {0: [-1.0, 1.0]}, "action": {}}
    result = apply(episode, config)  # must not raise IndexError
    assert result.dropped_frame_indices == []
    assert result.episode.state.shape == (5, 0)
    assert result.episode.action.shape == (5, 0)
    assert result.episode.timestamps.shape[0] == 5
