import pytest

lerobot = pytest.importorskip("lerobot")

from dataclasses import replace

import numpy as np

from episode import Episode
from common.schema import ProcessConfig
from stage2_trend_alignment import apply


def _episode_with_lag(lag: int, num_frames=100):
    rng = np.random.RandomState(0)
    action = rng.uniform(-1, 1, size=(num_frames, 2))
    # shift action so state-delta lags behind action by `lag` frames
    shifted_action = np.roll(action, lag, axis=0)
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=np.cumsum(action, axis=0),
        action=shifted_action,
    )


def test_zero_lag_is_not_skipped():
    episode = _episode_with_lag(lag=0)
    config = ProcessConfig(id="x", max_lag_frames=3, da_threshold=0.5)
    result = apply(episode, config)
    assert result.skip_reason is None
    assert result.stats["lag"] == 0


def test_lag_within_tolerance_is_shifted_and_aligned():
    episode = _episode_with_lag(lag=2)
    config = ProcessConfig(id="x", max_lag_frames=5, da_threshold=0.5)
    result = apply(episode, config)
    assert result.skip_reason is None
    assert result.episode.state.shape[0] == result.episode.action.shape[0]


def test_lag_beyond_tolerance_is_skipped():
    episode = _episode_with_lag(lag=20)
    config = ProcessConfig(id="x", max_lag_frames=3, da_threshold=0.5)
    result = apply(episode, config)
    assert result.skip_reason == "trend_misaligned"


def test_single_frame_episode_is_skipped_gracefully():
    episode = Episode(
        episode_index=0,
        timestamps=np.array([0.0]),
        state=np.zeros((1, 2)),
        action=np.zeros((1, 2)),
    )
    config = ProcessConfig(id="x")
    result = apply(episode, config)  # must not raise
    assert result.skip_reason == "insufficient_frames_for_trend_alignment"


def test_zero_dim_state_and_action_does_not_crash():
    # state/action with zero columns (no dims to correlate) must not crash
    # np.median()/int() on an empty per-dim lag list.
    num_frames = 10
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=np.zeros((num_frames, 0)),
        action=np.zeros((num_frames, 0)),
    )
    config = ProcessConfig(id="x")
    result = apply(episode, config)  # must not raise
    assert result.skip_reason is not None


def test_frames_are_trimmed_in_lockstep_with_state_on_nonzero_lag():
    """episode.frames is currently dormant in the real pipeline
    (load_lerobot_episodes never populates it), but stage3/check3 already
    rebuild it via keep_mask when they drop frames -- stage2 must do the
    same on a lag trim, or a future frame-loading wire-up will hand a
    length-T frames array alongside a length-(T-|lag|) state/action.
    """
    num_frames = 100
    episode = _episode_with_lag(lag=2, num_frames=num_frames)
    # Distinct per-frame markers (frame i's pixel value == i) so the exact
    # trim offset/direction can be verified, not just "some shorter length".
    original_frames = {"cam0": np.arange(num_frames, dtype=np.float64).reshape(num_frames, 1, 1)}
    episode = replace(episode, frames=original_frames)

    config = ProcessConfig(id="x", max_lag_frames=5, da_threshold=0.5)
    result = apply(episode, config)

    assert result.skip_reason is None
    detected_lag = result.stats["lag"]
    assert detected_lag != 0  # otherwise this test isn't exercising the trim path at all
    new_state_length = result.episode.state.shape[0]
    assert result.episode.frames["cam0"].shape[0] == new_state_length
    # Mirrors state's own trim exactly (frames[lag:] for positive lag,
    # frames[:lag] for negative lag) -- not action's opposite-end trim.
    expected = original_frames["cam0"][detected_lag:] if detected_lag > 0 else original_frames["cam0"][:detected_lag]
    np.testing.assert_array_equal(result.episode.frames["cam0"], expected)


def test_frames_are_trimmed_in_lockstep_with_state_on_negative_lag():
    num_frames = 100
    episode = _episode_with_lag(lag=-2, num_frames=num_frames)
    original_frames = {"cam0": np.arange(num_frames, dtype=np.float64).reshape(num_frames, 1, 1)}
    episode = replace(episode, frames=original_frames)

    config = ProcessConfig(id="x", max_lag_frames=5, da_threshold=0.5)
    result = apply(episode, config)

    assert result.skip_reason is None
    detected_lag = result.stats["lag"]
    assert detected_lag != 0
    new_state_length = result.episode.state.shape[0]
    assert result.episode.frames["cam0"].shape[0] == new_state_length
    expected = original_frames["cam0"][detected_lag:] if detected_lag > 0 else original_frames["cam0"][:detected_lag]
    np.testing.assert_array_equal(result.episode.frames["cam0"], expected)


def test_frames_are_left_alone_on_zero_lag():
    episode = _episode_with_lag(lag=0)
    original_frames = {"cam0": np.arange(100, dtype=np.float64).reshape(100, 1, 1)}
    episode = replace(episode, frames=original_frames)

    config = ProcessConfig(id="x", max_lag_frames=3, da_threshold=0.5)
    result = apply(episode, config)

    assert result.stats["lag"] == 0
    assert result.episode.frames["cam0"].shape[0] == result.episode.state.shape[0]
    np.testing.assert_array_equal(result.episode.frames["cam0"], original_frames["cam0"])


def test_absolute_action_frame_diffs_action_before_correlating():
    """action_frame="absolute" means the raw `action` array holds target
    positions, not per-frame deltas -- comparing it directly against
    state_delta (a rate-of-change quantity) is a unit mismatch that produces
    a wrong lag (independently verified: wrong sign, -3 instead of +3, with
    a much weaker directional-agreement score). Diffing the absolute action
    before correlating (the mirror of Qwen-RobotManip's "integrate delta
    actions to recover absolute values before comparison" -- differencing
    instead of integrating avoids the unbounded numerical drift a long
    episode's cumulative sum would accumulate) fixes this.
    """
    rng = np.random.RandomState(0)
    num_frames = 100
    true_lag = 3
    vel = rng.uniform(-1, 1, size=(num_frames, 2))
    shifted_vel = np.roll(vel, true_lag, axis=0)
    state = np.cumsum(shifted_vel, axis=0)
    action_absolute = np.cumsum(vel, axis=0)  # target position, not a delta
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=state,
        action=action_absolute,
    )
    config = ProcessConfig(id="x", action_frame="absolute", max_lag_frames=5, da_threshold=0.9)
    result = apply(episode, config)
    assert result.skip_reason is None
    assert result.stats["lag"] == true_lag


def test_minimal_two_frame_episode_does_not_crash():
    episode = Episode(
        episode_index=0,
        timestamps=np.arange(2, dtype=np.float64),
        state=np.array([[0.0, 0.0], [1.0, -1.0]]),
        action=np.array([[0.5, -0.5], [0.5, -0.5]]),
    )
    config = ProcessConfig(id="x", max_lag_frames=5, da_threshold=0.5)
    result = apply(episode, config)  # must not raise
    assert result.episode.state.shape[0] == result.episode.action.shape[0]
