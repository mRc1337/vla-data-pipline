import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from shared.episode import Episode
from common.schema import ProcessConfig
from stage1_sudden_change import apply


def _smooth_episode(num_frames=60, num_dims=2):
    t = np.linspace(0, 2 * np.pi, num_frames)
    state = np.stack([np.sin(t), np.cos(t)], axis=1)[:, :num_dims]
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=state.copy(),
        action=state.copy(),
    )


def test_smooth_episode_is_not_rejected_and_barely_changed():
    episode = _smooth_episode()
    config = ProcessConfig(id="x")
    result = apply(episode, config)
    assert result.rejected is False
    assert np.allclose(result.episode.state, episode.state, atol=0.05)


def test_single_spike_is_flagged_and_interpolated():
    episode = _smooth_episode()
    episode.state[30, 0] += 10.0  # inject a sudden spike
    config = ProcessConfig(id="x", residual_threshold=0.2)
    result = apply(episode, config)
    assert result.rejected is False
    assert 30 in result.dropped_frame_indices
    # interpolated value should be close to the surrounding smooth signal, not the spike
    assert abs(result.episode.state[30, 0] - episode.state[29, 0]) < 1.0


def test_too_many_spikes_rejects_episode():
    episode = _smooth_episode(num_frames=20)
    episode.state[:, 0] += np.random.RandomState(0).uniform(5, 10, size=20)
    config = ProcessConfig(id="x", residual_threshold=0.2, episode_reject_threshold=0.3)
    result = apply(episode, config)
    assert result.rejected is True


@pytest.mark.parametrize(
    "num_frames,polyorder,expect_skip",
    [
        (1, 1, True),
        (1, 2, True),
        (1, 3, True),
        (2, 0, False),
        (2, 2, True),
        (2, 3, True),
        (3, 2, False),
        (3, 3, True),
        (4, 2, False),
        (4, 3, True),
        (5, 2, False),
        (5, 3, False),
    ],
)
def test_window_selection_matrix_matches_expected_skip_pattern(num_frames, polyorder, expect_skip):
    # Uses the shipped defaults (savgol_window=11) and only varies num_frames/polyorder,
    # reproducing the exact matrix from the review finding. Must never raise ValueError.
    # (Whether a non-skip case ends up rejected/accepted is a separate concern driven by
    # the residual/accel/jerk thresholds on these tiny arbitrary episodes -- not asserted
    # here; the only thing under test is that window selection never crashes savgol_filter
    # and correctly decides skip vs. proceed.)
    episode = _smooth_episode(num_frames=num_frames, num_dims=2)
    config = ProcessConfig(id="x", savgol_polyorder=polyorder)
    result = apply(episode, config)  # must not raise ValueError
    if expect_skip:
        assert result.skip_reason == "episode_too_short_for_savgol"
    else:
        assert result.skip_reason is None


def test_single_frame_episode_with_zero_polyorder_skips_without_raising():
    # Regression test: num_frames=1 combined with savgol_polyorder=0 (a legal
    # ProcessConfig value) used to fall through the window-selection guard
    # (window=1, 1 <= polyorder(0) is False) straight into np.gradient, which
    # raises ValueError because it needs >=2 points along the differentiated
    # axis regardless of the chosen savgol window/polyorder. apply() must
    # never raise for short episodes; it must return the skip reason instead.
    episode = _smooth_episode(num_frames=1, num_dims=2)
    config = ProcessConfig(id="x", savgol_polyorder=0)
    result = apply(episode, config)  # must not raise ValueError
    assert result.skip_reason == "episode_too_short_for_savgol"


def test_interpolation_skipped_stat_set_when_too_few_valid_frames():
    # Extremely tight thresholds flag nearly every frame; a permissive
    # episode_reject_threshold keeps the episode from being rejected outright,
    # but with fewer than 2 valid frames left, interpolation can't run and the
    # raw (uncleaned) state must be reported honestly via the stat.
    episode = _smooth_episode(num_frames=5, num_dims=2)
    config = ProcessConfig(
        id="x",
        residual_threshold=1e-12,
        accel_threshold=1e-12,
        jerk_threshold=1e-12,
        episode_reject_threshold=1.0,
    )
    result = apply(episode, config)
    assert result.rejected is False
    assert len(result.dropped_frame_indices) >= 4  # fewer than 2 valid frames remain
    assert result.stats.get("interpolation_skipped") is True
    # state was left untouched since there weren't enough valid frames to interpolate from
    assert np.array_equal(result.episode.state, episode.state)
