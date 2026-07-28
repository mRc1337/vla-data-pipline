import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from episode import Episode
from common.schema import ProcessConfig
from check3_video_quality import apply


def _episode_with_frames(frames: np.ndarray):
    num_frames = frames.shape[0]
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=np.zeros((num_frames, 1)),
        action=np.zeros((num_frames, 1)),
        frames={"head": frames},
    )


def test_skips_when_no_frames():
    episode = Episode(
        episode_index=0, timestamps=np.arange(3, dtype=np.float64), state=np.zeros((3, 1)), action=np.zeros((3, 1))
    )
    config = ProcessConfig(id="x")
    result = apply(episode, config)
    assert result.skip_reason == "no_video_frames"


def test_black_frame_is_dropped():
    # Textured (not solid-color) non-black frames: a perfectly flat frame has
    # zero Laplacian variance regardless of its brightness, so an all-200
    # frame would (correctly) trip the blur check too and confound this
    # test's intent of isolating black-frame detection.
    rng = np.random.RandomState(1)
    frames = rng.randint(50, 200, size=(5, 8, 8, 3)).astype(np.uint8)
    frames[2] = 0  # fully black frame
    episode = _episode_with_frames(frames)
    config = ProcessConfig(id="x", black_threshold=10.0, blur_threshold=1.0, still_min_consecutive_frames=100)
    result = apply(episode, config)
    assert 2 in result.dropped_frame_indices
    assert result.episode.state.shape[0] == 4


def test_still_run_is_dropped():
    rng = np.random.RandomState(0)
    frames = rng.randint(50, 200, size=(40, 8, 8, 3)).astype(np.uint8)
    frames[10:35] = frames[10]  # 25 identical frames -> a still run
    episode = _episode_with_frames(frames)
    config = ProcessConfig(id="x", black_threshold=1.0, blur_threshold=1.0, still_threshold=0.5, still_min_consecutive_frames=20)
    result = apply(episode, config)
    assert set(range(10, 35)).issubset(set(result.dropped_frame_indices))


def test_still_run_includes_anchor_frame():
    # Regression for an off-by-one: the raw frame-diff signal at index i only
    # reflects "frame i looks like frame i-1", so a run of 25 identical
    # frames [10..34] only ever produces 24 True diffs (indices 11..34) --
    # frame 10 itself (the anchor the rest of the run duplicates) never gets
    # a "this looks like the previous frame" signal of its own and must be
    # pulled in explicitly. Assert the exact set (not just a subset) to lock
    # in that frame 10 is included, not just 11-34.
    rng = np.random.RandomState(0)
    frames = rng.randint(50, 200, size=(40, 8, 8, 3)).astype(np.uint8)
    frames[10:35] = frames[10]
    episode = _episode_with_frames(frames)
    config = ProcessConfig(id="x", black_threshold=1.0, blur_threshold=1.0, still_threshold=0.5, still_min_consecutive_frames=20)
    result = apply(episode, config)
    assert set(result.dropped_frame_indices) == set(range(10, 35))


def test_single_frame_episode_does_not_crash():
    rng = np.random.RandomState(2)
    frames = rng.randint(50, 200, size=(1, 8, 8, 3)).astype(np.uint8)
    episode = _episode_with_frames(frames)
    config = ProcessConfig(id="x")
    result = apply(episode, config)
    assert result.episode.state.shape[0] == 1
    assert result.dropped_frame_indices == []


def test_non_positive_still_min_consecutive_frames_does_not_crash():
    rng = np.random.RandomState(3)
    frames = rng.randint(50, 200, size=(10, 8, 8, 3)).astype(np.uint8)
    frames[3:6] = frames[3]  # a short held run
    episode = _episode_with_frames(frames)
    config = ProcessConfig(
        id="x", black_threshold=1.0, blur_threshold=1.0, still_threshold=0.5, still_min_consecutive_frames=0
    )
    result = apply(episode, config)
    # min_run <= 0 means "no minimum" -- any held frame-to-frame repeat
    # should be flaggable, not treated as an impossible/ignored threshold.
    assert set(range(3, 6)).issubset(set(result.dropped_frame_indices))


def test_float_frames_in_unit_range_not_falsely_flagged():
    # Regression: casting a [0, 1]-range float frame to uint8 truncates
    # everything to 0, which would make every frame read as both black and
    # maximally blurry. Episode.frames has no documented dtype/range
    # contract, so a real (if unusual) producer could hand this in.
    rng = np.random.RandomState(4)
    frames = rng.uniform(0.3, 0.9, size=(5, 8, 8, 3)).astype(np.float64)
    episode = _episode_with_frames(frames)
    # Thresholds scaled for the [0, 1] range these frames are actually in
    # (rather than the 0-255 scale black_threshold/blur_threshold otherwise
    # assume) so this test isolates the uint8-cast regression instead of the
    # separate, inherent black_threshold-scale ambiguity that comes from
    # Episode.frames having no documented dtype/range contract.
    config = ProcessConfig(id="x", black_threshold=0.1, blur_threshold=0.1, still_min_consecutive_frames=100)
    result = apply(episode, config)
    assert result.dropped_frame_indices == []
