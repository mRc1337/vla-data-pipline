"""Synthetic LeRobotDataset builders for tests that need real lerobot I/O
(as opposed to the stage/check unit tests, which construct Episode numpy
arrays directly and never touch lerobot).

Verified against the installed lerobot==0.4.4 API (lerobot.datasets.lerobot_dataset.
LeRobotDataset). Notable differences from a naive v2.1-era guess:
- `add_frame()` requires a "task" key in the frame dict (raises ValueError:
  "Missing features: {'task'}" otherwise).
- `LeRobotDataset.create(...)`/`.save_episode()` buffer episode metadata and only
  flush it to disk deterministically when `.finalize()` is called (relying on
  `__del__` for cleanup leaves the metadata parquet writer potentially unflushed
  and triggers a spurious "Exception ignored in __del__" at interpreter shutdown).
- Encoding a video feature requires an integer `fps` passed to
  `LeRobotDataset.create()`: PyAV's `add_stream(vcodec, fps, ...)` needs
  something with a `.numerator` attribute, and a plain Python `float` (even
  a whole number like `10.0`) raises `AttributeError: 'float' object has no
  attribute 'numerator'`. This only bites the video-encoding path -- the
  non-video path accepts (and, per `common/io.py`, relies on) `fps` being a
  float at runtime despite its `int` type hint.
- The default video codec (`libsvtav1`) hangs indefinitely (observed: still
  running after 60s+, force-killed) when encoding *very* small frames (8x8,
  16x16) -- SVT-AV1 warns "AQ mode 2 is unsupported with source dimensions"
  for those sizes and apparently never recovers. 32x32 encodes the same
  5-frame clip in well under a second with no warning, so `include_video`
  below uses 32x32 rather than the smaller sizes one might reach for first.
  (An explicit `vcodec="h264"` also encodes 8x8 instantly, confirming this
  is an SVT-AV1-at-tiny-resolution quirk, not a fundamental encoder limit --
  but changing the default codec is out of scope here.)
- A decoded video frame -- `dataset[i][<video_key>]` -- is a `torch.Tensor`
  of shape `(C, H, W)` (channel-first), dtype `float32`, values in `[0, 1]`.
  This is *not* the `(H, W, 3)` uint8 `0-255` layout `common/io.py`'s
  `load_lerobot_episodes` produces for `Episode.frames` -- that function
  transposes and rescales.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 32x32 rather than a smaller size: see the libsvtav1-hangs-at-tiny-
# resolutions note in the module docstring above.
_VIDEO_HEIGHT = 32
_VIDEO_WIDTH = 32
_VIDEO_KEY = "observation.image"


def make_synthetic_dataset(
    root: Path,
    repo_id: str,
    num_episodes: int,
    num_frames: int,
    state_dim: int = 4,
    action_dim: int = 4,
    fps: float = 10.0,
    task: str = "synthetic",
    include_video: bool = False,
) -> Path:
    """Build a tiny lerobot dataset for I/O tests.

    `task` and `include_video` default to the pre-existing behavior
    (a fixed "synthetic" task string, no video feature) so every caller
    written before these parameters existed keeps working unchanged.
    """
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    if include_video:
        features[_VIDEO_KEY] = {
            "dtype": "video",
            "shape": (_VIDEO_HEIGHT, _VIDEO_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
    # Video encoding requires an integer fps (see module docstring); the
    # non-video path keeps accepting/passing through the float `fps` as-is,
    # matching pre-existing behavior for every caller that doesn't ask for
    # video.
    create_fps = int(fps) if include_video else fps
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=create_fps, root=root, features=features, use_videos=include_video
    )
    rng = np.random.RandomState(0)
    for _episode in range(num_episodes):
        for _frame in range(num_frames):
            frame = {
                "observation.state": rng.uniform(-1, 1, size=state_dim).astype(np.float32),
                "action": rng.uniform(-1, 1, size=action_dim).astype(np.float32),
                "task": task,
            }
            if include_video:
                frame[_VIDEO_KEY] = rng.randint(0, 256, size=(_VIDEO_HEIGHT, _VIDEO_WIDTH, 3), dtype=np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return root
