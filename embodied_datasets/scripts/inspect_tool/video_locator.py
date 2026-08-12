"""Maps an (dataset root, episode_index, view_key) to the on-disk mp4 file
segment holding that episode's frames for that camera view.

lerobot packs multiple episodes into one physical video file up to
`video_files_size_in_mb` (see LeRobotDatasetMetadata) -- so a video file is
not "one episode, one file". Each episode's own window within a shared file
is given by that episode's `videos/{view_key}/from_timestamp` /
`.../to_timestamp` metadata fields, not by file boundaries. Verified
empirically against lerobot==0.4.4: three 5-frame/10fps synthetic episodes
end up sharing a single `chunk-000/file-000.mp4`, with
from_timestamp/to_timestamp of (0.0, 0.5), (0.5, 1.0), (1.0, 1.5)
respectively.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def video_clip_info(dataset_root: Path, episode_index: int, view_key: str) -> Optional[dict]:
    """Returns {"relative_path": str, "from_timestamp": float, "to_timestamp":
    float} describing where in `dataset_root`'s video files this episode's
    `view_key` frames live. None if `view_key` isn't a video feature in this
    dataset, or `episode_index` is out of range -- both are normal ("this
    dataset/episode has no video yet") rather than error conditions, so
    callers can skip that side of a raw/final comparison instead of
    crashing.
    """
    meta = LeRobotDatasetMetadata(repo_id=dataset_root.name, root=dataset_root)
    if meta.features.get(view_key, {}).get("dtype") != "video":
        return None
    if episode_index < 0 or episode_index >= len(meta.episodes):
        return None

    relative_path = meta.get_video_file_path(episode_index, view_key)
    episode_meta = meta.episodes[episode_index]
    return {
        "relative_path": str(relative_path),
        "from_timestamp": float(episode_meta[f"videos/{view_key}/from_timestamp"]),
        "to_timestamp": float(episode_meta[f"videos/{view_key}/to_timestamp"]),
    }
