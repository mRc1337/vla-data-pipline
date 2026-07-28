"""Check2: video-state consistency (FK projection vs SAM3 segmentation
IoU), gated by urdf_available AND has_camera_calibration. Real SAM3
integration is out of scope for this phase (design doc section 12);
NullClient short-circuits with a skip_reason. See design doc section 7 row 7.
"""
from __future__ import annotations

from episode import Episode, StageResult
from common.schema import ProcessConfig
from common.service_clients import NullClient, get_video_state_consistency_client


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.urdf_available:
        return StageResult(episode=episode, skip_reason="urdf_not_available")
    if not config.has_camera_calibration:
        return StageResult(episode=episode, skip_reason="camera_calibration_not_available")

    client = get_video_state_consistency_client(config.sam3_service_url)
    if isinstance(client, NullClient):
        return StageResult(episode=episode, skip_reason="sam3_service_not_configured")

    raise NotImplementedError("real SAM3 client path is unreachable until one is implemented")
