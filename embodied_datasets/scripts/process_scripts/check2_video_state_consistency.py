"""Check2: video-state consistency (FK projection vs SAM3 box+text-prompt
segmentation IoU), gated by urdf_available AND has_camera_calibration. See
docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from episode import CameraCalibration, Episode, StageResult
from fk_backend import FkChain
from common.schema import ProcessConfig
from common.service_clients import NullClient, get_video_state_consistency_client

MAX_SAMPLED_FRAMES = 5


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.urdf_available:
        return StageResult(episode=episode, skip_reason="urdf_not_available")
    if not config.has_camera_calibration or not episode.camera_calibration:
        return StageResult(episode=episode, skip_reason="camera_calibration_not_available")

    client = get_video_state_consistency_client(config.sam3_model_id, config.sam3_text_prompt, config.sam3_hf_token_env)
    if isinstance(client, NullClient):
        return StageResult(episode=episode, skip_reason="sam3_service_not_configured")

    if not episode.frames:
        return StageResult(episode=episode, skip_reason="no_frames_available")
    view = next(iter(episode.frames))
    calibration = episode.camera_calibration.get(view)
    if calibration is None:
        return StageResult(episode=episode, skip_reason="camera_calibration_missing_for_view")

    if not config.gripper_radius_m:
        return StageResult(episode=episode, skip_reason="gripper_radius_not_configured")

    joint_dim = config.dof_per_arm
    if not joint_dim or joint_dim <= 0 or not config.urdf_path:
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    try:
        chain = FkChain(config.urdf_path)
    except ValueError:
        # See stage4_fk_consistency.py's identical except-clause: a URDF
        # whose structure doesn't match ikpy's assumptions is a property of
        # the dataset's URDF, not a bug in this pipeline.
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")
    if joint_dim != len(chain._active_link_indices):
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    frames_view = episode.frames[view]
    num_frames = frames_view.shape[0]
    if num_frames == 0 or episode.state.shape[1] < joint_dim:
        return StageResult(episode=episode, skip_reason="fk_check_not_feasible")

    ious: List[float] = []
    for t in _sample_frame_indices(num_frames):
        position_base, _quat = chain.forward(episode.state[t, :joint_dim])
        projected = _project_to_pixel(position_base, calibration)
        if projected is None:
            continue
        u, v, z_cam = projected
        pixel_radius = calibration.fx * config.gripper_radius_m / z_cam
        frame_shape = frames_view[t].shape[:2]
        expected_mask = _disk_mask(frame_shape, (u, v), pixel_radius)
        box_prompt = (u - pixel_radius, v - pixel_radius, u + pixel_radius, v + pixel_radius)
        actual_mask = client.segment(frames_view[t], box_prompt)
        ious.append(_iou(expected_mask, actual_mask))

    if not ious:
        return StageResult(episode=episode, skip_reason="fk_projection_behind_camera")

    mean_iou = float(np.mean(ious))
    stats = {"mean_iou": mean_iou, "sampled_frame_ious": ious}
    if mean_iou < config.iou_threshold:
        return StageResult(episode=episode, skip_reason="video_state_inconsistent", stats=stats)
    return StageResult(episode=episode, stats=stats)


def _sample_frame_indices(total: int) -> List[int]:
    """Same sampling rule as `common/vlm_client.py::_sample_frames` (used by Check1):
    T <= MAX_SAMPLED_FRAMES returns every index; otherwise first, last, evenly spaced in between."""
    if total <= MAX_SAMPLED_FRAMES:
        return list(range(total))
    indices = np.linspace(0, total - 1, num=MAX_SAMPLED_FRAMES).round().astype(int)
    return indices.tolist()


def _project_to_pixel(
    position_base: np.ndarray, calibration: CameraCalibration
) -> Optional[Tuple[float, float, float]]:
    """Pinhole-projects a 3D point in the robot base frame into pixel
    coordinates (u, v) plus its camera-frame depth z_cam. Returns None if
    the point is behind the camera (z_cam <= 0)."""
    p_h = np.array([position_base[0], position_base[1], position_base[2], 1.0])
    p_cam = calibration.extrinsics @ p_h
    x_cam, y_cam, z_cam = p_cam[0], p_cam[1], p_cam[2]
    if z_cam <= 0:
        return None
    u = calibration.fx * x_cam / z_cam + calibration.cx
    v = calibration.fy * y_cam / z_cam + calibration.cy
    return float(u), float(v), float(z_cam)


def _disk_mask(shape: Tuple[int, int], center: Tuple[float, float], radius: float) -> np.ndarray:
    """Rasterizes a filled disk of `radius` pixels centered at `center`
    (u, v) into a (H, W) bool mask of `shape`. cv2.circle clips
    automatically when the disk extends past the frame edges."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), max(int(round(radius)), 1), 1, -1)
    return mask.astype(bool)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
