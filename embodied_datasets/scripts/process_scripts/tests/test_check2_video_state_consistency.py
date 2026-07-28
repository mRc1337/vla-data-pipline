import pytest

lerobot = pytest.importorskip("lerobot")

from pathlib import Path

import numpy as np

from episode import CameraCalibration, Episode
from common.schema import ProcessConfig
from check2_video_state_consistency import apply, _disk_mask, _iou, _project_to_pixel

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")

# R maps base X -> camera Z, base Y -> camera X, base Z -> camera Y (a
# proper rotation: R's columns are (0,0,1), (1,0,0), (0,1,0)). Chosen so
# that the simple_arm.urdf tool0 position at joint angles [0, 0] -- (1.5, 0, 0)
# in the base frame -- projects to exactly the image center with fx=fy=100,
# cx=cy=16: z_cam=1.5, x_cam=y_cam=0, so u=v=16.
_EXTRINSICS_BASE_X_FACES_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _calibration():
    return CameraCalibration(fx=100.0, fy=100.0, cx=16.0, cy=16.0, extrinsics=_EXTRINSICS_BASE_X_FACES_CAMERA)


def _episode(camera_calibration=None, num_frames=3):
    joints = np.zeros((num_frames, 2))  # both joints at 0 -> FK position [1.5, 0, 0]
    frames = {"observation.image": np.zeros((num_frames, 32, 32, 3), dtype=np.uint8)}
    return Episode(
        episode_index=0,
        timestamps=np.arange(num_frames, dtype=np.float64),
        state=joints,
        action=joints.copy(),
        frames=frames,
        camera_calibration=camera_calibration or {},
    )


def _config(**overrides):
    defaults = dict(
        id="x",
        urdf_available=True,
        has_camera_calibration=True,
        sam3_model_id="facebook/sam3",
        sam3_text_prompt="robot gripper",
        sam3_hf_token_env="TEST_SAM3_TOKEN",
        gripper_radius_m=0.05,
        urdf_path=URDF_PATH,
        dof_per_arm=2,
        iou_threshold=0.5,
    )
    defaults.update(overrides)
    return ProcessConfig(**defaults)


@pytest.fixture(autouse=True)
def _sam3_token_env(monkeypatch):
    monkeypatch.setenv("TEST_SAM3_TOKEN", "fake-token")


def test_skips_when_urdf_not_available():
    episode = _episode(camera_calibration={"observation.image": _calibration()})
    config = _config(urdf_available=False)
    result = apply(episode, config)
    assert result.skip_reason == "urdf_not_available"


def test_skips_when_camera_calibration_not_available():
    episode = _episode(camera_calibration={})
    config = _config(has_camera_calibration=True)
    result = apply(episode, config)
    assert result.skip_reason == "camera_calibration_not_available"


def test_reports_sam3_not_configured_when_model_id_unset():
    episode = _episode(camera_calibration={"observation.image": _calibration()})
    config = _config(sam3_model_id=None)
    result = apply(episode, config)
    assert result.skip_reason == "sam3_service_not_configured"


def test_raises_when_model_id_set_but_hf_token_env_missing():
    episode = _episode(camera_calibration={"observation.image": _calibration()})
    config = _config(sam3_hf_token_env=None)
    with pytest.raises(RuntimeError):
        apply(episode, config)


def test_skips_when_calibration_missing_for_chosen_view():
    episode = _episode(camera_calibration={"some_other_view": _calibration()})
    config = _config()
    result = apply(episode, config)
    assert result.skip_reason == "camera_calibration_missing_for_view"


def test_skips_when_gripper_radius_not_configured():
    episode = _episode(camera_calibration={"observation.image": _calibration()})
    config = _config(gripper_radius_m=None)
    result = apply(episode, config)
    assert result.skip_reason == "gripper_radius_not_configured"


def test_project_to_pixel_matches_hand_computed_value():
    position_base = np.array([1.5, 0.0, 0.0])
    result = _project_to_pixel(position_base, _calibration())
    assert result is not None
    u, v, z_cam = result
    assert u == pytest.approx(16.0)
    assert v == pytest.approx(16.0)
    assert z_cam == pytest.approx(1.5)


def test_project_to_pixel_returns_none_when_behind_camera():
    position_base = np.array([-1.5, 0.0, 0.0])  # maps to z_cam = -1.5
    result = _project_to_pixel(position_base, _calibration())
    assert result is None


def test_disk_mask_shape_and_dtype():
    mask = _disk_mask((32, 32), (16.0, 16.0), 5.0)
    assert mask.shape == (32, 32)
    assert mask.dtype == bool
    assert bool(mask[16, 16]) is True


def test_iou_of_identical_masks_is_one():
    mask = _disk_mask((32, 32), (16.0, 16.0), 5.0)
    assert _iou(mask, mask) == pytest.approx(1.0)


def test_iou_of_disjoint_masks_is_zero():
    mask_a = np.zeros((32, 32), dtype=bool)
    mask_a[0:5, 0:5] = True
    mask_b = np.zeros((32, 32), dtype=bool)
    mask_b[27:32, 27:32] = True
    assert _iou(mask_a, mask_b) == pytest.approx(0.0)


def test_apply_reports_high_mean_iou_when_sam3_mask_matches_disk(monkeypatch):
    calibration = _calibration()
    episode = _episode(camera_calibration={"observation.image": calibration})
    config = _config()

    def fake_segment(self, frame, box_prompt):
        x0, y0, x1, y1 = box_prompt
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        radius = (x1 - x0) / 2.0
        return _disk_mask(frame.shape[:2], center, radius)

    from common.sam3_client import LocalSam3Client

    monkeypatch.setattr(LocalSam3Client, "segment", fake_segment)

    result = apply(episode, config)
    assert result.skip_reason is None
    assert result.stats["mean_iou"] == pytest.approx(1.0)
    assert result.rejected is False


def test_apply_flags_video_state_inconsistent_when_sam3_mask_does_not_match(monkeypatch):
    episode = _episode(camera_calibration={"observation.image": _calibration()})
    config = _config()

    def fake_segment(self, frame, box_prompt):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[0, 0] = True  # far from the projected point (image center), no overlap
        return mask

    from common.sam3_client import LocalSam3Client

    monkeypatch.setattr(LocalSam3Client, "segment", fake_segment)

    result = apply(episode, config)
    assert result.skip_reason == "video_state_inconsistent"
    assert result.stats["mean_iou"] == pytest.approx(0.0)
    assert result.rejected is False
