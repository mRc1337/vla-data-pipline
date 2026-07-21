import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from common.episode import Episode
from common.schema import ProcessConfig
from check2_video_state_consistency import apply


def _episode():
    return Episode(
        episode_index=0,
        timestamps=np.arange(3, dtype=np.float64),
        state=np.zeros((3, 1)),
        action=np.zeros((3, 1)),
    )


def test_skips_when_urdf_not_available():
    episode = _episode()
    config = ProcessConfig(id="x", urdf_available=False, has_camera_calibration=True)
    result = apply(episode, config)
    assert result.skip_reason == "urdf_not_available"


def test_skips_when_camera_calibration_missing():
    episode = _episode()
    config = ProcessConfig(id="x", urdf_available=True, has_camera_calibration=False)
    result = apply(episode, config)
    assert result.skip_reason == "camera_calibration_not_available"


def test_reports_sam3_not_configured_when_url_unset():
    episode = _episode()
    config = ProcessConfig(id="x", urdf_available=True, has_camera_calibration=True, sam3_service_url=None)
    result = apply(episode, config)
    assert result.skip_reason == "sam3_service_not_configured"


def test_raises_for_unimplemented_real_client():
    episode = _episode()
    config = ProcessConfig(
        id="x", urdf_available=True, has_camera_calibration=True, sam3_service_url="http://localhost:9001"
    )
    with pytest.raises(NotImplementedError):
        apply(episode, config)
