import pytest

lerobot = pytest.importorskip("lerobot")

import numpy as np

from common.episode import Episode
from common.schema import ProcessConfig
from check1_instruction_consistency import apply


def _episode(instruction):
    return Episode(
        episode_index=0,
        timestamps=np.arange(3, dtype=np.float64),
        state=np.zeros((3, 1)),
        action=np.zeros((3, 1)),
        language_instruction=instruction,
    )


def test_skips_when_gate_field_false():
    episode = _episode("pick up the cup")
    config = ProcessConfig(id="x", has_language_instruction=False)
    result = apply(episode, config)
    assert result.skip_reason == "no_language_instruction"


def test_skips_when_episode_has_no_instruction_string():
    episode = _episode(None)
    config = ProcessConfig(id="x", has_language_instruction=True)
    result = apply(episode, config)
    assert result.skip_reason == "no_language_instruction"


def test_reports_vlm_not_configured_when_url_unset():
    episode = _episode("pick up the cup")
    config = ProcessConfig(id="x", has_language_instruction=True, vlm_service_url=None)
    result = apply(episode, config)
    assert result.skip_reason == "vlm_service_not_configured"


def test_raises_for_unimplemented_real_client():
    episode = _episode("pick up the cup")
    config = ProcessConfig(id="x", has_language_instruction=True, vlm_service_url="http://localhost:9000")
    with pytest.raises(NotImplementedError):
        apply(episode, config)
