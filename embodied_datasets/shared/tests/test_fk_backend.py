import pytest

lerobot = pytest.importorskip("lerobot")

import math
from pathlib import Path

import numpy as np

from shared.fk_backend import FkChain

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


def test_forward_at_zero_angles_is_fully_extended():
    chain = FkChain(URDF_PATH)
    position, quat = chain.forward(np.array([0.0, 0.0]))
    assert np.allclose(position[:2], [1.5, 0.0], atol=1e-6)
    assert quat.shape == (4,)


def test_forward_at_quarter_turns():
    chain = FkChain(URDF_PATH)
    position, _quat = chain.forward(np.array([math.pi / 2, 0.0]))
    assert np.allclose(position[:2], [0.0, 1.5], atol=1e-6)
