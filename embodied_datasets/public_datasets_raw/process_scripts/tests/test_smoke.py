import pytest

lerobot = pytest.importorskip("lerobot")


def test_lerobot_importable():
    assert lerobot is not None
