import numpy as np

from common_convert.unit_conversion import degrees_to_radians


def test_degrees_to_radians_converts_known_values():
    degrees = np.array([0.0, 90.0, 180.0])
    radians = degrees_to_radians(degrees)
    assert np.allclose(radians, [0.0, np.pi / 2, np.pi])
