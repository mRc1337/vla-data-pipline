"""Common unit-conversion helpers for per-dataset convert() implementations.
Exists because a real bug (SO-100 real-data validation, see design doc
section 1) was caused by a per-dataset conversion treating degrees as
radians.
"""
from __future__ import annotations

import numpy as np


def degrees_to_radians(values: np.ndarray) -> np.ndarray:
    return np.deg2rad(values)
