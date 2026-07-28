"""Local-inference SAM3 client for Check2 video-state consistency. See
docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md
section 3.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class LocalSam3Client:
    """VideoStateConsistencyClient backed by a local SAM3 model. Model
    weights are loaded lazily -- only on the first .segment() call, not at
    construction -- and cached on the instance for reuse across calls.
    Constructing a LocalSam3Client is therefore cheap and dependency-free;
    only code paths that actually call .segment() need a real predictor.
    """

    def __init__(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path
        self._predictor = None

    def segment(self, frame: np.ndarray, point_prompt: Tuple[float, float]) -> np.ndarray:
        if self._predictor is None:
            self._predictor = self._load_predictor()
        return self._run_point_prompt(self._predictor, frame, point_prompt)

    def _load_predictor(self):
        """Real SAM3 import + model construction goes here once the actual
        SAM3 Python SDK is confirmed -- see design doc section 3 and
        section 6 (known limitation: SDK interface not yet confirmed).
        Tests monkeypatch this method rather than loading real weights."""
        raise NotImplementedError(
            f"SAM3 predictor loading is not implemented yet -- checkpoint_path={self._checkpoint_path!r} "
            "needs a real SAM3 SDK integration before LocalSam3Client can run for real."
        )

    def _run_point_prompt(self, predictor, frame: np.ndarray, point_prompt: Tuple[float, float]) -> np.ndarray:
        """Real SAM3 point-prompt inference call goes here -- see the
        docstring on _load_predictor. Tests monkeypatch this method."""
        raise NotImplementedError("SAM3 point-prompt inference is not implemented yet.")
