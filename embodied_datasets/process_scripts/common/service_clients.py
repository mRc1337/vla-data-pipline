"""Pluggable VLM/SAM3 client interfaces for Check1/Check2. Real HTTP
clients are not implemented in this phase (see design doc section 12) --
only the Python interface and the NullClient fallback used when the
corresponding *_service_url config field is unset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class ConsistencyVerdict:
    consistent: bool
    reason: Optional[str] = None


class InstructionConsistencyClient(Protocol):
    def check(self, instruction: str, frames: List[np.ndarray]) -> ConsistencyVerdict: ...


class VideoStateConsistencyClient(Protocol):
    def segment(self, frame: np.ndarray) -> np.ndarray: ...


class NullClient:
    """Default fallback: any call reports 'not configured' so the caller
    can turn that into a StageResult.skip_reason."""

    def check(self, instruction: str, frames: List[np.ndarray]) -> ConsistencyVerdict:
        return ConsistencyVerdict(consistent=True, reason="vlm_service_not_configured")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        raise RuntimeError("sam3_service_not_configured")


def get_instruction_consistency_client(vlm_service_url: Optional[str]) -> InstructionConsistencyClient:
    if not vlm_service_url:
        return NullClient()
    raise NotImplementedError(
        "Real VLM HTTP client is not implemented yet -- configure "
        "vlm_service_url=None until the service is selected and deployed."
    )


def get_video_state_consistency_client(sam3_service_url: Optional[str]) -> VideoStateConsistencyClient:
    if not sam3_service_url:
        return NullClient()
    raise NotImplementedError(
        "Real SAM3 HTTP client is not implemented yet -- configure "
        "sam3_service_url=None until the service is selected and deployed."
    )
