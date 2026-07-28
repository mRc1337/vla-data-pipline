"""Pluggable VLM/SAM3 client interfaces for Check1/Check2.
get_instruction_consistency_client() wires up a real OpenAIVLMClient (see
docs/superpowers/specs/2026-07-27-vlm-client-check1-design.md) when
vlm_service_url is configured. get_video_state_consistency_client() wires
up a real LocalSam3Client (see
docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md)
when sam3_checkpoint_path is configured. NullClient is the fallback used
when the corresponding config field is unset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np


@dataclass
class ConsistencyVerdict:
    consistent: bool
    reason: Optional[str] = None


class InstructionConsistencyClient(Protocol):
    def check(self, instruction: str, frames: List[np.ndarray]) -> ConsistencyVerdict: ...


class VideoStateConsistencyClient(Protocol):
    def segment(self, frame: np.ndarray, point_prompt: Tuple[float, float]) -> np.ndarray: ...


class NullClient:
    """Default fallback: any call reports 'not configured' so the caller
    can turn that into a StageResult.skip_reason."""

    def check(self, instruction: str, frames: List[np.ndarray]) -> ConsistencyVerdict:
        return ConsistencyVerdict(consistent=True, reason="vlm_service_not_configured")

    def segment(self, frame: np.ndarray, point_prompt: Tuple[float, float]) -> np.ndarray:
        raise RuntimeError("sam3_service_not_configured")


def get_instruction_consistency_client(
    vlm_service_url: Optional[str],
    vlm_model_name: Optional[str],
    vlm_api_key_env: Optional[str],
) -> InstructionConsistencyClient:
    if not vlm_service_url:
        return NullClient()
    if not vlm_api_key_env:
        raise RuntimeError(
            f"vlm_service_url={vlm_service_url!r} is configured but vlm_api_key_env is unset -- "
            "refusing to silently fall back to NullClient for a service the config says should be real."
        )
    api_key = os.environ.get(vlm_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"vlm_api_key_env={vlm_api_key_env!r} names an environment variable that is not set -- "
            "refusing to silently fall back to NullClient for a service the config says should be real."
        )
    from common.vlm_client import OpenAIVLMClient  # local import: vlm_client.py imports ConsistencyVerdict from this module at top level

    return OpenAIVLMClient(base_url=vlm_service_url, model=vlm_model_name or "qwen2.5-vl-7b-instruct", api_key=api_key)


def get_video_state_consistency_client(sam3_checkpoint_path: Optional[str]) -> VideoStateConsistencyClient:
    if not sam3_checkpoint_path:
        return NullClient()
    from common.sam3_client import LocalSam3Client  # local import: mirrors get_instruction_consistency_client's OpenAIVLMClient import

    return LocalSam3Client(checkpoint_path=sam3_checkpoint_path)
