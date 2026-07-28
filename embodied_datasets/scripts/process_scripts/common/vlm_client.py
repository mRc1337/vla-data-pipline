"""OpenAI-compatible VLM client for Check1 instruction consistency. See
docs/superpowers/specs/2026-07-27-vlm-client-check1-design.md sections 3-4.
"""
from __future__ import annotations

import base64
import json
import time
from typing import List

import cv2
import numpy as np
from openai import OpenAI

from common.service_clients import ConsistencyVerdict

MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
MAX_SAMPLED_FRAMES = 5

SYSTEM_PROMPT = (
    "You are checking whether a natural-language instruction matches what a "
    "robot is doing in a short video, sampled as a few frames. First "
    "describe the scene, then describe the action being performed, then "
    "decide whether the instruction is consistent with that action. "
    "Respond with exactly one JSON object with these keys: "
    '"scene_description" (string), "action_description" (string), '
    '"consistent" (boolean), "reason" (string).'
)


def _sample_frames(view: np.ndarray) -> List[np.ndarray]:
    """Uniformly sample up to MAX_SAMPLED_FRAMES frames from a (T,H,W,C)
    view: first, last, and evenly spaced in between. T <= MAX_SAMPLED_FRAMES
    returns every frame unmodified. T == 0 returns an empty list."""
    total = view.shape[0]
    if total == 0:
        return []
    if total <= MAX_SAMPLED_FRAMES:
        return [view[i] for i in range(total)]
    indices = np.linspace(0, total - 1, num=MAX_SAMPLED_FRAMES).round().astype(int)
    return [view[i] for i in indices]


def _encode_frame_to_data_url(frame: np.ndarray) -> str:
    """frame is (H,W,C) uint8 RGB (lerobot_io.py's decode convention).
    cv2.imencode expects BGR, hence the channel swap before encoding."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame for VLM request")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(instruction: str, sampled_frames: List[np.ndarray]) -> list:
    content = [{"type": "text", "text": f"Instruction: {instruction}"}]
    for frame in sampled_frames:
        content.append({"type": "image_url", "image_url": {"url": _encode_frame_to_data_url(frame)}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class OpenAIVLMClient:
    """InstructionConsistencyClient backed by an OpenAI-compatible chat
    completions endpoint (e.g. a self-hosted Qwen2.5-VL server)."""

    def __init__(self, base_url: str, model: str, api_key: str):
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def check(self, instruction: str, frames: List[np.ndarray]) -> ConsistencyVerdict:
        try:
            view = frames[0]
        except IndexError:
            return ConsistencyVerdict(consistent=True, reason="no_frames_available")

        sampled = _sample_frames(view)
        if not sampled:
            return ConsistencyVerdict(consistent=True, reason="no_frames_available")

        messages = _build_messages(instruction, sampled)

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                payload = json.loads(response.choices[0].message.content)
                return ConsistencyVerdict(
                    consistent=bool(payload["consistent"]),
                    reason=payload.get("reason"),
                )
            except Exception:  # noqa: BLE001 -- any failure fails safe, design doc section 4 step 5
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
        return ConsistencyVerdict(consistent=True, reason="vlm_call_failed")
