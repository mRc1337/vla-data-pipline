"""Check1: language-instruction consistency, gated by
has_language_instruction. Real VLM integration is out of scope for this
phase (design doc section 12); NullClient's verdict.reason surfaces
directly as the StageResult.skip_reason. See design doc section 7 row 6.
"""
from __future__ import annotations

from episode import Episode, StageResult
from common.schema import ProcessConfig
from common.service_clients import NullClient, get_instruction_consistency_client


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.has_language_instruction or not episode.language_instruction:
        return StageResult(episode=episode, skip_reason="no_language_instruction")

    client = get_instruction_consistency_client(config.vlm_service_url)
    if isinstance(client, NullClient):
        verdict = client.check(episode.language_instruction, list(episode.frames.values()))
        return StageResult(episode=episode, skip_reason=verdict.reason)

    raise NotImplementedError("real VLM client path is unreachable until one is implemented")
