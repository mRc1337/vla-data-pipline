"""Check1: language-instruction consistency, gated by
has_language_instruction. Delegates to whichever InstructionConsistencyClient
get_instruction_consistency_client() returns (NullClient or a real
OpenAIVLMClient) -- see
docs/superpowers/specs/2026-07-27-vlm-client-check1-design.md. The client's
verdict.reason surfaces directly as the StageResult.skip_reason regardless
of verdict.consistent; this module does not reject episodes on an
inconsistent verdict (unchanged from the original NullClient-only
behavior -- see the 2026-07-17 design doc section 7 row 6).
"""
from __future__ import annotations

from episode import Episode, StageResult
from common.schema import ProcessConfig
from common.service_clients import get_instruction_consistency_client


def apply(episode: Episode, config: ProcessConfig) -> StageResult:
    if not config.has_language_instruction or not episode.language_instruction:
        return StageResult(episode=episode, skip_reason="no_language_instruction")

    client = get_instruction_consistency_client(config.vlm_service_url, config.vlm_model_name, config.vlm_api_key_env)
    verdict = client.check(episode.language_instruction, list(episode.frames.values()))
    return StageResult(episode=episode, skip_reason=verdict.reason)
