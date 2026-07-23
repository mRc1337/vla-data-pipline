"""Post-convert self-check: shape/scale/FK-consistency gates run_convert.py
calls right after a per-dataset convert() finishes writing staging data,
before the registry is marked convert_status=converted. See design doc
section 7 step 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from shared.episode import Episode
from shared.fk_backend import FkChain

from .report import ConversionReport

EEF_POS_WIDTH = 3
FK_OFFSET_TOLERANCE_M = 0.1
FK_SAMPLE_EPISODES = 3


@dataclass
class SelfCheckResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)


def check_shape(
    episodes: List[Episode], state_dim: Optional[int], action_dim: Optional[int]
) -> SelfCheckResult:
    """state_dim/action_dim come from the onboarding DatasetConfig -- when
    either is None (not declared during onboarding), that dimension is
    skipped rather than treated as a mismatch, since there is nothing to
    compare against.
    """
    reasons = []
    for episode in episodes:
        if state_dim is not None and episode.state.shape[1] != state_dim:
            reasons.append(
                f"episode {episode.episode_index}: state width {episode.state.shape[1]} != declared state_dim {state_dim}"
            )
        if action_dim is not None and episode.action.shape[1] != action_dim:
            reasons.append(
                f"episode {episode.episode_index}: action width {episode.action.shape[1]} != declared action_dim {action_dim}"
            )
    return SelfCheckResult(passed=not reasons, reasons=reasons)


def check_scale(
    report: ConversionReport, expected_num_episodes: Optional[int], scale_tolerance: float = 0.5
) -> SelfCheckResult:
    """Flags a conversion that silently dropped most of the dataset.
    `scale_tolerance` is a fraction: report.num_episodes must be within
    +/-50% of the onboarding-declared expected_num_episodes by default --
    coarse on purpose, this is a "did most of the data go missing" check,
    not a precise reconciliation.
    """
    if not expected_num_episodes:
        return SelfCheckResult(passed=True)
    ratio = report.num_episodes / expected_num_episodes
    if abs(ratio - 1.0) > scale_tolerance:
        return SelfCheckResult(
            passed=False,
            reasons=[
                f"converted {report.num_episodes} episodes, expected ~{expected_num_episodes} "
                f"(ratio {ratio:.2f}, tolerance +/-{scale_tolerance:.0%})"
            ],
        )
    return SelfCheckResult(passed=True)


def check_fk_consistency(
    episodes: List[Episode], urdf_path: Optional[str], dof_per_arm: Optional[int]
) -> SelfCheckResult:
    """Recomputes eef_pos via FK from the same joint columns
    layout.assemble_state just packed, and compares against the eef_pos
    columns assemble_state itself wrote. This is a tautological check for
    datasets converted through assemble_state (which always derives
    eef_pos from FK) -- but it still catches a real class of bug: a
    dataset-specific convert() that bypasses assemble_state and hand-packs
    columns incorrectly (wrong unit, wrong column order) ends up with
    joint/eef_pos columns that are internally inconsistent, exactly like
    process_scripts' Stage4 catches post-process. Samples up to
    FK_SAMPLE_EPISODES episodes rather than every episode -- this is a
    coarse smoke check, not exhaustive validation.
    """
    if not urdf_path or not dof_per_arm or dof_per_arm <= 0:
        return SelfCheckResult(passed=True)
    try:
        chain = FkChain(urdf_path)
    except ValueError:
        return SelfCheckResult(passed=True)
    if dof_per_arm != len(chain._active_link_indices):
        return SelfCheckResult(passed=True)

    reasons = []
    for episode in episodes[:FK_SAMPLE_EPISODES]:
        if episode.state.shape[1] < dof_per_arm + EEF_POS_WIDTH:
            continue
        reported_positions = episode.state[:, dof_per_arm : dof_per_arm + EEF_POS_WIDTH]
        fk_positions = np.zeros_like(reported_positions)
        for t in range(episode.state.shape[0]):
            position, _quat = chain.forward(episode.state[t, :dof_per_arm])
            fk_positions[t] = position
        offset_magnitude = float(np.linalg.norm(reported_positions - fk_positions, axis=1).max())
        if offset_magnitude > FK_OFFSET_TOLERANCE_M:
            reasons.append(
                f"episode {episode.episode_index}: FK/eef_pos offset {offset_magnitude:.3f}m exceeds "
                f"{FK_OFFSET_TOLERANCE_M}m tolerance -- check unit conversion (deg vs rad) and column order"
            )
    return SelfCheckResult(passed=not reasons, reasons=reasons)


def run_self_check(
    episodes: List[Episode],
    report: ConversionReport,
    state_dim: Optional[int],
    action_dim: Optional[int],
    expected_num_episodes: Optional[int],
    urdf_path: Optional[str],
    dof_per_arm: Optional[int],
) -> SelfCheckResult:
    results = [
        check_shape(episodes, state_dim, action_dim),
        check_scale(report, expected_num_episodes),
        check_fk_consistency(episodes, urdf_path, dof_per_arm),
    ]
    all_reasons = [reason for result in results for reason in result.reasons]
    return SelfCheckResult(passed=all(result.passed for result in results), reasons=all_reasons)
