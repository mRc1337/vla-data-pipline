"""Orchestrates Stage1-5 + Check1-3 + unify_representation for one dataset,
writes the cleaned output, updates datasets_registry.yaml, and generates
the dataset README. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 10.

Loads convert_scripts/common (RegistryEntry/DatasetConfig/load_registry/
save_registry/load_dataset_config) under the alias "registry_common" via
importlib.util instead of sys.path, because process_scripts/common and
convert_scripts/common are both literally named "common" -- putting both
directories on sys.path would make whichever imports first win for every
subsequent `import common` in the process (verified empirically while
writing this plan). This loader sidesteps that entirely.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.episode import Episode  # noqa: E402
from common.io import load_process_config  # noqa: E402
from shared.lerobot_io import load_lerobot_episodes, write_lerobot_episodes  # noqa: E402
from common.schema import ProcessConfig  # noqa: E402

import stage1_sudden_change  # noqa: E402
import stage2_trend_alignment  # noqa: E402
import stage3_extreme_value  # noqa: E402
import stage4_fk_consistency  # noqa: E402
import stage5_orientation_alignment  # noqa: E402
import check1_instruction_consistency  # noqa: E402
import check2_video_state_consistency  # noqa: E402
import check3_video_quality  # noqa: E402
import unify_representation  # noqa: E402

CONVERT_SCRIPTS_COMMON_DIR = Path(__file__).resolve().parents[1] / "convert_scripts" / "common"


def _load_registry_common() -> ModuleType:
    alias = "registry_common"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias, CONVERT_SCRIPTS_COMMON_DIR / "__init__.py", submodule_search_locations=[str(CONVERT_SCRIPTS_COMMON_DIR)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    module = sys.modules[alias]
    module.schema = importlib.import_module(f"{alias}.schema")
    module.io = importlib.import_module(f"{alias}.io")
    module.paths = importlib.import_module(f"{alias}.paths")
    return module


def _apply_gate_fields(config: ProcessConfig, dataset_config) -> None:
    action_space = getattr(dataset_config.action_space, "value", dataset_config.action_space)
    config.fk_check_feasible = bool(dataset_config.urdf_available) and action_space in {"joint_position", "eef_pose"}
    config.urdf_available = bool(dataset_config.urdf_available)
    config.has_camera_calibration = bool(dataset_config.has_camera_calibration)
    config.has_language_instruction = bool(dataset_config.has_language_instruction)
    config.embodiment_class = getattr(dataset_config.embodiment_class, "value", dataset_config.embodiment_class)
    config.num_arms = dataset_config.num_arms or 1
    config.dof_per_arm = dataset_config.dof_per_arm
    config.gripper_type = getattr(dataset_config.gripper_type, "value", dataset_config.gripper_type) or "unknown"
    config.has_mobile_base = bool(dataset_config.has_mobile_base)


def _resolve_fps(dataset_config) -> float:
    """`dataset_config.fps` (convert_scripts' DatasetConfig) is Optional --
    some registry entries may not have it populated yet. lerobot derives
    every frame's timestamp from frame_index / fps, so it needs a concrete
    scalar regardless of `dataset_config.fps_variable` (that flag just notes
    the *original* recording had non-uniform fps; it doesn't change what we
    write here, since lerobot has no per-frame-variable-fps concept). Fall
    back to the pre-existing hardcoded default (1.0) only when fps is
    genuinely unknown, rather than crashing.
    """
    if dataset_config is not None and dataset_config.fps:
        return float(dataset_config.fps)
    return 1.0


def run_dataset(dataset_id: str, staging_path: Path, output_path: Path, process_config_path: Path, dataset_config=None) -> dict:
    config = load_process_config(process_config_path)
    if dataset_config is not None:
        _apply_gate_fields(config, dataset_config)
    fps = _resolve_fps(dataset_config)

    episodes = load_lerobot_episodes(staging_path)
    log: List[tuple] = []

    survivors = []
    for episode in episodes:
        result = stage1_sudden_change.apply(episode, config)
        log.append(("stage1_sudden_change", episode.episode_index, result.skip_reason, result.rejected))
        if result.rejected:
            continue
        episode = result.episode

        result = stage2_trend_alignment.apply(episode, config)
        log.append(("stage2_trend_alignment", episode.episode_index, result.skip_reason, result.rejected))
        if result.skip_reason:
            continue
        survivors.append(result.episode)

    config.extreme_value_bounds = stage3_extreme_value.compute_bounds(survivors, config) if survivors else None

    final_episodes: List[Episode] = []
    # unify_representation's canonical_mask is provably dataset-constant (it
    # depends only on config.dof_per_arm/num_arms/gripper_type/
    # has_mobile_base, all fixed for the whole run, not per-episode data) --
    # so it's captured once from whichever episode's result first produces
    # it (non-skipped) rather than threaded through every Episode object.
    canonical_mask: Optional[np.ndarray] = None
    for episode in survivors:
        result = stage3_extreme_value.apply(episode, config)
        log.append(("stage3_extreme_value", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        if episode.state.shape[0] == 0:
            log.append(("run_pipeline", episode.episode_index, "all_frames_dropped", True))
            continue

        result = stage4_fk_consistency.apply(episode, config)
        log.append(("stage4_fk_consistency", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        result = stage5_orientation_alignment.apply(episode, config)
        log.append(("stage5_orientation_alignment", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        result = check1_instruction_consistency.apply(episode, config)
        log.append(("check1_instruction_consistency", episode.episode_index, result.skip_reason, result.rejected))

        result = check2_video_state_consistency.apply(episode, config)
        log.append(("check2_video_state_consistency", episode.episode_index, result.skip_reason, result.rejected))

        result = check3_video_quality.apply(episode, config)
        log.append(("check3_video_quality", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        if episode.state.shape[0] == 0:
            log.append(("run_pipeline", episode.episode_index, "all_frames_dropped", True))
            continue

        result = unify_representation.apply(episode, config)
        log.append(("unify_representation", episode.episode_index, result.skip_reason, result.rejected))
        episode = result.episode

        # For robot-collected embodiment classes (result.skip_reason is None),
        # unify_representation computes a cross-embodiment canonical 80-dim
        # projection of episode.state but does NOT itself replace
        # episode.state (see unify_representation.py's apply() docstring/
        # tests -- it returns the original episode object with the
        # canonical vector only in result.stats). Replacing it here, rather
        # than in unify_representation.apply, keeps that module's contract
        # (compute, don't mutate) and makes this the single place that
        # decides what actually gets written to the output dataset.
        # episode.action is explicitly left untouched -- Task 15 only
        # canonicalizes state, by design.
        if result.skip_reason is None:
            episode = replace(episode, state=result.stats["canonical_state"])
            if canonical_mask is None:
                canonical_mask = result.stats["canonical_mask"]

        final_episodes.append(episode)

    if final_episodes:
        write_lerobot_episodes(final_episodes, output_path, fps=fps, robot_type=dataset_id, canonical_mask=canonical_mask)

    total_frames = sum(ep.state.shape[0] for ep in final_episodes)
    return {
        "input_episodes": len(episodes),
        "output_episodes": len(final_episodes),
        "output_frames": total_frames,
        "fps": fps,
        "log": log,
    }


README_TEMPLATE = """# {name}

## 基本信息
- 来源: {source_url}
- License: {license}

## 本体信息
- embodiment_class: {embodiment_class}
- robot_platform: {robot_platform}
- num_arms: {num_arms}
- gripper_type: {gripper_type}

## 规模
- 清洗前 episode 数: {input_episodes}
- 清洗后 episode 数: {output_episodes}
- 清洗后帧数: {output_frames}

## 表示层
- state_dim: {state_dim}
- action_dim: {action_dim}
- world_frame_convention: {world_frame_convention}

## 处理记录
{processing_log}

## 已知局限
- 80 维统一表示层里的灵巧手槎位是 21 维（历史设计依据：曾onboard过的 humanoidbench
  Shadow Hand 实测 21 DOF；该数据集已从注册表移除，目前注册表内机器人采集灵巧手
  数据集的实测最大自由度为 16，见 arcap）；未来若出现超过 21 维的机器人
  灵巧手会被截断。MANO/人手视频数据集（human_hand/human_full_body）完全不经过这一层，
  不受此限制，完整参数保留在 data_root/staging/ 原始数据中。详见
  embodied_datasets/README.md 的"跨本体统一表示层"一节。
"""


def generate_dataset_readme(dataset_config, config: ProcessConfig, stats: dict) -> str:
    processing_log_lines = [
        f"- {stage}: episode {episode_index} -> {skip_reason or ('rejected' if rejected else 'ok')}"
        for stage, episode_index, skip_reason, rejected in stats["log"]
    ]
    return README_TEMPLATE.format(
        name=dataset_config.name,
        source_url=dataset_config.source_url or "unknown",
        license=getattr(dataset_config.license, "value", dataset_config.license) or "unknown",
        embodiment_class=config.embodiment_class or "unknown",
        robot_platform=getattr(dataset_config.robot_platform, "value", dataset_config.robot_platform) or "unknown",
        num_arms=config.num_arms,
        gripper_type=config.gripper_type,
        input_episodes=stats["input_episodes"],
        output_episodes=stats["output_episodes"],
        output_frames=stats["output_frames"],
        state_dim=dataset_config.state_dim or "unknown",
        action_dim=dataset_config.action_dim or "unknown",
        world_frame_convention=config.world_frame_convention,
        processing_log="\n".join(processing_log_lines) or "- (no episodes processed)",
    )


def _dir_size_bytes(path: Path) -> int:
    """Mirrors convert_scripts/common/scan_downloaded_datasets.py's
    `_dir_size_bytes` (module-private there, so not imported directly across
    the process_scripts/convert_scripts boundary) -- same rglob-and-sum
    approach, feeding the same `round(size_bytes / 1e9, 2)` convention
    run_scan_downloaded_datasets.py uses for `storage_size_gb`.
    """
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def compute_final_local_path(output_path: Path, data_root: Path) -> str:
    """Registry field is documented (design doc section 5.1) as relative to
    `data_root/public_datasets/lerobot_v3_0/`, not `data_root/` -- so for
    `output_path == data_root/public_datasets/lerobot_v3_0/<dataset_id>`
    this yields just `<dataset_id>`. Mirrors common/paths.py's `final_dir`
    -- keep both in sync if that layout ever changes.
    """
    return str(output_path.relative_to(data_root / "public_datasets" / "lerobot_v3_0"))


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the process_scripts cleaning pipeline for one dataset.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    registry_common = _load_registry_common()
    data_root = registry_common.paths.resolve_data_root(args.data_root)
    scripts_root = Path(__file__).resolve().parents[1]
    embodied_root = Path(__file__).resolve().parents[2]
    registry_path = embodied_root / "datasets_registry.yaml"
    dataset_config_path = scripts_root / "convert_scripts" / "configs" / f"{args.dataset_id}.yaml"
    process_config_path = Path(__file__).resolve().parent / "configs" / f"{args.dataset_id}.yaml"

    entries = registry_common.io.load_registry(registry_path)
    entry = next((e for e in entries if e.id == args.dataset_id), None)
    if entry is None:
        print(f"error: {args.dataset_id!r} not found in {registry_path}", file=sys.stderr)
        return 1
    if entry.convert_status != registry_common.schema.ConvertStatus.CONVERTED:
        print(
            f"error: {args.dataset_id!r} has convert_status={entry.convert_status!r}, "
            "expected 'converted' -- run convert_scripts first",
            file=sys.stderr,
        )
        return 1

    dataset_config = registry_common.io.load_dataset_config(dataset_config_path)
    staging_path = registry_common.paths.staging_dir(data_root, args.dataset_id)
    output_path = registry_common.paths.final_dir(data_root, args.dataset_id)

    stats = run_dataset(args.dataset_id, staging_path, output_path, process_config_path, dataset_config=dataset_config)

    if stats["output_episodes"] == 0:
        # run_dataset() never called write_lerobot_episodes (it early-returns
        # on an empty episode list), so output_path was never created on
        # disk -- there is nothing to write a README into, and nothing new
        # was actually persisted. Skip the README write (rather than
        # mkdir-ing an empty directory just to satisfy it) and mark the
        # registry FAILED instead of PROCESSED, since "processed" would
        # misrepresent a run that produced zero usable output. Update the
        # registry AFTER this decision (not before), so a save only ever
        # reflects a state that matches what's on disk.
        print(
            f"warning: {args.dataset_id!r} produced 0 output episodes ({stats['input_episodes']} input) -- "
            f"nothing written to {output_path}, skipping README generation",
            file=sys.stderr,
        )
        entry.process_status = registry_common.schema.ProcessStatus.FAILED
        entry.num_episodes = 0
        entry.num_frames = 0
        entry.duration_hours = 0.0
        registry_common.io.save_registry(entries, registry_path)
        print(f"processed {args.dataset_id}: {stats['input_episodes']} -> 0 episodes (FAILED)")
        return 1

    # Dataset output (and now the README) is fully written to disk before we
    # touch the registry at all -- so if the README write below were to
    # raise, save_registry() above/below never runs and the registry keeps
    # its pre-run state rather than being marked PROCESSED for a run that
    # didn't actually finish.
    config = load_process_config(process_config_path)
    _apply_gate_fields(config, dataset_config)
    readme_content = generate_dataset_readme(dataset_config, config, stats)
    (output_path / "README.md").write_text(readme_content, encoding="utf-8")

    entry.process_status = registry_common.schema.ProcessStatus.PROCESSED
    entry.num_episodes = stats["output_episodes"]
    entry.num_frames = stats["output_frames"]
    entry.final_local_path = compute_final_local_path(output_path, data_root)
    entry.duration_hours = stats["output_frames"] / stats["fps"] / 3600.0
    entry.storage_size_gb = round(_dir_size_bytes(output_path) / 1e9, 2)
    registry_common.io.save_registry(entries, registry_path)

    print(f"processed {args.dataset_id}: {stats['input_episodes']} -> {stats['output_episodes']} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
