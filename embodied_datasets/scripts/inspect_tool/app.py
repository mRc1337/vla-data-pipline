"""Streamlit UI for inspecting process_scripts' before/after cleaning
results. Standalone tool -- imports individual process_scripts modules
directly (via instrumented_pipeline.py), never run_pipeline.py itself.

Run with:
    streamlit run embodied_datasets/scripts/inspect_tool/app.py -- \
        --input /path/to/raw --output /path/to/output --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

_PROCESS_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "process_scripts"
if str(_PROCESS_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESS_SCRIPTS_DIR))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from common.io import load_process_config  # noqa: E402
from lerobot_io import load_lerobot_episodes  # noqa: E402

from instrumented_pipeline import run_dataset_instrumented  # noqa: E402
from metadata_io import load_metadata, metadata_exists  # noqa: E402
from views import action_bands, band_overlay_figure, episode_status_label, slice_band, state_bands  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect process_scripts cleaning/alignment before vs after.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(sys.argv[1:])


@st.cache_data(show_spinner="Running pipeline (first time only for this output path)...")
def _load_or_run(input_path: str, output_path: str, config_path: str) -> dict:
    output = Path(output_path)
    if output.exists() and metadata_exists(output):
        return load_metadata(output)
    return run_dataset_instrumented(Path(input_path), output, Path(config_path))


@st.cache_data(show_spinner="Loading raw dataset...")
def _load_raw_episodes(input_path: str):
    return load_lerobot_episodes(Path(input_path))


@st.cache_data(show_spinner="Loading cleaned dataset...")
def _load_final_episodes(output_path: str):
    if not Path(output_path).exists():
        return []
    return load_lerobot_episodes(Path(output_path))


@st.cache_data(show_spinner="Loading canonical masks...")
def _load_canonical_masks(output_path: str) -> dict:
    """Episode objects from load_lerobot_episodes never carry
    observation.state_canonical_mask/action_canonical_mask -- those are
    plain lerobot features write_lerobot_episodes adds, not fields
    load_lerobot_episodes reads back. Both masks are dataset-constant (the
    same value on every frame), so reading frame 0 is enough. Reads the
    lerobot dataset directly rather than extending load_lerobot_episodes
    itself, to keep this glue code local to inspect_tool."""
    output = Path(output_path)
    if not output.exists():
        return {"state": None, "action": None}
    dataset = LeRobotDataset(repo_id=output.name, root=output)
    if len(dataset) == 0:
        return {"state": None, "action": None}
    row = dataset[0]
    masks = {}
    for key, feature_key in (("state", "observation.state_canonical_mask"), ("action", "action_canonical_mask")):
        masks[key] = row[feature_key].numpy().astype(bool) if feature_key in row else None
    return masks


def _render_overview(metadata: dict) -> None:
    episodes = metadata["episodes"]
    kept = sum(1 for ep in episodes if ep["survived"])
    st.header("Dataset overview")
    col1, col2 = st.columns(2)
    col1.metric("Input episodes", len(episodes))
    col2.metric("Output episodes", kept)

    stage_counts: dict = {}
    for ep in episodes:
        for stage in ep["stages"]:
            if stage["rejected"] or stage["dropped_frame_indices"]:
                key = stage["stage"]
                stage_counts[key] = stage_counts.get(key, 0) + 1
    if stage_counts:
        st.subheader("Episodes affected per stage")
        st.table({"stage": list(stage_counts.keys()), "episodes_affected": list(stage_counts.values())})


def _render_episode_detail(episode_record: dict, raw_episode, final_episode, config, masks: dict) -> None:
    st.header(f"Episode {episode_record['episode_index']}")
    st.write(f"Status: **{episode_status_label(episode_record)}**")
    st.write(f"Frames: {episode_record['input_frame_count']} -> {episode_record['output_frame_count']}")

    st.subheader("Language instruction")
    st.write(raw_episode.language_instruction or "(none)")
    check1 = next((s for s in episode_record["stages"] if s["stage"] == "check1_instruction_consistency"), None)
    if check1 is not None:
        st.write(f"check1 verdict (skip_reason): {check1['skip_reason']}")

    st.subheader("Video frames (raw vs cleaned)")
    for view, raw_frames in raw_episode.frames.items():
        st.write(f"View: {view}")
        preview_count = min(5, raw_frames.shape[0])
        st.image(list(raw_frames[:preview_count]), caption=[f"raw #{i}" for i in range(preview_count)])
        if final_episode is not None and view in final_episode.frames:
            final_frames = final_episode.frames[view]
            preview_count = min(5, final_frames.shape[0])
            st.image(list(final_frames[:preview_count]), caption=[f"final #{i}" for i in range(preview_count)])

    st.subheader("State")
    for band in state_bands(config.num_arms):
        raw_slice = slice_band(raw_episode.state, None, band)
        final_slice = None
        if final_episode is not None:
            final_slice = slice_band(final_episode.state, masks["state"], band)
        label = band.label + (" (unpopulated)" if final_slice is not None and not final_slice["populated"] else "")
        st.caption(f"{label} (raw vs final)")
        st.plotly_chart(band_overlay_figure(band.label, raw_slice, final_slice), width="stretch", key=f"state_{band.label}")

    st.subheader("Action")
    for band in action_bands(config.num_arms):
        raw_slice = slice_band(raw_episode.action, None, band)
        final_slice = None
        if final_episode is not None:
            final_slice = slice_band(final_episode.action, masks["action"], band)
        label = band.label + (" (unpopulated)" if final_slice is not None and not final_slice["populated"] else "")
        st.caption(f"{label} (raw vs final)")
        st.plotly_chart(band_overlay_figure(band.label, raw_slice, final_slice), width="stretch", key=f"action_{band.label}")

    st.subheader("Per-stage record")
    st.json(episode_record["stages"])


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="process_scripts inspector", layout="wide")
    st.title("process_scripts cleaning/alignment inspector")

    metadata = _load_or_run(args.input, args.output, args.config)
    raw_episodes = {ep.episode_index: ep for ep in _load_raw_episodes(args.input)}
    final_episodes = {ep.episode_index: ep for ep in _load_final_episodes(args.output)}
    masks = _load_canonical_masks(args.output)
    config = load_process_config(Path(args.config))

    _render_overview(metadata)

    episode_indices = [ep["episode_index"] for ep in metadata["episodes"]]
    records_by_index = {ep["episode_index"]: ep for ep in metadata["episodes"]}
    selected = st.sidebar.selectbox(
        "Episode",
        episode_indices,
        format_func=lambda idx: f"{idx}: {episode_status_label(records_by_index[idx])}",
    )
    if selected is not None and selected in raw_episodes:
        _render_episode_detail(
            records_by_index[selected],
            raw_episodes[selected],
            final_episodes.get(selected),
            config,
            masks,
        )



if __name__ == "__main__":
    main()
