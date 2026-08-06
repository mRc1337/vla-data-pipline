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
import threading
from pathlib import Path
from typing import Optional

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

# How often the sidebar selector and main body fragments poll
# st.session_state.pipeline_state for progress. Polling never stops once the
# pipeline finishes -- the redraw is cheap (same "done" snapshot every tick)
# and this is a single-user local dev tool, so a stop/start fragment split
# isn't worth the extra code.
_POLL_INTERVAL = "2s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect process_scripts cleaning/alignment before vs after.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(sys.argv[1:])


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


def _ensure_pipeline_started(input_path: str, output_path: str, config_path: str, raw_episodes: list) -> None:
    """Idempotently makes sure exactly one pipeline run is in flight (or
    already finished) for this (input, output, config) triple. Tracked in
    st.session_state.pipeline_key/pipeline_state so repeated Streamlit
    script reruns within the same browser session never start a second
    thread. Fast path: if output_path already has `_inspect_metadata.json`
    on disk, there's nothing to run -- load it directly and mark state
    "done" immediately, same as the old blocking `_load_or_run` did.

    `raw_episodes` (already decoded by the caller via `_load_raw_episodes`)
    is handed straight to run_dataset_instrumented instead of letting it
    decode input_path again itself -- decoding the same dataset from two
    threads at once (this background thread and app.py's main thread) races
    on a class-level attribute inside HF `datasets`' thread_map/ensure_lock
    and intermittently raises `AttributeError: type object 'tqdm' has no
    attribute '_lock'`.

    Caveat this doesn't handle: refreshing the browser mid-run starts a
    brand-new Streamlit session (session_state is gone), and if
    `_inspect_metadata.json` hasn't been written yet this will start a
    second background thread writing the same output directory concurrently
    with the still-running first one. Don't refresh the page while a run is
    in progress; not solved here (would need a cross-session file lock)."""
    key = (input_path, output_path, config_path)
    output = Path(output_path)

    if output.exists() and metadata_exists(output):
        if st.session_state.get("pipeline_key") != key:
            metadata = load_metadata(output)
            st.session_state.pipeline_key = key
            st.session_state.pipeline_lock = threading.Lock()
            st.session_state.pipeline_state = {
                "status": "done",
                "episode_records": {ep["episode_index"]: ep for ep in metadata["episodes"]},
                "error": None,
                "metadata": metadata,
            }
        return

    if st.session_state.get("pipeline_key") == key and "pipeline_state" in st.session_state:
        return  # already started (running/done/error) for these args

    lock = threading.Lock()
    state = {"status": "running", "episode_records": {}, "error": None, "metadata": None}
    st.session_state.pipeline_key = key
    st.session_state.pipeline_lock = lock
    st.session_state.pipeline_state = state

    def on_episode_done(episode_index: int, record: dict) -> None:
        # `state` is a plain dict, not st.session_state itself -- only this
        # one dict/lock pair is ever touched from the background thread, so
        # there's no cross-thread write to Streamlit's session_state proxy.
        with lock:
            state["episode_records"][episode_index] = record

    def target() -> None:
        try:
            metadata = run_dataset_instrumented(
                Path(input_path), output, Path(config_path),
                on_episode_done=on_episode_done, episodes=raw_episodes,
            )
            with lock:
                state["status"] = "done"
                state["metadata"] = metadata
        except Exception as exc:  # surfaced in the UI, not just server logs
            with lock:
                state["status"] = "error"
                state["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=target, daemon=True)
    st.session_state.pipeline_thread = thread
    thread.start()


def _snapshot_pipeline_state() -> dict:
    """Reads the live st.session_state.pipeline_state under its lock. Must
    be called fresh from inside whichever function needs current progress
    (never passed in as a parameter to a fragment) -- a `run_every`
    fragment's periodic reruns re-invoke the same function with whatever
    arguments were bound on its last real call, so only values fetched from
    session_state inside the function body are guaranteed current."""
    state = st.session_state.pipeline_state
    lock = st.session_state.pipeline_lock
    with lock:
        return {
            "status": state["status"],
            "episode_records": dict(state["episode_records"]),
            "error": state["error"],
            "metadata": state["metadata"],
        }


def _render_overview(metadata: dict, total_count: Optional[int] = None) -> None:
    episodes = metadata["episodes"]
    kept = sum(1 for ep in episodes if ep["survived"])
    st.header("Dataset overview")
    if total_count is not None and len(episodes) < total_count:
        st.caption(f"Partial results: {len(episodes)}/{total_count} episodes processed so far")
    col1, col2 = st.columns(2)
    col1.metric("Input episodes", len(episodes) if total_count is None else total_count)
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


def _render_episode_detail(episode_record: Optional[dict], raw_episode, final_episode, config, masks: dict) -> None:
    st.header(f"Episode {raw_episode.episode_index}")
    if episode_record is None:
        st.write("Status: **pending** (not processed yet)")
    else:
        st.write(f"Status: **{episode_status_label(episode_record)}**")
        st.write(f"Frames: {episode_record['input_frame_count']} -> {episode_record['output_frame_count']}")

    st.subheader("Language instruction")
    st.write(raw_episode.language_instruction or "(none)")
    if episode_record is not None:
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

    if episode_record is not None:
        st.subheader("Per-stage record")
        st.json(episode_record["stages"])


@st.fragment(run_every=_POLL_INTERVAL)
def _render_sidebar(episode_indices: list) -> None:
    records_snapshot = _snapshot_pipeline_state()["episode_records"]

    def _label(idx: int) -> str:
        record = records_snapshot.get(idx)
        return f"{idx}: pending..." if record is None else f"{idx}: {episode_status_label(record)}"

    st.selectbox("Episode", episode_indices, format_func=_label, key="episode_select")


@st.fragment(run_every=_POLL_INTERVAL)
def _render_body(input_path: str, output_path: str, raw_episodes: dict, config) -> None:
    snapshot = _snapshot_pipeline_state()
    status = snapshot["status"]
    total_count = len(raw_episodes)
    done_count = len(snapshot["episode_records"])

    if status == "running":
        st.progress(done_count / max(total_count, 1), text=f"Running pipeline: {done_count}/{total_count} episodes processed")
    elif status == "error":
        st.error(f"Pipeline failed: {snapshot['error']}")
        return
    else:
        st.success(f"Pipeline finished: {done_count}/{total_count} episodes processed")

    if status == "done":
        _render_overview(snapshot["metadata"])
    else:
        _render_overview({"episodes": list(snapshot["episode_records"].values())}, total_count=total_count)

    selected = st.session_state.get("episode_select")
    if selected is None or selected not in raw_episodes:
        return

    record = snapshot["episode_records"].get(selected)
    if status == "done":
        final_by_index = {ep.episode_index: ep for ep in _load_final_episodes(output_path)}
        final_episode = final_by_index.get(selected)
        masks = _load_canonical_masks(output_path)
    else:
        final_episode = None
        masks = {"state": None, "action": None}

    if record is None:
        st.info(f"Episode {selected} is still being processed -- showing raw data only.")

    _render_episode_detail(record, raw_episodes[selected], final_episode, config, masks)


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="process_scripts inspector", layout="wide")
    st.title("process_scripts cleaning/alignment inspector")

    raw_episodes = {ep.episode_index: ep for ep in _load_raw_episodes(args.input)}
    config = load_process_config(Path(args.config))
    episode_indices = sorted(raw_episodes.keys())

    _ensure_pipeline_started(args.input, args.output, args.config, list(raw_episodes.values()))

    with st.sidebar:
        _render_sidebar(episode_indices)

    _render_body(args.input, args.output, raw_episodes, config)


if __name__ == "__main__":
    main()
