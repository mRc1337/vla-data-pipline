"""Renders inspect_tool's synced video+chart playback dashboard for one
episode as a single st.components.v1.html iframe. Play/pause/scrub/speed
are pure client-side JS against the payload build_payload() already
assembled -- none of them trigger a Streamlit rerun, which is what makes
playback smooth (see docs/superpowers/specs/2026-08-06-inspect-tool-sync-
player-design.md section 4.2). build_html() is split out from render() so
the generated HTML string is testable without a live Streamlit session.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit.components.v1 as components

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_PLOTLY_JS = (_STATIC_DIR / "plotly.min.js").read_text(encoding="utf-8")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 8px; }
  .toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .toolbar button { font-size: 16px; padding: 4px 10px; }
  .toolbar input[type=range] { flex: 1; }
  .module-title { font-weight: 600; margin: 16px 0 6px; }
  .view-row { display: flex; gap: 12px; margin-bottom: 10px; align-items: flex-start; }
  .view-row .side { flex: 1; }
  .side-label { font-size: 12px; color: #666; margin-bottom: 2px; }
  video { width: 100%; max-width: 360px; background: #000; }
  .video-missing {
    width: 100%; max-width: 360px; height: 200px; background: #eee;
    display: flex; align-items: center; justify-content: center;
    color: #888; font-size: 13px;
  }
  .chart-row { margin-bottom: 4px; }
  .chart-label { font-size: 13px; color: #333; }
</style>
</head>
<body>
<div class="toolbar">
  <button id="play-btn">&#9654;</button>
  <input id="scrub" type="range" min="0" max="1000" value="0">
  <select id="speed-select">
    <option value="0.5">0.5x</option>
    <option value="1" selected>1x</option>
    <option value="2">2x</option>
  </select>
  <span id="time-label">0.00 / 0.00s</span>
</div>
<div id="image-module"></div>
<div id="state-module"></div>
<div id="action-module"></div>
<script>__PLOTLY_JS__</script>
<script>
const DATA = __PAYLOAD_JSON__;
(function () {
  const elementsById = {};

  // Collect video element references from pre-generated HTML
  for (const view of DATA.views) {
    const clips = DATA.videos[view] || {};
    if (clips.raw) {
      const video = document.getElementById(clips.raw.element_id);
      if (video) {
        video.src = clips.raw.url;
        elementsById[clips.raw.element_id] = video;
      }
    }
    if (clips.final) {
      const video = document.getElementById(clips.final.element_id);
      if (video) {
        video.src = clips.final.url;
        elementsById[clips.final.element_id] = video;
      }
    }
  }

  // Initialize Plotly charts for state and action modules
  for (const groupKey of ["state", "action"]) {
    const bands = DATA.charts[groupKey] || [];
    for (const band of bands) {
      const container = document.getElementById(band.div_id);
      if (container) {
        Plotly.newPlot(container, band.figure.data, band.figure.layout, {displayModeBar: false});
        Plotly.relayout(container, {
          shapes: [
            {type: "line", xref: "x", yref: "paper", y0: 0, y1: 1, x0: 0, x1: 0,
             line: {color: "#000", width: 1, dash: "dot"}},
            {type: "line", xref: "x", yref: "paper", y0: 0, y1: 1, x0: 0, x1: 0,
             line: {color: "#000", width: 2}, opacity: DATA.final_frame_count > 0 ? 0.8 : 0},
          ],
        });
      }
    }
  }

  const masterDuration = DATA.fps > 0 ? DATA.raw_frame_count / DATA.fps : 0;
  const playBtn = document.getElementById("play-btn");
  const scrub = document.getElementById("scrub");
  const speedSelect = document.getElementById("speed-select");
  const timeLabel = document.getElementById("time-label");

  let playing = false;
  let currentT = 0;
  let speed = 1;
  let lastFrameTime = null;

  function frameIndex(numFrames, t) {
    if (!numFrames || numFrames <= 0) return 0;
    const idx = Math.floor(t * DATA.fps);
    return Math.max(0, Math.min(numFrames - 1, idx));
  }

  function setVideoTime(clip) {
    if (!clip) return;
    const media = elementsById[clip.element_id];
    if (!media) return;
    const duration = clip.to_timestamp - clip.from_timestamp;
    const local = Math.min(currentT, duration);
    const target = clip.from_timestamp + local;
    if (Math.abs(media.currentTime - target) > 0.03) {
      media.currentTime = target;
    }
    media.playbackRate = speed;
    if (playing && media.paused) {
      media.play().catch(() => {});
    } else if (!playing && !media.paused) {
      media.pause();
    }
  }

  function applyTime() {
    for (const view of DATA.views) {
      const clips = DATA.videos[view] || {};
      setVideoTime(clips.raw);
      setVideoTime(clips.final);
    }
    const rawIdx = frameIndex(DATA.raw_frame_count, currentT);
    const finalIdx = frameIndex(DATA.final_frame_count, currentT);
    for (const groupKey of ["state", "action"]) {
      for (const band of DATA.charts[groupKey]) {
        Plotly.relayout(band.div_id, {
          "shapes[0].x0": rawIdx, "shapes[0].x1": rawIdx,
          "shapes[1].x0": finalIdx, "shapes[1].x1": finalIdx,
        });
      }
    }
    scrub.value = masterDuration > 0 ? Math.round((currentT / masterDuration) * 1000) : 0;
    timeLabel.textContent = currentT.toFixed(2) + " / " + masterDuration.toFixed(2) + "s";
  }

  function tick(now) {
    if (!playing) return;
    if (lastFrameTime !== null) {
      currentT = Math.min(masterDuration, currentT + ((now - lastFrameTime) / 1000) * speed);
    }
    lastFrameTime = now;
    applyTime();
    if (currentT >= masterDuration) {
      playing = false;
      playBtn.innerHTML = "&#9654;";
      applyTime();
      return;
    }
    requestAnimationFrame(tick);
  }

  playBtn.addEventListener("click", () => {
    if (playing) {
      playing = false;
      playBtn.innerHTML = "&#9654;";
      applyTime();
      return;
    }
    if (currentT >= masterDuration) currentT = 0;
    playing = true;
    lastFrameTime = null;
    playBtn.innerHTML = "&#10074;&#10074;";
    requestAnimationFrame(tick);
  });

  scrub.addEventListener("input", () => {
    playing = false;
    playBtn.innerHTML = "&#9654;";
    currentT = (parseFloat(scrub.value) / 1000) * masterDuration;
    applyTime();
  });

  speedSelect.addEventListener("change", () => {
    speed = parseFloat(speedSelect.value);
  });

  applyTime();
})();
</script>
</body>
</html>
"""


def _render_video_side_html(view: str, side: str, clip: dict | None) -> str:
    """Renders one raw/final <video> (or missing-placeholder) side of a
    view-row. Shared by both the raw and final columns of the image module."""
    lines = ['<div class="side">']
    lines.append(f'<div class="side-label">{view} ({side})</div>')
    if clip:
        lines.append(f'<video id="{clip["element_id"]}" muted playsinline></video>')
    else:
        lines.append(f'<div class="video-missing">{side} not available</div>')
    lines.append("</div>")
    return "\n".join(lines)


def _render_chart_module_html(bands: list, title: str) -> str:
    """Renders a chart module (title + one chart-row per band). Shared by the
    state and action modules, which differ only in title and payload key."""
    if not bands:
        return ""
    lines = [f'<div class="module-title">{title}</div>']
    for band in bands:
        label_text = band["label"]
        if not band.get("populated", False):
            label_text += " (unpopulated)"
        lines.append('<div class="chart-row">')
        lines.append(f'<div class="chart-label">{label_text}</div>')
        lines.append(f'<div id="{band["div_id"]}" style="height: 220px;"></div>')
        lines.append("</div>")
    return "\n".join(lines)


def build_html(payload: dict) -> str:
    """Pure string-building step, split out from render() so it's testable
    without a live Streamlit session/ScriptRunContext. Generates the video and
    chart HTML structure from the payload."""
    html = _HTML_TEMPLATE.replace("__PLOTLY_JS__", _PLOTLY_JS).replace(
        "__PAYLOAD_JSON__", json.dumps(payload)
    )

    # Pre-generate image module HTML
    image_html_lines = []
    if payload.get("views"):
        image_html_lines.append('<div class="module-title">Image</div>')
        for view in payload["views"]:
            clips = payload.get("videos", {}).get(view, {})
            raw_clip = clips.get("raw")
            final_clip = clips.get("final")

            image_html_lines.append('<div class="view-row">')
            image_html_lines.append(_render_video_side_html(view, "raw", raw_clip))
            image_html_lines.append(_render_video_side_html(view, "final", final_clip))
            image_html_lines.append("</div>")

    image_module_html = "\n".join(image_html_lines)

    # Pre-generate chart module HTML
    state_module_html = _render_chart_module_html(
        payload.get("charts", {}).get("state", []), "State"
    )
    action_module_html = _render_chart_module_html(
        payload.get("charts", {}).get("action", []), "Action"
    )

    # Replace module divs with generated content
    html = html.replace('<div id="image-module"></div>', image_module_html)
    html = html.replace('<div id="state-module"></div>', state_module_html)
    html = html.replace('<div id="action-module"></div>', action_module_html)

    return html


def render(payload: dict, height: int = 1400) -> None:
    components.html(build_html(payload), height=height, scrolling=True)
