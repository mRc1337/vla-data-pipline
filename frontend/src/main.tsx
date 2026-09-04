import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  Alert, Button, Card, Collapse, Descriptions, Empty, Input, InputNumber, Layout,
  List, Pagination, Progress, Row, Col, Segmented, Select, Space, Spin, Statistic, Table, Tag, message,
} from "antd";
import "./style.css";

type Dataset = {
  uid: string; episodes: number; frames: number; duration: number; bytes: number;
  cameras: string[]; codebase_version: string; root?: string; schema?: Record<string, unknown>;
};
type DatasetCollection = {
  id: string; name: string; datasets: Dataset[]; episodes: number; frames: number; duration: number; bytes: number;
};
type Episode = {
  dataset_uid: string; episode_index: number; frames: number; duration: number;
  instruction?: string | null; task_index?: number | null; metadata_json?: string;
};
type DatasetTask = { task_index: number; name: string; episodes: number };
type VideoRef = {
  camera: string; relative_path: string; url: string;
  timestamp_start?: number; timestamp_end?: number;
  source_start?: number; source_end?: number; duration?: number;
  chunk_index?: number; file_index?: number; integrity_status?: string;
  width?: number | null; height?: number | null; fps?: number | null;
  codec?: string | null; pixel_format?: string | null; channels?: number | null;
  has_audio?: boolean | null; physical_frames?: number | null; frame_count?: number | null;
  source_bytes?: number | null; file_integrity_status?: string;
  is_proxy?: boolean;
};
type ProxyVideo = { camera: string; status: "pending" | "ready"; url: string; duration: number; frame_count: number; fps: number };
type ProxyJob = {
  job_id: string; dataset_uid: string; episode_index: number;
  status: "queued" | "generating" | "ready" | "failed";
  error?: string | null; videos: ProxyVideo[];
};
type StageResult = {
  stage_id: number; run_id: string; stage?: string; detector_version?: string;
  coordinate_system?: string;
  verdict?: string; anomaly_count?: number; severity?: string | null; score?: number | null;
  reason_codes?: string[]; detail_loaded?: boolean;
  summary?: Record<string, unknown>; records?: Array<Record<string, unknown>>;
  detail?: Record<string, unknown> | null;
  artifact_status?: "available" | "episode_pending" | "upstream_filtered" | "not_generated";
  placeholder?: string | null;
  visualization_spec?: {
    name?: string; description?: string; visualization?: string; expected_fields?: string[];
  } | null;
};
type EpisodeTimeline = {
  coordinate_system: string; frame_count: number; fps: number; duration: number; dataset_from_index: number;
};
type Preview = {
  dataset: Dataset; episode: Episode; timeline?: EpisodeTimeline; videos: VideoRef[]; stage_results: StageResult[];
  curation?: { format?: string | null; has_repairs?: boolean; has_validity?: boolean };
};
type SeriesRow = Record<string, unknown>;
type SeriesView = "raw" | "valid" | "repaired" | "diff";
type ScanTask = { scan_id: string; status: string; current: number; total: number; datasets: number; skipped: number; eta_seconds: number | null };
type PlaybackStatus = "idle" | "seeking" | "buffering" | "stalled" | "playing" | "paused" | "error";
type BrowserVideoMetadata = { width: number; height: number; duration: number };
type SearchStageBadge = {
  stage_id: number; artifact_status: string; verdict: string; anomaly_count: number;
  range_count: number; severity?: string | null; score?: number | null; reason_codes: string[];
};
type EpisodeSearchItem = {
  dataset_uid: string; collection_name: string; task_index?: number | null; task_name?: string | null;
  episode_index: number; instruction?: string | null; frame_count: number; duration: number;
  camera_count: number; primary_camera?: string | null; title: string; thumbnail_url: string;
  thumbnail_status: string; thumbnail_error?: string; match_reasons: string[]; stage_badges: SearchStageBadge[];
};
type EpisodeSearchResponse = {
  query: string; page: number; page_size: number; total: number;
  dataset_count: number; task_count: number; items: EpisodeSearchItem[];
};
type SearchIndexTask = {
  task_id: string; status: string; current: number; total: number; phase?: string;
  dataset_uid?: string; error?: string | null; result?: { datasets: number; episodes: number };
};
type SearchStageFacets = Record<string, {
  verdicts: Array<{ value: string; count: number }>;
  artifact_statuses: Array<{ value: string; count: number }>;
}>;

const playbackStatusLabels: Record<PlaybackStatus, string> = {
  idle: "待播放", seeking: "正在定位", buffering: "正在缓冲",
  stalled: "读取停滞", playing: "播放中", paused: "已暂停", error: "播放失败",
};

const stageFilterOptions: Record<number, Array<{ label: string; value: string }>> = {
  1: [
    { label: "通过", value: "pass" }, { label: "有突变", value: "anomaly" },
    { label: "Episode 已过滤", value: "filtered" },
  ],
  2: [
    { label: "通过", value: "pass" }, { label: "趋势不一致", value: "fail" },
    { label: "无法评分", value: "unscored" },
  ],
  3: [
    { label: "通过", value: "pass" }, { label: "有极值", value: "anomaly" },
    { label: "Episode 已过滤", value: "filtered" },
  ],
  4: [
    { label: "通过", value: "pass" }, { label: "警告", value: "warning" },
    { label: "失败", value: "fail" },
  ],
  5: [
    { label: "已对齐", value: "aligned" }, { label: "非训练候选", value: "not_candidate" },
  ],
  6: [
    { label: "已完成", value: "complete" }, { label: "需复核", value: "needs_review" },
  ],
  7: [
    { label: "通过", value: "pass" }, { label: "失败", value: "fail" },
    { label: "跳过", value: "skip" },
  ],
  8: [
    { label: "保留", value: "retain" },
    { label: "排除窗口", value: "exclude_affected_sample_windows" },
    { label: "排除 Episode", value: "exclude_episode_from_training" },
  ],
};

const artifactFilterOptions = [
  { label: "已有产物", value: "status:available" },
  { label: "该 Episode 待处理", value: "status:episode_pending" },
  { label: "前序阶段已过滤", value: "status:upstream_filtered" },
  { label: "未生成产物", value: "status:not_generated" },
];

function searchStatusLabel(value: string): string {
  const labels: Record<string, string> = {
    pass: "通过", anomaly: "有异常", filtered: "Episode 已过滤", fail: "失败",
    unscored: "无法评分", warning: "警告", aligned: "已对齐", not_candidate: "非训练候选",
    complete: "已完成", needs_review: "需复核", insufficient_confidence: "低置信度",
    skip: "跳过", retain: "保留", exclude_affected_sample_windows: "排除窗口",
    exclude_episode_from_training: "排除 Episode", available: "已有产物",
    episode_pending: "该 Episode 待处理", upstream_filtered: "前序阶段已过滤",
    not_generated: "未生成产物",
  };
  return labels[value] || value;
}

function stageBadgePresentation(stage: SearchStageBadge): { text: string; color?: string } {
  const id = stage.stage_id;
  if (stage.artifact_status === "not_generated") return { text: `S${id} 未生成` };
  if (stage.artifact_status === "episode_pending") return { text: `S${id} 待处理`, color: "gold" };
  if (stage.artifact_status === "upstream_filtered") return { text: `S${id} 前序过滤`, color: "orange" };
  if (id === 1 && stage.verdict === "anomaly") return { text: `S1 突变 ${stage.anomaly_count}帧`, color: "orange" };
  if (id === 3 && stage.verdict === "anomaly") return { text: `S3 极值 ${stage.range_count || stage.anomaly_count}区间`, color: "orange" };
  if (id === 6 && stage.severity === "warning") return { text: "S6 需复核", color: "gold" };
  if (id === 7 && stage.verdict !== "pass") return { text: `S7 ${searchStatusLabel(stage.verdict)}`, color: stage.verdict === "fail" ? "red" : "gold" };
  if (id === 8 && stage.verdict === "exclude_affected_sample_windows") return { text: "S8 排除窗口", color: "orange" };
  if (id === 8 && stage.verdict === "exclude_episode_from_training") return { text: "S8 排除 Episode", color: "red" };
  if (["fail", "filtered", "not_candidate"].includes(stage.verdict)) return { text: `S${id} ${searchStatusLabel(stage.verdict)}`, color: "red" };
  if (["warning", "unscored", "needs_review"].includes(stage.verdict)) return { text: `S${id} ${searchStatusLabel(stage.verdict)}`, color: "gold" };
  if (["pass", "complete", "aligned", "retain"].includes(stage.verdict)) {
    return { text: `S${id} ${searchStatusLabel(stage.verdict)}`, color: "green" };
  }
  return { text: `S${id} ${searchStatusLabel(stage.verdict)}`, color: "gold" };
}

function EpisodeSearchCard({ item, onOpen }: { item: EpisodeSearchItem; onOpen: () => void }) {
  const [imageFailed, setImageFailed] = useState(false);
  useEffect(() => setImageFailed(false), [item.thumbnail_url]);
  const showImage = item.thumbnail_status === "ready" && !imageFailed;
  return <Card hoverable className="episode-search-card" onClick={onOpen} cover={
    <div className="episode-cover">
      {showImage
        ? <img loading="lazy" decoding="async" src={item.thumbnail_url} alt={`${item.title} 主视角首帧`} onError={() => setImageFailed(true)} />
        : <div className="episode-cover-placeholder">
          <span>{item.collection_name.slice(0, 2).toUpperCase()}</span>
          <small>{item.thumbnail_status === "unavailable" ? "无可用主视角" : item.thumbnail_status === "failed" ? "封面生成失败" : "封面生成中"}</small>
        </div>}
      <span className="episode-duration">{item.duration.toFixed(1)}s</span>
    </div>
  }>
    <Card.Meta title={<span title={item.title}>{item.title}</span>} description={
      <>
        <div className="episode-card-line">
          Episode {String(item.episode_index).padStart(4, "0")} · {item.frame_count} 帧 · {item.camera_count} 相机
        </div>
        {item.dataset_uid !== item.collection_name && <div className="episode-card-line" title={item.dataset_uid}>子数据集：{item.dataset_uid}</div>}
        {item.task_name && <div className="episode-card-line" title={item.task_name}>Task：{item.task_name}</div>}
        {item.match_reasons.length > 0 && <div className="episode-card-tags"><Tag color="blue">命中 {item.match_reasons.join(" / ")}</Tag></div>}
        <div className="episode-card-tags">{item.stage_badges.map((stage) => {
          const badge = stageBadgePresentation(stage);
          return <Tag key={stage.stage_id} color={badge.color}>{badge.text}</Tag>;
        })}</div>
      </>
    } />
  </Card>;
}

// The catalog indexes physical LeRobot roots. Public datasets commonly put
// one task in each root (for example arcap/open_bottle), so group those roots
// for display while retaining each member UID for API requests.
function collectionName(dataset: Dataset): string {
  const parts = (dataset.root || dataset.uid).split(/[\\/]+/).filter(Boolean);
  const versionIndex = parts.findIndex((part) => /^lerobot_v\d+_\d+$/i.test(part));
  if (versionIndex >= 0 && parts[versionIndex + 1]) return parts[versionIndex + 1];
  return dataset.uid;
}

function buildCollections(rows: Dataset[]): DatasetCollection[] {
  const grouped = new Map<string, Dataset[]>();
  rows.filter((dataset) => !dataset.root?.includes("/data_curation/stage"))
    .forEach((dataset) => {
      const name = collectionName(dataset);
      const members = grouped.get(name) || [];
      members.push(dataset);
      grouped.set(name, members);
    });
  return Array.from(grouped.entries()).map(([name, members]) => ({
    id: name,
    name,
    datasets: members.sort((left, right) => left.uid.localeCompare(right.uid)),
    episodes: members.reduce((sum, item) => sum + (item.episodes || 0), 0),
    frames: members.reduce((sum, item) => sum + (item.frames || 0), 0),
    duration: members.reduce((sum, item) => sum + (item.duration || 0), 0),
    bytes: members.reduce((sum, item) => sum + (item.bytes || 0), 0),
  })).sort((left, right) => left.name.localeCompare(right.name));
}

function numeric(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  if (Array.isArray(value)) return value.length ? numeric(value[0]) : null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

function vector(value: unknown): number[] {
  if (Array.isArray(value)) return value.flatMap(vector).filter(Number.isFinite);
  const result = numeric(value);
  return result === null ? [] : [result];
}

function featureElementNames(dataset: Dataset | undefined, field: string, width: number): string[] {
  const feature = dataset?.schema?.[field] as { names?: unknown } | undefined;
  const names = Array.isArray(feature?.names) ? feature.names : [];
  return Array.from({ length: width }, (_, index) => (
    typeof names[index] === "string" && names[index] ? names[index] : `${field}[${index}]`
  ));
}

type FrameInterval = { start: number; end: number };

function stageFrameIntervals(results: StageResult[], timeline: EpisodeTimeline): FrameInterval[] {
  const frameIntervals: Array<{ start: number; end: number }> = [];
  const addRecord = (stage: StageResult, record: Record<string, unknown>, endExclusive = false) => {
      const status = String(record.status || record.label || record.decision || "warning").toLowerCase();
      if (["pass", "ok", "clean", "valid"].includes(status)) return;
      if (record.valid === true && record.flagged_frame !== true) return;
      if (record.flagged_frame === false || record.failed === false && record.frame_index !== undefined) return;
      const coordinate = String(record.coordinate_system || stage.coordinate_system || "episode_frame");
      let start = numeric(record.frame_start ?? record.start_frame ?? record.frame_index);
      let end = numeric(record.frame_end ?? record.end_frame ?? record.frame_index ?? start);
      if (start === null) {
        const timestamp = numeric(record.timestamp_start ?? record.start_timestamp ?? record.timestamp);
        if (timestamp !== null) start = timestamp * timeline.fps;
      }
      if (end === null) {
        const timestamp = numeric(record.timestamp_end ?? record.end_timestamp ?? record.timestamp);
        if (timestamp !== null) end = timestamp * timeline.fps;
      }
      if (start === null || end === null) return;
      if (endExclusive && end > start) end -= 1;
      if (["dataset_frame", "global_frame"].includes(coordinate)) {
        start -= timeline.dataset_from_index;
        end -= timeline.dataset_from_index;
      }
      start = Math.min(Math.max(start, 0), Math.max(0, timeline.frame_count - 1));
      end = Math.min(Math.max(end, start), Math.max(0, timeline.frame_count - 1));
      frameIntervals.push({ start, end });
  };
  results.forEach((stage) => {
    (stage.records || []).forEach((record) => addRecord(stage, record));
    if (stage.stage_id === 7) {
      const frames = Array.isArray(stage.detail?.frames) ? stage.detail.frames : [];
      frames.forEach((frame) => {
        if (frame && typeof frame === "object" && String((frame as Record<string, unknown>).decision) === "fail") {
          addRecord(stage, frame as Record<string, unknown>);
        }
      });
    }
    if (stage.stage_id === 8) {
      const ranges = Array.isArray(stage.detail?.invalid_ranges) ? stage.detail.invalid_ranges : [];
      ranges.forEach((range) => {
        if (range && typeof range === "object") addRecord(stage, range as Record<string, unknown>, true);
      });
    }
  });
  frameIntervals.sort((left, right) => left.start - right.start || left.end - right.end);
  const merged: Array<{ start: number; end: number }> = [];
  frameIntervals.forEach((interval) => {
    const previous = merged[merged.length - 1];
    if (previous && interval.start <= previous.end + 1) {
      previous.end = Math.max(previous.end, interval.end);
    } else {
      merged.push({ ...interval });
    }
  });
  return merged;
}

function stageIntervals(results: StageResult[], timeline: EpisodeTimeline): Array<[{ xAxis: number }, { xAxis: number }]> {
  return stageFrameIntervals(results, timeline).map(({ start, end }) => [
    { xAxis: start / timeline.fps },
    { xAxis: Math.min(timeline.frame_count, end + 1) / timeline.fps },
  ]);
}

function stageAnomalyTypes(stage: StageResult): string[] {
  const labels = new Set<string>();
  const reasonLabels: Record<string, string> = {
    stage1_sudden_change: "突变",
    stage1_episode_rejected: "Stage 1 整条 Episode 过滤",
    stage1_invalid: "继承的 Stage 1 无效帧",
    state_action_trend_mismatch: "State-Action 趋势不一致",
    stage3_extreme_value: "极值",
  };
  (stage.records || []).forEach((record) => {
    const reasons = Array.isArray(record.reason_codes) ? record.reason_codes : [];
    reasons.forEach((reason) => labels.add(reasonLabels[String(reason)] || String(reason)));
    if (record.reason_code) labels.add(reasonLabels[String(record.reason_code)] || String(record.reason_code));
    const stateDims = Array.isArray(record.failed_state_dimensions) ? record.failed_state_dimensions : [];
    const actionDims = Array.isArray(record.failed_action_dimensions) ? record.failed_action_dimensions : [];
    if (record.flagged_frame === true) {
      const kind = stage.stage_id === 1 ? "突变" : stage.stage_id === 3 ? "极值" : "帧异常";
      if (stateDims.length) labels.add(`${kind} · state[${stateDims.join(", ")}]`);
      if (actionDims.length) labels.add(`${kind} · action[${actionDims.join(", ")}]`);
      if (!stateDims.length && !actionDims.length) labels.add(kind);
    }
    if (record.failed === true && stage.stage_id === 2) labels.add("State-Action 趋势不一致");
    if (record.accepted === false || record.reject_episode === true) {
      labels.add(stage.stage_id === 2 ? "State-Action 趋势不一致" : `Stage ${stage.stage_id} Episode 过滤`);
    }
  });
  return Array.from(labels);
}

function recordsFrom(stage: StageResult, fileName: string): Array<Record<string, unknown>> {
  return (stage.records || []).filter((record) => String(record.file || "").endsWith(`/${fileName}`));
}

function firstRecordFrom(stage: StageResult, fileName: string): Record<string, unknown> | undefined {
  return recordsFrom(stage, fileName)[0];
}

function artifactStatusTag(status: StageResult["artifact_status"]) {
  if (status === "available") return <Tag color="green">已有产物</Tag>;
  if (status === "episode_pending") return <Tag color="gold">该 Episode 待处理</Tag>;
  if (status === "upstream_filtered") return <Tag color="orange">前序阶段已过滤</Tag>;
  return <Tag>暂无产物</Tag>;
}

function resultTag(value: unknown) {
  const status = String(value ?? "unknown").toLowerCase();
  const text = typeof value === "boolean" ? (value ? "是" : "否") : searchStatusLabel(status);
  if (["pass", "passed", "complete", "available", "retain", "true"].includes(status)) {
    return <Tag color="green">{text}</Tag>;
  }
  if (["fail", "failed", "invalid", "exclude_episode_from_training", "false"].includes(status)) {
    return <Tag color="red">{text}</Tag>;
  }
  return <Tag color="gold">{text}</Tag>;
}

function fixed(value: unknown, digits = 4, suffix = ""): string {
  const number = numeric(value);
  return number === null ? "—" : `${number.toFixed(digits)}${suffix}`;
}

function formatBytes(value: unknown): string {
  const bytes = numeric(value);
  if (bytes === null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = bytes;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function vectorText(value: unknown, digits = 5): string {
  const values = vector(value);
  return values.length ? `[${values.map((item) => item.toFixed(digits)).join(", ")}]` : "—";
}

function videoWindow(video: VideoRef, episodeDuration: number): { start: number; end: number; duration: number } {
  const start = Math.max(0, numeric(video.source_start ?? video.timestamp_start) ?? 0);
  const metadataEnd = numeric(video.source_end ?? video.timestamp_end);
  const end = metadataEnd !== null && metadataEnd > start
    ? metadataEnd
    : start + Math.max(0, episodeDuration);
  return { start, end: Math.max(start, end), duration: Math.max(0, end - start) };
}

const MEDIA_WAIT_TIMEOUT_MS = 30_000;

function waitForMediaEvent(video: HTMLVideoElement, eventName: "loadedmetadata", timeout = MEDIA_WAIT_TIMEOUT_MS): Promise<void> {
  return new Promise((resolve, reject) => {
    let timer = 0;
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener(eventName, onReady);
      video.removeEventListener("error", onError);
    };
    const onReady = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error(`视频 ${video.dataset.camera || "unknown"} 加载失败`)); };
    video.addEventListener(eventName, onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
    timer = window.setTimeout(() => {
      cleanup();
      reject(new Error(`等待视频 ${video.dataset.camera || "unknown"} ${eventName} 超时`));
    }, timeout);
  });
}

async function seekMedia(video: HTMLVideoElement, target: number, tolerance = 0.1): Promise<void> {
  if (video.readyState < HTMLMediaElement.HAVE_METADATA) await waitForMediaEvent(video, "loadedmetadata");
  // With preload="metadata", some browsers do not emit `seeked` until a
  // play request starts fetching media bytes. Set all target positions first,
  // then let the following play() calls drive loading instead of deadlocking
  // while waiting for `seeked` here.
  if (Math.abs(video.currentTime - target) > tolerance) video.currentTime = target;
}

function withTimeout<T>(promise: Promise<T>, messageText: string, timeout = MEDIA_WAIT_TIMEOUT_MS): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(messageText)), timeout);
    promise.then(
      (value) => { window.clearTimeout(timer); resolve(value); },
      (error) => { window.clearTimeout(timer); reject(error); },
    );
  });
}

function mediaErrorText(video: HTMLVideoElement): string {
  const code = video.error?.code;
  const detail = video.error?.message;
  return `视频 ${video.dataset.camera || "unknown"} 播放失败${code ? `（错误码 ${code}）` : ""}${detail ? `：${detail}` : ""}`;
}

function Curve({ title, rows, field, elementNames, fps, intervals = [], playhead, onSeek }: {
  title: string; rows: SeriesRow[]; field: string; elementNames: string[]; fps: number;
  intervals?: Array<[{ xAxis: number }, { xAxis: number }]>;
  playhead?: number;
  onSeek?: (time: number) => void;
}) {
  const first = rows.find((row) => vector(row[field]).length);
  const width = first ? vector(first[field]).length : 0;
  const availableDimensions = useMemo(() => Array.from({ length: width }, (_, index) => index), [width]);
  const selectionKey = `${field}:${width}:${elementNames.join("|")}`;
  const [selectedDimensions, setSelectedDimensions] = useState<number[]>(availableDimensions);
  useEffect(() => setSelectedDimensions(availableDimensions), [availableDimensions, selectionKey]);
  const chartSeries = useMemo(() => selectedDimensions.map((dimension) => ({
    dimension,
    data: rows.map((row, index) => {
      const values = vector(row[field]);
      const timestamp = numeric(row.episode_time ?? row.timestamp) ?? index / fps;
      return [timestamp, values[dimension] ?? null];
    }),
  })), [field, fps, rows, selectedDimensions]);
  const option = useMemo(() => ({
    animation: false,
    tooltip: { trigger: "axis" },
    legend: { type: "scroll", top: 4 },
    grid: { left: 48, right: 18, top: 38, bottom: 32 },
    xAxis: { type: "value", name: "s", min: 0 },
    yAxis: { type: "value" },
    series: chartSeries.map(({ dimension, data }, seriesIndex) => ({
      name: elementNames[dimension] || `${field}[${dimension}]`, type: "line", showSymbol: false,
      data,
      markArea: seriesIndex === 0 && intervals.length ? {
        silent: true, itemStyle: { color: "rgba(245, 63, 63, .16)" }, data: intervals,
      } : undefined,
      markLine: seriesIndex === 0 && playhead !== undefined ? {
        silent: true, symbol: "none", lineStyle: { color: "#1677ff", width: 1.5 },
        label: { show: false }, data: [{ xAxis: playhead }],
      } : undefined,
    })),
  }), [chartSeries, elementNames, field, intervals, playhead]);
  if (!rows.length || !width) return <Card size="small" title={title}><Alert type="info" showIcon message="该字段暂无可绘制数据，请先运行 standard 扫描或检查 Parquet schema。" /></Card>;
  const onEvents = onSeek ? {
    click: (params: { value?: unknown }) => {
      const value = Array.isArray(params.value) ? Number(params.value[0]) : Number(params.value);
      if (Number.isFinite(value)) onSeek(value);
    },
  } : undefined;
  const options = availableDimensions.map((dimension) => ({
    value: dimension, label: elementNames[dimension] || `${field}[${dimension}]`,
  }));
  return <Card size="small" title={title} extra={<Select<number[]> mode="multiple" allowClear
    className="curve-element-select" maxTagCount="responsive" optionFilterProp="label"
    placeholder="选择要显示的元素" value={selectedDimensions} options={options}
    onChange={setSelectedDimensions} />}>
    {!selectedDimensions.length && <Alert type="info" showIcon message="请选择至少一个元素进行可视化" />}
    <ReactECharts option={option} onEvents={onEvents} style={{ height: 270, cursor: onSeek ? "crosshair" : undefined }} notMerge lazyUpdate />
  </Card>;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function StageVisualizations({ stages, timeline, onSeek, onLoadStage, loadingStages }: {
  stages: StageResult[]; timeline: EpisodeTimeline; onSeek: (time: number) => void;
  onLoadStage: (stageId: number) => void; loadingStages: Set<number>;
}) {
  const rows = Array.from({ length: 8 }, (_, offset) => {
    const stageId = offset + 1;
    const matches = stages.filter((item) => item.stage_id === stageId);
    const available = matches.find((item) => item.artifact_status === "available")
      || matches.find((item) => item.artifact_status === "upstream_filtered")
      || matches.find((item) => item.artifact_status === "episode_pending")
      || matches[0];
    if (stageId <= 3 && available) {
      return {
        ...available,
        records: matches.flatMap((item) => item.records || []),
      };
    }
    return available || {
      stage_id: stageId, run_id: "placeholder", artifact_status: "not_generated" as const,
      stage: `Stage ${stageId}`, placeholder: `未发现 Stage ${stageId} 产物`,
    };
  });
  useEffect(() => {
    rows.forEach((stage) => {
      if (stage.artifact_status === "available"
        && !stage.detail_loaded && !loadingStages.has(stage.stage_id)) {
        onLoadStage(stage.stage_id);
      }
    });
  }, [stages]);
  const items = rows.map((stage) => {
        const spec = stage.visualization_spec || {};
        const label = <Space wrap>
          <Tag color="blue">Stage {stage.stage_id}</Tag>
          <span>{spec.name || stage.stage}</span>
          {artifactStatusTag(stage.artifact_status)}
        </Space>;
        if (stage.artifact_status === "available" && !stage.detail_loaded) {
          const children = <Spin spinning={loadingStages.has(stage.stage_id)}>
            <Alert type="info" showIcon
              message={loadingStages.has(stage.stage_id) ? "正在读取 Stage 详情" : "展开后按需读取 Stage 详情"}
              description={`摘要已从本地索引加载：${searchStatusLabel(String(stage.verdict || "available"))}${stage.anomaly_count ? ` · ${stage.anomaly_count} 个异常` : ""}`} />
          </Spin>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id <= 3) {
          if (stage.artifact_status !== "available") {
            const children = <>
              <Alert type={stage.artifact_status === "not_generated" ? "warning" : "info"} showIcon
                message={stage.placeholder || "该 Episode 尚无可读取产物"}
                description="当前状态来自实际 Stage 目录与 Manifest，不将缺失产物解释为检测通过。" />
              <Descriptions size="small" column={1} className="stage-details">
                <Descriptions.Item label="检测目标">{spec.description || "等待阶段产物"}</Descriptions.Item>
                <Descriptions.Item label="产物生成后展示">{spec.visualization || "检测状态、异常位置与审计证据"}</Descriptions.Item>
                <Descriptions.Item label="预期字段"><Space size={[4, 4]} wrap>{(spec.expected_fields || []).map((field) => <Tag key={field}>{field}</Tag>)}</Space></Descriptions.Item>
              </Descriptions>
            </>;
            return { key: String(stage.stage_id), label, children };
          }
          const frameFlagRecords = recordsFrom(stage, "frame_flags.parquet");
          const validityRecords = recordsFrom(stage, "step_validity.parquet").filter((record) => (
            record[`stage${stage.stage_id}_valid`] === false
          ));
          const annotationRecords = (stage.records || []).filter((record) => !record.file);
          const anomalyStage = { ...stage, records: [...frameFlagRecords, ...validityRecords, ...annotationRecords] };
          const types = stageAnomalyTypes(anomalyStage);
          const ranges = stageFrameIntervals([anomalyStage], timeline);
          const rejected = (stage.records || []).some((record) => (
            record.reject_episode === true || record.accepted === false
          ));
          const summaryRecord = firstRecordFrom(stage, "episode_summary.parquet");
          const filterRecord = firstRecordFrom(stage, "episode_filter.parquet");
          const flaggedFrames = numeric(summaryRecord?.flagged_frames) ?? frameFlagRecords.length;
          const resultTag = rejected
            ? <Tag color="red">Episode 已过滤</Tag>
            : flaggedFrames > 0 || ranges.length > 0
              ? <Tag color="orange">发现异常</Tag>
              : <Tag color="green">检测通过</Tag>;

          if (stage.stage_id === 2) {
            const episodeFlags = firstRecordFrom(stage, "episode_flags.parquet");
            const metrics = recordsFrom(stage, "dimension_metrics.parquet");
            const failedMetrics = metrics.filter((record) => record.failed === true);
            const minimumDa = numeric(episodeFlags?.minimum_da);
            const config = (stage.summary?.config || {}) as Record<string, unknown>;
            const threshold = numeric(config.stage2_da_threshold);
            const stage2Rejected = rejected || episodeFlags?.reject_episode === true || failedMetrics.length > 0;
            const children = <>
              <Alert type={stage2Rejected ? "error" : "success"} showIcon
                message={stage2Rejected ? "State-Action 趋势不一致，Episode 被过滤" : "State-Action 趋势一致，Episode 通过"}
                description="Stage 2 是 Episode/维度级判定，不生成帧级异常区间。下表直接读取 dimension_metrics.parquet。" />
              <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
                <Descriptions.Item label="检测结果">{stage2Rejected ? <Tag color="red">未通过</Tag> : <Tag color="green">通过</Tag>}</Descriptions.Item>
                <Descriptions.Item label="最小方向一致率">{minimumDa === null ? "—" : minimumDa.toFixed(4)}</Descriptions.Item>
                <Descriptions.Item label="判定阈值">{threshold === null ? "—" : threshold.toFixed(4)}</Descriptions.Item>
                <Descriptions.Item label="参与评分维度">{String(episodeFlags?.scored_dimensions ?? metrics.length)}</Descriptions.Item>
                <Descriptions.Item label="失败维度">{vector(episodeFlags?.failed_dimensions).join(", ") || "无"}</Descriptions.Item>
                <Descriptions.Item label="Episode 过滤记录">{
                  filterRecord?.accepted === false ? "已过滤" : filterRecord?.accepted === true ? "保留" : "—"
                }</Descriptions.Item>
              </Descriptions>
              <Table size="small" pagination={false} rowKey={(record, index) => `${record.state_dimension}-${record.action_dimension}-${index}`}
                dataSource={metrics} scroll={{ x: 720 }} locale={{ emptyText: "产物中没有维度级指标" }} columns={[
                  { title: "State 维度", dataIndex: "state_dimension", key: "state" },
                  { title: "Action 维度", dataIndex: "action_dimension", key: "action" },
                  { title: "方向一致率", dataIndex: "directional_agreement", key: "da", render: (value) => numeric(value)?.toFixed(4) ?? "—" },
                  { title: "时延（帧）", dataIndex: "lag_frames", key: "lag" },
                  { title: "活跃步数", dataIndex: "active_steps", key: "steps" },
                  { title: "结果", dataIndex: "failed", key: "failed", render: (value) => value === true ? <Tag color="red">失败</Tag> : <Tag color="green">通过</Tag> },
                ]} />
            </>;
            return { key: String(stage.stage_id), label, children };
          }

          const config = (stage.summary?.config || {}) as Record<string, unknown>;
          const children = <>
            <Alert type={rejected ? "error" : ranges.length ? "warning" : "success"} showIcon
              message={rejected
                ? "检测到异常，整条 Episode 已过滤"
                : ranges.length
                  ? `发现 ${ranges.length} 个异常区间，已覆盖到 State/Action 曲线中`
                  : "检测完成，当前 Episode 未发现异常"} />
            <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
              <Descriptions.Item label="检测结果">{resultTag}</Descriptions.Item>
              <Descriptions.Item label="异常类型">
                {types.length
                  ? <Space size={[4, 4]} wrap>{types.map((type) => <Tag color="orange" key={type}>{type}</Tag>)}</Space>
                  : <Tag color="green">无异常</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="异常位置">
                {ranges.length ? <Space size={[4, 4]} wrap>{ranges.map((range) => {
                  const text = range.start === range.end ? `Frame ${range.start}` : `Frame ${range.start}–${range.end}`;
                  return <Button type="link" danger size="small" key={`${range.start}-${range.end}`}
                    onClick={() => onSeek(range.start / timeline.fps)}>{text}</Button>;
                })}</Space> : rejected ? <Tag color="red">整条 Episode 被过滤</Tag> : <Tag color="green">无帧级异常</Tag>}
              </Descriptions.Item>
              <Descriptions.Item label="输入帧数">{String(summaryRecord?.num_frames ?? summaryRecord?.input_frames ?? filterRecord?.num_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="异常帧数">{String(flaggedFrames)}</Descriptions.Item>
              <Descriptions.Item label="输出有效帧">{String(summaryRecord?.valid_frames ?? summaryRecord?.output_frames ?? "—")}</Descriptions.Item>
              {stage.stage_id === 1 && <Descriptions.Item label="过滤策略">{String(summaryRecord?.exclusion_policy ?? config.stage1_exclusion ?? "—")}</Descriptions.Item>}
              {stage.stage_id === 3 && <Descriptions.Item label="边界扩展系数 α">{String(config.stage3_alpha ?? "—")}</Descriptions.Item>}
            </Descriptions>
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id === 4 && stage.artifact_status === "available") {
          const episodeSummary = firstRecordFrom(stage, "episode_summary.parquet");
          const transform = firstRecordFrom(stage, "kinematic_transform.parquet");
          const frameFlags = recordsFrom(stage, "frame_flags.parquet");
          const stageValidity = recordsFrom(stage, "step_validity.parquet").filter((record) => record.stage4_valid === false);
          const ranges = stageFrameIntervals([{ ...stage, records: [...frameFlags, ...stageValidity] }], timeline);
          const accepted = episodeSummary?.accepted !== false;
          const strategy = (stage.summary?.kinematic_strategy || {}) as Record<string, unknown>;
          const thresholds = (strategy.thresholds || {}) as Record<string, unknown>;
          const children = <>
            <Alert type={accepted ? ranges.length ? "warning" : "success" : "error"} showIcon
              message={accepted
                ? ranges.length ? `发现 ${ranges.length} 个运动学异常区间` : "运动学一致性检测通过"
                : "运动学一致性检测未通过，Episode 已过滤"}
              description="位置与姿态误差均直接来自 Stage 4 的 episode_summary.parquet。" />
            <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
              <Descriptions.Item label="状态">{resultTag(episodeSummary?.status ?? accepted)}</Descriptions.Item>
              <Descriptions.Item label="检测帧数">{String(episodeSummary?.s4_evaluated_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="软/硬异常帧">{String(episodeSummary?.s4_soft_mismatch_frames ?? "—")} / {String(episodeSummary?.s4_hard_mismatch_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="位置误差 中位/P95/最大" span={2}>
                {fixed(episodeSummary?.position_error_median_m, 6, " m")} / {fixed(episodeSummary?.position_error_p95_m, 6, " m")} / {fixed(episodeSummary?.position_error_max_m, 6, " m")}
              </Descriptions.Item>
              <Descriptions.Item label="硬异常比例">{fixed((numeric(episodeSummary?.s4_hard_mismatch_ratio) ?? 0) * 100, 3, "%")}</Descriptions.Item>
              <Descriptions.Item label="姿态误差 中位/P95/最大" span={2}>
                {fixed(episodeSummary?.orientation_error_median_deg, 4, "°")} / {fixed(episodeSummary?.orientation_error_p95_deg, 4, "°")} / {fixed(episodeSummary?.orientation_error_max_deg, 4, "°")}
              </Descriptions.Item>
              <Descriptions.Item label="模型">{String(episodeSummary?.model ?? strategy.robot_model ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="Base 世界坐标" span={2}>{vectorText(transform?.base_world_xyz_m ?? [episodeSummary?.base_world_x_m, episodeSummary?.base_world_y_m, episodeSummary?.base_world_z_m])}</Descriptions.Item>
              <Descriptions.Item label="软/硬位置阈值">{fixed(thresholds.soft_position_m, 4, " m")} / {fixed(thresholds.hard_position_m, 4, " m")}</Descriptions.Item>
              <Descriptions.Item label="异常位置" span={3}>
                {ranges.length ? <Space size={[4, 4]} wrap>{ranges.map((range) => <Button type="link" danger size="small"
                  key={`${range.start}-${range.end}`} onClick={() => onSeek(range.start / timeline.fps)}>
                  {range.start === range.end ? `Frame ${range.start}` : `Frame ${range.start}–${range.end}`}
                </Button>)}</Space> : <Tag color="green">无 Stage 4 异常帧</Tag>}
              </Descriptions.Item>
            </Descriptions>
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id === 5 && stage.artifact_status === "available") {
          const transform = firstRecordFrom(stage, "episode_transform.parquet");
          const transformation = (stage.summary?.transformation || {}) as Record<string, unknown>;
          const candidate = transform?.s4_training_candidate !== false;
          const children = <>
            <Alert type={candidate ? "success" : "warning"} showIcon
              message={candidate ? "坐标系对齐产物可用于训练" : "上游运动学结果不满足训练候选条件"}
              description="Stage 5 对当前 Episode 应用固定世界坐标到 Panda base 的平移；该产物没有帧级异常标签。" />
            <Descriptions size="small" column={{ xs: 1, md: 2 }} className="stage-details">
              <Descriptions.Item label="Base 世界坐标">{vectorText(transform?.base_world_xyz_m)}</Descriptions.Item>
              <Descriptions.Item label="Base 世界旋转">{vectorText(transform?.base_rotation_world)}</Descriptions.Item>
              <Descriptions.Item label="Stage 4 已评估">{resultTag(transform?.s4_evaluated)}</Descriptions.Item>
              <Descriptions.Item label="训练候选">{resultTag(transform?.s4_training_candidate)}</Descriptions.Item>
              <Descriptions.Item label="位置变换" span={2}>{String(transformation.position_transform || "—")}</Descriptions.Item>
              <Descriptions.Item label="姿态变换" span={2}>{String(transformation.orientation_transform || "—")}</Descriptions.Item>
              <Descriptions.Item label="Action 处理" span={2}>{String(transformation.action_transform || "—")}</Descriptions.Item>
              <Descriptions.Item label="全量已转换行数">{String(stage.summary?.rows_transformed ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="视频策略">{String(stage.summary?.videos ?? "—")}</Descriptions.Item>
            </Descriptions>
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id === 6 && stage.artifact_status === "available" && stage.detail) {
          const detail = stage.detail;
          const task = (detail.task || {}) as Record<string, unknown>;
          const scene = (detail.scene || {}) as Record<string, unknown>;
          const segmentation = (detail.temporal_segmentation || {}) as Record<string, unknown>;
          const quality = (detail.quality || {}) as Record<string, unknown>;
          const segments = (Array.isArray(segmentation.segments) ? segmentation.segments : []) as Array<Record<string, unknown>>;
          const objects = (Array.isArray(scene.objects) ? scene.objects : []) as Array<Record<string, unknown>>;
          const counts = (stage.summary?.counts || {}) as Record<string, unknown>;
          const children = <>
            <Alert type="info" showIcon message="当前 Stage 6 产物是语义子任务与场景证据"
              description="产物未提供独立的 instruction-consistency 数值分数，因此页面展示可审查证据，不虚构 pass/fail。" />
            <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
              <Descriptions.Item label="状态">{String(detail.status || "unknown")}</Descriptions.Item>
              <Descriptions.Item label="模型">{String(detail.model || "—")}</Descriptions.Item>
              <Descriptions.Item label="全量进度">{String(counts.complete ?? "—")} / {String(stage.summary?.total_episodes ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="原始指令" span={3}>{stringList(task.original_instructions).join("；") || "未提供"}</Descriptions.Item>
              <Descriptions.Item label="场景摘要" span={3}>{String(scene.summary || scene.description || "未提供")}</Descriptions.Item>
            </Descriptions>
            <Row gutter={[12, 12]} className="stage-details">
              <Col xs={24} lg={12}>
                <Card size="small" title="预期任务计划">
                  <List size="small" dataSource={stringList(task.expected_task_plan)} locale={{ emptyText: "未提供任务计划" }}
                    renderItem={(item, index) => <List.Item><Tag>{index + 1}</Tag>{item}</List.Item>} />
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card size="small" title="场景对象">
                  <Space size={[4, 6]} wrap>{objects.map((object, index) => <Tag key={`${String(object.name)}-${index}`}
                    color={object.confidence === "high" ? "green" : object.confidence === "medium" ? "gold" : "default"}>
                    {String(object.name || "unnamed")} · {String(object.task_role || object.category || "object")}
                  </Tag>)}</Space>
                  {!objects.length && <span>未提供对象列表</span>}
                </Card>
              </Col>
            </Row>
            <Card size="small" title="语义子任务时间轴（点击区间跳转视频）" className="stage-details">
              {segments.length ? <div className="semantic-timeline">{segments.map((segment, index) => {
                const start = numeric(segment.start_frame) ?? 0;
                const end = numeric(segment.end_frame_exclusive) ?? start + 1;
                const width = Math.max(4, (Math.max(1, end - start) / Math.max(1, timeline.frame_count)) * 100);
                return <button type="button" className={`semantic-segment phase-${String(segment.temporal_phase || "unknown")}`}
                  style={{ flexBasis: `${width}%` }} key={`${start}-${end}-${index}`}
                  title={`${String(segment.subtask_label || "subtask")} · Frame ${start}–${Math.max(start, end - 1)}`}
                  onClick={() => onSeek(start / timeline.fps)}>
                  <span>{index + 1}. {String(segment.subtask_label || "subtask")}</span>
                  <small>F{start}–{Math.max(start, end - 1)}</small>
                </button>;
              })}</div> : <Alert type="info" showIcon message="该 Episode 没有语义分段" />}
              {segments.length > 0 && <Table size="small" pagination={false} rowKey={(segment) => `${segment.start_frame}-${segment.end_frame_exclusive}`}
                dataSource={segments} scroll={{ x: 900 }} columns={[
                  { title: "帧区间", key: "range", width: 110, render: (_v, segment) => {
                    const start = numeric(segment.start_frame) ?? 0;
                    const end = (numeric(segment.end_frame_exclusive) ?? start + 1) - 1;
                    return <Button type="link" size="small" onClick={() => onSeek(start / timeline.fps)}>F{start}–{end}</Button>;
                  } },
                  { title: "阶段", dataIndex: "temporal_phase", key: "phase", width: 120, render: (value) => <Tag>{String(value || "unknown")}</Tag> },
                  { title: "语义子任务", dataIndex: "subtask_label", key: "label", width: 280 },
                  { title: "对象 / 目标", key: "objects", width: 210, render: (_v, segment) => [
                    ...stringList(segment.manipulated_objects), ...(segment.destination ? [String(segment.destination)] : []),
                  ].join(" → ") || "—" },
                  { title: "置信度", dataIndex: "confidence", key: "confidence", width: 90, render: (value) => <Tag color={value === "high" ? "green" : "gold"}>{String(value || "—")}</Tag> },
                  { title: "证据帧", key: "evidence", render: (_v, segment) => <Space size={[2, 2]} wrap>{stringList(segment.evidence_frames).map((frame) => <Button
                    type="link" size="small" key={frame} onClick={() => onSeek(Number(frame) / timeline.fps)}>F{frame}</Button>)}</Space> },
                ]} />}
            </Card>
            {stringList(quality.uncertainties).length > 0 && <Alert className="stage-details" type="warning" showIcon
              message="模型不确定性" description={stringList(quality.uncertainties).join("；")} />}
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id === 7 && stage.artifact_status === "available" && stage.detail) {
          const detail = stage.detail;
          const frames = (Array.isArray(detail.frames) ? detail.frames : []) as Array<Record<string, unknown>>;
          const episodeSummary = (detail.summary || {}) as Record<string, unknown>;
          const model = (detail.model || {}) as Record<string, unknown>;
          const counts = (stage.summary?.decision_counts || {}) as Record<string, unknown>;
          const decision = String(detail.decision || "unknown");
          const chartOption = {
            animation: false,
            tooltip: { trigger: "axis" },
            legend: { top: 4 },
            grid: { left: 48, right: 18, top: 42, bottom: 42 },
            xAxis: { type: "category", name: "Frame", data: frames.map((frame) => String(frame.frame_index ?? "—")) },
            yAxis: { type: "value", min: 0, max: 1 },
            series: [
              { name: "IoU", type: "line", data: frames.map((frame) => numeric(frame.iou)), connectNulls: false },
              { name: "Render coverage", type: "line", data: frames.map((frame) => numeric(frame.render_coverage)), connectNulls: false },
              { name: "SAM2 score", type: "line", data: frames.map((frame) => numeric(frame.sam2_score)), connectNulls: false },
            ],
          };
          const children = <>
            <Alert type={decision === "fail" ? "error" : decision === "pass" ? "success" : "warning"} showIcon
              message={<Space wrap>视频－状态一致性判定 {resultTag(decision)}</Space>}
              description={String(detail.reason || "产物未提供判定原因")} />
            {detail.filtering_authorized === false && <Alert className="stage-details" type="info" showIcon
              message="当前产物是审计结果，尚未授权自动过滤"
              description={detail.thresholds_calibrated === false ? "阈值尚未完成标定，insufficient_confidence 不等同于检测失败。" : undefined} />}
            <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
              <Descriptions.Item label="模型">{String(model.variant || model.type || stage.summary?.model || "—")}</Descriptions.Item>
              <Descriptions.Item label="采样帧">{String(episodeSummary.sampled_frames ?? frames.length)}</Descriptions.Item>
              <Descriptions.Item label="通过/失败/低置信度">{String(episodeSummary.pass_frames ?? "—")} / {String(episodeSummary.fail_frames ?? "—")} / {String(episodeSummary.insufficient_confidence_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="中位 IoU">{fixed(episodeSummary.median_iou)}</Descriptions.Item>
              <Descriptions.Item label="中位 SAM2 分数">{fixed(episodeSummary.median_sam2_score)}</Descriptions.Item>
              <Descriptions.Item label="全量进度">{String(stage.summary?.episode_count ?? "—")} / {String(stage.summary?.requested_episode_count ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="全量判定统计" span={3}>
                <Space wrap><Tag color="green">pass {String(counts.pass ?? 0)}</Tag><Tag color="red">fail {String(counts.fail ?? 0)}</Tag><Tag color="gold">insufficient {String(counts.insufficient_confidence ?? 0)}</Tag></Space>
              </Descriptions.Item>
            </Descriptions>
            {frames.length > 0 ? <>
              <Card size="small" title="采样帧一致性分数（点击跳转视频）" className="stage-details">
                <ReactECharts option={chartOption} style={{ height: 260 }} notMerge lazyUpdate onEvents={{
                  click: (params: { dataIndex?: number }) => {
                    const frame = frames[params.dataIndex ?? -1];
                    const index = numeric(frame?.frame_index);
                    if (index !== null) onSeek(index / timeline.fps);
                  },
                }} />
              </Card>
              <Table size="small" pagination={false} rowKey={(frame, index) => `${frame.frame_index}-${index}`}
                dataSource={frames} scroll={{ x: 900 }} columns={[
                  { title: "帧", dataIndex: "frame_index", key: "frame", render: (value) => <Button type="link" size="small" onClick={() => onSeek(Number(value) / timeline.fps)}>F{String(value)}</Button> },
                  { title: "判定", dataIndex: "decision", key: "decision", render: resultTag },
                  { title: "IoU", dataIndex: "iou", key: "iou", render: (value) => fixed(value) },
                  { title: "Render coverage", dataIndex: "render_coverage", key: "render", render: (value) => fixed(value) },
                  { title: "SAM2", dataIndex: "sam2_score", key: "sam2", render: (value) => fixed(value) },
                  { title: "原因", dataIndex: "reason", key: "reason" },
                ]} />
            </> : <Alert className="stage-details" type="warning" showIcon message="该 Episode 没有可评分采样帧" />}
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        if (stage.stage_id === 8 && stage.artifact_status === "available" && stage.detail) {
          const detail = stage.detail;
          const cameras = (Array.isArray(detail.per_camera_results) ? detail.per_camera_results : []) as Array<Record<string, unknown>>;
          const ranges = (Array.isArray(detail.invalid_ranges) ? detail.invalid_ranges : []) as Array<Record<string, unknown>>;
          const limitations = stringList(detail.known_limitations);
          const dispositions = (stage.summary?.disposition_counts || {}) as Record<string, unknown>;
          const disposition = String(detail.data_disposition || "unknown");
          const rejected = disposition === "exclude_episode_from_training";
          const partial = disposition === "exclude_affected_sample_windows";
          const dispositionText = rejected ? "整条 Episode 排除" : partial ? "排除受影响窗口" : disposition === "retain" ? "保留" : disposition;
          const children = <>
            <Alert type={rejected ? "error" : partial ? "warning" : "success"} showIcon
              message={`视频质量处置：${dispositionText}`}
              description={detail.filtering_authorized === false ? "当前为审计模式，产物中的处置建议尚未授权自动修改训练集。" : undefined} />
            <Descriptions size="small" column={{ xs: 1, md: 3 }} className="stage-details">
              <Descriptions.Item label="状态">{resultTag(detail.status)}</Descriptions.Item>
              <Descriptions.Item label="总帧数">{String(detail.total_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="无效帧">{String(detail.invalid_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="冗余静止帧">{String(detail.redundant_static_frames ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="受保护关键帧">{String(detail.protected_keyframes ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="全量进度">{String(stage.summary?.episode_count ?? "—")} / {String(stage.summary?.total_episodes ?? "—")}</Descriptions.Item>
              <Descriptions.Item label="全量处置统计" span={3}>
                <Space wrap><Tag color="green">保留 {String(dispositions.retain ?? 0)}</Tag><Tag color="red">整条排除 {String(dispositions.exclude_episode_from_training ?? 0)}</Tag><Tag color="orange">窗口排除 {String(dispositions.exclude_affected_sample_windows ?? 0)}</Tag></Space>
              </Descriptions.Item>
              <Descriptions.Item label="异常区间" span={3}>
                {ranges.length ? <Space size={[4, 4]} wrap>{ranges.map((range, index) => {
                  const start = numeric(range.start_frame) ?? 0;
                  const endExclusive = numeric(range.end_frame) ?? start + 1;
                  const reasons = stringList(range.reasons).join(", ") || "video_quality";
                  return <Button type="link" danger size="small" key={`${start}-${endExclusive}-${index}`}
                    onClick={() => onSeek(start / timeline.fps)}>F{start}–{Math.max(start, endExclusive - 1)} · {reasons}</Button>;
                })}</Space> : <Tag color="green">无视频质量异常区间</Tag>}
              </Descriptions.Item>
            </Descriptions>
            <Table size="small" pagination={false} rowKey={(camera, index) => `${camera.camera}-${index}`}
              dataSource={cameras} scroll={{ x: 900 }} locale={{ emptyText: "产物中没有相机统计" }} columns={[
                { title: "相机", dataIndex: "camera", key: "camera" },
                { title: "状态", dataIndex: "status", key: "status", render: resultTag },
                { title: "损坏帧", dataIndex: "corrupted_frames", key: "corrupted" },
                { title: "黑屏帧", dataIndex: "black_frames", key: "black" },
                { title: "模糊帧", dataIndex: "blurred_frames", key: "blurred" },
                { title: "中位亮度", dataIndex: "median_luminance", key: "luminance", render: (value) => fixed(value, 2) },
                { title: "中位清晰度", dataIndex: "median_blur_score", key: "blur", render: (value) => fixed(value, 2) },
                { title: "原因", dataIndex: "reason", key: "reason", render: (value) => String(value || "—") },
              ]} />
            {limitations.length > 0 && <Alert className="stage-details" type="warning" showIcon
              message="已知限制" description={limitations.join("；")} />}
          </>;
          return { key: String(stage.stage_id), label, children };
        }
        const children = <>
          <Alert type={stage.artifact_status === "not_generated" ? "warning" : "info"} showIcon
            message={stage.placeholder || "该 Episode 尚无可视化产物"}
            description="占位只说明数据状态，不代表该 Stage 已通过。" />
          <Descriptions size="small" column={1} className="stage-details">
            <Descriptions.Item label="论文/流程目标">{spec.description || "等待阶段产物"}</Descriptions.Item>
            <Descriptions.Item label="产物生成后展示">{spec.visualization || "指标、异常区间与审计证据"}</Descriptions.Item>
            <Descriptions.Item label="预期字段"><Space size={[4, 4]} wrap>{(spec.expected_fields || []).map((field) => <Tag key={field}>{field}</Tag>)}</Space></Descriptions.Item>
          </Descriptions>
        </>;
        return { key: String(stage.stage_id), label, children };
      });
  return <Card title="Stage 1–8 检测与专项可视化" className="section-card">
    <Collapse defaultActiveKey={rows.map((stage) => String(stage.stage_id))} items={items} onChange={(keys) => {
      const active = Array.isArray(keys) ? keys : [keys];
      active.forEach((key) => {
        const stageId = Number(key);
        const stage = rows.find((item) => item.stage_id === stageId);
        if (stage?.artifact_status === "available" && !stage.detail_loaded) onLoadStage(stageId);
      });
    }} />
  </Card>;
}

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const collections = useMemo(() => buildCollections(datasets), [datasets]);
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>();
  const selectedCollection = useMemo(
    () => collections.find((item) => item.id === selectedCollectionId),
    [collections, selectedCollectionId],
  );
  const [selected, setSelected] = useState<Dataset>();
  const [tasks, setTasks] = useState<DatasetTask[]>([]);
  const [taskIndex, setTaskIndex] = useState<number>();
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episodeIndex, setEpisodeIndex] = useState<number>();
  const [preview, setPreview] = useState<Preview>();
  const [loadingStageDetails, setLoadingStageDetails] = useState<Set<number>>(new Set());
  const [series, setSeries] = useState<SeriesRow[]>([]);
  const [seriesView, setSeriesView] = useState<SeriesView>("raw");
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [scanTask, setScanTask] = useState<ScanTask>();
  const [loadingEpisode, setLoadingEpisode] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackStatus, setPlaybackStatus] = useState<PlaybackStatus>("idle");
  const [playbackError, setPlaybackError] = useState<string>();
  const [proxyJob, setProxyJob] = useState<ProxyJob>();
  const [proxyError, setProxyError] = useState<string>();
  const [useOriginalVideo, setUseOriginalVideo] = useState(false);
  const [browserVideoMetadata, setBrowserVideoMetadata] = useState<Record<string, BrowserVideoMetadata>>({});
  const [currentTime, setCurrentTime] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const currentTimeRef = useRef(0);
  const playbackRequest = useRef(0);
  const lastFollowerSync = useRef(0);
  const lastUiUpdate = useRef(0);
  const lastHardSeek = useRef<Record<string, number>>({});
  const workbenchRef = useRef<HTMLDivElement>(null);
  const navigationTarget = useRef<EpisodeSearchItem | undefined>(undefined);
  const initialSearchStarted = useRef(false);
  const searchSequence = useRef(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchDatasets, setSearchDatasets] = useState<string[]>([]);
  const [searchStageValues, setSearchStageValues] = useState<Record<number, string[]>>({});
  const [searchSort, setSearchSort] = useState("relevance");
  const [searchResult, setSearchResult] = useState<EpisodeSearchResponse>();
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchIndexTask, setSearchIndexTask] = useState<SearchIndexTask>();
  const [searchIndexAvailable, setSearchIndexAvailable] = useState<boolean>();
  const [searchStageFacets, setSearchStageFacets] = useState<SearchStageFacets>({});

  const refresh = () => fetch("/api/datasets").then((response) => response.json()).then(setDatasets)
    .catch(() => message.error("后端未启动"));

  const chooseCollection = (collection: DatasetCollection) => {
    navigationTarget.current = undefined;
    setSelectedCollectionId(collection.id);
    setSelected(collection.datasets.length === 1 ? collection.datasets[0] : undefined);
    setTasks([]); setTaskIndex(undefined); setEpisodes([]); setEpisodeIndex(undefined);
    setPreview(undefined); setSeries([]);
  };

  const makeSearchBody = (
    page = 1,
    overrides?: { query?: string; datasets?: string[]; stages?: Record<number, string[]>; sort?: string },
  ) => {
    const stages = overrides?.stages ?? searchStageValues;
    return {
      query: overrides?.query ?? searchQuery,
      datasets: overrides?.datasets ?? searchDatasets,
      stage_filters: Object.entries(stages).flatMap(([stageId, values]) => {
        if (!values.length) return [];
        return [{
          stage_id: Number(stageId),
          verdicts: values.filter((value) => !value.startsWith("status:")),
          artifact_statuses: values.filter((value) => value.startsWith("status:")).map((value) => value.slice(7)),
        }];
      }),
      sort: overrides?.sort ?? searchSort,
      page,
      page_size: 24,
    };
  };

  const executeSearch = async (
    page = 1,
    prewarm = true,
    overrides?: { query?: string; datasets?: string[]; stages?: Record<number, string[]>; sort?: string },
  ) => {
    const sequence = ++searchSequence.current;
    const body = makeSearchBody(page, overrides);
    setSearchLoading(true);
    try {
      const response = await fetch("/api/search/episodes", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Episode 搜索失败");
      if (sequence !== searchSequence.current) return;
      let result = payload as EpisodeSearchResponse;
      setSearchResult(result);
      setSearchLoading(false);
      if (!prewarm || !result.items.length) return;
      const prewarmResponse = await fetch("/api/thumbnails/prewarm", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ episodes: result.items.map((item) => ({
          dataset_uid: item.dataset_uid, episode_index: item.episode_index,
        })) }),
      });
      if (!prewarmResponse.ok) return;
      const deadline = Date.now() + 30000;
      while (sequence === searchSequence.current && Date.now() < deadline
        && result.items.some((item) => ["not_generated", "queued", "generating"].includes(item.thumbnail_status))) {
        await new Promise((resolve) => window.setTimeout(resolve, 800));
        const poll = await fetch("/api/search/episodes", {
          method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
        });
        if (!poll.ok || sequence !== searchSequence.current) return;
        result = await poll.json() as EpisodeSearchResponse;
        setSearchResult(result);
      }
    } catch (error) {
      if (sequence === searchSequence.current) {
        setSearchLoading(false);
        message.error(error instanceof Error ? error.message : "Episode 搜索失败");
      }
    }
  };

  const rebuildSearchIndex = async () => {
    try {
      const response = await fetch("/api/search/index", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ datasets: searchDatasets }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "搜索索引任务创建失败");
      let task = payload as SearchIndexTask;
      setSearchIndexTask(task);
      while (!["succeeded", "failed"].includes(task.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 600));
        const status = await fetch(`/api/search/index/${encodeURIComponent(task.task_id)}`);
        const statusPayload = await status.json();
        if (!status.ok) throw new Error(statusPayload.detail || "搜索索引状态读取失败");
        task = statusPayload as SearchIndexTask;
        setSearchIndexTask(task);
      }
      if (task.status !== "succeeded") throw new Error(task.error || "搜索索引构建失败");
      setSearchIndexAvailable(true);
      fetch("/api/search/facets").then((response) => response.json()).then((payload) => setSearchStageFacets(payload.stages || {}));
      await executeSearch(1);
    } catch (error) {
      setSearchIndexTask(undefined);
      message.error(error instanceof Error ? error.message : "搜索索引构建失败");
    }
  };

  const resetSearch = () => {
    const emptyStages: Record<number, string[]> = {};
    setSearchQuery(""); setSearchDatasets([]); setSearchStageValues(emptyStages); setSearchSort("relevance");
    void executeSearch(1, true, { query: "", datasets: [], stages: emptyStages, sort: "relevance" });
  };

  const openSearchEpisode = (item: EpisodeSearchItem) => {
    const dataset = datasets.find((value) => value.uid === item.dataset_uid);
    if (!dataset) { message.error(`目录中未找到子数据集 ${item.dataset_uid}`); return; }
    navigationTarget.current = item;
    setSelectedCollectionId(item.collection_name);
    if (selected?.uid === dataset.uid) {
      const nextTask = item.task_index ?? undefined;
      if (taskIndex === nextTask) {
        setEpisodeIndex(item.episode_index);
        navigationTarget.current = undefined;
        window.setTimeout(() => workbenchRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
      } else {
        setTaskIndex(nextTask);
      }
    } else {
      setSelected(dataset);
    }
  };

  const scan = async (mode: "quick" | "standard" | "deep" = "quick") => {
    try {
      const response = await fetch("/api/catalog/scan", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      if (!response.ok) throw new Error();
      let job: ScanTask = await response.json();
      setScanTask(job);
      while (!["succeeded", "failed", "cancelled"].includes(job.status)) {
        await new Promise((resolve) => setTimeout(resolve, 500));
        job = await fetch(`/api/catalog/scans/${job.scan_id}`).then((item) => item.json());
        setScanTask(job);
      }
      if (job.status !== "succeeded") throw new Error();
      await refresh();
      message.success(`${mode} 扫描完成：${job.datasets} 个数据集`);
    } catch {
      message.error("扫描失败，请检查后端和数据路径");
    }
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!datasets.length || initialSearchStarted.current) return;
    initialSearchStarted.current = true;
    fetch("/api/search/index").then((response) => response.json()).then((stats) => {
      const available = Number(stats.episodes || 0) > 0;
      setSearchIndexAvailable(available);
      if (available) {
        fetch("/api/search/facets").then((response) => response.json()).then((payload) => setSearchStageFacets(payload.stages || {}));
        void executeSearch(1);
      }
    }).catch(() => setSearchIndexAvailable(false));
  }, [datasets]);

  useEffect(() => {
    if (!selected) { setTasks([]); setTaskIndex(undefined); return; }
    let cancelled = false;
    setTasks([]); setTaskIndex(undefined); setEpisodes([]); setEpisodeIndex(undefined);
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/tasks`)
      .then((response) => response.json()).then((items: DatasetTask[]) => {
        if (cancelled) return;
        setTasks(items);
        const target = navigationTarget.current;
        if (target?.dataset_uid === selected.uid) setTaskIndex(target.task_index ?? undefined);
      })
      .catch(() => { if (!cancelled) message.error("Task 列表读取失败"); });
    return () => { cancelled = true; };
  }, [selected]);

  useEffect(() => {
    if (!selected) { setEpisodes([]); setEpisodeIndex(undefined); return; }
    let cancelled = false;
    setPreview(undefined); setSeries([]); setEpisodeIndex(undefined);
    const query = taskIndex === undefined ? "" : `?task_index=${taskIndex}`;
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes${query}`)
      .then((response) => response.json()).then((items: Episode[]) => {
        if (cancelled) return;
        setEpisodes(items);
        const target = navigationTarget.current;
        if (target?.dataset_uid === selected.uid
          && (target.task_index == null || target.task_index === taskIndex)) {
          setEpisodeIndex(target.episode_index);
          navigationTarget.current = undefined;
          window.setTimeout(() => workbenchRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
        } else if (items.length) setEpisodeIndex(items[0].episode_index);
      }).catch(() => { if (!cancelled) message.error("Episode 列表读取失败"); });
    return () => { cancelled = true; };
  }, [selected, taskIndex]);

  useEffect(() => {
    if (!selected || episodeIndex === undefined) return;
    let cancelled = false;
    playbackRequest.current += 1;
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      video.pause();
      video.playbackRate = 1;
    });
    currentTimeRef.current = 0;
    lastUiUpdate.current = 0;
    setIsPlaying(false); setPlaybackStatus("idle"); setPlaybackError(undefined);
    setLoadingEpisode(true); setCurrentTime(0); setCurrentFrame(0);
    setPreview(undefined); setLoadingStageDetails(new Set());
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes/${episodeIndex}/preview`)
      .then((response) => response.json()).then((episodePreview) => {
      if (cancelled) return;
      setPreview(episodePreview);
    }).catch(() => { if (!cancelled) message.error("Episode 预览读取失败"); })
      .finally(() => { if (!cancelled) setLoadingEpisode(false); });
    return () => { cancelled = true; };
  }, [episodeIndex, selected]);

  const loadStageDetail = async (stageId: number) => {
    if (!selected || episodeIndex === undefined || loadingStageDetails.has(stageId)) return;
    const requestedUid = selected.uid;
    const requestedEpisode = episodeIndex;
    setLoadingStageDetails((current) => new Set(current).add(stageId));
    try {
      const response = await fetch(
        `/api/datasets/${encodeURIComponent(requestedUid)}/episodes/${requestedEpisode}/stages/${stageId}`,
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Stage ${stageId} 详情读取失败`);
      setPreview((current) => {
        if (!current || current.dataset.uid !== requestedUid
          || current.episode.episode_index !== requestedEpisode) return current;
        return {
          ...current,
          stage_results: current.stage_results.map((stage) => (
            stage.stage_id === stageId ? payload as StageResult : stage
          )),
        };
      });
    } catch (error) {
      message.error(error instanceof Error ? error.message : `Stage ${stageId} 详情读取失败`);
    } finally {
      setLoadingStageDetails((current) => {
        const next = new Set(current); next.delete(stageId); return next;
      });
    }
  };

  useEffect(() => {
    const format = preview?.curation?.format;
    if (format === "vla_curation_filter" && ["repaired", "diff"].includes(seriesView)) {
      setSeriesView("valid");
    } else if (format === "vla_curation_overlay" && seriesView === "valid") {
      setSeriesView("raw");
    } else if (!format && seriesView !== "raw") {
      setSeriesView("raw");
    }
  }, [preview?.curation?.format, seriesView]);

  useEffect(() => {
    if (!selected || episodeIndex === undefined) return;
    let cancelled = false;
    setLoadingSeries(true); setSeries([]);
    const query = new URLSearchParams({
      fields: "timestamp,frame_index,observation.state,action",
      limit: "5000",
      view: seriesView,
    });
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes/${episodeIndex}/series?${query}`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "曲线数据读取失败");
        return payload;
      })
      .then((payload) => { if (!cancelled) setSeries(payload.rows || []); })
      .catch((error) => { if (!cancelled) message.error(error instanceof Error ? error.message : "曲线数据读取失败"); })
      .finally(() => { if (!cancelled) setLoadingSeries(false); });
    return () => { cancelled = true; };
  }, [episodeIndex, selected, seriesView]);

  useEffect(() => {
    if (!preview) {
      setProxyJob(undefined); setProxyError(undefined); setUseOriginalVideo(false);
      return undefined;
    }
    let cancelled = false;
    setProxyJob(undefined); setProxyError(undefined); setUseOriginalVideo(false);
    const requestProxy = async () => {
      try {
        const response = await fetch(
          `/api/datasets/${encodeURIComponent(preview.dataset.uid)}/episodes/${preview.episode.episode_index}/video-proxies?prewarm=2`,
          { method: "POST" },
        );
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "代理视频任务创建失败");
        let job = payload as ProxyJob;
        if (!cancelled) setProxyJob(job);
        while (!cancelled && ["queued", "generating"].includes(job.status)) {
          await new Promise((resolve) => window.setTimeout(resolve, 750));
          if (cancelled) return;
          const statusResponse = await fetch(`/api/video-proxy-jobs/${encodeURIComponent(job.job_id)}`);
          const statusPayload = await statusResponse.json();
          if (!statusResponse.ok) throw new Error(statusPayload.detail || "代理视频状态读取失败");
          job = statusPayload as ProxyJob;
          if (!cancelled) setProxyJob(job);
        }
        if (!cancelled && job.status === "failed") setProxyError(job.error || "代理视频生成失败");
      } catch (error) {
        if (!cancelled) setProxyError(error instanceof Error ? error.message : "代理视频生成失败");
      }
    };
    void requestProxy();
    return () => { cancelled = true; };
  }, [preview?.dataset.uid, preview?.episode.episode_index]);

  const fps = Number(preview?.timeline?.fps || ((preview?.dataset.schema?.timestamp as { fps?: number } | undefined)?.fps)
    || ((selected?.schema?.timestamp as { fps?: number } | undefined)?.fps) || 20);
  const timeline = useMemo<EpisodeTimeline>(() => ({
    coordinate_system: preview?.timeline?.coordinate_system || "episode_relative",
    frame_count: preview?.timeline?.frame_count || preview?.episode.frames || 0,
    fps,
    duration: preview?.timeline?.duration || preview?.episode.duration || 0,
    dataset_from_index: preview?.timeline?.dataset_from_index || 0,
  }), [fps, preview]);
  const intervals = useMemo(() => stageIntervals(preview?.stage_results || [], timeline), [preview, timeline]);
  const readyProxyVideos = proxyJob?.status === "ready" ? proxyJob.videos : undefined;
  const activeVideos = useMemo<VideoRef[]>(() => {
    if (!preview) return [];
    if (useOriginalVideo) return preview.videos;
    if (!readyProxyVideos) return [];
    const byCamera = new Map(readyProxyVideos.map((video) => [video.camera, video]));
    return preview.videos.flatMap((video) => {
      const proxy = byCamera.get(video.camera);
      return proxy?.status === "ready" ? [{
        ...video, url: proxy.url, source_start: 0, source_end: timeline.duration,
        timestamp_start: 0, timestamp_end: timeline.duration, duration: timeline.duration,
        fps: proxy.fps, frame_count: proxy.frame_count, codec: "h264", pixel_format: "yuv420p",
        integrity_status: "pass", is_proxy: true,
      }] : [];
    });
  }, [preview, readyProxyVideos, timeline.duration, useOriginalVideo]);

  const technicalVideos = activeVideos.length ? activeVideos : (preview?.videos || []);

  useEffect(() => {
    if (!preview) return;
    playbackRequest.current += 1;
    currentTimeRef.current = 0;
    lastFollowerSync.current = 0;
    lastUiUpdate.current = 0;
    lastHardSeek.current = {};
    setBrowserVideoMetadata({});
    setIsPlaying(false); setPlaybackStatus("idle"); setPlaybackError(undefined);
    setCurrentTime(0); setCurrentFrame(0);
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      video.pause();
      video.playbackRate = 1;
      const meta = activeVideos.find((item) => item.camera === video.dataset.camera);
      if (meta && video.readyState >= 1) video.currentTime = videoWindow(meta, timeline.duration).start;
    });
  }, [activeVideos, preview, timeline.duration]);
  const episodeOptions = useMemo(() => episodes.map((item) => ({
    value: item.episode_index,
    label: `Episode ${item.episode_index} · ${item.frames || "?"} frames${item.instruction ? ` · ${item.instruction.slice(0, 72)}` : ""}`,
  })), [episodes]);
  const taskOptions = useMemo(() => tasks.map((item) => ({
    value: item.task_index,
    label: `Task ${item.task_index} · ${item.episodes} episodes · ${item.name}`,
  })), [tasks]);

  const pausePlayback = (status: PlaybackStatus = "paused") => {
    playbackRequest.current += 1;
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      video.pause();
      video.playbackRate = 1;
    });
    setIsPlaying(false);
    setPlaybackStatus(status);
    setCurrentTime(currentTimeRef.current);
    setCurrentFrame(Math.min(
      Math.max(0, timeline.frame_count - 1),
      Math.floor(currentTimeRef.current * fps + 1e-6),
    ));
  };

  const syncTime = (source: HTMLVideoElement) => {
    const sourceMeta = activeVideos.find((item) => item.camera === source.dataset.camera);
    if (!sourceMeta) return;
    const sourceWindow = videoWindow(sourceMeta, timeline.duration);
    if (source.currentTime < sourceWindow.start) source.currentTime = sourceWindow.start;
    if (source.currentTime >= sourceWindow.end - 0.5 / fps) {
      source.currentTime = sourceWindow.end;
      currentTimeRef.current = timeline.duration;
      setCurrentTime(timeline.duration);
      setCurrentFrame(Math.max(0, timeline.frame_count - 1));
      pausePlayback();
      return;
    }
    const relativeTime = Math.min(Math.max(source.currentTime - sourceWindow.start, 0), timeline.duration);
    const now = performance.now();
    // The primary camera owns the playback clock. Followers are corrected at
    // most four times per second: small drift is ignored, medium drift uses a
    // gentle rate adjustment, and only large drift causes a real seek.
    if (now - lastFollowerSync.current >= 250) {
      lastFollowerSync.current = now;
      Object.values(videoRefs.current).forEach((video) => {
        if (!video || video === source || video.seeking || video.paused) return;
        const meta = activeVideos.find((item) => item.camera === video.dataset.camera);
        if (!meta) return;
        const window = videoWindow(meta, timeline.duration);
        const target = Math.min(window.end, window.start + relativeTime);
        const drift = target - video.currentTime;
        const camera = video.dataset.camera || "unknown";
        if (Math.abs(drift) > 0.5 && now - (lastHardSeek.current[camera] || 0) >= 1000) {
          lastHardSeek.current[camera] = now;
          video.playbackRate = 1;
          video.currentTime = target;
        } else if (Math.abs(drift) > 0.2) {
          video.playbackRate = Math.min(1.05, Math.max(0.95, 1 + drift * 0.2));
        } else {
          video.playbackRate = 1;
        }
      });
    }
    currentTimeRef.current = relativeTime;
    if (now - lastUiUpdate.current >= 200) {
      lastUiUpdate.current = now;
      setCurrentTime(relativeTime);
      setCurrentFrame(Math.min(Math.max(0, timeline.frame_count - 1), Math.floor(relativeTime * fps + 1e-6)));
    }
  };
  const seek = (time: number) => {
    const duration = timeline.duration;
    const relativeTime = Math.min(Math.max(time, 0), duration);
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      const meta = activeVideos.find((item) => item.camera === video.dataset.camera);
      if (!meta) return;
      const videoDuration = videoWindow(meta, duration);
      video.currentTime = Math.min(videoDuration.end, videoDuration.start + relativeTime);
    });
    currentTimeRef.current = relativeTime;
    setCurrentTime(relativeTime);
    setCurrentFrame(Math.min(Math.max(0, timeline.frame_count - 1), Math.floor(relativeTime * fps + 1e-6)));
  };

  const startPlayback = async () => {
    if (!activeVideos.length) return;
    const request = ++playbackRequest.current;
    const relativeTime = currentTimeRef.current >= timeline.duration - 1 / fps ? 0 : currentTimeRef.current;
    const videos = Object.values(videoRefs.current).filter((video): video is HTMLVideoElement => Boolean(video));
    setPlaybackError(undefined);
    setPlaybackStatus("seeking");
    currentTimeRef.current = relativeTime;
    setCurrentTime(relativeTime);
    setCurrentFrame(Math.min(Math.max(0, timeline.frame_count - 1), Math.floor(relativeTime * fps + 1e-6)));
    try {
      await Promise.all(videos.map((video) => {
        const meta = activeVideos.find((item) => item.camera === video.dataset.camera);
        if (!meta) return Promise.resolve();
        const window = videoWindow(meta, timeline.duration);
        video.playbackRate = 1;
        return seekMedia(video, Math.min(window.end, window.start + relativeTime));
      }));
      if (request !== playbackRequest.current) return;
      setPlaybackStatus("buffering");
      await Promise.all(videos.map((video) => withTimeout(
        video.play(),
        `等待视频 ${video.dataset.camera || "unknown"} 开始播放超时`,
      )));
      if (request !== playbackRequest.current) {
        videos.forEach((video) => video.pause());
        return;
      }
      setIsPlaying(true);
      setPlaybackStatus("playing");
    } catch (error) {
      if (request !== playbackRequest.current) return;
      videos.forEach((video) => video.pause());
      setIsPlaying(false);
      setPlaybackStatus("error");
      setPlaybackError(error instanceof Error ? error.message : "视频播放失败");
    }
  };

  const togglePlayback = () => {
    if (isPlaying || ["seeking", "buffering"].includes(playbackStatus)) {
      pausePlayback();
      return;
    }
    void startPlayback();
  };

  useEffect(() => {
    if (!isPlaying || !activeVideos.length) return undefined;
    const primary = videoRefs.current[activeVideos[0].camera];
    if (!primary) return undefined;
    type FrameVideo = HTMLVideoElement & {
      requestVideoFrameCallback?: (callback: () => void) => number;
      cancelVideoFrameCallback?: (handle: number) => void;
    };
    const frameVideo = primary as FrameVideo;
    let handle = 0;
    let animationHandle = 0;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      syncTime(primary);
      if (frameVideo.requestVideoFrameCallback) handle = frameVideo.requestVideoFrameCallback(tick);
      else animationHandle = window.requestAnimationFrame(tick);
    };
    tick();
    return () => {
      cancelled = true;
      if (handle && frameVideo.cancelVideoFrameCallback) frameVideo.cancelVideoFrameCallback(handle);
      if (animationHandle) window.cancelAnimationFrame(animationHandle);
    };
  }, [activeVideos, isPlaying, preview, timeline.duration]);

  const stateWidth = vector(series.find((row) => vector(row["observation.state"]).length)?.["observation.state"]).length;
  const actionWidth = vector(series.find((row) => vector(row.action).length)?.action).length;
  const stateElementNames = useMemo(
    () => featureElementNames(preview?.dataset || selected, "observation.state", stateWidth),
    [preview?.dataset, selected, stateWidth],
  );
  const actionElementNames = useMemo(
    () => featureElementNames(preview?.dataset || selected, "action", actionWidth),
    [actionWidth, preview?.dataset, selected],
  );
  const seriesViewOptions = preview?.curation?.format === "vla_curation_filter"
    ? [{ label: "原始数据", value: "raw" }, { label: "有效帧", value: "valid" }]
    : preview?.curation?.has_repairs
      ? [{ label: "原始", value: "raw" }, { label: "修复后", value: "repaired" }, { label: "差值", value: "diff" }]
      : [{ label: "原始数据", value: "raw" }];
  const searchOptionsForStage = (stageId: number) => {
    const options = new Map<string, { label: string; value: string }>();
    [...stageFilterOptions[stageId], ...artifactFilterOptions].forEach((option) => options.set(option.value, option));
    (searchStageFacets[String(stageId)]?.verdicts || []).forEach((item) => {
      const existing = options.get(item.value);
      options.set(item.value, { value: item.value, label: `${existing?.label || searchStatusLabel(item.value)} (${item.count})` });
    });
    (searchStageFacets[String(stageId)]?.artifact_statuses || []).forEach((item) => {
      const value = `status:${item.value}`;
      const existing = options.get(value);
      options.set(value, { value, label: `${existing?.label || searchStatusLabel(item.value)} (${item.count})` });
    });
    return Array.from(options.values());
  };

  return <Layout className="shell">
    <Layout.Header>
      <h1>VLA Data Governance Workbench</h1>
      <Space>
        {scanTask && <Tag color={scanTask.status === "succeeded" ? "green" : "blue"}>扫描 {scanTask.status} {scanTask.total ? `${scanTask.current}/${scanTask.total}` : ""}</Tag>}
        <Button onClick={() => void scan("quick")}>快速扫描</Button>
        <Button onClick={() => void scan("standard")}>标准扫描</Button>
      </Space>
    </Layout.Header>
    <Layout.Content>
      <Card className="search-panel" title="Episode 搜索" extra={<Space>
        {searchIndexTask && <Tag color={searchIndexTask.status === "succeeded" ? "green" : searchIndexTask.status === "failed" ? "red" : "blue"}>
          索引 {searchIndexTask.status}{searchIndexTask.total ? ` ${searchIndexTask.current}/${searchIndexTask.total}` : ""}
        </Tag>}
        <Button loading={searchIndexTask?.status === "queued" || searchIndexTask?.status === "running"} onClick={() => void rebuildSearchIndex()}>
          更新搜索索引
        </Button>
      </Space>}>
        <div className="search-primary-row">
          <Input.Search
            allowClear value={searchQuery} placeholder="搜索数据集、子数据集、Task、Instruction 或 Episode 编号…"
            enterButton="搜索" onChange={(event) => setSearchQuery(event.target.value)} onSearch={() => void executeSearch(1)}
          />
          <Button onClick={resetSearch}>重置</Button>
        </div>
        <div className="search-filter-grid">
          <Select mode="multiple" allowClear maxTagCount="responsive" value={searchDatasets}
            placeholder="数据集" onChange={setSearchDatasets}
            options={collections.map((collection) => ({
              value: collection.name, label: `${collection.name} · ${collection.episodes} episodes`,
            }))} />
          {Array.from({ length: 8 }, (_, index) => index + 1).map((stageId) => <Select
            key={stageId} mode="multiple" allowClear maxTagCount={1}
            value={searchStageValues[stageId] || []} placeholder={`Stage ${stageId}`}
            options={searchOptionsForStage(stageId)}
            onChange={(values) => setSearchStageValues((current) => ({ ...current, [stageId]: values }))}
          />)}
          <Select value={searchSort} onChange={setSearchSort} options={[
            { value: "relevance", label: "相关性" }, { value: "episode", label: "Episode 顺序" },
            { value: "duration_asc", label: "时长从短到长" }, { value: "duration_desc", label: "时长从长到短" },
          ]} />
        </div>
        {searchIndexTask && ["queued", "running"].includes(searchIndexTask.status) && <Alert className="search-index-alert" type="info" showIcon
          message={`正在建立 Episode 与 Stage 搜索索引${searchIndexTask.dataset_uid ? `：${searchIndexTask.dataset_uid}` : ""}`}
          description="索引任务只读取紧凑元数据和 Stage 产物，不解码视频；搜索结果会在完成后自动刷新。" />}
        {searchIndexAvailable === false && !searchIndexTask && <Alert className="search-index-alert" type="warning" showIcon
          message="尚未建立 Episode 搜索索引"
          description="点击“更新搜索索引”后才会读取紧凑元数据；不会解码或复制视频。" />}
        <Spin spinning={searchLoading}>
          {searchResult && <>
            <div className="search-summary">
              找到 <strong>{searchResult.total}</strong> 个 Episode · {searchResult.task_count} 个 Task · {searchResult.dataset_count} 个数据集
            </div>
            {searchResult.items.length
              ? <Row gutter={[16, 20]}>{searchResult.items.map((item) => <Col xs={24} sm={12} md={8} xl={6} key={`${item.dataset_uid}:${item.episode_index}`}>
                <EpisodeSearchCard item={item} onOpen={() => openSearchEpisode(item)} />
              </Col>)}</Row>
              : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有符合当前条件的 Episode" />}
            {searchResult.total > searchResult.page_size && <Pagination className="search-pagination"
              current={searchResult.page} pageSize={searchResult.page_size} total={searchResult.total}
              showSizeChanger={false} onChange={(page) => void executeSearch(page)} />}
          </>}
        </Spin>
      </Card>
      <Row gutter={16}>
        <Col xs={24} lg={7}>
          <Card title="数据集目录" extra={<Tag>{collections.length}</Tag>}>
            <List dataSource={collections} className="dataset-list" renderItem={(collection) => <List.Item onClick={() => chooseCollection(collection)} className={selectedCollection?.id === collection.id ? "selected" : "dataset-item"}>
              <List.Item.Meta title={collection.name} description={`${collection.episodes} episodes · ${collection.frames} frames · ${collection.datasets.length} 个数据集`} />
              <Tag color="green">{collection.datasets.length > 1 ? `${collection.datasets.length} members` : collection.datasets[0]?.codebase_version}</Tag>
            </List.Item>} />
          </Card>
        </Col>
        <Col xs={24} lg={17}>
          {!selectedCollection && <Card><Alert type="info" showIcon message="请先从左侧选择数据集集合" /></Card>}
          {selectedCollection && !selected && <Card title={`数据集 / ${selectedCollection.name}`}>
            <Alert type="info" showIcon message="该集合包含多个物理数据集，请先选择其中一个，再选择 Task 和 Episode。" />
            <Select showSearch virtual optionFilterProp="label" placeholder="选择子数据集 / Task 集合" style={{ width: "100%", marginTop: 12 }}
              options={selectedCollection.datasets.map((item) => ({ value: item.uid, label: `${item.uid} · ${item.episodes} episodes · ${item.frames} frames` }))}
              onChange={(uid) => { navigationTarget.current = undefined; setSelected(selectedCollection.datasets.find((item) => item.uid === uid)); }} />
          </Card>}
          {selected && <div ref={workbenchRef}>
            <Card title={`Episode 预览 / ${selectedCollection?.name || selected.uid}`}>
              <div className={`episode-selector-row episode-selector-primary${selectedCollection && selectedCollection.datasets.length > 1 ? "" : " single"}`}>
                {selectedCollection && selectedCollection.datasets.length > 1 && <div className="episode-selector-field">
                  <span>子数据集</span>
                  <Select showSearch virtual optionFilterProp="label" value={selected.uid}
                    placeholder="选择子数据集" options={selectedCollection.datasets.map((item) => ({ value: item.uid, label: `${item.uid} · ${item.episodes} episodes` }))}
                    onChange={(uid) => { navigationTarget.current = undefined; setSelected(selectedCollection.datasets.find((item) => item.uid === uid)); setTaskIndex(undefined); }} />
                </div>}
                <div className="episode-selector-field">
                  <span>Task</span>
                  <Select allowClear showSearch virtual optionFilterProp="label" placeholder="选择 Task" value={taskIndex} options={taskOptions}
                    onChange={(value) => { navigationTarget.current = undefined; setTaskIndex(value); }} />
                </div>
              </div>
              <div className="episode-selector-row episode-selector-secondary">
                <div className="episode-selector-field">
                  <span>Episode</span>
                  <Select showSearch virtual optionFilterProp="label" placeholder="选择 Episode" value={episodeIndex} options={episodeOptions} onChange={setEpisodeIndex} />
                </div>
                {episodeIndex !== undefined && <div className="episode-selector-field episode-number-field">
                  <span>编号</span>
                  <InputNumber min={0} max={Math.max(0, selected.episodes - 1)} value={episodeIndex} onChange={(value) => value !== null && setEpisodeIndex(value)} />
                </div>}
              </div>
              <Row gutter={16} className="episode-statistics">
                <Col><Statistic title="Episodes" value={selected.episodes} /></Col>
                <Col><Statistic title="Frames" value={selected.frames} /></Col>
                <Col><Statistic title="Cameras" value={preview?.videos.length || selected.cameras.length} /></Col>
                <Col><Statistic title="当前帧" value={currentFrame} suffix={`/ ${preview ? Math.max(0, timeline.frame_count - 1) : "?"}`} /></Col>
              </Row>
              {loadingEpisode && <Progress percent={60} status="active" showInfo={false} />}
              {preview && <Descriptions size="small" column={2} className="episode-meta">
                <Descriptions.Item label="Instruction" span={2}>{preview.episode.instruction || "未提供"}</Descriptions.Item>
                <Descriptions.Item label="时长">{(preview.episode.duration || 0).toFixed(2)} s</Descriptions.Item>
                <Descriptions.Item label="时间戳">{currentTime.toFixed(3)} s</Descriptions.Item>
              </Descriptions>}
            </Card>
            {preview && <Card title="多相机同步视频" className="section-card">
              <Space className="playback-state" wrap>
                <Tag color={playbackStatus === "error" ? "red" : playbackStatus === "playing" ? "green" : ["buffering", "seeking", "stalled"].includes(playbackStatus) ? "blue" : "default"}>
                  {playbackStatusLabels[playbackStatus]}
                </Tag>
                <Tag color={useOriginalVideo ? "orange" : readyProxyVideos ? "green" : "blue"}>
                  {useOriginalVideo ? "原始分片" : readyProxyVideos ? "Episode H.264 代理" : "代理生成中"}
                </Tag>
                <Button size="small" disabled={useOriginalVideo && !readyProxyVideos}
                  onClick={() => setUseOriginalVideo((value) => !value)}>
                  {useOriginalVideo ? "切换到代理视频" : "查看原始视频"}
                </Button>
                {(playbackStatus === "buffering" || playbackStatus === "seeking") && <span>正在等待所有相机就绪，请勿连续点击播放。</span>}
                {playbackStatus === "stalled" && <span>视频分片读取暂时停滞，恢复供数后会自动继续。</span>}
              </Space>
              {playbackError && <Alert type="error" showIcon message={playbackError} closable onClose={() => setPlaybackError(undefined)} />}
              {!useOriginalVideo && !readyProxyVideos && !proxyError && <Alert type="info" showIcon
                message="正在生成该 Episode 的 H.264 代理视频"
                description="首次打开需要短暂转码；完成后将从 0 秒直接播放，后续访问复用缓存。" />}
              {proxyError && !useOriginalVideo && <Alert type="error" showIcon
                message="代理视频生成失败" description={proxyError}
                action={<Button size="small" onClick={() => setUseOriginalVideo(true)}>使用原始视频</Button>} />}
              {!useOriginalVideo && proxyJob && ["queued", "generating"].includes(proxyJob.status)
                && <Progress percent={proxyJob.status === "generating" ? 65 : 15} status="active" showInfo={false} />}
              <Collapse className="video-metadata-collapse" items={[{
                key: "video-metadata",
                label: `视频技术信息 · ${technicalVideos.length} 个相机（点击展开）`,
                children: <Space direction="vertical" size={10} style={{ width: "100%" }}>
                  <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 4 }}>
                    <Descriptions.Item label="Episode 分辨率基准">各相机独立记录</Descriptions.Item>
                    <Descriptions.Item label="Episode FPS">{fixed(timeline.fps, 2)}</Descriptions.Item>
                    <Descriptions.Item label="Episode 帧数">{timeline.frame_count}</Descriptions.Item>
                    <Descriptions.Item label="Episode 时长">{fixed(timeline.duration, 3, " s")}</Descriptions.Item>
                  </Descriptions>
                  {technicalVideos.map((video) => {
                    const observed = browserVideoMetadata[video.camera];
                    const width = observed?.width || numeric(video.width);
                    const height = observed?.height || numeric(video.height);
                    const sourceWindow = videoWindow(video, timeline.duration);
                    return <Card key={video.camera} size="small" title={video.camera}
                      extra={<Tag color={video.is_proxy ? "green" : "blue"}>{video.is_proxy ? "Episode H.264 代理" : "原始 MP4 分片"}</Tag>}>
                      <Descriptions size="small" column={{ xs: 1, sm: 2, lg: 3 }}>
                        <Descriptions.Item label="分辨率">{width && height ? `${width} × ${height}` : "—"}</Descriptions.Item>
                        <Descriptions.Item label="FPS">{fixed(video.fps ?? timeline.fps, 2)}</Descriptions.Item>
                        <Descriptions.Item label="编码">{video.codec || (video.is_proxy ? "h264" : "—")}</Descriptions.Item>
                        <Descriptions.Item label="像素格式">{video.pixel_format || "—"}</Descriptions.Item>
                        <Descriptions.Item label="通道/音频">{video.channels ?? "—"} channels · {video.has_audio ? "有音频" : "无音频"}</Descriptions.Item>
                        <Descriptions.Item label="当前媒体时长">{observed && Number.isFinite(observed.duration) ? `${observed.duration.toFixed(3)} s` : "加载视频后显示"}</Descriptions.Item>
                        <Descriptions.Item label="Episode 时间窗">{sourceWindow.start.toFixed(3)}–{sourceWindow.end.toFixed(3)} s</Descriptions.Item>
                        <Descriptions.Item label="物理分片帧数">{video.physical_frames ?? "未索引"}</Descriptions.Item>
                        <Descriptions.Item label="源文件大小">{formatBytes(video.source_bytes)}</Descriptions.Item>
                        <Descriptions.Item label="分片编号">chunk-{String(video.chunk_index ?? "?").padStart(3, "0")} / file-{String(video.file_index ?? "?").padStart(3, "0")}</Descriptions.Item>
                        <Descriptions.Item label="完整性">{video.integrity_status || "—"} · {video.file_integrity_status || "not_indexed"}</Descriptions.Item>
                        <Descriptions.Item label="源文件" span={3}><code className="video-source-path">{video.relative_path}</code></Descriptions.Item>
                      </Descriptions>
                    </Card>;
                  })}
                </Space>,
              }]} />
              <Row gutter={[12, 12]}>{activeVideos.map((video) => <Col xs={24} md={12} key={video.camera}>
                <div className="camera-title">{video.camera} · {video.is_proxy
                  ? `Episode 代理 · 0.00–${timeline.duration.toFixed(2)} s`
                  : `file-${String(video.file_index ?? "?").padStart(3, "0")} · 源时间窗 ${videoWindow(video, timeline.duration).start.toFixed(2)}–${videoWindow(video, timeline.duration).end.toFixed(2)} s`}</div>
                {video.integrity_status === "duration_mismatch" && <Alert type="warning" showIcon message="该相机视频窗口长度与 Episode 长度不一致" />}
                <video className="episode-video" playsInline muted preload="metadata" src={video.url} data-camera={video.camera}
                  onClick={togglePlayback} ref={(element) => { videoRefs.current[video.camera] = element; }}
                  onLoadedMetadata={(event) => {
                    setBrowserVideoMetadata((current) => ({ ...current, [video.camera]: {
                      width: event.currentTarget.videoWidth,
                      height: event.currentTarget.videoHeight,
                      duration: event.currentTarget.duration,
                    } }));
                    const target = videoWindow(video, timeline.duration).start;
                    if (Math.abs(event.currentTarget.currentTime - target) > 0.1) event.currentTarget.currentTime = target;
                  }}
                  onWaiting={() => { if (isPlaying) setPlaybackStatus("buffering"); }}
                  onStalled={() => { if (isPlaying) setPlaybackStatus("stalled"); }}
                  onCanPlay={() => { if (isPlaying) setPlaybackStatus("playing"); }}
                  onError={(event) => {
                    const errorText = mediaErrorText(event.currentTarget);
                    pausePlayback("error");
                    setPlaybackError(errorText);
                  }} />
              </Col>)}</Row>
              <div className="video-controls">
                <Space>
                  <Button type="primary" onClick={togglePlayback} disabled={!activeVideos.length} danger={playbackStatus === "error"}>
                    {isPlaying || playbackStatus === "seeking" || playbackStatus === "buffering" ? "暂停" : "播放"}
                  </Button>
                  <Button disabled={!activeVideos.length} onClick={() => seek(currentTime - 1 / fps)}>上一帧</Button>
                  <Button disabled={!activeVideos.length} onClick={() => seek(currentTime + 1 / fps)}>下一帧</Button>
                </Space>
                <span>Episode {currentTime.toFixed(2)} / {timeline.duration.toFixed(2)} s · Frame {currentFrame}</span>
                <input aria-label="Episode 时间轴" type="range" min={0} max={timeline.duration || 1} step={1 / fps}
                  disabled={!activeVideos.length} value={Math.min(currentTime, timeline.duration || 1)} onChange={(event) => seek(Number(event.target.value))} />
              </div>
            </Card>}
            <Card title="state / action 曲线" className="section-card" extra={<Segmented
              value={seriesView}
              options={seriesViewOptions}
              onChange={(value) => setSeriesView(value as SeriesView)}
            />}>
              {loadingSeries && <Progress percent={60} status="active" showInfo={false} />}
              {seriesView === "diff" && <Alert type="info" showIcon message="差值 = 修复后 − 原始；未修复位置为 0" />}
              {seriesView === "valid" && <Alert type="info" showIcon message="无效帧显示为曲线断点；原始 Parquet 和视频未被修改" />}
              {!series.length && <Alert type="info" showIcon message="暂无曲线数据；请执行标准扫描，或确认该数据集包含 Parquet state/action 字段。" />}
              <Row gutter={[12, 12]}>
                <Col xs={24} xl={12}><Curve title="State" rows={series} field="observation.state" elementNames={stateElementNames} fps={fps} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
                <Col xs={24} xl={12}><Curve title="Action" rows={series} field="action" elementNames={actionElementNames} fps={fps} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
              </Row>
            </Card>
            {preview && <StageVisualizations stages={preview.stage_results} timeline={timeline} onSeek={seek}
              onLoadStage={(stageId) => void loadStageDetail(stageId)} loadingStages={loadingStageDetails} />}
          </div>}
        </Col>
      </Row>
    </Layout.Content>
  </Layout>;
}

createRoot(document.getElementById("root")!).render(<App />);
