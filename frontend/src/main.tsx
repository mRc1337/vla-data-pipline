import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  Alert, Button, Card, Collapse, Descriptions, InputNumber, Layout,
  List, Progress, Row, Col, Select, Space, Statistic, Table, Tag, message,
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
};
type StageResult = {
  stage_id: number; run_id: string; stage?: string; detector_version?: string;
  coordinate_system?: string;
  summary?: Record<string, unknown>; records?: Array<Record<string, unknown>>;
};
type EpisodeTimeline = {
  coordinate_system: string; frame_count: number; fps: number; duration: number; dataset_from_index: number;
};
type Preview = { dataset: Dataset; episode: Episode; timeline?: EpisodeTimeline; videos: VideoRef[]; stage_results: StageResult[] };
type SeriesRow = Record<string, unknown>;
type Task = { task_id: string; status: string; progress: number; stage_id: number; error?: string };
type ScanTask = { scan_id: string; status: string; current: number; total: number; datasets: number; skipped: number; eta_seconds: number | null };

const stages = Array.from({ length: 8 }, (_, i) => ({ value: i + 1, label: `Stage ${i + 1}` }));

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
  if (Array.isArray(value)) return value.length ? numeric(value[0]) : null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

function vector(value: unknown): number[] {
  if (Array.isArray(value)) return value.flatMap(vector).filter(Number.isFinite);
  const result = numeric(value);
  return result === null ? [] : [result];
}

function stageIntervals(results: StageResult[], timeline: EpisodeTimeline): Array<[{ xAxis: number }, { xAxis: number }]> {
  const intervals: Array<[{ xAxis: number }, { xAxis: number }]> = [];
  results.forEach((stage) => (stage.records || []).forEach((record) => {
    const status = String(record.status || record.label || "warning").toLowerCase();
    if (["pass", "ok", "clean"].includes(status)) return;
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
    if (["dataset_frame", "global_frame"].includes(coordinate)) {
      start -= timeline.dataset_from_index;
      end -= timeline.dataset_from_index;
    }
    start = Math.min(Math.max(start, 0), Math.max(0, timeline.frame_count - 1));
    end = Math.min(Math.max(end, start), Math.max(0, timeline.frame_count - 1));
    intervals.push([{ xAxis: start / timeline.fps }, { xAxis: end / timeline.fps }]);
  }));
  return intervals;
}

function videoWindow(video: VideoRef, episodeDuration: number): { start: number; end: number; duration: number } {
  const start = Math.max(0, numeric(video.source_start ?? video.timestamp_start) ?? 0);
  const metadataEnd = numeric(video.source_end ?? video.timestamp_end);
  const end = metadataEnd !== null && metadataEnd > start
    ? metadataEnd
    : start + Math.max(0, episodeDuration);
  return { start, end: Math.max(start, end), duration: Math.max(0, end - start) };
}

function Curve({ title, rows, field, dimensions, fps, intervals = [], playhead, onSeek }: {
  title: string; rows: SeriesRow[]; field: string; dimensions?: number[]; fps: number;
  intervals?: Array<[{ xAxis: number }, { xAxis: number }]>;
  playhead?: number;
  onSeek?: (time: number) => void;
}) {
  const first = rows.find((row) => vector(row[field]).length);
  const width = first ? vector(first[field]).length : 0;
  const dimensionKey = dimensions?.join(",") || `all:${width}`;
  const indexes = useMemo(
    () => dimensions || Array.from({ length: width }, (_, i) => i),
    [dimensionKey, width],
  );
  const chartSeries = useMemo(() => indexes.map((dimension) => ({
    dimension,
    data: rows.map((row, index) => {
      const values = vector(row[field]);
      const timestamp = numeric(row.episode_time ?? row.timestamp) ?? index / fps;
      return [timestamp, values[dimension] ?? null];
    }),
  })), [field, fps, indexes, rows]);
  const option = useMemo(() => ({
    animation: false,
    title: { text: title, left: 8, textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis" },
    legend: { type: "scroll", top: 24 },
    grid: { left: 48, right: 18, top: 58, bottom: 32 },
    xAxis: { type: "value", name: "s", min: 0 },
    yAxis: { type: "value" },
    series: chartSeries.map(({ dimension, data }, seriesIndex) => ({
      name: `${field}[${dimension}]`, type: "line", showSymbol: false,
      data,
      markArea: seriesIndex === 0 && intervals.length ? {
        silent: true, itemStyle: { color: "rgba(245, 63, 63, .16)" }, data: intervals,
      } : undefined,
      markLine: seriesIndex === 0 && playhead !== undefined ? {
        silent: true, symbol: "none", lineStyle: { color: "#1677ff", width: 1.5 },
        label: { show: false }, data: [{ xAxis: playhead }],
      } : undefined,
    })),
  }), [chartSeries, field, intervals, playhead, title]);
  if (!rows.length || !width) return <Card size="small" title={title}><Alert type="info" showIcon message="该字段暂无可绘制数据，请先运行 standard 扫描或检查 Parquet schema。" /></Card>;
  const onEvents = onSeek ? {
    click: (params: { value?: unknown }) => {
      const value = Array.isArray(params.value) ? Number(params.value[0]) : Number(params.value);
      if (Number.isFinite(value)) onSeek(value);
    },
  } : undefined;
  return <ReactECharts option={option} onEvents={onEvents} style={{ height: 270, cursor: onSeek ? "crosshair" : undefined }} notMerge lazyUpdate />;
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
  const [series, setSeries] = useState<SeriesRow[]>([]);
  const [scanTask, setScanTask] = useState<ScanTask>();
  const [stage, setStage] = useState(1);
  const [task, setTask] = useState<Task>();
  const [loadingEpisode, setLoadingEpisode] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const syncing = useRef(false);

  const refresh = () => fetch("/api/datasets").then((response) => response.json()).then(setDatasets)
    .catch(() => message.error("后端未启动"));

  const chooseCollection = (collection: DatasetCollection) => {
    setSelectedCollectionId(collection.id);
    setSelected(collection.datasets.length === 1 ? collection.datasets[0] : undefined);
    setTasks([]); setTaskIndex(undefined); setEpisodes([]); setEpisodeIndex(undefined);
    setPreview(undefined); setSeries([]);
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
    if (!selected) { setTasks([]); setTaskIndex(undefined); return; }
    setTasks([]); setTaskIndex(undefined); setEpisodes([]); setEpisodeIndex(undefined);
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/tasks`)
      .then((response) => response.json()).then(setTasks)
      .catch(() => message.error("Task 列表读取失败"));
  }, [selected]);

  useEffect(() => {
    if (!selected || taskIndex === undefined) { setEpisodes([]); setEpisodeIndex(undefined); return; }
    setPreview(undefined); setSeries([]); setEpisodeIndex(undefined);
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes?task_index=${taskIndex}`)
      .then((response) => response.json()).then((items: Episode[]) => {
        setEpisodes(items);
        if (items.length) setEpisodeIndex(items[0].episode_index);
      }).catch(() => message.error("Episode 列表读取失败"));
  }, [selected, taskIndex]);

  useEffect(() => {
    if (!selected || episodeIndex === undefined) return;
    let cancelled = false;
    Object.values(videoRefs.current).forEach((video) => video?.pause());
    setIsPlaying(false); setLoadingEpisode(true); setCurrentTime(0); setCurrentFrame(0);
    Promise.all([
      fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes/${episodeIndex}/preview`).then((response) => response.json()),
      fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes/${episodeIndex}/series?fields=timestamp,frame_index,observation.state,action&limit=5000`).then((response) => response.json()),
    ]).then(([episodePreview, episodeSeries]) => {
      if (cancelled) return;
      setPreview(episodePreview); setSeries(episodeSeries.rows || []);
    }).catch(() => { if (!cancelled) message.error("Episode 预览读取失败"); })
      .finally(() => { if (!cancelled) setLoadingEpisode(false); });
    return () => { cancelled = true; };
  }, [episodeIndex, selected]);

  const run = async () => {
    if (!selected) return;
    const response = await fetch("/api/pipelines/run", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset_uid: selected.uid, stage_id: stage }),
    });
    setTask(await response.json());
  };

  useEffect(() => {
    if (!task || ["succeeded", "failed", "cancelled"].includes(task.status)) return undefined;
    const id = setInterval(() => fetch(`/api/tasks/${task.task_id}`).then((response) => response.json()).then((next: Task) => {
      setTask(next);
      if (next.status === "succeeded" && selected && episodeIndex !== undefined) {
        void fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes/${episodeIndex}/preview`)
          .then((response) => response.json()).then(setPreview);
      }
    }), 1000);
    return () => clearInterval(id);
  }, [episodeIndex, selected, task]);

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

  useEffect(() => {
    if (!preview) return;
    setIsPlaying(false); setCurrentTime(0); setCurrentFrame(0);
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      video.pause();
      const meta = preview.videos.find((item) => item.camera === video.dataset.camera);
      if (meta && video.readyState >= 1) video.currentTime = videoWindow(meta, timeline.duration).start;
    });
  }, [preview, timeline.duration]);
  const episodeOptions = useMemo(() => episodes.map((item) => ({
    value: item.episode_index,
    label: `Episode ${item.episode_index} · ${item.frames || "?"} frames${item.instruction ? ` · ${item.instruction.slice(0, 72)}` : ""}`,
  })), [episodes]);
  const taskOptions = useMemo(() => tasks.map((item) => ({
    value: item.task_index,
    label: `Task ${item.task_index} · ${item.episodes} episodes · ${item.name}`,
  })), [tasks]);

  const syncTime = (source: HTMLVideoElement) => {
    if (syncing.current) return;
    const sourceMeta = preview?.videos.find((item) => item.camera === source.dataset.camera);
    if (!sourceMeta) return;
    const sourceWindow = videoWindow(sourceMeta, timeline.duration);
    if (source.currentTime < sourceWindow.start) source.currentTime = sourceWindow.start;
    if (source.currentTime > sourceWindow.end) {
      source.currentTime = sourceWindow.end;
      Object.values(videoRefs.current).forEach((video) => video?.pause());
      setIsPlaying(false);
    }
    const relativeTime = Math.min(Math.max(source.currentTime - sourceWindow.start, 0), timeline.duration);
    syncing.current = true;
    Object.values(videoRefs.current).forEach((video) => {
      if (!video || video === source) return;
      const meta = preview?.videos.find((item) => item.camera === video.dataset.camera);
      if (!meta) return;
      const window = videoWindow(meta, timeline.duration);
      const target = Math.min(window.end, window.start + relativeTime);
      if (Math.abs(video.currentTime - target) > 0.05) video.currentTime = target;
    });
    syncing.current = false;
    setCurrentTime(relativeTime);
    setCurrentFrame(Math.min(Math.max(0, timeline.frame_count - 1), Math.floor(relativeTime * fps + 1e-6)));
  };
  const seek = (time: number) => {
    const duration = timeline.duration;
    const relativeTime = Math.min(Math.max(time, 0), duration);
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) return;
      const meta = preview?.videos.find((item) => item.camera === video.dataset.camera);
      if (!meta) return;
      const videoDuration = videoWindow(meta, duration);
      video.currentTime = Math.min(videoDuration.end, videoDuration.start + relativeTime);
    });
    setCurrentTime(relativeTime);
    setCurrentFrame(Math.min(Math.max(0, timeline.frame_count - 1), Math.floor(relativeTime * fps + 1e-6)));
  };

  const togglePlayback = () => {
    if (isPlaying) {
      Object.values(videoRefs.current).forEach((video) => video?.pause());
      setIsPlaying(false);
      return;
    }
    if (currentTime >= timeline.duration - 1 / fps) seek(0);
    const videos = Object.values(videoRefs.current).filter((video): video is HTMLVideoElement => Boolean(video));
    void Promise.allSettled(videos.map((video) => video.play())).then((results) => {
      const allStarted = videos.length > 0 && results.every((result) => result.status === "fulfilled");
      if (!allStarted) Object.values(videoRefs.current).forEach((video) => video?.pause());
      setIsPlaying(allStarted);
    });
  };

  useEffect(() => {
    if (!isPlaying || !preview?.videos.length) return undefined;
    const primary = videoRefs.current[preview.videos[0].camera];
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
  }, [isPlaying, preview, timeline.duration]);

  const selectedStateRows = series.filter((row) => row["observation.state"] !== undefined);
  const stageColumns = [
    { title: "Stage", dataIndex: "stage_id", key: "stage_id", render: (value: number) => <Tag color="blue">Stage {value}</Tag> },
    { title: "运行", dataIndex: "run_id", key: "run_id", ellipsis: true },
    { title: "Episode 记录", key: "records", render: (_: unknown, row: StageResult) => row.records?.length || 0 },
    { title: "汇总", key: "summary", render: (_: unknown, row: StageResult) => Object.entries(row.summary || {}).map(([key, value]) => `${key}: ${String(value)}`).join(" · ") || "—" },
  ];

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
              onChange={(uid) => setSelected(selectedCollection.datasets.find((item) => item.uid === uid))} />
          </Card>}
          {selected && <>
            <Card title={`Episode 预览 / ${selectedCollection?.name || selected.uid}`} extra={<Space>
              {selectedCollection && selectedCollection.datasets.length > 1 && <Select showSearch virtual optionFilterProp="label" value={selected.uid}
                placeholder="选择子数据集" style={{ width: 300 }} options={selectedCollection.datasets.map((item) => ({ value: item.uid, label: `${item.uid} · ${item.episodes} episodes` }))}
                onChange={(uid) => { setSelected(selectedCollection.datasets.find((item) => item.uid === uid)); setTaskIndex(undefined); }} />}
              <Select allowClear showSearch virtual optionFilterProp="label" placeholder="选择 Task" value={taskIndex} options={taskOptions} onChange={setTaskIndex} style={{ width: 380 }} />
              <Select disabled={taskIndex === undefined} showSearch virtual optionFilterProp="label" placeholder={taskIndex === undefined ? "先选择 Task" : "选择 Episode"} value={episodeIndex} options={episodeOptions} onChange={setEpisodeIndex} style={{ width: 360 }} />
              {episodeIndex !== undefined && <InputNumber disabled={taskIndex === undefined} min={0} max={Math.max(0, selected.episodes - 1)} value={episodeIndex} onChange={(value) => value !== null && setEpisodeIndex(value)} />}
            </Space>}>
              <Row gutter={16}>
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
              <Row gutter={[12, 12]}>{preview.videos.map((video) => <Col xs={24} md={12} key={video.camera}>
                <div className="camera-title">{video.camera} · file-{String(video.file_index ?? "?").padStart(3, "0")} · 源时间窗 {videoWindow(video, timeline.duration).start.toFixed(2)}–{videoWindow(video, timeline.duration).end.toFixed(2)} s</div>
                {video.integrity_status === "duration_mismatch" && <Alert type="warning" showIcon message="该相机视频窗口长度与 Episode 长度不一致" />}
                <video className="episode-video" playsInline muted preload="metadata" src={video.url} data-camera={video.camera}
                  onClick={togglePlayback} ref={(element) => { videoRefs.current[video.camera] = element; }}
                  onLoadedMetadata={(event) => { event.currentTarget.currentTime = videoWindow(video, timeline.duration).start; }} />
              </Col>)}</Row>
              <div className="video-controls">
                <Space>
                  <Button type="primary" onClick={togglePlayback}>{isPlaying ? "暂停" : "播放"}</Button>
                  <Button onClick={() => seek(currentTime - 1 / fps)}>上一帧</Button>
                  <Button onClick={() => seek(currentTime + 1 / fps)}>下一帧</Button>
                </Space>
                <span>Episode {currentTime.toFixed(2)} / {timeline.duration.toFixed(2)} s · Frame {currentFrame}</span>
                <input aria-label="Episode 时间轴" type="range" min={0} max={timeline.duration || 1} step={1 / fps}
                  value={Math.min(currentTime, timeline.duration || 1)} onChange={(event) => seek(Number(event.target.value))} />
              </div>
            </Card>}
            {preview && <Card title="异常区间与 Stage 对比" className="section-card">
              {intervals.length > 0 && <Alert type="warning" showIcon message={`发现 ${intervals.length} 个异常区间，已覆盖到曲线图中`} />}
              <Table size="small" pagination={false} rowKey={(row) => `${row.stage_id}-${row.run_id}`} columns={stageColumns} dataSource={preview.stage_results} locale={{ emptyText: "暂无 Stage 产物；可先运行 Stage 或扫描产物目录" }} />
            </Card>}
            <Card title="state / action 曲线" className="section-card">
              {!series.length && <Alert type="info" showIcon message="暂无曲线数据；请执行标准扫描，或确认该数据集包含 Parquet state/action 字段。" />}
              <Row gutter={[12, 12]}>
                <Col xs={24} xl={12}><Curve title="State" rows={series} field="observation.state" fps={fps} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
                <Col xs={24} xl={12}><Curve title="Action" rows={series} field="action" fps={fps} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
                <Col xs={24} xl={12}><Curve title="末端位置（state 0–2）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[0, 1, 2]} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
                <Col xs={24} xl={12}><Curve title="姿态（state 3–5）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[3, 4, 5]} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
                <Col xs={24} xl={12}><Curve title="关节 / 夹爪（state 6–7）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[6, 7]} intervals={intervals} playhead={currentTime} onSeek={seek} /></Col>
              </Row>
            </Card>
            <Card title="运行治理 Stage" className="section-card">
              <Space><Select value={stage} options={stages} onChange={setStage} /><Button type="primary" onClick={() => void run()}>运行 Stage</Button></Space>
              {task && <p>任务 {task.task_id}: <Tag color={task.status === "succeeded" ? "green" : "blue"}>{task.status}</Tag> {Math.round(task.progress * 100)}% {task.error && <span>{task.error}</span>}</p>}
              {preview?.stage_results.length ? <Collapse className="stage-details" items={preview.stage_results.map((result) => ({ key: `${result.stage_id}-${result.run_id}`, label: `Stage ${result.stage_id} · ${result.run_id}`, children: <pre>{JSON.stringify(result.records || result.summary || {}, null, 2)}</pre> }))} /> : null}
            </Card>
          </>}
        </Col>
      </Row>
    </Layout.Content>
  </Layout>;
}

createRoot(document.getElementById("root")!).render(<App />);
