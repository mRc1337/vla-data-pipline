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
  cameras: string[]; codebase_version: string; schema?: Record<string, unknown>;
};
type Episode = {
  dataset_uid: string; episode_index: number; frames: number; duration: number;
  instruction?: string | null; metadata_json?: string;
};
type VideoRef = {
  camera: string; relative_path: string; url: string;
  timestamp_start?: number; timestamp_end?: number;
};
type StageResult = {
  stage_id: number; run_id: string; stage?: string; detector_version?: string;
  summary?: Record<string, unknown>; records?: Array<Record<string, unknown>>;
};
type Preview = { dataset: Dataset; episode: Episode; videos: VideoRef[]; stage_results: StageResult[] };
type SeriesRow = Record<string, unknown>;
type Task = { task_id: string; status: string; progress: number; stage_id: number; error?: string };
type ScanTask = { scan_id: string; status: string; current: number; total: number; datasets: number; skipped: number; eta_seconds: number | null };

const stages = Array.from({ length: 8 }, (_, i) => ({ value: i + 1, label: `Stage ${i + 1}` }));

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

function stageIntervals(results: StageResult[], fps: number): Array<[{ xAxis: number }, { xAxis: number }]> {
  const intervals: Array<[{ xAxis: number }, { xAxis: number }]> = [];
  results.forEach((stage) => (stage.records || []).forEach((record) => {
    const status = String(record.status || record.label || "warning").toLowerCase();
    if (["pass", "ok", "clean"].includes(status)) return;
    const start = numeric(record.frame_start ?? record.start_frame ?? record.frame_index);
    const end = numeric(record.frame_end ?? record.end_frame ?? record.frame_index ?? start);
    if (start !== null && end !== null) intervals.push([{ xAxis: start / fps }, { xAxis: Math.max(end, start) / fps }]);
  }));
  return intervals;
}

function Curve({ title, rows, field, dimensions, fps, intervals = [] }: {
  title: string; rows: SeriesRow[]; field: string; dimensions?: number[]; fps: number;
  intervals?: Array<[{ xAxis: number }, { xAxis: number }]>;
}) {
  const first = rows.find((row) => vector(row[field]).length);
  const width = first ? vector(first[field]).length : 0;
  const indexes = dimensions || Array.from({ length: width }, (_, i) => i);
  const option = useMemo(() => ({
    animation: false,
    title: { text: title, left: 8, textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis" },
    legend: { type: "scroll", top: 24 },
    grid: { left: 48, right: 18, top: 58, bottom: 32 },
    xAxis: { type: "value", name: "s", min: 0 },
    yAxis: { type: "value" },
    series: indexes.map((dimension) => ({
      name: `${field}[${dimension}]`, type: "line", showSymbol: false,
      data: rows.map((row, index) => {
        const values = vector(row[field]);
        const timestamp = numeric(row.timestamp) ?? index / fps;
        return [timestamp, values[dimension] ?? null];
      }),
    })),
    markArea: intervals.length ? { itemStyle: { color: "rgba(245, 63, 63, .16)" }, data: intervals } : undefined,
  }), [field, fps, indexes, intervals, rows, title]);
  if (!rows.length || !width) return <Card size="small" title={title}><Alert type="info" showIcon message="该字段暂无可绘制数据，请先运行 standard 扫描或检查 Parquet schema。" /></Card>;
  return <ReactECharts option={option} style={{ height: 270 }} notMerge lazyUpdate />;
}

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<Dataset>();
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episodeIndex, setEpisodeIndex] = useState<number>();
  const [preview, setPreview] = useState<Preview>();
  const [series, setSeries] = useState<SeriesRow[]>([]);
  const [scanTask, setScanTask] = useState<ScanTask>();
  const [stage, setStage] = useState(1);
  const [task, setTask] = useState<Task>();
  const [loadingEpisode, setLoadingEpisode] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const syncing = useRef(false);
  const playbackSyncing = useRef(false);

  const refresh = () => fetch("/api/datasets").then((response) => response.json()).then(setDatasets)
    .catch(() => message.error("后端未启动"));

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
    if (!selected) { setEpisodes([]); setEpisodeIndex(undefined); return; }
    setPreview(undefined); setSeries([]); setEpisodeIndex(undefined);
    fetch(`/api/datasets/${encodeURIComponent(selected.uid)}/episodes`)
      .then((response) => response.json()).then((items: Episode[]) => {
        setEpisodes(items);
        if (items.length) setEpisodeIndex(items[0].episode_index);
      }).catch(() => message.error("Episode 列表读取失败"));
  }, [selected]);

  useEffect(() => {
    if (!selected || episodeIndex === undefined) return;
    let cancelled = false;
    setLoadingEpisode(true); setCurrentTime(0); setCurrentFrame(0);
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

  const fps = Number(((preview?.dataset.schema?.timestamp as { fps?: number } | undefined)?.fps)
    || ((selected?.schema?.timestamp as { fps?: number } | undefined)?.fps) || 20);
  const intervals = useMemo(() => stageIntervals(preview?.stage_results || [], fps), [fps, preview]);
  const episodeOptions = useMemo(() => episodes.map((item) => ({
    value: item.episode_index,
    label: `Episode ${item.episode_index} · ${item.frames || "?"} frames${item.instruction ? ` · ${item.instruction.slice(0, 72)}` : ""}`,
  })), [episodes]);

  const syncTime = (source: HTMLVideoElement) => {
    if (syncing.current) return;
    syncing.current = true;
    Object.values(videoRefs.current).forEach((video) => {
      if (video && video !== source && Math.abs(video.currentTime - source.currentTime) > 0.05) video.currentTime = source.currentTime;
    });
    syncing.current = false;
    const time = source.currentTime || 0;
    setCurrentTime(time); setCurrentFrame(Math.round(time * fps));
  };
  const syncPlayback = (source: HTMLVideoElement, playing: boolean) => {
    if (playbackSyncing.current) return;
    playbackSyncing.current = true;
    Object.values(videoRefs.current).forEach((video) => {
      if (!video || video === source) return;
      video.currentTime = source.currentTime;
      if (playing) void video.play().catch(() => undefined); else video.pause();
    });
    window.setTimeout(() => { playbackSyncing.current = false; }, 0);
  };
  const seek = (time: number) => {
    Object.values(videoRefs.current).forEach((video) => { if (video) video.currentTime = time; });
    setCurrentTime(time); setCurrentFrame(Math.round(time * fps));
  };

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
          <Card title="数据集目录" extra={<Tag>{datasets.length}</Tag>}>
            <List dataSource={datasets} className="dataset-list" renderItem={(dataset) => <List.Item onClick={() => setSelected(dataset)} className={selected?.uid === dataset.uid ? "selected" : "dataset-item"}>
              <List.Item.Meta title={dataset.uid} description={`${dataset.episodes} episodes · ${dataset.frames} frames`} />
              <Tag color="green">{dataset.codebase_version}</Tag>
            </List.Item>} />
          </Card>
        </Col>
        <Col xs={24} lg={17}>
          {!selected && <Card><Alert type="info" showIcon message="请先从左侧选择数据集" /></Card>}
          {selected && <>
            <Card title={`Episode 预览 / ${selected.uid}`} extra={<Space>
              <Select showSearch virtual optionFilterProp="label" placeholder="选择 Episode" value={episodeIndex} options={episodeOptions} onChange={setEpisodeIndex} style={{ width: 360 }} />
              {episodeIndex !== undefined && <InputNumber min={0} max={Math.max(0, selected.episodes - 1)} value={episodeIndex} onChange={(value) => value !== null && setEpisodeIndex(value)} />}
            </Space>}>
              <Row gutter={16}>
                <Col><Statistic title="Episodes" value={selected.episodes} /></Col>
                <Col><Statistic title="Frames" value={selected.frames} /></Col>
                <Col><Statistic title="Cameras" value={preview?.videos.length || selected.cameras.length} /></Col>
                <Col><Statistic title="当前帧" value={currentFrame} suffix={`/ ${preview?.episode.frames || "?"}`} /></Col>
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
                <div className="camera-title">{video.camera}</div>
                <video className="episode-video" controls preload="metadata" src={video.url} ref={(element) => { videoRefs.current[video.camera] = element; }}
                  onTimeUpdate={(event) => syncTime(event.currentTarget)} onPlay={(event) => syncPlayback(event.currentTarget, true)} onPause={(event) => syncPlayback(event.currentTarget, false)} />
              </Col>)}</Row>
              <div className="video-controls"><span>当前时间 {currentTime.toFixed(2)} s</span><input type="range" min={0} max={preview.episode.duration || 1} step={0.01} value={Math.min(currentTime, preview.episode.duration || 1)} onChange={(event) => seek(Number(event.target.value))} /></div>
            </Card>}
            {preview && <Card title="异常区间与 Stage 对比" className="section-card">
              {intervals.length > 0 && <Alert type="warning" showIcon message={`发现 ${intervals.length} 个异常区间，已覆盖到曲线图中`} />}
              <Table size="small" pagination={false} rowKey={(row) => `${row.stage_id}-${row.run_id}`} columns={stageColumns} dataSource={preview.stage_results} locale={{ emptyText: "暂无 Stage 产物；可先运行 Stage 或扫描产物目录" }} />
            </Card>}
            <Card title="state / action 曲线" className="section-card">
              {!series.length && <Alert type="info" showIcon message="暂无曲线数据；请执行标准扫描，或确认该数据集包含 Parquet state/action 字段。" />}
              <Row gutter={[12, 12]}>
                <Col xs={24} xl={12}><Curve title="State" rows={series} field="observation.state" fps={fps} intervals={intervals} /></Col>
                <Col xs={24} xl={12}><Curve title="Action" rows={series} field="action" fps={fps} intervals={intervals} /></Col>
                <Col xs={24} xl={12}><Curve title="末端位置（state 0–2）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[0, 1, 2]} intervals={intervals} /></Col>
                <Col xs={24} xl={12}><Curve title="姿态（state 3–5）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[3, 4, 5]} intervals={intervals} /></Col>
                <Col xs={24} xl={12}><Curve title="关节 / 夹爪（state 6–7）" rows={selectedStateRows} field="observation.state" fps={fps} dimensions={[6, 7]} intervals={intervals} /></Col>
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
