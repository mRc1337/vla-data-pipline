import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Layout, List, Card, Button, Tag, Statistic, Row, Col, Select, message } from "antd";
import "./style.css";

type Dataset = { uid:string; episodes:number; frames:number; duration:number; bytes:number; cameras:string[]; codebase_version:string };
type Task = { task_id:string; status:string; progress:number; stage_id:number };
type ScanTask = { scan_id:string; status:string; current:number; total:number; datasets:number; skipped:number; eta_seconds:number|null };

const stages = Array.from({length:8}, (_, i) => ({value:i+1, label:`Stage ${i+1}`}));
function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]); const [selected, setSelected] = useState<Dataset>(); const [scanTask, setScanTask] = useState<ScanTask>();
  const [stage, setStage] = useState(1); const [task, setTask] = useState<Task>();
  const refresh = () => fetch("/api/datasets").then(r=>r.json()).then(setDatasets).catch(()=>message.error("后端未启动"));
  const scan = async () => { try { const r=await fetch("/api/catalog/scan", {method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({mode:"quick"})}); if (!r.ok) throw new Error(); let job:ScanTask=await r.json(); setScanTask(job); while (!["succeeded","failed","cancelled"].includes(job.status)) { await new Promise(resolve=>setTimeout(resolve,500)); job=await fetch(`/api/catalog/scans/${job.scan_id}`).then(x=>x.json()); setScanTask(job); } if (job.status !== "succeeded") throw new Error(); await refresh(); message.success(`目录扫描完成：${job.datasets} 个数据集`); } catch { message.error("扫描失败，请检查后端和数据路径"); } };
  useEffect(() => { void refresh(); }, []);
  const run = async () => { if (!selected) return; const r=await fetch("/api/pipelines/run", {method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({dataset_uid:selected.uid,stage_id:stage})}); setTask(await r.json()); };
  useEffect(()=>{ if (!task || ["succeeded","failed"].includes(task.status)) return; const id=setInterval(()=>fetch(`/api/tasks/${task.task_id}`).then(r=>r.json()).then(setTask),1000); return()=>clearInterval(id); },[task]);
  return <Layout className="shell"><Layout.Header><h1>VLA Data Governance Workbench</h1><span>{scanTask && <Tag color={scanTask.status==="succeeded"?"green":"blue"}>扫描 {scanTask.status} {scanTask.total?`${scanTask.current}/${scanTask.total}`:""}</Tag>}<Button onClick={scan}>扫描并刷新目录</Button></span></Layout.Header><Layout.Content><Row gutter={16}><Col span={8}><Card title="数据集目录"><List dataSource={datasets} renderItem={d=><List.Item onClick={()=>setSelected(d)} className={selected?.uid===d.uid?"selected":""}><List.Item.Meta title={d.uid} description={`${d.episodes} episodes · ${d.frames} frames`} /><Tag color="green">{d.codebase_version}</Tag></List.Item>} /></Card></Col><Col span={16}><Card title={selected ? `Episode / ${selected.uid}` : "选择数据集"}>{selected ? <><Row gutter={16}><Col><Statistic title="Episodes" value={selected.episodes}/></Col><Col><Statistic title="Frames" value={selected.frames}/></Col><Col><Statistic title="Cameras" value={selected.cameras.length}/></Col></Row><p>相机：{selected.cameras.join(", ") || "未发现"}</p><div className="timeline">异常时间轴 / state-action 曲线将在选择 episode 后显示</div><Select value={stage} options={stages} onChange={setStage} /><Button type="primary" onClick={run}>运行 Stage</Button>{task && <p>任务 {task.task_id}: <Tag>{task.status}</Tag> {Math.round(task.progress*100)}%</p>}</> : "请先扫描或选择数据集"}</Card></Col></Row></Layout.Content></Layout>;
}
createRoot(document.getElementById("root")!).render(<App />);
