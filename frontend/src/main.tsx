import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Layout, List, Card, Button, Tag, Statistic, Row, Col, Select, message } from "antd";
import "./style.css";

type Dataset = { uid:string; episodes:number; frames:number; duration:number; bytes:number; cameras:string[]; codebase_version:string };
type Task = { task_id:string; status:string; progress:number; stage_id:number };

const stages = Array.from({length:8}, (_, i) => ({value:i+1, label:`Stage ${i+1}`}));
function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]); const [selected, setSelected] = useState<Dataset>();
  const [stage, setStage] = useState(1); const [task, setTask] = useState<Task>();
  const refresh = () => fetch("/api/datasets").then(r=>r.json()).then(setDatasets).catch(()=>message.error("后端未启动"));
  useEffect(refresh, []);
  const run = async () => { if (!selected) return; const r=await fetch("/api/pipelines/run", {method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({dataset_uid:selected.uid,stage_id:stage})}); setTask(await r.json()); };
  useEffect(()=>{ if (!task || ["succeeded","failed"].includes(task.status)) return; const id=setInterval(()=>fetch(`/api/tasks/${task.task_id}`).then(r=>r.json()).then(setTask),1000); return()=>clearInterval(id); },[task]);
  return <Layout className="shell"><Layout.Header><h1>VLA Data Governance Workbench</h1><Button onClick={refresh}>刷新目录</Button></Layout.Header><Layout.Content><Row gutter={16}><Col span={8}><Card title="数据集目录"><List dataSource={datasets} renderItem={d=><List.Item onClick={()=>setSelected(d)} className={selected?.uid===d.uid?"selected":""}><List.Item.Meta title={d.uid} description={`${d.episodes} episodes · ${d.frames} frames`} /><Tag color="green">{d.codebase_version}</Tag></List.Item>} /></Card></Col><Col span={16}><Card title={selected ? `Episode / ${selected.uid}` : "选择数据集"}>{selected ? <><Row gutter={16}><Col><Statistic title="Episodes" value={selected.episodes}/></Col><Col><Statistic title="Frames" value={selected.frames}/></Col><Col><Statistic title="Cameras" value={selected.cameras.length}/></Col></Row><p>相机：{selected.cameras.join(", ") || "未发现"}</p><div className="timeline">异常时间轴 / state-action 曲线将在选择 episode 后显示</div><Select value={stage} options={stages} onChange={setStage} /><Button type="primary" onClick={run}>运行 Stage</Button>{task && <p>任务 {task.task_id}: <Tag>{task.status}</Tag> {Math.round(task.progress*100)}%</p>}</> : "请先扫描或选择数据集"}</Card></Col></Row></Layout.Content></Layout>;
}
createRoot(document.getElementById("root")!).render(<App />);
