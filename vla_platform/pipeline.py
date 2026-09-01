from __future__ import annotations

import asyncio
import json
import os
import subprocess
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class CurationStage(Protocol):
    stage_id: int
    name: str
    version: str
    async def validate_input(self, dataset: dict[str, Any]) -> None: ...
    async def process_episode(self, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    async def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class StageResult:
    stage_id: int
    name: str
    version: str = "1.0"
    async def validate_input(self, dataset: dict[str, Any]) -> None:
        if dataset.get("codebase_version") not in (None, "unknown", "v3.0"):
            raise ValueError("stage input must be LeRobot v3.0")
    async def process_episode(self, episode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {"episode_index": episode.get("episode_index", 0), "status": "pass", "labels": []}
    async def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"episodes": len(results), "passed": sum(r.get("status") == "pass" for r in results)}


STAGE_NAMES = {
    1: "Sudden Change Detection", 2: "State-Action Trend Alignment", 3: "Extreme Value Detection",
    4: "Kinematic Consistency", 5: "Orientation Alignment", 6: "Instruction Consistency",
    7: "Video State Consistency", 8: "Video Quality Filtering",
}


def default_stages() -> dict[int, StageResult]:
    return {i: StageResult(i, name) for i, name in STAGE_NAMES.items()}


@dataclass
class Task:
    task_id: str
    dataset_uid: str
    stage_id: int
    status: str = "queued"
    progress: float = 0
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False


class PipelineRunner:
    def __init__(self, catalog: Any, output_root: str | Path):
        self.catalog, self.output_root = catalog, Path(output_root)
        self.tasks: dict[str, Task] = {}
        self.stages = default_stages()

    async def submit(self, dataset_uid: str, stage_id: int, config: dict[str, Any] | None = None) -> Task:
        if stage_id not in self.stages:
            raise ValueError(f"unknown stage {stage_id}")
        if not self.catalog.get_dataset(dataset_uid):
            raise FileNotFoundError(dataset_uid)
        task = Task(str(uuid.uuid4()), dataset_uid, stage_id)
        self.tasks[task.task_id] = task
        asyncio.create_task(self._run(task, config or {}))
        return task

    async def _run(self, task: Task, config: dict[str, Any]) -> None:
        task.status = "running"
        started_at = time.time()
        stage = self.stages[task.stage_id]
        try:
            dataset = self.catalog.get_dataset(task.dataset_uid)
            await stage.validate_input(dataset)
            episodes = self.catalog.list_episodes(task.dataset_uid)
            results = []
            for index, episode in enumerate(episodes):
                if task.cancel_requested:
                    task.status = "cancelled"
                    return
                results.append(await stage.process_episode(episode, config))
                task.progress = (index + 1) / max(len(episodes), 1)
            task.summary = await stage.aggregate(results)
            run_root = self.output_root / f"stage{task.stage_id}" / task.dataset_uid / task.task_id
            run_root.mkdir(parents=True, exist_ok=True)
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                git_commit = "unknown"
            (run_root / "manifest.json").write_text(json.dumps({"dataset_uid": task.dataset_uid,
                "source": dataset["root"], "stage_id": task.stage_id, "stage": stage.name,
                "detector_version": stage.version, "config": config, "input_codebase_version": dataset.get("codebase_version"),
                "git_commit": git_commit, "worker": {"pid": os.getpid(), "gpu": os.getenv("CUDA_VISIBLE_DEVICES")},
                "created_at": task.created_at, "started_at": started_at, "finished_at": time.time()}, indent=2))
            (run_root / "reports").mkdir(exist_ok=True)
            (run_root / "reports" / "summary.json").write_text(json.dumps(task.summary, indent=2))
            task.status = "succeeded"
        except Exception as exc:  # task state is surfaced to the UI
            task.status, task.error = "failed", str(exc)

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> Task | None:
        task = self.tasks.get(task_id)
        if task and task.status in {"queued", "running"}:
            task.cancel_requested = True
        return task
