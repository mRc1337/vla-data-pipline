"""Optional distributed worker entry point.

The single-node API defaults to asyncio. When Redis/Celery are enabled,
``celery -A vla_platform.celery_app worker`` executes the same stage contract
with an explicit queue per resource class.
"""
import asyncio
import os

from .api import catalog, runner

try:
    from celery import Celery
except ImportError:  # pragma: no cover - optional deployment dependency
    Celery = None

if Celery is not None:
    celery_app = Celery("vla-data-pipeline", broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
                        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1"))
    celery_app.conf.task_routes = {"vla_platform.celery_app.run_stage": {"queue": "cpu"}}

    @celery_app.task(name="vla_platform.celery_app.run_stage")
    def run_stage(dataset_uid: str, stage_id: int, config: dict | None = None) -> dict:
        async def execute() -> dict:
            task = await runner.submit(dataset_uid, stage_id, config or {})
            while task.status in {"queued", "running"}:
                await asyncio.sleep(0.1)
            return task.__dict__
        return asyncio.run(execute())
else:
    celery_app = None
