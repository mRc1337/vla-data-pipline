"""Process entry point reserved for Celery/RQ deployment.

The API uses an asyncio task runner for the zero-dependency local profile. A
production deployment can replace this module with a Celery worker while
keeping the CurationStage interface and task schema stable.
"""
import asyncio


async def main() -> None:
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
