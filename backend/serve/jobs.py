"""In-memory async job store.

Jobs run one at a time in a thread pool (the pipeline is CPU/IO bound: a GEE
pull of ~30-90 s then tiled torch inference). State is kept in a dict and lost
on process restart, along with the on-disk masks under ``JOBS_DIR`` - this is a
single-instance demo service, not a durable queue.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import time
import traceback
import uuid

from .config import JOB_TIMEOUT_S, JOBS_DIR

STATUSES = ("queued", "fetching", "inferring", "estimating", "done", "failed")


@dataclasses.dataclass
class Job:
    id: str
    created_utc: float
    spec: dict
    status: str = "queued"
    progress: float = 0.0
    message: str = "queued"
    result: dict | None = None
    error: str | None = None
    finished_utc: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_utc": self.created_utc,
            "finished_utc": self.finished_utc,
            "spec": self.spec,
            "result": self.result,
            "error": self.error,
            "mask_url": f"/jobs/{self.id}/mask.png"
            if self.result and self.result.get("mask_ready") else None,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def job_dir(self, job_id: str):
        d = JOBS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def create(self, spec: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], created_utc=time.time(), spec=spec)
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def run(self, job_id: str, run_pipeline) -> None:
        """Execute ``run_pipeline(job, update)`` in a worker thread with a timeout.

        ``run_pipeline`` is the blocking function from serve.pipeline; ``update``
        lets it push status/progress back onto the Job.
        """
        job = self._jobs[job_id]

        def update(status: str | None = None, progress: float | None = None,
                   message: str | None = None) -> None:
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, min(1.0, progress))
            if message is not None:
                job.message = message

        loop = asyncio.get_running_loop()
        try:
            job.status, job.message = "fetching", "starting"
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_pipeline, job, update),
                timeout=JOB_TIMEOUT_S,
            )
            job.result = result
            job.status = "done"
            job.progress = 1.0
            job.message = "done"
        except asyncio.TimeoutError:
            job.status = "failed"
            job.error = (f"job exceeded the {JOB_TIMEOUT_S:.0f}s time limit "
                         "(Earth Engine pull or inference too slow for this AOI).")
            job.message = "timed out"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = "failed"
            traceback.print_exc()
        finally:
            job.finished_utc = time.time()

    def cleanup(self, job_id: str) -> None:
        shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
