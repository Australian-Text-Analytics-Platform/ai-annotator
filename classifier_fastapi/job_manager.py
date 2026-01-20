"""
Job Manager

Job tracking and management with optional JSON file persistence.
"""
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import asyncio
import sys

from loguru import logger
from pydantic import BaseModel, ConfigDict

from classifier_fastapi.api.models import JobStatus, JobProgress
from classifier_fastapi.storage import FileJobStorage
from classifier_fastapi.settings import get_settings


class Job(BaseModel):
    """Job model for tracking classification jobs"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    status: JobStatus
    progress: JobProgress
    request_data: Dict
    results: List = []
    errors: List = []
    cost: Optional[Dict] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Memory tracking (not persisted)
    _results_bytes: int = 0
    _disk_only_mode: bool = False


class JobManager:
    """Manages classification jobs with optional file persistence"""

    MEMORY_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100MB default

    def __init__(
        self,
        max_jobs: int = 1000,
        storage: FileJobStorage | None = None,
        persist_batch_size: int = 50,
        memory_threshold_mb: int = 100,
    ):
        self._jobs: Dict[str, Job] = {}
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()
        self._storage = storage
        self._persist_batch_size = persist_batch_size
        self._pending_results: Dict[str, List[Dict]] = {}  # Buffer for batching
        self.MEMORY_THRESHOLD_BYTES = memory_threshold_mb * 1024 * 1024

    async def _persist(self, job: Job) -> None:
        """Persist job to storage. Logs warning on failure but doesn't raise."""
        if not self._storage:
            return
        try:
            await self._storage.save(job.job_id, job.model_dump(mode="json"))
        except Exception as e:
            logger.warning(f"Failed to persist job {job.job_id}: {e}")

    async def create_job(self, request_data: Dict) -> str:
        """Create a new job"""
        async with self._lock:
            if len(self._jobs) >= self._max_jobs:
                self._cleanup_old_jobs()

            job_id = str(uuid.uuid4())
            job = Job(
                job_id=job_id,
                status=JobStatus.PENDING,
                progress=JobProgress(
                    total=len(request_data.get("texts", [])),
                    completed=0,
                    failed=0,
                    percentage=0.0,
                ),
                request_data=request_data,
                created_at=datetime.utcnow(),
            )
            self._jobs[job_id] = job
            self._pending_results[job_id] = []
            await self._persist(job)
            return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_progress(self, job_id: str, completed: int, failed: int):
        """Update job progress (not persisted immediately - batched with results)"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.progress.completed = completed
                job.progress.failed = failed
                total = job.progress.total
                if total > 0:
                    job.progress.percentage = (completed + failed) / total * 100

    async def update_status(self, job_id: str, status: JobStatus):
        """Update job status and persist immediately (status changes are important)"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.status = status
                if status == JobStatus.RUNNING and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job.completed_at = datetime.utcnow()
                await self._persist(job)

    async def add_result(self, job_id: str, result: Dict):
        """Add result with batched persistence and memory threshold."""
        async with self._lock:
            if job := self._jobs.get(job_id):
                # Track memory usage
                result_size = sys.getsizeof(str(result))
                job._results_bytes += result_size

                # Check memory threshold
                if (
                    job._results_bytes > self.MEMORY_THRESHOLD_BYTES
                    and not job._disk_only_mode
                ):
                    logger.info(
                        f"Job {job_id} exceeded {self.MEMORY_THRESHOLD_BYTES // (1024*1024)}MB threshold, "
                        "switching to disk-only mode"
                    )
                    job._disk_only_mode = True
                    job.results = []  # Clear from memory

                # Add to in-memory results if not in disk-only mode
                if not job._disk_only_mode:
                    job.results.append(result)

                # Buffer for batched persistence
                self._pending_results.setdefault(job_id, []).append(result)

                # Persist when batch size reached
                if len(self._pending_results[job_id]) >= self._persist_batch_size:
                    await self._flush_results(job_id)

    async def _flush_results(self, job_id: str) -> None:
        """Flush pending results to disk."""
        if job := self._jobs.get(job_id):
            self._pending_results[job_id] = []
            await self._persist(job)

    async def flush_all_pending(self, job_id: str) -> None:
        """Force flush any remaining results (call at job completion)."""
        async with self._lock:
            if self._pending_results.get(job_id):
                await self._flush_results(job_id)

    async def add_error(self, job_id: str, error: Dict):
        """Add an error to job"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.errors.append(error)

    async def set_cost(self, job_id: str, cost: Dict):
        """Set job cost"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.cost = cost

    async def list_jobs(
        self, status: JobStatus | None = None, limit: int = 100
    ) -> List[Dict]:
        """List jobs, optionally filtered by status."""
        async with self._lock:
            jobs = list(self._jobs.values())

            if status is not None:
                jobs = [j for j in jobs if j.status == status]

            # Sort by created_at descending (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            return [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "progress": j.progress.model_dump(),
                    "created_at": j.created_at.isoformat(),
                    "started_at": j.started_at.isoformat() if j.started_at else None,
                    "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                }
                for j in jobs[:limit]
            ]

    async def load_jobs_from_storage(self) -> int:
        """Load existing jobs from storage on startup. Mark running jobs as failed."""
        if not self._storage:
            return 0

        count = 0
        for job_id in await self._storage.list_jobs():
            if job_id not in self._jobs:
                data = await self._storage.load(job_id)
                if data:
                    try:
                        job = Job.model_validate(data)

                        # Mark interrupted jobs as failed
                        if job.status == JobStatus.RUNNING:
                            logger.warning(
                                f"Job {job_id} was running when server stopped, marking as failed"
                            )
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.utcnow()
                            job.errors.append(
                                {
                                    "error": "Job interrupted due to server restart",
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                            await self._storage.save(job_id, job.model_dump(mode="json"))

                        self._jobs[job_id] = job
                        self._pending_results[job_id] = []
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load job {job_id}: {e}")
        return count

    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs to stay under max_jobs"""
        completed_jobs = [
            (job_id, job)
            for job_id, job in self._jobs.items()
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
        ]
        completed_jobs.sort(key=lambda x: x[1].completed_at or x[1].created_at)

        for job_id, _ in completed_jobs[: len(completed_jobs) // 2]:
            del self._jobs[job_id]
            if job_id in self._pending_results:
                del self._pending_results[job_id]


# Singleton instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create job manager singleton"""
    global _job_manager
    if _job_manager is None:
        settings = get_settings()

        storage = None
        if settings.JOB_PERSIST_ENABLED:
            storage = FileJobStorage(storage_dir=settings.JOB_STORAGE_DIR)

        _job_manager = JobManager(
            storage=storage,
            persist_batch_size=settings.JOB_PERSIST_BATCH_SIZE,
            memory_threshold_mb=settings.JOB_MEMORY_THRESHOLD_MB,
        )
    return _job_manager


def reset_job_manager() -> None:
    """Reset the job manager singleton (useful for testing)"""
    global _job_manager
    _job_manager = None
