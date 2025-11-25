"""
Job Manager

In-memory job tracking and management for async classification jobs.
"""
from typing import Dict, Optional, List
from datetime import datetime
import uuid
import asyncio
from pydantic import BaseModel

from classifier_fastapi.api.models import JobStatus, JobProgress


class Job(BaseModel):
    """Job model for tracking classification jobs"""
    job_id: str
    status: JobStatus
    progress: JobProgress
    request_data: Dict
    results: List = []
    errors: List = []
    cost: Optional[Dict] = None  # Allow any type for flexibility with token counts
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobManager:
    """Manages classification jobs in memory"""
    
    def __init__(self, max_jobs: int = 1000):
        self._jobs: Dict[str, Job] = {}
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()

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
                    total=len(request_data.get('texts', [])),
                    completed=0,
                    failed=0,
                    percentage=0.0
                ),
                request_data=request_data,
                created_at=datetime.utcnow()
            )
            self._jobs[job_id] = job
            return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_progress(self, job_id: str, completed: int, failed: int):
        """Update job progress"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.progress.completed = completed
                job.progress.failed = failed
                total = job.progress.total
                if total > 0:
                    job.progress.percentage = (completed + failed) / total * 100

    async def update_status(self, job_id: str, status: JobStatus):
        """Update job status"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.status = status
                if status == JobStatus.RUNNING and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job.completed_at = datetime.utcnow()

    async def add_result(self, job_id: str, result: Dict):
        """Add a result to job"""
        async with self._lock:
            if job := self._jobs.get(job_id):
                job.results.append(result)

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

    def _cleanup_old_jobs(self):
        """Remove oldest completed/failed jobs to stay under max_jobs"""
        completed_jobs = [
            (job_id, job) for job_id, job in self._jobs.items()
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
        ]
        completed_jobs.sort(key=lambda x: x[1].completed_at or x[1].created_at)

        for job_id, _ in completed_jobs[:len(completed_jobs) // 2]:
            del self._jobs[job_id]


# Singleton instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create job manager singleton"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
