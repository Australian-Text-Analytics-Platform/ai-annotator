"""
Job storage backend using JSON files.
"""
from pathlib import Path
from typing import Protocol
import json

import aiofiles
import aiofiles.os


class JobStorage(Protocol):
    """Protocol for job storage backends."""

    async def save(self, job_id: str, data: dict) -> None:
        ...

    async def load(self, job_id: str) -> dict | None:
        ...

    async def delete(self, job_id: str) -> bool:
        ...

    async def list_jobs(self) -> list[str]:
        ...

    async def exists(self, job_id: str) -> bool:
        ...


class FileJobStorage:
    """JSON file-based job storage."""

    def __init__(self, storage_dir: str = "data/jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        # Sanitize job_id to prevent path traversal
        safe_id = job_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.storage_dir / f"{safe_id}.json"

    async def save(self, job_id: str, data: dict) -> None:
        """Save job to disk with atomic write."""
        path = self._job_path(job_id)
        temp_path = path.with_suffix(".tmp")

        content = json.dumps(data, indent=2, default=str)
        async with aiofiles.open(temp_path, "w") as f:
            await f.write(content)

        # Atomic rename (POSIX) - prevents corruption on crash
        await aiofiles.os.rename(temp_path, path)

    async def load(self, job_id: str) -> dict | None:
        """Load job from disk."""
        path = self._job_path(job_id)

        if not path.exists():
            return None

        try:
            async with aiofiles.open(path, "r") as f:
                content = await f.read()
            return json.loads(content)
        except (json.JSONDecodeError, IOError):
            return None

    async def delete(self, job_id: str) -> bool:
        """Delete job file."""
        path = self._job_path(job_id)
        if path.exists():
            await aiofiles.os.remove(path)
            return True
        return False

    async def list_jobs(self) -> list[str]:
        """List all job IDs."""
        return [p.stem for p in self.storage_dir.glob("*.json")]

    async def exists(self, job_id: str) -> bool:
        """Check if job exists."""
        return self._job_path(job_id).exists()
