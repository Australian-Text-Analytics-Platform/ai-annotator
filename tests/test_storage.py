"""
Test job storage and persistence

Tests for FileJobStorage and JobManager persistence integration.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from classifier_fastapi.storage import FileJobStorage
from classifier_fastapi.job_manager import JobManager, Job, reset_job_manager
from classifier_fastapi.api.models import JobStatus, JobProgress


class TestFileJobStorage:
    """Test FileJobStorage class"""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a FileJobStorage instance"""
        return FileJobStorage(storage_dir=temp_storage_dir)

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage):
        """Test basic save and load functionality"""
        job_data = {
            "job_id": "test-123",
            "status": "pending",
            "progress": {"total": 10, "completed": 0, "failed": 0, "percentage": 0.0},
            "request_data": {"texts": ["text1", "text2"]},
            "results": [],
            "errors": [],
        }

        await storage.save("test-123", job_data)
        loaded = await storage.load("test-123")

        assert loaded is not None
        assert loaded["job_id"] == "test-123"
        assert loaded["status"] == "pending"
        assert loaded["progress"]["total"] == 10

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, storage):
        """Test loading a non-existent job returns None"""
        loaded = await storage.load("nonexistent-job")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_exists(self, storage):
        """Test exists check"""
        await storage.save("test-job", {"status": "pending"})

        assert await storage.exists("test-job") is True
        assert await storage.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        """Test deleting a job"""
        await storage.save("test-job", {"status": "pending"})
        assert await storage.exists("test-job") is True

        result = await storage.delete("test-job")
        assert result is True
        assert await storage.exists("test-job") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage):
        """Test deleting a non-existent job returns False"""
        result = await storage.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, storage):
        """Test listing all jobs"""
        await storage.save("job-1", {"status": "pending"})
        await storage.save("job-2", {"status": "running"})
        await storage.save("job-3", {"status": "completed"})

        jobs = await storage.list_jobs()
        assert len(jobs) == 3
        assert set(jobs) == {"job-1", "job-2", "job-3"}

    @pytest.mark.asyncio
    async def test_atomic_write(self, storage, temp_storage_dir):
        """Test that writes are atomic (no .tmp files left behind)"""
        await storage.save("test-job", {"status": "pending"})

        tmp_files = list(Path(temp_storage_dir).glob("*.tmp"))
        assert len(tmp_files) == 0

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, storage):
        """Test that path traversal attempts are sanitized"""
        # Attempt path traversal
        await storage.save("../../../etc/passwd", {"status": "malicious"})

        # Should be saved with sanitized name, not in actual path
        loaded = await storage.load("../../../etc/passwd")
        assert loaded is not None
        assert loaded["status"] == "malicious"

        # Original path should not exist
        assert not Path("/etc/passwd.json").exists()

    @pytest.mark.asyncio
    async def test_corrupted_json_returns_none(self, storage, temp_storage_dir):
        """Test that corrupted JSON files return None"""
        # Write corrupted JSON directly
        job_path = Path(temp_storage_dir) / "corrupted.json"
        job_path.write_text("{ invalid json }")

        loaded = await storage.load("corrupted")
        assert loaded is None


class TestJobManagerPersistence:
    """Test JobManager with persistence"""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a FileJobStorage instance"""
        return FileJobStorage(storage_dir=temp_storage_dir)

    @pytest.fixture
    def job_manager(self, storage):
        """Create a JobManager with storage"""
        return JobManager(
            max_jobs=100,
            storage=storage,
            persist_batch_size=5,  # Small batch for testing
            memory_threshold_mb=1,  # 1MB for testing
        )

    @pytest.mark.asyncio
    async def test_create_job_persists(self, job_manager, storage):
        """Test that creating a job persists it to storage"""
        job_id = await job_manager.create_job({"texts": ["text1", "text2"]})

        # Check storage directly
        loaded = await storage.load(job_id)
        assert loaded is not None
        assert loaded["status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_status_persists(self, job_manager, storage):
        """Test that status updates are persisted immediately"""
        job_id = await job_manager.create_job({"texts": ["text1"]})
        await job_manager.update_status(job_id, JobStatus.RUNNING)

        loaded = await storage.load(job_id)
        assert loaded["status"] == "running"
        assert loaded["started_at"] is not None

    @pytest.mark.asyncio
    async def test_batched_results_persistence(self, job_manager, storage):
        """Test that results are persisted in batches"""
        job_id = await job_manager.create_job({"texts": ["text"] * 10})
        await job_manager.update_status(job_id, JobStatus.RUNNING)

        # Add 4 results (less than batch size of 5)
        for i in range(4):
            await job_manager.add_result(job_id, {"index": i, "classification": "test"})

        # Check that results are in memory but may not be persisted yet
        job = await job_manager.get_job(job_id)
        assert len(job.results) == 4

        # Add 5th result to trigger persistence
        await job_manager.add_result(job_id, {"index": 4, "classification": "test"})

        # Verify persistence happened
        loaded = await storage.load(job_id)
        assert len(loaded["results"]) == 5

    @pytest.mark.asyncio
    async def test_flush_pending_results(self, job_manager, storage):
        """Test that flush_all_pending persists remaining results"""
        job_id = await job_manager.create_job({"texts": ["text"] * 10})
        await job_manager.update_status(job_id, JobStatus.RUNNING)

        # Add 3 results (less than batch size)
        for i in range(3):
            await job_manager.add_result(job_id, {"index": i, "classification": "test"})

        # Flush remaining
        await job_manager.flush_all_pending(job_id)

        # Verify all results are persisted
        loaded = await storage.load(job_id)
        assert len(loaded["results"]) == 3

    @pytest.mark.asyncio
    async def test_load_jobs_from_storage(self, storage, temp_storage_dir):
        """Test loading jobs from storage on startup"""
        # Create jobs directly in storage
        job_data = {
            "job_id": "existing-job",
            "status": "completed",
            "progress": {"total": 5, "completed": 5, "failed": 0, "percentage": 100.0},
            "request_data": {"texts": ["text1"]},
            "results": [{"index": 0, "classification": "positive"}],
            "errors": [],
            "cost": None,
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:00:01",
            "completed_at": "2024-01-15T10:00:10",
        }
        await storage.save("existing-job", job_data)

        # Create new job manager and load from storage
        new_manager = JobManager(storage=storage)
        loaded = await new_manager.load_jobs_from_storage()

        assert loaded == 1

        job = await new_manager.get_job("existing-job")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert len(job.results) == 1

    @pytest.mark.asyncio
    async def test_running_jobs_marked_failed_on_restart(self, storage):
        """Test that running jobs are marked as failed on restart"""
        # Create a running job directly in storage
        job_data = {
            "job_id": "running-job",
            "status": "running",
            "progress": {"total": 10, "completed": 5, "failed": 0, "percentage": 50.0},
            "request_data": {"texts": ["text1"]},
            "results": [],
            "errors": [],
            "cost": None,
            "created_at": "2024-01-15T10:00:00",
            "started_at": "2024-01-15T10:00:01",
            "completed_at": None,
        }
        await storage.save("running-job", job_data)

        # Create new job manager and load from storage
        new_manager = JobManager(storage=storage)
        await new_manager.load_jobs_from_storage()

        job = await new_manager.get_job("running-job")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert any("server restart" in str(e).lower() for e in job.errors)

    @pytest.mark.asyncio
    async def test_progress_not_persisted_immediately(self, job_manager, storage):
        """Test that progress updates are not persisted immediately"""
        job_id = await job_manager.create_job({"texts": ["text"] * 10})
        await job_manager.update_status(job_id, JobStatus.RUNNING)

        # Get the storage state after status update
        loaded_before = await storage.load(job_id)
        assert loaded_before["progress"]["completed"] == 0

        # Update progress (should not persist)
        await job_manager.update_progress(job_id, completed=5, failed=0)

        # Storage should still show old progress
        loaded_after = await storage.load(job_id)
        assert loaded_after["progress"]["completed"] == 0

        # Memory should have new progress
        job = await job_manager.get_job(job_id)
        assert job.progress.completed == 5


class TestJobManagerListJobs:
    """Test JobManager list_jobs functionality"""

    @pytest.fixture
    def job_manager(self):
        """Create a JobManager without storage for simpler tests"""
        return JobManager(max_jobs=100)

    @pytest.mark.asyncio
    async def test_list_all_jobs(self, job_manager):
        """Test listing all jobs"""
        await job_manager.create_job({"texts": ["text1"]})
        await job_manager.create_job({"texts": ["text2"]})
        await job_manager.create_job({"texts": ["text3"]})

        jobs = await job_manager.list_jobs()
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status(self, job_manager):
        """Test filtering jobs by status"""
        job1 = await job_manager.create_job({"texts": ["text1"]})
        job2 = await job_manager.create_job({"texts": ["text2"]})
        job3 = await job_manager.create_job({"texts": ["text3"]})

        await job_manager.update_status(job1, JobStatus.RUNNING)
        await job_manager.update_status(job2, JobStatus.COMPLETED)

        pending_jobs = await job_manager.list_jobs(status=JobStatus.PENDING)
        assert len(pending_jobs) == 1
        assert pending_jobs[0]["job_id"] == job3

        running_jobs = await job_manager.list_jobs(status=JobStatus.RUNNING)
        assert len(running_jobs) == 1
        assert running_jobs[0]["job_id"] == job1

    @pytest.mark.asyncio
    async def test_list_jobs_with_limit(self, job_manager):
        """Test listing jobs with limit"""
        for i in range(10):
            await job_manager.create_job({"texts": [f"text{i}"]})

        jobs = await job_manager.list_jobs(limit=5)
        assert len(jobs) == 5

    @pytest.mark.asyncio
    async def test_list_jobs_sorted_by_created_at(self, job_manager):
        """Test that jobs are sorted by created_at descending"""
        job1 = await job_manager.create_job({"texts": ["text1"]})
        job2 = await job_manager.create_job({"texts": ["text2"]})
        job3 = await job_manager.create_job({"texts": ["text3"]})

        jobs = await job_manager.list_jobs()
        # Most recent first
        assert jobs[0]["job_id"] == job3
        assert jobs[1]["job_id"] == job2
        assert jobs[2]["job_id"] == job1


class TestStorageFailureTolerance:
    """Test that storage failures don't break job processing"""

    @pytest.mark.asyncio
    async def test_storage_failure_logs_warning(self, caplog):
        """Test that storage failures log warnings but don't raise"""
        from unittest.mock import AsyncMock, MagicMock

        # Create a mock storage that fails
        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(side_effect=IOError("Disk full"))

        job_manager = JobManager(storage=mock_storage)

        # Should not raise, just log warning
        job_id = await job_manager.create_job({"texts": ["text1"]})

        assert job_id is not None
        # Job should exist in memory
        job = await job_manager.get_job(job_id)
        assert job is not None
