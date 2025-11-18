"""
Test job status and management endpoints

Tests the /jobs/ endpoints for job status retrieval and management.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from datetime import datetime

from classifier_fastapi.api.models import JobStatus, JobProgress
from classifier_fastapi.job_manager import Job


class TestJobsEndpoint:
    """Test job status endpoint"""

    def test_jobs_requires_authentication(self, client: TestClient):
        """Test that /jobs/{job_id} requires authentication"""
        response = client.get("/jobs/test-job-id")
        assert response.status_code == 403

    def test_jobs_with_valid_auth(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that /jobs/{job_id} works with valid authentication"""
        response = client.get("/jobs/nonexistent-job", headers=auth_headers)
        # Should return 404, not 403
        assert response.status_code == 404

    def test_job_not_found_returns_404(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that non-existent job returns 404"""
        response = client.get("/jobs/nonexistent-job-id", headers=auth_headers)
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]


class TestJobStatusRetrieval:
    """Test job status retrieval"""

    @pytest.mark.asyncio
    async def test_get_pending_job_status(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test retrieving a pending job status"""
        # Create a job
        job_id = await test_job_manager.create_job({
            "texts": ["text1", "text2"],
            "user_schema": {},
            "provider": "openai",
            "model": "gpt-4o-mini"
        })

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"
        assert "progress" in data
        assert data["progress"]["total"] == 2
        assert data["progress"]["completed"] == 0
        assert data["progress"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_get_running_job_status(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test retrieving a running job status"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1", "text2"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "running"
        assert data["started_at"] is not None

    @pytest.mark.asyncio
    async def test_get_completed_job_with_results(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test retrieving a completed job with results"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1", "text2"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)

        # Add results
        await test_job_manager.add_result(job_id, {
            "index": 0,
            "text": "text1",
            "classification": "positive",
            "prompt": "classify this"
        })
        await test_job_manager.add_result(job_id, {
            "index": 1,
            "text": "text2",
            "classification": "negative",
            "prompt": "classify this"
        })

        await test_job_manager.update_status(job_id, JobStatus.COMPLETED)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert data["completed_at"] is not None
        assert data["results"] is not None
        assert len(data["results"]) == 2
        assert data["results"][0]["classification"] == "positive"
        assert data["results"][1]["classification"] == "negative"

    @pytest.mark.asyncio
    async def test_get_failed_job_with_errors(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test retrieving a failed job with error details"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)

        # Add error
        await test_job_manager.add_error(job_id, {
            "error": "API key invalid",
            "type": "AuthenticationError"
        })

        await test_job_manager.update_status(job_id, JobStatus.FAILED)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "failed"
        assert data["errors"] is not None
        assert len(data["errors"]) > 0
        assert "API key invalid" in str(data["errors"])


class TestJobProgressTracking:
    """Test job progress tracking"""

    @pytest.mark.asyncio
    async def test_job_progress_updates(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that job progress updates correctly"""
        job_id = await test_job_manager.create_job({
            "texts": ["t1", "t2", "t3", "t4", "t5"],
        })

        await test_job_manager.update_status(job_id, JobStatus.RUNNING)
        await test_job_manager.update_progress(job_id, completed=2, failed=1)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["progress"]["total"] == 5
        assert data["progress"]["completed"] == 2
        assert data["progress"]["failed"] == 1
        # 3 out of 5 processed = 60%
        assert data["progress"]["percentage"] == 60.0

    @pytest.mark.asyncio
    async def test_job_progress_percentage_calculation(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that job progress percentage is calculated correctly"""
        job_id = await test_job_manager.create_job({
            "texts": ["t1", "t2", "t3", "t4"],
        })

        await test_job_manager.update_progress(job_id, completed=3, failed=0)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["progress"]["percentage"] == 75.0


class TestJobCost:
    """Test job cost information"""

    @pytest.mark.asyncio
    async def test_job_cost_included_when_available(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that job cost is included when available"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })

        await test_job_manager.set_cost(job_id, {
            "total_usd": 0.0015,
            "total_tokens": 1500
        })

        await test_job_manager.update_status(job_id, JobStatus.COMPLETED)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["cost"] is not None
        assert data["cost"]["total_usd"] == 0.0015
        assert data["cost"]["total_tokens"] == 1500


class TestCancelJob:
    """Test job cancellation"""

    def test_cancel_job_requires_authentication(self, client: TestClient):
        """Test that DELETE /jobs/{job_id} requires authentication"""
        response = client.delete("/jobs/test-job-id")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_cancel_pending_job(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test canceling a pending job"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })

        response = client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert "cancelled" in data["message"].lower()

        # Verify job is cancelled
        job = await test_job_manager.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_running_job(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test canceling a running job"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)

        response = client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        # Verify job is cancelled
        job = await test_job_manager.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed_job(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that completed job cannot be cancelled"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.COMPLETED)

        response = client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 400
        assert "Cannot cancel" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_cannot_cancel_failed_job(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that failed job cannot be cancelled"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.FAILED)

        response = client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 400

    def test_cancel_nonexistent_job_returns_404(
        self,
        client: TestClient,
        auth_headers: dict
    ):
        """Test that canceling non-existent job returns 404"""
        response = client.delete("/jobs/nonexistent-job", headers=auth_headers)
        assert response.status_code == 404


@pytest.mark.asyncio
class TestJobsAsync:
    """Async tests for jobs endpoint"""

    async def test_get_job_status_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test get job status with async client"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })

        response = await async_client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id

    async def test_cancel_job_async(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test cancel job with async client"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })

        response = await async_client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200


class TestJobTimestamps:
    """Test job timestamp fields"""

    @pytest.mark.asyncio
    async def test_job_created_at_timestamp(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that job has created_at timestamp"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["created_at"] is not None
        # Should be valid ISO format
        datetime.fromisoformat(data["created_at"])

    @pytest.mark.asyncio
    async def test_job_started_at_timestamp(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that running job has started_at timestamp"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["started_at"] is not None
        datetime.fromisoformat(data["started_at"])

    @pytest.mark.asyncio
    async def test_job_completed_at_timestamp(
        self,
        client: TestClient,
        auth_headers: dict,
        test_job_manager
    ):
        """Test that completed job has completed_at timestamp"""
        job_id = await test_job_manager.create_job({
            "texts": ["text1"],
        })
        await test_job_manager.update_status(job_id, JobStatus.RUNNING)
        await test_job_manager.update_status(job_id, JobStatus.COMPLETED)

        response = client.get(f"/jobs/{job_id}", headers=auth_headers)
        data = response.json()

        assert data["completed_at"] is not None
        datetime.fromisoformat(data["completed_at"])
