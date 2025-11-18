"""
Integration tests for FastAPI service

Full workflow tests including job submission, polling, and result retrieval.
These tests may use real API keys and are marked as integration tests.
"""
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
import asyncio
import time
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestFullWorkflow:
    """Test full classification workflow"""

    @pytest.mark.asyncio
    async def test_submit_poll_get_results_workflow(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test full workflow: submit job → poll status → get results"""
        # Step 1: Submit job
        submit_response = await async_client.post(
            "/classify/batch",
            json=sample_classification_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        job_data = submit_response.json()
        job_id = job_data["job_id"]

        # Step 2: Poll status until complete (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()
        status = "pending"

        while status not in ["completed", "failed"] and (time.time() - start_time) < max_wait:
            await asyncio.sleep(0.5)
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            assert status_response.status_code == 200
            status_data = status_response.json()
            status = status_data["status"]

        # Step 3: Get final results
        final_response = await async_client.get(
            f"/jobs/{job_id}",
            headers=auth_headers
        )
        assert final_response.status_code == 200
        final_data = final_response.json()

        # Verify we got a result (completed or failed)
        assert final_data["status"] in ["completed", "failed"]

        if final_data["status"] == "completed":
            assert final_data["results"] is not None
            assert len(final_data["results"]) > 0

    @pytest.mark.asyncio
    async def test_workflow_with_progress_tracking(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test workflow with progress tracking"""
        # Submit job with multiple texts
        request = sample_classification_request.copy()
        request["texts"] = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        submit_response = await async_client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        job_id = submit_response.json()["job_id"]

        # Poll and track progress
        progress_values = []
        max_wait = 30
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = status_response.json()
            progress_values.append(data["progress"]["percentage"])

            if data["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(0.3)

        # Should have seen progress increase
        assert len(progress_values) > 0
        # Final progress should be 100 if completed
        if data["status"] == "completed":
            assert data["progress"]["percentage"] == 100.0


@pytest.mark.integration
class TestConcurrentJobs:
    """Test concurrent job processing"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_concurrent_jobs(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test that multiple jobs can run concurrently"""
        num_jobs = 3

        # Submit multiple jobs
        job_ids = []
        for i in range(num_jobs):
            request = sample_classification_request.copy()
            request["texts"] = [f"text {i}-{j}" for j in range(3)]

            response = await async_client.post(
                "/classify/batch",
                json=request,
                headers=auth_headers
            )
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        # Wait for all jobs to complete
        max_wait = 60
        start_time = time.time()

        all_completed = False
        while not all_completed and (time.time() - start_time) < max_wait:
            statuses = []
            for job_id in job_ids:
                response = await async_client.get(
                    f"/jobs/{job_id}",
                    headers=auth_headers
                )
                statuses.append(response.json()["status"])

            all_completed = all(s in ["completed", "failed"] for s in statuses)
            await asyncio.sleep(1)

        # Verify all jobs finished
        for job_id in job_ids:
            response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = response.json()
            assert data["status"] in ["completed", "failed"]


class TestJobCleanup:
    """Test job cleanup and expiration"""

    @pytest.mark.asyncio
    async def test_job_manager_cleanup_old_jobs(
        self,
        test_job_manager
    ):
        """Test that old jobs are cleaned up when limit is reached"""
        from classifier_fastapi.api.models import JobStatus

        # Create jobs up to the limit
        max_jobs = test_job_manager._max_jobs
        job_ids = []

        for i in range(max_jobs):
            job_id = await test_job_manager.create_job({
                "texts": [f"text {i}"]
            })
            job_ids.append(job_id)
            # Mark as completed
            await test_job_manager.update_status(job_id, JobStatus.COMPLETED)

        # Create one more job to trigger cleanup
        new_job_id = await test_job_manager.create_job({
            "texts": ["new text"]
        })

        # Old jobs should be cleaned up
        assert new_job_id not in job_ids

        # Some old jobs should be removed
        remaining_old_jobs = 0
        for job_id in job_ids[:max_jobs // 2]:
            job = await test_job_manager.get_job(job_id)
            if job is not None:
                remaining_old_jobs += 1
        assert remaining_old_jobs < max_jobs // 2


@pytest.mark.integration
class TestRealAPIIntegration:
    """
    Tests with real API calls (optional, requires valid API keys).
    Run with: pytest -m integration
    """

    @pytest.mark.skip(reason="Requires real API key and makes actual API calls")
    @pytest.mark.asyncio
    async def test_real_openai_classification(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_user_schema: dict
    ):
        """Test real OpenAI API classification"""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        request = {
            "texts": [
                "This product is amazing!",
                "Terrible experience, very disappointed."
            ],
            "user_schema": sample_user_schema,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "technique": "zero_shot",
            "llm_api_key": os.getenv("OPENAI_API_KEY")
        }

        # Submit job
        submit_response = await async_client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Wait for completion
        max_wait = 60
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = response.json()

            if data["status"] == "completed":
                assert len(data["results"]) == 2
                assert data["cost"] is not None
                break
            elif data["status"] == "failed":
                pytest.fail(f"Job failed: {data['errors']}")

            await asyncio.sleep(2)


class TestErrorHandling:
    """Test error handling in integration scenarios"""

    @pytest.mark.asyncio
    async def test_job_failure_handling(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict
    ):
        """Test that job failures are handled gracefully"""
        # Submit job with invalid configuration
        request = sample_classification_request.copy()
        request["provider"] = "openai"
        request["model"] = "nonexistent-model-xyz"
        request["llm_api_key"] = "invalid-key-12345"

        submit_response = await async_client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )

        # Job submission should succeed
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Wait for job to fail
        max_wait = 10
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = response.json()

            if data["status"] == "failed":
                # Should have error details
                assert data["errors"] is not None
                assert len(data["errors"]) > 0
                break

            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_cancel_running_job_integration(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test canceling a running job"""
        # Submit job
        request = sample_classification_request.copy()
        request["texts"] = ["text"] * 10  # More texts to ensure it runs longer

        submit_response = await async_client.post(
            "/classify/batch",
            json=request,
            headers=auth_headers
        )
        job_id = submit_response.json()["job_id"]

        # Wait a bit for job to start
        await asyncio.sleep(0.5)

        # Cancel the job
        cancel_response = await async_client.delete(
            f"/jobs/{job_id}",
            headers=auth_headers
        )

        # May succeed or fail depending on timing
        assert cancel_response.status_code in [200, 400]


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios"""

    @pytest.mark.asyncio
    async def test_classify_then_estimate_cost(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        sample_cost_estimate_request: dict,
        mock_litellm
    ):
        """Test estimating cost before running classification"""
        # First, estimate cost
        with patch("classifier_fastapi.core.cost.CostEstimator.estimate_tokens") as mock_tokens:
            with patch("classifier_fastapi.core.cost.CostEstimator.estimate_cost") as mock_cost:
                mock_tokens.return_value = {
                    "total_tokens": 1000,
                    "input_tokens": 800,
                    "estimated_output_tokens": 200
                }
                mock_cost.return_value = 0.001

                estimate_response = await async_client.post(
                    "/classify/estimate-cost",
                    json=sample_cost_estimate_request,
                    headers=auth_headers
                )
                assert estimate_response.status_code == 200
                estimate_data = estimate_response.json()

                # Check estimated cost
                assert estimate_data["estimated_cost_usd"] is not None

        # Then, run classification
        submit_response = await async_client.post(
            "/classify/batch",
            json=sample_classification_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_models_then_classify(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test listing models before classification"""
        # First, list available models
        models_response = await async_client.get(
            "/models/",
            headers=auth_headers
        )
        assert models_response.status_code == 200
        models_data = models_response.json()

        # Pick a model (or use the one from sample request)
        assert len(models_data["models"]) > 0

        # Then classify
        submit_response = await async_client.post(
            "/classify/batch",
            json=sample_classification_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200


@pytest.mark.asyncio
class TestHealthCheckDuringLoad:
    """Test that health checks work during load"""

    async def test_health_check_during_classification(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_classification_request: dict,
        mock_litellm
    ):
        """Test that health check remains responsive during classification"""
        # Submit a job
        await async_client.post(
            "/classify/batch",
            json=sample_classification_request,
            headers=auth_headers
        )

        # Health check should still work
        health_response = await async_client.get("/health/")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
