"""
Real Integration Tests for Cost Estimation Accuracy

Tests that compare estimated costs to actual costs using real API calls.
Requires API keys in .env file: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
"""
import pytest
from httpx import AsyncClient
import asyncio
import os
from dotenv import load_dotenv


@pytest.mark.integration
@pytest.mark.asyncio
class TestCostEstimationAccuracy:
    """Test accuracy of cost estimation compared to actual costs"""

    async def test_openai_cost_estimation_accuracy(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_user_schema: dict
    ):
        """Test OpenAI cost estimation vs actual cost (requires OPENAI_API_KEY)"""
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in .env file")

        texts = [
            "The product quality is excellent and exceeded my expectations.",
            "Terrible experience, would not recommend to anyone.",
            "It's okay, nothing special but does the job.",
        ]

        base_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "technique": "zero_shot",
            "modifier": "no_modifier",
            "temperature": 1.0,
            "top_p": 1.0,
            "llm_api_key": os.getenv("OPENAI_API_KEY")
        }

        # Step 1: Get cost estimate
        estimate_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "technique": "zero_shot"
        }

        estimate_response = await async_client.post(
            "/classify/estimate-cost",
            json=estimate_request,
            headers=auth_headers
        )
        assert estimate_response.status_code == 200
        estimate_data = estimate_response.json()

        estimated_cost = estimate_data["estimated_cost_usd"]
        estimated_tokens = estimate_data["estimated_tokens"]
        estimated_input = estimate_data["input_tokens"]
        estimated_output = estimate_data["output_tokens"]

        print(f"\n=== OpenAI Cost Estimation ===")
        print(f"Estimated cost: ${estimated_cost:.6f}")
        print(f"Estimated total tokens: {estimated_tokens}")
        print(f"Estimated input tokens: {estimated_input}")
        print(f"Estimated output tokens: {estimated_output}")

        # Step 2: Run actual classification
        submit_response = await async_client.post(
            "/classify/batch",
            json=base_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Step 3: Wait for completion and get actual cost
        max_wait = 60
        start_time = asyncio.get_event_loop().time()
        actual_cost = None
        actual_tokens = None
        actual_input = None
        actual_output = None

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = status_response.json()

            if data["status"] == "completed":
                cost_info = data.get("cost")
                if cost_info:
                    actual_cost = cost_info.get("total_usd")
                    actual_tokens = cost_info.get("total_tokens")
                    actual_input = cost_info.get("input_tokens")
                    actual_output = cost_info.get("output_tokens")
                break
            elif data["status"] == "failed":
                pytest.fail(f"Classification job failed: {data.get('errors')}")

            await asyncio.sleep(2)

        # Step 4: Compare and validate
        assert actual_cost is not None, "No actual cost returned from job"
        assert actual_tokens is not None, "No actual token count returned"

        print(f"\n=== OpenAI Actual Cost ===")
        print(f"Actual cost: ${actual_cost:.6f}")
        print(f"Actual total tokens: {actual_tokens}")
        print(f"Actual input tokens: {actual_input}")
        print(f"Actual output tokens: {actual_output}")

        # Calculate accuracy
        cost_error = abs(estimated_cost - actual_cost) / actual_cost * 100
        token_error = abs(estimated_tokens - actual_tokens) / actual_tokens * 100

        if actual_output:
            output_token_error = abs(estimated_output - actual_output) / actual_output * 100
            print(f"\nOutput token error: {output_token_error:.1f}%")
            print(f"Estimated output: {estimated_output}, Actual output: {actual_output}")

        print(f"\n=== Accuracy Metrics ===")
        print(f"Cost error: {cost_error:.1f}%")
        print(f"Token error: {token_error:.1f}%")

        # Assertions - allow reasonable margin of error
        # Token estimates should be within 50% (estimation is inherently approximate)
        assert token_error < 50, f"Token estimation error too high: {token_error:.1f}%"

        # Cost should be within 100% (double is acceptable for estimates)
        # This is generous because output tokens are hard to predict
        assert cost_error < 100, f"Cost estimation error too high: {cost_error:.1f}%"

        # Estimates should not be too low (should warn users of potential costs)
        # Allow estimates to be lower by up to 50% in edge cases
        relative_difference = (estimated_cost - actual_cost) / actual_cost * 100
        assert relative_difference > -50, f"Cost estimate too low: {relative_difference:.1f}% below actual"

    async def test_gemini_cost_estimation_accuracy(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_user_schema: dict
    ):
        """Test Gemini cost estimation vs actual cost (requires GEMINI_API_KEY)"""
        load_dotenv()

        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set in .env file")

        texts = [
            "Amazing product, highly recommended!",
            "Not worth the money, very disappointed.",
        ]

        base_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "gemini",
            "model": "gemini/gemini-2.5-flash-lite",
            "technique": "zero_shot",
            "modifier": "no_modifier",
            "temperature": 1.0,
            "top_p": 1.0,
            "llm_api_key": os.getenv("GEMINI_API_KEY")
        }

        # Get estimate
        estimate_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "gemini",
            "model": "gemini/gemini-2.5-flash-lite",
            "technique": "zero_shot"
        }

        estimate_response = await async_client.post(
            "/classify/estimate-cost",
            json=estimate_request,
            headers=auth_headers
        )
        assert estimate_response.status_code == 200
        estimate_data = estimate_response.json()

        estimated_cost = estimate_data["estimated_cost_usd"]
        estimated_tokens = estimate_data["estimated_tokens"]

        print(f"\n=== Gemini Cost Estimation ===")
        print(f"Estimated cost: ${estimated_cost:.6f}" if estimated_cost else "Estimated cost: N/A")
        print(f"Estimated tokens: {estimated_tokens}")

        # Run classification
        submit_response = await async_client.post(
            "/classify/batch",
            json=base_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Wait for completion
        max_wait = 60
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = status_response.json()

            if data["status"] == "completed":
                cost_info = data.get("cost")
                if cost_info:
                    actual_cost = cost_info.get("total_usd")
                    actual_tokens = cost_info.get("total_tokens")

                    print(f"\n=== Gemini Actual Cost ===")
                    print(f"Actual cost: ${actual_cost:.6f}" if actual_cost else "Actual cost: N/A")
                    print(f"Actual tokens: {actual_tokens}")

                    if estimated_cost and actual_cost:
                        cost_error = abs(estimated_cost - actual_cost) / actual_cost * 100
                        print(f"\nCost error: {cost_error:.1f}%")
                        assert cost_error < 100, f"Cost estimation error too high: {cost_error:.1f}%"
                break
            elif data["status"] == "failed":
                pytest.fail(f"Classification job failed: {data.get('errors')}")

            await asyncio.sleep(2)

    async def test_anthropic_cost_estimation_accuracy(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_user_schema: dict
    ):
        """Test Anthropic cost estimation vs actual cost (requires ANTHROPIC_API_KEY)"""
        load_dotenv()

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set in .env file")

        texts = [
            "Excellent service and great value for money.",
            "Poor quality, would not buy again.",
        ]

        base_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "technique": "zero_shot",
            "modifier": "no_modifier",
            "temperature": 1.0,
            "top_p": 1.0,
            "llm_api_key": os.getenv("ANTHROPIC_API_KEY")
        }

        # Get estimate
        estimate_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
            "technique": "zero_shot"
        }

        estimate_response = await async_client.post(
            "/classify/estimate-cost",
            json=estimate_request,
            headers=auth_headers
        )
        assert estimate_response.status_code == 200
        estimate_data = estimate_response.json()

        estimated_cost = estimate_data["estimated_cost_usd"]
        estimated_tokens = estimate_data["estimated_tokens"]

        print(f"\n=== Anthropic Cost Estimation ===")
        print(f"Estimated cost: ${estimated_cost:.6f}" if estimated_cost else "Estimated cost: N/A")
        print(f"Estimated tokens: {estimated_tokens}")

        # Run classification
        submit_response = await async_client.post(
            "/classify/batch",
            json=base_request,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # Wait for completion
        max_wait = 60
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = status_response.json()

            if data["status"] == "completed":
                cost_info = data.get("cost")
                if cost_info:
                    actual_cost = cost_info.get("total_usd")
                    actual_tokens = cost_info.get("total_tokens")

                    print(f"\n=== Anthropic Actual Cost ===")
                    print(f"Actual cost: ${actual_cost:.6f}" if actual_cost else "Actual cost: N/A")
                    print(f"Actual tokens: {actual_tokens}")

                    if estimated_cost and actual_cost:
                        cost_error = abs(estimated_cost - actual_cost) / actual_cost * 100
                        print(f"\nCost error: {cost_error:.1f}%")
                        assert cost_error < 100, f"Cost estimation error too high: {cost_error:.1f}%"
                break
            elif data["status"] == "failed":
                pytest.fail(f"Classification job failed: {data.get('errors')}")

            await asyncio.sleep(2)

    async def test_output_token_underestimation_detection(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_user_schema: dict
    ):
        """Test that output tokens are not severely underestimated"""
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in .env file")

        # Use longer texts that might generate more output
        texts = [
            "This is a comprehensive review of the product with many details about the quality, features, and overall experience.",
            "I want to provide extensive feedback on this service including all aspects of customer support and value proposition.",
        ]

        estimate_request = {
            "texts": texts,
            "user_schema": sample_user_schema,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "technique": "zero_shot"
        }

        estimate_response = await async_client.post(
            "/classify/estimate-cost",
            json=estimate_request,
            headers=auth_headers
        )
        estimate_data = estimate_response.json()
        estimated_output = estimate_data["output_tokens"]

        print(f"\n=== Output Token Estimation Test ===")
        print(f"Estimated output tokens: {estimated_output}")
        print(f"Estimated output per text: {estimated_output / len(texts):.0f}")

        # Run actual classification
        base_request = estimate_request.copy()
        base_request["llm_api_key"] = os.getenv("OPENAI_API_KEY")
        base_request["modifier"] = "no_modifier"
        base_request["temperature"] = 1.0
        base_request["top_p"] = 1.0

        submit_response = await async_client.post(
            "/classify/batch",
            json=base_request,
            headers=auth_headers
        )
        job_id = submit_response.json()["job_id"]

        # Wait for completion
        max_wait = 60
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            status_response = await async_client.get(
                f"/jobs/{job_id}",
                headers=auth_headers
            )
            data = status_response.json()

            if data["status"] == "completed":
                cost_info = data.get("cost")
                if cost_info and cost_info.get("output_tokens"):
                    actual_output = cost_info["output_tokens"]

                    print(f"Actual output tokens: {actual_output}")
                    print(f"Actual output per text: {actual_output / len(texts):.0f}")

                    # Check that estimate is not more than 3x too low
                    ratio = actual_output / estimated_output
                    print(f"Actual/Estimated ratio: {ratio:.2f}x")

                    # With default of 8 tokens, estimate should be very close to actual
                    # Allow 0.5x to 2x range (actual can vary by model and technique)
                    assert ratio < 2.0, f"Output tokens underestimated: {ratio:.2f}x actual"
                    assert ratio > 0.5, f"Output tokens overestimated: actual is {ratio:.2f}x of estimate"
                break
            elif data["status"] == "failed":
                pytest.fail(f"Job failed: {data.get('errors')}")

            await asyncio.sleep(2)
