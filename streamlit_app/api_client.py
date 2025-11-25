"""
FastAPI Client for Streamlit App

Handles all HTTP communication with the FastAPI backend.
"""
import os
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FastAPIClient:
    """Client for interacting with the FastAPI classification service"""

    def __init__(self, base_url: str = "http://localhost:8002"):
        """
        Initialize the FastAPI client

        Args:
            base_url: Base URL of the FastAPI service

        Raises:
            ValueError: If SERVICE_API_KEY environment variable is not set
        """
        self.base_url = base_url.rstrip('/')

        # Read service API key from environment
        service_api_key = os.getenv("SERVICE_API_KEY", "")
        if not service_api_key:
            raise ValueError(
                "SERVICE_API_KEY environment variable not set. "
                "Please set it to authenticate with the FastAPI backend."
            )

        self.headers = {"X-API-Key": service_api_key}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the FastAPI service

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            json_data: Optional JSON payload
            timeout: Request timeout in seconds

        Returns:
            Response JSON as dictionary

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"

        with httpx.Client(timeout=timeout) as client:
            if method.upper() == "GET":
                response = client.get(url, headers=self.headers)
            elif method.upper() == "POST":
                response = client.post(url, headers=self.headers, json=json_data)
            elif method.upper() == "DELETE":
                response = client.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

    def get_models(self) -> Dict[str, Any]:
        """
        Get list of available models and providers

        Returns:
            Dict with 'models' and 'providers' keys

        Example response:
            {
                "models": [
                    {
                        "name": "gpt-4o-mini",
                        "provider": "openai",
                        "context_window": 128000,
                        "input_cost_per_1m_tokens": 0.15,
                        "output_cost_per_1m_tokens": 0.60
                    }
                ],
                "providers": ["openai", "ollama"]
            }
        """
        return self._make_request("GET", "/models/")

    def estimate_cost(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate cost of a classification job

        Args:
            request_data: Cost estimation request payload

        Returns:
            Cost estimate response

        Example response:
            {
                "estimated_tokens": 1500,
                "estimated_cost_usd": 0.0023,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "num_texts": 10,
                "input_tokens": 1200,
                "output_tokens": 300,
                "warnings": []
            }
        """
        return self._make_request("POST", "/classify/estimate-cost", json_data=request_data)

    def submit_batch_job(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a batch classification job

        Args:
            request_data: Classification request payload

        Returns:
            Job creation response

        Example response:
            {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "Job created successfully",
                "created_at": "2025-01-15T10:30:00Z"
            }
        """
        return self._make_request("POST", "/classify/batch", json_data=request_data, timeout=60.0)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status and results of a classification job

        Args:
            job_id: Unique job identifier

        Returns:
            Job status response

        Example response:
            {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "progress": {
                    "total": 10,
                    "completed": 10,
                    "failed": 0,
                    "percentage": 100.0
                },
                "results": [
                    {
                        "index": 0,
                        "text": "Sample text",
                        "classification": "Positive",
                        "prompt": "..."
                    }
                ],
                "errors": None,
                "cost": {
                    "total_usd": 0.0023,
                    "total_tokens": 1500
                },
                "created_at": "2025-01-15T10:30:00Z",
                "started_at": "2025-01-15T10:30:01Z",
                "completed_at": "2025-01-15T10:30:05Z"
            }
        """
        return self._make_request("GET", f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running or pending job

        Args:
            job_id: Unique job identifier

        Returns:
            Cancellation response

        Example response:
            {
                "message": "Job cancelled successfully",
                "job_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        """
        return self._make_request("DELETE", f"/jobs/{job_id}")
