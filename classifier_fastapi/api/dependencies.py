"""dependencies.py

FastAPI dependencies for authentication and validation.
"""

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from classifier_fastapi.auth import get_auth_service, AuthService
from classifier_fastapi.settings import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_current_api_key(
    api_key: str = Security(api_key_header)
) -> str:
    """
    Validate service API key.

    Raises:
        HTTPException: If API key is invalid

    Returns:
        Validated API key
    """
    auth_service = get_auth_service()
    try:
        return auth_service.validate_api_key(api_key)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid API key")


async def validate_batch_size(batch_size: int) -> int:
    """
    Validate batch size against configured maximum.

    Args:
        batch_size: Number of texts in batch

    Raises:
        HTTPException: If batch size exceeds maximum

    Returns:
        Validated batch size
    """
    settings = get_settings()

    if batch_size > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {batch_size} exceeds maximum {settings.MAX_BATCH_SIZE}"
        )

    if batch_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be greater than 0"
        )

    return batch_size
