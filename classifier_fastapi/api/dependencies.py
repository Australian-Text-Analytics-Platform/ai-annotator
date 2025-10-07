"""
FastAPI Dependencies

Authentication and validation dependencies.
"""
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

# TODO: Implement authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_current_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate service API key"""
    # TODO: Implement actual validation
    return api_key
