"""
Authentication

API key based authentication for the FastAPI service.
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from typing import Set

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


class AuthService:
    """Authentication service for API key validation"""
    
    def __init__(self, valid_api_keys: Set[str]):
        self.valid_api_keys = valid_api_keys

    def validate_api_key(self, api_key: str = Security(api_key_header)) -> str:
        """Validate API key"""
        if api_key not in self.valid_api_keys:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        return api_key


def get_auth_service() -> AuthService:
    """Get authentication service with configured API keys"""
    # TODO: Load from settings
    from classifier_fastapi.settings import get_settings
    settings = get_settings()
    return AuthService(valid_api_keys=settings.service_api_keys)
