"""
Security and authentication utilities for ASL Translator API
"""
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
import os
import logging

logger = logging.getLogger(__name__)

# Import security logging
try:
    from security_logging import SecurityLogger
except ImportError:
    # Fallback if security_logging not available
    class SecurityLogger:
        @staticmethod
        def log_authentication_failure(request, reason, api_key_prefix=None):
            logger.warning(f"Authentication failure: {reason}")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_valid_api_keys():
    """Retrieve valid API keys from environment"""
    api_keys_env = os.getenv("API_KEYS", "")
    if not api_keys_env:
        logger.warning("No API keys configured in environment")
        return []
    return [key.strip() for key in api_keys_env.split(",") if key.strip()]


async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    """
    Verify API key from request header
    
    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header
        
    Returns:
        str: Validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not api_key:
        SecurityLogger.log_authentication_failure(request, "No API key provided")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Please provide X-API-Key header."
        )
    
    valid_keys = get_valid_api_keys()
    
    if not valid_keys:
        logger.error("No API keys configured - authentication disabled")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication not configured"
        )
    
    if api_key not in valid_keys:
        SecurityLogger.log_authentication_failure(request, "Invalid API key", api_key[:8])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    logger.info(f"API key authenticated successfully from IP: {request.client.host}, Key prefix: {api_key[:8]}...")
    return api_key


async def optional_api_key(request: Request, api_key: str = Security(api_key_header)):
    """
    Optional API key verification (for endpoints that work with or without auth)
    
    Returns:
        str or None: API key if provided and valid, None otherwise
    """
    if not api_key:
        return None
    
    valid_keys = get_valid_api_keys()
    if api_key in valid_keys:
        logger.debug(f"Optional API key validated: {api_key[:8]}...")
        return api_key
    
    return None
