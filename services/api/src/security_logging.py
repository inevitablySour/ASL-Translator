"""
Security event logging for monitoring and incident detection
"""
import logging
from datetime import datetime
from typing import Optional
from fastapi import Request

logger = logging.getLogger("security")


class SecurityLogger:
    """Logger for security-related events"""
    
    @staticmethod
    def log_authentication_failure(request: Request, reason: str, api_key_prefix: Optional[str] = None):
        """Log failed authentication attempts"""
        logger.warning(
            f"Authentication failure - IP: {request.client.host}, "
            f"Path: {request.url.path}, Reason: {reason}, "
            f"API Key: {api_key_prefix}..." if api_key_prefix else "No API key provided"
        )
    
    @staticmethod
    def log_rate_limit_exceeded(request: Request, endpoint: str):
        """Log rate limit violations"""
        logger.warning(
            f"Rate limit exceeded - IP: {request.client.host}, "
            f"Endpoint: {endpoint}, User-Agent: {request.headers.get('user-agent', 'Unknown')}"
        )
    
    @staticmethod
    def log_validation_error(request: Request, endpoint: str, error: str):
        """Log input validation failures"""
        logger.info(
            f"Validation error - IP: {request.client.host}, "
            f"Endpoint: {endpoint}, Error: {error}"
        )
    
    @staticmethod
    def log_suspicious_activity(request: Request, activity_type: str, details: str):
        """Log suspicious or potentially malicious activity"""
        logger.warning(
            f"Suspicious activity detected - IP: {request.client.host}, "
            f"Type: {activity_type}, Details: {details}, "
            f"User-Agent: {request.headers.get('user-agent', 'Unknown')}"
        )
    
    @staticmethod
    def log_admin_action(request: Request, action: str, resource: str, api_key_prefix: str):
        """Log administrative actions"""
        logger.info(
            f"Admin action - IP: {request.client.host}, "
            f"Action: {action}, Resource: {resource}, "
            f"API Key: {api_key_prefix}..."
        )
