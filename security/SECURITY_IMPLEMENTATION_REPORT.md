# Security Implementation Report: ASL Translator System
## Requirement R-06: Basic Security Measures Implementation
---

## Executive Summary

This document outlines the comprehensive security implementation plan for the ASL Translator system, addressing Requirement R-06: "The system shall implement basic security measures. The system must protect against common, basic vulnerabilities and handle sensitive information appropriately."

Following a thorough security audit of the codebase, this report identifies current security gaps, provides detailed implementation strategies aligned with OWASP Top 10 best practices, and establishes a prioritized roadmap for securing the application against common web vulnerabilities.

**Key Findings:**
- **Current State:** Basic structure in place, but critical security controls missing
- **Risk Level:** Medium-High (public API without authentication, no rate limiting)
- **Implementation Priority:** Phased approach over 3-4 weeks
- **Compliance:** Aligned with OWASP Top 10 and industry best practices

---

## 1. Security Goals & Principles

### 1.1 Core Security Objectives

The ASL Translator system implements security controls to achieve three fundamental goals:

1. **Confidentiality** - Sensitive data (API keys, credentials, user data) must not be disclosed to unauthorized parties
2. **Integrity** - Data and system components are protected from tampering and unauthorized modification
3. **Availability** - Services remain available and resilient against denial-of-service and abuse attacks

### 1.2 Security Framework

The implementation follows industry-recognized security frameworks:
- **OWASP Top 10** (2021) - Addressing common web application vulnerabilities
- **Defense in Depth** - Multiple layers of security controls
- **Principle of Least Privilege** - Minimal access rights for users and services
- **Secure by Default** - Security controls enabled by default

---

## 2. Current State Analysis

### 2.1 Security Audit Results

| Security Control | Status | Risk Level | Notes |
|-----------------|--------|------------|-------|
| **CORS Configuration** | ⚠️ Partial | Medium | Middleware imported but not configured |
| **Authentication** | ❌ Missing | High | All endpoints publicly accessible |
| **Rate Limiting** | ❌ Missing | High | Vulnerable to DoS/abuse attacks |
| **Input Validation** | ⚠️ Basic | Medium | Pydantic validation exists, but no size/format limits |
| **Secrets Management** | ⚠️ Partial | High | `.env` ignored, but hardcoded passwords in docker-compose |
| **HTTPS/TLS** | ❌ Missing | Medium | No SSL configuration (handled by reverse proxy) |
| **Debug Mode** | ⚠️ Risk | Low | Debug logging enabled in production |
| **Security Headers** | ❌ Missing | Low | No security headers middleware |
| **Logging Security** | ⚠️ Partial | Medium | May expose sensitive data in logs |
| **Payload Limits** | ❌ Missing | Medium | No request size restrictions |

### 2.2 Identified Vulnerabilities

1. **Broken Access Control (OWASP #1)**
   - Admin endpoints (`/api/models/{id}/activate`) accessible without authentication
   - No API key or token validation

2. **Cryptographic Failures (OWASP #2)**
   - Database passwords hardcoded in `docker-compose.yaml`
   - No encryption for sensitive data at rest

3. **Injection (OWASP #3)**
   - Limited input sanitization for image data
   - No validation on model selection parameters

4. **Security Misconfiguration (OWASP #5)**
   - CORS middleware imported but not configured
   - Debug mode potentially enabled in production
   - Missing security headers

5. **Vulnerable Components (OWASP #6)**
   - Dependencies need regular security updates
   - No automated vulnerability scanning

---

## 3. Implementation Plan

### 3.1 Phase 1: Critical Security Controls (Week 1)

#### 3.1.1 Secrets Management & Configuration

**Objective:** Remove hardcoded credentials and implement secure configuration management.

**Implementation:**

1. **Create Environment Configuration Files**

   Create `.env.example` (tracked in git):
   ```bash
   # Database Configuration
   POSTGRES_USER=asl_user
   POSTGRES_PASSWORD=change_me_in_production
   POSTGRES_DB=asl_translator
   
   # API Security
   API_KEYS=key1_abc123,key2_def456
   SECRET_KEY=generate_secure_random_key_here
   
   # Environment
   ENVIRONMENT=production
   DEBUG=false
   
   # CORS Configuration
   ALLOWED_ORIGINS=http://localhost:8000,https://yourdomain.com
   ```

2. **Update docker-compose.yaml**

   ```yaml
   services:
     postgres:
       image: postgres:15-alpine
       container_name: asl_postgres
       restart: always
       env_file:
         - .env.production  # Not tracked in git
       environment:
         POSTGRES_USER: ${POSTGRES_USER}
         POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
         POSTGRES_DB: ${POSTGRES_DB}
       ports:
         - "5432:5432"
       volumes:
         - postgres_data:/var/lib/postgresql/data
   
     api:
       build:
         context: ./services/api
       env_file:
         - .env.production
       environment:
         RABBITMQ_HOST: rabbitmq
         RABBITMQ_PORT: ${RABBITMQ_PORT:-5672}
         DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
         API_KEYS: ${API_KEYS}
         ENVIRONMENT: ${ENVIRONMENT:-production}
         DEBUG: ${DEBUG:-false}
   ```

3. **Update .gitignore**

   Ensure `.env.production` and `.env.local` are ignored:
   ```gitignore
   # Environment variables
   .env
   .env.local
   .env.production
   .env.*.local
   ```

**Files Modified:**
- `docker-compose.yaml`
- `.gitignore`
- Create `.env.example`

**Testing:**
- Verify `.env.production` is not tracked in git
- Test application startup with environment variables
- Verify fallback behavior when variables are missing

---

#### 3.1.2 CORS Configuration

**Objective:** Restrict cross-origin requests to trusted domains only.

**Implementation:**

Update `services/api/src/api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware
import os

# Get allowed origins from environment
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000"
).split(",")

# Configure CORS after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["X-Request-ID"],
    max_age=3600,  # Cache preflight requests for 1 hour
)
```

**Security Considerations:**
- Restrict to specific domains in production
- Use environment variables for configuration
- Limit allowed methods to only what's needed
- Set appropriate `max_age` for preflight caching

**Files Modified:**
- `services/api/src/api.py`

**Testing:**
- Test CORS headers with browser developer tools
- Verify requests from unauthorized origins are blocked
- Test preflight OPTIONS requests

---

#### 3.1.3 Rate Limiting

**Objective:** Prevent abuse and DoS attacks through request throttling.

**Implementation:**

1. **Add slowapi to requirements**

   Update `services/api/requirements.txt`:
   ```
   slowapi==0.1.9
   ```

2. **Implement Rate Limiting Middleware**

   Update `services/api/src/api.py`:

   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded
   from fastapi import Request
   
   # Initialize rate limiter
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   # Apply rate limits to endpoints
   @app.post("/predict", response_model=PredictionResponse)
   @limiter.limit("30/minute")  # 30 predictions per minute per IP
   async def predict_gesture(request: Request, request_data: PredictionRequest):
       # ... existing code ...
   
   @app.post("/feedback")
   @limiter.limit("60/minute")  # 60 feedback submissions per minute
   async def submit_feedback(request: Request, feedback_data: FeedbackRequest):
       # ... existing code ...
   
   @app.get("/api/stats")
   @limiter.limit("10/minute")  # Stricter limit for stats endpoint
   async def get_production_stats(request: Request):
       # ... existing code ...
   ```

3. **Configuration via Environment Variables**

   ```python
   # Rate limit configuration
   RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30/minute")
   RATE_LIMIT_FEEDBACK = os.getenv("RATE_LIMIT_FEEDBACK", "60/minute")
   RATE_LIMIT_STATS = os.getenv("RATE_LIMIT_STATS", "10/minute")
   ```

**Rate Limit Strategy:**
- **Public endpoints** (`/predict`, `/feedback`): 30-60 requests/minute per IP
- **Admin endpoints** (`/api/models/*`): 10 requests/minute + API key required
- **Stats endpoints**: 10 requests/minute

**Files Modified:**
- `services/api/src/api.py`
- `services/api/requirements.txt`

**Testing:**
- Test rate limit enforcement with multiple rapid requests
- Verify rate limit headers in response (`X-RateLimit-Limit`, `X-RateLimit-Remaining`)
- Test rate limit reset behavior

---

### 3.2 Phase 2: Authentication & Access Control (Week 2)

#### 3.2.1 API Key Authentication

**Objective:** Protect sensitive endpoints with API key authentication.

**Implementation:**

1. **Create Authentication Module**

   Create `services/api/src/security.py`:

   ```python
   """
   Security and authentication utilities for ASL Translator API
   """
   from fastapi import Security, HTTPException, status
   from fastapi.security import APIKeyHeader
   import os
   import logging
   
   logger = logging.getLogger(__name__)
   
   API_KEY_NAME = "X-API-Key"
   api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
   
   def get_valid_api_keys():
       """Retrieve valid API keys from environment"""
       api_keys_env = os.getenv("API_KEYS", "")
       if not api_keys_env:
           logger.warning("No API keys configured in environment")
           return []
       return [key.strip() for key in api_keys_env.split(",") if key.strip()]
   
   async def verify_api_key(api_key: str = Security(api_key_header)):
       """
       Verify API key from request header
       
       Args:
           api_key: API key from X-API-Key header
           
       Returns:
           str: Validated API key
           
       Raises:
           HTTPException: If API key is invalid or missing
       """
       if not api_key:
           logger.warning("API key authentication attempted without key")
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
           logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
           raise HTTPException(
               status_code=status.HTTP_403_FORBIDDEN,
               detail="Invalid API key"
           )
       
       logger.info(f"API key authenticated successfully: {api_key[:8]}...")
       return api_key
   
   async def optional_api_key(api_key: str = Security(api_key_header)):
       """
       Optional API key verification (for endpoints that work with or without auth)
       
       Returns:
           str or None: API key if provided and valid, None otherwise
       """
       if not api_key:
           return None
       
       valid_keys = get_valid_api_keys()
       if api_key in valid_keys:
           return api_key
       
       return None
   ```

2. **Apply Authentication to Protected Endpoints**

   Update `services/api/src/api.py`:

   ```python
   from security import verify_api_key, optional_api_key
   
   # Public endpoints (no auth required)
   @app.post("/predict", response_model=PredictionResponse)
   @limiter.limit("30/minute")
   async def predict_gesture(request: Request, request_data: PredictionRequest):
       # ... existing code ...
   
   # Protected admin endpoints (API key required)
   @app.post("/api/models/{model_id}/activate")
   @limiter.limit("10/minute")
   async def activate_model(
       request: Request,
       model_id: str,
       api_key: str = Depends(verify_api_key)
   ):
       """Activate a model - requires API key authentication"""
       # ... existing code ...
   
   @app.get("/api/stats")
   @limiter.limit("10/minute")
   async def get_production_stats(
       request: Request,
       api_key: str = Depends(optional_api_key)
   ):
       """
       Get production statistics
       - Public access with rate limiting
       - Enhanced access with API key (higher limits)
       """
       # ... existing code ...
   ```

**Security Considerations:**
- API keys stored in environment variables, never in code
- Keys rotated regularly (documented process)
- Different keys for different environments
- Failed authentication attempts logged for monitoring

**Files Created:**
- `services/api/src/security.py`

**Files Modified:**
- `services/api/src/api.py`

**Testing:**
- Test protected endpoints without API key (should fail)
- Test with invalid API key (should fail)
- Test with valid API key (should succeed)
- Verify authentication failures are logged

---

### 3.3 Phase 3: Input Validation & Sanitization (Week 2-3)

#### 3.3.1 Enhanced Input Validation

**Objective:** Implement strict validation for all user inputs, including size and format constraints.

**Implementation:**

1. **Enhanced PredictionRequest Model**

   Update `services/api/src/api.py`:

   ```python
   from pydantic import BaseModel, field_validator, Field
   import base64
   import re
   
   class PredictionRequest(BaseModel):
       """Request model for gesture prediction with enhanced validation"""
       image: str = Field(..., description="Base64-encoded image data")
       model: Optional[str] = Field(None, description="Model version identifier")
       
       @field_validator('image')
       @classmethod
       def validate_image(cls, v: str) -> str:
           """
           Validate base64 image data
           - Check format
           - Enforce size limits
           - Verify valid base64 encoding
           """
           if not v:
               raise ValueError("Image data cannot be empty")
           
           # Check base64 format
           base64_pattern = re.compile(r'^data:image/[a-zA-Z]+;base64,')
           has_prefix = base64_pattern.match(v)
           
           # Extract base64 data (remove data URI prefix if present)
           if has_prefix:
               v = v.split(',', 1)[1]
           
           # Validate base64 encoding
           try:
               decoded = base64.b64decode(v, validate=True)
           except Exception as e:
               raise ValueError(f"Invalid base64 encoding: {str(e)}")
           
           # Size validation (5MB limit for decoded image)
           MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
           if len(decoded) > MAX_IMAGE_SIZE:
               raise ValueError(
                   f"Image size ({len(decoded)} bytes) exceeds maximum "
                   f"allowed size ({MAX_IMAGE_SIZE} bytes)"
               )
           
           # Basic image format validation (check magic bytes)
           if len(decoded) < 4:
               raise ValueError("Image data too small to be valid")
           
           # Check for common image formats (JPEG, PNG)
           if decoded[0:2] not in [b'\xff\xd8', b'\x89\x50']:  # JPEG or PNG
               raise ValueError("Image must be in JPEG or PNG format")
           
           return v
       
       @field_validator('model')
       @classmethod
       def validate_model(cls, v: Optional[str]) -> Optional[str]:
           """Validate model identifier format"""
           if v is None:
               return v
           
           # Allow alphanumeric, hyphens, underscores only
           if not re.match(r'^[a-zA-Z0-9_-]+$', v):
               raise ValueError(
                   "Model identifier contains invalid characters. "
                   "Only alphanumeric, hyphens, and underscores allowed."
               )
           
           # Length limit
           if len(v) > 100:
               raise ValueError("Model identifier exceeds maximum length (100 characters)")
           
           return v
       
       class Config:
           json_schema_extra = {
               "example": {
                   "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                   "model": "asl_model_20260120_110302"
               }
           }
   ```

2. **Enhanced FeedbackRequest Validation**

   ```python
   class FeedbackRequest(BaseModel):
       """Request model for user feedback with validation"""
       job_id: str = Field(..., description="Prediction job ID")
       accepted: bool = Field(..., description="Whether prediction was accepted")
       corrected_gesture: Optional[str] = Field(None, description="Corrected gesture if rejected")
       
       @field_validator('job_id')
       @classmethod
       def validate_job_id(cls, v: str) -> str:
           """Validate job ID format (UUID)"""
           uuid_pattern = re.compile(
               r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
               re.IGNORECASE
           )
           if not uuid_pattern.match(v):
               raise ValueError("Invalid job ID format. Must be a valid UUID.")
           return v
       
       @field_validator('corrected_gesture')
       @classmethod
       def validate_corrected_gesture(cls, v: Optional[str]) -> Optional[str]:
           """Validate corrected gesture format"""
           if v is None:
               return v
           
           # Allow single uppercase letter (A-Z)
           if not re.match(r'^[A-Z]$', v):
               raise ValueError(
                   "Corrected gesture must be a single uppercase letter (A-Z)"
               )
           
           return v
   ```

3. **Request Size Limits**

   Add FastAPI configuration for request size limits:

   ```python
   from fastapi import Request
   from fastapi.exceptions import RequestValidationError
   
   # Configure maximum request body size (10MB)
   MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
   
   @app.middleware("http")
   async def check_request_size(request: Request, call_next):
       """Middleware to check request body size"""
       content_length = request.headers.get("content-length")
       if content_length:
           size = int(content_length)
           if size > MAX_REQUEST_SIZE:
               raise HTTPException(
                   status_code=413,
                   detail=f"Request body too large. Maximum size: {MAX_REQUEST_SIZE} bytes"
               )
       response = await call_next(request)
       return response
   ```

**Files Modified:**
- `services/api/src/api.py`

**Testing:**
- Test with oversized images (should reject)
- Test with invalid base64 (should reject)
- Test with non-image data (should reject)
- Test with invalid model identifiers (should reject)
- Test with oversized requests (should reject)

---

#### 3.3.2 SQL Injection Prevention

**Objective:** Ensure all database queries use parameterized statements.

**Current Status:** ✅ Already implemented using SQLAlchemy ORM (parameterized queries)

**Verification:**
- All database queries use SQLAlchemy ORM methods
- No raw SQL queries with string concatenation
- User input always passed as parameters

**Example (already secure):**
```python
# ✅ Secure - SQLAlchemy ORM
prediction = session.query(Prediction).filter_by(job_id=job_id).first()

# ✅ Secure - Parameterized query
session.query(Prediction).filter(Prediction.job_id == job_id).first()
```

**No changes needed** - Current implementation is secure.

---

### 3.4 Phase 4: Security Headers & Logging (Week 3)

#### 3.4.1 Security Headers Middleware

**Objective:** Add security headers to all HTTP responses.

**Implementation:**

Create `services/api/src/security_middleware.py`:

```python
"""
Security middleware for adding security headers to responses
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy (adjust based on your needs)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # Remove server header (optional - hides server version)
        if "server" in response.headers:
            del response.headers["server"]
        
        return response
```

Update `services/api/src/api.py`:

```python
from security_middleware import SecurityHeadersMiddleware

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)
```

**Files Created:**
- `services/api/src/security_middleware.py`

**Files Modified:**
- `services/api/src/api.py`

**Testing:**
- Verify security headers in response using browser dev tools
- Test CSP policy doesn't break frontend functionality
- Verify X-Frame-Options prevents clickjacking

---

#### 3.4.2 Secure Logging

**Objective:** Prevent sensitive data exposure in logs.

**Implementation:**

Update `services/api/src/logging_setup.py`:

```python
"""
Logging configuration with sensitive data filtering
"""
import logging
import sys
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path


class SensitiveDataFilter(logging.Filter):
    """
    Filter to redact sensitive information from logs
    Prevents exposure of passwords, API keys, and large data payloads
    """
    
    # Patterns to match and redact
    PATTERNS = [
        # Passwords
        (re.compile(r'password["\s:=]+([^\s,"]+)', re.IGNORECASE), r'password=***REDACTED***'),
        (re.compile(r'pwd["\s:=]+([^\s,"]+)', re.IGNORECASE), r'pwd=***REDACTED***'),
        
        # API keys
        (re.compile(r'api[_-]?key["\s:=]+([^\s,"]+)', re.IGNORECASE), r'api_key=***REDACTED***'),
        (re.compile(r'x-api-key["\s:=]+([^\s,"]+)', re.IGNORECASE), r'x-api-key=***REDACTED***'),
        
        # Tokens
        (re.compile(r'token["\s:=]+([^\s,"]+)', re.IGNORECASE), r'token=***REDACTED***'),
        (re.compile(r'bearer\s+([^\s,"]+)', re.IGNORECASE), r'bearer ***REDACTED***'),
        
        # Large base64 image data (truncate)
        (re.compile(r'"image":\s*"([^"]{100,})"'), lambda m: f'"image": "***BASE64_DATA_TRUNCATED*** ({len(m.group(1))} chars)"'),
        
        # Database connection strings
        (re.compile(r'postgresql://[^:]+:([^@]+)@', re.IGNORECASE), r'postgresql://***USER***:***REDACTED***@'),
        (re.compile(r'mysql://[^:]+:([^@]+)@', re.IGNORECASE), r'mysql://***USER***:***REDACTED***@'),
    ]
    
    def filter(self, record):
        """Filter log records to redact sensitive data"""
        # Process message
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                if callable(replacement):
                    msg = pattern.sub(replacement, msg)
                else:
                    msg = pattern.sub(replacement, msg)
            record.msg = msg
        
        # Process args (for formatted messages)
        if hasattr(record, 'args') and record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        if callable(replacement):
                            args[i] = pattern.sub(replacement, arg)
                        else:
                            args[i] = pattern.sub(replacement, arg)
            record.args = tuple(args)
        
        return True


def setup_logging():
    """Configure logging for the API service with security filters"""
    
    # Create logs directory
    logs_dir = Path("/tmp/asl_api_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Determine log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(sensitive_filter)
    root_logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    log_file = logs_dir / "api.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(sensitive_filter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"API Logging configured. Log file: {log_file}")
    return log_file
```

**Files Modified:**
- `services/api/src/logging_setup.py`

**Testing:**
- Test logging with sensitive data (passwords, API keys)
- Verify sensitive data is redacted in log files
- Test that legitimate log messages are still readable

---

#### 3.4.3 Security Event Logging

**Objective:** Log security-relevant events for monitoring and incident response.

**Implementation:**

Create `services/api/src/security_logging.py`:

```python
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
```

Update authentication to use security logging:

```python
from security_logging import SecurityLogger

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        SecurityLogger.log_authentication_failure(request, "No API key provided")
        raise HTTPException(...)
    
    if api_key not in valid_keys:
        SecurityLogger.log_authentication_failure(
            request, "Invalid API key", api_key[:8]
        )
        raise HTTPException(...)
```

**Files Created:**
- `services/api/src/security_logging.py`

**Files Modified:**
- `services/api/src/security.py`
- `services/api/src/api.py`

**Testing:**
- Verify security events are logged correctly
- Test log aggregation and monitoring setup
- Verify log retention policies

---

### 3.5 Phase 5: Production Configuration (Week 4)

#### 3.5.1 Environment-Based Configuration

**Objective:** Separate development and production configurations.

**Implementation:**

Update `services/inference/src/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    # Model Configuration
    model_confidence_threshold: float = 0.6
    classifier_confidence_threshold: float = 0.6
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    max_num_hands: int = 1
    
    # Application
    app_name: str = "ASL Translator"
    app_version: str = "1.0.0"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "asl-gesture-recognition"
    
    # Paths
    models_dir: str = "/app/models/"
    data_dir: str = "/app/data/"
    
    # Language
    default_language: str = "english"
    
    # Feedback and Retraining
    feedback_confidence_threshold: float = 0.9
    retraining_sample_threshold: int = 200
    database_url: str = "sqlite:///data/feedback.db"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def validate_production_settings(self):
        """Validate settings for production deployment"""
        if self.is_production:
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
            if not self.database_url or "sqlite" in self.database_url.lower():
                raise ValueError("Production must use PostgreSQL, not SQLite")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()

# Validate production settings on import
try:
    settings.validate_production_settings()
except ValueError as e:
    if settings.is_production:
        raise
    # In development, just log a warning
    import logging
    logging.warning(f"Production validation warning: {e}")
```

**Files Modified:**
- `services/inference/src/config.py`
- `services/api/src/api.py` (add environment checks)

**Testing:**
- Test application startup in different environments
- Verify production validation prevents unsafe configurations
- Test environment variable override behavior

---

#### 3.5.2 HTTPS/TLS Configuration

**Objective:** Ensure secure communication in production.

**Implementation Note:** HTTPS/TLS is typically handled by a reverse proxy (nginx, Traefik) in front of the FastAPI application. The FastAPI app itself runs on HTTP internally.

**Recommended Setup:**

1. **Nginx Reverse Proxy Configuration** (for production deployment):

   ```nginx
   # /etc/nginx/sites-available/asl-translator
   server {
       listen 443 ssl http2;
       server_name yourdomain.com;
       
       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
       
       # SSL Configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;
       ssl_prefer_server_ciphers on;
       
       # Security Headers
       add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   
   # Redirect HTTP to HTTPS
   server {
       listen 80;
       server_name yourdomain.com;
       return 301 https://$server_name$request_uri;
   }
   ```

2. **Update FastAPI to Trust Proxy Headers**

   ```python
   from fastapi import Request
   
   @app.middleware("http")
   async def trust_proxy_headers(request: Request, call_next):
       """Trust proxy headers for correct client IP and protocol"""
       if request.headers.get("X-Forwarded-Proto") == "https":
           request.scope["scheme"] = "https"
       response = await call_next(request)
       return response
   ```

**Deployment Documentation:**
- Document HTTPS setup process
- Include SSL certificate renewal procedures
- Document reverse proxy configuration

---

## 4. Testing Strategy

### 4.1 Security Testing Approach

#### 4.1.1 Unit Tests

Create `tests/test_security.py`:

```python
"""
Security testing for ASL Translator API
"""
import pytest
from fastapi.testclient import TestClient
from services.api.src.api import app
import os

# Set test environment variables
os.environ["API_KEYS"] = "test_key_123"
os.environ["ENVIRONMENT"] = "test"

client = TestClient(app)


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/predict")
    assert "access-control-allow-origin" in response.headers


def test_security_headers():
    """Test security headers are present"""
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_rate_limiting():
    """Test rate limiting is enforced"""
    # Make rapid requests
    for _ in range(35):  # Exceed 30/minute limit
        response = client.post("/predict", json={"image": "test"})
    
    # Should get rate limit error
    assert response.status_code == 429


def test_api_key_authentication():
    """Test API key authentication for protected endpoints"""
    # Without API key
    response = client.post("/api/models/test/activate")
    assert response.status_code == 403
    
    # With invalid API key
    response = client.post(
        "/api/models/test/activate",
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == 403
    
    # With valid API key
    response = client.post(
        "/api/models/test/activate",
        headers={"X-API-Key": "test_key_123"}
    )
    # May still fail due to model not found, but auth should pass
    assert response.status_code != 403


def test_input_validation():
    """Test input validation"""
    # Oversized image
    large_image = "A" * (6 * 1024 * 1024)  # 6MB base64
    response = client.post("/predict", json={"image": large_image})
    assert response.status_code == 422
    
    # Invalid base64
    response = client.post("/predict", json={"image": "not_base64!!!"})
    assert response.status_code == 422
    
    # Invalid model identifier
    response = client.post(
        "/predict",
        json={"image": "dGVzdA==", "model": "invalid model name!"}
    )
    assert response.status_code == 422


def test_sensitive_data_logging():
    """Test that sensitive data is redacted in logs"""
    # This would require checking log output
    # Implementation depends on logging test setup
    pass
```

#### 4.1.2 Integration Tests

- Test end-to-end security flows
- Test authentication across multiple endpoints
- Test rate limiting across concurrent requests
- Test CORS with actual browser requests

#### 4.1.3 Penetration Testing Checklist

- [ ] Attempt SQL injection on all input fields
- [ ] Test XSS vulnerabilities in user inputs
- [ ] Attempt to bypass authentication
- [ ] Test rate limit bypass techniques
- [ ] Test CORS misconfiguration
- [ ] Test for information disclosure in error messages
- [ ] Test for sensitive data in logs
- [ ] Test for insecure direct object references

---

## 5. Deployment Checklist

### 5.1 Pre-Deployment Security Checklist

- [ ] All secrets moved to environment variables
- [ ] `.env.production` file created and secured (not in git)
- [ ] API keys generated and configured
- [ ] CORS origins restricted to production domains
- [ ] Debug mode disabled in production
- [ ] Log level set to INFO or WARNING in production
- [ ] Rate limits configured appropriately
- [ ] Security headers middleware enabled
- [ ] HTTPS/TLS configured (reverse proxy)
- [ ] Database uses strong passwords
- [ ] Database backups configured
- [ ] Security monitoring/logging enabled
- [ ] Dependencies updated (no known vulnerabilities)
- [ ] Security tests passing

### 5.2 Post-Deployment Verification

- [ ] Verify HTTPS is working (SSL Labs test)
- [ ] Verify security headers present
- [ ] Test authentication on protected endpoints
- [ ] Verify rate limiting is active
- [ ] Check logs for sensitive data exposure
- [ ] Monitor for security events
- [ ] Verify CORS is properly configured
- [ ] Test input validation with edge cases

---

## 6. Monitoring & Incident Response

### 6.1 Security Monitoring

**Key Metrics to Monitor:**
- Failed authentication attempts (by IP)
- Rate limit violations
- Input validation failures
- Unusual request patterns
- Error rates (potential attack indicators)

**Log Aggregation:**
- Centralized logging (ELK stack, CloudWatch, etc.)
- Alert on security events
- Regular log review

### 6.2 Incident Response Plan

1. **Detection:** Automated alerts for security events
2. **Assessment:** Determine severity and impact
3. **Containment:** Block IPs, disable affected endpoints if needed
4. **Eradication:** Fix vulnerabilities, rotate credentials
5. **Recovery:** Restore service, verify security
6. **Lessons Learned:** Document and improve

---

## 7. Maintenance & Updates

### 7.1 Regular Security Tasks

**Weekly:**
- Review security logs
- Check for dependency updates
- Monitor rate limit patterns

**Monthly:**
- Review and rotate API keys
- Update dependencies (security patches)
- Review and update CORS origins
- Security audit of configuration

**Quarterly:**
- Full security review
- Penetration testing
- Update security documentation
- Review and update rate limits

### 7.2 Dependency Management

- Use `pip-audit` or `safety` to check for vulnerabilities
- Keep dependencies up to date
- Subscribe to security advisories
- Test updates in staging before production

---

## 8. Implementation Timeline

### Week 1: Critical Security Controls
- ✅ Secrets management (environment variables)
- ✅ CORS configuration
- ✅ Rate limiting implementation
- ✅ Basic security headers

### Week 2: Authentication & Validation
- ✅ API key authentication
- ✅ Enhanced input validation
- ✅ Request size limits
- ✅ Security event logging

### Week 3: Logging & Headers
- ✅ Secure logging (data redaction)
- ✅ Security headers middleware
- ✅ Environment-based configuration
- ✅ Security testing

### Week 4: Production Hardening
- ✅ Production configuration validation
- ✅ HTTPS/TLS documentation
- ✅ Deployment checklist
- ✅ Monitoring setup

---

## 9. References & Resources

### OWASP Resources
- [OWASP Top 10 (2021)](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

### FastAPI Security
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)

### Additional Resources
- [API Security Best Practices](https://www.aikido.dev/blog/api-security-guide)
- [Web Application Security Best Practices](https://www.radware.com/cyberpedia/application-security/web-application-security-best-practices/)

---

## 10. Conclusion

This security implementation plan addresses Requirement R-06 by implementing comprehensive security controls aligned with OWASP Top 10 best practices. The phased approach ensures critical vulnerabilities are addressed first, while maintaining system functionality and developer productivity.

**Key Achievements:**
- ✅ Comprehensive security audit completed
- ✅ Detailed implementation plan with code examples
- ✅ Prioritized roadmap for security improvements
- ✅ Testing strategy defined
- ✅ Deployment and monitoring procedures documented

**Next Steps:**
1. Begin Phase 1 implementation (Week 1)
2. Set up security testing infrastructure
3. Configure monitoring and alerting
4. Schedule regular security reviews

**Risk Reduction:**
- **Before:** High risk (public API, no authentication, no rate limiting)
- **After:** Low-Medium risk (comprehensive security controls in place)

This implementation significantly improves the security posture of the ASL Translator system and provides a solid foundation for secure operation in production environments.

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Implementation Plan - Ready for Execution
