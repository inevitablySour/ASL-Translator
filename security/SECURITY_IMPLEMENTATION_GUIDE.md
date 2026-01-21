# Security Implementation Guide - Step by Step
## Complete Walkthrough for ASL Translator Security Implementation

This guide provides exact step-by-step instructions to implement all security measures from the Security Implementation Report.

---

## Prerequisites

Before starting, ensure you have:
- ✅ Git repository initialized
- ✅ Docker and Docker Compose installed
- ✅ Python 3.11+ installed locally (for testing)
- ✅ Access to your project directory
- ✅ Basic understanding of FastAPI and Docker

**Estimated Time:** 4-6 hours total (can be done in phases)

---

## Phase 1: Critical Security Controls (Week 1)

### Step 1.1: Set Up Environment Variables

#### 1.1.1 Create `.env.example` file

**Location:** `ASL-Translator/.env.example`

```bash
# Navigate to project root
cd "c:\Users\erwin\OneDrive - Zuyd Hogeschool\zuyd STUDIE saved\year2\ai_ops\proj2.0\ASL-Translator"
```

**Create the file:**

```bash
# Windows PowerShell
New-Item -Path ".env.example" -ItemType File -Force
```

**Add content to `.env.example`:**

```env
# Database Configuration
POSTGRES_USER=asl_user
POSTGRES_PASSWORD=change_me_in_production_use_strong_password
POSTGRES_DB=asl_translator

# API Security
API_KEYS=dev_key_abc123,admin_key_def456
SECRET_KEY=generate_secure_random_key_here_min_32_chars

# Environment
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# CORS Configuration (comma-separated)
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Rate Limiting
RATE_LIMIT_PREDICT=30/minute
RATE_LIMIT_FEEDBACK=60/minute
RATE_LIMIT_STATS=10/minute
RATE_LIMIT_ADMIN=10/minute
```

#### 1.1.2 Create `.env.production` file (NOT tracked in git)

```bash
# Copy example to production (you'll customize this)
Copy-Item ".env.example" ".env.production"
```

**Edit `.env.production`** with your actual production values:
- Generate a strong password for `POSTGRES_PASSWORD`
- Generate secure API keys for `API_KEYS`
- Set `ENVIRONMENT=production`
- Set `DEBUG=false`
- Update `ALLOWED_ORIGINS` with your production domain

#### 1.1.3 Update `.gitignore`

**Location:** `ASL-Translator/.gitignore`

**Add these lines if not already present:**

```gitignore
# Environment variables
.env
.env.local
.env.production
.env.*.local
.env.*.production
```

**Verify it's working:**

```bash
# Check if .env.production is ignored
git status
# Should NOT show .env.production
```

---

### Step 1.2: Update docker-compose.yaml

**Location:** `ASL-Translator/docker-compose.yaml`

**Replace the entire file with:**

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: asl_postgres
    restart: always
    env_file:
      - .env.production
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 2s
      timeout: 5s
      retries: 10
      start_period: 10s
  
  api:
    build: 
      context: ./services/api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models:ro
    env_file:
      - .env.production
    environment:
      RABBITMQ_HOST: rabbitmq
      RABBITMQ_PORT: ${RABBITMQ_PORT:-5672}
      MODEL_QUEUE: inference_queue
      API_QUEUE: api_queue
      DATA_QUEUE: data_queue
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      API_KEYS: ${API_KEYS}
      ENVIRONMENT: ${ENVIRONMENT:-production}
      DEBUG: ${DEBUG:-false}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      ALLOWED_ORIGINS: ${ALLOWED_ORIGINS:-http://localhost:8000}
      RATE_LIMIT_PREDICT: ${RATE_LIMIT_PREDICT:-30/minute}
      RATE_LIMIT_FEEDBACK: ${RATE_LIMIT_FEEDBACK:-60/minute}
      RATE_LIMIT_STATS: ${RATE_LIMIT_STATS:-10/minute}
      RATE_LIMIT_ADMIN: ${RATE_LIMIT_ADMIN:-10/minute}
    depends_on:
      - rabbitmq
      - postgres
  
  inference:
    build: 
      context: ./services/inference
    volumes:
      - ./models:/app/models:ro
      - ./services/api/src:/app/api_src:ro
      - ./data:/app/data:ro
    env_file:
      - .env.production
    environment:
      RABBITMQ_HOST: rabbitmq
      RABBITMQ_PORT: ${RABBITMQ_PORT:-5672}
      MODEL_QUEUE: inference_queue
      API_QUEUE: api_queue
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      ENVIRONMENT: ${ENVIRONMENT:-production}
      DEBUG: ${DEBUG:-false}
    depends_on:
      - postgres
      - rabbitmq
  
  training:
    build:
      context: ./services/training
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./services/api/src:/app/api_src:ro
    env_file:
      - .env.production
    environment:
      RABBITMQ_HOST: rabbitmq
      PYTHONUNBUFFERED: "1"
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      ENVIRONMENT: ${ENVIRONMENT:-production}
      DEBUG: ${DEBUG:-false}
    depends_on:
      - rabbitmq
      - postgres
      - api
  
  rabbitmq: 
    image: rabbitmq:3-management
    container_name: rabbitmq
    restart: always
    ports:
      - "5672:5672"
      - "15672:15672"

volumes:
  postgres_data:
```

**Test the configuration:**

```bash
# Validate docker-compose syntax
docker compose -f docker-compose.yaml config
```

---

### Step 1.3: Add Rate Limiting Dependency

**Location:** `ASL-Translator/services/api/requirements.txt`

**Add this line:**

```
slowapi==0.1.9
```

**Verify the file looks correct:**

```bash
# View the file
cat services/api/requirements.txt
# Should include slowapi==0.1.9
```

---

### Step 1.4: Implement Rate Limiting in API

**Location:** `ASL-Translator/services/api/src/api.py`

**Step 1.4.1: Add imports at the top (around line 6)**

Find this line:
```python
from fastapi.middleware.cors import CORSMiddleware
```

**Add these imports right after it:**

```python
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
```

**Step 1.4.2: Initialize rate limiter (after app creation, around line 112)**

Find this line:
```python
app = FastAPI(lifespan=lifespan)
```

**Add right after it:**

```python
app = FastAPI(lifespan=lifespan)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get rate limit configuration from environment
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30/minute")
RATE_LIMIT_FEEDBACK = os.getenv("RATE_LIMIT_FEEDBACK", "60/minute")
RATE_LIMIT_STATS = os.getenv("RATE_LIMIT_STATS", "10/minute")
RATE_LIMIT_ADMIN = os.getenv("RATE_LIMIT_ADMIN", "10/minute")
```

**Step 1.4.3: Add rate limiting decorators to endpoints**

**Find the `/predict` endpoint (around line 135):**

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: PredictionRequest):
```

**Change to:**

```python
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict_gesture(request: Request, request_data: PredictionRequest):
```

**Update the function body** - change `request` parameter references:

Find:
```python
async def predict_gesture(request: PredictionRequest):
    start_time = time.time()
    global connection_producer, connection_consumer
    job_id = create_ID()
    
    try:
        logger.info(f"[JOB {job_id}] Received prediction request")
        
        payload = {"job_id": job_id, "image": request.image, "model": request.model}
```

**Change to:**

```python
async def predict_gesture(request: Request, request_data: PredictionRequest):
    start_time = time.time()
    global connection_producer, connection_consumer
    job_id = create_ID()
    
    try:
        logger.info(f"[JOB {job_id}] Received prediction request")
        
        payload = {"job_id": job_id, "image": request_data.image, "model": request_data.model}
```

**Find the `/feedback` endpoint (around line 203):**

```python
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
```

**Change to:**

```python
@app.post("/feedback")
@limiter.limit(RATE_LIMIT_FEEDBACK)
async def submit_feedback(request: Request, feedback_data: FeedbackRequest):
```

**Update the function body:**

Find:
```python
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a prediction
    Only accepts feedback for predictions with landmarks (high confidence)
    """
    if not feedback_manager:
        raise HTTPException(status_code=500, detail="Feedback system not available")
    
    try:
        logger.info(f"Received feedback for job {request.job_id}: accepted={request.accepted}")
        
        result = feedback_manager.submit_feedback(
            job_id=request.job_id,
            accepted=request.accepted,
            corrected_gesture=request.corrected_gesture
        )
```

**Change to:**

```python
async def submit_feedback(request: Request, feedback_data: FeedbackRequest):
    """
    Submit user feedback for a prediction
    Only accepts feedback for predictions with landmarks (high confidence)
    """
    if not feedback_manager:
        raise HTTPException(status_code=500, detail="Feedback system not available")
    
    try:
        logger.info(f"Received feedback for job {feedback_data.job_id}: accepted={feedback_data.accepted}")
        
        result = feedback_manager.submit_feedback(
            job_id=feedback_data.job_id,
            accepted=feedback_data.accepted,
            corrected_gesture=feedback_data.corrected_gesture
        )
```

**Find the `/api/stats` endpoint (around line 266):**

```python
@app.get("/api/stats")
async def get_production_stats():
```

**Change to:**

```python
@app.get("/api/stats")
@limiter.limit(RATE_LIMIT_STATS)
async def get_production_stats(request: Request):
```

**Find the `/api/models/{model_id}/activate` endpoint (around line 392):**

```python
@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str):
```

**Change to:**

```python
@app.post("/api/models/{model_id}/activate")
@limiter.limit(RATE_LIMIT_ADMIN)
async def activate_model(request: Request, model_id: str):
```

**Save the file and verify syntax:**

```bash
# Check Python syntax
python -m py_compile services/api/src/api.py
```

---

### Step 1.5: Implement CORS Configuration

**Location:** `ASL-Translator/services/api/src/api.py`

**Find where the app is created (around line 112):**

```python
app = FastAPI(lifespan=lifespan)
```

**Add CORS configuration right after rate limiter initialization:**

```python
# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get rate limit configuration from environment
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30/minute")
RATE_LIMIT_FEEDBACK = os.getenv("RATE_LIMIT_FEEDBACK", "60/minute")
RATE_LIMIT_STATS = os.getenv("RATE_LIMIT_STATS", "10/minute")
RATE_LIMIT_ADMIN = os.getenv("RATE_LIMIT_ADMIN", "10/minute")

# Configure CORS
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000"
).split(",")

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

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 1.6: Test Phase 1 Implementation

**1.6.1: Rebuild Docker containers**

```bash
# Stop existing containers
docker compose down

# Rebuild with new dependencies
docker compose build api

# Start services
docker compose up -d
```

**1.6.2: Test rate limiting**

```bash
# Install httpie or use curl
# Windows PowerShell - install httpie first: pip install httpie

# Test normal request (should work)
http POST http://localhost:8000/predict image=="test" --print=HhBb

# Rapid requests (should hit rate limit after 30)
# Create test script: test_rate_limit.ps1
```

**Create `test_rate_limit.ps1`:**

```powershell
# Test rate limiting
for ($i=1; $i -le 35; $i++) {
    Write-Host "Request $i"
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
            -Method POST `
            -ContentType "application/json" `
            -Body '{"image":"dGVzdA=="}' `
            -ErrorAction Stop
        Write-Host "Status: $($response.StatusCode)"
    } catch {
        Write-Host "Error: $($_.Exception.Response.StatusCode.value__)"
        if ($_.Exception.Response.StatusCode.value__ -eq 429) {
            Write-Host "Rate limit hit!"
            break
        }
    }
    Start-Sleep -Milliseconds 100
}
```

**Run the test:**

```bash
powershell -ExecutionPolicy Bypass -File test_rate_limit.ps1
```

**1.6.3: Test CORS headers**

Open browser console and test:

```javascript
// In browser console at http://localhost:8000
fetch('http://localhost:8000/api/stats', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
})
.then(r => {
    console.log('CORS Headers:', r.headers.get('access-control-allow-origin'));
    return r.json();
})
.then(data => console.log('Data:', data));
```

**Expected:** Should see CORS headers in response.

---

## Phase 2: Authentication & Access Control (Week 2)

### Step 2.1: Create Security Module

**Location:** `ASL-Translator/services/api/src/security.py`

**Create new file:**

```bash
New-Item -Path "services/api/src/security.py" -ItemType File -Force
```

**Add this content:**

```python
"""
Security and authentication utilities for ASL Translator API
"""
from fastapi import Security, HTTPException, status, Request
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
        logger.warning(f"API key authentication attempted without key from IP: {request.client.host}")
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
        logger.warning(f"Invalid API key attempted from IP: {request.client.host}, Key prefix: {api_key[:8]}...")
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
```

**Save and verify:**

```bash
python -m py_compile services/api/src/security.py
```

---

### Step 2.2: Apply Authentication to Protected Endpoints

**Location:** `ASL-Translator/services/api/src/api.py`

**Step 2.2.1: Add import**

**Find imports section (around line 1-30), add:**

```python
from security import verify_api_key, optional_api_key
```

**Step 2.2.2: Protect admin endpoint**

**Find the `/api/models/{model_id}/activate` endpoint:**

```python
@app.post("/api/models/{model_id}/activate")
@limiter.limit(RATE_LIMIT_ADMIN)
async def activate_model(request: Request, model_id: str):
```

**Change to:**

```python
@app.post("/api/models/{model_id}/activate")
@limiter.limit(RATE_LIMIT_ADMIN)
async def activate_model(
    request: Request,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
```

**Add Depends import if not present:**

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
```

**Step 2.2.3: Make stats endpoint optionally authenticated**

**Find the `/api/stats` endpoint:**

```python
@app.get("/api/stats")
@limiter.limit(RATE_LIMIT_STATS)
async def get_production_stats(request: Request):
```

**Change to:**

```python
@app.get("/api/stats")
@limiter.limit(RATE_LIMIT_STATS)
async def get_production_stats(
    request: Request,
    api_key: str = Depends(optional_api_key)
):
```

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 2.3: Test Authentication

**2.3.1: Rebuild and restart**

```bash
docker compose restart api
# Or rebuild if needed
docker compose build api && docker compose up -d api
```

**2.3.2: Test without API key (should fail)**

```bash
# PowerShell
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/models/test/activate" `
    -Method POST `
    -ErrorAction Stop
```

**Expected:** Should get 403 Forbidden

**2.3.3: Test with invalid API key (should fail)**

```bash
$headers = @{
    "X-API-Key" = "invalid_key_123"
}
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/models/test/activate" `
    -Method POST `
    -Headers $headers `
    -ErrorAction Stop
```

**Expected:** Should get 403 Forbidden

**2.3.4: Test with valid API key (should work)**

```bash
# Get API key from .env.production
# Assuming it's "dev_key_abc123" (first key in API_KEYS)

$headers = @{
    "X-API-Key" = "dev_key_abc123"
}
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/models/test/activate" `
        -Method POST `
        -Headers $headers `
        -ErrorAction Stop
    Write-Host "Success! Status: $($response.StatusCode)"
} catch {
    # May fail due to model not found, but should NOT be 403
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    if ($_.Exception.Response.StatusCode.value__ -eq 403) {
        Write-Host "ERROR: Authentication failed!"
    } else {
        Write-Host "Authentication passed (other error expected)"
    }
}
```

**Expected:** Should NOT get 403 (may get 404 if model doesn't exist, which is OK)

---

## Phase 3: Input Validation & Sanitization (Week 2-3)

### Step 3.1: Enhance PredictionRequest Model

**Location:** `ASL-Translator/services/api/src/api.py`

**Step 3.1.1: Update imports**

**Find imports section, add:**

```python
from pydantic import BaseModel, field_validator, Field
import re
```

**Step 3.1.2: Replace PredictionRequest class**

**Find this class (around line 57):**

```python
class PredictionRequest(BaseModel):
    image: str
    model: Optional[str] = None
```

**Replace with:**

```python
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
        
        # Check base64 format (may have data URI prefix)
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

**Step 3.1.3: Enhance FeedbackRequest**

**Find this class (around line 72):**

```python
class FeedbackRequest(BaseModel):
    job_id: str
    accepted: bool
    corrected_gesture: Optional[str] = None
```

**Replace with:**

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

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 3.2: Add Request Size Limits Middleware

**Location:** `ASL-Translator/services/api/src/api.py`

**Step 3.2.1: Add middleware after CORS configuration**

**Find where CORS middleware is added, add after it:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["X-Request-ID"],
    max_age=3600,
)

# Request size limit middleware
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

@app.middleware("http")
async def check_request_size(request: Request, call_next):
    """Middleware to check request body size"""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size: {MAX_REQUEST_SIZE} bytes"
                )
        except ValueError:
            pass  # Invalid content-length, let it through to be handled by FastAPI
    
    response = await call_next(request)
    return response
```

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 3.3: Test Input Validation

**3.3.1: Test oversized image**

```bash
# Create test with large base64 string
$largeImage = "A" * (6 * 1024 * 1024)  # 6MB
$body = @{
    image = $largeImage
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop
} catch {
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    # Should be 422 (validation error) or 413 (too large)
}
```

**Expected:** Should get 422 or 413 error

**3.3.2: Test invalid base64**

```bash
$body = @{
    image = "not_base64!!!"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop
} catch {
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    # Should be 422
}
```

**Expected:** Should get 422 validation error

**3.3.3: Test invalid model identifier**

```bash
$body = @{
    image = "dGVzdA=="
    model = "invalid model name!"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop
} catch {
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    # Should be 422
}
```

**Expected:** Should get 422 validation error

---

## Phase 4: Security Headers & Logging (Week 3)

### Step 4.1: Create Security Headers Middleware

**Location:** `ASL-Translator/services/api/src/security_middleware.py`

**Create new file:**

```bash
New-Item -Path "services/api/src/security_middleware.py" -ItemType File -Force
```

**Add this content:**

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

**Save and verify:**

```bash
python -m py_compile services/api/src/security_middleware.py
```

---

### Step 4.2: Add Security Headers to API

**Location:** `ASL-Translator/services/api/src/api.py`

**Step 4.2.1: Add import**

**Find imports section, add:**

```python
from security_middleware import SecurityHeadersMiddleware
```

**Step 4.2.2: Add middleware**

**Find where CORS middleware is added, add after it:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["X-Request-ID"],
    max_age=3600,
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)
```

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 4.3: Implement Secure Logging

**Location:** `ASL-Translator/services/api/src/logging_setup.py`

**Replace the entire file with:**

```python
"""
Logging configuration with sensitive data filtering
"""
import logging
import sys
import re
import os
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

**Save and verify:**

```bash
python -m py_compile services/api/src/logging_setup.py
```

---

### Step 4.4: Create Security Event Logging

**Location:** `ASL-Translator/services/api/src/security_logging.py`

**Create new file:**

```bash
New-Item -Path "services/api/src/security_logging.py" -ItemType File -Force
```

**Add this content:**

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

**Save and verify:**

```bash
python -m py_compile services/api/src/security_logging.py
```

---

### Step 4.5: Integrate Security Logging

**Location:** `ASL-Translator/services/api/src/security.py`

**Update the file to use security logging:**

**Add import at top:**

```python
from security_logging import SecurityLogger
```

**Update `verify_api_key` function:**

**Find:**

```python
    if not api_key:
        logger.warning(f"API key authentication attempted without key from IP: {request.client.host}")
        raise HTTPException(...)
```

**Change to:**

```python
    if not api_key:
        SecurityLogger.log_authentication_failure(request, "No API key provided")
        raise HTTPException(...)
```

**Find:**

```python
    if api_key not in valid_keys:
        logger.warning(f"Invalid API key attempted from IP: {request.client.host}, Key prefix: {api_key[:8]}...")
        raise HTTPException(...)
```

**Change to:**

```python
    if api_key not in valid_keys:
        SecurityLogger.log_authentication_failure(request, "Invalid API key", api_key[:8])
        raise HTTPException(...)
```

**Save and verify:**

```bash
python -m py_compile services/api/src/security.py
```

---

### Step 4.6: Test Security Headers and Logging

**4.6.1: Test security headers**

```bash
# PowerShell
$response = Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
$response.Headers

# Check for:
# - X-Content-Type-Options
# - X-Frame-Options
# - X-XSS-Protection
# - Content-Security-Policy
```

**Expected:** All security headers should be present

**4.6.2: Test secure logging**

```bash
# Make a request with API key
$headers = @{
    "X-API-Key" = "dev_key_abc123"
}
Invoke-WebRequest -Uri "http://localhost:8000/api/stats" `
    -Method GET `
    -Headers $headers

# Check logs (inside Docker container)
docker exec asl-translator-api-1 cat /tmp/asl_api_logs/api.log | Select-String "API key"
# Should see redacted API key (only first 8 chars)
```

**Expected:** API keys should be redacted in logs

---

## Phase 5: Production Configuration (Week 4)

### Step 5.1: Update Config for Environment Detection

**Location:** `ASL-Translator/services/inference/src/config.py`

**Update the Settings class:**

**Find:**

```python
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    model_confidence_threshold: float = 0.6

    # Application
    app_name: str = "ASL Translator"
    app_version: str = "1.0.0"
    debug: bool = False
```

**Add environment property:**

```python
class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    model_confidence_threshold: float = 0.6

    # Application
    app_name: str = "ASL Translator"
    app_version: str = "1.0.0"
    
    # ... rest of settings ...
    
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
```

**At the end of the file, add validation:**

```python
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

**Save and verify:**

```bash
python -m py_compile services/inference/src/config.py
```

---

### Step 5.2: Add Environment Checks to API

**Location:** `ASL-Translator/services/api/src/api.py`

**Add environment check at startup:**

**Find the lifespan function, add after connection setup:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_producer, connection_consumer, consumer_thread

    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    if environment == "production" and debug_mode:
        logger.error("CRITICAL: Debug mode enabled in production!")
        raise RuntimeError("Debug mode must be disabled in production")
    
    # Setup the connection with the producer and the consumer
    connection_producer = producer.connect_with_broker()
    connection_consumer = consumer.connect_with_broker()
    # ... rest of function ...
```

**Save and verify:**

```bash
python -m py_compile services/api/src/api.py
```

---

### Step 5.3: Final Testing and Verification

**5.3.1: Rebuild all services**

```bash
docker compose down
docker compose build
docker compose up -d
```

**5.3.2: Run comprehensive tests**

**Create `test_security_comprehensive.ps1`:**

```powershell
Write-Host "=== Security Implementation Tests ===" -ForegroundColor Green

# Test 1: CORS Headers
Write-Host "`n1. Testing CORS headers..." -ForegroundColor Yellow
$response = Invoke-WebRequest -Uri "http://localhost:8000/" -Method GET
if ($response.Headers['Access-Control-Allow-Origin']) {
    Write-Host "   ✓ CORS headers present" -ForegroundColor Green
} else {
    Write-Host "   ✗ CORS headers missing" -ForegroundColor Red
}

# Test 2: Security Headers
Write-Host "`n2. Testing security headers..." -ForegroundColor Yellow
$headers = $response.Headers
$securityHeaders = @('X-Content-Type-Options', 'X-Frame-Options', 'X-XSS-Protection')
$allPresent = $true
foreach ($header in $securityHeaders) {
    if ($headers[$header]) {
        Write-Host "   ✓ $header present" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $header missing" -ForegroundColor Red
        $allPresent = $false
    }
}

# Test 3: Rate Limiting
Write-Host "`n3. Testing rate limiting..." -ForegroundColor Yellow
$rateLimitHit = $false
for ($i=1; $i -le 35; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
            -Method POST `
            -ContentType "application/json" `
            -Body '{"image":"dGVzdA=="}' `
            -ErrorAction Stop
    } catch {
        if ($_.Exception.Response.StatusCode.value__ -eq 429) {
            Write-Host "   ✓ Rate limit enforced at request $i" -ForegroundColor Green
            $rateLimitHit = $true
            break
        }
    }
    Start-Sleep -Milliseconds 50
}
if (-not $rateLimitHit) {
    Write-Host "   ✗ Rate limit not enforced" -ForegroundColor Red
}

# Test 4: Authentication
Write-Host "`n4. Testing authentication..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/api/models/test/activate" `
        -Method POST `
        -ErrorAction Stop
    Write-Host "   ✗ Authentication not enforced" -ForegroundColor Red
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 403) {
        Write-Host "   ✓ Authentication enforced (403 Forbidden)" -ForegroundColor Green
    } else {
        Write-Host "   ? Unexpected status: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Yellow
    }
}

# Test 5: Input Validation
Write-Host "`n5. Testing input validation..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body '{"image":"invalid_base64!!!"}' `
        -ErrorAction Stop
    Write-Host "   ✗ Input validation not working" -ForegroundColor Red
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 422) {
        Write-Host "   ✓ Input validation working (422 Unprocessable)" -ForegroundColor Green
    } else {
        Write-Host "   ? Unexpected status: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Tests Complete ===" -ForegroundColor Green
```

**Run the test:**

```bash
powershell -ExecutionPolicy Bypass -File test_security_comprehensive.ps1
```

---

## Final Checklist

Before deploying to production, verify:

- [ ] `.env.production` exists and is NOT in git
- [ ] All hardcoded passwords removed from docker-compose.yaml
- [ ] API keys generated and configured
- [ ] CORS origins set to production domains only
- [ ] Rate limits configured appropriately
- [ ] Debug mode disabled in production
- [ ] Log level set to INFO or WARNING
- [ ] All security headers present
- [ ] Authentication working on protected endpoints
- [ ] Input validation working
- [ ] Logs don't contain sensitive data
- [ ] Docker containers rebuild successfully
- [ ] All tests pass

---

## Troubleshooting

### Issue: Rate limiting not working
**Solution:** Check that `slowapi` is installed and `Request` is imported correctly

### Issue: CORS errors in browser
**Solution:** Verify `ALLOWED_ORIGINS` includes your frontend domain

### Issue: Authentication always fails
**Solution:** Check that `API_KEYS` environment variable is set correctly (comma-separated)

### Issue: Logs still show sensitive data
**Solution:** Verify `SensitiveDataFilter` is added to all handlers in `logging_setup.py`

### Issue: Docker build fails
**Solution:** Check that all new Python files are in the correct directories and syntax is valid

---

## Next Steps

1. **Monitor Security Events:** Set up log aggregation and alerting
2. **Regular Updates:** Keep dependencies updated
3. **Security Audits:** Schedule regular security reviews
4. **Documentation:** Update API documentation with authentication requirements
5. **User Guide:** Create guide for API key management

---

**Implementation Complete!** 🎉

Your ASL Translator system now has comprehensive security controls in place. All critical vulnerabilities have been addressed, and the system is ready for secure production deployment.
