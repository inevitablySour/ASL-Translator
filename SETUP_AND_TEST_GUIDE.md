# ASL Translator - Detailed Setup and Testing Guide

**Last Updated:** January 21, 2026  
**Status:** Production-Ready  
**Purpose:** Complete instructions for running and testing the ASL Translator project

---

## Prerequisites - What You Need Before Starting

### System Requirements

- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 4GB (8GB recommended)
- **Disk Space:** Minimum 5GB free (10GB recommended)
- **Internet Connection:** Required for initial setup

### Software to Install (in this order)

#### 1. Docker and Docker Compose

**Why:** The entire project runs in Docker containers. This provides consistent environments across machines.

**Installation Steps:**

**Windows:**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Run the installer
3. Follow installation wizard (use default settings)
4. Restart your computer when prompted
5. Open PowerShell and verify installation:
   ```powershell
   docker --version
   docker-compose --version
   ```
6. Expected output should show version numbers (e.g., Docker version 24.x.x)

**macOS:**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Open the .dmg file and drag Docker to Applications folder
3. Open Docker from Applications
4. Complete the setup wizard
5. Open Terminal and verify:
   ```bash
   docker --version
   docker-compose --version
   ```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose -y
sudo usermod -aG docker $USER
# Log out and log back in to apply group changes
```

#### 2. Python 3.12 or Higher

**Why:** Project uses Python 3.12.6 for all scripts and testing

**Installation Steps:**

**Windows:**
1. Download from https://www.python.org/downloads/
2. Run installer
3. Check "Add Python to PATH"
4. Click "Install Now"
5. Verify in PowerShell:
   ```powershell
   python --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.12
```

**Linux:**
```bash
sudo apt-get install python3.12 python3.12-venv
```

#### 3. Git (Optional but Recommended)

**Why:** For version control and committing your changes

**Installation:** Download from https://git-scm.com/

---

## Step 1: Extract and Navigate to Project

### Windows (PowerShell)

```powershell
# Navigate to where you saved the project
cd "C:\Users\YourUsername\OneDrive - Zuyd Hogeschool\zuyd STUDIE saved\year2\ai_ops\proj2.0\ASL-Translator"

# Verify you're in the right directory
dir  # You should see docker-compose.yaml, requirements.txt, etc.
```

### macOS/Linux (Terminal)

```bash
cd /path/to/ASL-Translator
ls  # Should show docker-compose.yaml, requirements.txt, etc.
```

---

## Step 2: Configure Environment Variables

### Create Production Environment File

The project requires a `.env.production` file with sensitive configuration. This file is already created in the repository with default values.

**Option A: Using Existing .env.production (Quick Start)**

If `.env.production` already exists:
```bash
# Windows PowerShell
cat .env.production

# macOS/Linux Terminal
cat .env.production
```

You should see output like:
```
POSTGRES_USER=asl_user
POSTGRES_PASSWORD=secure_asl_password_12345
POSTGRES_DB=asl_translator_db
API_KEYS=test_key_12345,test_key_67890
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:8000,http://localhost:3000
RABBITMQ_PORT=5672
RATE_LIMIT_PREDICT=30/minute
RATE_LIMIT_FEEDBACK=60/minute
RATE_LIMIT_STATS=10/minute
RATE_LIMIT_ADMIN=10/minute
```

**Option B: Creating Custom .env.production (Production Deployment)**

If you need to customize values:

**Windows PowerShell:**
```powershell
# Create or edit the file
$env_content = @"
POSTGRES_USER=asl_user
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=asl_translator_db
API_KEYS=your_api_key_here
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:8000
RABBITMQ_PORT=5672
RATE_LIMIT_PREDICT=30/minute
RATE_LIMIT_FEEDBACK=60/minute
RATE_LIMIT_STATS=10/minute
RATE_LIMIT_ADMIN=10/minute
"@

$env_content | Out-File -Encoding UTF8 .env.production
```

**macOS/Linux:**
```bash
cat > .env.production << 'EOF'
POSTGRES_USER=asl_user
POSTGRES_PASSWORD=secure_password_123
POSTGRES_DB=asl_translator_db
API_KEYS=your_api_key_here
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:8000
RABBITMQ_PORT=5672
RATE_LIMIT_PREDICT=30/minute
RATE_LIMIT_FEEDBACK=60/minute
RATE_LIMIT_STATS=10/minute
RATE_LIMIT_ADMIN=10/minute
EOF
```

---

## Step 3: Start the Application

### Option A: Using Shell Scripts (Recommended - Automatic)

The project includes convenience scripts that handle Docker setup automatically.

**Windows PowerShell:**
```powershell
# Start all services (takes 2-3 minutes first time)
.\docker-up.ps1
```

**What happens automatically:**
1. Builds Docker images if not already built
2. Starts PostgreSQL database container
3. Starts API service container
4. Starts RabbitMQ message queue
5. Starts inference service
6. Syncs database from SQLite to PostgreSQL
7. Runs migrations if needed

**Expected output:** You'll see progress messages ending with services running.

**macOS/Linux:**
```bash
# Make scripts executable first time
chmod +x docker-up.sh
chmod +x docker-down.sh

# Start all services
./docker-up.sh
```

### Option B: Manual Docker Compose (if scripts don't work)

**Windows PowerShell:**
```powershell
docker-compose up -d --build
```

**macOS/Linux:**
```bash
docker-compose up -d --build
```

**Flags explained:**
- `-d` = Run in detached mode (runs in background)
- `--build` = Rebuild Docker images before starting

### Verify Services Started

**Windows PowerShell:**
```powershell
# Check running containers
docker ps

# Or more detailed view
docker-compose ps
```

**macOS/Linux:**
```bash
docker ps
docker-compose ps
```

**Expected output:** You should see containers running:
```
CONTAINER ID   IMAGE                    STATUS
12345abcde     asl_postgres            Up 2 minutes
67890fghij     asl_api                 Up 2 minutes
```

---

## Step 4: Verify API Service is Running

### Test API Health Endpoint

**Windows PowerShell:**
```powershell
# Test if API is responding
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

**macOS/Linux (with curl installed):**
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:30:00Z"
}
```

**If not responding:**
1. Wait 30-60 seconds for services to fully start
2. Check logs: `docker logs asl_api`
3. Verify Docker is running: `docker ps`

### Access API Documentation

Once API is running, you can view interactive documentation:

- **Swagger UI:** Open http://localhost:8000/docs in your web browser
- **ReDoc:** Open http://localhost:8000/redoc in your web browser

You should see API endpoints documented with ability to test them.

---

## Step 5: Run Security Tests

The project includes comprehensive security tests. Running these verifies all security implementations work correctly.

### Test 1: Install Test Dependencies

**Windows PowerShell:**
```powershell
# Install pytest if not already installed
pip install pytest==8.4.2 pydantic

# Verify installation
pytest --version
```

**macOS/Linux:**
```bash
pip3 install pytest==8.4.2 pydantic
pytest --version
```

### Test 2: Run Security Test Suite

**Windows PowerShell:**
```powershell
# Navigate to project root directory
cd C:\Users\YourUsername\...\ASL-Translator

# Run all security tests
pytest tests/test_security_implementation.py -v

# Or get more detailed output
pytest tests/test_security_implementation.py -v --tb=short
```

**macOS/Linux:**
```bash
pytest tests/test_security_implementation.py -v
```

**What you should see:**
```
test_security_implementation.py::TestConfigurationManagement::test_env_example_exists PASSED
test_security_implementation.py::TestConfigurationManagement::test_env_example_contains_required_vars PASSED
test_security_implementation.py::TestRateLimiting::test_slowapi_in_requirements PASSED
...
================================================ 44 passed, 1 skipped in 0.12s ================================================
```

**Expected Results:**
- Passed: 44
- Failed: 0
- Skipped: 1 (expected - external dependency limitation)
- Success Rate: 97.8%

### Test 3: Run Specific Test Categories

You can run specific security phase tests:

**Configuration Tests Only:**
```powershell
pytest tests/test_security_implementation.py::TestConfigurationManagement -v
```

**Authentication Tests Only:**
```powershell
pytest tests/test_security_implementation.py::TestAuthentication -v
```

**Input Validation Tests Only:**
```powershell
pytest tests/test_security_implementation.py::TestInputValidation -v
```

**All compliance tests:**
```powershell
pytest tests/test_security_implementation.py::TestRequirementR06Compliance -v
```

### Test 4: Generate Test Report

```powershell
# Run tests and save detailed report
pytest tests/test_security_implementation.py -v --tb=short > test_report.txt

# View the report
Get-Content test_report.txt
```

---

## Step 6: Test API Endpoints

Once the API is running, you can test actual functionality. There are two ways:

### Method A: Using Web UI (Easiest)

1. Open http://localhost:8000/docs in your browser
2. Click on an endpoint (e.g., `/predict`)
3. Click "Try it out"
4. Enter sample data
5. Click "Execute"
6. View response

### Method B: Using PowerShell Commands

**Test Prediction Endpoint (Core Functionality):**

```powershell
# Create test image data (small base64 encoded image)
$testImage = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# Create request body
$body = @{
    image = $testImage
    gesture = "A"
} | ConvertTo-Json

# Test without API key (should work)
$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"

Write-Host "Response: $response"
```

**Test with API Key (Admin Endpoint):**

```powershell
# Get API key from .env.production
$apiKey = "test_key_12345"  # Use your actual API key

# Test admin endpoint
$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/activate-model" `
  -Method Post `
  -Headers @{"X-API-Key" = $apiKey} `
  -Body (@{model_id = "test-model"} | ConvertTo-Json) `
  -ContentType "application/json"

Write-Host "Response: $response"
```

**Test Rate Limiting:**

```powershell
# Try to exceed rate limit with rapid requests
for ($i = 1; $i -le 35; $i++) {
    $response = Invoke-RestMethod `
      -Uri "http://localhost:8000/health" `
      -Method Get `
      -ErrorAction SilentlyContinue
    
    if ($response) {
        Write-Host "Request $i: Success"
    } else {
        Write-Host "Request $i: Rate limited (expected after ~30)"
    }
}
```

---

## Step 7: Check Database

### View PostgreSQL Database

**Connect to PostgreSQL from Command Line:**

```powershell
# Windows - install psql first, or use Docker
docker exec -it asl_postgres psql -U asl_user -d asl_translator_db

# Then in psql prompt
\dt  # Show all tables
SELECT * FROM users;  # Show users table
\q   # Quit
```

**macOS/Linux:**
```bash
docker exec -it asl_postgres psql -U asl_user -d asl_translator_db
```

### Check SQLite Backup

```powershell
# Windows
dir data/feedback.db

# macOS/Linux
ls -lh data/feedback.db
```

---

## Step 8: View Logs

### API Service Logs

```powershell
# Windows
docker logs asl_api

# Show last 100 lines
docker logs --tail 100 asl_api

# Follow live logs (Ctrl+C to stop)
docker logs -f asl_api
```

**macOS/Linux:**
```bash
docker logs asl_api
docker logs -f asl_api
```

### Database Logs

```powershell
docker logs asl_postgres
```

### All Logs

```powershell
docker-compose logs
```

---

## Step 9: Stop the Application

### Using Script (Recommended)

**Windows PowerShell:**
```powershell
.\docker-down.ps1
```

**What happens:**
1. Exports PostgreSQL data to SQLite backup
2. Syncs all database changes
3. Stops all containers gracefully
4. Preserves all data for next startup

**macOS/Linux:**
```bash
./docker-down.sh
```

### Manual Stop (if script fails)

```powershell
docker-compose down
```

---

## Step 10: Troubleshooting Common Issues

### Issue 1: Docker Daemon Not Running

**Error Message:** `Cannot connect to Docker daemon`

**Solution:**
- Windows: Open Docker Desktop application
- macOS: Open Docker from Applications
- Linux: Run `sudo systemctl start docker`

### Issue 2: Port Already in Use

**Error Message:** `port 8000 is already allocated`

**Solution:**
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Stop the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different ports in docker-compose.yaml
```

### Issue 3: Out of Disk Space

**Error Message:** `no space left on device`

**Solution:**
```powershell
# Clean up Docker resources
docker system prune -a --volumes

# This removes unused images, containers, and volumes
```

### Issue 4: Database Connection Failed

**Error Message:** `could not connect to server`

**Solution:**
```powershell
# Check if postgres container is running
docker ps | findstr postgres

# Check postgres logs
docker logs asl_postgres

# Restart postgres
docker-compose restart postgres
```

### Issue 5: Tests Failing with Import Errors

**Error Message:** `ModuleNotFoundError: No module named 'pydantic'`

**Solution:**
```powershell
# Install required packages
pip install -r requirements.txt
pip install pytest pydantic fastapi slowapi

# Try tests again
pytest tests/test_security_implementation.py -v
```

### Issue 6: API Responds with 403 Forbidden

**Error Message:** `403 Forbidden - Invalid API Key`

**Solution:**
1. Verify you're using correct API key from `.env.production`
2. Check header is exactly: `X-API-Key: your_key_here`
3. Ensure API key has no extra spaces or quotes

---

## Complete Testing Workflow (All Steps in Order)

Here's a quick reference for running through everything in sequence:

### Windows PowerShell

```powershell
# 1. Navigate to project
cd "C:\Path\To\ASL-Translator"

# 2. Start services
.\docker-up.ps1
Write-Host "Waiting 30 seconds for services to fully start..."
Start-Sleep -Seconds 30

# 3. Verify API health
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
Write-Host "API is responding!"

# 4. Run security tests
pytest tests/test_security_implementation.py -v

# 5. Check database
docker exec -it asl_postgres psql -U asl_user -d asl_translator_db -c "\dt"

# 6. View logs if needed
docker logs asl_api

# 7. Stop when done
.\docker-down.ps1
Write-Host "All services stopped and data backed up!"
```

### macOS/Linux Bash

```bash
# 1. Navigate to project
cd /path/to/ASL-Translator

# 2. Start services
./docker-up.sh
echo "Waiting 30 seconds for services to fully start..."
sleep 30

# 3. Verify API health
curl http://localhost:8000/health
echo -e "\nAPI is responding!"

# 4. Run security tests
pytest tests/test_security_implementation.py -v

# 5. Check database
docker exec -it asl_postgres psql -U asl_user -d asl_translator_db -c "\dt"

# 6. View logs if needed
docker logs asl_api

# 7. Stop when done
./docker-down.sh
echo "All services stopped and data backed up!"
```

---

## Success Indicators - How to Know It's Working

| Component | Success Indicator | How to Check |
|-----------|------------------|-------------|
| Docker | Containers running | `docker ps` shows 4-5 containers |
| API Service | Responds to requests | `curl http://localhost:8000/health` returns 200 |
| Database | Connected and ready | `docker logs asl_postgres` shows "ready to accept connections" |
| Security Tests | 44/45 passing | `pytest` output shows "44 passed, 1 skipped" |
| Rate Limiting | Working | After ~30 requests, get HTTP 429 response |
| Logging | Active | `docker logs asl_api` shows security events |
| Data Persistence | Functional | Data survives `docker-down` and `docker-up` |

---

## Performance Expectations

| Task | Expected Duration | Notes |
|------|------------------|-------|
| First startup | 2-3 minutes | Builds Docker images and containers |
| Subsequent startup | 30-60 seconds | Images already built |
| Security tests | 0.12 seconds | Very fast, all in-process tests |
| API health check | < 100ms | Should be nearly instant |
| Single prediction | 100-500ms | Depends on model size and system |
| Shutdown/sync | 30-60 seconds | Exports data to SQLite |

---

## Next Steps After Successful Testing

1. **Review Security Reports:** Read `security/FINAL_SECURITY_REPORT.md` for implementation details
2. **Deploy to Production:** Follow deployment guidelines in documentation
3. **Monitor Logs:** Set up log monitoring for production alerts
4. **Regular Testing:** Run test suite periodically to catch regressions
5. **Update Dependencies:** Check for security updates monthly

---

## Getting Help

If you encounter issues not covered in troubleshooting:

1. Check Docker logs: `docker logs [service_name]`
2. Review QUICKSTART.md for alternative instructions
3. Check .env.production for correct configuration
4. Verify all prerequisites installed correctly
5. Ensure sufficient disk space and RAM available

---

**Last Updated:** January 21, 2026  
**Status:** Production-Ready  
**Test Success Rate:** 97.8% (44/45 tests passing)
