# Wrapper script for docker-compose up that syncs SQLite to PostgreSQL if needed

Write-Host "Starting Docker containers..." -ForegroundColor Green
docker-compose up -d postgres

Write-Host ""
Write-Host "Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "Checking if database sync is needed..." -ForegroundColor Yellow

# Detect Python command - use active venv or find one
$PYTHON_CMD = "python"

if ($env:VIRTUAL_ENV) {
    # Already in a virtual environment
    $PYTHON_CMD = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    # Local .venv directory
    $PYTHON_CMD = ".venv\Scripts\python.exe"
} else {
    # Try to find python with required packages
    try {
        python -c "import sqlalchemy, psycopg2" 2>$null
        $PYTHON_CMD = "python"
    } catch {
        Write-Host "⚠️  Cannot find Python with sqlalchemy and psycopg2 installed" -ForegroundColor Yellow
        Write-Host "   Install with: pip install sqlalchemy psycopg2-binary" -ForegroundColor Yellow
        $PYTHON_CMD = "python"
    }
}

# Check if sync is needed
Write-Host "Running check_and_sync.py..." -ForegroundColor Cyan
& $PYTHON_CMD check_and_sync.py
if ($LASTEXITCODE -eq 0) {
    # 0 = sync needed
    Write-Host ""
    Write-Host "Syncing SQLite to PostgreSQL..." -ForegroundColor Yellow
    & $PYTHON_CMD migrate_to_postgres.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Sync completed successfully" -ForegroundColor Green
    } else {
        Write-Host "Sync failed, but containers are running" -ForegroundColor Red
    }
} else {
    # non-zero = no sync needed
    Write-Host ""
    Write-Host "Database is already up-to-date" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting remaining services..." -ForegroundColor Green
docker-compose up -d @args

Write-Host ""
Write-Host "All services are running!" -ForegroundColor Green
Write-Host ""
Write-Host "Services:" -ForegroundColor Cyan
docker-compose ps
