# Wrapper script for docker-compose down that exports PostgreSQL to SQLite first

Write-Host "Exporting PostgreSQL data to SQLite before shutdown..." -ForegroundColor Yellow

# Check if PostgreSQL container is running
$postgresRunning = docker ps | Select-String "asl_postgres"

if ($postgresRunning) {
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
            try {
                python3 -c "import sqlalchemy, psycopg2" 2>$null
                $PYTHON_CMD = "python3"
            } catch {
                Write-Host "Cannot find Python with sqlalchemy and psycopg2 installed" -ForegroundColor Yellow
                Write-Host "   Install with: pip install sqlalchemy psycopg2-binary" -ForegroundColor Yellow
                $PYTHON_CMD = "python3"
            }
        }
    }
    
    Write-Host "Running export_to_sqlite.py..." -ForegroundColor Cyan
    & $PYTHON_CMD export_to_sqlite.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Data exported successfully" -ForegroundColor Green
    } else {
        Write-Host "Export failed, but continuing with shutdown..." -ForegroundColor Red
        Write-Host "   Make sure dependencies are installed: pip install sqlalchemy psycopg2-binary" -ForegroundColor Yellow
    }
} else {
    Write-Host "PostgreSQL container not running, skipping export" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Stopping Docker containers..." -ForegroundColor Green
docker-compose down @args

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
