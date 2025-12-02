# ASL Translator Run Script
Write-Host "Starting ASL Translator..." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Run the application
Write-Host "`nStarting application..." -ForegroundColor Green
Write-Host "Access at: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop`n" -ForegroundColor Yellow

python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
