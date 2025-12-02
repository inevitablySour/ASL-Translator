@echo off
echo Starting ASL Translator...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the application
echo.
echo Starting ASL Translator application...
echo Access the application at: http://127.0.0.1:8000
echo API Documentation at: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

pause
