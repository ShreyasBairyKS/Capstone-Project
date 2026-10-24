@echo off
setlocal enabledelayedexpansion

title VisionQAI Industrial Inspection Station - Standalone Native Mode
cd /d "%~dp0"

echo ==============================================================================
echo   VISIONQAI DEFECT DETECTION PIPELINE (STANDALONE NATIVE RUNTIME)
echo ==============================================================================
echo.

:: 1. Check Python virtual environment
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at %~dp0.venv
    echo Please create the virtual environment and install requirements first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\pip install -r pipeline\requirements.txt
    pause
    exit /b 1
)

:: 2. Check if frontend dist exists
if not exist "%~dp0pipeline\frontend\dist\index.html" (
    echo [INFO] Compiled frontend assets not found. Building now...
    cd "%~dp0pipeline\frontend"
    call npm run build
    cd /d "%~dp0"
)

:: 3. Launch FastAPI backend in dedicated process window
echo [INFO] Starting VisionQAI unified server on port 8000...
start "VisionQAI Server" /D "%~dp0" "%~dp0.venv\Scripts\python.exe" -m uvicorn pipeline.api.app:app --host 127.0.0.1 --port 8000

:: 4. Poll health endpoint
echo [INFO] Awaiting server readiness...
set /a ATTEMPTS=0
:POLL_LOOP
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
set /a ATTEMPTS+=1

curl -s -f http://127.0.0.1:8000/api/health >nul 2>&1
if !ERRORLEVEL! equ 0 (
    goto HEALTHY
)

if %ATTEMPTS% geq 20 (
    echo.
    echo [WARNING] Server startup taking longer than expected.
    goto OPEN_BROWSER
)

echo|set /p="."
goto POLL_LOOP

:HEALTHY
echo.
echo [SUCCESS] VisionQAI Server is ONLINE!

:OPEN_BROWSER
echo.
echo ==============================================================================
echo   OPERATIONAL ENDPOINTS:
echo   - Operator Dashboard  : http://localhost:8000/visionqai/
echo   - Direct OpenAPI Docs : http://localhost:8000/visionqai/docs
echo   - Health Diagnostics  : http://localhost:8000/api/health
echo ==============================================================================
echo.
echo [INFO] Launching default web browser to Operator Dashboard...
start http://localhost:8000/visionqai/

echo.
echo Quick commands:
echo   - Run stress test suite  : .\.venv\Scripts\python.exe pipeline\stress_test.py
echo   - Stop server process    : .\stop_system.bat
echo.
exit /b 0
