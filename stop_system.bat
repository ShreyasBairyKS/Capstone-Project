@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

title Stop VisionQAI System

echo ==============================================================================
echo   STOPPING VISIONQAI INDUSTRIAL INSPECTION SYSTEM
echo ==============================================================================
echo.

:: 1. Stop Docker Compose containers if running
where docker >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Stopping Docker containers...
    docker compose down 2>nul
)

:: 2. Terminate any lingering native Python uvicorn processes on port 8000
echo [INFO] Releasing Port 8000 if occupied...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

:: 3. Terminate any lingering processes on Port 5173
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }" >nul 2>&1

echo.
echo [SUCCESS] VisionQAI containers and processes stopped cleanly.
echo.
powershell -NoProfile -Command "Start-Sleep -Seconds 1"
exit /b 0
