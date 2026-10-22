@echo off
setlocal enabledelayedexpansion

title VisionQAI Industrial Inspection Station - Docker Mode

echo ==============================================================================
echo   VISIONQAI AUTOMATED DEFECT DETECTION PIPELINE (CONTAINERIZED RUNTIME)
echo ==============================================================================
echo.

:: 1. Check Docker CLI presence
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker executable not found in PATH.
    echo Please install Docker Desktop with WSL2 backend before running this script.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: 2. Check Docker Daemon status
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker daemon is not currently running.
    echo Please open Docker Desktop, wait for the engine to start, and try again.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: 3. Read version tags from .env (fallback to v1)
set "BACKEND_VER=v1"
set "FRONTEND_VER=v1"
set "ENGINE_VER=v1"

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "KEY=%%A"
        set "VAL=%%B"
        if "!KEY!"=="VISIONQAI_BACKEND_VERSION" set "BACKEND_VER=!VAL!"
        if "!KEY!"=="VISIONQAI_FRONTEND_VERSION" set "FRONTEND_VER=!VAL!"
        if "!KEY!"=="VISIONQAI_ENGINE_VERSION" set "ENGINE_VER=!VAL!"
    )
)

echo [CONFIG] Target Engine Version  : %ENGINE_VER%
echo [CONFIG] Target Backend Image   : visionqai_backend:%BACKEND_VER%
echo [CONFIG] Target Frontend Image  : visionqai_frontend:%FRONTEND_VER%
echo.

:: 4. Check for rebuild flag or missing images
set "DO_BUILD=0"
if "%1"=="--build" set "DO_BUILD=1"
if "%1"=="-b" set "DO_BUILD=1"

for /f %%i in ('docker images -q visionqai_backend:%BACKEND_VER% 2^>nul') do set "BK_IMG=%%i"
for /f %%i in ('docker images -q visionqai_frontend:%FRONTEND_VER% 2^>nul') do set "FT_IMG=%%i"

if "%BK_IMG%"=="" set "DO_BUILD=1"
if "%FT_IMG%"=="" set "DO_BUILD=1"

if "%DO_BUILD%"=="1" (
    echo [INFO] Generating / Updating container images...
    docker compose build
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Container image build failed.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Existing versioned images found. Skipping build (use --build to force).
)

:: 5. Launch containers
echo [INFO] Starting VisionQAI inspection containers...
docker compose up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start Docker Compose services.
    pause
    exit /b 1
)

:: 6. Healthcheck Polling
echo.
echo [INFO] Awaiting backend health verification and TensorRT FP16 engine readiness...
set /a ATTEMPTS=0
:POLL_LOOP
powershell -NoProfile -Command "Start-Sleep -Seconds 3"
set /a ATTEMPTS+=1

curl -s -f http://127.0.0.1:8000/api/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    goto HEALTHY
)

if %ATTEMPTS% geq 30 (
    echo.
    echo [WARNING] Healthcheck timed out after 90 seconds.
    echo Containers may still be compiling the TensorRT engine on GPU.
    echo Check logs with: docker compose logs -f backend
    goto OPEN_BROWSER
)

echo|set /p="."
goto POLL_LOOP

:HEALTHY
echo.
echo [SUCCESS] Backend service and inference engine are ONLINE!

:OPEN_BROWSER
echo.
echo ==============================================================================
echo   OPERATIONAL ENDPOINTS:
echo   - Operator Dashboard  : http://localhost:80/visionqai/
echo   - Direct Backend API  : http://localhost:8000/visionqai/docs
echo   - Health Diagnostics  : http://localhost:8000/api/health
echo ==============================================================================
echo.
echo [INFO] Launching default web browser to Operator Dashboard...
start http://localhost:80/visionqai/

echo.
echo Quick commands:
echo   - View live backend logs : docker compose logs -f backend
echo   - Run stress test suite  : docker exec -it visionqai_backend python pipeline/stress_test.py
echo   - Stop all services      : .\stop_system.bat
echo.
exit /b 0
