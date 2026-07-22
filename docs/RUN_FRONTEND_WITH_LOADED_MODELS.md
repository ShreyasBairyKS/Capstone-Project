# Run Frontend With Loaded Models

Use these commands from the project root:

```powershell
cd "C:\Users\Ullas N\Desktop\Capstone-Project"
```

## Option 1: Start Everything With The Script

This is the recommended command because it prepares environment values, checks the required YOLO model files, starts the services, and can warm the inference pipeline.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1
```

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1 -Device cpu
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1 -SkipWarmup
powershell -ExecutionPolicy Bypass -File .\scripts\start_full_stack.ps1 -SkipBranchCheck
```

## Option 2: Start Manually

Start MongoDB and Redis:

```powershell
docker compose -f .\docker\docker-compose.yml up -d mongo redis
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, start the frontend:

```powershell
cd .\dashboard
npm run dev -- --host 127.0.0.1
```

Open the dashboard:

```text
http://127.0.0.1:3000
```

Backend API:

```text
http://127.0.0.1:8000
```

## Required Model Files

The live inspection pipeline expects these files:

```text
runs/detect/bottle_cap_det_v2/weights/best.pt
runs/detect/water_surface_v1/weights/best.pt
models/cap_classifier_best.pth
```

Check that they exist:

```powershell
Test-Path .\runs\detect\bottle_cap_det_v2\weights\best.pt
Test-Path .\runs\detect\water_surface_v1\weights\best.pt
Test-Path .\models\cap_classifier_best.pth
```

All three commands should return:

```text
True
```

## Confirm The Models Are Loaded

Check backend health:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health | ConvertTo-Json
```

Expected important values:

```json
{
  "status": "ok",
  "model_loaded": true,
  "standard_pipeline_loaded": false,
  "yolo_fill_ready": true
}
```

`standard_pipeline_loaded` can be `false` because the old ONNX pipeline files are not present. For live inspection, the important field is `yolo_fill_ready: true`, which means the bottle detector, fill-level model, and cap classifier files are available.

## Confirm Live Inspection Settings

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/inspections/live-settings `
  -Headers @{ "X-API-Key" = "dev-insecure-key" } |
  ConvertTo-Json
```

Expected:

```json
{
  "pipeline_mode": "yolo_fill_level",
  "use_cap_classifier": true
}
```

## Warm Up The YOLO Pipeline

The YOLO models are loaded lazily on the first inspection. To warm them manually, submit one image through the API using the dashboard manual inspection page, or run the full-stack script without `-SkipWarmup`.

## Stop Services

Stop Docker services:

```powershell
docker compose -f .\docker\docker-compose.yml down
```

Stop frontend/backend dev processes if they were started manually:

```powershell
Get-Process node,python -ErrorAction SilentlyContinue | Stop-Process
```

