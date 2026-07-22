# Manual Setup On Another Laptop

This guide is for running the frontend UI on a fresh laptop using the same GitHub repo, without using `scripts/start_full_stack.ps1`.

The setup runs:

- MongoDB and Redis using Docker
- FastAPI backend locally with Python
- React/Vite frontend locally with npm
- YOLO fill-level pipeline with bottle detection, cap classifier, and fill-level model

## 1. Install Required Software

Install these first:

- Git
- Docker Desktop
- Python 3.11.x
- Node.js LTS

After installing Docker Desktop, open it once and wait until Docker says it is running.

Check the tools:

```powershell
git --version
docker --version
docker compose version
python --version
node --version
npm --version
```

Python should be `3.11.x`.

## 2. Clone The Repository

Open PowerShell and go to the folder where the project should live.

```powershell
cd "C:\Users\<YOUR_USERNAME>\Desktop"
git clone <YOUR_GITHUB_REPO_URL> Capstone-Project
cd .\Capstone-Project
```

If the working branch is required, switch to it:

```powershell
git checkout till-this-sem-working-inference
git pull
```

If the repo uses another branch name, replace `till-this-sem-working-inference` with that branch.

## 3. Confirm Model Files Are Present

The frontend will show the model as loaded only when these files exist:

```text
runs/detect/bottle_cap_det_v2/weights/best.pt
runs/detect/water_surface_v1/weights/best.pt
models/cap_classifier_best.pth
```

Run:

```powershell
Test-Path .\runs\detect\bottle_cap_det_v2\weights\best.pt
Test-Path .\runs\detect\water_surface_v1\weights\best.pt
Test-Path .\models\cap_classifier_best.pth
```

Expected output:

```text
True
True
True
```

If any command returns `False`, the model files are missing from that laptop. Copy the missing files into the exact paths above, then run the checks again.

## 4. Create Backend `.env`

Create `.env` from the example:

```powershell
Copy-Item .\.env.example .\.env
```

Open `.env` in Notepad:

```powershell
notepad .\.env
```

Make sure these values are present:

```env
TIER=edge
DEVICE_ID=edge_node_01

YOLO_FILL_DETECTOR_WEIGHTS=runs/detect/bottle_cap_det_v2/weights/best.pt
YOLO_FILL_WATER_WEIGHTS=runs/detect/water_surface_v1/weights/best.pt
YOLO_FILL_CAP_CLASSIFIER_WEIGHTS=models/cap_classifier_best.pth
YOLO_FILL_DEVICE=cpu

DATABASE_URL=sqlite:///./visionfood_dev.db
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=visionfood

REDIS_URL=redis://localhost:6379
REDIS_LIVE_STREAM=inspections:live
REDIS_STREAM_MAX_LEN=1000

API_KEY=dev-insecure-key

LOG_LEVEL=INFO
LOG_FORMAT=json
AUDIT_LOG_PATH=logs/audit.jsonl

REMEDY_ENABLED=true
REMEDY_MAX_ATTEMPTS=2
SKU_PROFILES_DIR=configs/sku_profiles
```

Save and close Notepad.

## 5. Create Frontend `.env`

Create the dashboard env file:

```powershell
Copy-Item .\dashboard\.env.example .\dashboard\.env
```

Open it:

```powershell
notepad .\dashboard\.env
```

Make sure it contains:

```env
VITE_API_BASE=/api
VITE_API_KEY=dev-insecure-key
VITE_WS_URL=ws://localhost:8000/ws/live
```

Important: `VITE_API_KEY` must match `API_KEY` in the backend `.env`.

Save and close Notepad.

## 6. Create Python Virtual Environment

From the project root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install backend dependencies:

```powershell
pip install -r requirements.txt
```

This may take a while because PyTorch, Ultralytics, OpenCV, and backend packages are installed.

## 7. Install Frontend Dependencies

Open another PowerShell window or stay in the same one.

From the project root:

```powershell
cd .\dashboard
npm install
cd ..
```

## 8. Start MongoDB And Redis

Make sure Docker Desktop is running.

From the project root:

```powershell
docker compose -f .\docker\docker-compose.yml up -d mongo redis
```

Check containers:

```powershell
docker ps
```

You should see containers for:

```text
mongo:7
redis:7-alpine
```

## 9. Start The Backend

Open a PowerShell window in the project root.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave this PowerShell window open.

The backend should show:

```text
Uvicorn running on http://127.0.0.1:8000
```

It is okay if the old standard ONNX pipeline logs a warning. The live inspection UI uses the YOLO fill-level pipeline.

## 10. Verify Backend And Model Status

Open a new PowerShell window.

Run:

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

Meaning:

- `status: ok` means backend is running.
- `model_loaded: true` means the frontend should show AI Model as loaded.
- `yolo_fill_ready: true` means the YOLO detector, fill-level model, and cap classifier files were found.
- `standard_pipeline_loaded: false` is acceptable if the older ONNX files are not being used.

Now check live inspection settings:

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

## 11. Start The Frontend

Open another PowerShell window.

Go to the dashboard folder:

```powershell
cd "C:\Users\<YOUR_USERNAME>\Desktop\Capstone-Project\dashboard"
```

Start Vite:

```powershell
npm run dev -- --host 127.0.0.1
```

Leave this PowerShell window open.

Open the frontend in the browser:

```text
http://127.0.0.1:3000
```

The sidebar should show:

```text
API Connected
AI Model Loaded
```

The live stream may show offline until a websocket/live inspection event is available, but API and AI Model should be connected/loaded.

## 12. Run A Manual Inspection Test

In the frontend:

1. Open `http://127.0.0.1:3000`.
2. Go to the manual inspection or inspect page.
3. Upload a bottle image.
4. Select the YOLO/fill-level pipeline if the UI gives the option.
5. Keep cap classifier enabled if you want cap quality detection.
6. Run inspection.

Expected result:

- Annotated image appears in the frontend.
- Detection confidence/details appear in the inference dashboard panel.
- The backend terminal may take longer on the first request because YOLO models are loaded lazily.

## 13. Optional: Test Inference Directly Through The API

Use this only if you want to confirm the backend without the frontend.

Replace the image path with an image that exists on the laptop:

```powershell
$imagePath = "C:\path\to\test-image.jpg"
$imageB64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($imagePath))
$payload = @{
  image_b64 = $imageB64
  sku = "default"
  product_id = "manual-test"
  attempt_count = 0
  pipeline_mode = "yolo_fill_level"
  use_cap_classifier = $true
} | ConvertTo-Json -Compress -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/inspections `
  -Method Post `
  -Headers @{ "X-API-Key" = "dev-insecure-key" } `
  -ContentType "application/json" `
  -Body $payload |
  ConvertTo-Json -Depth 10
```

If this succeeds, the model pipeline is connected.

## 14. Stop Everything

Stop frontend:

```text
Press Ctrl+C in the frontend PowerShell window.
```

Stop backend:

```text
Press Ctrl+C in the backend PowerShell window.
```

Stop MongoDB and Redis:

```powershell
docker compose -f .\docker\docker-compose.yml down
```

## Troubleshooting

### Frontend Shows API Error

Check backend:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health | ConvertTo-Json
```

If it cannot connect, the backend is not running. Start it again:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Frontend Shows AI Model Not Loaded

Check the model files:

```powershell
Test-Path .\runs\detect\bottle_cap_det_v2\weights\best.pt
Test-Path .\runs\detect\water_surface_v1\weights\best.pt
Test-Path .\models\cap_classifier_best.pth
```

Then check health:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health | ConvertTo-Json
```

The important value is:

```json
"yolo_fill_ready": true
```

If it is `false`, one or more model files are missing or in the wrong folder.

### API Gives 401 Unauthorized

Make sure both env files use the same key:

Backend `.env`:

```env
API_KEY=dev-insecure-key
```

Frontend `dashboard/.env`:

```env
VITE_API_KEY=dev-insecure-key
```

After changing frontend env values, stop and restart `npm run dev`.

### Backend Cannot Connect To MongoDB Or Redis

Start Docker services:

```powershell
docker compose -f .\docker\docker-compose.yml up -d mongo redis
```

Check:

```powershell
docker ps
```

### Port Already In Use

For backend port `8000`:

```powershell
Get-Process python -ErrorAction SilentlyContinue
```

Stop old Python backend processes if needed:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process
```

For frontend port `3000`:

```powershell
Get-Process node -ErrorAction SilentlyContinue | Stop-Process
```

Then start backend/frontend again.

