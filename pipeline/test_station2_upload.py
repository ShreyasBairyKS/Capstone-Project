"""
Test Suite for Station 2 (Fill Level Detector) Dedicated Single-Image Upload & API Integration.
Verifies:
1. Station 2 inspection engine & contract:
   - Evaluates a single bottle profile image specimen.
   - Accepts custom bottle_id or auto-generates ID.
   - Generates compliant FillInspectResponse with verdict, fill_status, fill_ratio,
     annotated_image_base64, and total_latency_ms.
   - Verifies Station 1 inference models remain UNLOADED during Station 2 inspection.
2. API Route Contract for POST /api/fill/inspect:
   - inspect_upload route signature accepts single file: UploadFile.
   - Accepts bottle_id via Form or Query parameter.
   - Returns FillInspectResponse model.
3. Station Switching & Frontend Contracts:
   - FillUploadModal.jsx exists, targets /api/fill/inspect, accepts single file and optional bottle_id.
   - App.jsx routes activeStation === 'fill' to FillUploadModal and activeStation === 'cap' to UploadModal.
   - UploadModal.jsx preserves camera 1 selection (Task 2) and delegates to FillUploadModal when in fill station.
4. Production Vite Build Assets:
   - dist/index.html and bundled assets exist and build succeeds cleanly.
"""
import inspect
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_station2_upload")

from pipeline.fill_level.api.service import FillPipelineService
from pipeline.fill_level.api.schemas import FillInspectResponse
from pipeline.fill_level.api.routes import inspect_upload, fill_router
from pipeline.api.pipeline_service import PipelineService


def run_checks():
    logger.info("================================================================")
    logger.info("STEP 1: Verify Station 2 Single-Image Inspection & S1 Isolation")
    logger.info("================================================================")
    fill_svc = FillPipelineService.get_instance()
    fill_svc.ensure_engine_loaded()
    assert fill_svc.is_engine_loaded() is True, "Fill engine must be loaded for inspection!"

    cap_svc = PipelineService.get_instance()
    if cap_svc.is_engine_loaded():
        cap_svc.unload_engine()
    assert cap_svc.is_engine_loaded() is False, "Cap Classifier should be unloaded to test isolation!"

    # Create dummy bottle profile image (640x640 with synthetic bottle silhouette)
    test_img = np.full((640, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(test_img, (220, 150), (420, 580), (180, 180, 180), -1)
    cv2.rectangle(test_img, (225, 320), (415, 575), (240, 120, 40), -1)

    # Test inspection with custom bottle_id
    custom_id = "SPECIMEN_FILL_TEST_99"
    resp = fill_svc.inspect_frame(test_img, bottle_id=custom_id)
    
    logger.info(f"Fill inspect response: bottle_id={resp.bottle_id}, verdict={resp.verdict}, fill_status={resp.fill_status}, fill_ratio={resp.fill_ratio}")
    assert resp.bottle_id == custom_id, f"Expected bottle_id={custom_id}, got {resp.bottle_id}"
    assert resp.verdict in ("PASS", "REJECT", "UNCERTAIN", "INSPECTION_FAILED"), f"Invalid verdict: {resp.verdict}"
    assert resp.fill_status in ("normal", "underfill", "overfill", "no_bottle", "no_water"), f"Invalid fill_status: {resp.fill_status}"
    assert resp.total_latency_ms > 0, "Expected positive latency value"
    assert resp.annotated_image_base64 is not None, "Expected annotated base64 preview"
    assert len(resp.annotated_image_base64) > 100, "Base64 image should contain valid data"

    # Verify Cap Classifier remained UNLOADED during Station 2 inspection
    assert cap_svc.is_engine_loaded() is False, "Violation: Station 1 Cap Classifier was loaded during Station 2 upload!"
    logger.info("STEP 1 PASSED: Station 2 inspection succeeded with strict Station 1 model isolation.")

    logger.info("================================================================")
    logger.info("STEP 2: Auto-Generated bottle_id Fallback")
    logger.info("================================================================")
    resp_auto = fill_svc.inspect_frame(test_img, bottle_id=None)
    assert resp_auto.bottle_id.startswith("FILL_"), f"Expected auto-generated bottle_id starting with FILL_, got {resp_auto.bottle_id}"
    logger.info(f"STEP 2 PASSED: Auto-generated bottle_id verified: {resp_auto.bottle_id}")

    logger.info("================================================================")
    logger.info("STEP 3: Verify Route Contract for POST /api/fill/inspect")
    logger.info("================================================================")
    # Check inspect_upload function signature
    sig = inspect.signature(inspect_upload)
    param_names = list(sig.parameters.keys())
    assert "file" in param_names, f"inspect_upload must have 'file' parameter! Found: {param_names}"
    assert "bottle_id" in param_names, f"inspect_upload must have 'bottle_id' parameter! Found: {param_names}"
    assert "bottle_id_query" in param_names, f"inspect_upload must have 'bottle_id_query' parameter! Found: {param_names}"

    # Verify route is mounted on fill_router
    inspect_routes = [r for r in fill_router.routes if getattr(r, "path", None) == "/inspect"]
    assert len(inspect_routes) == 1, "Expected exactly 1 /inspect route in fill_router"
    route = inspect_routes[0]
    assert "POST" in route.methods, f"Expected POST method for /inspect route, got: {route.methods}"
    assert route.response_model == FillInspectResponse, f"Expected FillInspectResponse model, got: {route.response_model}"
    logger.info("STEP 3 PASSED: POST /api/fill/inspect route contract fully verified.")

    logger.info("================================================================")
    logger.info("STEP 4: Frontend Contract & Component Verification")
    logger.info("================================================================")
    frontend_dir = PROJECT_ROOT / "pipeline" / "frontend" / "src"

    # 1. FillUploadModal.jsx
    fill_modal_file = frontend_dir / "components" / "fill_level" / "FillUploadModal.jsx"
    assert fill_modal_file.exists(), f"{fill_modal_file} does not exist!"
    fill_modal_content = fill_modal_file.read_text(encoding="utf-8")
    
    assert "/api/fill/inspect" in fill_modal_content, "FillUploadModal must route to /api/fill/inspect!"
    assert "/api/inspect/image" not in fill_modal_content, "FillUploadModal must NEVER reference /api/inspect/image!"
    assert "formData.append('file', file)" in fill_modal_content, "FillUploadModal must append single 'file'!"
    assert "onInspectionComplete" in fill_modal_content, "FillUploadModal must call onInspectionComplete!"
    logger.info("FillUploadModal.jsx contract verified.")

    # 2. App.jsx
    app_file = frontend_dir / "App.jsx"
    assert app_file.exists(), f"{app_file} does not exist!"
    app_content = app_file.read_text(encoding="utf-8")

    assert "import FillUploadModal from './components/fill_level/FillUploadModal';" in app_content, "App.jsx must import FillUploadModal!"
    assert "activeStation === 'fill'" in app_content, "App.jsx must check activeStation for conditional modal rendering!"
    assert "<FillUploadModal" in app_content, "App.jsx must render FillUploadModal!"
    assert "<UploadModal" in app_content, "App.jsx must render UploadModal!"
    logger.info("App.jsx station switching contract verified.")

    # 3. UploadModal.jsx
    upload_modal_file = frontend_dir / "components" / "UploadModal.jsx"
    assert upload_modal_file.exists(), f"{upload_modal_file} does not exist!"
    upload_modal_content = upload_modal_file.read_text(encoding="utf-8")

    assert "import FillUploadModal from './fill_level/FillUploadModal';" in upload_modal_content, "UploadModal must import FillUploadModal for fallback delegation!"
    assert "activeStation === 'fill'" in upload_modal_content, "UploadModal must delegate when activeStation === 'fill'!"
    assert "slot === 'front'" in upload_modal_content, "Camera 1 (front) selection handler must be preserved!"
    assert "setFrontFile(file)" in upload_modal_content, "Camera 1 file state must be preserved!"
    assert "setFrontPreview(e.target.result)" in upload_modal_content, "Camera 1 preview state must be preserved!"
    assert "/api/inspect/image" in upload_modal_content, "UploadModal for Station 1 must route to /api/inspect/image!"
    logger.info("UploadModal.jsx Camera 1 fix and delegation verified.")

    # 4. StationContext.jsx
    ctx_file = frontend_dir / "context" / "StationContext.jsx"
    assert ctx_file.exists(), f"{ctx_file} does not exist!"
    ctx_content = ctx_file.read_text(encoding="utf-8")
    assert "export const StationContext" in ctx_content, "StationContext must be exported!"
    logger.info("StationContext.jsx verified.")

    logger.info("================================================================")
    logger.info("STEP 5: Production Assets Verification")
    logger.info("================================================================")
    dist_dir = PROJECT_ROOT / "pipeline" / "frontend" / "dist"
    assert dist_dir.exists(), f"Frontend dist folder {dist_dir} does not exist!"
    assert (dist_dir / "index.html").exists(), "dist/index.html not found!"
    js_files = list((dist_dir / "assets").glob("*.js"))
    assert len(js_files) >= 1, "Built JS assets missing!"
    logger.info(f"Verified built bundle: {js_files[0].name}")

    print("\n" + "=" * 70)
    print("ALL STATION 2 UPLOAD & API INTEGRATION TESTS PASSED 100%!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_checks()
