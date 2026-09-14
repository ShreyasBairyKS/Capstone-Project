"""
Verification Script for Step 3: Stream & Ingestion Manager (Multi-Camera Array).
Tests:
  1. CameraSourceConfig and SynchronizedFrameBundle schemas
  2. Offline paired multi-view bundling (Front + Rear angles)
  3. ThreadedCameraWorker non-blocking capture test
  4. Complete 3-Stage End-to-End Chain:
     Part 3 (StreamManager) -> Part 1 (InferenceEngine) -> Part 2 (QualityGate)
     Simulating continuous conveyor line inspections!
"""
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingestion import CameraSourceConfig, SynchronizedFrameBundle, ThreadedCameraWorker, StreamManager
from pipeline.inference import InferenceEngine
from pipeline.decision import QualityGate, InspectionVerdict, GlobalInspectionResult

TEST_IMG_1 = PROJECT_ROOT / "runs" / "segment" / "predict" / "IMG20260824154700.jpg"
TEST_IMG_2 = PROJECT_ROOT / "runs" / "segment" / "predict2" / "IMG20260824154411.jpg"


def run_tests():
    print("=" * 75)
    print("  STEP 3 VERIFICATION: STREAM & INGESTION MANAGER (MULTI-CAMERA ARRAY)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # TEST 1: Schema & Bundle Validation
    # -------------------------------------------------------------------------
    print("\n[1/4] Testing Camera Configuration & Bundle Assembly...")
    cfg_front = CameraSourceConfig(camera_id="camera_front", source_type="image_pair", source_path=str(TEST_IMG_1))
    cfg_rear = CameraSourceConfig(camera_id="camera_rear", source_type="image_pair", source_path=str(TEST_IMG_2))
    assert cfg_front.camera_id == "camera_front"
    assert cfg_rear.camera_id == "camera_rear"
    print("  ✓ CameraSourceConfig validated for front & rear cameras.")

    bundle = StreamManager.create_offline_bundle(
        camera_images={"camera_front": TEST_IMG_1, "camera_rear": TEST_IMG_2},
        bottle_id="BOTTLE_SIM_0001",
    )
    assert isinstance(bundle, SynchronizedFrameBundle)
    assert bundle.camera_count == 2
    assert "camera_front" in bundle.frames
    assert "camera_rear" in bundle.frames
    print(f"  ✓ Synchronized bundle created: {bundle.bottle_id}")
    print(f"  ✓ Camera count: {bundle.camera_count} views | Shapes: {[f.shape for f in bundle.frames.values()]}")

    # -------------------------------------------------------------------------
    # TEST 2: Threaded Camera Worker Simulation
    # -------------------------------------------------------------------------
    print("\n[2/4] Testing ThreadedCameraWorker Non-Blocking Ingestion...")
    worker_cfg = CameraSourceConfig(
        camera_id="test_worker_cam",
        source_type="video",
        source_path=str(TEST_IMG_1),  # OpenCV VideoCapture can read an image file as a static 1-frame stream
    )
    worker = ThreadedCameraWorker(worker_cfg)
    time.sleep(0.3)  # Let background thread capture initial frame

    is_new, frame, ts = worker.read_latest()
    assert frame is not None, "Worker must produce a frame"
    print(f"  ✓ Non-blocking frame read: Shape={frame.shape} | Timestamp={ts:.4f}")
    
    # Second read should be instantaneous (zero-wait)
    t0 = time.perf_counter()
    _, frame_cached, _ = worker.read_latest()
    read_latency_ms = (time.perf_counter() - t0) * 1000
    assert read_latency_ms < 5.0, f"Threaded read took too long: {read_latency_ms:.2f} ms"
    print(f"  ✓ Zero-buffer latency confirmed: Cached read took {read_latency_ms:.3f} ms (< 5ms target)")
    worker.stop()
    print("  ✓ Threaded worker stopped cleanly.")

    # -------------------------------------------------------------------------
    # TEST 3: Multi-Bottle Conveyor Simulation (3 Consecutive Bottles)
    # -------------------------------------------------------------------------
    print("\n[3/4] Initializing Full 3-Stage Pipeline (Ingestion -> Inference -> QualityGate)...")
    t_init = time.perf_counter()
    engine = InferenceEngine(warmup=True, conf_floor=0.35)
    gate = QualityGate()
    print(f"  ✓ Pipeline engines ready in {(time.perf_counter() - t_init)*1000:.1f} ms")

    # We will simulate 2 conveyor cycles:
    # Bottle 1: Multi-view pair (front defect, rear good) -> Expected REJECT
    # Bottle 2: Multi-view pair (both rear good image) -> Expected PASS
    conveyor_queue = [
        {"id": "BOTTLE_CONVEYOR_001", "views": {"camera_front": TEST_IMG_1, "camera_rear": TEST_IMG_2}},
        {"id": "BOTTLE_CONVEYOR_002", "views": {"camera_front": TEST_IMG_2, "camera_rear": TEST_IMG_2}},
    ]

    print("\n[4/4] Processing Conveyor Stream through Full 3-Stage Pipeline...")
    for idx, item in enumerate(conveyor_queue, 1):
        print(f"\n  --- Conveyor Cycle #{idx}: {item['id']} ---")
        
        # Stage 1: Ingestion
        t_cycle = time.perf_counter()
        bundle = StreamManager.create_offline_bundle(item["views"], bottle_id=item["id"])
        
        # Stage 2: Batched Inference on GPU/CPU
        multi_out = engine.predict_multi_camera(bundle.frames)
        
        # Stage 3: Quality Gate Decision
        decision = gate.evaluate_multi_camera(multi_out, bottle_id=bundle.bottle_id)
        cycle_ms = (time.perf_counter() - t_cycle) * 1000

        assert isinstance(decision, GlobalInspectionResult)
        print(f"    -> Bottle ID     : {decision.bottle_id}")
        print(f"    -> Global Verdict: {decision.global_verdict} (Compliant={decision.is_compliant})")
        print(f"    -> Reject Actuator Trigger: {decision.trigger_reject_actuator}")
        print(f"    -> Primary Defect: {decision.primary_defect}")
        print(f"    -> Flagged Views : {decision.flagged_cameras}")
        print(f"    -> Cycle Time    : {cycle_ms:.1f} ms")

        # Specific assertions
        if idx == 1:
            assert decision.global_verdict == InspectionVerdict.REJECT
            assert decision.trigger_reject_actuator is True
            assert "camera_front" in decision.flagged_cameras
        elif idx == 2:
            assert decision.global_verdict == InspectionVerdict.PASS
            assert decision.trigger_reject_actuator is False
            assert len(decision.flagged_cameras) == 0

    print("\n" + "=" * 75)
    print("  >>> STEP 3 VERIFICATION PASSED: Stream & Ingestion Manager is 100% Operational <<<")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
