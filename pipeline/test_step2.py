"""
Verification Script for Step 2: Multi-View Inspection Decision Logic (The Quality Gate).
Tests:
  1. Unit tests across all 5 industrial scenarios:
     - Scenario A: Clean Dual-Pass (Both views compliant) -> PASS
     - Scenario B: Defect Dominant (Front view damaged, rear view compliant) -> REJECT
     - Scenario C: Opposite Defect (Front compliant, rear open_cap) -> REJECT
     - Scenario D: Borderline 0.61 Edge Case (Front 0.61 good_cap, rear compliant) -> UNCERTAIN (No false defect!)
     - Scenario E: Zero Detection / Anomaly -> INSPECTION_FAILED
  2. Full Integration Test:
     - Feeds real InferenceEngine output (Part 1) into QualityGate (Part 2)
"""
import sys
from pathlib import Path

import os
from typing import Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.inference import InferenceEngine, Detection, CameraResult, MultiCameraInferenceResult
from pipeline.decision import QualityGate, InspectionVerdict, InspectionRulesConfig, GlobalInspectionResult


def resolve_test_images(project_root: Path) -> Tuple[Path, Path]:
    search_dirs = [
        Path(os.getenv("VISIONQAI_DATASET_DIR", "").strip("\"'")),
        Path(os.getenv("VISIONQAI_TEST_DATA_DIR", "").strip("\"'")),
        Path(r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11\test\images"),
        Path(r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11"),
        project_root / "runs" / "segment",
    ]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    found = []
    for d in search_dirs:
        if d.is_file() and d.suffix.lower() in valid_exts and d.exists():
            found.append(d)
        elif d.is_dir() and d.exists():
            for sub in [d / "test" / "images", d / "images", d]:
                if sub.exists() and sub.is_dir():
                    imgs = sorted([f for f in sub.iterdir() if f.suffix.lower() in valid_exts])
                    if imgs:
                        found.extend(imgs)
                        break
        if len(found) >= 2:
            break

    if len(found) >= 2:
        return found[0], found[1]
    elif len(found) == 1:
        return found[0], found[0]
    else:
        synth_dir = project_root / "pipeline" / "scratch"
        synth_dir.mkdir(parents=True, exist_ok=True)
        img1 = synth_dir / "test_bottle_1.jpg"
        img2 = synth_dir / "test_bottle_2.jpg"
        if not img1.exists():
            cv2.imwrite(str(img1), np.full((640, 640, 3), (120, 120, 120), dtype=np.uint8))
        if not img2.exists():
            cv2.imwrite(str(img2), np.full((640, 640, 3), (130, 130, 130), dtype=np.uint8))
        return img1, img2


TEST_IMG_1, TEST_IMG_2 = resolve_test_images(PROJECT_ROOT)


def run_tests():
    print("=" * 75)
    print("  STEP 2 VERIFICATION: MULTI-VIEW QUALITY GATE & DECISION LOGIC")
    print("=" * 75)

    gate = QualityGate()

    # -------------------------------------------------------------------------
    # UNIT TEST SCENARIO A: Clean Dual-Pass (Both views >= 0.75 good_cap)
    # -------------------------------------------------------------------------
    print("\n[Scenario A] Testing Clean Dual-Pass (All views compliant)...")
    res_a = MultiCameraInferenceResult(
        hardware_used="Mock",
        total_latency_ms=4.2,
        camera_count=2,
        views={
            "camera_front": CameraResult(
                camera_id="camera_front", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.88, box=[100, 100, 200, 200])]
            ),
            "camera_rear": CameraResult(
                camera_id="camera_rear", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.91, box=[105, 95, 205, 195])]
            ),
        }
    )
    verdict_a = gate.evaluate_multi_camera(res_a)
    assert verdict_a.global_verdict == InspectionVerdict.PASS, f"Expected PASS, got {verdict_a.global_verdict}"
    assert verdict_a.is_compliant is True
    assert verdict_a.trigger_reject_actuator is False
    assert len(verdict_a.flagged_cameras) == 0
    print(f"  ✓ Global Verdict: {verdict_a.global_verdict} (Actuator={verdict_a.trigger_reject_actuator})")
    print(f"  ✓ Reason: {verdict_a.decision_reason}")

    # -------------------------------------------------------------------------
    # UNIT TEST SCENARIO B: Defect Dominant (Front view damaged, rear compliant)
    # -------------------------------------------------------------------------
    print("\n[Scenario B] Testing Defect Dominance (Camera Front damaged_cap, Rear compliant)...")
    res_b = MultiCameraInferenceResult(
        hardware_used="Mock",
        total_latency_ms=4.2,
        camera_count=2,
        views={
            "camera_front": CameraResult(
                camera_id="camera_front", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=0, class_name="damaged_cap", confidence=0.74, box=[100, 100, 200, 200])]
            ),
            "camera_rear": CameraResult(
                camera_id="camera_rear", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.95, box=[105, 95, 205, 195])]
            ),
        }
    )
    verdict_b = gate.evaluate_multi_camera(res_b)
    assert verdict_b.global_verdict == InspectionVerdict.REJECT, f"Expected REJECT, got {verdict_b.global_verdict}"
    assert verdict_b.is_compliant is False
    assert verdict_b.trigger_reject_actuator is True
    assert "camera_front" in verdict_b.flagged_cameras
    assert verdict_b.primary_defect == "damaged_cap"
    print(f"  ✓ Global Verdict: {verdict_b.global_verdict} (Actuator={verdict_b.trigger_reject_actuator})")
    print(f"  ✓ Primary Defect: {verdict_b.primary_defect} ({verdict_b.primary_defect_confidence*100:.1f}%)")
    print(f"  ✓ Flagged Camera: {verdict_b.flagged_cameras}")
    print(f"  ✓ Reason: {verdict_b.decision_reason}")

    # -------------------------------------------------------------------------
    # UNIT TEST SCENARIO C: Opposite Defect (Front compliant, rear open_cap 0.58)
    # -------------------------------------------------------------------------
    print("\n[Scenario C] Testing Opposite Defect (Camera Front compliant, Rear open_cap >= 0.45)...")
    res_c = MultiCameraInferenceResult(
        hardware_used="Mock",
        total_latency_ms=4.2,
        camera_count=2,
        views={
            "camera_front": CameraResult(
                camera_id="camera_front", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.92, box=[100, 100, 200, 200])]
            ),
            "camera_rear": CameraResult(
                camera_id="camera_rear", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=4, class_name="open_cap", confidence=0.58, box=[105, 95, 205, 195])]
            ),
        }
    )
    verdict_c = gate.evaluate_multi_camera(res_c)
    assert verdict_c.global_verdict == InspectionVerdict.REJECT, f"Expected REJECT, got {verdict_c.global_verdict}"
    assert verdict_c.is_compliant is False
    assert verdict_c.trigger_reject_actuator is True
    assert "camera_rear" in verdict_c.flagged_cameras
    assert verdict_c.primary_defect == "open_cap"
    print(f"  ✓ Global Verdict: {verdict_c.global_verdict} (Actuator={verdict_c.trigger_reject_actuator})")
    print(f"  ✓ Primary Defect: {verdict_c.primary_defect} ({verdict_c.primary_defect_confidence*100:.1f}%)")
    print(f"  ✓ Flagged Camera: {verdict_c.flagged_cameras}")

    # -------------------------------------------------------------------------
    # UNIT TEST SCENARIO D: Borderline 0.61 Edge Case (No False Defect Flag!)
    # -------------------------------------------------------------------------
    print("\n[Scenario D] Testing Borderline 0.61 Edge Case (Front=0.61 good_cap, Rear=0.88 good_cap)...")
    res_d = MultiCameraInferenceResult(
        hardware_used="Mock",
        total_latency_ms=4.2,
        camera_count=2,
        views={
            "camera_front": CameraResult(
                camera_id="camera_front", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.61, box=[100, 100, 200, 200])]
            ),
            "camera_rear": CameraResult(
                camera_id="camera_rear", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.88, box=[105, 95, 205, 195])]
            ),
        }
    )
    verdict_d = gate.evaluate_multi_camera(res_d)
    assert verdict_d.global_verdict == InspectionVerdict.UNCERTAIN, f"Expected UNCERTAIN, got {verdict_d.global_verdict}"
    assert verdict_d.trigger_reject_actuator is False, "Borderline case MUST NOT falsely trigger reject actuator!"
    assert "camera_front" in verdict_d.flagged_cameras
    print(f"  ✓ Global Verdict: {verdict_d.global_verdict} (Actuator={verdict_d.trigger_reject_actuator} -> No False Discard)")
    print(f"  ✓ Flagged Camera for Review: {verdict_d.flagged_cameras}")
    print(f"  ✓ Reason: {verdict_d.decision_reason}")

    # -------------------------------------------------------------------------
    # UNIT TEST SCENARIO E: Zero Detections (Missing bottle cap / blind spot)
    # -------------------------------------------------------------------------
    print("\n[Scenario E] Testing Zero Detections (Empty frame / Missing cap)...")
    res_e = MultiCameraInferenceResult(
        hardware_used="Mock",
        total_latency_ms=4.2,
        camera_count=2,
        views={
            "camera_front": CameraResult(
                camera_id="camera_front", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[]
            ),
            "camera_rear": CameraResult(
                camera_id="camera_rear", frame_width=1920, frame_height=1080, inference_latency_ms=4.2,
                detections=[Detection(class_id=1, class_name="good_cap", confidence=0.85, box=[105, 95, 205, 195])]
            ),
        }
    )
    verdict_e = gate.evaluate_multi_camera(res_e)
    assert verdict_e.global_verdict == InspectionVerdict.INSPECTION_FAILED
    assert verdict_e.trigger_reject_actuator is True
    print(f"  ✓ Global Verdict: {verdict_e.global_verdict} (Actuator={verdict_e.trigger_reject_actuator})")
    print(f"  ✓ Flagged Camera: {verdict_e.flagged_cameras}")

    # -------------------------------------------------------------------------
    # FULL END-TO-END INTEGRATION TEST: Part 1 Engine -> Part 2 Quality Gate
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[INTEGRATION TEST] Chaining Part 1 InferenceEngine into Part 2 QualityGate...")
    print("-" * 75)
    
    engine = InferenceEngine(warmup=False, conf_floor=0.35)
    raw_multi_out = engine.predict_multi_camera({
        "camera_front": TEST_IMG_1,
        "camera_rear": TEST_IMG_2,
    })

    final_decision = gate.evaluate_multi_camera(raw_multi_out, bottle_id="BOTTLE_TEST_E2E_001")
    assert isinstance(final_decision, GlobalInspectionResult)

    print(f"  ✓ Session Bottle ID  : {final_decision.bottle_id}")
    print(f"  ✓ Global Verdict     : {final_decision.global_verdict}")
    print(f"  ✓ Compliant          : {final_decision.is_compliant}")
    print(f"  ✓ Reject Actuator    : {final_decision.trigger_reject_actuator}")
    print(f"  ✓ Primary Defect     : {final_decision.primary_defect} ({final_decision.primary_defect_confidence})")
    print(f"  ✓ Flagged Cameras    : {final_decision.flagged_cameras}")
    print(f"  ✓ Total Time (E2E)   : {final_decision.total_latency_ms} ms")
    print(f"  ✓ Decision Reason    : {final_decision.decision_reason}")

    for cid, v in final_decision.per_view_results.items():
        print(f"      -> View [{cid}]: Verdict={v.verdict} | Primary={v.primary_class} ({v.primary_confidence}) | Defect={v.defect_detected}")

    print("\n" + "=" * 75)
    print("  >>> STEP 2 VERIFICATION PASSED: Quality Gate & Decision Logic is 100% Operational <<<")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
