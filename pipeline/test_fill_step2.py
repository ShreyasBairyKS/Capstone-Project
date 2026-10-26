"""
Automated Verification Script for Part 2: Fill Quality Gate and Decision Logic.
Validates:
1. Nominal Pass evaluation (within 40% - 85%).
2. Underfill Rejection & pneumatic actuator triggering (< 40%).
3. Overfill Rejection & pneumatic actuator triggering (> 85%).
4. Dry/Empty container rejection (no_water).
5. Empty conveyor gap handling (no_bottle -> INSPECTION_FAILED).
6. Borderline gray-zone routing to UNCERTAIN.
7. Sub-millisecond decision evaluation latency benchmark.
8. End-to-end integration: FillLevelInferenceEngine -> FillQualityGate on real sample images.
"""
import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.fill_level.inference.schemas import BottleBox, MeniscusBox, SingleImageFillResult
from pipeline.fill_level.inference.engine import FillLevelInferenceEngine
from pipeline.fill_level.decision.rules import FillRulesConfig
from pipeline.fill_level.decision.schemas import FillVerdict, FillInspectionDecision
from pipeline.fill_level.decision.gate import FillQualityGate

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step2")


def test_nominal_pass():
    """Validates certified pass on nominal volume."""
    logger.info("--- Test 1: Nominal Pass (72.3% Fill) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=12.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.95),
        meniscus=MeniscusBox(box=[130, 180, 270, 190], water_y=185, confidence=0.88),
        fill_ratio=0.723,
        fill_verdict="normal"
    )
    dec: FillInspectionDecision = gate.evaluate(mock_res, bottle_id="BOTTLE_NOMINAL_001")
    assert dec.verdict == FillVerdict.PASS, f"Expected PASS, got {dec.verdict}"
    assert dec.is_compliant is True
    assert dec.trigger_reject_actuator is False
    assert dec.fill_status == "normal"
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_underfill_rejection():
    """Validates underfill detection and actuator trigger."""
    logger.info("--- Test 2: Underfill Rejection (32.1% Fill) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=12.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.95),
        meniscus=MeniscusBox(box=[130, 380, 270, 390], water_y=385, confidence=0.85),
        fill_ratio=0.321,
        fill_verdict="underfill"
    )
    dec = gate.evaluate(mock_res, bottle_id="BOTTLE_UNDERFILL_001")
    assert dec.verdict == FillVerdict.REJECT, f"Expected REJECT, got {dec.verdict}"
    assert dec.is_compliant is False
    assert dec.trigger_reject_actuator is True
    assert dec.fill_status == "underfill"
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_overfill_rejection():
    """Validates overfill detection and actuator trigger."""
    logger.info("--- Test 3: Overfill Rejection (90.2% Fill) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=12.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.95),
        meniscus=MeniscusBox(box=[130, 90, 270, 100], water_y=95, confidence=0.91),
        fill_ratio=0.902,
        fill_verdict="overfill"
    )
    dec = gate.evaluate(mock_res, bottle_id="BOTTLE_OVERFILL_001")
    assert dec.verdict == FillVerdict.REJECT, f"Expected REJECT, got {dec.verdict}"
    assert dec.is_compliant is False
    assert dec.trigger_reject_actuator is True
    assert dec.fill_status == "overfill"
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_dry_bottle_rejection():
    """Validates that a container without meniscus triggers REJECT and actuator."""
    logger.info("--- Test 4: Dry/Empty Container (no_water) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=10.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.92),
        meniscus=None,
        fill_ratio=0.0,
        fill_verdict="no_water"
    )
    dec = gate.evaluate(mock_res, bottle_id="BOTTLE_DRY_001")
    assert dec.verdict == FillVerdict.REJECT, f"Expected REJECT, got {dec.verdict}"
    assert dec.is_compliant is False
    assert dec.trigger_reject_actuator is True
    assert dec.fill_status == "no_water"
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_conveyor_gap():
    """Validates that an empty conveyor gap produces INSPECTION_FAILED without firing actuator."""
    logger.info("--- Test 5: Empty Conveyor Gap (no_bottle) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=8.0,
        bottle=None,
        meniscus=None,
        fill_ratio=None,
        fill_verdict="no_bottle"
    )
    dec = gate.evaluate(mock_res, bottle_id="BOTTLE_GAP_001")
    assert dec.verdict == FillVerdict.INSPECTION_FAILED, f"Expected INSPECTION_FAILED, got {dec.verdict}"
    assert dec.is_compliant is False
    assert dec.trigger_reject_actuator is False
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_borderline_gray_zone():
    """Validates that borderline fill ratios near threshold with low confidence route to UNCERTAIN."""
    logger.info("--- Test 6: Borderline Gray-Zone (39.0% Fill, low conf) ---")
    gate = FillQualityGate(FillRulesConfig(borderline_band=0.02, borderline_conf_floor=0.50))
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=12.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.88),
        meniscus=MeniscusBox(box=[130, 350, 270, 360], water_y=355, confidence=0.35),
        fill_ratio=0.390,  # 39.0% is within 0.02 of 0.40 threshold
        fill_verdict="underfill"
    )
    dec = gate.evaluate(mock_res, bottle_id="BOTTLE_BORDERLINE_001")
    assert dec.verdict == FillVerdict.UNCERTAIN, f"Expected UNCERTAIN, got {dec.verdict}"
    assert dec.is_compliant is False
    assert dec.trigger_reject_actuator is False
    assert dec.fill_status == "borderline"
    logger.info(f"  [PASS] Verdict: {dec.verdict} | Actuator: {dec.trigger_reject_actuator} | Reason: {dec.decision_reason}")


def test_latency_benchmark():
    """Validates sub-millisecond evaluation speed."""
    logger.info("--- Test 7: Decision Latency Benchmark (1000 iterations) ---")
    gate = FillQualityGate()
    mock_res = SingleImageFillResult(
        hardware_used="CPU",
        inference_latency_ms=10.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.95),
        meniscus=MeniscusBox(box=[130, 180, 270, 190], water_y=185, confidence=0.88),
        fill_ratio=0.723,
        fill_verdict="normal"
    )

    t0 = time.perf_counter()
    for i in range(1000):
        _ = gate.evaluate(mock_res, bottle_id=f"BOTTLE_BENCH_{i}")
    total_time = (time.perf_counter() - t0) * 1000.0
    avg_latency = total_time / 1000.0

    logger.info(f"  [PASS] Total 1000 evals: {total_time:.2f}ms | Avg Latency: {avg_latency*1000:.1f}µs ({avg_latency:.4f}ms)")
    assert avg_latency < 0.5, f"Expected <0.5ms average latency, got {avg_latency}ms"


def test_end_to_end_integration():
    """Tests FillLevelInferenceEngine -> FillQualityGate on real sample images."""
    logger.info("--- Test 8: End-to-End Inference + Decision Pipeline ---")
    engine = FillLevelInferenceEngine(warmup=False)
    gate = FillQualityGate()

    samples_dir = PROJECT_ROOT / "pipeline" / "fill_level" / "sample_images"
    sample_files = [
        ("sample_bottle_1_normal.jpg", FillVerdict.PASS),
        ("sample_bottle_2_overfill.jpg", FillVerdict.REJECT),
        ("sample_bottle_3_normal.jpg", FillVerdict.PASS),
    ]

    for filename, expected_verdict in sample_files:
        path = samples_dir / filename
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        inf_res = engine.predict_single_frame(img)
        dec = gate.evaluate(inf_res, bottle_id=filename.replace(".jpg", ""))
        logger.info(
            f"  {filename}: Expected={expected_verdict.value:<7} | "
            f"Actual={dec.verdict.value:<7} | "
            f"Ratio={f'{dec.fill_ratio*100:.1f}%' if dec.fill_ratio is not None else 'N/A'} | "
            f"Actuator={dec.trigger_reject_actuator} | "
            f"Total Latency={dec.total_latency_ms:.1f}ms"
        )
        assert dec.verdict == expected_verdict, f"Mismatch on {filename}: expected {expected_verdict}, got {dec.verdict}"

    # Blank frame end-to-end
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    blank_inf = engine.predict_single_frame(blank)
    blank_dec = gate.evaluate(blank_inf, bottle_id="BOTTLE_BLANK")
    assert blank_dec.verdict == FillVerdict.INSPECTION_FAILED
    assert blank_dec.trigger_reject_actuator is False
    logger.info(f"  Blank Frame: Expected=INSPECTION_FAILED | Actual={blank_dec.verdict.value} | Actuator={blank_dec.trigger_reject_actuator}")


def main():
    logger.info("==================================================================")
    logger.info("STARTING PART 2 VERIFICATION: FILL QUALITY GATE & DECISION LOGIC")
    logger.info("==================================================================")

    test_nominal_pass()
    test_underfill_rejection()
    test_overfill_rejection()
    test_dry_bottle_rejection()
    test_conveyor_gap()
    test_borderline_gray_zone()
    test_latency_benchmark()
    test_end_to_end_integration()

    logger.info("==================================================================")
    logger.info("ALL PART 2 TESTS COMPLETED WITH 100% SUCCESS!")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
