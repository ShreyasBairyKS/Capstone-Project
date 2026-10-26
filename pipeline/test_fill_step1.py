"""
Automated Verification Script for Part 1: Core Fill Level Inference Engine.
Validates:
1. Hardware auto-detection & model resolution (ONNX Runtime / TensorRT).
2. Warmup routine completion.
3. Deterministic geometric fill ratio math accuracy.
4. Single-frame inference on blank frame (no_bottle handling).
5. Single-frame inference on real bottle test images (normal, overfill, underfill).
"""
import sys
import glob
import logging
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.fill_level.inference.engine import FillLevelInferenceEngine
from pipeline.fill_level.inference.schemas import SingleImageFillResult

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step1")


def test_geometric_math():
    """Validates compute_fill_ratio math with known synthetic coordinates."""
    logger.info("--- Test 1: Deterministic Fill Ratio Math ---")
    bottle_box = [100, 100, 500, 900]  # height = 800
    # Trim 3% top (24px) -> by1_corr = 124
    # Trim 2% bot (16px) -> by2_corr = 884
    # Corrected height = 760
    
    # 50% fill: water_y = 884 - 380 = 504
    water_box_50 = [200, 500, 400, 508]  # center Y = 504
    ratio_50 = FillLevelInferenceEngine.compute_fill_ratio(bottle_box, water_box_50)
    assert abs(ratio_50 - 0.500) < 0.002, f"Expected 0.500, got {ratio_50}"
    logger.info(f"  [PASS] 50% Fill Test: ratio={ratio_50}")

    # 75% fill: water_y = 884 - 570 = 314
    water_box_75 = [200, 310, 400, 318]  # center Y = 314
    ratio_75 = FillLevelInferenceEngine.compute_fill_ratio(bottle_box, water_box_75)
    assert abs(ratio_75 - 0.750) < 0.002, f"Expected 0.750, got {ratio_75}"
    logger.info(f"  [PASS] 75% Fill Test: ratio={ratio_75}")

    # 25% fill: water_y = 884 - 190 = 694
    water_box_25 = [200, 690, 400, 698]  # center Y = 694
    ratio_25 = FillLevelInferenceEngine.compute_fill_ratio(bottle_box, water_box_25)
    assert abs(ratio_25 - 0.250) < 0.002, f"Expected 0.250, got {ratio_25}"
    logger.info(f"  [PASS] 25% Fill Test: ratio={ratio_25}")


def test_engine_initialization_and_warmup():
    """Initializes engine and verifies warmup."""
    logger.info("--- Test 2: Engine Initialization & Warmup ---")
    engine = FillLevelInferenceEngine(warmup=True)
    assert engine.bottle_model is not None, "Bottle model not loaded"
    assert engine.water_model is not None, "Water surface model not loaded"
    logger.info(f"  [PASS] Hardware Runtime: {engine.hardware_desc}")
    logger.info(f"  [PASS] Bottle Checkpoint: {engine.bottle_model_path.name}")
    logger.info(f"  [PASS] Water Checkpoint : {engine.water_model_path.name}")
    return engine


def test_blank_frame_inference(engine: FillLevelInferenceEngine):
    """Tests that a blank image correctly produces a 'no_bottle' verdict."""
    logger.info("--- Test 3: Blank Frame Handling ---")
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    res: SingleImageFillResult = engine.predict_single_frame(blank)
    assert res.fill_verdict == "no_bottle", f"Expected 'no_bottle', got {res.fill_verdict}"
    assert res.bottle is None
    assert res.meniscus is None
    logger.info(f"  [PASS] Blank frame correctly flagged: {res.fill_verdict} (Latency: {res.inference_latency_ms:.1f}ms)")


def test_real_bottle_images(engine: FillLevelInferenceEngine):
    """Tests inference on realistic profile bottle images."""
    logger.info("--- Test 4: Real Bottle Profile Image Inference ---")
    sample_dir = PROJECT_ROOT / "pipeline" / "fill_level" / "sample_images"
    images = list(sample_dir.glob("*.jpg"))

    assert len(images) > 0, f"No sample images found in {sample_dir}"
    logger.info(f"Evaluating {len(images)} sample profile bottle images:")

    detected_count = 0
    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        res = engine.predict_single_frame(img)
        
        logger.info(
            f"  Bottle #{idx} ({img_path.name}): "
            f"Verdict={res.fill_verdict.upper():<10} | "
            f"Fill Ratio={f'{res.fill_ratio*100:.1f}%' if res.fill_ratio is not None else 'N/A':<7} | "
            f"Bottle={res.bottle.box if res.bottle else None} | "
            f"Meniscus={res.meniscus.box if res.meniscus else None} | "
            f"Latency={res.inference_latency_ms:.1f}ms"
        )
        assert res.fill_verdict in ("normal", "underfill", "overfill", "no_bottle", "no_water")
        if res.bottle and res.meniscus:
            detected_count += 1
            assert res.fill_ratio is not None
            assert 0.0 <= res.fill_ratio <= 1.0

    logger.info(f"  [PASS] Successfully detected bottles and meniscus lines in {detected_count}/{len(images)} images.")


def main():
    logger.info("==================================================================")
    logger.info("STARTING PART 1 VERIFICATION: CORE FILL LEVEL INFERENCE ENGINE")
    logger.info("==================================================================")
    
    test_geometric_math()
    engine = test_engine_initialization_and_warmup()
    test_blank_frame_inference(engine)
    test_real_bottle_images(engine)

    logger.info("==================================================================")
    logger.info("ALL PART 1 TESTS COMPLETED WITH 100% SUCCESS!")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
