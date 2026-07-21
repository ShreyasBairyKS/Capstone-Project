"""
barcode_check.py
────────────────
Barcode decode and content verification using pyzbar.

Why a content check alongside the anomaly model:
  Anomaly models detect surface defects — they cannot verify text content.
  A barcode may look perfectly printed (no surface anomaly) but encode the
  wrong SKU if the wrong label reel was loaded. This check catches that.

Two sub-checks:
  1. Decode check  — Can pyzbar read the barcode at all?
                     A smeared or damaged barcode fails here.
  2. Content check — Does the decoded string match the expected SKU?
                     Only active if expected_sku is set in thresholds.yaml.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BarcodeCheckResult:
    decoded_value:    Optional[str]   # Decoded string, or None if decode failed
    decode_success:   bool
    content_match:    bool            # True if SKU matches (or no SKU check configured)
    passed:           bool
    failure_reason:   Optional[str]


def check_barcode(
    barcode_roi_crop: np.ndarray,
    expected_sku: str = "",
    require_decode: bool = True,
) -> BarcodeCheckResult:
    """
    Decode the barcode from the ROI_BARCODE crop and verify content.

    Args:
        barcode_roi_crop: Cropped BGR image of the barcode region.
        expected_sku:     Expected barcode string. Empty = skip content check.
        require_decode:   If True, a failed decode → FAIL verdict.

    Returns:
        BarcodeCheckResult.
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError:
        return BarcodeCheckResult(
            decoded_value=None,
            decode_success=False,
            content_match=True,
            passed=True,
            failure_reason="pyzbar not installed — barcode check skipped",
        )

    # Grayscale + CLAHE contrast enhancement for better decode on low-contrast crops
    gray    = cv2.cvtColor(barcode_roi_crop, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    decoded = None
    for img in [gray, enhanced]:
        results = pyzbar_decode(img)
        if results:
            decoded = results[0].data.decode("utf-8").strip()
            break

    decode_success  = decoded is not None
    content_match   = True
    failure_reason  = None

    if not decode_success:
        if require_decode:
            failure_reason = "Barcode could not be decoded (damaged or smeared)"
    elif expected_sku and decoded != expected_sku:
        content_match  = False
        failure_reason = f"SKU mismatch: expected '{expected_sku}', got '{decoded}'"

    passed = (decode_success or not require_decode) and content_match

    return BarcodeCheckResult(
        decoded_value=decoded,
        decode_success=decode_success,
        content_match=content_match,
        passed=passed,
        failure_reason=failure_reason,
    )
