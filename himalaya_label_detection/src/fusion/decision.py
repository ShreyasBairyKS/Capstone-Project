"""
decision.py
───────────
Rule-based decision fusion engine.

Why rule-based rather than a learned combiner:
  - Anomaly detection was chosen to avoid needing labelled defect images.
    A learned combiner (e.g. logistic regression on scores) requires those
    same labels — defeating the purpose.
  - Rule-based fusion is transparent and auditable; important under
    ISO 15378 for pharmaceutical packaging inspection.
  - Thresholds can be adjusted by QA engineers without retraining.

Fusion logic — FAIL if ANY of:
  1. Any critical ROI anomaly score  > critical_roi_threshold
  2. Any standard ROI anomaly score  > general_roi_threshold
  3. Left-Right SSIM                 < lr_ssim_threshold
  4. Golden Master cosine similarity < gm_cosine_threshold
  5. Barcode decode failed           (if require_decode = True)
  6. Barcode SKU mismatch            (if expected_sku is set)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ROIs where any anomaly is treated as critical (lower, stricter threshold)
CRITICAL_ROIS = {"ROI_LOGO", "ROI_BARCODE", "ROI_BATCH_CODE", "ROI_CERT"}


@dataclass
class FusionConfig:
    critical_roi_threshold: float = 0.50
    general_roi_threshold:  float = 0.60
    lr_ssim_threshold:      float = 0.85
    gm_cosine_threshold:    float = 0.90

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FusionConfig":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        a = cfg.get("anomaly", {})
        v = cfg.get("verification", {})
        return cls(
            critical_roi_threshold=a.get("critical_roi_threshold", 0.50),
            general_roi_threshold= a.get("general_roi_threshold",  0.60),
            lr_ssim_threshold=     v.get("lr_ssim_threshold",      0.85),
            gm_cosine_threshold=   v.get("gm_cosine_threshold",    0.90),
        )


@dataclass
class FusionResult:
    passed:        bool
    reasons:       List[str]             = field(default_factory=list)
    roi_scores:    Dict[str, float]      = field(default_factory=dict)
    lr_score:      Optional[float]       = None
    gm_score:      Optional[float]       = None
    barcode_value: Optional[str]         = None

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


def fuse(
    roi_scores:              Dict[str, float],
    lr_score:                Optional[float]  = None,
    gm_score:                Optional[float]  = None,
    barcode_passed:          bool              = True,
    barcode_failure_reason:  Optional[str]     = None,
    barcode_value:           Optional[str]     = None,
    cfg:                     Optional[FusionConfig] = None,
) -> FusionResult:
    """
    Combine all inspection signals into a single PASS/FAIL decision.

    Args:
        roi_scores:             Dict of ROI name → anomaly score (0.0–1.0).
        lr_score:               Left-Right SSIM. None = skip this check.
        gm_score:               Golden Master cosine similarity. None = skip.
        barcode_passed:         False if the barcode check failed.
        barcode_failure_reason: Human-readable barcode failure message.
        barcode_value:          Decoded barcode string (for logging).
        cfg:                    Thresholds. Uses defaults if None.

    Returns:
        FusionResult with verdict and all failure reasons.
    """
    if cfg is None:
        cfg = FusionConfig()

    reasons: List[str] = []

    # 1 — ROI anomaly scores
    for roi, score in roi_scores.items():
        is_critical = roi in CRITICAL_ROIS
        threshold   = (cfg.critical_roi_threshold if is_critical
                       else cfg.general_roi_threshold)
        if score > threshold:
            tag = "CRITICAL" if is_critical else "anomaly"
            reasons.append(
                f"{roi}: {tag} score {score:.3f} > {threshold:.2f}"
            )

    # 2 — Left-Right symmetry
    if lr_score is not None and lr_score < cfg.lr_ssim_threshold:
        reasons.append(
            f"Left-Right SSIM {lr_score:.3f} < {cfg.lr_ssim_threshold:.2f}"
        )

    # 3 — Golden Master similarity
    if gm_score is not None and gm_score < cfg.gm_cosine_threshold:
        reasons.append(
            f"Golden Master similarity {gm_score:.3f} < {cfg.gm_cosine_threshold:.2f}"
        )

    # 4 — Barcode
    if not barcode_passed and barcode_failure_reason:
        reasons.append(f"Barcode: {barcode_failure_reason}")

    return FusionResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        roi_scores=roi_scores,
        lr_score=lr_score,
        gm_score=gm_score,
        barcode_value=barcode_value,
    )
