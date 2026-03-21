"""
inference/pipeline.py — EdgeInferencePipeline

Orchestrates the full inference flow for a single product image:
  Frame → YOLOv11 Detection → EfficientViT Classification → MC Dropout UQ
  → Verdict Logic → REMEDY Engine → InspectionResult

This is the Phase 2 deliverable. The class skeleton is provided in Phase 0
so imports across the codebase work during development.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np

from core.config import EdgeConfig, settings
from core.logging import get_logger
from core.schemas import (
    Detection,
    InspectionResult,
    UQResult,
    Verdict,
)
from inference.preprocessor import extract_crop

log = get_logger(__name__)


class EdgeInferencePipeline:
    """
    Full single-product inspection pipeline.

    Phases:
      Phase 2: YOLOv11 + EfficientViT + UQ + verdict logic
      Phase 3: REMEDY engine integration (SeverityScorer + TriageRouter)

    Usage:
        pipeline = EdgeInferencePipeline()
        result = pipeline.inspect(frame, product_id="P001")
    """

    def __init__(self, config: Optional[EdgeConfig] = None) -> None:
        self.config = config or settings
        self._detector = None
        self._classifier = None
        self._uq = None
        self._remedy_scorer = None
        self._remedy_router = None
        self._models_loaded = False
        log.info("pipeline_created", tier=self.config.TIER, device=self.config.DEVICE_ID)

    def load_models(self) -> None:
        """Explicitly load all ONNX models. Called on API startup."""
        from inference.models.yolov11_detector import YOLOv11Detector
        from inference.models.efficientvit_classifier import EfficientViTClassifier
        from inference.models.uq_inspector import MCDropoutUQ
        from remedy.severity_scorer import SeverityScorer
        from remedy.triage_router import TriageRouter

        self._detector = YOLOv11Detector(config=self.config)
        self._detector._load_session()

        # Classifier exported with dropout active (training=True)
        self._classifier = EfficientViTClassifier(config=self.config, enable_dropout=True)
        self._classifier._load_session()

        self._uq = MCDropoutUQ(self._classifier._session, config=self.config)
        self._remedy_scorer = SeverityScorer(config=self.config)
        self._remedy_router = TriageRouter(config=self.config)
        self._models_loaded = True
        log.info("pipeline_models_loaded")

    def inspect(
        self,
        frame: np.ndarray,
        product_id: Optional[str] = None,
        sku: str = "default",
        attempt_count: int = 0,
    ) -> InspectionResult:
        """
        Inspect a single product frame and return a full InspectionResult.

        Args:
            frame:         BGR uint8 numpy array from camera
            product_id:    Optional product identifier for traceability
            sku:           Product SKU for REMEDY profile lookup
            attempt_count: Number of previous inspection attempts (for REMEDY)

        Returns:
            InspectionResult with verdict, detections, severity, and remedy action
        """
        t0 = time.perf_counter()
        inspection_id = str(uuid.uuid4())

        # Phase 2: Replace stub with actual detection
        detections: list[Detection] = self._run_detection(frame)

        # Phase 2: Run UQ only when defects are detected
        uq_result: Optional[UQResult] = None
        if detections:
            uq_result = self._run_uq(frame, detections)

        # Verdict logic (Phase 2)
        verdict, escalated = self._apply_verdict_logic(detections, uq_result)

        # REMEDY (Phase 3)
        severity_result = None
        remediation_action = None
        if verdict in (Verdict.FAIL, Verdict.ESCALATE) and self.config.REMEDY_ENABLED:
            severity_result, remediation_action = self._run_remedy(
                detections, uq_result, sku, attempt_count
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = InspectionResult(
            inspection_id=inspection_id,
            product_id=product_id,
            sku=sku,
            timestamp=datetime.utcnow(),
            verdict=verdict,
            escalated=escalated,
            detections=detections,
            uq_result=uq_result,
            severity_result=severity_result,
            remediation_action=remediation_action,
            latency_ms=round(latency_ms, 2),
            device_id=self.config.DEVICE_ID,
        )

        log.info(
            "inspection_complete",
            inspection_id=inspection_id,
            verdict=verdict.value,
            n_detections=len(detections),
            latency_ms=round(latency_ms, 2),
        )
        return result

    # ---------------------------------------------------------------------- #
    # Internal steps — each implemented in the corresponding phase
    # ---------------------------------------------------------------------- #

    def _run_detection(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv11 detection and return filtered detections."""
        if self._detector is None:
            from inference.models.yolov11_detector import YOLOv11Detector
            self._detector = YOLOv11Detector(config=self.config)
        return self._detector.detect(frame)

    def _run_uq(self, frame: np.ndarray, detections: list[Detection]) -> UQResult:
        """Run MC Dropout UQ on the highest-confidence detection crop."""
        if self._classifier is None:
            from inference.models.efficientvit_classifier import EfficientViTClassifier
            self._classifier = EfficientViTClassifier(
                config=self.config, enable_dropout=True
            )
        if self._uq is None:
            from inference.models.uq_inspector import MCDropoutUQ
            self._uq = MCDropoutUQ(
                self._classifier._session, config=self.config
            )

        # Use the highest-confidence detection's bounding box
        primary = max(detections, key=lambda d: d.confidence)
        bb = primary.bbox
        crop = extract_crop(frame, bb.x1, bb.y1, bb.x2, bb.y2)
        return self._uq.estimate(crop)

    def _apply_verdict_logic(
        self,
        detections: list[Detection],
        uq: Optional[UQResult],
    ) -> tuple[Verdict, bool]:
        """
        Determine inspection verdict from detections and uncertainty.

        Logic (corrected — see review notes):
          No detections           → PASS
          Detections exist:
            conf ≥ AUTO_PASS      AND not uncertain → FAIL (confirmed defect)
            conf ≥ ESCALATE       → FAIL + escalated flag
            conf ≥ HUMAN_REVIEW   → ESCALATE (human queue)
            else                  → REVIEW (uncertain, low confidence)
        """
        if not detections:
            return Verdict.PASS, False

        mean_conf = (
            uq.mean_confidence if uq else max(d.confidence for d in detections)
        )
        is_uncertain = uq.is_uncertain if uq else False

        if mean_conf >= self.config.AUTO_PASS_THRESHOLD and not is_uncertain:
            return Verdict.FAIL, False

        if mean_conf >= self.config.ESCALATE_THRESHOLD:
            return Verdict.FAIL, True

        if mean_conf >= self.config.HUMAN_REVIEW_THRESHOLD:
            return Verdict.ESCALATE, True

        return Verdict.REVIEW, True

    def _run_remedy(self, detections, uq, sku, attempt_count):
        """Score severity and route the primary defect to a remediation action."""
        if self._remedy_scorer is None:
            from remedy.severity_scorer import SeverityScorer
            self._remedy_scorer = SeverityScorer(config=self.config)
        if self._remedy_router is None:
            from remedy.triage_router import TriageRouter
            self._remedy_router = TriageRouter(config=self.config)

        primary = max(detections, key=lambda d: d.confidence)
        severity = self._remedy_scorer.score(primary, uq, attempt_count)
        action = self._remedy_router.route(primary, severity, attempt_count)
        return severity, action
