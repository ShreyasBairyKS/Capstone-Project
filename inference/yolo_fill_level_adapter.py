"""
Adapter that exposes inference/yoloWithFillLevel.py through the API contract.

The original script remains usable from the command line. This module imports it
as a library, keeps the YOLO models warm across requests, and maps its per-bottle
results into the dashboard's InspectionResult shape.
"""

from __future__ import annotations

import base64
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import EdgeConfig, settings
from core.schemas import (
    BoundingBox,
    DEFECT_CLASS_NAMES,
    DefectClass,
    Detection,
    InspectionResult,
    Verdict,
)
from remedy.severity_scorer import SeverityScorer
from remedy.triage_router import TriageRouter


class YoloFillLevelPipeline:
    """Lazy-loaded YOLO + fill-level pipeline used by the live dashboard."""

    def __init__(self, config: Optional[EdgeConfig] = None) -> None:
        self.config = config or settings
        self._detector = None
        self._fill_model = None
        self._classifier = None
        self._cls_device = None
        self._classifier_path: Optional[Path] = None
        self._severity_scorer = SeverityScorer(config=self.config)
        self._triage_router = TriageRouter(config=self.config)

    def inspect(
        self,
        frame: np.ndarray,
        product_id: Optional[str] = None,
        sku: str = "default",
        attempt_count: int = 0,
        use_cap_classifier: bool = True,
        product_category: Optional[str] = None,
        product_sub_type: Optional[str] = None,
        container_contents: Optional[str] = None,
    ) -> InspectionResult:
        """Run yoloWithFillLevel.py and return an InspectionResult."""
        t0 = time.perf_counter()
        self._ensure_models(use_cap_classifier=use_cap_classifier)

        from inference.yoloWithFillLevel import draw, full_inspect

        result = full_inspect(
            self._detector,
            self._fill_model,
            self._classifier if use_cap_classifier else None,
            self._cls_device,
            frame,
            self.config.YOLO_FILL_DEVICE,
        )

        annotated = draw(frame, result)
        annotated_b64 = self._encode_jpeg_b64(annotated)

        detections = self._detections_from_result(result, frame.shape)
        verdict = Verdict.FAIL if detections else Verdict.PASS
        escalated = False

        severity_result = None
        remediation_action = None
        if detections and self.config.REMEDY_ENABLED:
            primary = max(detections, key=lambda d: d.confidence)
            severity_result = self._severity_scorer.score(primary, None, attempt_count)
            remediation_action = self._triage_router.route(primary, severity_result, attempt_count)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return InspectionResult(
            inspection_id=str(uuid.uuid4()),
            product_id=product_id,
            sku=sku,
            timestamp=datetime.utcnow(),
            product_category=product_category,
            product_sub_type=product_sub_type,
            container_contents=container_contents,
            verdict=verdict,
            escalated=escalated,
            detections=detections,
            severity_result=severity_result,
            remediation_action=remediation_action,
            annotated_image_b64=annotated_b64,
            inference_summary=self._summary_from_result(result, use_cap_classifier),
            latency_ms=round(latency_ms, 2),
            device_id=self.config.DEVICE_ID,
        )

    def _ensure_models(self, use_cap_classifier: bool) -> None:
        from ultralytics import YOLO

        from inference.yoloWithFillLevel import BottleCapDetector, load_classifier

        det_weights = self._require_path(self.config.YOLO_FILL_DETECTOR_WEIGHTS)
        fill_weights = self._require_path(self.config.YOLO_FILL_WATER_WEIGHTS)

        if self._detector is None:
            self._detector = BottleCapDetector(
                str(det_weights),
                device=self.config.YOLO_FILL_DEVICE,
                det_conf=self.config.YOLO_FILL_CONF_THRESHOLD,
                zoom_scale=self.config.YOLO_FILL_ZOOM_SCALE,
            )

        if self._fill_model is None:
            self._fill_model = YOLO(str(fill_weights))

        if use_cap_classifier:
            cls_weights = self._require_path(self.config.YOLO_FILL_CAP_CLASSIFIER_WEIGHTS)
            if self._classifier is None or self._classifier_path != cls_weights:
                self._classifier, self._cls_device = load_classifier(
                    str(cls_weights),
                    self.config.YOLO_FILL_DEVICE,
                )
                self._classifier_path = cls_weights

    @staticmethod
    def _require_path(path: Path) -> Path:
        resolved = Path(path)
        if not resolved.exists():
            raise RuntimeError(f"YOLO fill-level model file not found: {resolved}")
        return resolved

    def _detections_from_result(self, result: dict, shape: tuple[int, ...]) -> list[Detection]:
        detections: list[Detection] = []

        for bottle in result.get("bottles", []):
            fill_level = bottle.get("fill_level")
            if fill_level in {"underfill", "overfill"}:
                class_name = (
                    DefectClass.FILL_LEVEL_LOW
                    if fill_level == "underfill"
                    else DefectClass.FILL_LEVEL_HIGH
                )
                bbox = bottle.get("water_bbox") or bottle.get("bottle_bbox")
                confidence = bottle.get("water_conf") or bottle.get("bottle_conf") or 0.5
                detections.append(self._make_detection(class_name, confidence, bbox, shape))

            if bottle.get("cap_verdict") in {"Missing Cap", "Defective Cap"}:
                bbox = bottle.get("cap_bbox") or self._fallback_cap_bbox(bottle.get("bottle_bbox"))
                confidence = (
                    bottle.get("cap_quality_conf")
                    or bottle.get("cap_detection_conf")
                    or bottle.get("bottle_conf")
                    or 0.5
                )
                detections.append(
                    self._make_detection(
                        DefectClass.CAP_FITTING_ANOMALY,
                        confidence,
                        bbox,
                        shape,
                    )
                )

        return detections

    @staticmethod
    def _fallback_cap_bbox(bottle_bbox: list[int] | None) -> list[int]:
        if not bottle_bbox:
            return [0, 0, 1, 1]
        x1, y1, x2, y2 = bottle_bbox
        cap_h = max(1, int((y2 - y1) * 0.2))
        return [x1, y1, x2, min(y2, y1 + cap_h)]

    def _make_detection(
        self,
        class_name: DefectClass,
        confidence: float,
        bbox_px: list[int],
        shape: tuple[int, ...],
    ) -> Detection:
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox_px
        bbox = BoundingBox(
            x1=self._clamp01(x1 / max(w, 1)),
            y1=self._clamp01(y1 / max(h, 1)),
            x2=self._clamp01(x2 / max(w, 1)),
            y2=self._clamp01(y2 / max(h, 1)),
        )
        return Detection(
            class_id=DEFECT_CLASS_NAMES.index(class_name.value),
            class_name=class_name,
            confidence=self._clamp01(float(confidence)),
            bbox=bbox,
            bbox_area_ratio=bbox.area_ratio,
        )

    @staticmethod
    def _summary_from_result(result: dict, use_cap_classifier: bool) -> dict:
        bottles = []
        for bottle in result.get("bottles", []):
            bottles.append(
                {
                    "bottle_index": bottle.get("bottle_idx"),
                    "bottle_bbox": bottle.get("bottle_bbox"),
                    "bottle_confidence": bottle.get("bottle_conf"),
                    "cap": {
                        "verdict": bottle.get("cap_verdict"),
                        "bbox": bottle.get("cap_bbox"),
                        "quality": bottle.get("cap_quality"),
                        "quality_confidence": bottle.get("cap_quality_conf"),
                        "detection_confidence": bottle.get("cap_detection_conf"),
                        "detection_source": bottle.get("cap_detection_source"),
                    },
                    "fill": {
                        "level": bottle.get("fill_level"),
                        "ratio": bottle.get("fill_ratio"),
                        "water_bbox": bottle.get("water_bbox"),
                        "water_confidence": bottle.get("water_conf"),
                    },
                }
            )

        return {
            "pipeline": "yolo_fill_level",
            "cap_classifier_enabled": use_cap_classifier,
            "bottle_count": len(result.get("bottles", [])),
            "caps_pass1": result.get("caps_p1", 0),
            "caps_pass2": result.get("caps_p2", 0),
            "annotations": bottles,
        }

    @staticmethod
    def _encode_jpeg_b64(frame: np.ndarray) -> Optional[str]:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
