"""
Quality Gate and Multi-View Decision Engine.
Evaluates single and multi-camera inference results against quality control rules.
"""
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict
import logging

from pipeline.inference.schemas import CameraResult, MultiCameraInferenceResult, Detection
from .schemas import InspectionVerdict, ViewVerdict, GlobalInspectionResult
from .rules import InspectionRulesConfig

logger = logging.getLogger("pipeline.decision")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class QualityGate:
    """
    Industrial Quality Gate implementing:
    1. Asymmetric thresholding (strict for pass, sensitive for defect).
    2. Defect-dominant multi-view aggregation (any camera sees defect -> REJECT).
    3. Gray-zone handling (0.40 - 0.75 routed to UNCERTAIN without false defect alarms).
    4. Hardware actuator trigger decision (<10ms).
    """

    def __init__(self, config: Optional[InspectionRulesConfig] = None):
        self.config = config or InspectionRulesConfig()
        logger.info(
            f"QualityGate initialized | Pass Conf: >={self.config.pass_conf_threshold*100:.0f}% | "
            f"Defect Conf: >={self.config.defect_conf_threshold*100:.0f}% | "
            f"Uncertain Floor: {self.config.uncertain_floor*100:.0f}%"
        )

    def evaluate_view(self, camera_result: CameraResult) -> ViewVerdict:
        """
        Evaluates detections from a single camera view in complete isolation.
        """
        cid = camera_result.camera_id
        detections = camera_result.detections

        # 1. Zero Detections Check
        if not detections:
            return ViewVerdict(
                camera_id=cid,
                verdict=InspectionVerdict.INSPECTION_FAILED,
                decision_reason="Zero detections found: No cap or extreme occlusion in this view.",
                detections=[],
            )

        # 2. Check for confirmed defects first (Defect Sensitivity)
        # Any detection where class is in defect_classes and confidence >= defect_conf_threshold
        defect_candidates = [
            d for d in detections
            if self.config.is_defect(d.class_name) and d.confidence >= self.config.defect_conf_threshold
        ]

        if defect_candidates:
            # Sort defects by confidence descending
            top_defect = max(defect_candidates, key=lambda d: d.confidence)
            return ViewVerdict(
                camera_id=cid,
                verdict=InspectionVerdict.REJECT,
                primary_class=top_defect.class_name,
                primary_confidence=top_defect.confidence,
                defect_detected=True,
                defect_name=top_defect.class_name,
                defect_confidence=top_defect.confidence,
                decision_reason=f"Defect confirmed: '{top_defect.class_name}' with {top_defect.confidence*100:.1f}% confidence.",
                detections=detections,
            )

        # 3. Check for certified pass (High-bar requirement)
        top_detection = max(detections, key=lambda d: d.confidence)
        
        if top_detection.class_name == self.config.pass_class:
            if top_detection.confidence >= self.config.pass_conf_threshold:
                # Certified PASS
                return ViewVerdict(
                    camera_id=cid,
                    verdict=InspectionVerdict.PASS,
                    primary_class=top_detection.class_name,
                    primary_confidence=top_detection.confidence,
                    defect_detected=False,
                    decision_reason=f"Compliant cap verified ({top_detection.confidence*100:.1f}% confidence).",
                    detections=detections,
                )
            elif top_detection.confidence >= self.config.uncertain_floor:
                # Borderline good_cap (0.40 <= conf < 0.75) -> UNCERTAIN, NOT REJECT
                # Guards against surface glare or edge angle false defect alarms
                return ViewVerdict(
                    camera_id=cid,
                    verdict=InspectionVerdict.UNCERTAIN,
                    primary_class=top_detection.class_name,
                    primary_confidence=top_detection.confidence,
                    defect_detected=False,
                    decision_reason=(
                        f"Borderline confidence on '{top_detection.class_name}' ({top_detection.confidence*100:.1f}%). "
                        f"Below pass bar ({self.config.pass_conf_threshold*100:.0f}%) but no defect identified. "
                        f"Routed for review without false rejection."
                    ),
                    detections=detections,
                )

        # 4. Fallback for unclassified anomalies or very low confidence
        return ViewVerdict(
            camera_id=cid,
            verdict=InspectionVerdict.UNCERTAIN,
            primary_class=top_detection.class_name,
            primary_confidence=top_detection.confidence,
            defect_detected=False,
            decision_reason=f"Uncertain anomaly: '{top_detection.class_name}' at low confidence ({top_detection.confidence*100:.1f}%).",
            detections=detections,
        )

    def evaluate_multi_camera(
        self,
        multi_result: MultiCameraInferenceResult,
        bottle_id: Optional[str] = None,
    ) -> GlobalInspectionResult:
        """
        Aggregates multi-camera results under the 'Defect-Dominant' rule.
        """
        t0 = time.perf_counter()
        session_id = bottle_id or f"BOTTLE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step A: Evaluate each view in complete isolation
        per_view_results: Dict[str, ViewVerdict] = {}
        for cid, cam_res in multi_result.views.items():
            per_view_results[cid] = self.evaluate_view(cam_res)

        # Step B: Multi-View Aggregation Logic
        reject_views = [cid for cid, v in per_view_results.items() if v.verdict == InspectionVerdict.REJECT]
        failed_views = [cid for cid, v in per_view_results.items() if v.verdict == InspectionVerdict.INSPECTION_FAILED]
        uncertain_views = [cid for cid, v in per_view_results.items() if v.verdict == InspectionVerdict.UNCERTAIN]
        pass_views = [cid for cid, v in per_view_results.items() if v.verdict == InspectionVerdict.PASS]

        # Rule 1: Defect Dominant (Any defect on any side -> REJECT)
        if reject_views:
            # Find the primary defect (highest confidence among all defect views)
            top_defect_verdict = max(
                (per_view_results[cid] for cid in reject_views),
                key=lambda v: v.defect_confidence or 0.0,
            )
            global_verdict = InspectionVerdict.REJECT
            is_compliant = False
            trigger_actuator = True
            primary_defect = top_defect_verdict.defect_name
            primary_defect_conf = top_defect_verdict.defect_confidence
            flagged_cameras = reject_views
            reason = (
                f"REJECT: Defect '{primary_defect}' ({primary_defect_conf*100:.1f}%) observed on "
                f"{len(reject_views)} camera view(s): {', '.join(reject_views)}."
            )

        # Rule 2: Inspection Failure (Missing cap / blind spot)
        elif failed_views:
            global_verdict = InspectionVerdict.INSPECTION_FAILED
            is_compliant = False
            trigger_actuator = True  # In manufacturing, missing/unidentifiable cap triggers diversion
            primary_defect = "NO_CAP_OR_BLIND_SPOT"
            primary_defect_conf = 1.0
            flagged_cameras = failed_views
            reason = f"INSPECTION_FAILED: Camera(s) {', '.join(failed_views)} detected no bottle cap."

        # Rule 3: Uncertain (Borderline scores, no confirmed defect)
        elif uncertain_views:
            global_verdict = InspectionVerdict.UNCERTAIN
            is_compliant = False
            trigger_actuator = False  # Avoid false mechanical discard; route to review queue
            primary_defect = None
            primary_defect_conf = None
            flagged_cameras = uncertain_views
            reason = (
                f"UNCERTAIN: Camera(s) {', '.join(uncertain_views)} reported borderline confidence. "
                f"Saved to review queue without false defect discard."
            )

        # Rule 4: All Compliant (Every view passed)
        else:
            global_verdict = InspectionVerdict.PASS
            is_compliant = True
            trigger_actuator = False
            primary_defect = None
            primary_defect_conf = None
            flagged_cameras = []
            reason = f"PASS: All {len(pass_views)} camera angles verified compliant bottle cap."

        decision_latency_ms = (time.perf_counter() - t0) * 1000
        total_time_ms = round(multi_result.total_latency_ms + decision_latency_ms, 2)

        return GlobalInspectionResult(
            bottle_id=session_id,
            timestamp=timestamp,
            global_verdict=global_verdict,
            is_compliant=is_compliant,
            trigger_reject_actuator=trigger_actuator,
            primary_defect=primary_defect,
            primary_defect_confidence=round(primary_defect_conf, 4) if primary_defect_conf else None,
            flagged_cameras=flagged_cameras,
            decision_reason=reason,
            total_latency_ms=total_time_ms,
            per_view_results=per_view_results,
        )
