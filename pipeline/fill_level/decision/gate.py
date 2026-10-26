"""
Fill Level Quality Gate and Decision Engine.
Evaluates single-image inference results against quality control rules,
manages gray-zone / borderline routing, and fires the pneumatic reject actuator.
"""
import time
from datetime import datetime, timezone
from typing import Optional
import logging

from pipeline.fill_level.inference.schemas import SingleImageFillResult
from .rules import FillRulesConfig
from .schemas import FillVerdict, FillInspectionDecision

logger = logging.getLogger("pipeline.fill_level.decision")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FillQualityGate:
    """
    Industrial Quality Gate for Liquid Fill Level Inspection.
    1. Rejection Rules:
       - Underfill (< 40.0%): Triggers pneumatic ejector to route bottle to Refill station.
       - Overfill (> 85.0%): Triggers pneumatic ejector to route bottle to Drain station.
       - Empty/Dry Container: Triggers pneumatic ejector.
    2. Borderline / Gray-Zone Handling:
       - Ratios near boundaries (e.g. 38%-40% or 85%-87%) with low confidence route to UNCERTAIN
         for human review in the Active Learning station without false defect alarms.
    3. Conveyor Gap Handling:
       - Blank / empty conveyor frames produce INSPECTION_FAILED without firing actuator.
    4. Deterministic Real-time Latency:
       - Decision rule evaluation strictly executes in < 0.5ms.
    """

    def __init__(self, config: Optional[FillRulesConfig] = None):
        self.config = config or FillRulesConfig()
        logger.info(
            f"FillQualityGate initialized | Nominal Range: {self.config.underfill_threshold*100:.0f}% - "
            f"{self.config.overfill_threshold*100:.0f}% | Gray-Zone Band: ±{self.config.borderline_band*100:.1f}%"
        )

    def evaluate(
        self,
        result: SingleImageFillResult,
        bottle_id: str,
        timestamp: Optional[str] = None
    ) -> FillInspectionDecision:
        """
        Evaluates single-frame machine vision output and generates the authoritative verdict.
        """
        t0 = time.perf_counter()
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # 1. Empty conveyor gap or container occlusion
        if result.fill_verdict == "no_bottle" or result.bottle is None:
            dec_latency = (time.perf_counter() - t0) * 1000.0
            return FillInspectionDecision(
                bottle_id=bottle_id,
                timestamp=ts,
                verdict=FillVerdict.INSPECTION_FAILED,
                is_compliant=False,
                trigger_reject_actuator=False,
                fill_status="no_bottle",
                fill_ratio=None,
                decision_reason="Empty conveyor gap or container occlusion: No bottle detected.",
                decision_latency_ms=round(dec_latency, 3),
                total_latency_ms=round(result.inference_latency_ms + dec_latency, 2),
                bottle=None,
                meniscus=None,
            )

        # 2. Bottle detected, but no liquid meniscus found (Empty or opaque bottle)
        if result.fill_verdict == "no_water" or result.meniscus is None:
            dec_latency = (time.perf_counter() - t0) * 1000.0
            return FillInspectionDecision(
                bottle_id=bottle_id,
                timestamp=ts,
                verdict=FillVerdict.REJECT,
                is_compliant=False,
                trigger_reject_actuator=True,
                fill_status="no_water",
                fill_ratio=0.0,
                decision_reason="Defect confirmed: Bottle detected but liquid meniscus is missing (Empty/Dry container).",
                decision_latency_ms=round(dec_latency, 3),
                total_latency_ms=round(result.inference_latency_ms + dec_latency, 2),
                bottle=result.bottle,
                meniscus=None,
            )

        # 3. Liquid meniscus present: Evaluate ratio and confidence
        ratio = result.fill_ratio if result.fill_ratio is not None else 0.0
        meniscus_conf = result.meniscus.confidence if result.meniscus else 0.0

        # Check for borderline gray-zone with low confidence
        if self.config.is_borderline(ratio) and meniscus_conf < self.config.borderline_conf_floor:
            verdict = FillVerdict.UNCERTAIN
            is_compliant = False
            trigger_actuator = False
            status = "borderline"
            reason = (
                f"Borderline liquid level ({ratio*100:.1f}%) with marginal meniscus confidence ({meniscus_conf*100:.1f}%): "
                f"Routed for Active Learning review."
            )
        # Check underfill defect
        elif self.config.is_underfill(ratio):
            verdict = FillVerdict.REJECT
            is_compliant = False
            trigger_actuator = True
            status = "underfill"
            reason = (
                f"Defect confirmed: Underfill detected at {ratio*100:.1f}% volume "
                f"(minimum tolerance: {self.config.underfill_threshold*100:.1f}%). Route to Refill Station."
            )
        # Check overfill defect
        elif self.config.is_overfill(ratio):
            verdict = FillVerdict.REJECT
            is_compliant = False
            trigger_actuator = True
            status = "overfill"
            reason = (
                f"Defect confirmed: Overfill detected at {ratio*100:.1f}% volume "
                f"(maximum tolerance: {self.config.overfill_threshold*100:.1f}%). Route to Drain Station."
            )
        # Certified nominal pass
        else:
            verdict = FillVerdict.PASS
            is_compliant = True
            trigger_actuator = False
            status = "normal"
            reason = (
                f"Compliant fill level verified ({ratio*100:.1f}% volume within nominal range "
                f"{self.config.underfill_threshold*100:.0f}%-{self.config.overfill_threshold*100:.0f}%)."
            )

        dec_latency = (time.perf_counter() - t0) * 1000.0
        return FillInspectionDecision(
            bottle_id=bottle_id,
            timestamp=ts,
            verdict=verdict,
            is_compliant=is_compliant,
            trigger_reject_actuator=trigger_actuator,
            fill_status=status,
            fill_ratio=ratio,
            decision_reason=reason,
            decision_latency_ms=round(dec_latency, 3),
            total_latency_ms=round(result.inference_latency_ms + dec_latency, 2),
            bottle=result.bottle,
            meniscus=result.meniscus,
        )
