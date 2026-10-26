"""
Fill Level Audit Logger and Asset Bundler.
Saves training-ready unblemished raw frames, normalized YOLO format labels,
structured telemetry JSON, and coordinates SQLite persistence.
"""
import io
import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import cv2
import numpy as np

from pipeline.fill_level.decision.schemas import FillVerdict, FillInspectionDecision
from .schemas import (
    FillAuditConfig,
    FillKPISummary,
    FillInspectionFilter,
    FillInspectionRecord,
)
from .db import FillAuditDB

logger = logging.getLogger("pipeline.fill_level.audit.logger")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FillAuditLogger:
    """
    Coordinates isolated bottle bundling and persistent SQLite database operations for Station 2.
    """

    def __init__(self, config: Optional[FillAuditConfig] = None):
        self.config = config or FillAuditConfig()
        self.db = FillAuditDB(self.config.db_path)
        self.bottles_dir = Path(self.config.base_dir) / self.config.bottles_subfolder
        self.bottles_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FillAuditLogger online | Storage: {self.bottles_dir} | DB: {self.config.db_path}")

    def log_inspection(
        self,
        frame: np.ndarray,
        decision: FillInspectionDecision
    ) -> FillInspectionRecord:
        """
        Persists inspection telemetry to database and writes training-ready assets to disk.
        """
        t0 = time.perf_counter()
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in decision.bottle_id)
        bundle_path = self.bottles_dir / safe_id
        bundle_path.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]

        # 1. Save Unburned Raw Frame (Training-Ready)
        raw_img_path = bundle_path / "raw_frame.jpg"
        if self.config.save_raw_image and frame is not None and frame.size > 0:
            cv2.imwrite(str(raw_img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 2. Generate Normalized YOLO Ground-Truth Labels for Active Learning
        labels_dir = bundle_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = labels_dir / "water_surface.txt"

        yolo_lines: List[str] = []
        if decision.meniscus is not None:
            wx1, wy1, wx2, wy2 = decision.meniscus.box
            xc = max(0.0, min(1.0, ((wx1 + wx2) / 2.0) / w))
            yc = max(0.0, min(1.0, ((wy1 + wy2) / 2.0) / h))
            bw = max(0.0, min(1.0, (wx2 - wx1) / w))
            bh = max(0.0, min(1.0, (wy2 - wy1) / h))
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_file, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        # 3. Save Structured Telemetry JSON
        meta_file = bundle_path / "inspection_data.json"
        meta_dict = {
            "bottle_id": decision.bottle_id,
            "timestamp": decision.timestamp,
            "global_verdict": decision.verdict.value,
            "fill_status": decision.fill_status,
            "fill_ratio": decision.fill_ratio,
            "is_compliant": decision.is_compliant,
            "trigger_reject_actuator": decision.trigger_reject_actuator,
            "decision_reason": decision.decision_reason,
            "total_latency_ms": decision.total_latency_ms,
            "bottle_box": decision.bottle.box if decision.bottle else None,
            "bottle_confidence": decision.bottle.confidence if decision.bottle else None,
            "meniscus_box": decision.meniscus.box if decision.meniscus else None,
            "meniscus_confidence": decision.meniscus.confidence if decision.meniscus else None,
            "water_y": decision.meniscus.water_y if decision.meniscus else None,
            "raw_frame_file": "raw_frame.jpg",
            "yolo_label_file": "labels/water_surface.txt"
        }
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(meta_dict, mf, indent=2)

        # 4. Insert Row into SQLite Database
        db_payload = {
            "bottle_id": decision.bottle_id,
            "timestamp": decision.timestamp,
            "epoch_time": time.time(),
            "global_verdict": decision.verdict.value,
            "fill_status": decision.fill_status,
            "fill_ratio": decision.fill_ratio,
            "water_y": decision.meniscus.water_y if decision.meniscus else None,
            "bottle_h": decision.bottle.height if decision.bottle else None,
            "water_conf": decision.meniscus.confidence if decision.meniscus else None,
            "bottle_conf": decision.bottle.confidence if decision.bottle else None,
            "is_compliant": 1 if decision.is_compliant else 0,
            "trigger_actuator": 1 if decision.trigger_reject_actuator else 0,
            "decision_reason": decision.decision_reason,
            "total_latency_ms": decision.total_latency_ms,
            "bundle_dir": str(bundle_path)
        }
        rowid = self.db.insert_inspection(db_payload)

        record = FillInspectionRecord(
            id=rowid,
            bottle_id=decision.bottle_id,
            timestamp=decision.timestamp,
            epoch_time=db_payload["epoch_time"],
            global_verdict=decision.verdict.value,
            fill_status=decision.fill_status,
            fill_ratio=decision.fill_ratio,
            water_y=db_payload["water_y"],
            bottle_h=db_payload["bottle_h"],
            water_conf=db_payload["water_conf"],
            bottle_conf=db_payload["bottle_conf"],
            is_compliant=decision.is_compliant,
            trigger_actuator=decision.trigger_reject_actuator,
            decision_reason=decision.decision_reason,
            total_latency_ms=decision.total_latency_ms,
            bundle_dir=str(bundle_path)
        )

        log_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"Logged fill inspection {decision.bottle_id} to DB (rowid: {rowid}, {log_ms:.1f}ms)")
        return record

    def get_kpis(self, window_seconds: Optional[float] = None) -> FillKPISummary:
        return self.db.get_kpis(window_seconds)

    def get_ratio_histogram(self, bins: int = 10, window_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.db.get_ratio_histogram(bins, window_seconds)

    def query_inspections(self, filt: FillInspectionFilter) -> List[FillInspectionRecord]:
        return self.db.query_inspections(filt)

    def count_inspections(self, filt: FillInspectionFilter) -> int:
        return self.db.count_inspections(filt)

    def get_bottle(self, bottle_id: str) -> Optional[FillInspectionRecord]:
        return self.db.get_bottle(bottle_id)

    def get_adjacent_bottles(self, bottle_id: str) -> Dict[str, Optional[str]]:
        return self.db.get_adjacent_bottles(bottle_id)

    def update_bottle_annotation(
        self,
        bottle_id: str,
        global_verdict: Optional[str] = None,
        fill_status: Optional[str] = None,
        is_compliant: Optional[bool] = None
    ) -> bool:
        return self.db.update_bottle_annotation(
            bottle_id=bottle_id,
            global_verdict=global_verdict,
            fill_status=fill_status,
            is_compliant=is_compliant
        )

    def export_csv(self, filt: FillInspectionFilter) -> str:
        """Exports matching audit records to an RFC 4180 CSV string."""
        records = self.db.query_inspections(filt)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")

        # Headers
        writer.writerow([
            "id", "bottle_id", "timestamp", "global_verdict", "fill_status",
            "fill_ratio", "water_y", "bottle_h", "water_conf", "bottle_conf",
            "is_compliant", "trigger_actuator", "total_latency_ms", "decision_reason"
        ])

        for r in records:
            writer.writerow([
                r.id,
                r.bottle_id,
                r.timestamp,
                r.global_verdict,
                r.fill_status,
                f"{r.fill_ratio:.3f}" if r.fill_ratio is not None else "",
                r.water_y if r.water_y is not None else "",
                r.bottle_h if r.bottle_h is not None else "",
                f"{r.water_conf:.3f}" if r.water_conf is not None else "",
                f"{r.bottle_conf:.3f}" if r.bottle_conf is not None else "",
                1 if r.is_compliant else 0,
                1 if r.trigger_actuator else 0,
                f"{r.total_latency_ms:.2f}",
                r.decision_reason
            ])

        return output.getvalue()
