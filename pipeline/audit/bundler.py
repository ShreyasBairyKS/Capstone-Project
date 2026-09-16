"""
Bottle Bundler for Part 5: Logging, Defect Audit & Telemetry.
Handles isolated per-bottle directory creation, unannotated raw frame storage,
YOLOv11 segmentation polygon label exports, and inspection metadata persistence.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import cv2
import numpy as np

from pipeline.audit.schemas import AuditConfig
from pipeline.decision.schemas import GlobalInspectionResult, InspectionVerdict

logger = logging.getLogger("pipeline.audit.bundler")


class BottleBundler:
    """
    Manages filesystem archiving of inspection sessions.
    Strictly guarantees bottle-level isolation with pristine raw frames
    and auto-generated YOLO active learning annotations.
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.base_dir = Path(self.config.base_dir)
        self.bottles_root = self.base_dir / self.config.bottles_subfolder
        self.bottles_root.mkdir(parents=True, exist_ok=True)

    def bundle_bottle(
        self,
        bottle_id: str,
        frames: Dict[str, np.ndarray],
        decision: GlobalInspectionResult
    ) -> Optional[str]:
        """
        Creates an isolated bundle directory for a bottle inspection event.

        Args:
            bottle_id: Unique bottle inspection identifier.
            frames: Dictionary of camera_id -> raw numpy BGR images.
            decision: Authoritative multi-view inspection result.

        Returns:
            Relative path string to the bundle folder (e.g. 'audit/bottles/BOTTLE_xxx'),
            or None if saving was skipped due to config policy.
        """
        # Enforce retention policy: Skip compliant frames if configured to save only defects
        if decision.global_verdict == InspectionVerdict.PASS and not self.config.save_pass_images:
            return None

        # Sanitize bottle_id for safe filesystem path
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        bundle_path = self.bottles_root / safe_id
        bundle_path.mkdir(parents=True, exist_ok=True)

        labels_dir = bundle_path / "labels"
        if self.config.save_yolo_labels:
            labels_dir.mkdir(parents=True, exist_ok=True)

        image_manifest = {}
        label_manifest = {}

        # 1. Archive Raw Frames (Clean, Pristine, Completely Unannotated)
        for cam_id, frame in frames.items():
            if frame is None or frame.size == 0:
                continue

            img_filename = f"{cam_id}.jpg"
            img_dest = bundle_path / img_filename
            cv2.imwrite(
                str(img_dest),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            )
            image_manifest[cam_id] = img_filename

            # 2. Export Active Learning YOLO Segmentation Polygon Labels
            if self.config.save_yolo_labels and cam_id in decision.per_view_results:
                view_result = decision.per_view_results[cam_id]
                h, w = frame.shape[:2]
                label_lines = []

                for det in view_result.detections:
                    if det.mask and len(det.mask) >= 3:
                        # Normalize polygon coordinates to [0.0, 1.0]
                        coords = []
                        for pt in det.mask:
                            x_norm = max(0.0, min(1.0, pt[0] / float(w)))
                            y_norm = max(0.0, min(1.0, pt[1] / float(h)))
                            coords.append(f"{x_norm:.6f} {y_norm:.6f}")
                        label_lines.append(f"{det.class_id} " + " ".join(coords))
                    else:
                        # Bounding box 4-corner polygon fallback
                        x1, y1, x2, y2 = det.box
                        box_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        coords = []
                        for pt in box_poly:
                            x_norm = max(0.0, min(1.0, pt[0] / float(w)))
                            y_norm = max(0.0, min(1.0, pt[1] / float(h)))
                            coords.append(f"{x_norm:.6f} {y_norm:.6f}")
                        label_lines.append(f"{det.class_id} " + " ".join(coords))

                label_filename = f"{cam_id}.txt"
                label_dest = labels_dir / label_filename
                with open(label_dest, "w", encoding="utf-8") as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                label_manifest[cam_id] = f"labels/{label_filename}"

        # 3. Write Comprehensive Inspection Metadata JSON
        inspection_data = decision.model_dump(mode="json")
        inspection_data["bundle_manifest"] = {
            "images": image_manifest,
            "labels": label_manifest,
            "isolated_bundle_dir": str(bundle_path)
        }

        meta_dest = bundle_path / "inspection_data.json"
        with open(meta_dest, "w", encoding="utf-8") as f:
            json.dump(inspection_data, f, indent=2)

        rel_bundle_dir = str(bundle_path.relative_to(Path(".").resolve()) if bundle_path.is_absolute() and bundle_path.is_relative_to(Path(".").resolve()) else bundle_path)
        logger.debug(f"Archived isolated bottle bundle: {rel_bundle_dir}")
        return str(bundle_path).replace("\\", "/")
