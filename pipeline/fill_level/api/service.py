"""
Fill Pipeline Service for Station 2 (Fill Level Inspection).
Singleton coordinator combining:
- Part 1: FillLevelInferenceEngine (On-demand dynamic lazy loading into GPU VRAM)
- Part 2: FillQualityGate (Underfill, overfill, and borderline decision rules)
- Part 3: FillVisualOverlayRenderer (Non-destructive glowing meniscus & ruler rendering)
- Part 4: FillAuditLogger (Dedicated audit/fill_level.db & training-ready raw frames)
"""
import gc
import io
import json
import os
import threading
import zipfile
import base64
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch

from pipeline.env_loader import env_path
from pipeline.fill_level.inference.engine import FillLevelInferenceEngine
from pipeline.fill_level.inference.schemas import BottleBox, MeniscusBox
from pipeline.fill_level.decision.gate import FillQualityGate
from pipeline.fill_level.decision.schemas import FillVerdict, FillInspectionDecision
from pipeline.fill_level.visualizer.renderer import FillVisualOverlayRenderer
from pipeline.fill_level.audit.logger import FillAuditLogger
from pipeline.fill_level.audit.schemas import (
    FillAuditConfig,
    FillKPISummary,
    FillInspectionFilter,
    FillInspectionRecord,
)
from .schemas import (
    FillConfigResponse,
    FillConfigUpdateRequest,
    FillInspectResponse,
    FillActiveLearningCandidate,
    SaveFillAnnotationRequest,
    FillHealthResponse,
)

logger = logging.getLogger("pipeline.fill_level.api.service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FillPipelineService:
    """
    Central orchestration service for Station 2 (Fill Level Inspection).
    Implements on-demand dynamic lazy loading and VRAM deallocation, ensuring
    zero GPU memory overhead when Station 1 is active.
    """

    _instance: Optional["FillPipelineService"] = None

    def __init__(self):
        logger.info("Initializing FillPipelineService components...")
        self.quality_gate = FillQualityGate()
        self.visualizer = FillVisualOverlayRenderer()
        self.audit_logger = FillAuditLogger()
        self._engine: Optional[FillLevelInferenceEngine] = None  # LAZY: NOT LOADED AT BOOT
        # Serializes load/unload: /load, the WebSocket auto-load and inspect calls can race, and a
        # first-time TensorRT compile must never run twice concurrently.
        self._infer_lock = threading.RLock()  # serialize inspections; unload waits for in-flight inference
        self._engine_lock = threading.RLock()
        
        # Simulation images: VISIONQAI_FILL_SIM_DIR (absolute, or relative to repo root); default = bundled samples
        self.sample_images_dir = env_path("VISIONQAI_FILL_SIM_DIR", Path(__file__).resolve().parent.parent / "sample_images")
        # Runtime-adjustable detector confidences (applied to the engine whenever it is loaded)
        self.bottle_conf_threshold: Optional[float] = None
        self.water_conf_threshold: Optional[float] = None
        self._sample_index = 0
        self._sample_images: List[Path] = []
        self._find_sample_images()
        logger.info("FillPipelineService initialized in STANDBY mode (0 MB VRAM used).")

    @classmethod
    def get_instance(cls) -> "FillPipelineService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _find_sample_images(self):
        """Loads available bottle profile sample images for simulation mode."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if self.sample_images_dir.is_dir():
            self._sample_images = sorted(
                p for p in self.sample_images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
            )
        if self._sample_images:
            logger.info(f"Station 2 simulation: {len(self._sample_images)} image(s) from {self.sample_images_dir}")
        else:
            logger.warning(f"No simulation images found in {self.sample_images_dir} (VISIONQAI_FILL_SIM_DIR); using synthetic frame.")

    @property
    def engine(self) -> Optional[FillLevelInferenceEngine]:
        """Returns the active inference engine instance, or None if in standby."""
        return self._engine

    def is_engine_loaded(self) -> bool:
        """Returns True if Fill Level models are currently allocated in VRAM."""
        return self._engine is not None

    def ensure_engine_loaded(self) -> FillLevelInferenceEngine:
        """Dynamically allocates Fill Level models into GPU VRAM on demand."""
        with self._engine_lock:
            if self._engine is None:
                logger.info("Dynamically loading Fill Level models into GPU VRAM on demand...")
                engine = FillLevelInferenceEngine(warmup=True)
                self._apply_config_to_engine(engine)
                self._engine = engine
                logger.info("Fill Level models online and ready in VRAM.")
            return self._engine

    def unload_engine(self) -> bool:
        """Deallocates Fill Level models from VRAM and flushes CUDA memory."""
        with self._infer_lock, self._engine_lock:
            if self._engine is None:
                return False
            logger.info("Unloading Fill Level models from GPU VRAM...")
            self._engine = None
            # Drop reference cycles held by the YOLO/TensorRT wrappers before flushing the CUDA cache,
            # otherwise the allocator cannot return the memory.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Fill Level models unloaded. GPU VRAM freed.")
            return True

    def _apply_config_to_engine(self, engine: FillLevelInferenceEngine) -> None:
        """Pushes the live gate/confidence settings into an engine instance."""
        engine.underfill_threshold = self.quality_gate.config.underfill_threshold
        engine.overfill_threshold = self.quality_gate.config.overfill_threshold
        if self.bottle_conf_threshold is not None:
            engine.bottle_conf_thresh = self.bottle_conf_threshold
        if self.water_conf_threshold is not None:
            engine.water_conf_thresh = self.water_conf_threshold

    def get_config(self) -> FillConfigResponse:
        """Returns the active Station 2 runtime thresholds."""
        gate = self.quality_gate.config
        eng = self._engine
        bottle_conf = self.bottle_conf_threshold
        if bottle_conf is None:
            bottle_conf = eng.bottle_conf_thresh if eng else float(os.getenv("VISIONQAI_FILL_DET_CONF", "0.25"))
        water_conf = self.water_conf_threshold
        if water_conf is None:
            water_conf = eng.water_conf_thresh if eng else float(os.getenv("VISIONQAI_FILL_WATER_CONF", "0.15"))
        return FillConfigResponse(
            underfill_threshold=gate.underfill_threshold,
            overfill_threshold=gate.overfill_threshold,
            borderline_band=gate.borderline_band,
            borderline_conf_floor=gate.borderline_conf_floor,
            bottle_conf_threshold=bottle_conf,
            water_conf_threshold=water_conf,
            hardware_runtime=eng.hardware_desc if eng else "Standby (Unloaded)",
        )

    def update_config(self, req: FillConfigUpdateRequest) -> FillConfigResponse:
        """Validates and applies Station 2 threshold changes live (gate, overlay ruler, engine)."""
        gate = self.quality_gate.config
        under = req.underfill_threshold if req.underfill_threshold is not None else gate.underfill_threshold
        over = req.overfill_threshold if req.overfill_threshold is not None else gate.overfill_threshold
        if under >= over:
            raise ValueError("underfill_threshold must be lower than overfill_threshold.")

        # Validate against the same bounds as FillRulesConfig before mutating anything
        candidate = gate.model_dump()
        candidate["underfill_threshold"] = under
        candidate["overfill_threshold"] = over
        if req.borderline_band is not None:
            candidate["borderline_band"] = req.borderline_band
        if req.borderline_conf_floor is not None:
            candidate["borderline_conf_floor"] = req.borderline_conf_floor
        validated = type(gate)(**candidate)  # raises pydantic.ValidationError when out of range
        for field in ("underfill_threshold", "overfill_threshold", "borderline_band", "borderline_conf_floor"):
            setattr(gate, field, getattr(validated, field))

        if req.bottle_conf_threshold is not None:
            self.bottle_conf_threshold = req.bottle_conf_threshold
        if req.water_conf_threshold is not None:
            self.water_conf_threshold = req.water_conf_threshold

        self.visualizer.config.underfill_thresh = gate.underfill_threshold
        self.visualizer.config.overfill_thresh = gate.overfill_threshold
        if self._engine is not None:
            self._apply_config_to_engine(self._engine)
        logger.info(f"Station 2 config updated: {self.get_config().model_dump()}")
        return self.get_config()

    def get_health(self) -> FillHealthResponse:
        """Reports health and VRAM allocation state."""
        eng = self._engine  # single read: may be unloaded concurrently
        loaded = eng is not None
        hw_desc = eng.hardware_desc if loaded else "Standby (Unloaded)"
        eng_ver = eng.engine_version if loaded else "v1"
        return FillHealthResponse(
            status="ok",
            station="Station 2: Fill Level Inspection",
            models_loaded=loaded,
            hardware_runtime=hw_desc,
            engine_version=eng_ver,
            underfill_threshold=self.quality_gate.config.underfill_threshold,
            overfill_threshold=self.quality_gate.config.overfill_threshold,
        )

    def inspect_frame(
        self,
        frame: np.ndarray,
        bottle_id: Optional[str] = None,
        render_base64: bool = True
    ) -> FillInspectResponse:
        """Thread-safe entry point: one inspection at a time per engine (see _infer_lock)."""
        with self._infer_lock:
            return self._inspect_frame(frame, bottle_id, render_base64)

    def _inspect_frame(
        self,
        frame: np.ndarray,
        bottle_id: Optional[str] = None,
        render_base64: bool = True
    ) -> FillInspectResponse:
        """
        Executes a complete Station 2 inspection cycle:
        Inference -> Quality Gate -> SQLite Audit Log -> Dynamic Visual Overlay.
        """
        engine = self.ensure_engine_loaded()
        b_id = bottle_id or f"FILL_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        # Step 1: Forward Pass on Single Bottle Image
        inf_result = engine.predict_single_frame(frame)

        # Step 2: Quality Gate Threshold Evaluation
        decision: FillInspectionDecision = self.quality_gate.evaluate(
            result=inf_result,
            bottle_id=b_id
        )

        # Step 3: Archive Unburned Raw Frame and Record to SQLite
        audit_record = self.audit_logger.log_inspection(frame=frame, decision=decision)

        # Step 4: Render Dynamic On-Demand Visual Overlay
        annotated_b64 = None
        if render_base64:
            annotated_frame = self.visualizer.render(frame, decision)
            annotated_b64 = self.visualizer.encode_base64(annotated_frame)

        return FillInspectResponse(
            bottle_id=decision.bottle_id,
            timestamp=decision.timestamp,
            verdict=decision.verdict.value,
            is_compliant=decision.is_compliant,
            trigger_reject_actuator=decision.trigger_reject_actuator,
            fill_status=decision.fill_status,
            fill_ratio=decision.fill_ratio,
            decision_reason=decision.decision_reason,
            total_latency_ms=round(decision.total_latency_ms, 2),
            inference_latency_ms=round(inf_result.inference_latency_ms, 2),
            annotated_image_base64=annotated_b64,
            bottle_box=decision.bottle.box if decision.bottle else None,
            bottle_confidence=decision.bottle.confidence if decision.bottle else None,
            meniscus_box=decision.meniscus.box if decision.meniscus else None,
            meniscus_confidence=decision.meniscus.confidence if decision.meniscus else None,
            water_y=decision.meniscus.water_y if decision.meniscus else None,
            bundle_dir=audit_record.bundle_dir
        )

    def run_simulated_cycle(self) -> FillInspectResponse:
        """Executes a simulated conveyor cycle by reading the next bottle sample image."""
        if not self._sample_images:
            # Fallback synthetic profile image
            syn = np.full((640, 640, 3), 40, dtype=np.uint8)
            cv2.rectangle(syn, (150, 80), (490, 580), (200, 200, 200), -1)
            return self.inspect_frame(syn)

        img_path = self._sample_images[self._sample_index % len(self._sample_images)]
        self._sample_index += 1

        img = cv2.imread(str(img_path))
        if img is None:
            syn = np.full((640, 640, 3), 40, dtype=np.uint8)
            return self.inspect_frame(syn)

        return self.inspect_frame(img)

    def get_stats(self, window_seconds: Optional[float] = None) -> FillKPISummary:
        """Returns real-time KPI metrics for Station 2."""
        return self.audit_logger.get_kpis(window_seconds)

    def get_ratio_histogram(self, bins: int = 10, window_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Returns the measured fill-ratio distribution used by the dashboard histogram."""
        return self.audit_logger.get_ratio_histogram(bins, window_seconds)

    def get_defects(
        self,
        limit: int = 50,
        offset: int = 0,
        verdict: Optional[str] = None,
        fill_status: Optional[str] = None,
        search: Optional[str] = None,
        start_epoch: Optional[float] = None,
        end_epoch: Optional[float] = None,
        paginate: bool = False
    ) -> Any:
        """Queries paginated defect inspections for Station 2."""
        filt = FillInspectionFilter(
            limit=limit,
            offset=offset,
            verdict=verdict,
            fill_status=fill_status,
            search=search,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        records = self.audit_logger.query_inspections(filt)
        if paginate:
            total = self.audit_logger.count_inspections(filt)
            return {
                "items": records,
                "total_count": total,
                "limit": limit,
                "offset": offset
            }
        return records

    def _bundle_path(self, bottle_id: str) -> Path:
        """Bundle folder derived from the configured audit dir (DB-stored paths are not portable across PCs)."""
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        return self.audit_logger.bottles_dir / safe_id

    @staticmethod
    def _load_bundle_metadata(bundle_path: Path) -> Dict[str, Any]:
        """Reads inspection_data.json from a bottle bundle; returns {} if missing or unreadable."""
        meta_file = bundle_path / "inspection_data.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    return json.load(mf)
            except Exception as e:
                logger.warning(f"Could not read {meta_file}: {e}")
        return {}

    def get_bottle_detail(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete bottle telemetry, adjacent IDs for S2 Replay stepper,
        and dynamically renders overlay for deep-dive inspection.
        """
        record = self.audit_logger.get_bottle(bottle_id)
        if not record:
            return None

        adjacent = self.audit_logger.get_adjacent_bottles(bottle_id)
        bundle_path = self._bundle_path(record.bottle_id)

        # Load raw frame and dynamically render overlay
        raw_b64 = None
        annotated_b64 = None
        image_size = None
        raw_path = bundle_path / "raw_frame.jpg"
        if raw_path.exists():
            raw_img = cv2.imread(str(raw_path))
            if raw_img is not None:
                image_size = {"width": int(raw_img.shape[1]), "height": int(raw_img.shape[0])}
                # Raw base64
                _, raw_enc = cv2.imencode(".jpg", raw_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                raw_b64 = f"data:image/jpeg;base64,{base64.b64encode(raw_enc).decode('utf-8')}"
                
                # Reconstruct Decision object from the persisted telemetry (real boxes, not placeholders)
                meta = self._load_bundle_metadata(bundle_path)
                b_box = meta.get("bottle_box")
                w_box = meta.get("meniscus_box")
                mock_bottle = (
                    BottleBox(box=[int(v) for v in b_box], confidence=record.bottle_conf or 0.0)
                    if b_box and len(b_box) == 4 else None
                )
                mock_meniscus = (
                    MeniscusBox(
                        box=[int(v) for v in w_box],
                        water_y=int(meta.get("water_y") if meta.get("water_y") is not None else (w_box[1] + w_box[3]) // 2),
                        confidence=record.water_conf or 0.0,
                    )
                    if w_box and len(w_box) == 4 else None
                )

                dec = FillInspectionDecision(
                    bottle_id=record.bottle_id,
                    timestamp=record.timestamp,
                    verdict=FillVerdict(record.global_verdict),
                    is_compliant=record.is_compliant,
                    trigger_reject_actuator=record.trigger_actuator,
                    fill_status=record.fill_status,
                    fill_ratio=record.fill_ratio,
                    decision_reason=record.decision_reason,
                    decision_latency_ms=0.1,
                    total_latency_ms=record.total_latency_ms,
                    bottle=mock_bottle,
                    meniscus=mock_meniscus
                )
                annotated_img = self.visualizer.render(raw_img, dec)
                annotated_b64 = self.visualizer.encode_base64(annotated_img)

        meta_dict = self._load_bundle_metadata(bundle_path)

        return {
            "record": record.dict(),
            "adjacent": adjacent,
            "metadata": meta_dict,
            "image_size": image_size,
            "raw_image_base64": raw_b64,
            "annotated_image_base64": annotated_b64,
        }

    def export_csv(
        self,
        verdict: Optional[str] = None,
        fill_status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 10000
    ) -> str:
        """Exports matching audit records to RFC 4180 CSV string."""
        filt = FillInspectionFilter(
            limit=limit,
            offset=0,
            verdict=verdict,
            fill_status=fill_status,
            search=search
        )
        return self.audit_logger.export_csv(filt)

    def get_active_learning_candidates(
        self,
        uncertain_only: bool = True,
        search: Optional[str] = None,
        limit: int = 50
    ) -> List[FillActiveLearningCandidate]:
        """Queries borderline fill inspections for human-in-the-loop triage in S3."""
        filt = FillInspectionFilter(
            limit=limit,
            offset=0,
            verdict="UNCERTAIN" if uncertain_only else None,
            search=search
        )
        records = self.audit_logger.query_inspections(filt)
        candidates: List[FillActiveLearningCandidate] = []

        for r in records:
            b_dir = self._bundle_path(r.bottle_id)
            label_file = b_dir / "labels" / "water_surface.txt"
            has_labels = label_file.exists() and label_file.stat().st_size > 0
            is_unc = (r.global_verdict == "UNCERTAIN") or (r.fill_status == "borderline")

            candidates.append(
                FillActiveLearningCandidate(
                    bottle_id=r.bottle_id,
                    timestamp=r.timestamp,
                    epoch_time=r.epoch_time,
                    global_verdict=r.global_verdict,
                    fill_status=r.fill_status,
                    fill_ratio=r.fill_ratio,
                    meniscus_confidence=r.water_conf,
                    has_labels=has_labels,
                    is_uncertain=is_unc
                )
            )
        return candidates

    def save_active_learning_annotation(self, req: SaveFillAnnotationRequest) -> Dict[str, Any]:
        """Saves operator-verified ground-truth annotations to YOLO format and updates SQLite."""
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in req.bottle_id)
        bundle_path = self.audit_logger.bottles_dir / safe_id
        bundle_path.mkdir(parents=True, exist_ok=True)

        labels_dir = bundle_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Update YOLO label file if water_box provided
        if req.water_box and len(req.water_box) == 4:
            raw_path = bundle_path / "raw_frame.jpg"
            if raw_path.exists():
                img = cv2.imread(str(raw_path))
                if img is not None:
                    h, w = img.shape[:2]
                    wx1, wy1, wx2, wy2 = req.water_box
                    xc = max(0.0, min(1.0, ((wx1 + wx2) / 2.0) / w))
                    yc = max(0.0, min(1.0, ((wy1 + wy2) / 2.0) / h))
                    bw = max(0.0, min(1.0, (wx2 - wx1) / w))
                    bh = max(0.0, min(1.0, (wy2 - wy1) / h))
                    with open(labels_dir / "water_surface.txt", "w", encoding="utf-8") as lf:
                        lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Update SQLite database
        is_comp = (req.global_verdict.upper() == "PASS") if req.global_verdict else None
        self.audit_logger.update_bottle_annotation(
            bottle_id=req.bottle_id,
            global_verdict=req.global_verdict,
            fill_status=req.fill_status,
            is_compliant=is_comp
        )

        return {
            "status": "success",
            "bottle_id": req.bottle_id,
            "message": "Fill level ground-truth annotations committed successfully."
        }

    def export_dataset_zip(self) -> bytes:
        """Packs verified fill level bottles into a standardized YOLOv11 dataset zip archive."""
        buffer = io.BytesIO()
        yaml_content = (
            "path: fill_dataset\n"
            "train: images/train\n"
            "val: images/train\n"
            "names:\n"
            "  0: water_surface\n"
        )
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("fill_dataset/data.yaml", yaml_content)
            for b_folder in self.audit_logger.bottles_dir.iterdir():
                if not b_folder.is_dir():
                    continue
                label_p = b_folder / "labels" / "water_surface.txt"
                raw_p = b_folder / "raw_frame.jpg"
                if label_p.exists() and raw_p.exists() and label_p.stat().st_size > 0:
                    b_id = b_folder.name
                    zf.write(str(raw_p), arcname=f"fill_dataset/images/train/{b_id}.jpg")
                    zf.write(str(label_p), arcname=f"fill_dataset/labels/train/{b_id}.txt")
        buffer.seek(0)
        return buffer.getvalue()

    def shutdown(self) -> None:
        """Cleanly releases VRAM and closes audit database connections."""
        logger.info("Shutting down FillPipelineService...")
        self.unload_engine()
        if hasattr(self.audit_logger, "close"):
            self.audit_logger.close()
        logger.info("FillPipelineService shutdown complete.")
