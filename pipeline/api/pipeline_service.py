"""
Pipeline Service for Part 6: Backend Service & API Layer.
Singleton service coordinator combining:
- Part 1: InferenceEngine (YOLOv11m-seg with isolated NMS)
- Part 2: QualityGate (Asymmetric dual-threshold decision rules)
- Part 3: StreamManager (Multi-camera frame bundle ingestion)
- Part 4: VisualOverlayRenderer (Semi-transparent mask overlays & dynamic replay)
- Part 5: AuditLogger (Isolated bottle bundling, YOLO labels & SQLite telemetry)
"""
import base64
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from pipeline.inference.engine import InferenceEngine
from pipeline.inference.schemas import MultiCameraInferenceResult
from pipeline.decision.gate import QualityGate
from pipeline.decision.schemas import GlobalInspectionResult, InspectionVerdict
from pipeline.ingestion.stream_manager import StreamManager
from pipeline.visualizer.renderer import VisualOverlayRenderer
from pipeline.visualizer.schemas import VisualizerConfig
from pipeline.audit.logger import AuditLogger
from pipeline.audit.schemas import AuditConfig, KPISummary, InspectionFilter
from pipeline.api.schemas import (
    InspectResponse,
    ViewInspectSummary,
    ConfigUpdateRequest,
    ConfigResponse,
    DefectItem
)

logger = logging.getLogger("pipeline.api.service")


class PipelineService:
    """
    Central orchestration service for the inspection pipeline.
    Maintains thread-safe instances of model inference, quality decision logic,
    real-time overlay rendering, and telemetry storage.
    """

    _instance: Optional["PipelineService"] = None

    def __init__(self):
        logger.info("Initializing PipelineService engines...")
        self.inference_engine = InferenceEngine()
        self.quality_gate = QualityGate()
        self.stream_manager = StreamManager()
        self.visualizer = VisualOverlayRenderer(VisualizerConfig(mask_alpha=0.40))
        self.audit_logger = AuditLogger(AuditConfig(async_saving=False))  # Direct saving for predictable demo
        self.sample_images: List[Path] = []
        self._find_sample_images()
        logger.info("PipelineService fully initialized.")

    @classmethod
    def get_instance(cls) -> "PipelineService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _find_sample_images(self) -> None:
        """Finds project sample images for offline simulation mode."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        p1 = base_dir / "runs" / "segment" / "predict" / "IMG20260824154700.jpg"
        p2 = base_dir / "runs" / "segment" / "predict2" / "IMG20260824154411.jpg"
        for p in [p1, p2]:
            if p.exists():
                self.sample_images.append(p)

    def inspect_frames(
        self,
        frames: Dict[str, np.ndarray],
        bottle_id: Optional[str] = None,
        render_base64: bool = True
    ) -> InspectResponse:
        """
        Executes a full inspection cycle across multi-camera views:
        Ingestion -> Inference -> Quality Gate -> Audit Logging -> Visualizer.
        """
        b_id = bottle_id or f"BOTTLE_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        # Step 1: Batched Forward Pass
        inf_result: MultiCameraInferenceResult = self.inference_engine.predict_multi_camera(frames)

        # Step 2: Quality Gate Multi-View Evaluation
        decision: GlobalInspectionResult = self.quality_gate.evaluate_multi_camera(
            multi_result=inf_result,
            bottle_id=b_id
        )

        # Step 3: Archive to Isolated Bottle Directory & SQLite Database
        audit_record = self.audit_logger.log_inspection(
            frames=frames,
            decision=decision,
            wait_sync=True
        )

        # Step 4: Render Visual Overlays & Convert to View Summaries
        view_summaries: Dict[str, ViewInspectSummary] = {}
        for cam_id, frame in frames.items():
            view_verdict = decision.per_view_results.get(cam_id)
            ann_frame = self.visualizer.render_view(
                frame=frame,
                view_data=view_verdict if view_verdict else {},
                global_result=decision,
                camera_id=cam_id
            )

            b64_str = None
            if render_base64:
                # Downscale large 12MP frames for fast web transmission
                disp_frame = ann_frame
                if disp_frame.shape[1] > 1280:
                    scale = 1280.0 / disp_frame.shape[1]
                    disp_frame = cv2.resize(disp_frame, (1280, int(disp_frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                success, encoded = cv2.imencode(".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    b64_str = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('utf-8')}"

            v_str = view_verdict.verdict.value if view_verdict and hasattr(view_verdict.verdict, "value") else "PASS"
            view_summaries[cam_id] = ViewInspectSummary(
                camera_id=cam_id,
                verdict=v_str,
                primary_class=view_verdict.primary_class if view_verdict else None,
                primary_confidence=view_verdict.primary_confidence if view_verdict else None,
                defect_detected=view_verdict.defect_detected if view_verdict else False,
                defect_name=view_verdict.defect_name if view_verdict else None,
                defect_confidence=view_verdict.defect_confidence if view_verdict else None,
                decision_reason=view_verdict.decision_reason if view_verdict else "No detections",
                detection_count=len(view_verdict.detections) if view_verdict else 0,
                annotated_image_base64=b64_str
            )

        g_verdict_str = decision.global_verdict.value if hasattr(decision.global_verdict, "value") else str(decision.global_verdict)

        return InspectResponse(
            bottle_id=decision.bottle_id,
            timestamp=decision.timestamp,
            global_verdict=g_verdict_str,
            is_compliant=decision.is_compliant,
            trigger_reject_actuator=decision.trigger_reject_actuator,
            primary_defect=decision.primary_defect,
            primary_defect_confidence=decision.primary_defect_confidence,
            flagged_cameras=decision.flagged_cameras,
            decision_reason=decision.decision_reason,
            total_latency_ms=round(decision.total_latency_ms, 2),
            camera_count=decision.camera_count,
            bundle_dir=audit_record.bundle_dir,
            views=view_summaries
        )

    def get_stats(self, window_seconds: Optional[float] = None) -> KPISummary:
        """Returns real-time KPI metrics."""
        return self.audit_logger.get_kpis(window_seconds)

    def get_defects(
        self,
        limit: int = 50,
        offset: int = 0,
        verdict: Optional[str] = None,
        defect_type: Optional[str] = None
    ) -> List[DefectItem]:
        """Returns paginated list of defect inspections."""
        filt = InspectionFilter(
            limit=limit,
            offset=offset,
            verdict=verdict or "REJECT",
            defect_type=defect_type
        )
        records = self.audit_logger.query_inspections(filt)
        return [
            DefectItem(
                id=r.id,
                bottle_id=r.bottle_id,
                timestamp=r.timestamp,
                epoch_time=r.epoch_time,
                global_verdict=r.global_verdict,
                is_compliant=r.is_compliant,
                trigger_actuator=r.trigger_actuator,
                primary_defect=r.primary_defect,
                primary_defect_confidence=r.primary_defect_confidence,
                flagged_cameras=r.flagged_cameras,
                total_latency_ms=r.total_latency_ms,
                bundle_dir=r.bundle_dir
            )
            for r in records
        ]

    def get_bottle_detail(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete bottle DB record and JSON manifest."""
        rec = self.audit_logger.get_bottle_record(bottle_id)
        if not rec:
            return None
        manifest = self.audit_logger.get_bottle_manifest(bottle_id)
        return {
            "record": rec.model_dump(),
            "manifest": manifest
        }

    def get_camera_image_bytes(
        self,
        bottle_id: str,
        camera_id: str,
        render: bool = False
    ) -> Optional[Tuple[bytes, str]]:
        """
        Retrieves image bytes for a camera view of a bottle.
        If render=False, serves pristine raw JPEG.
        If render=True, dynamically renders on-demand overlays.
        """
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        bundle_path = Path(self.audit_logger.config.base_dir) / self.audit_logger.config.bottles_subfolder / safe_id
        if not bundle_path.exists():
            return None

        if not render:
            img_path = bundle_path / f"{camera_id}.jpg"
            if not img_path.exists():
                return None
            with open(img_path, "rb") as f:
                return f.read(), "image/jpeg"
        else:
            # Dynamic On-Demand Replay Rendering
            try:
                ann_frame = self.visualizer.render_from_bundle(bundle_path, camera_id=camera_id)
                success, encoded = cv2.imencode(".jpg", ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if success:
                    return encoded.tobytes(), "image/jpeg"
            except Exception as e:
                logger.error(f"Error dynamically rendering replay for {bottle_id} ({camera_id}): {e}")
                return None
        return None

    def update_config(self, req: ConfigUpdateRequest) -> ConfigResponse:
        """Applies dynamic runtime updates to thresholds without restart."""
        if req.pass_conf_threshold is not None:
            self.quality_gate.config.pass_conf_threshold = req.pass_conf_threshold
        if req.defect_conf_threshold is not None:
            self.quality_gate.config.defect_conf_threshold = req.defect_conf_threshold
        if req.uncertain_floor is not None:
            self.quality_gate.config.uncertain_floor = req.uncertain_floor
        if req.save_pass_images is not None:
            self.audit_logger.config.save_pass_images = req.save_pass_images
        if req.mask_alpha is not None:
            self.visualizer.config.mask_alpha = req.mask_alpha

        return self.get_config()

    def get_config(self) -> ConfigResponse:
        """Returns active runtime configuration."""
        return ConfigResponse(
            pass_conf_threshold=self.quality_gate.config.pass_conf_threshold,
            defect_conf_threshold=self.quality_gate.config.defect_conf_threshold,
            uncertain_floor=self.quality_gate.config.uncertain_floor,
            save_pass_images=self.audit_logger.config.save_pass_images,
            save_defect_images=self.audit_logger.config.save_defect_images,
            save_yolo_labels=self.audit_logger.config.save_yolo_labels,
            mask_alpha=self.visualizer.config.mask_alpha,
            hardware_runtime=self.inference_engine.hardware_desc
        )

    def run_simulated_cycle(self) -> InspectResponse:
        """Executes a simulated inspection cycle using available project images."""
        frames = {}
        if len(self.sample_images) >= 2:
            f1 = cv2.imread(str(self.sample_images[0]))
            f2 = cv2.imread(str(self.sample_images[1]))
            if f1 is not None and f2 is not None:
                frames["camera_front"] = f1
                frames["camera_rear"] = f2
        elif len(self.sample_images) == 1:
            f1 = cv2.imread(str(self.sample_images[0]))
            if f1 is not None:
                frames["camera_front"] = f1
                frames["camera_rear"] = f1.copy()

        if not frames:
            # Synthetic fallback
            frames["camera_front"] = np.full((640, 640, 3), (120, 120, 120), dtype=np.uint8)
            frames["camera_rear"] = np.full((640, 640, 3), (120, 120, 120), dtype=np.uint8)

        return self.inspect_frames(frames=frames)

    def shutdown(self) -> None:
        """Shuts down background workers cleanly."""
        self.audit_logger.shutdown()
        self.stream_manager.shutdown_all()
        logger.info("PipelineService cleanly shut down.")
