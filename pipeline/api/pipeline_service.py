"""
Pipeline Service for Part 6: Backend Service & API Layer.
Singleton service coordinator combining:
- Part 1: InferenceEngine (YOLOv11m-seg with isolated NMS)
- Part 2: QualityGate (Asymmetric dual-threshold decision rules)
- Part 3: StreamManager (Multi-camera frame bundle ingestion)
- Part 4: VisualOverlayRenderer (Semi-transparent mask overlays & dynamic replay)
- Part 5: AuditLogger (Isolated bottle bundling, YOLO labels & SQLite telemetry)
"""
import os
import io
import json
import zipfile
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
    DefectItem,
    AnnotationPolygonItem,
    ActiveLearningCameraView,
    ActiveLearningCandidateItem,
    ActiveLearningCandidatesResponse,
    ActiveLearningBottleDetail,
    SaveAnnotationRequest
)

logger = logging.getLogger("pipeline.api.service")

CLASS_NAMES: Dict[int, str] = {
    0: "damaged_cap",
    1: "good_cap",
    2: "misplaced_cap",
    3: "no_cap",
    4: "open_cap",
    5: "wet_cap"
}


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
        self._sim_index: int = 0
        self._find_sample_images()
        logger.info("PipelineService fully initialized.")

    @classmethod
    def get_instance(cls) -> "PipelineService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _find_sample_images(self) -> None:
        """Finds project sample images for offline simulation mode from configured dataset directory."""
        self.sample_images.clear()
        base_dir = Path(__file__).resolve().parent.parent.parent

        # Search candidates in priority order
        search_dirs: List[Path] = []
        for env_var in ("VISIONQAI_DATASET_DIR", "VISIONQAI_TEST_DATA_DIR"):
            val = os.getenv(env_var)
            if val:
                search_dirs.append(Path(val.strip("\"'")))

        # Configured dataset path from user specification
        search_dirs.append(Path(r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11"))
        search_dirs.append(Path("/app/dataset"))
        search_dirs.append(base_dir / "test")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        discovered: List[Path] = []

        for root_p in search_dirs:
            if not root_p.exists():
                continue
            # Check candidate subdirectories
            sub_candidates = [
                root_p / "test" / "images",
                root_p / "valid" / "images",
                root_p / "train" / "images",
                root_p / "images",
                root_p / "test",
                root_p,
            ]
            for sdir in sub_candidates:
                if sdir.exists() and sdir.is_dir():
                    img_files = sorted([f for f in sdir.iterdir() if f.suffix.lower() in valid_exts])
                    if img_files:
                        discovered.extend(img_files)
                        break
            if discovered:
                break

        # Check local runs fallback if no dataset found
        if not discovered:
            for fallback_dir in [base_dir / "runs" / "segment" / "predict", base_dir / "runs" / "segment" / "predict2"]:
                if fallback_dir.exists():
                    discovered.extend(sorted([f for f in fallback_dir.iterdir() if f.suffix.lower() in valid_exts]))

        # De-duplicate while preserving order
        seen = set()
        for p in discovered:
            if p not in seen and p.exists():
                seen.add(p)
                self.sample_images.append(p)

        logger.info(f"Discovered {len(self.sample_images)} dataset images for simulation.")

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
        defect_type: Optional[str] = None,
        search: Optional[str] = None,
        start_epoch: Optional[float] = None,
        end_epoch: Optional[float] = None,
        paginate: bool = False
    ) -> Any:
        """Returns paginated list or structured response of defect inspections."""
        filt = InspectionFilter(
            limit=limit,
            offset=offset,
            verdict=verdict,
            defect_type=defect_type,
            search=search,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        records = self.audit_logger.query_inspections(filt)
        items = [
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

        if paginate:
            total_count = self.audit_logger.count_inspections(filt)
            return {
                "items": items,
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }
        return items

    def export_audit_csv(
        self,
        verdict: Optional[str] = None,
        defect_type: Optional[str] = None,
        search: Optional[str] = None,
        start_epoch: Optional[float] = None,
        end_epoch: Optional[float] = None,
        limit: int = 10000
    ) -> str:
        """Exports matching audit records as RFC 4180 CSV string."""
        filt = InspectionFilter(
            limit=limit,
            offset=0,
            verdict=verdict,
            defect_type=defect_type,
            search=search,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        return self.audit_logger.export_csv(filt)

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

    def get_latest_bottle_detail(self) -> Optional[Dict[str, Any]]:
        """Retrieves complete bottle DB record and JSON manifest for the most recent inspection."""
        rec = self.audit_logger.get_latest_bottle()
        if not rec:
            return None
        manifest = self.audit_logger.get_bottle_manifest(rec.bottle_id)
        return {
            "record": rec.model_dump(),
            "manifest": manifest
        }

    def get_adjacent_bottles(self, bottle_id: str, window: int = 5) -> Dict[str, Any]:
        """Retrieves chronological sequence of bottles immediately preceding and following a target bottle."""
        return self.audit_logger.get_adjacent_bottles(bottle_id, window=window)

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
        """Executes a simulated inspection cycle across 3 synchronized side camera angles."""
        if not self.sample_images:
            self._find_sample_images()

        frames = {}
        loaded = []
        chosen_names = []

        if self.sample_images:
            total = len(self.sample_images)
            for offset in range(min(3, total)):
                idx = (self._sim_index + offset) % total
                img_p = self.sample_images[idx]
                img = cv2.imread(str(img_p))
                if img is not None:
                    loaded.append(img)
                    chosen_names.append(img_p.name)
            self._sim_index = (self._sim_index + 3) % total

        if len(loaded) >= 3:
            frames["camera_front"] = loaded[0]
            frames["camera_rear"] = loaded[1]
            frames["camera_side"] = loaded[2]
        elif len(loaded) == 2:
            frames["camera_front"] = loaded[0]
            frames["camera_rear"] = loaded[1]
            frames["camera_side"] = loaded[0].copy()
        elif len(loaded) == 1:
            frames["camera_front"] = loaded[0]
            frames["camera_rear"] = loaded[0].copy()
            frames["camera_side"] = loaded[0].copy()
        else:
            # Synthetic fallback with minor intensity variation
            frames["camera_front"] = np.full((640, 640, 3), (120, 120, 120), dtype=np.uint8)
            frames["camera_rear"] = np.full((640, 640, 3), (115, 115, 115), dtype=np.uint8)
            frames["camera_side"] = np.full((640, 640, 3), (125, 125, 125), dtype=np.uint8)

        if chosen_names:
            logger.info(f"Simulated cycle fed with images: {chosen_names}")

        return self.inspect_frames(frames=frames)

    def get_active_learning_candidates(
        self,
        uncertain_only: bool = True,
        search: Optional[str] = None,
        limit: int = 50
    ) -> ActiveLearningCandidatesResponse:
        """
        Retrieves candidate bottle inspections for active learning verification.
        Default is uncertain/borderline cases, with option to search/re-evaluate any bottle.
        """
        rows = self.audit_logger.query_active_learning_candidates(
            uncertain_only=uncertain_only,
            search=search,
            limit=limit
        )
        uncertain_total = self.audit_logger.count_uncertain()

        items: List[ActiveLearningCandidateItem] = []
        base_dir = Path(self.audit_logger.config.base_dir) / self.audit_logger.config.bottles_subfolder

        for r in rows:
            safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in r["bottle_id"])
            bundle_dir = base_dir / safe_id
            labels_dir = bundle_dir / "labels"
            has_labels = labels_dir.exists() and any(labels_dir.glob("*.txt"))

            conf = r["primary_defect_confidence"]
            verdict = r["global_verdict"]
            is_unc = (verdict == "UNCERTAIN") or (conf is not None and 0.20 <= conf <= 0.80)

            items.append(
                ActiveLearningCandidateItem(
                    bottle_id=r["bottle_id"],
                    timestamp=r["timestamp"],
                    epoch_time=r["epoch_time"],
                    global_verdict=verdict,
                    primary_defect=r["primary_defect"],
                    primary_defect_confidence=conf,
                    camera_count=r["camera_count"] or 1,
                    has_labels=has_labels,
                    is_uncertain=is_unc
                )
            )

        return ActiveLearningCandidatesResponse(
            items=items,
            total_count=len(items),
            uncertain_count=uncertain_total
        )

    def get_active_learning_bottle(self, bottle_id: str) -> Optional[ActiveLearningBottleDetail]:
        """
        Fetches detailed bottle views and existing polygon annotations for active learning editor.
        """
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        bundle_path = Path(self.audit_logger.config.base_dir) / self.audit_logger.config.bottles_subfolder / safe_id
        if not bundle_path.exists():
            return None

        rec = self.audit_logger.get_bottle_record(bottle_id)
        if not rec:
            return None

        # Find camera images in bundle directory
        cameras: Dict[str, ActiveLearningCameraView] = {}
        labels_dir = bundle_path / "labels"

        image_files = sorted(list(bundle_path.glob("camera_*.jpg")) or list(bundle_path.glob("*.jpg")))
        for img_file in image_files:
            cam_id = img_file.stem
            w, h = 640, 640
            try:
                img_mat = cv2.imread(str(img_file))
                if img_mat is not None:
                    h, w = img_mat.shape[:2]
            except Exception:
                pass

            annotations: List[AnnotationPolygonItem] = []
            label_file = labels_dir / f"{cam_id}.txt"
            if label_file.exists():
                try:
                    with open(label_file, "r", encoding="utf-8") as lf:
                        for line_idx, line in enumerate(lf):
                            line_str = line.strip()
                            if not line_str:
                                continue
                            parts = line_str.split()
                            if len(parts) >= 5:
                                cid = int(parts[0])
                                coords = [float(p) for p in parts[1:]]
                                pts = []
                                for pi in range(0, len(coords) - 1, 2):
                                    pts.append([round(coords[pi], 6), round(coords[pi + 1], 6)])
                                if len(pts) >= 3:
                                    annotations.append(
                                        AnnotationPolygonItem(
                                            id=f"{cam_id}_poly_{line_idx}",
                                            class_id=cid,
                                            class_name=CLASS_NAMES.get(cid, f"class_{cid}"),
                                            points=pts
                                        )
                                    )
                except Exception as e:
                    logger.error(f"Error reading labels for {bottle_id} ({cam_id}): {e}")

            cameras[cam_id] = ActiveLearningCameraView(
                camera_id=cam_id,
                image_url=f"/api/bottles/{bottle_id}/image/{cam_id}?render=false",
                width=w,
                height=h,
                annotations=annotations
            )

        default_cam = "camera_front" if "camera_front" in cameras else (list(cameras.keys())[0] if cameras else "")

        return ActiveLearningBottleDetail(
            bottle_id=bottle_id,
            timestamp=rec.timestamp,
            global_verdict=rec.global_verdict,
            primary_defect=rec.primary_defect,
            primary_defect_confidence=rec.primary_defect_confidence,
            cameras=cameras,
            default_camera=default_cam
        )

    def save_active_learning_annotation(
        self,
        bottle_id: str,
        req: SaveAnnotationRequest
    ) -> Dict[str, Any]:
        """
        Commits verified ground-truth annotations to disk YOLO format and updates SQLite.
        """
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        bundle_path = Path(self.audit_logger.config.base_dir) / self.audit_logger.config.bottles_subfolder / safe_id
        if not bundle_path.exists():
            bundle_path.mkdir(parents=True, exist_ok=True)

        labels_dir = bundle_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for ann in req.annotations:
            if not ann.points or len(ann.points) < 3:
                continue
            coord_strs = []
            for pt in ann.points:
                x_norm = max(0.0, min(1.0, float(pt[0])))
                y_norm = max(0.0, min(1.0, float(pt[1])))
                coord_strs.append(f"{x_norm:.6f} {y_norm:.6f}")
            lines.append(f"{ann.class_id} " + " ".join(coord_strs))

        dest_txt = labels_dir / f"{req.camera_id}.txt"
        temp_txt = labels_dir / f"{req.camera_id}.txt.tmp"
        with open(temp_txt, "w", encoding="utf-8") as tf:
            tf.write("\n".join(lines) + ("\n" if lines else ""))
        temp_txt.replace(dest_txt)

        is_compliant = None
        if req.global_verdict is not None:
            is_compliant = (req.global_verdict.upper() == "PASS")
        self.audit_logger.update_bottle_annotation(
            bottle_id=bottle_id,
            global_verdict=req.global_verdict,
            primary_defect=req.primary_defect,
            is_compliant=is_compliant
        )

        json_path = bundle_path / "inspection_data.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if req.global_verdict:
                    data["global_verdict"] = req.global_verdict
                if req.primary_defect is not None:
                    data["primary_defect"] = req.primary_defect
                data["active_learning_verified"] = True
                data["active_learning_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(data, jf, indent=2)
            except Exception as e:
                logger.error(f"Error updating inspection_data.json for {bottle_id}: {e}")

        return {
            "status": "success",
            "bottle_id": bottle_id,
            "camera_id": req.camera_id,
            "annotation_count": len(req.annotations),
            "message": "Ground-truth annotations committed successfully."
        }

    def export_dataset_zip(self) -> bytes:
        """
        Packs all annotated bottles into a standardized YOLOv11 dataset zip archive.
        """
        buffer = io.BytesIO()
        base_dir = Path(self.audit_logger.config.base_dir) / self.audit_logger.config.bottles_subfolder

        yaml_content = (
            "path: dataset\n"
            "train: images/train\n"
            "val: images/train\n"
            "names:\n"
            "  0: damaged_cap\n"
            "  1: good_cap\n"
            "  2: misplaced_cap\n"
            "  3: no_cap\n"
            "  4: open_cap\n"
            "  5: wet_cap\n"
        )

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dataset/data.yaml", yaml_content)

            if base_dir.exists():
                for b_folder in base_dir.iterdir():
                    if not b_folder.is_dir():
                        continue
                    labels_dir = b_folder / "labels"
                    if not labels_dir.exists():
                        continue

                    safe_b_id = b_folder.name
                    for label_path in labels_dir.glob("*.txt"):
                        cam_id = label_path.stem
                        img_path = b_folder / f"{cam_id}.jpg"
                        if img_path.exists() and label_path.stat().st_size > 0:
                            zf.write(
                                str(img_path),
                                arcname=f"dataset/images/train/{safe_b_id}_{cam_id}.jpg"
                            )
                            zf.write(
                                str(label_path),
                                arcname=f"dataset/labels/train/{safe_b_id}_{cam_id}.txt"
                            )

        buffer.seek(0)
        return buffer.getvalue()

    def shutdown(self) -> None:
        """Shuts down background workers cleanly."""
        self.audit_logger.shutdown()
        self.stream_manager.shutdown_all()
        logger.info("PipelineService cleanly shut down.")
