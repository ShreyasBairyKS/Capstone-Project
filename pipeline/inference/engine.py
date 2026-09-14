"""
Core Inference Engine for Bottle Cap Defect Detection
Handles model loading, hardware auto-detection (NVIDIA GPU with CPU fallback),
warm-up, batched multi-camera forward passes, and isolated per-image NMS.
"""
import os
import time
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from .schemas import Detection, CameraResult, MultiCameraInferenceResult

logger = logging.getLogger("pipeline.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class InferenceEngine:
    """
    Production-grade inference engine with:
    1. Hardware auto-detection: Prioritizes NVIDIA CUDA GPU with automatic multi-threaded CPU fallback.
    2. Model format hierarchy: Checks for TensorRT (.engine) -> ONNX (.onnx) -> PyTorch (.pt).
    3. Warm-up routine: Eliminates initial latency spikes by pre-allocating GPU/memory contexts.
    4. Batched multi-camera inference: Stacks multi-view frames into a single forward pass.
    5. Strictly isolated per-image NMS: Prevents cross-camera coordinate contamination.
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "YOLOv11seg-X model"
        / "YOLO_TrainingScripts"
        / "student_yolo11m_distilled"
        / "weights"
        / "best.pt"
    )

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        conf_floor: float = 0.35,
        imgsz: int = 640,
        warmup: bool = True,
    ):
        self.conf_floor = conf_floor
        self.imgsz = imgsz

        # 1. Resolve model path
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Resolved model checkpoint: {self.model_path}")

        # 2. Configure hardware & fallback
        self.device, self.hardware_desc = self._select_device(device)
        logger.info(f"Active hardware runtime: {self.hardware_desc} (target: {self.device})")

        # 3. Load model
        t0 = time.perf_counter()
        self.model = YOLO(str(self.model_path), task="segment")
        load_time = (time.perf_counter() - t0) * 1000
        logger.info(f"Model loaded in {load_time:.1f} ms | Classes ({len(self.model.names)}): {self.model.names}")

        # 4. Warm-up
        if warmup:
            self._warmup()

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """
        Auto-resolves and compiles the best model runtime:
        1. If CUDA GPU available: Automatically compiles best.pt -> best.engine (TensorRT FP16)
           if not already present on this machine/container, then loads .engine.
        2. If CPU: Automatically exports best.pt -> best.onnx if not present, then loads .onnx.
        3. Gracefully falls back to best.pt if compilation is unsupported.
        """
        if model_path is not None:
            p = Path(model_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Specified model path does not exist: {model_path}")

        base_dir = self.DEFAULT_MODEL_PATH.parent
        engine_candidate = base_dir / "best.engine"
        onnx_candidate = base_dir / "best.onnx"

        # ---------------------------------------------------------------------
        # 1. NVIDIA GPU / TensorRT Auto-Compilation Strategy
        # ---------------------------------------------------------------------
        if torch.cuda.is_available():
            if engine_candidate.exists():
                logger.info(f"Found existing compiled TensorRT engine: {engine_candidate}")
                return engine_candidate
            elif self.DEFAULT_MODEL_PATH.exists():
                logger.info(
                    "NVIDIA GPU detected, but no pre-compiled TensorRT engine found for this machine. "
                    "Auto-compiling best.pt -> best.engine (FP16) on-the-fly for this hardware..."
                )
                try:
                    from pipeline.inference.export import export_model
                    compiled_engine = export_model(
                        checkpoint_path=self.DEFAULT_MODEL_PATH,
                        export_format="engine",
                        half=True,
                        dynamic=True
                    )
                    logger.info(f"Auto-compilation complete! Generated: {compiled_engine}")
                    return compiled_engine
                except Exception as e:
                    logger.warning(
                        f"Auto-compilation to TensorRT was bypassed ({e}). "
                        "Continuing with ONNX or PyTorch fallback."
                    )

        # ---------------------------------------------------------------------
        # 2. ONNX Auto-Generation Strategy
        # ---------------------------------------------------------------------
        if onnx_candidate.exists():
            logger.info(f"Selecting optimized ONNX runtime: {onnx_candidate}")
            return onnx_candidate
        elif self.DEFAULT_MODEL_PATH.exists():
            try:
                from pipeline.inference.export import export_model
                logger.info("No ONNX model found. Auto-exporting best.pt -> best.onnx on-the-fly...")
                generated_onnx = export_model(
                    checkpoint_path=self.DEFAULT_MODEL_PATH,
                    export_format="onnx",
                    half=False,
                    dynamic=True
                )
                logger.info(f"Auto-export to ONNX complete! Generated: {generated_onnx}")
                return generated_onnx
            except Exception as e:
                logger.warning(f"Auto-export to ONNX bypassed ({e}), using master PyTorch checkpoint.")

        # ---------------------------------------------------------------------
        # 3. Master PyTorch Checkpoint Fallback
        # ---------------------------------------------------------------------
        if self.DEFAULT_MODEL_PATH.exists():
            logger.info(f"Selecting master PyTorch checkpoint: {self.DEFAULT_MODEL_PATH}")
            return self.DEFAULT_MODEL_PATH
        else:
            raise FileNotFoundError(f"Could not locate default model checkpoint at: {self.DEFAULT_MODEL_PATH}")

    def _select_device(self, requested_device: Optional[str]) -> Tuple[str, str]:
        """Auto-detects NVIDIA GPU availability or falls back to multi-threaded CPU."""
        if requested_device is not None:
            dev = requested_device.lower()
            if "cuda" in dev or dev.isdigit():
                if torch.cuda.is_available():
                    dev_name = torch.cuda.get_device_name(0)
                    return dev, f"NVIDIA GPU ({dev_name})"
                else:
                    logger.warning(f"Requested device '{requested_device}' but CUDA is not available. Falling back to CPU.")
                    return "cpu", "CPU Fallback (Multi-threaded)"
            return dev, f"CPU ({torch.get_num_threads()} threads)"

        # Auto-detection
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "0", f"NVIDIA GPU ({gpu_name}, {vram_gb:.1f} GB VRAM)"
        else:
            logger.info("No NVIDIA CUDA GPU detected. Utilizing multi-threaded CPU fallback.")
            return "cpu", f"CPU Fallback ({torch.get_num_threads()} threads)"

    def _warmup(self, iterations: int = 2):
        """Runs dummy frames to eliminate initial inference latency spikes."""
        logger.info(f"Warming up inference engine on {self.hardware_desc} ({iterations} passes)...")
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for i in range(iterations):
            _ = self.model.predict(
                source=dummy_frame,
                conf=self.conf_floor,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        logger.info("Warm-up complete. Ready for real-time inspection.")

    def _extract_detections(self, ultralytics_result) -> List[Detection]:
        """Extracts structured detections including polygon masks and areas from a single Results object."""
        detections: List[Detection] = []
        boxes = ultralytics_result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        masks_xy = None
        if ultralytics_result.masks is not None and ultralytics_result.masks.xy is not None:
            masks_xy = ultralytics_result.masks.xy

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            polygon = None
            mask_area = None
            if masks_xy is not None and idx < len(masks_xy):
                poly_arr = masks_xy[idx]
                if len(poly_arr) > 0:
                    polygon = poly_arr.tolist()
                    mask_area = float(cv2.contourArea(poly_arr.astype(np.float32)))

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    box=[round(c, 2) for c in xyxy],
                    mask=[[round(pt[0], 2), round(pt[1], 2)] for pt in polygon] if polygon else None,
                    mask_area_pixels=round(mask_area, 2) if mask_area is not None else None,
                )
            )
        return detections

    def predict_single(
        self,
        image_input: Union[str, Path, np.ndarray],
        camera_id: str = "camera_default",
        conf_floor: Optional[float] = None,
    ) -> CameraResult:
        """
        Executes inference on a single image frame (Mode A).
        """
        threshold = conf_floor if conf_floor is not None else self.conf_floor
        t0 = time.perf_counter()

        # Load image dimension if path
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Failed to read image at: {image_input}")
            h, w = img_bgr.shape[:2]
            source = img_bgr
        else:
            h, w = image_input.shape[:2]
            source = image_input

        results = self.model.predict(
            source=source,
            conf=threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        detections = self._extract_detections(results[0])

        return CameraResult(
            camera_id=camera_id,
            frame_width=w,
            frame_height=h,
            detections=detections,
            inference_latency_ms=round(latency_ms, 2),
        )

    def predict_multi_camera(
        self,
        camera_frames: Dict[str, Union[str, Path, np.ndarray]],
        conf_floor: Optional[float] = None,
    ) -> MultiCameraInferenceResult:
        """
        Executes batched inference across an array of synchronized camera frames (Mode B).
        Guarantees strictly isolated per-image NMS.
        """
        if not camera_frames:
            raise ValueError("No camera frames provided for multi-camera inference.")

        threshold = conf_floor if conf_floor is not None else self.conf_floor
        camera_ids = list(camera_frames.keys())
        sources = []
        dims = []

        # Prepare batch
        for cid in camera_ids:
            inp = camera_frames[cid]
            if isinstance(inp, (str, Path)):
                img = cv2.imread(str(inp))
                if img is None:
                    raise ValueError(f"Failed to read frame for camera '{cid}' at: {inp}")
                sources.append(img)
                dims.append((img.shape[1], img.shape[0]))
            else:
                sources.append(inp)
                dims.append((inp.shape[1], inp.shape[0]))

        # Batched forward pass in a single GPU call
        t0 = time.perf_counter()
        results = self.model.predict(
            source=sources,
            conf=threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        total_latency_ms = (time.perf_counter() - t0) * 1000

        # Build isolated per-camera outputs
        views: Dict[str, CameraResult] = {}
        for idx, cid in enumerate(camera_ids):
            w, h = dims[idx]
            # Isolated detections for this specific batch index
            view_detections = self._extract_detections(results[idx])

            views[cid] = CameraResult(
                camera_id=cid,
                frame_width=w,
                frame_height=h,
                detections=view_detections,
                inference_latency_ms=round(total_latency_ms, 2),
            )

        return MultiCameraInferenceResult(
            hardware_used=f"{self.hardware_desc} (Batch Size={len(camera_ids)})",
            total_latency_ms=round(total_latency_ms, 2),
            camera_count=len(camera_ids),
            views=views,
        )
