"""
Core Fill Level Inference Engine (Station 2).
Handles model loading, hardware auto-detection (NVIDIA GPU with CPU fallback),
on-the-fly FP16 TensorRT compilation (batch=1), single-image forward pass,
spatial association, and deterministic geometric fill ratio calculation.
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

from pipeline.env_loader import REPO_ROOT, env_path
from pipeline.fill_level.config import fill_threshold_from_env
from .schemas import BottleBox, MeniscusBox, SingleImageFillResult

logger = logging.getLogger("pipeline.fill_level.inference")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FillLevelInferenceEngine:
    """
    Dedicated inference engine for liquid fill level verification.
    1. Independent Model Hierarchy & Versioning:
       - Tracks separate fill version via VISIONQAI_FILL_ENGINE_VERSION (default: v1).
       - Automatically compiles TensorRT FP16 engine on GPU with batch=1.
       - Gracefully falls back to pre-compiled ONNX or PyTorch on CPU.
    2. Single-Image Conveyor Inspection:
       - Optimized for single full-bottle machine vision frames (batch=1).
    3. Spatial Association & Deterministic Fill Math:
       - Associates water surface meniscus line inside detected bottle boundaries.
       - Corrects for outer padding margin (3% top, 2% bottom trim).
       - Computes normalized liquid fill ratio [0.0, 1.0].
    """

    # Override with VISIONQAI_FILL_MODELS_DIR (absolute, or relative to the repo root)
    DEFAULT_MODELS_DIR = env_path("VISIONQAI_FILL_MODELS_DIR", REPO_ROOT / "models" / "fill-level-detector")

    BOTTLE_TOP_TRIM = 0.03   # Trim 3% top padding from bottle box
    BOTTLE_BOT_TRIM = 0.02   # Trim 2% bottom padding from bottle box

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        underfill_threshold: Optional[float] = None,
        overfill_threshold: Optional[float] = None,
        imgsz: int = 640,
        warmup: bool = True,
    ):
        self.imgsz = imgsz
        self.models_dir = Path(models_dir) if models_dir else self.DEFAULT_MODELS_DIR
        self.engine_version = os.getenv("VISIONQAI_FILL_ENGINE_VERSION", "v1").strip().lower()
        
        self.underfill_threshold = underfill_threshold if underfill_threshold is not None else fill_threshold_from_env("VISIONQAI_FILL_UNDERFILL_THRESHOLD", 0.40)
        self.overfill_threshold = overfill_threshold if overfill_threshold is not None else fill_threshold_from_env("VISIONQAI_FILL_OVERFILL_THRESHOLD", 0.85)
        self.bottle_conf_thresh = float(os.getenv("VISIONQAI_FILL_DET_CONF", "0.25"))
        self.water_conf_thresh = float(os.getenv("VISIONQAI_FILL_WATER_CONF", "0.15"))

        logger.info(
            f"Initializing FillLevelInferenceEngine | Engine Version: {self.engine_version} | "
            f"Underfill: <{self.underfill_threshold*100:.1f}% | Overfill: >{self.overfill_threshold*100:.1f}%"
        )

        # 1. Hardware auto-detection
        self.device, self.hardware_desc = self._select_device(device)
        logger.info(f"Active hardware runtime: {self.hardware_desc} (target: {self.device})")

        # 2. Resolve model paths (TensorRT -> ONNX -> PyTorch)
        self.bottle_model_path = self._resolve_model_path(
            model_key="bottle_detector",
            pt_filename="bottle_detector_best.pt",
            onnx_filename="bottle_detector_best.onnx"
        )
        self.water_model_path = self._resolve_model_path(
            model_key="water_surface",
            pt_filename="best.pt",
            onnx_filename="water_surface_best.onnx"
        )

        # 3. Load models
        t0 = time.perf_counter()
        self.bottle_model = YOLO(str(self.bottle_model_path), task="detect")
        self.water_model = YOLO(str(self.water_model_path), task="detect")
        load_time = (time.perf_counter() - t0) * 1000
        logger.info(f"Fill level models loaded in {load_time:.1f} ms")
        logger.info(f"  Bottle model: {self.bottle_model_path.name} ({self.bottle_model.names})")
        logger.info(f"  Water model : {self.water_model_path.name} ({self.water_model.names})")

        # 4. Warm-up
        if warmup:
            self._warmup()

    def _select_device(self, requested_device: Optional[str]) -> Tuple[str, str]:
        """Detects NVIDIA GPU availability or defaults to multi-threaded CPU."""
        if requested_device is not None:
            dev = requested_device.lower()
            if "cuda" in dev or dev.isdigit():
                if torch.cuda.is_available():
                    dev_name = torch.cuda.get_device_name(0)
                    return dev, f"NVIDIA GPU ({dev_name})"
                logger.warning(f"Requested device '{requested_device}' but CUDA is unavailable. Falling back to CPU.")
                return "cpu", "CPU Fallback (Multi-threaded)"
            return dev, f"CPU ({torch.get_num_threads()} threads)"

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "0", f"NVIDIA GPU ({gpu_name}, {vram_gb:.1f} GB VRAM)"
        
        return "cpu", f"CPU Fallback ({torch.get_num_threads()} threads)"

    def _resolve_model_path(
        self,
        model_key: str,
        pt_filename: str,
        onnx_filename: str
    ) -> Path:
        """
        Auto-resolves and compiles the target model runtime:
        1. On CUDA GPU: Compiles {model_key}_{version}.engine (FP16, batch=1, imgsz=640) if missing.
        2. On CPU / Fallback: Selects {model_key}_best.onnx or master PyTorch checkpoint.
        """
        base_dir = self.models_dir
        pt_path = base_dir / pt_filename
        onnx_path = base_dir / onnx_filename
        engine_filename = f"{model_key}_{self.engine_version}.engine"
        engine_path = base_dir / engine_filename

        force_recompile = os.getenv("VISIONQAI_FILL_FORCE_RECOMPILE", "0").strip().lower() in ("1", "true", "yes")

        # 1. NVIDIA TensorRT Compilation Strategy (batch=1)
        if torch.cuda.is_available():
            if force_recompile and engine_path.exists():
                logger.info(f"Force recompile requested (VISIONQAI_FILL_FORCE_RECOMPILE=1). Unlinking {engine_path.name}...")
                engine_path.unlink(missing_ok=True)

            if engine_path.exists():
                logger.info(f"Found existing compiled TensorRT engine ({self.engine_version}): {engine_path}")
                return engine_path

            if pt_path.exists():
                logger.info(
                    f"NVIDIA GPU detected. Auto-compiling {pt_path.name} -> {engine_filename} "
                    f"(FP16, batch=1, imgsz={self.imgsz}) on-the-fly..."
                )
                try:
                    from pipeline.inference.export import export_model
                    compiled_engine = export_model(
                        checkpoint_path=pt_path,
                        export_format="engine",
                        half=True,
                        dynamic=False,
                        imgsz=self.imgsz,
                        batch=1,
                        output_name=engine_filename
                    )
                    logger.info(f"Auto-compilation complete! Generated: {compiled_engine}")
                    return compiled_engine
                except Exception as e:
                    logger.warning(f"TensorRT auto-compilation bypassed for {model_key} ({e}). Falling back to ONNX/PyTorch.")

        # 2. ONNX Runtime Strategy
        if onnx_path.exists():
            logger.info(f"Selecting optimized ONNX runtime: {onnx_path}")
            return onnx_path
        elif pt_path.exists():
            try:
                from pipeline.inference.export import export_model
                logger.info(f"No ONNX found for {model_key}. Auto-exporting {pt_path.name} -> {onnx_path.name}...")
                generated_onnx = export_model(
                    checkpoint_path=pt_path,
                    export_format="onnx",
                    half=False,
                    dynamic=True,
                    imgsz=self.imgsz,
                    batch=1,
                    output_name=onnx_filename
                )
                return generated_onnx
            except Exception as e:
                logger.warning(f"Auto-export to ONNX bypassed for {model_key} ({e}).")

        # 3. Master PyTorch Checkpoint Fallback
        if pt_path.exists():
            logger.info(f"Selecting master PyTorch checkpoint: {pt_path}")
            return pt_path

        raise FileNotFoundError(f"Could not locate checkpoint '{pt_filename}' or '{onnx_filename}' in {base_dir}")

    def _warmup(self, iterations: int = 2):
        """Runs dummy single-frame forward passes to eliminate first-cycle latency."""
        logger.info(f"Warming up FillLevelInferenceEngine on {self.hardware_desc} ({iterations} passes)...")
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            _ = self.bottle_model.predict(dummy, verbose=False)
            _ = self.water_model.predict(dummy, verbose=False)
        logger.info("FillLevelInferenceEngine warmup complete.")

    @classmethod
    def compute_fill_ratio(cls, bottle_bbox: List[int], water_bbox: List[int]) -> float:
        """
        Computes normalized liquid fill ratio from bottle and meniscus bounding boxes.
        Trims 3% from top and 2% from bottom to account for object detector padding margin.
        Returns float between 0.0 (empty) and 1.0 (completely full).
        """
        bx1, by1, bx2, by2 = bottle_bbox
        bh = by2 - by1

        by1_corr = by1 + int(bh * cls.BOTTLE_TOP_TRIM)
        by2_corr = by2 - int(bh * cls.BOTTLE_BOT_TRIM)
        bottle_h_corr = max(by2_corr - by1_corr, 1)

        # Meniscus Center Y (where liquid meets air)
        water_y = (water_bbox[1] + water_bbox[3]) // 2
        # Clamp meniscus within bottle boundaries
        water_y = max(by1_corr, min(by2_corr, water_y))

        fill_from_bottom = by2_corr - water_y
        fill_ratio = fill_from_bottom / bottle_h_corr
        return round(float(fill_ratio), 3)

    def predict_single_frame(self, frame: np.ndarray) -> SingleImageFillResult:
        """
        Executes complete fill level inspection cycle on a single bottle profile frame:
        1. Detects bottle bounding box [bx1, by1, bx2, by2].
        2. Detects water surface line [wx1, wy1, wx2, wy2].
        3. Spatially associates meniscus horizontally within bottle bounds.
        4. Calculates normalized liquid fill ratio and assigns status verdict.
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Stage 1: Detect Bottle
        r_bottle = self.bottle_model.predict(
            frame,
            conf=self.bottle_conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        best_bottle: Optional[BottleBox] = None
        for box in r_bottle.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.bottle_model.names.get(cls_id, "").lower()
            # In bottle_detector_best.pt, class 0 is 'bottle'
            if "bottle" in cls_name or cls_id == 0:
                coords = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                if best_bottle is None or conf > best_bottle.confidence:
                    best_bottle = BottleBox(box=coords, confidence=round(conf, 3))

        # If no bottle detected (e.g. empty conveyor gap)
        if best_bottle is None:
            latency = (time.perf_counter() - t0) * 1000.0
            return SingleImageFillResult(
                hardware_used=self.hardware_desc,
                inference_latency_ms=round(latency, 2),
                bottle=None,
                meniscus=None,
                fill_ratio=None,
                fill_verdict="no_bottle",
                water_y=None,
                bottle_h=None
            )

        bx1, by1, bx2, by2 = best_bottle.box
        bottle_h = by2 - by1

        # Stage 2: Detect Water Surface Meniscus Line
        r_water = self.water_model.predict(
            frame,
            conf=self.water_conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        best_meniscus: Optional[MeniscusBox] = None
        for box in r_water.boxes:
            wx1, wy1, wx2, wy2 = [int(v) for v in box.xyxy[0].tolist()]
            water_cx = (wx1 + wx2) // 2
            conf = float(box.conf[0])

            # Spatial Association: Meniscus must be horizontally within bottle bounds
            # and vertically inside the bottle
            if bx1 <= water_cx <= bx2 and (by1 <= wy1 <= by2 or by1 <= wy2 <= by2):
                if best_meniscus is None or conf > best_meniscus.confidence:
                    water_y = (wy1 + wy2) // 2
                    best_meniscus = MeniscusBox(
                        box=[wx1, wy1, wx2, wy2],
                        water_y=water_y,
                        confidence=round(conf, 3)
                    )

        # Stage 3: Evaluate Fill Ratio & Verdict
        latency = (time.perf_counter() - t0) * 1000.0

        if best_meniscus is None:
            # Bottle detected, but no water surface found (empty or opaque container)
            return SingleImageFillResult(
                hardware_used=self.hardware_desc,
                inference_latency_ms=round(latency, 2),
                bottle=best_bottle,
                meniscus=None,
                fill_ratio=0.0,
                fill_verdict="no_water",
                water_y=None,
                bottle_h=bottle_h
            )

        fill_ratio = self.compute_fill_ratio(best_bottle.box, best_meniscus.box)

        if fill_ratio < self.underfill_threshold:
            verdict = "underfill"
        elif fill_ratio > self.overfill_threshold:
            verdict = "overfill"
        else:
            verdict = "normal"

        return SingleImageFillResult(
            hardware_used=self.hardware_desc,
            inference_latency_ms=round(latency, 2),
            bottle=best_bottle,
            meniscus=best_meniscus,
            fill_ratio=fill_ratio,
            fill_verdict=verdict,
            water_y=best_meniscus.water_y,
            bottle_h=bottle_h
        )
