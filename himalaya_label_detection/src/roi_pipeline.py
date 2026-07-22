"""
roi_pipeline.py
───────────────
ROI-folder-based inspection pipeline for Himalaya NSC label inspection.

Dataset facts:
  - Full NSC images: RGB, 1504×8000 px BMP (unwrapped cylindrical label)
  - 3 pre-cropped grayscale ROI folders: ROI_LOGO / ROI_INGREDIENT / ROI_BARCODE
  - 78 good images, 42 bad images

At inference on a NEW raw RGB image:
  1. Load raw RGB image (BMP)
  2. Align to golden master (ORB + Homography) — solves translational variance
  3. Convert aligned image to grayscale (to match training crop format)
  4. Extract 3 ROI crops at the same coordinates used during data collection
  5. Score each grayscale crop with its dedicated EfficientAD model
  6. Apply per-ROI threshold → PASS / FAIL per ROI
  7. Extract bounding boxes from anomaly heatmap (connected components)
  8. Map bounding box coords back to full aligned image space
  9. Draw annotated output image (coloured boxes on grayscale→BGR canvas)
  10. Return JSON result + annotated image

Usage:
    inspector = ROIInspector.from_config("config/")
    result = inspector.inspect(cv2.imread("label.bmp", cv2.IMREAD_GRAYSCALE))
    print(result.overall_verdict)           # "PASS" or "FAIL"
    cv2.imwrite("output.png", result.annotated_image)
    print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.postprocessing.bbox import (
    BoundingBox,
    ROIDetectionResult,
    heatmap_to_bboxes,
    map_bboxes_to_full_image,
    overlay_heatmap,
    draw_bboxes,
)


# ── ROI coordinate configuration ──────────────────────────────────────────────

@dataclass
class ROICoord:
    """
    Pixel coordinates of one ROI in the aligned full image.
    These must match EXACTLY how the pre-cropped training folders were extracted.
    Fill these in from roi_config.json after Person C measures them.
    """
    name: str
    x: int   # left edge in aligned image
    y: int   # top edge in aligned image
    w: int   # width
    h: int   # height
    critical: bool = False   # critical ROIs lower the FAIL threshold

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


# ── Per-ROI scorer ────────────────────────────────────────────────────────────

class ROIAnomalyScorer:
    """Loads and runs one anomaly model for one ROI."""

    MODEL_INPUT_SIZE = (256, 256)  # must match what was used in training

    def __init__(self, roi_name: str, model_dir: Path):
        self.roi_name  = roi_name
        self.model_dir = model_dir
        self._model    = None
        self._load()

    def _load(self) -> None:
        ckpt_paths = list(self.model_dir.rglob("*.ckpt"))
        if not ckpt_paths:
            print(f"[{self.roi_name}] WARN: No checkpoint in {self.model_dir} — stub scorer active")
            return

        meta_path  = self.model_dir / "model_meta.json"
        model_type = "efficientad"
        if meta_path.exists():
            with open(meta_path) as f:
                model_type = json.load(f).get("model_type", "efficientad")

        try:
            if model_type == "efficientad":
                from anomalib.models import EfficientAd
                self._model = EfficientAd.load_from_checkpoint(str(ckpt_paths[0]))
            else:
                from anomalib.models import Patchcore
                self._model = Patchcore.load_from_checkpoint(str(ckpt_paths[0]))
            self._model.eval()
            print(f"[{self.roi_name}] Loaded {model_type} from {ckpt_paths[0].name}")
        except Exception as exc:
            print(f"[{self.roi_name}] WARN: Could not load model: {exc} — stub scorer active")

    def score(self, crop: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Score a grayscale crop.

        Returns:
            (anomaly_score, anomaly_map)
            anomaly_score: float 0.0–1.0
            anomaly_map:   float32 H×W or None
        """
        if self._model is None:
            return 0.0, None

        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage

        # Grayscale → RGB (anomalib models expect 3-channel input)
        if crop.ndim == 2:
            pil = PILImage.fromarray(crop).convert("RGB")
        else:
            pil = PILImage.fromarray(crop[:, :, ::-1])

        tensor = TF.to_tensor(TF.resize(pil, list(self.MODEL_INPUT_SIZE)))
        tensor = TF.normalize(tensor,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        batch = {"image": tensor.unsqueeze(0)}

        with torch.no_grad():
            output = self._model(batch)

        score = float(output["pred_score"].item())
        amap  = output.get("anomaly_map")
        amap_np = amap.squeeze().cpu().numpy() if amap is not None else None

        return score, amap_np


# ── Inspection result ─────────────────────────────────────────────────────────

@dataclass
class ROIInspectionResult:
    image_path:        str
    overall_verdict:   str                          # "PASS" or "FAIL"
    overall_pass:      bool
    latency_ms:        float
    roi_results:       Dict[str, ROIDetectionResult]  # per-ROI details
    annotated_image:   Optional[np.ndarray]            # BGR image with boxes + heatmap
    fail_reasons:      List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path":      self.image_path,
            "overall_verdict": self.overall_verdict,
            "overall_pass":    self.overall_pass,
            "latency_ms":      round(self.latency_ms, 1),
            "fail_reasons":    self.fail_reasons,
            "roi_results":     {
                name: r.to_dict()
                for name, r in self.roi_results.items()
            },
        }


# ── Main inspector ────────────────────────────────────────────────────────────

class ROIInspector:
    """
    ROI-folder-based Himalaya NSC label inspector.

    One EfficientAD model per ROI. Per-ROI thresholds. Bounding box output.
    """

    # Visualization colors per ROI (BGR)
    # 3 ROI folders: grayscale crops from RGB source images
    ROI_COLORS = {
        "ROI_LOGO":       (0,   0, 255),    # Red   — critical (logo defect = reject)
        "ROI_BARCODE":    (0,   0, 255),    # Red   — critical (barcode defect = reject)
        "ROI_INGREDIENT": (0, 165, 255),    # Orange — medium priority
    }
    DEFAULT_COLOR = (0, 255, 255)           # Yellow fallback for any other ROI name

    def __init__(
        self,
        roi_coords:       Dict[str, ROICoord],
        roi_scorers:      Dict[str, ROIAnomalyScorer],
        thresholds:       Dict[str, float],
        aligner=None,                         # optional: LabelAligner instance
        bbox_percentile:  float = 92.0,       # heatmap threshold percentile
        bbox_min_area_px: int   = 40,         # ignore tiny noise blobs
        bbox_margin_px:   int   = 8,          # expand bbox by this many pixels
    ):
        self.roi_coords       = roi_coords
        self.roi_scorers      = roi_scorers
        self.thresholds       = thresholds
        self.aligner          = aligner
        self.bbox_percentile  = bbox_percentile
        self.bbox_min_area_px = bbox_min_area_px
        self.bbox_margin_px   = bbox_margin_px

    # ── Public API ─────────────────────────────────────────────────────────────

    def inspect(
        self,
        raw_image: np.ndarray,
        image_path: str = "",
    ) -> ROIInspectionResult:
        """
        Inspect one label image.

        Args:
            raw_image:    Grayscale or BGR image (will be converted to grayscale).
            image_path:   Optional file path string for logging/JSON output.

        Returns:
            ROIInspectionResult with verdict, per-ROI details, and annotated image.
        """
        t0 = time.perf_counter()

        # Stage 1: Ensure RGB (full NSC images are RGB BMP)
        # Keep the RGB version for visualization, convert to gray for model scoring
        if raw_image.ndim == 2:
            # Already grayscale — convert to BGR for consistent processing
            rgb = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
        else:
            rgb = raw_image.copy()   # BGR from cv2.imread

        # Stage 2: Align RGB image to golden master (ORB + Homography)
        # This fixes translational variance (logo shifts vertically across images)
        if self.aligner is not None:
            try:
                align_result = self.aligner.align(rgb)
                aligned_rgb = align_result.image
            except Exception as exc:
                print(f"[WARN] Alignment failed: {exc} — using raw image without alignment")
                aligned_rgb = rgb
        else:
            aligned_rgb = rgb

        # Stage 3: Convert aligned RGB to grayscale for ROI scoring
        # Training was done on grayscale crops — inference must match
        aligned_gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)

        # Stage 3 & 4: Score each ROI
        roi_results: Dict[str, ROIDetectionResult] = {}
        fail_reasons: List[str] = []

        for roi_name, coord in self.roi_coords.items():
            if roi_name not in self.roi_scorers:
                continue

            # Extract grayscale ROI crop (must match training format)
            crop = aligned_gray[coord.y: coord.y + coord.h,
                                coord.x: coord.x + coord.w]

            if crop.size == 0:
                print(f"[{roi_name}] WARN: Empty crop — check ROI coordinates")
                continue

            # Score through model
            scorer = self.roi_scorers[roi_name]
            anomaly_score, anomaly_map = scorer.score(crop)

            # Get threshold for this ROI
            threshold = self.thresholds.get(roi_name, 0.5)
            is_anomalous = anomaly_score > threshold

            # Extract bounding boxes from heatmap
            bboxes_in_crop: List[BoundingBox] = []
            bboxes_in_full: List[BoundingBox] = []

            if anomaly_map is not None and is_anomalous:
                bboxes_in_crop = heatmap_to_bboxes(
                    anomaly_map,
                    threshold_percentile=self.bbox_percentile,
                    min_area_px=self.bbox_min_area_px,
                    margin_px=self.bbox_margin_px,
                )
                # Map coordinates from model input space → full aligned image space
                bboxes_in_full = map_bboxes_to_full_image(
                    bboxes_in_crop,
                    roi_offset=(coord.x, coord.y),
                    roi_original_size=(coord.w, coord.h),
                    model_input_size=ROIAnomalyScorer.MODEL_INPUT_SIZE,
                )

            roi_results[roi_name] = ROIDetectionResult(
                roi_name=roi_name,
                anomaly_score=anomaly_score,
                anomaly_map=anomaly_map,
                bounding_boxes=bboxes_in_full,   # coords in full image space
                is_anomalous=is_anomalous,
                threshold_used=threshold,
            )

            if is_anomalous:
                fail_reasons.append(
                    f"{roi_name}: score={anomaly_score:.3f} > threshold={threshold:.3f}"
                    f" | {len(bboxes_in_full)} anomaly region(s) detected"
                )

        # Stage 5: Decision
        overall_pass = len(fail_reasons) == 0
        latency_ms   = (time.perf_counter() - t0) * 1000

        # Stage 8: Annotate output image — draw on RGB aligned image
        # (show colored bboxes on the full color label for easier QA review)
        annotated = self._draw_results(aligned_rgb, roi_results, overall_pass)

        return ROIInspectionResult(
            image_path=image_path,
            overall_verdict="PASS" if overall_pass else "FAIL",
            overall_pass=overall_pass,
            latency_ms=latency_ms,
            roi_results=roi_results,
            annotated_image=annotated,
            fail_reasons=fail_reasons,
        )

    # ── Visualization ──────────────────────────────────────────────────────────

    def _draw_results(
        self,
        aligned_bgr: np.ndarray,
        roi_results: Dict[str, ROIDetectionResult],
        overall_pass: bool,
    ) -> np.ndarray:
        """
        Draw on the full aligned RGB/BGR image:
          - ROI bounding rectangles (green=pass, red=fail)
          - Heatmap overlay on anomalous ROIs
          - Anomaly bounding boxes with score labels
          - PASS/FAIL banner at top
        """
        # aligned_bgr is already BGR (from cv2.imread / alignment pipeline)
        canvas = aligned_bgr.copy() if aligned_bgr.ndim == 3 else cv2.cvtColor(aligned_bgr, cv2.COLOR_GRAY2BGR)

        for roi_name, result in roi_results.items():
            coord = self.roi_coords.get(roi_name)
            if coord is None:
                continue

            roi_color   = self.ROI_COLORS.get(roi_name, self.DEFAULT_COLOR)
            border_color = (0, 0, 200) if result.is_anomalous else (0, 180, 0)

            # Draw ROI boundary
            cv2.rectangle(canvas,
                          (coord.x, coord.y),
                          (coord.x + coord.w, coord.y + coord.h),
                          border_color, 2)

            # ROI name label
            label_y = max(coord.y + 18, 18)
            cv2.putText(canvas, roi_name,
                        (coord.x + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)

            # Heatmap overlay on anomalous ROIs
            if result.is_anomalous and result.anomaly_map is not None:
                roi_region = canvas[coord.y:coord.y + coord.h,
                                    coord.x:coord.x + coord.w]
                # Resize anomaly map to ROI size for overlay
                amap_resized = cv2.resize(
                    result.anomaly_map.astype(np.float32),
                    (coord.w, coord.h),
                    interpolation=cv2.INTER_LINEAR,
                )
                roi_with_heat = overlay_heatmap(
                    roi_region,
                    amap_resized,
                    alpha=0.40,
                )
                canvas[coord.y:coord.y + coord.h,
                       coord.x:coord.x + coord.w] = roi_with_heat

            # Draw anomaly bounding boxes (already in full image coords)
            if result.bounding_boxes:
                canvas = draw_bboxes(
                    canvas,
                    result.bounding_boxes,
                    color=roi_color,
                    thickness=2,
                    show_score=True,
                    font_scale=0.4,
                )

        # PASS / FAIL banner at top
        banner_color = (0, 160, 0) if overall_pass else (0, 0, 200)
        verdict_text = "PASS" if overall_pass else "FAIL"
        cv2.rectangle(canvas, (0, 0), (160, 36), banner_color, -1)
        cv2.putText(canvas, verdict_text,
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        models_root: str | Path | None = None,
    ) -> "ROIInspector":
        """
        Build ROIInspector from config/ directory.

        Reads:
          - config/roi_thresholds.json      (per-ROI thresholds)
          - config/roi_coord_config.json    (ROI pixel coordinates)
          - models/rois/<roi_name>/         (trained model checkpoints)
        """
        config_dir   = Path(config_dir)
        project_root = config_dir.parent

        if models_root is None:
            models_root = project_root / "models" / "rois"
        models_root = Path(models_root)

        # ── Load thresholds ───────────────────────────────────────────────────
        thresholds_path = config_dir / "roi_thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                tdata = json.load(f)
            # Support both flat {"roi": value} and nested {"thresholds": {...}}
            thresholds = tdata.get("thresholds", tdata)
            print(f"[ROIInspector] Loaded thresholds: {thresholds}")
        else:
            print(f"[ROIInspector] WARN: {thresholds_path} not found — using default 0.5")
            thresholds = {}

        # ── Load ROI coordinates ──────────────────────────────────────────────
        coord_path = config_dir / "roi_coord_config.json"
        if coord_path.exists():
            with open(coord_path) as f:
                coord_data = json.load(f)
            roi_coords = {
                name: ROICoord(
                    name=name,
                    x=d["x"], y=d["y"],
                    w=d["w"], h=d["h"],
                    critical=d.get("critical", False),
                )
                for name, d in coord_data.items()
            }
        else:
            print(f"[ROIInspector] WARN: {coord_path} not found.")
            print(f"  → Create {coord_path} with ROI pixel coordinates.")
            print(f"    See README or ask Person C to fill this in.\n")
            roi_coords = {}

        # ── Load ROI scorers ──────────────────────────────────────────────────
        roi_scorers = {}
        if models_root.exists():
            for model_dir in sorted(models_root.iterdir()):
                if model_dir.is_dir() and model_dir.name != "__pycache__":
                    roi_scorers[model_dir.name] = ROIAnomalyScorer(
                        roi_name=model_dir.name,
                        model_dir=model_dir,
                    )
        else:
            print(f"[ROIInspector] WARN: Models root not found: {models_root}")

        # ── Optionally load aligner ───────────────────────────────────────────
        aligner = None
        try:
            from src.preprocessing.align import LabelAligner
            import yaml
            pcfg_path = config_dir / "pipeline_config.yaml"
            if pcfg_path.exists():
                with open(pcfg_path) as f:
                    pcfg = yaml.safe_load(f)
                gm_path = project_root / pcfg["paths"]["golden_master"]
                if gm_path.exists():
                    aligner = LabelAligner(
                        golden_master_path=gm_path,
                        target_size=(pcfg["image"]["width"], pcfg["image"]["height"]),
                        orb_max_features=pcfg["alignment"]["orb_max_features"],
                    )
                    print("[ROIInspector] Aligner loaded ✓")
                else:
                    print(f"[ROIInspector] WARN: Golden master not found at {gm_path}")
        except Exception as exc:
            print(f"[ROIInspector] WARN: Could not load aligner: {exc}")

        # Apply defaults for ROIs with no threshold set
        for roi_name in roi_coords:
            thresholds.setdefault(roi_name, 0.5)

        return cls(
            roi_coords=roi_coords,
            roi_scorers=roi_scorers,
            thresholds=thresholds,
            aligner=aligner,
        )
