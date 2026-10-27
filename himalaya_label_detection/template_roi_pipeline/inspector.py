"""
inspector.py
─────────────────────────────────────────────────────────────────────────────
End-to-end inspection for Himalaya NSC label scans:

    full scan (1504 x 8000)
        -> grayscale
        -> ROITemplateMatcher: split into ROI_1..ROI_4 (one occurrence each)
        -> per-ROI ONNX EfficientAD classifier, fed the GRAYSCALE crop
        -> per-ROI PASS/FAIL (score vs calibrated threshold)
        -> aggregated final verdict

ONNX contract (see scripts/export_grayscale_roi_onnx.py):
    input  "image"       float32 [1,3,256,256]  RGB in [0,1], no mean/std
    output "anomaly_map" float32 [1,1,256,256]
    score = mean(anomaly_map) by default (score_stat selects max / p99);  FAIL when score > threshold

input_mode:
    "gray"  (default) the classifier gets the grayscale crop replicated to 3 channels.
            Use this with the grayscale-trained models (models/gray_scale_rois).
    "color" the classifier gets the RGB crop. The ONNX files currently committed in
            models/rois were trained on COLOR crops; fed grayscale they flag every
            crop as anomalous (mean map ~0.5 on clean crops vs ~0 in color mode).

Final verdict rule: the label is PASS only if all four ROIs were located AND
every ROI score is <= its threshold. A ROI that cannot be located is reported
as MISSING and fails the label (fail_on_missing=True), since an unverified
region must not be passed silently.

Usage:
    from template_roi_pipeline import TemplateROIInspector   # himalaya_label_detection/ on sys.path
    inspector = TemplateROIInspector.from_config("himalaya_label_detection/template_roi_pipeline/config",
                                                 models_root="models/rois")
    result = inspector.inspect(cv2.imread("scan.bmp", cv2.IMREAD_GRAYSCALE))
    print(result.verdict, result.fail_reasons)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .roi_template_matcher import (
    ROI_NAMES,
    ROIMatch,
    ROITemplateMatcher,
)

MODEL_INPUT_SIZE = 256


# ── Per-ROI classifier ────────────────────────────────────────────────────────

class ONNXROIClassifier:
    """One exported EfficientAD ONNX model for one ROI."""

    SCORE_STATS = ("mean", "max", "p99")

    def __init__(self, roi_name: str, model_path: Path, intra_op_threads: int = 0, score_stat: str = "mean"):
        """intra_op_threads=0 lets onnxruntime use all cores; cap it when running several sessions at once."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        if score_stat not in self.SCORE_STATS:
            raise ValueError(f"score_stat must be one of {self.SCORE_STATS}")
        self.roi_name = roi_name
        self.score_stat = score_stat
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def preprocess(crop: np.ndarray) -> np.ndarray:
        """uint8 crop -> float32 [1,3,256,256], RGB / 255. A 2-D gray crop is replicated to 3 channels;
        a 3-D crop is taken as BGR (OpenCV order)."""
        small = cv2.resize(crop, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        elif small.ndim == 3 and small.shape[2] == 3:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported crop shape {crop.shape}")
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

    def score(self, crop: np.ndarray) -> tuple[float, np.ndarray]:
        amap = self.session.run(["anomaly_map"], {"image": self.preprocess(crop)})[0]
        amap = np.asarray(amap, dtype=np.float32)[0, 0]
        value = {"mean": amap.mean, "max": amap.max, "p99": lambda: np.percentile(amap, 99)}[self.score_stat]()
        return float(value), amap


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ROIVerdict:
    roi_name: str
    status: str                       # "PASS" | "FAIL" | "MISSING"
    score: Optional[float]
    threshold: Optional[float]
    match: ROIMatch
    anomaly_map: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score": None if self.score is None else round(self.score, 6),
            "threshold": None if self.threshold is None else round(self.threshold, 6),
            **self.match.to_dict(),
        }


@dataclass
class InspectionResult:
    verdict: str                      # "PASS" | "FAIL"
    rois: Dict[str, ROIVerdict]
    fail_reasons: List[str]
    latency_ms: float
    image_path: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "verdict": self.verdict,
            "fail_reasons": self.fail_reasons,
            "latency_ms": round(self.latency_ms, 1),
            "rois": {n: v.to_dict() for n, v in self.rois.items()},
        }


# ── Inspector ─────────────────────────────────────────────────────────────────

class TemplateROIInspector:
    def __init__(
        self,
        matcher: ROITemplateMatcher,
        classifiers: Dict[str, ONNXROIClassifier],
        thresholds: Dict[str, float],
        fail_on_missing: bool = True,
        input_mode: str = "gray",
    ):
        if input_mode not in ("gray", "color"):
            raise ValueError("input_mode must be 'gray' or 'color'")
        self.input_mode = input_mode
        missing = [r for r in ROI_NAMES if r not in classifiers or r not in thresholds]
        if missing:
            raise ValueError(f"Need a classifier and threshold for every ROI; missing: {missing}")
        self.matcher = matcher
        self.classifiers = classifiers
        self.thresholds = thresholds
        self.fail_on_missing = fail_on_missing

    def inspect(self, image: np.ndarray, image_path: str = "") -> InspectionResult:
        """
        image: uint8 full scan, gray (H,W) or BGR (H,W,3). Localisation always runs on grayscale;
        in input_mode="color" the classifiers are fed the BGR crops, so a BGR image is required.
        """
        t0 = time.perf_counter()
        if self.input_mode == "color" and image.ndim != 3:
            raise ValueError("input_mode='color' needs a BGR image")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source = image if self.input_mode == "color" else gray
        matches = self.matcher.locate(gray)
        crops = self.matcher.extract(source, matches)

        rois: Dict[str, ROIVerdict] = {}
        reasons: List[str] = []
        for name in ROI_NAMES:
            m, crop, thr = matches[name], crops[name], self.thresholds[name]
            if crop is None:
                rois[name] = ROIVerdict(name, "MISSING", None, thr, m)
                if self.fail_on_missing:
                    reasons.append(f"{name}: not located (template score {m.score:.3f})")
                continue
            # The classifiers were trained on crops cut from the scan rotated 90° clockwise
            # (upright text, see scripts/crop_roi.py), while matching runs in the raw frame.
            upright = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            score, amap = self.classifiers[name].score(upright)
            status = "FAIL" if score > thr else "PASS"
            rois[name] = ROIVerdict(name, status, score, thr, m, amap)
            if status == "FAIL":
                reasons.append(f"{name}: score {score:.4f} > threshold {thr:.4f}")

        verdict = "FAIL" if reasons else "PASS"
        return InspectionResult(verdict, rois, reasons, (time.perf_counter() - t0) * 1000, image_path)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        models_root: str | Path,
        thresholds_file: str | Path | None = None,
        min_match_score: float | None = None,
        fail_on_missing: bool = True,
        input_mode: str = "gray",
        score_stat: str = "mean",
    ) -> "TemplateROIInspector":
        config_dir, models_root = Path(config_dir), Path(models_root)
        thresholds_file = Path(thresholds_file) if thresholds_file else config_dir / "roi_onnx_thresholds.json"
        if not thresholds_file.is_file():
            raise FileNotFoundError(
                f"{thresholds_file} not found. Calibrate with scripts/run_template_inference.py --calibrate."
            )
        doc = json.loads(thresholds_file.read_text(encoding="utf-8"))
        thresholds = {k: float(v) for k, v in doc.get("thresholds", doc).items() if k in ROI_NAMES}

        classifiers = {}
        for name in ROI_NAMES:
            path = models_root / name / "weights" / "model.onnx"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ONNX model: {path}")
            classifiers[name] = ONNXROIClassifier(name, path, score_stat=score_stat)

        matcher = ROITemplateMatcher.from_config(config_dir, min_match_score)
        return cls(matcher, classifiers, thresholds, fail_on_missing, input_mode)
