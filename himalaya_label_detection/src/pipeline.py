"""
pipeline.py
───────────
HimalayaLabelInspector — full end-to-end inference pipeline.

Stage order at inference time:
  1. Align          ORB + Homography → golden master coordinates
  2. ROI Extract    9 functional crops from the aligned image
  3. Anomaly Score  PatchCore — one score + anomaly map per ROI crop
  4. Verify         Left-Right SSIM  |  Golden Master cosine  |  Barcode decode
  5. Fuse           Rule-based PASS / FAIL
  6. Heatmap        Only on FAIL — blend anomaly overlay, extract defect crop

Usage:
    inspector = HimalayaLabelInspector.from_config("config/")
    result    = inspector.inspect(bgr_image)
    print(result.verdict)           # "PASS" or "FAIL"
    if result.heatmap:
        cv2.imshow("defect", result.heatmap.overlay)
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.preprocessing.align import LabelAligner, AlignmentResult
from src.preprocessing.roi_extractor import ROIExtractor
from src.verification.lr_check import LRCheckResult, check_left_right
from src.verification.gm_check import GoldenMasterChecker, GMCheckResult
from src.verification.barcode_check import BarcodeCheckResult, check_barcode
from src.fusion.decision import FusionConfig, FusionResult, fuse
from src.postprocessing.heatmap import (
    HeatmapResult, generate_heatmap, combine_roi_heatmaps
)


# ── Anomaly scorer ────────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Loads a trained PatchCore checkpoint and scores image crops.

    Falls back to a zero-score stub when anomalib is not installed,
    so the rest of the pipeline (verification, fusion, heatmap) can be
    developed and tested without a GPU or a trained model.
    """

    def __init__(self, model_dir: Path):
        self._model              = None
        self._model_dir          = Path(model_dir)
        self._anomalib_available = False
        self._load()

    def _load(self) -> None:
        try:
            from anomalib.models import Patchcore
            ckpt_paths = list(self._model_dir.rglob("*.ckpt"))
            if not ckpt_paths:
                print(f"[WARN] No checkpoint in {self._model_dir} — stub scorer active.")
                return
            self._model = Patchcore.load_from_checkpoint(str(ckpt_paths[0]))
            self._model.eval()
            self._anomalib_available = True
            print(f"[AnomalyScorer] Loaded: {ckpt_paths[0].name}")
        except ImportError:
            print("[WARN] anomalib not installed — stub scorer active (all scores 0.0).")
        except Exception as exc:
            print(f"[WARN] Model load failed: {exc} — stub scorer active.")

    def score(self, crop: np.ndarray) -> tuple[float, Optional[np.ndarray]]:
        """
        Score a single image crop.

        Returns:
            (anomaly_score, anomaly_map)
            anomaly_score : float 0.0–1.0 (normalised by anomalib)
            anomaly_map   : float32 H×W array, or None if not available
        """
        if not self._anomalib_available or self._model is None:
            return 0.0, None

        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image

        rgb    = crop[:, :, ::-1].copy()
        pil    = Image.fromarray(rgb)
        tensor = TF.to_tensor(TF.resize(pil, [256, 256]))
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


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InspectionResult:
    verdict:             str                        # "PASS" or "FAIL"
    passed:              bool
    alignment_inliers:   int
    roi_scores:          Dict[str, float]
    lr_result:           Optional[LRCheckResult]
    gm_result:           Optional[GMCheckResult]
    barcode_result:      Optional[BarcodeCheckResult]
    fusion:              FusionResult
    heatmap:             Optional[HeatmapResult]    = None
    anomaly_maps:        Dict[str, np.ndarray]      = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Verdict: {self.verdict}",
                 f"Alignment inliers: {self.alignment_inliers}"]
        if not self.passed:
            lines.append("Failure reasons:")
            for r in self.fusion.reasons:
                lines.append(f"  · {r}")
        return "\n".join(lines)


# ── Inspector ─────────────────────────────────────────────────────────────────

class HimalayaLabelInspector:
    """Full end-to-end Himalaya label inspection pipeline."""

    def __init__(
        self,
        aligner:                LabelAligner,
        roi_extractor:          ROIExtractor,
        anomaly_scorer:         AnomalyScorer,
        gm_checker:             GoldenMasterChecker,
        fusion_cfg:             FusionConfig,
        expected_sku:           str  = "",
        require_barcode_decode: bool = True,
        run_lr_check:           bool = True,
        run_gm_check:           bool = True,
        run_barcode_check:      bool = True,
    ):
        self.aligner                = aligner
        self.roi_extractor          = roi_extractor
        self.anomaly_scorer         = anomaly_scorer
        self.gm_checker             = gm_checker
        self.fusion_cfg             = fusion_cfg
        self.expected_sku           = expected_sku
        self.require_barcode_decode = require_barcode_decode
        self.run_lr_check           = run_lr_check
        self.run_gm_check           = run_gm_check
        self.run_barcode_check      = run_barcode_check

    # ── Public API ────────────────────────────────────────────────────────────

    def inspect(self, raw_image: np.ndarray) -> InspectionResult:
        """
        Run full inspection on a raw (unaligned) BGR label image.

        Returns:
            InspectionResult with verdict and all intermediate signals.
        """
        # Stage 1 — Align
        align_result: AlignmentResult = self.aligner.align(raw_image)
        aligned = align_result.image

        # Stage 2 — Extract ROIs
        roi_result = self.roi_extractor.extract(aligned)

        # Stage 3 — Score each ROI
        roi_scores:   Dict[str, float]      = {}
        anomaly_maps: Dict[str, np.ndarray] = {}

        for roi_name, crop in roi_result.crops.items():
            if not roi_result.valid.get(roi_name, False):
                roi_scores[roi_name] = 0.0
                continue
            score, amap = self.anomaly_scorer.score(crop)
            roi_scores[roi_name] = score
            if amap is not None:
                anomaly_maps[roi_name] = amap

        # Stage 4 — Global verification
        lr_result      = None
        gm_result      = None
        barcode_result = None

        if self.run_lr_check:
            lr_result = check_left_right(
                aligned, threshold=self.fusion_cfg.lr_ssim_threshold
            )

        if self.run_gm_check:
            try:
                gm_result = self.gm_checker.score(
                    aligned, threshold=self.fusion_cfg.gm_cosine_threshold
                )
            except RuntimeError:
                pass    # embedding not built yet — skip silently

        if self.run_barcode_check and "ROI_BARCODE" in roi_result.crops:
            barcode_result = check_barcode(
                roi_result.crops["ROI_BARCODE"],
                expected_sku=self.expected_sku,
                require_decode=self.require_barcode_decode,
            )

        # Stage 5 — Decision fusion
        fusion_result = fuse(
            roi_scores=roi_scores,
            lr_score=lr_result.score if lr_result else None,
            gm_score=gm_result.similarity if gm_result else None,
            barcode_passed=(barcode_result.passed
                            if barcode_result else True),
            barcode_failure_reason=(barcode_result.failure_reason
                                    if barcode_result else None),
            barcode_value=(barcode_result.decoded_value
                           if barcode_result else None),
            cfg=self.fusion_cfg,
        )

        # Stage 6 — Heatmap (FAIL path only)
        heatmap_result = None
        if not fusion_result.passed and anomaly_maps:
            combined_map = combine_roi_heatmaps(
                aligned, anomaly_maps, self.roi_extractor.configs
            )
            heatmap_result = generate_heatmap(aligned, combined_map)

        return InspectionResult(
            verdict="PASS" if fusion_result.passed else "FAIL",
            passed=fusion_result.passed,
            alignment_inliers=align_result.num_inliers,
            roi_scores=roi_scores,
            lr_result=lr_result,
            gm_result=gm_result,
            barcode_result=barcode_result,
            fusion=fusion_result,
            heatmap=heatmap_result,
            anomaly_maps=anomaly_maps,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_dir: str | Path) -> "HimalayaLabelInspector":
        """
        Build a HimalayaLabelInspector from the config/ directory.
        All paths are resolved relative to the project root.
        """
        config_dir   = Path(config_dir)
        project_root = config_dir.parent

        with open(config_dir / "pipeline_config.yaml") as f:
            pcfg = yaml.safe_load(f)

        thresholds_path = config_dir / "thresholds.yaml"
        fusion_cfg = (FusionConfig.from_yaml(thresholds_path)
                      if thresholds_path.exists() else FusionConfig())

        barcode_cfg: dict = {}
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                barcode_cfg = yaml.safe_load(f).get("barcode", {})

        target_size = (pcfg["image"]["width"], pcfg["image"]["height"])

        aligner = LabelAligner(
            golden_master_path=project_root / pcfg["paths"]["golden_master"],
            target_size=target_size,
            orb_max_features=pcfg["alignment"]["orb_max_features"],
            match_keep_top=pcfg["alignment"]["match_keep_top"],
            ransac_threshold=pcfg["alignment"]["ransac_threshold"],
            min_match_count=pcfg["alignment"]["min_match_count"],
        )

        roi_extractor = ROIExtractor(config_dir / "roi_config.yaml")

        anomaly_scorer = AnomalyScorer(project_root / "models" / "full_label")

        gm_path    = project_root / "models" / "golden_master_embedding.pt"
        gm_checker = GoldenMasterChecker(gm_path if gm_path.exists() else None)

        return cls(
            aligner=aligner,
            roi_extractor=roi_extractor,
            anomaly_scorer=anomaly_scorer,
            gm_checker=gm_checker,
            fusion_cfg=fusion_cfg,
            expected_sku=barcode_cfg.get("expected_sku", ""),
            require_barcode_decode=barcode_cfg.get("require_decode", True),
        )
