"""
anchor.py
─────────────────────────────────────────────────────────────────────────────
Template-matching based ROI anchor finder for Himalaya NSC label images.

Each of the 3 ROI regions (LOGO, INGREDIENT, ADDRESS) has a distinctive
visual signature. We use a saved reference patch from each region and run
cv2.matchTemplate independently to find where each region sits in a new
full-image (8000 × 1504 px RGB BMP).

Why template matching (not ORB/Homography, not YOLO):
  - No training required
  - Tested score 0.75–0.93 across dataset
  - Robust to the 4620px vertical shift seen across images
  - Runs in ~40ms per ROI on CPU

Usage:
    finder = TemplateAnchorFinder(Path("config/"))
    crops  = finder.extract_crops(bgr_image)   # dict: roi_name → gray crop
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AnchorResult:
    roi_name: str
    y: int              # top row of this ROI in the full image
    score: float        # template match confidence (0–1)
    found: bool         # False if score < threshold


class TemplateAnchorFinder:
    """
    Finds each ROI region in a full 8000×1504 image using template matching.

    Expects in config_dir/:
        patch_logo.png        – reference patch for ROI_LOGO
        patch_ingredient.png  – reference patch for ROI_INGREDIENT
        patch_address.png     – reference patch for ROI_ADDRESS
        roi_coord_config.json – roi heights (h field per ROI)
    """

    # Patch filenames for each ROI
    PATCH_FILES = {
        "ROI_LOGO":       "patch_logo.png",
        "ROI_INGREDIENT": "patch_ingredient.png",
        "ROI_ADDRESS":    "patch_address.png",
    }

    # Score must exceed this to accept a match
    # Validated: all 15 tested images scored 0.75+ for logo
    MATCH_THRESHOLD: float = 0.50

    def __init__(self, config_dir: Path):
        config_dir = Path(config_dir)
        self.templates: dict[str, np.ndarray] = {}
        self.roi_heights: dict[str, int] = {}

        # Load ROI heights from coord config
        coord_file = config_dir / "roi_coord_config.json"
        if coord_file.exists():
            with open(coord_file) as f:
                coords = {k: v for k, v in json.load(f).items()
                          if not k.startswith("_")}
            for roi_name, vals in coords.items():
                if "h" in vals:
                    self.roi_heights[roi_name] = int(vals["h"])

        # Load template patches
        for roi_name, patch_file in self.PATCH_FILES.items():
            patch_path = config_dir / patch_file
            if patch_path.exists():
                patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
                if patch is not None:
                    self.templates[roi_name] = patch
                else:
                    print(f"[WARN] Could not read patch: {patch_path}")
            else:
                print(f"[WARN] Patch not found: {patch_path}  — ROI {roi_name} will be skipped")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def find_all(self, gray_image: np.ndarray) -> dict[str, AnchorResult]:
        """
        Run template matching for each ROI independently.
        Returns dict of roi_name → AnchorResult.
        Always returns topmost (smallest y) logo instance.
        """
        results: dict[str, AnchorResult] = {}
        for roi_name, template in self.templates.items():
            y, score = self._match(gray_image, template)
            results[roi_name] = AnchorResult(
                roi_name=roi_name,
                y=max(0, y),
                score=score,
                found=(y >= 0),
            )
        return results

    def extract_crops(
        self,
        bgr_image: np.ndarray,
    ) -> dict[str, Optional[np.ndarray]]:
        """
        Full pipeline: BGR image in → dict of grayscale ROI crops out.

        Returns None for any ROI whose template was not found.
        Caller should skip None crops.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) \
               if bgr_image.ndim == 3 else bgr_image.copy()

        anchors = self.find_all(gray)
        crops: dict[str, Optional[np.ndarray]] = {}

        for roi_name, anchor in anchors.items():
            if not anchor.found:
                print(f"[{roi_name}] WARN: not found (score={anchor.score:.3f})")
                crops[roi_name] = None
                continue

            h = self.roi_heights.get(roi_name)
            if not h:
                print(f"[{roi_name}] WARN: height not in roi_coord_config.json")
                crops[roi_name] = None
                continue

            y1 = anchor.y
            y2 = min(gray.shape[0], y1 + h)
            crop = gray[y1:y2, :]

            if crop.size == 0:
                print(f"[{roi_name}] WARN: empty crop at y={y1}:{y2}")
                crops[roi_name] = None
                continue

            crops[roi_name] = crop

        return crops

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _match(self, image: np.ndarray, template: np.ndarray) -> tuple[int, float]:
        """
        Run matchTemplate and return (y_topmost, best_score).
        Returns (-1, score) if no match exceeds MATCH_THRESHOLD.
        """
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val < self.MATCH_THRESHOLD:
            return -1, float(max_val)

        # Collect all positions above threshold
        locs = np.where(res >= self.MATCH_THRESHOLD)
        y_positions = sorted(locs[0].tolist())

        # Cluster positions within 500px (same logo instance)
        clustered: list[int] = []
        c = y_positions[0]
        prev = y_positions[0]
        for y in y_positions[1:]:
            if y - prev > 500:
                clustered.append(c)
                c = y
            prev = y
        clustered.append(c)

        # Return topmost match (smallest y)
        return clustered[0], float(max_val)
