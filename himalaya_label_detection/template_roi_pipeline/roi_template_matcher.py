"""
roi_template_matcher.py
─────────────────────────────────────────────────────────────────────────────
Template-matching ROI localiser for Himalaya NSC label images.

A full scan is a 1504 x 8000 px unwrapped label. The four inspection regions
(ROI_1..ROI_4, see config/roi_coord_config.json) each appear up to twice
because the label wraps around the tube. For every ROI we keep ONE
occurrence: the best-scoring one.

Why full-ROI templates:
  cv2.matchTemplate only scores positions where the template fits entirely
  inside the image, so an occurrence that is cut by the image edge can never
  win. That gives us the "use the complete, uncut occurrence" rule for free.

Search is coarse-to-fine: match on a 1/COARSE_SCALE downsample, then refine
at full resolution inside a small window around the coarse peak.

Usage:
    matcher = ROITemplateMatcher.from_config("himalaya_label_detection/template_roi_pipeline/config")
    matches = matcher.locate(gray_image)            # dict ROI_n -> ROIMatch
    crops   = matcher.extract(gray_image, matches)  # dict ROI_n -> gray crop
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ROI_NAMES = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")


@dataclass
class ROIMatch:
    roi_name: str
    x: int            # left edge in the full image
    y: int            # top edge in the full image
    w: int
    h: int
    score: float      # TM_CCOEFF_NORMED at full resolution, -1..1
    found: bool       # score >= min_score

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "match_score": round(self.score, 4), "found": self.found,
        }


class ROITemplateMatcher:
    COARSE_SCALE = 4          # coarse search runs on image / COARSE_SCALE
    REFINE_RADIUS = 12        # full-res refine window half-size, in px
    # Below this the ROI is reported as not found. ROI_1/ROI_3 score >= 0.83 on the
    # 120 NSC scans, but ROI_2/ROI_4 carry thin low-contrast text and bottom out
    # near 0.55 even when the crop is visibly correct.
    DEFAULT_MIN_SCORE = 0.45
    # ROIs are distinct label regions and never overlap. Low-contrast ROI templates can still score
    # ~0.57 on a *different* ROI's area (e.g. ROI_4's template on ROI_3's text panel), which is above
    # DEFAULT_MIN_SCORE, so two found ROIs overlapping by more than this fraction of the smaller one
    # means the lower-scoring match is false.
    MAX_OVERLAP = 0.10

    def __init__(self, templates: Dict[str, np.ndarray], min_score: float = DEFAULT_MIN_SCORE):
        self.templates = templates
        self.min_score = min_score
        s = self.COARSE_SCALE
        self._coarse = {
            name: cv2.resize(t, (max(1, t.shape[1] // s), max(1, t.shape[0] // s)),
                             interpolation=cv2.INTER_AREA)
            for name, t in templates.items()
        }

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_dir: str | Path, min_score: Optional[float] = None) -> "ROITemplateMatcher":
        """Load config_dir/roi_templates/<roi>_template.png for every ROI."""
        tdir = Path(config_dir) / "roi_templates"
        templates: Dict[str, np.ndarray] = {}
        for name in ROI_NAMES:
            path = tdir / f"{name.lower()}_template.png"
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(
                    f"Template missing: {path}. Run scripts/build_roi_templates.py first."
                )
            templates[name] = img
        return cls(templates, min_score if min_score is not None else cls.DEFAULT_MIN_SCORE)

    # ── Public API ────────────────────────────────────────────────────────────

    def locate(self, gray: np.ndarray) -> Dict[str, ROIMatch]:
        """Find the best occurrence of every ROI in a grayscale full image."""
        if gray.ndim != 2:
            raise ValueError("locate() expects a single-channel grayscale image")
        s = self.COARSE_SCALE
        coarse_img = cv2.resize(gray, (gray.shape[1] // s, gray.shape[0] // s),
                                interpolation=cv2.INTER_AREA)
        matches = {name: self._locate_one(gray, coarse_img, name) for name in self.templates}

        # Re-search the loser of any overlap with the other ROIs' areas excluded, so a ROI that is
        # really absent/damaged ends up "not found" instead of silently sitting on a neighbour.
        for _ in range(len(matches)):
            loser = self._overlap_loser(matches)
            if loser is None:
                break
            others = [m.xyxy for n, m in matches.items() if n != loser and m.found]
            matches[loser] = self._locate_one(gray, coarse_img, loser, exclude=others)
        return matches

    def extract(self, image: np.ndarray, matches: Dict[str, ROIMatch]) -> Dict[str, Optional[np.ndarray]]:
        """Crop each located ROI from `image` (gray or BGR, same size as the matched image). Unfound ROIs give None."""
        out: Dict[str, Optional[np.ndarray]] = {}
        for name, m in matches.items():
            if not m.found:
                out[name] = None
                continue
            x1, y1, x2, y2 = m.xyxy
            out[name] = image[y1:y2, x1:x2]
        return out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _overlap_loser(self, matches: Dict[str, ROIMatch]) -> Optional[str]:
        """Name of the lower-scoring ROI in the first found pair that overlaps too much, else None."""
        found = [m for m in matches.values() if m.found]
        for i, a in enumerate(found):
            for b in found[i + 1:]:
                ix = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                iy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                if ix > 0 and iy > 0 and ix * iy / min(a.w * a.h, b.w * b.h) > self.MAX_OVERLAP:
                    return a.roi_name if a.score < b.score else b.roi_name
        return None

    def _locate_one(self, gray: np.ndarray, coarse_img: np.ndarray, name: str,
                    exclude: Optional[list[tuple[int, int, int, int]]] = None) -> ROIMatch:
        tmpl = self.templates[name]
        th, tw = tmpl.shape
        s = self.COARSE_SCALE
        H, W = gray.shape

        if th > H or tw > W:
            return ROIMatch(name, 0, 0, tw, th, -1.0, False)

        ctmpl = self._coarse[name]
        if ctmpl.shape[0] > coarse_img.shape[0] or ctmpl.shape[1] > coarse_img.shape[1]:
            return ROIMatch(name, 0, 0, tw, th, -1.0, False)

        res = cv2.matchTemplate(coarse_img, ctmpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=-1.0)
        for rx1, ry1, rx2, ry2 in exclude or ():
            # drop candidate positions whose ROI would overlap an already-claimed area
            xs = np.arange(res.shape[1]) * s
            ys = np.arange(res.shape[0]) * s
            ox = np.clip(np.minimum(xs + tw, rx2) - np.maximum(xs, rx1), 0, None)
            oy = np.clip(np.minimum(ys + th, ry2) - np.maximum(ys, ry1), 0, None)
            overlap = np.outer(oy, ox) / min(tw * th, (rx2 - rx1) * (ry2 - ry1))
            res[overlap > self.MAX_OVERLAP] = -1.0
        _, _, _, (cx, cy) = cv2.minMaxLoc(res)
        gx, gy = cx * s, cy * s

        # Refine at full resolution around the coarse peak.
        r = self.REFINE_RADIUS
        x0, y0 = max(0, gx - r), max(0, gy - r)
        x1, y1 = min(W, gx + tw + r), min(H, gy + th + r)
        window = gray[y0:y1, x0:x1]
        fine = np.nan_to_num(cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED), nan=-1.0)
        _, score, _, (fx, fy) = cv2.minMaxLoc(fine)

        return ROIMatch(name, x0 + fx, y0 + fy, tw, th, float(score), float(score) >= self.min_score)


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image as uint8 grayscale (what the classifiers are fed)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def load_roi_sizes(config_dir: str | Path) -> Dict[str, tuple[int, int]]:
    """(w, h) per ROI from roi_coord_config.json."""
    with open(Path(config_dir) / "roi_coord_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: (int(v["w"]), int(v["h"])) for k, v in cfg.items() if k in ROI_NAMES}
