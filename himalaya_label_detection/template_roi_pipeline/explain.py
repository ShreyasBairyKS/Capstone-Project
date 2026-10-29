"""
explain.py
─────────────────────────────────────────────────────────────────────────────
Turns an InspectionResult into evidence a person can read:

  * per-ROI heatmap overlays of the EfficientAD anomaly map (where the model
    sees something unusual), with the strongest regions boxed and numbered
  * the same regions projected back onto the full scan
  * plain-language reasons for every ROI and for the final label

How to read the maps
    The ONNX model returns an anomaly map for each ROI crop. The ROI score is the
    mean of that map and the ROI fails when the score is above its threshold, so a
    pixel is "hot" when its value is above the threshold: such pixels pull the
    score towards a FAIL. A passing ROI can still show a few hot pixels; it only
    fails when they raise the average past the limit.

Everything here is presentation only: it never changes a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .inspector import InspectionResult, ROIVerdict

STATUS_BGR = {"PASS": (74, 160, 56), "FAIL": (62, 62, 214), "MISSING": (30, 150, 245)}
MAX_REGIONS = 3
MIN_REGION_AREA_PCT = 0.2      # ignore specks smaller than this share of the ROI
DISPLAY_WIDTH = 1600           # width of crops / overlays sent to the UI (the widest upright crop is ~1780 px; the page's magnifier needs the detail)


# ── Heat scaling ──────────────────────────────────────────────────────────────

MIN_HEAT_SPAN = 0.25           # colour range never shrinks below this, so clean crops stay faint

def heat_above_threshold(amap: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """
    Map values to 0..1 where 0 = at/below the ROI threshold and 1 = the map's hottest pixels.
    Real defects produce spikes far above the threshold (a rejected crop reaches 9+ against a limit of
    0.23) while clean crops top out near it, so the range follows the map but never drops below
    MIN_HEAT_SPAN: noise in a clean crop then stays faint instead of being stretched to full colour.
    """
    top = float(np.percentile(amap, 99.8))
    span = max(top - threshold, MIN_HEAT_SPAN)
    return np.clip((amap - threshold) / span, 0.0, 1.0).astype(np.float32), span


def heat_full(amap: np.ndarray) -> np.ndarray:
    """Plain min-max normalisation of the whole map (relative view, ignores the threshold)."""
    lo, hi = float(amap.min()), float(amap.max())
    return ((amap - lo) / max(hi - lo, 1e-6)).astype(np.float32)


# ── Regions ───────────────────────────────────────────────────────────────────

@dataclass
class HotRegion:
    index: int
    x0: float               # all coordinates normalised 0..1 on the upright crop
    y0: float
    x1: float
    y1: float
    area_pct: float         # share of the ROI covered by the region
    peak: float             # highest map value inside the region
    where: str
    accounts_for: float = 0.0     # how much of the ROI score this region adds (score units)
    score_without: float = 0.0    # the ROI score if this region were as quiet as the rest of the crop

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "box": [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)],
            "area_pct": round(self.area_pct, 2),
            "peak": round(self.peak, 4),
            "where": self.where,
            "accounts_for": round(self.accounts_for, 6),
            "score_without": round(self.score_without, 6),
        }


def describe_position(cx: float, cy: float) -> str:
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    v = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    if h == "centre" and v == "middle":
        return "centre"
    if h == "centre":
        return v
    if v == "middle":
        return h
    return f"{v}-{h}"


CORE_HEAT = 0.35               # a region's core: pixels at least this far up the colour range
SHOULDER_HEAT = 0.08           # ...and the surrounding pixels still clearly above the threshold


def find_hot_regions(amap: np.ndarray, threshold: float) -> List[HotRegion]:
    """
    Up to MAX_REGIONS connected areas where the map is clearly above the threshold, strongest first.

    Each region also carries a counterfactual: the ROI score if the region (core plus its shoulder) were
    replaced by the median of the crop. Because score = mean(map), that tells whether the region alone is
    what pushes the ROI over its limit.
    """
    heat, _ = heat_above_threshold(amap, threshold)
    core = (heat >= CORE_HEAT).astype(np.uint8)
    if not core.any():
        return []
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, shoulder_labels = cv2.connectedComponents((heat >= SHOULDER_HEAT).astype(np.uint8), connectivity=8)
    h, w = amap.shape
    baseline = float(np.median(amap))
    total = float(amap.sum())

    # One region per shoulder component (cores that merge into the same shoulder are one anomaly,
    # otherwise two boxes would claim the same pixels and the counterfactual would count them twice).
    ids = shoulder_labels[core > 0]
    found = []
    for sid in np.unique(ids[ids > 0]):
        shoulder = shoulder_labels == sid
        cores = shoulder & (core > 0)
        area_pct = 100.0 * float(cores.sum()) / (h * w)
        if area_pct < MIN_REGION_AREA_PCT:
            continue
        ys, xs = np.where(cores)
        removed = float(np.clip(amap[shoulder] - baseline, 0, None).sum())
        found.append((removed / amap.size, xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h,
                      area_pct, float(amap[cores].max()), (total - removed) / amap.size))
    found.sort(reverse=True)                      # biggest contribution to the score first
    regions = []
    for rank, (accounts, x0, y0, x1, y1, area_pct, peak, without) in enumerate(found[:MAX_REGIONS], start=1):
        regions.append(HotRegion(rank, x0, y0, x1, y1, area_pct, peak,
                                 describe_position((x0 + x1) / 2, (y0 + y1) / 2), accounts, without))
    return regions


# ── Image rendering ───────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _fit_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] <= width:
        return img
    scale = width / img.shape[1]
    return cv2.resize(img, (width, max(1, round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def render_crop(crop: np.ndarray, width: int = DISPLAY_WIDTH) -> np.ndarray:
    return _fit_width(_to_bgr(crop), width)


def render_overlay(
    crop: np.ndarray,
    amap: np.ndarray,
    threshold: float,
    mode: str = "above",
    regions: Optional[List[HotRegion]] = None,
    width: int = DISPLAY_WIDTH,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    mode "above": only pixels above the ROI threshold are tinted (transparent elsewhere).
    mode "full":  the whole map, min-max normalised, as a heatmap over the crop.
    """
    base = render_crop(crop, width)
    h, w = base.shape[:2]
    if mode == "above":
        heat, _ = heat_above_threshold(amap, threshold)
        a = alpha * np.clip(cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None]
    elif mode == "full":
        heat = heat_full(amap)
        a = np.full((h, w, 1), 0.55, np.float32)
    else:
        raise ValueError("mode must be 'above' or 'full'")
    colour = cv2.applyColorMap((np.clip(cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1) * 255)
                               .astype(np.uint8), cv2.COLORMAP_INFERNO if mode == "above" else cv2.COLORMAP_TURBO)
    out = (base.astype(np.float32) * (1 - a) + colour.astype(np.float32) * a).astype(np.uint8)
    for r in regions or ():
        p0, p1 = (round(r.x0 * w), round(r.y0 * h)), (round(r.x1 * w), round(r.y1 * h))
        cv2.rectangle(out, p0, p1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(out, p0, p1, (40, 40, 40), 1, cv2.LINE_AA)
        _chip(out, str(r.index), (p0[0], max(p0[1], 0)), (255, 255, 255), (40, 40, 40), scale=0.8)
    return out


def _chip(img: np.ndarray, text: str, org: tuple[int, int], bg: tuple, fg: tuple, scale: float = 0.5) -> None:
    """Text on a filled rectangle, anchored at its top-left corner (clamped inside the image)."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = max(2, round(scale * 6))
    x = int(np.clip(org[0], 0, max(0, img.shape[1] - tw - 2 * pad)))
    y = int(np.clip(org[1], 0, max(0, img.shape[0] - th - base - 2 * pad)))
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + base + 2 * pad), bg, -1)
    cv2.putText(img, text, (x + pad, y + pad + th), font, scale, fg, thick, cv2.LINE_AA)


def upright_to_scan(box: tuple, roi_xywh: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """
    Normalised box on the upright (rotated 90 deg clockwise) crop -> scan pixel box (x0, y0, x1, y1).
    A clockwise rotation sends scan-crop pixel (x, y) to upright (h - 1 - y, x), hence the inverse below.
    """
    nx0, ny0, nx1, ny1 = box
    rx, ry, rw, rh = roi_xywh
    return (rx + ny0 * rw, ry + (1 - nx1) * rh, rx + ny1 * rw, ry + (1 - nx0) * rh)


def render_overview(image: np.ndarray, result: InspectionResult, regions: Dict[str, List[HotRegion]],
                    scale: float = 0.25, margin: int = 120) -> tuple[np.ndarray, Dict[str, List[float]]]:
    """
    The scan turned upright (like the crops) and trimmed to the label area, every ROI boxed in its status
    colour and hot regions outlined. Also returns each ROI's box as fractions of the returned image
    (x0, y0, x1, y1) so a UI can overlay click targets.
    """
    located = [v.match for v in result.rois.values() if v.status != "MISSING"]
    left = max(0, min(m.x for m in located) - margin) if located else 0
    right = min(image.shape[1], max(m.x + m.w for m in located) + margin) if located else image.shape[1]

    small = cv2.resize(_to_bgr(image), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = small[:, round(left * scale):round(right * scale)]
    hs = small.shape[0]
    out = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
    oh, ow = out.shape[:2]

    def to_view(x0, y0, x1, y1):
        """scan pixel box -> box in the rotated overview (a clockwise turn sends (x, y) to (hs - y, x))."""
        sx0, sx1, sy0, sy1 = (x0 - left) * scale, (x1 - left) * scale, y0 * scale, y1 * scale
        return hs - sy1, sx0, hs - sy0, sx1

    fractions: Dict[str, List[float]] = {}
    lost = 0
    for name, v in result.rois.items():
        colour, m = STATUS_BGR[v.status], v.match
        if v.status == "MISSING":
            # the "best guess" position of a ROI that was not found means nothing, so it gets a note, not a box
            _chip(out, f"{name} not located", (4, 4 + lost * 30), colour, (255, 255, 255), scale=0.6)
            lost += 1
            continue
        bx0, by0, bx1, by1 = to_view(*m.xyxy)
        fractions[name] = [round(max(0.0, bx0 / ow), 5), round(max(0.0, by0 / oh), 5),
                           round(min(1.0, bx1 / ow), 5), round(min(1.0, by1 / oh), 5)]
        cv2.rectangle(out, (round(bx0), round(by0)), (round(bx1), round(by1)), colour,
                      3 if v.status != "MISSING" else 1, cv2.LINE_AA)
        for r in regions.get(name, ()):
            hx0, hy0, hx1, hy1 = to_view(*upright_to_scan((r.x0, r.y0, r.x1, r.y1), (m.x, m.y, m.w, m.h)))
            cv2.rectangle(out, (round(hx0), round(hy0)), (round(hx1), round(hy1)), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(out, (round(hx0), round(hy0)), (round(hx1), round(hy1)), colour, 1, cv2.LINE_AA)
        _chip(out, f"{name} {v.status}", (round(bx0) + 3, round(by0) + 3), colour, (255, 255, 255), scale=0.6)
    return out, fractions


# ── Text ──────────────────────────────────────────────────────────────────────

def explain_roi(v: ROIVerdict, regions: List[HotRegion]) -> dict:
    """Numbers + plain-language summary for one ROI."""
    base = {
        "roi": v.roi_name,
        "status": v.status,
        "threshold": None if v.threshold is None else round(v.threshold, 6),
        "match_score": round(v.match.score, 4),
        "box": [v.match.x, v.match.y, v.match.w, v.match.h],
    }
    if v.status == "MISSING":
        base.update(
            score=None, margin=None, gauge=None, area_over_pct=None, regions=[],
            summary=(f"{v.roi_name} could not be located: the best template match was {v.match.score:.2f}, "
                     "below the minimum needed to trust it. The region may be absent, cut off or badly "
                     "misaligned, so it cannot be verified and the label fails."),
        )
        return base

    amap, thr, score = v.anomaly_map, v.threshold, v.score
    margin = score - thr
    area_over = 100.0 * float((amap > thr).mean())
    reach = max(1.6 * abs(margin), 0.02 + 0.5 * abs(thr))
    base.update(
        score=round(score, 6),
        margin=round(margin, 6),
        gauge={"lo": round(thr - reach, 6), "hi": round(thr + reach, 6)},
        area_over_pct=round(area_over, 2),
        regions=[r.to_dict() for r in regions],
    )
    top = regions[0] if regions else None
    if v.status == "FAIL":
        text = (f"Anomaly score {score:.4f} is above the limit of {thr:.4f} by {margin:.4f}; "
                f"{area_over:.0f}% of the ROI is scored above the limit.")
        if top:
            text += (f" Region 1 ({top.where}, {top.area_pct:.1f}% of the area) adds {top.accounts_for:.4f} to the score. "
                     f"Without it the ROI would score {top.score_without:.4f}, ")
            text += ("under the limit, so that region is what causes the rejection."
                     if top.score_without <= thr else "still over the limit, so the anomaly is not confined to one place.")
    else:
        text = f"Anomaly score {score:.4f} is {abs(margin):.4f} below the limit of {thr:.4f}."
        text += (f" The most active area ({top.where}) adds only {top.accounts_for:.4f} to the score."
                 if top else " No area of the ROI rises clearly above the limit.")
    base["summary"] = text
    return base


def explain_label(result: InspectionResult, rois: Dict[str, dict]) -> dict:
    failing = [n for n, v in result.rois.items() if v.status == "FAIL"]
    missing = [n for n, v in result.rois.items() if v.status == "MISSING"]
    if result.verdict == "PASS":
        headline = "All four regions are within their limits."
    else:
        parts = []
        if failing:
            parts.append("anomaly detected in " + ", ".join(failing))
        if missing:
            parts.append(", ".join(missing) + " could not be located")
        headline = "Label rejected: " + "; ".join(parts) + "."
    # most informative first: biggest overshoot
    ranked = sorted((r for r in rois.values() if r["margin"] is not None), key=lambda r: -r["margin"])
    return {
        "headline": headline,
        "reasons": list(result.fail_reasons),
        "closest_to_limit": ranked[0]["roi"] if ranked else None,
    }
