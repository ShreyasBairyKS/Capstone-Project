"""
Fill Level Visualizer and Measurement Overlay Renderer.
Generates dynamic on-demand visual annotations for live streaming and dashboard inspection.
Preserves raw unblemished frames for database persistence and machine learning training.
"""
import base64
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from pipeline.fill_level.decision.schemas import FillVerdict, FillInspectionDecision
from .schemas import FillVisualizerConfig


# ── Color Palette (BGR) ──
CYAN_GLOW   = (243, 218, 0)     # Air-liquid meniscus line
CYAN_FAINT  = (200, 160, 0)     # Bottle bounding box
GREEN_PASS  = (80, 220, 80)     # Certified nominal pass
RED_REJECT  = (48, 59, 255)     # Underfill / defect reject
ORANGE_WARN = (0, 140, 255)     # Overfill alert
AMBER_UNC   = (0, 190, 255)     # Borderline / review
GRAY_TEXT   = (180, 190, 200)   # Auxiliary labels
DARK_BG     = (15, 12, 10)      # Translucent badge background
WHITE       = (255, 255, 255)


class FillVisualOverlayRenderer:
    """
    High-performance, non-destructive visual telemetry renderer.
    Renders on an in-memory buffer without modifying the raw image,
    ensuring saved audit images remain 100% clean and training-ready.
    """

    def __init__(self, config: Optional[FillVisualizerConfig] = None):
        self.config = config or FillVisualizerConfig()

    def render(
        self,
        frame: np.ndarray,
        decision: FillInspectionDecision,
        overlay_alpha: float = 0.85
    ) -> np.ndarray:
        """
        Renders non-destructive telemetry overlay onto a copy of the input frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        bottle = decision.bottle
        meniscus = decision.meniscus

        # 1. Bottle Outline
        if self.config.draw_bottle_box and bottle is not None:
            bx1, by1, bx2, by2 = bottle.box
            cv2.rectangle(out, (bx1, by1), (bx2, by2), CYAN_FAINT, 2, cv2.LINE_AA)
            cv2.putText(
                out, f"BOTTLE {bottle.confidence*100:.0f}%",
                (bx1 + 6, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN_FAINT, 1, cv2.LINE_AA
            )

        # 2. Calibrated Vertical Fill Ruler
        if self.config.draw_ruler and bottle is not None:
            self._draw_ruler(out, bottle.box, meniscus.water_y if meniscus else None)

        # 3. Glowing Meniscus Line
        if self.config.draw_meniscus_line and meniscus is not None:
            self._draw_meniscus(out, meniscus.box, meniscus.water_y)

        # 4. Status Telemetry Banner
        if self.config.draw_status_banner:
            self._draw_status_banner(out, decision)

        return out

    def _draw_meniscus(self, img: np.ndarray, water_box: list[int], water_y: int):
        """Draws vibrant horizontal air-liquid line with anti-aliased glow."""
        wx1, wy1, wx2, wy2 = water_box
        h, w = img.shape[:2]

        # Extend meniscus line across bottle width with subtle padding
        x1 = max(0, wx1 - 4)
        x2 = min(w - 1, wx2 + 4)

        # Semi-transparent outer glow band
        glow_h = 3
        y1_glow = max(0, water_y - glow_h)
        y2_glow = min(h - 1, water_y + glow_h)
        sub = img[y1_glow:y2_glow + 1, x1:x2 + 1]
        if sub.size > 0:
            glow_color = np.full(sub.shape, CYAN_GLOW, dtype=np.uint8)
            blended = cv2.addWeighted(sub, 0.45, glow_color, 0.55, 0)
            img[y1_glow:y2_glow + 1, x1:x2 + 1] = blended

        # Solid core line
        cv2.line(img, (x1, water_y), (x2, water_y), CYAN_GLOW, 2, cv2.LINE_AA)

        # Level indicator pill on the meniscus line
        tag = "LIQUID LEVEL"
        cv2.putText(
            img, tag, (max(0, x1 - 70), water_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN_GLOW, 1, cv2.LINE_AA
        )

    def _draw_ruler(self, img: np.ndarray, bottle_box: list[int], water_y: Optional[int]):
        """Draws calibrated vertical measurement ruler along bottle profile edge."""
        bx1, by1, bx2, by2 = bottle_box
        h, w = img.shape[:2]
        bh = by2 - by1

        # Corrected bounds (matches compute_fill_ratio)
        by1_corr = by1 + int(bh * 0.03)
        by2_corr = by2 - int(bh * 0.02)
        h_corr = max(by2_corr - by1_corr, 1)

        # Place ruler on right side of bottle if space permits, else inside edge
        rx = min(w - 20, bx2 + 10)
        tick_len = 8

        # Vertical rail
        cv2.line(img, (rx, by1_corr), (rx, by2_corr), (120, 130, 140), 1, cv2.LINE_AA)

        # Ticks: 0% (base), 40% (min), 50%, 75%, 85% (max), 100% (top)
        ticks = [
            (0.00, "0%", (160, 160, 160)),
            (self.config.underfill_thresh, f"{int(self.config.underfill_thresh*100)}% MIN", RED_REJECT),
            (0.50, "50%", (160, 160, 160)),
            (0.75, "75%", (160, 160, 160)),
            (self.config.overfill_thresh, f"{int(self.config.overfill_thresh*100)}% MAX", ORANGE_WARN),
            (1.00, "100%", (160, 160, 160)),
        ]

        for ratio_val, label, color in ticks:
            ty = int(by2_corr - (h_corr * ratio_val))
            if 0 <= ty < h:
                # Horizontal tick mark
                cv2.line(img, (rx, ty), (rx + tick_len, ty), color, 1 if color == (160, 160, 160) else 2, cv2.LINE_AA)
                cv2.putText(
                    img, label, (rx + tick_len + 3, ty + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA
                )

        # Highlight current liquid level position on ruler
        if water_y is not None and 0 <= water_y < h:
            cv2.circle(img, (rx, water_y), 4, CYAN_GLOW, -1, cv2.LINE_AA)
            cv2.circle(img, (rx, water_y), 5, WHITE, 1, cv2.LINE_AA)

    def _draw_status_banner(self, img: np.ndarray, decision: FillInspectionDecision):
        """Draws industrial translucent status badge on frame."""
        h, w = img.shape[:2]
        verdict = decision.verdict
        ratio = decision.fill_ratio

        # Determine theme color & badge text
        if verdict == FillVerdict.PASS:
            color = GREEN_PASS
            title = f"PASS - NOMINAL VOLUME ({ratio*100:.1f}%)" if ratio is not None else "PASS"
        elif verdict == FillVerdict.REJECT:
            if decision.fill_status == "underfill":
                color = RED_REJECT
                title = f"REJECT - UNDERFILL ({ratio*100:.1f}%)" if ratio is not None else "REJECT - UNDERFILL"
            elif decision.fill_status == "overfill":
                color = ORANGE_WARN
                title = f"REJECT - OVERFILL ({ratio*100:.1f}%)" if ratio is not None else "REJECT - OVERFILL"
            elif decision.fill_status == "no_water":
                color = RED_REJECT
                title = "REJECT - NO LIQUID DETECTED"
            else:
                color = RED_REJECT
                title = "REJECT - FILL ANOMALY"
        elif verdict == FillVerdict.UNCERTAIN:
            color = AMBER_UNC
            title = f"REVIEW - BORDERLINE FILL ({ratio*100:.1f}%)" if ratio is not None else "REVIEW - UNCERTAIN"
        else:
            color = GRAY_TEXT
            title = "STANDBY - NO CONTAINER"

        # Position banner at bottom left
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
        tw, th = text_size

        pad_x, pad_y = 12, 8
        bx1 = 16
        by2 = h - 16
        by1 = by2 - (th + pad_y * 2)
        bx2 = bx1 + (tw + pad_x * 2)

        # Translucent pill background
        sub = img[by1:by2, bx1:bx2]
        if sub.size > 0:
            bg_rect = np.full(sub.shape, DARK_BG, dtype=np.uint8)
            img[by1:by2, bx1:bx2] = cv2.addWeighted(sub, 0.25, bg_rect, 0.75, 0)
            # Colored border
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

        # Badge text
        cv2.putText(
            img, title, (bx1 + pad_x, by1 + pad_y + th - 2),
            font, font_scale, color, thickness, cv2.LINE_AA
        )

    def encode_base64(self, frame: np.ndarray, max_dim: int = 1280) -> str:
        """Encodes frame to Base64 JPEG string for real-time WebSocket transmission."""
        h, w = frame.shape[:2]
        disp_frame = frame
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            disp_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        success, encoded = cv2.imencode(
            ".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not success:
            raise RuntimeError("Failed to JPEG encode frame for base64 stream.")

        b64_str = base64.b64encode(encoded).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
