"""
make_icon.py
─────────────────────────────────────────────────────────────────────────────
Draws the app icon (the page's mountain mark on a blue tile) and writes desktop/himalaya.ico.
Needs only numpy and OpenCV, which the pipeline already depends on:

    python himalaya_label_detection/template_roi_pipeline/desktop/make_icon.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

SIZES = (256, 128, 64, 48, 32, 24, 16)
OUT = Path(__file__).with_name("himalaya.ico")


def draw(size: int) -> np.ndarray:
    """BGRA tile, drawn at 4x and shrunk so the edges are smooth."""
    s = size * 4
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
    t = np.clip((xx + yy) / 2, 0, 1)[..., None]                       # diagonal gradient, light blue to blue
    top, bottom = np.array([255, 180, 114], np.float32), np.array([232, 96, 33], np.float32)   # BGR
    rgb = top * (1 - t) + bottom * t
    # rounded-square mask
    r, m = 0.22 * s, 0.04 * s
    cx = np.clip(np.abs(xx * s - s / 2) - (s / 2 - m - r), 0, None)
    cy = np.clip(np.abs(yy * s - s / 2) - (s / 2 - m - r), 0, None)
    alpha = (np.hypot(cx, cy) <= r).astype(np.float32)
    img = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    # the mark: the page's polygon "M2 20 9.5 6l4 7 2.5-4L22 20Z" on a 24-unit grid, centred and scaled
    pts = np.array([(2, 20), (9.5, 6), (13.5, 13), (16, 9), (22, 20)], np.float32)
    pts = (pts - [12, 13]) * (s * 0.62 / 20) + s / 2
    cv2.fillPoly(img, [pts.astype(np.int32)], (255, 255, 255, 255), cv2.LINE_AA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def write_ico(path: Path = OUT) -> None:
    pngs = []
    for size in SIZES:
        ok, buf = cv2.imencode(".png", draw(size))
        assert ok
        pngs.append((size, buf.tobytes()))
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries, blobs = b"", b""
    for size, data in pngs:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset + len(blobs))
        blobs += data
    path.write_bytes(header + entries + blobs)


if __name__ == "__main__":
    write_ico()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
