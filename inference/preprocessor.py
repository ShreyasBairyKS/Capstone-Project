"""
inference/preprocessor.py — Shared image preprocessing utilities.

Used by both YOLOv11Detector and EfficientViTClassifier.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    pad_colour: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image to target_size×target_size preserving aspect ratio via padding.

    Returns:
        padded:  uint8 BGR image of shape (target_size, target_size, 3)
        ratio:   scale factor applied to original dimensions
        padding: (pad_left, pad_top) pixel offsets — needed to map coords back
    """
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, target_size - new_h - pad_top,
        pad_left, target_size - new_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=pad_colour,
    )
    return padded, ratio, (pad_left, pad_top)


def bgr_to_nchw_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 (H,W,3) → float32 NCHW (1,3,H,W) normalised [0,1].
    Colour channel converts BGR → RGB.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    return normalised.transpose(2, 0, 1)[np.newaxis, :]  # HWC → NCHW batch


def extract_crop(
    frame: np.ndarray,
    x1_norm: float,
    y1_norm: float,
    x2_norm: float,
    y2_norm: float,
    target_size: int = 224,
    pad_ratio: float = 0.05,
) -> np.ndarray:
    """
    Extract and resize a bounding-box crop from the original frame.

    Coordinates are normalised [0, 1]. Adds a small padding border to give
    the classifier slightly more context around the defect.

    Returns: BGR uint8 (target_size, target_size, 3)
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(x1_norm * w) - int(pad_ratio * w))
    y1 = max(0, int(y1_norm * h) - int(pad_ratio * h))
    x2 = min(w, int(x2_norm * w) + int(pad_ratio * w))
    y2 = min(h, int(y2_norm * h) + int(pad_ratio * h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame  # fallback — degenerate box
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def prepare_classifier_input(crop: np.ndarray) -> np.ndarray:
    """
    Prepare a 224×224 BGR crop for EfficientViT input.
    Returns float32 NCHW (1,3,224,224) normalised with ImageNet mean/std.
    """
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalised = (rgb - mean) / std
    return normalised.transpose(2, 0, 1)[np.newaxis, :]  # HWC → NCHW


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logits array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
