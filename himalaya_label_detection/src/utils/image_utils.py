"""
image_utils.py
Shared helper functions for loading, resizing, and saving images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_image(path: str | Path,
               target_size: Optional[Tuple[int, int]] = None,
               grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path:        Path to the image file.
        target_size: Optional (width, height) to resize to.
        grayscale:   If True, load as single-channel grayscale.

    Returns:
        numpy array in BGR (or GRAY if grayscale=True).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)

    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    """
    Save an image to disk, creating parent directories as needed.

    Args:
        image: numpy array (BGR or GRAY).
        path:  Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def list_images(directory: str | Path,
                extensions: Optional[set] = None) -> list[Path]:
    """
    Return a sorted list of image file paths in a directory.

    Args:
        directory:  Directory to scan.
        extensions: Set of lowercase extensions to include (e.g. {'.jpg', '.png'}).
                    Defaults to SUPPORTED_EXTENSIONS.

    Returns:
        Sorted list of Path objects.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in extensions
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale. No-op if already grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to 3-channel BGR. No-op if already BGR."""
    if image.ndim == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def side_by_side(img_a: np.ndarray, img_b: np.ndarray,
                 label_a: str = "A", label_b: str = "B") -> np.ndarray:
    """
    Stack two images side by side for visual comparison.
    Both images are converted to BGR and padded to the same height.
    """
    a = ensure_bgr(img_a.copy())
    b = ensure_bgr(img_b.copy())

    h = max(a.shape[0], b.shape[0])
    if a.shape[0] < h:
        a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if b.shape[0] < h:
        b = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    canvas = np.hstack([a, b])

    # Draw labels
    cv2.putText(canvas, label_a, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, label_b, (a.shape[1] + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return canvas


def diff_map(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Compute absolute difference between two same-size images.
    Returns a color-mapped heatmap (BGR) for visual inspection.
    """
    a = to_grayscale(img_a)
    b = to_grayscale(img_b)
    diff = cv2.absdiff(a, b)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
