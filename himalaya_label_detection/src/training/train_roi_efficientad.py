"""
train_roi_efficientad.py
────────────────────────
Per-ROI anomaly model training using pre-cropped grayscale ROI folders.

Assumes the 4 ROI folders already exist on the remote PC in the layout:
    <data_root>/
        <roi_folder_1>/
            good/    ← grayscale good crops (training data)
            bad/     ← grayscale bad crops  (threshold calibration only)
        <roi_folder_2>/
            good/
            bad/
        ...

Supports both EfficientAD (default, recommended) and PatchCore as the model.
EfficientAD trains in ~10-15 min per ROI on an A500 GPU.
PatchCore is faster to "train" (single forward pass) but needs more RAM.

Usage:
    # Train EfficientAD on all ROI folders found under data/rois/
    python scripts/train_all_rois.py

    # Or import and call directly:
    from src.training.train_roi_efficientad import train_roi_model
    train_roi_model("data/rois/ROI_LOGO", "models/ROI_LOGO", model_type="efficientad")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _count_images(folder: Path) -> int:
    """Count image files in a folder."""
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def verify_roi_folder(roi_dir: Path) -> dict:
    """
    Verify that an ROI folder has the expected good/ and bad/ subfolders
    with at least some images.

    Returns:
        dict with keys: ok (bool), n_good (int), n_bad (int), errors (list[str])
    """
    errors = []
    good_dir = roi_dir / "good"
    bad_dir  = roi_dir / "bad"

    if not good_dir.exists():
        errors.append(f"Missing 'good/' subfolder in {roi_dir}")
    if not bad_dir.exists():
        errors.append(f"Missing 'bad/' subfolder in {roi_dir} — needed for threshold calibration")

    n_good = _count_images(good_dir)
    n_bad  = _count_images(bad_dir)

    if n_good == 0:
        errors.append(f"No images found in {good_dir}")
    if n_bad == 0:
        errors.append(f"No images found in {bad_dir} — threshold calibration will be skipped")

    return {
        "ok":     len([e for e in errors if "good" in e]) == 0,
        "n_good": n_good,
        "n_bad":  n_bad,
        "errors": errors,
    }


def train_roi_model(
    roi_data_dir: str | Path,
    output_dir: str | Path,
    model_type: Literal["efficientad", "patchcore"] = "efficientad",
    image_size: tuple[int, int] = (256, 256),
    max_epochs: int = 100,
    train_batch_size: int = 8,
    num_workers: int = 4,
    model_size: str = "small",          # efficientad only: "small" | "medium"
    coreset_sampling_ratio: float = 0.1, # patchcore only
    num_neighbors: int = 9,              # patchcore only
) -> Path:
    """
    Train an anomaly detection model on one ROI folder's good images.

    The model is trained ONLY on good images — bad images are never used
    for training. Bad images are only needed for threshold calibration
    (run calibrate_roi_thresholds.py after training).

    Args:
        roi_data_dir:           Path to the ROI folder containing good/ subdir.
        output_dir:             Directory to save the trained model checkpoint.
        model_type:             "efficientad" (recommended) or "patchcore".
        image_size:             (H, W) to resize crops to during training.
        max_epochs:             Training epochs. EfficientAD: 100. PatchCore: 1.
        train_batch_size:       Batch size (lower if GPU OOM).
        num_workers:            DataLoader workers (0 on Windows if issues).
        model_size:             EfficientAD model size: "small" (faster) or "medium".
        coreset_sampling_ratio: PatchCore coreset fraction (0.1 = 10%).
        num_neighbors:          PatchCore k-NN neighbours.

    Returns:
        Path to the saved checkpoint directory.
    """
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
    except ImportError as exc:
        raise ImportError(
            "anomalib is not installed.\n"
            "Install with:  pip install anomalib==1.1.0"
        ) from exc

    roi_data_dir = Path(roi_data_dir)
    output_dir   = Path(output_dir)
    roi_name     = roi_data_dir.name

    # ── Verify folder structure ───────────────────────────────────────────────
    check = verify_roi_folder(roi_data_dir)
    for err in check["errors"]:
        print(f"  [WARN] {err}")
    if not check["ok"]:
        raise FileNotFoundError(
            f"ROI folder '{roi_data_dir}' is missing a valid good/ directory.\n"
            "Make sure the pre-cropped ROI folders are present."
        )
    print(f"\n[{roi_name}] Good crops: {check['n_good']}  |  Bad crops: {check['n_bad']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build anomalib datamodule ─────────────────────────────────────────────
    # anomalib Folder expects:
    #   root = parent of good/
    #   normal_dir = "good"   (relative to root)
    # bad/ is NOT passed here — it is only used during calibration, not training.
    datamodule = Folder(
        name=roi_name,
        root=str(roi_data_dir),
        normal_dir="good",
        image_size=image_size,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        num_workers=num_workers,
        # Grayscale crops — anomalib handles single-channel inputs
    )

    # ── Build model ───────────────────────────────────────────────────────────
    if model_type == "efficientad":
        from anomalib.models import EfficientAd
        model = EfficientAd(model_size=model_size)
        epochs = max_epochs
        print(f"[{roi_name}] Model: EfficientAD-{model_size.upper()}  |  Epochs: {epochs}")

    elif model_type == "patchcore":
        from anomalib.models import Patchcore
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        epochs = 1  # PatchCore: single forward pass to build memory bank
        print(f"[{roi_name}] Model: PatchCore  |  Coreset ratio: {coreset_sampling_ratio}")

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'efficientad' or 'patchcore'.")

    # ── Train ─────────────────────────────────────────────────────────────────
    engine = Engine(
        max_epochs=epochs,
        accelerator="auto",       # GPU if available, CPU fallback
        devices=1,
        default_root_dir=str(output_dir),
    )

    print(f"[{roi_name}] Starting training → {output_dir}")
    engine.fit(model=model, datamodule=datamodule)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "roi_name":    roi_name,
        "model_type":  model_type,
        "image_size":  list(image_size),
        "n_good":      check["n_good"],
        "n_bad":       check["n_bad"],
        "model_size":  model_size if model_type == "efficientad" else None,
        "epochs":      epochs,
        "data_dir":    str(roi_data_dir.resolve()),
        "output_dir":  str(output_dir.resolve()),
    }
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{roi_name}] ✓ Training complete  →  {output_dir}")
    return output_dir
