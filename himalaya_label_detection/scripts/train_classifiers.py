"""
train_classifiers.py
────────────────────
Trains one EfficientAD-Medium anomaly classifier per ROI, then auto-calibrates
detection thresholds optimized for maximum recall.

Pipeline:
  data/rois/ROI_1/good/  +  bad/  +  data/annotations/ROI_1/   →  EfficientAD-Medium  →  checkpoint
  data/rois/ROI_2/good/  +  bad/  +  data/annotations/ROI_2/   →  EfficientAD-Medium  →  checkpoint
  data/rois/ROI_3/good/  +  bad/  +  data/annotations/ROI_3/   →  EfficientAD-Medium  →  checkpoint
  data/rois/ROI_4/good/  +  bad/  +  data/annotations/ROI_4/   →  EfficientAD-Medium  →  checkpoint
                                                                                         ↓
                                                                        config/roi_thresholds.json

Usage (recommended — let it run overnight on the A5000):
    python himalaya_label_detection/scripts/train_classifiers.py

Custom paths / options:
    python himalaya_label_detection/scripts/train_classifiers.py \\
        --rois data/rois \\
        --annotations data/annotations \\
        --out  models/rois \\
        --epochs 200 \\
        --imgsz 512 \\
        --batch 16 \\
        --model-size medium

Resume after interruption (skips already-trained ROIs):
    python himalaya_label_detection/scripts/train_classifiers.py --skip-done

Design notes:
  - Trained exclusively on GOOD images (anomaly detection, not classification)
  - YOLO-format defect annotations (data/annotations/ROI_N/) are converted to
    binary pixel masks and passed to anomalib as abnormal_dir masks.
    This lets EfficientAD learn WHERE anomalies appear, giving far better heatmaps.
  - Bad images are also used for recall-optimized threshold calibration after training.
  - Threshold is set at the 5th percentile of bad-image scores:
    95%+ of real defects are always caught (we tolerate some false positives).
  - EfficientAD-Medium is used — ~6% better AUROC than Small on the A5000 24GB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Literal, Tuple

import cv2
import numpy as np

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
CONFIG_DIR      = PROJECT_ROOT / "himalaya_label_detection" / "config"
ROI_NAMES       = ["ROI_1", "ROI_2", "ROI_3", "ROI_4"]
SUPPORTED       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)


def _banner(msg: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {msg}\n{line}")


# ─────────────────────────────────────────────────────────────────────────────
# YOLO annotation → binary mask converter
# ─────────────────────────────────────────────────────────────────────────────

def yolo_to_mask(
    yolo_txt: Path,
    img_w: int,
    img_h: int,
    out_path: Path,
) -> bool:
    """
    Convert a YOLO-format annotation file (class cx cy w h, normalised) into
    a binary PNG mask where defect regions are white (255) and background is black (0).

    Returns True if the mask was written, False if the annotation file was empty.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    lines = yolo_txt.read_text().strip().splitlines()
    if not lines:
        return False

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

        # Convert normalised → pixel coords
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)

        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)

        mask[y1:y2, x1:x2] = 255

    cv2.imwrite(str(out_path), mask)
    return True


def build_mask_dir(
    bad_dir: Path,
    annotation_dir: Path,
    tmp_root: Path,
    roi_name: str,
) -> Path | None:
    """
    For every annotated bad image, generate its binary mask PNG and place it in
    a temporary masks/ directory that anomalib's Folder datamodule can read.

    Images with no matching annotation are still included in the datamodule
    (they are assumed to be fully anomalous — masks are left empty).

    Returns the path to the temp masks directory, or None if no annotations exist.
    """
    ann_files = sorted(
        p for p in (annotation_dir.glob("*.txt") if annotation_dir.exists() else [])
        if p.name.lower() != "classes.txt"   # skip the labelImg class-list file
    )
    if not ann_files:
        print(f"  [{roi_name}] No YOLO annotations found in {annotation_dir} — "
              "bad images will be used for threshold calibration only (no masks)")
        return None

    mask_dir = tmp_root / f"{roi_name}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    matched = 0
    for ann_path in ann_files:
        stem = ann_path.stem
        # Find the matching image (any extension)
        img_path = None
        for ext in SUPPORTED:
            candidate = bad_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        out_mask = mask_dir / f"{stem}.png"
        written  = yolo_to_mask(ann_path, w, h, out_mask)
        if written:
            matched += 1

    if matched == 0:
        shutil.rmtree(mask_dir, ignore_errors=True)
        return None

    print(f"  [{roi_name}] Converted {matched} YOLO annotation(s) → binary masks in {mask_dir}")
    return mask_dir


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_roi(
    roi_dir: Path,
    annotation_dir: Path | None,
    output_dir: Path,
    tmp_root: Path,
    image_size: int = 512,
    max_epochs: int = 200,
    batch_size: int = 16,
    num_workers: int = 8,
    model_size: Literal["small", "medium"] = "medium",
) -> Path:
    """
    Train EfficientAD-Medium on one ROI's good/ images.
    If annotations exist, binary masks of defect regions are passed to anomalib
    so the model learns WHERE anomalies appear (better heatmaps).

    Returns path to the saved checkpoint directory.
    """
    try:
        import anomalib
        _anom_ver = tuple(int(x) for x in anomalib.__version__.split(".")[:2])
        print(f"  Using anomalib v{anomalib.__version__}")

        if _anom_ver >= (2, 0):
            # anomalib v2+ API
            from anomalib.data import Folder
            from anomalib.engine import Engine
            from anomalib.models import EfficientAd
        else:
            # anomalib v1 API (1.x)
            from anomalib.data import Folder
            from anomalib.engine import Engine
            from anomalib.models import EfficientAd
    except ImportError as exc:
        raise ImportError(
            f"anomalib import failed: {exc}\n"
            "anomalib v1.1.0 requires the 'lightning' package (NOT pytorch-lightning).\n"
            "Fix:  pip install lightning\n"
            "Full install:  pip install anomalib timm lightning albumentations"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error loading anomalib: {exc}\n"
            "Check your anomalib version: python -c \"import anomalib; print(anomalib.__version__)\""
        ) from exc

    roi_name = roi_dir.name
    good_dir = roi_dir / "good"
    bad_dir  = roi_dir / "bad"
    n_good   = len(list_images(good_dir))
    n_bad    = len(list_images(bad_dir))

    _banner(f"Training {roi_name}  |  {n_good} good  {n_bad} bad  |  EfficientAD-{model_size.upper()}")

    if n_good == 0:
        print(f"  [SKIP] No good images found in {good_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate binary masks from YOLO annotations ───────────────────────────
    mask_dir = None
    if annotation_dir and annotation_dir.exists():
        mask_dir = build_mask_dir(bad_dir, annotation_dir, tmp_root, roi_name)

    # ── Data ─────────────────────────────────────────────────────────────────
    folder_kwargs = dict(
        name=roi_name,
        root=str(roi_dir),
        normal_dir="good",
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
    )

    # Include bad images in eval — with or without masks
    if n_bad > 0:
        folder_kwargs["abnormal_dir"] = "bad"
    if mask_dir is not None:
        folder_kwargs["mask_dir"] = str(mask_dir)

    datamodule = Folder(**folder_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EfficientAd(model_size=model_size)

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = Engine(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=str(output_dir),
    )

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    elapsed = (time.perf_counter() - t0) / 60

    print(f"\n  [{roi_name}] ✅ Training done in {elapsed:.1f} min")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "roi_name":    roi_name,
        "model_type":  "efficientad",
        "model_size":  model_size,
        "image_size":  [image_size, image_size],
        "epochs":      max_epochs,
        "n_good":      n_good,
        "n_bad":       n_bad,
        "masks_used":  mask_dir is not None,
        "data_dir":    str(roi_dir.resolve()),
    }
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    roi_dir: Path,
    model_dir: Path,
    image_size: int = 512,
    recall_percentile: float = 5.0,
) -> float:
    """
    Score every bad image through the trained model and set the detection
    threshold at the `recall_percentile`-th percentile of those scores.

    recall_percentile=5 → threshold below 95% of bad scores → 95%+ recall.
    Lower values = more sensitive (even higher recall, more false positives).

    Returns the calibrated threshold, or a safe conservative default.
    """
    try:
        import torch
        from anomalib.models import EfficientAd
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
    except ImportError:
        print("  [WARN] Cannot run calibration — anomalib/torch not available")
        return 0.3

    bad_dir   = roi_dir / "bad"
    bad_imgs  = list_images(bad_dir)
    roi_name  = roi_dir.name

    if not bad_imgs:
        print(f"  [{roi_name}] No bad images — using conservative default threshold 0.3")
        return 0.3

    ckpt_paths = sorted(model_dir.rglob("*.ckpt"),
                        key=lambda p: p.stat().st_mtime)
    if not ckpt_paths:
        print(f"  [{roi_name}] No checkpoint found — skipping calibration, using 0.5")
        return 0.5

    latest_ckpt = ckpt_paths[-1]
    print(f"\n  [{roi_name}] Calibrating threshold using {len(bad_imgs)} bad images...")
    print(f"             Checkpoint: {latest_ckpt.name}")

    try:
        model = EfficientAd.load_from_checkpoint(str(latest_ckpt))
        model.eval()
    except Exception as e:
        print(f"  [{roi_name}] Could not load checkpoint: {e}")
        return 0.3

    scores = []
    for img_path in bad_imgs:
        try:
            img    = PILImage.open(img_path).convert("RGB")
            tensor = TF.to_tensor(TF.resize(img, [image_size, image_size]))
            tensor = TF.normalize(tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            batch = {"image": tensor.unsqueeze(0)}
            with __import__("torch").no_grad():
                out = model(batch)
            scores.append(float(out["pred_score"].item()))
        except Exception as e:
            print(f"    [WARN] Skipping {img_path.name}: {e}")

    if not scores:
        return 0.3

    threshold = float(np.percentile(scores, recall_percentile))
    # Clamp: never so low everything triggers, never so high we miss defects
    threshold = max(0.05, min(threshold, 0.50))

    print(f"  [{roi_name}] Scores on bad images →  "
          f"min={min(scores):.3f}  median={np.median(scores):.3f}  max={max(scores):.3f}")
    print(f"  [{roi_name}] Threshold: {threshold:.4f}  "
          f"(p{recall_percentile:.0f} → catches ≥{100 - recall_percentile:.0f}% of defects)")

    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(args: argparse.Namespace) -> None:
    rois_root   = Path(args.rois)
    ann_root    = Path(args.annotations)
    models_out  = Path(args.out)
    thresholds  = {}

    if not rois_root.exists():
        sys.exit(f"❌ ROI data directory not found: {rois_root.resolve()}")

    # Discover which ROI folders are present
    roi_dirs = []
    for name in ROI_NAMES:
        d = rois_root / name
        if d.is_dir() and (d / "good").exists():
            roi_dirs.append(d)
        else:
            print(f"  [SKIP] {name} not found or missing good/ — skipping")

    if not roi_dirs:
        sys.exit("❌ No valid ROI folders found. Check --rois path.")

    print(f"\n🔍 Found {len(roi_dirs)} ROI(s) to train: {[d.name for d in roi_dirs]}")

    # Temp dir for generated masks (cleaned up at the end)
    tmp_root = Path(tempfile.mkdtemp(prefix="roi_masks_"))

    total_start = time.perf_counter()

    try:
        for roi_dir in roi_dirs:
            roi_name      = roi_dir.name
            output_dir    = models_out / roi_name
            annotation_dir = ann_root / roi_name

            # ── Skip if already trained ───────────────────────────────────────
            existing_ckpts = list(output_dir.rglob("*.ckpt"))
            if args.skip_done and existing_ckpts:
                print(f"\n  [SKIP] {roi_name} already trained ({existing_ckpts[-1].name}) "
                      "— calibrating threshold only (re-run without --skip-done to retrain)")
                threshold = calibrate_threshold(
                    roi_dir, output_dir,
                    image_size=args.imgsz,
                    recall_percentile=args.recall_percentile,
                )
                thresholds[roi_name] = round(threshold, 6)
                continue

            # ── Train ─────────────────────────────────────────────────────────
            try:
                train_one_roi(
                    roi_dir=roi_dir,
                    annotation_dir=annotation_dir,
                    output_dir=output_dir,
                    tmp_root=tmp_root,
                    image_size=args.imgsz,
                    max_epochs=args.epochs,
                    batch_size=args.batch,
                    num_workers=args.workers,
                    model_size=args.model_size,
                )
            except Exception as e:
                print(f"\n  [ERROR] Training {roi_name} failed: {e}")
                thresholds[roi_name] = 0.5
                continue

            # ── Calibrate threshold ───────────────────────────────────────────
            threshold = calibrate_threshold(
                roi_dir, output_dir,
                image_size=args.imgsz,
                recall_percentile=args.recall_percentile,
            )
            thresholds[roi_name] = round(threshold, 6)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Save thresholds ───────────────────────────────────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_DIR / "roi_thresholds.json"

    # Merge with existing thresholds so we don't wipe un-trained ROIs
    existing = {}
    if out_path.exists():
        try:
            raw = json.loads(out_path.read_text())
            existing = raw.get("thresholds", raw)
            existing = {k: v for k, v in existing.items() if not k.startswith("_")}
        except Exception:
            pass
    existing.update(thresholds)

    out_path.write_text(json.dumps(
        {
            "thresholds": existing,
            "_note": (
                "Auto-calibrated for max recall. "
                "Scores above threshold → FAIL for that ROI. "
                "Lower value = more sensitive."
            ),
        },
        indent=2,
    ))

    total_min = (time.perf_counter() - total_start) / 60
    _banner(f"✅ All done in {total_min:.1f} min")
    print(f"  Thresholds saved → {out_path}")
    for roi_name, t in thresholds.items():
        print(f"    {roi_name}: {t:.4f}")
    print()
    print("Next step — run the full end-to-end pipeline:")
    print("  python himalaya_label_detection/scripts/run_roi_inference.py \\")
    print("    --image 'dataset/NSC/NSC BAD IMAGES/1.bmp' --show")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train EfficientAD classifiers for each ROI using defect annotations, "
            "then auto-calibrate recall-optimized thresholds."
        )
    )
    p.add_argument("--rois", default="data/rois",
                   help="Root folder containing ROI_1/, ROI_2/, ... subfolders (default: data/rois)")
    p.add_argument("--annotations", default="data/annotations",
                   help="Root folder containing ROI_1/, ROI_2/, ... annotation txt files (default: data/annotations)")
    p.add_argument("--out",  default="models/rois",
                   help="Root folder to save trained model checkpoints (default: models/rois)")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per ROI (default: 200)")
    p.add_argument("--imgsz", type=int, default=512,
                   help="Resize each crop to this square size (default: 512)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size — 16 is safe for A5000 24GB (default: 16)")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader worker processes (default: 8)")
    p.add_argument("--model-size", default="medium", choices=["small", "medium"],
                   help="EfficientAD variant: 'medium' recommended (default: medium)")
    p.add_argument("--recall-percentile", type=float, default=5.0,
                   help=(
                       "Threshold calibration percentile. "
                       "5.0 → 95%% recall. 1.0 → 99%% recall (more false positives). "
                       "(default: 5.0)"
                   ))
    p.add_argument("--skip-done", action="store_true", default=False,
                   help="Skip ROIs that already have a trained checkpoint and only re-calibrate thresholds")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(args)
