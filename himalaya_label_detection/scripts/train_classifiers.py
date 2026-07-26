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

Prerequisites (run on the remote PC ONCE before training):
    pip install "numpy<2" anomalib==1.1.0 lightning timm albumentations imgaug kornia opencv-python

Usage (recommended — let it run overnight on the A5000):
    python himalaya_label_detection/scripts/train_classifiers.py

Custom paths / options:
    python himalaya_label_detection/scripts/train_classifiers.py \\
        --rois data/rois \\
        --annotations data/annotations \\
        --out  models/rois \\
        --epochs 200 \\
        --imgsz 256 \\
        --batch 16 \\
        --model-size medium

Resume after interruption (skips already-trained ROIs):
    python himalaya_label_detection/scripts/train_classifiers.py --skip-done

Design notes:
  - Trained exclusively on GOOD images (anomaly detection, not classification).
  - YOLO-format defect annotations (data/annotations/ROI_N/) are converted to
    binary pixel masks placed in data/rois/ROI_N/ground_truth/bad/.
    anomalib uses these masks for pixel-level evaluation metrics (AUROC, F1),
    which gives much better threshold calibration.
  - Bad images are used for recall-optimized threshold calibration after training.
  - Threshold is set at the 5th percentile of bad-image scores:
    95%+ of real defects are always caught (we tolerate some false positives).
  - EfficientAD-Medium is used — ~6% better AUROC than Small on the A5000 24GB.
  - Image size is 256x256 (EfficientAD's native input size — do NOT increase
    beyond this, it won't help and wastes VRAM).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Literal

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

        x1 = max(0, int((cx - bw / 2) * img_w))
        y1 = max(0, int((cy - bh / 2) * img_h))
        x2 = min(img_w, int((cx + bw / 2) * img_w))
        y2 = min(img_h, int((cy + bh / 2) * img_h))

        mask[y1:y2, x1:x2] = 255

    cv2.imwrite(str(out_path), mask)
    return True


def build_ground_truth_masks(
    roi_dir: Path,
    annotation_dir: Path,
    roi_name: str,
) -> Path | None:
    """
    For every annotated bad image, generate its binary mask PNG and place it in
    roi_dir/ground_truth/bad/ — this is the anomalib-native mask directory layout.

    anomalib's Folder datamodule expects:
        root/
          good/          (normal_dir)
          bad/           (abnormal_dir)
          ground_truth/  (mask_dir — relative to root)
            bad/
              <same filenames as bad/>

    Returns the relative mask_dir string ("ground_truth/bad"), or None.
    """
    bad_dir = roi_dir / "bad"

    ann_files = sorted(
        p for p in (annotation_dir.glob("*.txt") if annotation_dir.exists() else [])
        if p.name.lower() != "classes.txt"
    )
    if not ann_files:
        print(f"  [{roi_name}] No YOLO annotations found in {annotation_dir}")
        return None

    # Place masks INSIDE the roi_dir so anomalib can find them with a relative path
    mask_out = roi_dir / "ground_truth" / "bad"
    mask_out.mkdir(parents=True, exist_ok=True)

    matched = 0
    for ann_path in ann_files:
        stem = ann_path.stem
        # Find the matching image in bad/ (any extension)
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

        out_mask = mask_out / f"{stem}.png"
        if yolo_to_mask(ann_path, w, h, out_mask):
            matched += 1

    if matched == 0:
        shutil.rmtree(roi_dir / "ground_truth", ignore_errors=True)
        return None

    print(f"  [{roi_name}] Generated {matched} binary masks → {mask_out}")
    return "ground_truth/bad"   # relative to roi_dir (root)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_roi(
    roi_dir: Path,
    annotation_dir: Path | None,
    output_dir: Path,
    image_size: int = 256,
    max_epochs: int = 200,
    batch_size: int = 16,
    num_workers: int = 8,
    model_size: Literal["small", "medium"] = "medium",
) -> Path:
    """
    Train EfficientAD on one ROI's good/ images.

    Returns path to the saved checkpoint directory.
    """
    # ── Import anomalib — use direct paths to avoid loading all models ──────
    # Top-level imports trigger the full model zoo (video/CLIP/pkg_resources).
    # Direct paths + Lightning Trainer bypass all of that.
    try:
        import anomalib
        print(f"  anomalib v{anomalib.__version__}")

        # Direct model/data imports — bypasses video/CLIP/pkg_resources chain
        from anomalib.data.image.folder import Folder
        from anomalib.models.image.efficient_ad.lightning_model import EfficientAd

        # Use Lightning Trainer directly — anomalib models ARE LightningModules
        # This completely avoids anomalib.engine which pulls in the full model zoo
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint

    except ModuleNotFoundError as exc:
        missing = str(exc)
        if "pkg_resources" in missing:
            hint = "Fix:  python -m pip install --force-reinstall setuptools"
        elif "kornia" in missing:
            hint = "Fix:  pip install kornia"
        elif "sklearn" in missing or "scikit" in missing:
            hint = "Fix:  pip install scikit-learn"
        elif "imgaug" in missing:
            hint = "Fix:  pip install imgaug"
        elif "lightning" in missing:
            hint = "Fix:  pip install lightning"
        else:
            hint = "Fix:  pip install 'numpy<2' scikit-learn kornia imgaug timm lightning albumentations"
        raise ImportError(f"Import failed: {exc}\n{hint}") from exc

    roi_name = roi_dir.name
    good_dir = roi_dir / "good"
    bad_dir  = roi_dir / "bad"
    n_good   = len(list_images(good_dir))
    n_bad    = len(list_images(bad_dir))

    _banner(f"Training {roi_name}  |  {n_good} good  {n_bad} bad  |  EfficientAD-{model_size.upper()}")

    if n_good == 0:
        print(f"  [SKIP] No good images in {good_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate binary masks from YOLO annotations ───────────────────────────
    mask_rel = None
    if annotation_dir and annotation_dir.exists():
        mask_rel = build_ground_truth_masks(roi_dir, annotation_dir, roi_name)

    # ── Build anomalib Folder datamodule (v1.1.0 API) ─────────────────────────
    # EfficientAD's native input size is 256x256.
    # CRITICAL: anomalib's EfficientAD implementation requires train_batch_size=1.
    folder_kwargs = dict(
        name=roi_name,
        root=str(roi_dir),
        normal_dir="good",
        image_size=(image_size, image_size),
        train_batch_size=1,  # MUST be 1 for EfficientAD
        eval_batch_size=batch_size, # evaluation can be batched
        num_workers=num_workers,
    )

    if n_bad > 0:
        folder_kwargs["abnormal_dir"] = "bad"

    if mask_rel is not None:
        folder_kwargs["mask_dir"] = mask_rel
        # Use segmentation task when masks are available — gives pixel-level metrics
        try:
            from anomalib.utils.types import TaskType
            folder_kwargs["task"] = TaskType.SEGMENTATION
            print(f"  [{roi_name}] Task: SEGMENTATION (pixel-level masks available)")
        except ImportError:
            pass
    else:
        print(f"  [{roi_name}] Task: CLASSIFICATION (no masks)")

    print(f"  [{roi_name}] Folder kwargs: root={roi_dir}, normal=good, abnormal=bad, "
          f"mask={mask_rel}, imgsz={image_size}")

    datamodule = Folder(**folder_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    # EfficientAD has its own internal augmentation pipeline (teacher-student
    # with knowledge distillation). We do NOT add external augmentations —
    # the paper shows the internal pipeline is optimal.
    model = EfficientAd(model_size=model_size)

    # ── Trainer (Lightning Trainer used directly — avoids anomalib.engine) ────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "weights"),
        filename="best",
        monitor="pixel_AUROC",      # EfficientAD exposes this metric
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=str(output_dir),
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
    )

    t0 = time.perf_counter()
    trainer.fit(model=model, datamodule=datamodule)
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
        "masks_used":  mask_rel is not None,
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
    image_size: int = 256,
    recall_percentile: float = 5.0,
) -> float:
    """
    Score every bad image through the trained model and set the detection
    threshold at the `recall_percentile`-th percentile of those scores.

    recall_percentile=5 → threshold below 95% of bad scores → 95%+ recall.
    """
    try:
        import torch
        from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
    except ImportError:
        print("  [WARN] Cannot calibrate — missing torch/anomalib")
        return 0.3

    bad_dir   = roi_dir / "bad"
    bad_imgs  = list_images(bad_dir)
    roi_name  = roi_dir.name

    if not bad_imgs:
        print(f"  [{roi_name}] No bad images — using conservative default threshold 0.3")
        return 0.3

    ckpt_paths = sorted(model_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpt_paths:
        print(f"  [{roi_name}] No checkpoint found — using default 0.5")
        return 0.5

    latest = ckpt_paths[-1]
    print(f"\n  [{roi_name}] Calibrating on {len(bad_imgs)} bad images (ckpt: {latest.name})...")

    try:
        model = EfficientAd.load_from_checkpoint(str(latest))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"  [{roi_name}] Checkpoint load failed: {e}")
        return 0.3

    device = next(model.parameters()).device
    scores = []

    for img_path in bad_imgs:
        try:
            img    = PILImage.open(img_path).convert("RGB")
            tensor = TF.to_tensor(TF.resize(img, [image_size, image_size]))
            tensor = TF.normalize(tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            batch = {"image": tensor.unsqueeze(0).to(device)}
            with torch.no_grad():
                out = model(batch)
            scores.append(float(out["pred_score"].item()))
        except Exception as e:
            print(f"    [WARN] {img_path.name}: {e}")

    if not scores:
        return 0.3

    threshold = float(np.percentile(scores, recall_percentile))
    threshold = max(0.05, min(threshold, 0.50))

    print(f"  [{roi_name}] Bad scores →  min={min(scores):.3f}  "
          f"median={np.median(scores):.3f}  max={max(scores):.3f}")
    print(f"  [{roi_name}] Threshold: {threshold:.4f}  "
          f"(p{recall_percentile:.0f} → catches ≥{100 - recall_percentile:.0f}% of defects)")

    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Good-image scores (for false positive rate estimation)
# ─────────────────────────────────────────────────────────────────────────────

def score_good_images(
    roi_dir: Path,
    model_dir: Path,
    threshold: float,
    image_size: int = 256,
) -> None:
    """Score good images and report how many would be false positives."""
    try:
        import torch
        from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
    except ImportError:
        return

    good_imgs = list_images(roi_dir / "good")
    roi_name  = roi_dir.name
    if not good_imgs:
        return

    ckpt_paths = sorted(model_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpt_paths:
        return

    try:
        model = EfficientAd.load_from_checkpoint(str(ckpt_paths[-1]))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except Exception:
        return

    device = next(model.parameters()).device
    scores = []

    for img_path in good_imgs:
        try:
            img    = PILImage.open(img_path).convert("RGB")
            tensor = TF.to_tensor(TF.resize(img, [image_size, image_size]))
            tensor = TF.normalize(tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            batch = {"image": tensor.unsqueeze(0).to(device)}
            with torch.no_grad():
                out = model(batch)
            scores.append(float(out["pred_score"].item()))
        except Exception:
            pass

    if not scores:
        return

    fp = sum(1 for s in scores if s > threshold)
    print(f"  [{roi_name}] Good images: {len(scores)} scored  |  "
          f"max={max(scores):.3f}  |  {fp} would be false positives at threshold {threshold:.4f}")


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

    total_start = time.perf_counter()

    for roi_dir in roi_dirs:
        roi_name       = roi_dir.name
        output_dir     = models_out / roi_name
        annotation_dir = ann_root / roi_name

        # ── Skip if already trained ───────────────────────────────────────
        existing_ckpts = list(output_dir.rglob("*.ckpt"))
        if args.skip_done and existing_ckpts:
            print(f"\n  [SKIP] {roi_name} already trained ({existing_ckpts[-1].name})")
            threshold = calibrate_threshold(
                roi_dir, output_dir,
                image_size=args.imgsz,
                recall_percentile=args.recall_percentile,
            )
            thresholds[roi_name] = round(threshold, 6)
            score_good_images(roi_dir, output_dir, threshold, args.imgsz)
            continue

        # ── Train ─────────────────────────────────────────────────────────
        try:
            train_one_roi(
                roi_dir=roi_dir,
                annotation_dir=annotation_dir,
                output_dir=output_dir,
                image_size=args.imgsz,
                max_epochs=args.epochs,
                batch_size=args.batch,
                num_workers=args.workers,
                model_size=args.model_size,
            )
        except Exception as e:
            print(f"\n  [ERROR] Training {roi_name} failed: {e}")
            import traceback
            traceback.print_exc()
            thresholds[roi_name] = 0.5
            continue

        # ── Calibrate threshold ───────────────────────────────────────────
        threshold = calibrate_threshold(
            roi_dir, output_dir,
            image_size=args.imgsz,
            recall_percentile=args.recall_percentile,
        )
        thresholds[roi_name] = round(threshold, 6)

        # ── False positive check ──────────────────────────────────────────
        score_good_images(roi_dir, output_dir, threshold, args.imgsz)

    # ── Save thresholds ───────────────────────────────────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_DIR / "roi_thresholds.json"

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
            "_note": "Auto-calibrated for max recall. Lower = more sensitive.",
        },
        indent=2,
    ))

    total_min = (time.perf_counter() - total_start) / 60
    _banner(f"✅ All done in {total_min:.1f} min")
    print(f"  Thresholds saved → {out_path}")
    for roi_name, t in thresholds.items():
        print(f"    {roi_name}: {t:.4f}")
    print()
    print("Next step:")
    print("  python himalaya_label_detection/scripts/run_roi_inference.py \\")
    print("    --image \"dataset/NSC/NSC BAD IMAGES/1.bmp\" --show")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientAD classifiers for each ROI + auto-calibrate thresholds"
    )
    p.add_argument("--rois", default="data/rois",
                   help="Root folder with ROI_1/, ROI_2/, ... subfolders")
    p.add_argument("--annotations", default="data/annotations",
                   help="Root folder with ROI_1/, ROI_2/, ... YOLO annotation .txt files")
    p.add_argument("--out",  default="models/rois",
                   help="Save trained checkpoints here")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per ROI (default: 200)")
    p.add_argument("--imgsz", type=int, default=256,
                   help="Input size — EfficientAD's native size is 256 (default: 256)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader workers (default: 8)")
    p.add_argument("--model-size", default="medium", choices=["small", "medium"],
                   help="EfficientAD variant (default: medium)")
    p.add_argument("--recall-percentile", type=float, default=5.0,
                   help="Calibration percentile. 5=95%% recall, 1=99%% recall (default: 5)")
    p.add_argument("--skip-done", action="store_true", default=False,
                   help="Skip already-trained ROIs, only re-calibrate thresholds")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(args)
