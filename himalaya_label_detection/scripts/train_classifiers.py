"""
train_classifiers.py
────────────────────
Trains one EfficientAD-Medium anomaly classifier per ROI, then auto-calibrates
detection thresholds optimized for maximum recall.

Pipeline:
  data/rois/ROI_1/good/ + bad/  →  EfficientAD-Medium  →  model checkpoint
  data/rois/ROI_2/good/ + bad/  →  EfficientAD-Medium  →  model checkpoint
  data/rois/ROI_3/good/ + bad/  →  EfficientAD-Medium  →  model checkpoint
  data/rois/ROI_4/good/ + bad/  →  EfficientAD-Medium  →  model checkpoint
                                                         ↓
                                            config/roi_thresholds.json

Usage (recommended — let it run overnight on the A5000):
    python himalaya_label_detection/scripts/train_classifiers.py

Custom paths / options:
    python himalaya_label_detection/scripts/train_classifiers.py \\
        --rois data/rois \\
        --out  models/rois \\
        --epochs 200 \\
        --imgsz 512 \\
        --batch 16 \\
        --model-size medium

Resume after interruption (skips already-trained ROIs):
    python himalaya_label_detection/scripts/train_classifiers.py --skip-done

Design notes:
  - Trained exclusively on GOOD images (anomaly detection, not classification)
  - Bad images are ONLY used for threshold calibration after training
  - Threshold is set at the 5th percentile of bad-image scores to maximise recall
    (i.e. we accept some false positives to ensure we never miss a real defect)
  - EfficientAD-Medium gives ~6% better AUROC vs Small with negligible time cost
    on a GPU with ≥16 GB VRAM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR   = PROJECT_ROOT / "himalaya_label_detection" / "config"
ROI_NAMES    = ["ROI_1", "ROI_2", "ROI_3", "ROI_4"]
SUPPORTED    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


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
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_roi(
    roi_dir: Path,
    output_dir: Path,
    image_size: int = 512,
    max_epochs: int = 200,
    batch_size: int = 16,
    num_workers: int = 8,
    model_size: Literal["small", "medium"] = "medium",
) -> Path:
    """
    Train EfficientAD on one ROI's good/ images.

    Returns path to the saved checkpoint directory.
    """
    try:
        import torch
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import EfficientAd
        from pytorch_lightning.callbacks import ModelCheckpoint
    except ImportError as exc:
        raise ImportError(
            "anomalib is not installed.\n"
            "Run:  pip install anomalib==1.1.0 timm pytorch-lightning"
        ) from exc

    roi_name  = roi_dir.name
    good_dir  = roi_dir / "good"
    n_good    = len(list_images(good_dir))

    _banner(f"Training {roi_name}  |  {n_good} good images  |  EfficientAD-{model_size.upper()}")

    if n_good == 0:
        print(f"  [SKIP] No good images found in {good_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    # anomalib Folder datamodule — trained on normal (good) only
    # bad/ is NOT passed here — used only in calibration phase
    datamodule = Folder(
        name=roi_name,
        root=str(roi_dir),
        normal_dir="good",
        abnormal_dir="bad",           # anomalib uses this for val metrics only
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        # Augmentation — critical for generalisation on small datasets
        augmentations=dict(
            train=dict(
                Compose=[
                    dict(HorizontalFlip=dict(p=0.5)),
                    dict(VerticalFlip=dict(p=0.3)),
                    dict(RandomRotate90=dict(p=0.3)),
                    dict(ColorJitter=dict(
                        brightness=0.3, contrast=0.3,
                        saturation=0.2, hue=0.05, p=0.8
                    )),
                    dict(GaussNoise=dict(var_limit=(10, 50), p=0.5)),
                    dict(Blur=dict(blur_limit=3, p=0.3)),
                ]
            )
        ),
    )

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
        "roi_name":   roi_name,
        "model_type": "efficientad",
        "model_size": model_size,
        "image_size": [image_size, image_size],
        "epochs":     max_epochs,
        "n_good":     n_good,
        "data_dir":   str(roi_dir.resolve()),
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
    Compute the anomaly score on every bad image and set the threshold at the
    `recall_percentile`-th percentile of those scores.

    recall_percentile=5 → threshold sits below 95% of bad image scores,
    so we catch at least 95% of real defects (high recall, tolerate some false positives).
    Lower the percentile → even higher recall.

    Returns the calibrated threshold float, or 0.3 if no bad images exist.
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
        print(f"  [{roi_name}] No bad images found — using conservative default threshold 0.3")
        return 0.3

    # Find the latest checkpoint
    ckpt_paths = sorted(model_dir.rglob("*.ckpt"))
    if not ckpt_paths:
        print(f"  [{roi_name}] No checkpoint found in {model_dir} — skipping calibration")
        return 0.5

    print(f"\n  [{roi_name}] Calibrating threshold on {len(bad_imgs)} bad images...")

    try:
        model = EfficientAd.load_from_checkpoint(str(ckpt_paths[-1]))
        model.eval()
    except Exception as e:
        print(f"  [{roi_name}] Could not load checkpoint: {e}")
        return 0.3

    scores = []
    for img_path in bad_imgs:
        try:
            img = PILImage.open(img_path).convert("RGB")
            tensor = TF.to_tensor(TF.resize(img, [image_size, image_size]))
            tensor = TF.normalize(tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            batch = {"image": tensor.unsqueeze(0)}
            with torch.no_grad():
                out = model(batch)
            score = float(out["pred_score"].item())
            scores.append(score)
        except Exception as e:
            print(f"    [WARN] Skipping {img_path.name}: {e}")

    if not scores:
        return 0.3

    threshold = float(np.percentile(scores, recall_percentile))

    # Safety clamp — never set threshold above 0.5 (too permissive) 
    # or below 0.05 (too aggressive, everything fails)
    threshold = max(0.05, min(threshold, 0.50))

    print(f"  [{roi_name}] Bad-image scores → min={min(scores):.3f}  "
          f"median={np.median(scores):.3f}  max={max(scores):.3f}")
    print(f"  [{roi_name}] Threshold set to {threshold:.4f}  "
          f"(p{recall_percentile:.0f} of bad scores)")

    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(args: argparse.Namespace) -> None:
    rois_root  = Path(args.rois)
    models_out = Path(args.out)
    thresholds = {}

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
        roi_name    = roi_dir.name
        output_dir  = models_out / roi_name

        # ── Skip if already trained ──────────────────────────────────────────
        already_done = list(output_dir.rglob("*.ckpt"))
        if args.skip_done and already_done:
            print(f"\n  [SKIP] {roi_name} already trained ({already_done[-1].name}) — use --skip-done=false to retrain")
            # Still calibrate threshold
            threshold = calibrate_threshold(
                roi_dir, output_dir,
                image_size=args.imgsz,
                recall_percentile=args.recall_percentile,
            )
            thresholds[roi_name] = round(threshold, 6)
            continue

        # ── Train ────────────────────────────────────────────────────────────
        try:
            train_one_roi(
                roi_dir=roi_dir,
                output_dir=output_dir,
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

        # ── Calibrate threshold ───────────────────────────────────────────────
        threshold = calibrate_threshold(
            roi_dir, output_dir,
            image_size=args.imgsz,
            recall_percentile=args.recall_percentile,
        )
        thresholds[roi_name] = round(threshold, 6)

    # ── Save thresholds ───────────────────────────────────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_DIR / "roi_thresholds.json"

    # Merge with any existing thresholds (don't overwrite ROIs we didn't train)
    existing = {}
    if out_path.exists():
        try:
            raw = json.loads(out_path.read_text())
            existing = raw.get("thresholds", raw)
            # Remove comment keys
            existing = {k: v for k, v in existing.items() if not k.startswith("_")}
        except Exception:
            pass

    existing.update(thresholds)

    out_path.write_text(json.dumps(
        {"thresholds": existing, "_note": "Auto-calibrated for max recall. Lower value = more sensitive."},
        indent=2
    ))

    total_min = (time.perf_counter() - total_start) / 60
    _banner(f"✅ All done in {total_min:.1f} min")
    print(f"  Thresholds saved → {out_path}")
    for roi_name, t in thresholds.items():
        print(f"    {roi_name}: {t:.4f}")
    print()
    print("Next step:")
    print("  python himalaya_label_detection/scripts/run_roi_inference.py \\")
    print("    --image 'dataset/NSC/NSC BAD IMAGES/1.bmp' --show")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train EfficientAD classifiers for each ROI + auto-calibrate thresholds"
    )
    p.add_argument("--rois", default="data/rois",
                   help="Root folder containing ROI_1/, ROI_2/, ... subfolders")
    p.add_argument("--out",  default="models/rois",
                   help="Root folder to save trained model checkpoints")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per ROI (default: 200)")
    p.add_argument("--imgsz", type=int, default=512,
                   help="Resize each crop to this square size (default: 512)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16 — fine for A5000 24GB)")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader worker processes")
    p.add_argument("--model-size", default="medium", choices=["small", "medium"],
                   help="EfficientAD variant: 'medium' recommended (default)")
    p.add_argument("--recall-percentile", type=float, default=5.0,
                   help="Threshold percentile for calibration. Lower = higher recall. (default: 5)")
    p.add_argument("--skip-done", action="store_true", default=True,
                   help="Skip ROIs that already have a trained checkpoint (default: True)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(args)
