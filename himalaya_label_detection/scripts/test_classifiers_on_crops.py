"""
test_classifiers_on_crops.py
─────────────────────────────────────────────────────────────────────────────
Tests the PyTorch anomaly classifiers (EfficientAD) directly on pre-cropped
images, completely bypassing YOLO.

This helps isolate whether the poor recall is due to YOLO missing the ROIs,
or if the anomaly classifiers themselves are failing to detect the defects.

Usage:
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py --roi ROI_1
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from anomalib.models import EfficientAd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT  = PROJECT_ROOT / "models" / "rois" / "pytorch"
DATA_ROOT    = PROJECT_ROOT / "data" / "rois"

def load_model(roi_name: str, device: str) -> EfficientAd:
    ckpt_dir = MODELS_ROOT / roi_name / "weights"
    if not ckpt_dir.exists():
        sys.exit(f"❌  No weights folder found for {roi_name} at {ckpt_dir}")
        
    hits = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not hits:
        sys.exit(f"❌  No .ckpt found for {roi_name}")
        
    ckpt_path = hits[-1]
    print(f"  [{roi_name}] Loading: {ckpt_path.name}")
    
    model = EfficientAd.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    return model

def score_crop(model: EfficientAd, img_bgr: np.ndarray, device: str) -> float:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        
    if isinstance(out, dict) and "anomaly_map" in out:
        return float(out["anomaly_map"].mean().item())
    elif hasattr(out, "anomaly_map"):
        return float(out.anomaly_map.mean().item())
    elif isinstance(out, torch.Tensor):
        return float(out.mean().item())
    else:
        print(f"[WARN] Unknown output format from PyTorch: {type(out)}")
        return 0.0

def test_roi(roi_name: str, device: str):
    print("\n" + "═"*64)
    print(f"  Testing: {roi_name}")
    print("═"*64)
    
    roi_dir = DATA_ROOT / roi_name / "test"
    if not roi_dir.exists():
        print(f"  [SKIP] No test data found at {roi_dir}")
        return
        
    good_dir = roi_dir / "good"
    bad_dir = roi_dir / "bad"
    
    good_files = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.bmp")) + sorted(good_dir.glob("*.jpg"))
    bad_files = sorted(bad_dir.glob("*.png")) + sorted(bad_dir.glob("*.bmp")) + sorted(bad_dir.glob("*.jpg"))
    
    if not good_files and not bad_files:
        print(f"  [SKIP] No images found in {roi_dir}/good or bad")
        return
        
    print(f"  Found {len(good_files)} good crops, {len(bad_files)} bad crops.")
    
    model = load_model(roi_name, device)
    
    good_scores = []
    bad_scores = []
    
    t0 = time.time()
    
    for f in good_files:
        img = cv2.imread(str(f))
        if img is not None:
            good_scores.append(score_crop(model, img, device))
            
    for f in bad_files:
        img = cv2.imread(str(f))
        if img is not None:
            bad_scores.append(score_crop(model, img, device))
            
    t1 = time.time()
    
    if bad_scores:
        threshold = float(np.percentile(bad_scores, 5.0))
        print(f"\n  Threshold (5th %ile of bad): {threshold:.4f}")
    elif good_scores:
        threshold = float(max(good_scores)) * 1.1
        print(f"\n  Threshold (110% of max good): {threshold:.4f}")
    else:
        threshold = 999.0
        
    print(f"  Min Bad Score:  {min(bad_scores) if bad_scores else 'N/A':.4f}")
    print(f"  Max Good Score: {max(good_scores) if good_scores else 'N/A':.4f}")
    
    tp = sum(1 for s in bad_scores if s > threshold)
    fn = len(bad_scores) - tp
    fp = sum(1 for s in good_scores if s > threshold)
    tn = len(good_scores) - fp
    
    recall = tp / len(bad_scores) if bad_scores else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    print("\n  Confusion Matrix:")
    print(f"    TP (bad caught):   {tp}")
    print(f"    FN (missed def):   {fn}")
    print(f"    TN (good passed):  {tn}")
    print(f"    FP (false alarm):  {fp}")
    print(f"\n  Recall:    {recall*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Time:      {((t1-t0)/len(good_files+bad_files))*1000:.1f}ms per crop")
    
    if fn > 0:
        print("\n  ⚠️ WARNING: The anomaly classifier missed some defects. If this number is high,")
        print("  the classifier itself is failing, independent of YOLO.")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", type=str, default="all", help="Specific ROI to test (e.g. ROI_1)")
    args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    rois_to_test = []
    if args.roi != "all":
        rois_to_test = [args.roi]
    else:
        rois_to_test = [d.name for d in MODELS_ROOT.iterdir() if d.is_dir() and d.name.startswith("ROI_")]
        
    for roi in sorted(rois_to_test):
        test_roi(roi, device)

if __name__ == "__main__":
    main()
