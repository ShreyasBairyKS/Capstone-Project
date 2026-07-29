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
MODELS_ROOT  = PROJECT_ROOT / "models" / "rois"
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

def score_crop(model: EfficientAd, img_bgr: np.ndarray, device: str) -> Tuple[float, np.ndarray]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        
    if isinstance(out, dict) and "anomaly_map" in out:
        amap = out["anomaly_map"].squeeze().cpu().numpy()
        score = float(amap.mean())
    elif hasattr(out, "anomaly_map"):
        amap = out.anomaly_map.squeeze().cpu().numpy()
        score = float(amap.mean())
    elif isinstance(out, torch.Tensor):
        amap = out.squeeze().cpu().numpy()
        score = float(amap.mean())
    else:
        print(f"[WARN] Unknown output format from PyTorch: {type(out)}")
        return 0.0, np.zeros((256, 256))
        
    return score, amap

def overlay_heatmap(img_bgr: np.ndarray, amap: np.ndarray) -> np.ndarray:
    # Resize amap to match original image if needed (amap is 256x256)
    h, w = img_bgr.shape[:2]
    amap_resized = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize heatmap to 0-255
    min_val, max_val = amap_resized.min(), amap_resized.max()
    if max_val > min_val:
        norm_map = (amap_resized - min_val) / (max_val - min_val)
    else:
        norm_map = amap_resized
    
    heatmap = np.uint8(255 * norm_map)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Create side-by-side: original | overlay
    return np.hstack((img_bgr, overlay))

def test_roi(roi_name: str, device: str):
    print("\n" + "═"*64)
    print(f"  Testing: {roi_name}")
    print("═"*64)
    
    roi_dir = DATA_ROOT / roi_name
    if not roi_dir.exists():
        print(f"  [SKIP] No data found at {roi_dir}")
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
    
    results = [] # list of dicts: {path, true_label, score, amap, img}
    
    t0 = time.time()
    
    for f in good_files:
        img = cv2.imread(str(f))
        if img is not None:
            score, amap = score_crop(model, img, device)
            results.append({"path": f, "true_label": "good", "score": score, "amap": amap, "img": img})
            
    for f in bad_files:
        img = cv2.imread(str(f))
        if img is not None:
            score, amap = score_crop(model, img, device)
            results.append({"path": f, "true_label": "bad", "score": score, "amap": amap, "img": img})
            
    t1 = time.time()
    
    bad_scores = [r["score"] for r in results if r["true_label"] == "bad"]
    good_scores = [r["score"] for r in results if r["true_label"] == "good"]
    
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
    
    tp, fn, fp, tn = 0, 0, 0, 0
    
    # Save results to categorized folders
    out_base = PROJECT_ROOT / "results" / "classifier_testing" / roi_name
    for sub in ["tp", "tn", "fp", "fn"]:
        (out_base / sub).mkdir(parents=True, exist_ok=True)
        
    for r in results:
        is_bad = r["true_label"] == "bad"
        pred_bad = r["score"] > threshold
        
        if is_bad and pred_bad:
            cat = "tp"
            tp += 1
        elif is_bad and not pred_bad:
            cat = "fn"
            fn += 1
        elif not is_bad and pred_bad:
            cat = "fp"
            fp += 1
        else:
            cat = "tn"
            tn += 1
            
        vis = overlay_heatmap(r["img"], r["amap"])
        
        # Add score text
        color = (0, 0, 255) if pred_bad else (0, 200, 0)
        cv2.putText(vis, f"Score: {r['score']:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"Thresh: {threshold:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        out_path = out_base / cat / f"{r['score']:.2f}_{r['path'].name}"
        cv2.imwrite(str(out_path), vis)
    
    recall = tp / len(bad_scores) if bad_scores else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    print("\n  Confusion Matrix:")
    print(f"    TP (bad caught):   {tp}")
    print(f"    FN (missed def):   {fn}")
    print(f"    TN (good passed):  {tn}")
    print(f"    FP (false alarm):  {fp}")
    print(f"\n  Recall:    {recall*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Time:      {((t1-t0)/len(results))*1000:.1f}ms per crop")
    print(f"  Images saved to: {out_base}")
    
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
