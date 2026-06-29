import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from scripts.prepare_nsc_dataset import (
    load_config, load_image, run_quality_gate, 
    find_fiducial_offset, register_strip, extract_patches
)
from training.train_nsc_patchcore import FeatureExtractor, PatchCoreMemoryBank

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single NSC cylindrical tube image.")
    parser.add_argument("image", type=Path, help="Path to input .bmp image")
    parser.add_argument("--config", type=Path, default=Path("models/nsc_patchcore_config.yaml"), help="Path to PatchCore config")
    parser.add_argument("--membank", type=Path, default=Path("models/nsc_patchcore_membank.pt"), help="Path to PatchCore memory bank")
    parser.add_argument("--sku-config", type=Path, default=Path("configs/sku_profiles/nsc_tube_unwrap.yaml"), help="Path to SKU config")
    parser.add_argument("--template", type=Path, default=Path("configs/templates/nsc_logo_template.png"), help="Fiducial template path")
    parser.add_argument("--output", type=Path, default=Path("runs/inference_output.png"), help="Path to save heatmap overlay")
    parser.add_argument("--device", type=str, default="", help="Device: cuda:0 or cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        return

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    print(f"==================================================")
    print(f"  NSC PatchCore Inference Pipeline")
    print(f"==================================================")
    print(f"  Image  : {args.image.name}")
    print(f"  Device : {device}")

    # 2. Load configurations
    sku_config = load_config(args.sku_config)
    with open(args.config, "r") as f:
        pc_config = yaml.safe_load(f)
        
    threshold = pc_config.get("threshold", 1.2105)
    feature_dim = pc_config["feature_dim"]
    image_size = pc_config["image_size"]

    t0 = time.time()
    
    # 3. Load image
    print(f"\n[1/5] Loading image...")
    image_bgr = load_image(args.image)
    
    # 4. Quality Gate
    print(f"[2/5] Running Quality Gate...")
    qg_result = run_quality_gate(image_bgr, sku_config)
    if qg_result["status"] == "RECAPTURE":
        print(f"  [REJECT] Quality Gate Failed: {qg_result['reason']}")
        return
    print("  [OK] Quality Gate passed.")

    # 5. Registration
    print(f"[3/5] Registering Image...")
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    zone_rows = sku_config["registration"]["zone_rows"]
    x_offset, match_score = find_fiducial_offset(image_bgr, template, zone_rows)
    registered_strip = register_strip(image_bgr, x_offset)
    
    # Extract Patches
    print(f"[4/5] Extracting Patches...")
    # Use standard 384x384 patch size with stride 192
    patches = extract_patches(registered_strip, zone_rows, patch_size=image_size, stride=image_size // 2)
    print(f"  Extracted {len(patches)} patches.")

    # Convert patches to tensor
    # Preprocess: BGR -> RGB, HWC -> CHW, Normalize
    patch_tensors = []
    for p in patches:
        p_rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        p_tensor = torch.from_numpy(p_rgb).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        p_tensor = (p_tensor - mean) / std
        patch_tensors.append(p_tensor)
        
    batch_tensor = torch.stack(patch_tensors).to(device)

    # 6. PatchCore Inference
    print(f"[5/5] Anomaly Scoring...")
    feature_extractor = FeatureExtractor(target_size=feature_dim).to(device)
    feature_extractor.eval()
    
    membank = PatchCoreMemoryBank.load(args.membank, device="cpu")
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        # Extract features
        features = feature_extractor(batch_tensor).cpu()
        
    # Score features
    scores = membank.score(features)
    
    # Reshape scores to per-patch maximum
    spatial_size = feature_dim * feature_dim
    patch_scores = scores.reshape(len(patches), spatial_size).max(dim=1).values.numpy()
    
    max_score = np.max(patch_scores)
    is_anomaly = max_score > threshold
    
    t_total = time.time() - t0
    
    print(f"\n==================================================")
    print(f"  Result : {'[FAIL] ANOMALY DETECTED' if is_anomaly else '[PASS] GOOD'}")
    print(f"  Score  : {max_score:.4f} (Threshold: {threshold:.4f})")
    print(f"  Time   : {t_total:.2f}s")
    print(f"==================================================")
    
    # Generate Heatmap (Optional visualization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Reconstruct the label band visual to overlay heatmap
    label_start, label_end = zone_rows["label_band"]
    label_band = registered_strip[label_start:label_end, :, :].copy()
    h, w = label_band.shape[:2]
    
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    idx = 0
    stride = image_size // 2
    for y in range(0, h - image_size + 1, stride):
        for x in range(0, w - image_size + 1, stride):
            score = patch_scores[idx]
            heatmap[y:y+image_size, x:x+image_size] += score
            counts[y:y+image_size, x:x+image_size] += 1
            idx += 1
            
    heatmap /= np.maximum(counts, 1)
    
    # Normalize heatmap for visualization (clip to threshold)
    heatmap_norm = np.clip((heatmap - threshold) / (np.max(heatmap) - threshold + 1e-5), 0, 1)
    heatmap_norm = (heatmap_norm * 255).astype(np.uint8)
    
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(label_band, 0.5, heatmap_color, 0.5, 0)
    
    cv2.imwrite(str(args.output), overlay)
    print(f"  Saved heatmap overlay -> {args.output}")

if __name__ == "__main__":
    main()
