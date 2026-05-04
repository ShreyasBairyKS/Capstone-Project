"""
inference/yoloWithFillLevel.py
===============================
Full Pipeline: YOLO V3 Detection + Cap Classifier + Fill Level Detection

Pipeline:
    1. YOLO Pass 1 (full frame)  → detect bottles
    2. YOLO Pass 2 (smart zoom)  → detect caps (small-object fix)
    3. Cap Classifier            → good_cap / defective_cap / no_cap
    4. Water Surface YOLO        → detect water surface line on bottle crop
    5. Fill Level Math           → compute fill ratio → underfill / normal / overfill

Output per bottle:
    Cap verdict  : Good Cap / Defective Cap / Missing Cap
    Fill verdict : underfill / normal / overfill
    Combined     : both shown on annotated image

Usage:
    python inference/yoloWithFillLevel.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --fill-weights runs/detect/water_surface_v1/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source path/to/images \
        --device 0

    # Without cap classifier (YOLO only + fill level):
    python inference/yoloWithFillLevel.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --fill-weights runs/detect/water_surface_v1/weights/best.pt \
        --source path/to/images

    # Webcam:
    python inference/yoloWithFillLevel.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --fill-weights runs/detect/water_surface_v1/weights/best.pt \
        --source 0 --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Colors (BGR) ──
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
ORANGE = (0, 140, 255)
YELLOW = (0, 220, 220)
CYAN   = (220, 220, 0)
BLUE   = (220, 120, 0)
GRAY   = (128, 128, 128)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

# ── Classifier config ──
CLS_NAMES = ["good_cap", "defective_cap", "no_cap"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Fill level thresholds ──
# fill_ratio = (bottle_bottom - water_surface_y) / bottle_height
# 0.0 = water at top of bottle (full), 1.0 = water at bottom (empty)
UNDERFILL_THRESHOLD = 0.40   # fill ratio < 0.40 → underfill (less than 40% full)
OVERFILL_THRESHOLD  = 0.85   # fill ratio > 0.85 → overfill  (more than 85% full)

# Bottle bbox padding correction (YOLO adds a small border around objects)
BOTTLE_TOP_TRIM = 0.03   # trim 3% from top of bottle bbox
BOTTLE_BOT_TRIM = 0.02   # trim 2% from bottom of bottle bbox


# ─────────────────────────────────────────────────────────────────────────────
# CAP CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(weights_path: str, device: str):
    """Load MobileNetV3-Small cap quality classifier."""
    from torchvision.models import mobilenet_v3_small

    backbone = mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features

    class CapClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.dropout = nn.Dropout(p=0.3)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, len(CLS_NAMES)),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    dev = torch.device("cuda:0" if device == "0" else device)
    model = CapClassifier()
    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(dev).eval()
    return model, dev


def classify_cap(classifier, device, crop_bgr, size=224):
    """Classify cap crop → (class_name, confidence)."""
    from PIL import Image as PILImage

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    w, h = pil.size
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pil = pil.resize((new_w, new_h), PILImage.BILINEAR)
    canvas = PILImage.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(pil, ((size - new_w) // 2, (size - new_h) // 2))

    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(canvas).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(classifier(tensor), dim=1)
        idx = probs.argmax(1).item()
    return CLS_NAMES[idx], round(probs[0, idx].item(), 3)


# ─────────────────────────────────────────────────────────────────────────────
# FILL LEVEL MATH
# ─────────────────────────────────────────────────────────────────────────────

def compute_fill_ratio(bottle_bbox: list[int], water_bbox: list[int]) -> float:
    """
    Compute fill ratio from bottle and water surface bounding boxes.

    Args:
        bottle_bbox: [x1, y1, x2, y2] of bottle in original image coords
        water_bbox:  [x1, y1, x2, y2] of water surface in original image coords

    Returns:
        fill_ratio: 0.0 = completely full, 1.0 = completely empty
                    Values in between represent partial fill.
    """
    bx1, by1, bx2, by2 = bottle_bbox
    bottle_h = by2 - by1

    # Correct for YOLO bbox padding
    by1_corr = by1 + int(bottle_h * BOTTLE_TOP_TRIM)
    by2_corr = by2 - int(bottle_h * BOTTLE_BOT_TRIM)
    bottle_h_corr = max(by2_corr - by1_corr, 1)

    # Water surface Y = top of the water surface bbox (where water meets air)
    water_y = (water_bbox[1] + water_bbox[3]) // 2  # use center Y of water bbox

    # Clamp water_y within bottle bounds
    water_y = max(by1_corr, min(by2_corr, water_y))

    # Fill from bottom = distance from bottle bottom to water surface
    fill_from_bottom = by2_corr - water_y
    fill_ratio = fill_from_bottom / bottle_h_corr
    return round(fill_ratio, 3)


def fill_verdict(fill_ratio: float) -> str:
    if fill_ratio < UNDERFILL_THRESHOLD:
        return "underfill"
    elif fill_ratio > OVERFILL_THRESHOLD:
        return "overfill"
    else:
        return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO V3 DETECTOR (bottle + cap, 2-pass zoom)
# ─────────────────────────────────────────────────────────────────────────────

class BottleCapDetector:
    def __init__(self, weights, device="0", det_conf=0.25, cap_conf=0.15,
                 zoom_scale=2.5, cap_region_ratio=0.35, crop_pad=0.15):
        self.model = YOLO(weights)
        self.names = self.model.names
        self.device = device
        self.det_conf = det_conf
        self.cap_conf = cap_conf
        self.zoom_scale = zoom_scale
        self.cap_region_ratio = cap_region_ratio
        self.crop_pad = crop_pad

    def _is_bottle(self, name): return "bottle" in name.lower()
    def _is_cap(self, name): return "cap" in name.lower()

    def detect(self, img):
        h, w = img.shape[:2]

        # Pass 1: full frame
        r1 = self.model.predict(img, conf=self.det_conf, iou=0.45,
                                device=self.device, verbose=False)
        dets = []
        for box in r1[0].boxes:
            cls_id = int(box.cls[0])
            name = self.names.get(cls_id, str(cls_id))
            dets.append({
                "class": name, "conf": float(box.conf[0]),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                "source": "pass1",
            })

        bottles = [d for d in dets if self._is_bottle(d["class"])]
        caps_p1 = [d for d in dets if self._is_cap(d["class"])]
        skip_zoom = max((c["conf"] for c in caps_p1), default=0.0) >= 0.70

        # Pass 2: bottle-guided zoom
        caps_p2 = []
        if bottles and not skip_zoom:
            for bottle in sorted(bottles, key=lambda d: -d["conf"])[:5]:
                x1, y1, x2, y2 = bottle["bbox"]
                bw, bh = x2 - x1, y2 - y1
                cap_h = int(bh * self.cap_region_ratio)
                pad_w = int(bw * self.crop_pad)
                pad_h = int(cap_h * self.crop_pad)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w, x2 + pad_w)
                cy2 = min(h, y1 + cap_h + pad_h)

                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0: continue
                ch, cw = crop.shape[:2]
                new_w = min(int(cw * self.zoom_scale), 960)
                new_h = min(int(ch * self.zoom_scale), 960)
                if new_w < 32 or new_h < 32: continue
                zoomed = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                r2 = self.model.predict(zoomed, conf=self.cap_conf, iou=0.45,
                                        device=self.device, verbose=False)
                scale_x, scale_y = cw / new_w, ch / new_h
                for box in r2[0].boxes:
                    name = self.names.get(int(box.cls[0]), "")
                    if not self._is_cap(name): continue
                    zx1, zy1, zx2, zy2 = box.xyxy[0].tolist()
                    caps_p2.append({
                        "class": name, "conf": float(box.conf[0]),
                        "bbox": [int(cx1 + zx1*scale_x), int(cy1 + zy1*scale_y),
                                 int(cx1 + zx2*scale_x), int(cy1 + zy2*scale_y)],
                        "source": "pass2_zoom",
                    })

        all_caps = self._dedup(caps_p1, caps_p2)
        return {"bottles": bottles, "caps": all_caps,
                "caps_p1": len(caps_p1), "caps_p2": len(caps_p2)}

    def _dedup(self, p1, p2):
        if not p2: return p1
        if not p1: return p2
        merged = list(p1)
        for c2 in p2:
            dup = False
            for c1 in p1:
                if self._iou(c1["bbox"], c2["bbox"]) > 0.3:
                    dup = True
                    if c2["conf"] > c1["conf"]:
                        merged[merged.index(c1)] = c2
                    break
            if not dup: merged.append(c2)
        return merged

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def full_inspect(detector, fill_model, classifier, cls_device, img, device):
    """Run complete 4-stage pipeline on one image."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]

    # Stage 1+2: detect bottles + caps
    det = detector.detect(img)
    bottles = det["bottles"]
    caps = det["caps"]

    results_per_bottle = []

    for i, bottle in enumerate(bottles):
        bx1, by1, bx2, by2 = bottle["bbox"]

        # ── Cap quality (Stage 3) ──
        cap_quality = None
        cap_quality_conf = 0.0
        cap_bbox = None

        # Find best cap for this bottle (closest to bottle top)
        best_cap = None
        best_dist = float("inf")
        for cap in caps:
            cx1, cy1, cx2, cy2 = cap["bbox"]
            cap_center_x = (cx1 + cx2) / 2
            # Cap must be horizontally within bottle bounds
            if bx1 <= cap_center_x <= bx2:
                dist = abs(cy1 - by1)
                if dist < best_dist:
                    best_dist = dist
                    best_cap = cap

        if best_cap:
            cap_bbox = best_cap["bbox"]
            if classifier is not None:
                cx1, cy1, cx2, cy2 = cap_bbox
                pad_x = int((cx2 - cx1) * 0.1)
                pad_y = int((cy2 - cy1) * 0.1)
                crop = img[max(0, cy1-pad_y):min(h, cy2+pad_y),
                           max(0, cx1-pad_x):min(w, cx2+pad_x)]
                if crop.size > 0:
                    cap_quality, cap_quality_conf = classify_cap(classifier, cls_device, crop)

        # ── Fill level (Stage 4) ──
        fill_ratio = None
        fill_level = None

        if fill_model is not None:
            # Run fill model on full image (it detects water surface anywhere)
            r_fill = fill_model.predict(img, conf=0.25, iou=0.45,
                                        device=device, verbose=False)
            # Find water surface box closest to this bottle
            best_water = None
            best_water_iou = 0.0
            for box in r_fill[0].boxes:
                wx1, wy1, wx2, wy2 = [int(v) for v in box.xyxy[0].tolist()]
                water_cx = (wx1 + wx2) / 2
                # Water surface must be horizontally within bottle bounds
                if bx1 <= water_cx <= bx2 and by1 <= wy1 <= by2:
                    # Use highest confidence one
                    conf = float(box.conf[0])
                    if conf > best_water_iou:
                        best_water_iou = conf
                        best_water = [wx1, wy1, wx2, wy2]

            if best_water:
                fill_ratio = compute_fill_ratio(bottle["bbox"], best_water)
                fill_level = fill_verdict(fill_ratio)

        # ── Verdicts ──
        has_cap = best_cap is not None
        if has_cap:
            if classifier is not None:
                if cap_quality == "no_cap":
                    cap_verdict = "Missing Cap"
                elif cap_quality == "defective_cap":
                    cap_verdict = "Defective Cap"
                else:
                    cap_verdict = "Good Cap"
            else:
                cap_verdict = "Cap Present"
        else:
            cap_verdict = "Missing Cap"

        results_per_bottle.append({
            "bottle_idx": i,
            "bottle_bbox": bottle["bbox"],
            "cap_verdict": cap_verdict,
            "cap_quality": cap_quality,
            "cap_quality_conf": cap_quality_conf,
            "cap_bbox": cap_bbox,
            "fill_level": fill_level,
            "fill_ratio": fill_ratio,
        })

    latency = (time.perf_counter() - t0) * 1000.0
    return {
        "bottles": results_per_bottle,
        "caps_p1": det["caps_p1"],
        "caps_p2": det["caps_p2"],
        "latency_ms": round(latency, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw(img, result):
    out = img.copy()

    for br in result["bottles"]:
        bx1, by1, bx2, by2 = br["bottle_bbox"]

        # Bottle box
        cv2.rectangle(out, (bx1, by1), (bx2, by2), ORANGE, 2)
        cv2.putText(out, "bottle", (bx1+4, by2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)

        # Cap box
        if br["cap_bbox"]:
            cx1, cy1, cx2, cy2 = br["cap_bbox"]
            cap_v = br["cap_verdict"]
            cap_color = {
                "Good Cap": GREEN, "Defective Cap": RED, "Missing Cap": YELLOW,
                "Cap Present": CYAN,
            }.get(cap_v, CYAN)

            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), cap_color, 2)
            q_conf = br.get("cap_quality_conf", 0)
            q_label = f"{cap_v}"
            if q_conf > 0:
                q_label += f" {q_conf:.2f}"
            lw, lh = cv2.getTextSize(q_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(out, (cx1, cy1-lh-6), (cx1+lw+4, cy1), cap_color, -1)
            cv2.putText(out, q_label, (cx1+2, cy1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

        # Fill level badge (bottom of bottle box)
        fill = br.get("fill_level")
        ratio = br.get("fill_ratio")
        if fill:
            fill_color = {
                "normal": GREEN, "underfill": YELLOW, "overfill": RED
            }.get(fill, GRAY)
            ratio_str = f"{ratio:.2f}" if ratio is not None else ""
            fill_label = f"Fill: {fill} ({ratio_str})"
            lw, lh = cv2.getTextSize(fill_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(out, (bx1, by2), (bx1+lw+8, by2+lh+8), fill_color, -1)
            cv2.putText(out, fill_label, (bx1+4, by2+lh+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    # Info bar
    n = len(result["bottles"])
    info = f"{result['latency_ms']:.0f}ms | {n} bottle(s) | P2:{result['caps_p2']}"
    cv2.putText(out, info, (10, out.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: YOLO V3 + Cap Classifier + Fill Level Detection"
    )
    parser.add_argument("--weights", required=True,
                        help="YOLO bottle+cap detector weights (best.pt)")
    parser.add_argument("--fill-weights", required=True,
                        help="Water surface YOLO weights (water_surface_v1/best.pt)")
    parser.add_argument("--cls-weights", default="",
                        help="Cap classifier weights (optional)")
    parser.add_argument("--source", required=True,
                        help="Image, folder, or '0' for webcam")
    parser.add_argument("--device", default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--zoom", type=float, default=2.5)
    parser.add_argument("--show", action="store_true",
                        help="Show results in window")
    parser.add_argument("--underfill-thresh", type=float, default=UNDERFILL_THRESHOLD,
                        help=f"Fill ratio below this = underfill (default: {UNDERFILL_THRESHOLD})")
    parser.add_argument("--overfill-thresh", type=float, default=OVERFILL_THRESHOLD,
                        help=f"Fill ratio above this = overfill (default: {OVERFILL_THRESHOLD})")
    args = parser.parse_args()

    # Update thresholds from args
    global UNDERFILL_THRESHOLD, OVERFILL_THRESHOLD
    UNDERFILL_THRESHOLD = args.underfill_thresh
    OVERFILL_THRESHOLD = args.overfill_thresh

    # Load models
    detector = BottleCapDetector(args.weights, device=args.device,
                                  det_conf=args.conf, zoom_scale=args.zoom)
    fill_model = YOLO(args.fill_weights)

    classifier, cls_device = None, None
    if args.cls_weights and Path(args.cls_weights).exists():
        classifier, cls_device = load_classifier(args.cls_weights, args.device)

    # Output dir
    out_dir = Path("inference/annotated/yolo_with_fill_level")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Full Pipeline: YOLO V3 + Cap Classifier + Fill Level")
    print(f"{'='*60}")
    print(f"  Bottle/Cap YOLO : {args.weights}")
    print(f"  Fill YOLO       : {args.fill_weights}")
    print(f"  Cap Classifier  : {args.cls_weights or 'NONE (skipping cap classification)'}")
    print(f"  Underfill thresh: fill_ratio < {UNDERFILL_THRESHOLD}")
    print(f"  Overfill thresh : fill_ratio > {OVERFILL_THRESHOLD}")
    print(f"  Output          : {out_dir}")
    print()

    # ── Webcam ──
    if args.source in ("0", "webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Cannot open webcam"); return
        print("  Press 'q' to quit\n")
        while True:
            ret, frame = cap.read()
            if not ret: break
            result = full_inspect(detector, fill_model, classifier, cls_device,
                                  frame, args.device)
            frame = draw(frame, result)
            cv2.imshow("Full Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release(); cv2.destroyAllWindows()
        return

    # ── Images ──
    source = Path(args.source)
    assert source.exists(), f"Source not found: {source}"
    paths = [source] if source.is_file() else sorted(
        p for p in source.glob("*.*") if p.suffix.lower() in IMAGE_EXT
    )
    if not paths:
        print(f"  No images found at: {source}"); return

    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None: continue

        result = full_inspect(detector, fill_model, classifier, cls_device,
                              img, args.device)

        # Print per-bottle results
        for br in result["bottles"]:
            fill = br.get("fill_level") or "unknown"
            ratio = br.get("fill_ratio")
            ratio_str = f" (ratio={ratio:.2f})" if ratio else ""
            cap_icon = {"Good Cap": "✅", "Defective Cap": "⚠️",
                        "Missing Cap": "❌", "Cap Present": "✅"}.get(br["cap_verdict"], "?")
            fill_icon = {"normal": "🟢", "underfill": "🟡", "overfill": "🔴"}.get(fill, "⬜")
            print(f"  {cap_icon} {br['cap_verdict']:<15} | "
                  f"{fill_icon} Fill: {fill:<10}{ratio_str} | "
                  f"{img_path.name}  {result['latency_ms']:.0f}ms")

        annotated = draw(img, result)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        if args.show:
            cv2.imshow("Full Pipeline", annotated)
            cv2.waitKey(0)

    print(f"\n  Annotated → {out_dir}\n")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
