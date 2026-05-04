"""
inference/yoloWithClassifier.py
================================
Full Hybrid Pipeline: YOLO V3 Detection + MobileNetV3 Cap Classification

Pipeline:
    1. YOLO Pass 1 (full frame) → detect bottles
    2. YOLO Pass 2 (smart zoom) → crop top 35% of each bottle → zoom 2.5x → detect caps
    3. Classifier              → crop each detected cap → MobileNetV3 → good/defective/no_cap

Output verdicts:
    ✅ Good Cap      — cap detected + classifier says good
    ⚠️ Defective Cap — cap detected + classifier says defective
    ❌ Missing Cap   — bottle found but no cap (or classifier says no_cap)
    ⬜ No Bottle     — nothing detected

Usage:
    python inference/yoloWithClassifier.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source path/to/images

    python inference/yoloWithClassifier.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source 0 --show
"""

import argparse, time
from pathlib import Path
import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Colors (BGR)
GREEN  = (0, 200, 0)
RED    = (0, 0, 255)
ORANGE = (0, 140, 255)
GRAY   = (128, 128, 128)
CYAN   = (255, 255, 0)
YELLOW = (0, 255, 255)

# Classifier class names (must match training order)
CLS_NAMES = ["good_cap", "defective_cap", "no_cap"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(weights_path, device):
    """Load trained MobileNetV3-Small cap quality classifier."""
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

    dev = torch.device(device if device != "0" else "cuda:0")
    model = CapClassifier()
    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(dev).eval()
    return model, dev


def classify_cap(classifier, device, crop_bgr, size=224):
    """Classify a cropped cap image → (class_name, confidence)."""
    from PIL import Image as PILImage

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)

    # Letterbox resize (preserve aspect ratio)
    w, h = pil.size
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pil = pil.resize((new_w, new_h), PILImage.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top  = (size - new_h) // 2
    canvas = PILImage.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(pil, (pad_left, pad_top))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(canvas).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs = F.softmax(logits, dim=1)
        cls_idx = probs.argmax(1).item()
        conf = probs[0, cls_idx].item()

    return CLS_NAMES[cls_idx], round(conf, 3)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO V3 DETECTOR (same as only_yolo.py)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv3Detector:
    """2-pass YOLO detector with bottle-guided zoom."""

    def __init__(self, weights, device="0", det_conf=0.25, cap_conf=0.15,
                 zoom_scale=2.5, cap_region_ratio=0.35, crop_pad=0.15,
                 max_zoom_targets=5, skip_zoom_if_cap_conf=0.70,
                 save_zoomed_images=False, zoomed_output_dir=None):
        self.model = YOLO(weights)
        self.names = self.model.names
        self.device = device
        self.det_conf = det_conf
        self.cap_conf = cap_conf
        self.zoom_scale = zoom_scale
        self.cap_region_ratio = cap_region_ratio
        self.crop_pad = crop_pad
        self.max_zoom_targets = max_zoom_targets
        self.skip_zoom_if_cap_conf = skip_zoom_if_cap_conf
        self.save_zoomed_images = save_zoomed_images
        self.zoomed_output_dir = Path(zoomed_output_dir) if zoomed_output_dir else None
        if self.save_zoomed_images and self.zoomed_output_dir is not None:
            self.zoomed_output_dir.mkdir(parents=True, exist_ok=True)

    def _is_bottle(self, name): return "bottle" in name.lower()
    def _is_cap(self, name): return "cap" in name.lower()

    def detect(self, img, source_name="image"):
        """Run 2-pass detection. Returns verdict + detections."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]

        # PASS 1: Full frame
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

        best_cap_conf = max((c["conf"] for c in caps_p1), default=0.0)
        skip_zoom = best_cap_conf >= self.skip_zoom_if_cap_conf

        # PASS 2: Bottle-guided zoom
        caps_p2 = []
        if bottles and not skip_zoom:
            sorted_bottles = sorted(bottles, key=lambda d: -d["conf"])[:self.max_zoom_targets]
            for bottle in sorted_bottles:
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

                if self.save_zoomed_images and self.zoomed_output_dir is not None:
                    zoom_name = f"{source_name}_bottle{len(caps_p2) + 1:02d}_zoom.jpg"
                    cv2.imwrite(str(self.zoomed_output_dir / zoom_name), zoomed)

                r2 = self.model.predict(zoomed, conf=self.cap_conf, iou=0.45,
                                        device=self.device, verbose=False)

                scale_x = cw / new_w
                scale_y = ch / new_h
                for box in r2[0].boxes:
                    cls_id = int(box.cls[0])
                    name = self.names.get(cls_id, str(cls_id))
                    if not self._is_cap(name): continue
                    zx1, zy1, zx2, zy2 = box.xyxy[0].tolist()
                    caps_p2.append({
                        "class": name, "conf": float(box.conf[0]),
                        "bbox": [int(cx1 + zx1*scale_x), int(cy1 + zy1*scale_y),
                                 int(cx1 + zx2*scale_x), int(cy1 + zy2*scale_y)],
                        "source": "pass2_zoom",
                    })

        # Deduplicate
        all_caps = self._dedup(caps_p1, caps_p2)
        all_dets = [d for d in dets if self._is_bottle(d["class"])] + all_caps

        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "detections": all_dets,
            "bottles": bottles,
            "caps": all_caps,
            "caps_p1": len(caps_p1),
            "caps_p2": len(caps_p2),
            "latency_ms": round(latency, 1),
        }

    def _dedup(self, caps_p1, caps_p2):
        if not caps_p2: return caps_p1
        if not caps_p1: return caps_p2
        merged = list(caps_p1)
        for c2 in caps_p2:
            dup = False
            for c1 in caps_p1:
                if self._iou(c1["bbox"], c2["bbox"]) > 0.3:
                    dup = True
                    if c2["conf"] > c1["conf"]:
                        merged[merged.index(c1)] = c2
                    break
            if not dup: merged.append(c2)
        return merged

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_inspect(detector, classifier, cls_device, img, source_name="image"):
    """
    Full pipeline:
        1. YOLO V3 detects bottles + caps (with zoom)
        2. Each detected cap is cropped and classified
        3. Returns verdict + annotated detections
    """
    # Step 1: YOLO detection
    det_result = detector.detect(img, source_name=source_name)
    h, w = img.shape[:2]

    bottles = det_result["bottles"]
    caps = det_result["caps"]

    # Step 2: Classify each detected cap
    cap_quality = None
    cap_quality_conf = 0.0

    for cap_det in caps:
        cx1, cy1, cx2, cy2 = cap_det["bbox"]
        # Add 10% padding around cap for context
        pad_x = int((cx2 - cx1) * 0.1)
        pad_y = int((cy2 - cy1) * 0.1)
        cx1 = max(0, cx1 - pad_x)
        cy1 = max(0, cy1 - pad_y)
        cx2 = min(w, cx2 + pad_x)
        cy2 = min(h, cy2 + pad_y)

        cap_crop = img[cy1:cy2, cx1:cx2]
        if cap_crop.size == 0:
            continue

        quality, conf = classify_cap(classifier, cls_device, cap_crop)
        cap_det["quality"] = quality
        cap_det["quality_conf"] = conf

        # Use worst quality among all caps
        if quality == "defective_cap":
            cap_quality = "defective_cap"
            cap_quality_conf = conf
        elif quality == "no_cap":
            if cap_quality is None:
                cap_quality = "no_cap"
                cap_quality_conf = conf
        elif cap_quality is None:
            cap_quality = "good_cap"
            cap_quality_conf = conf

    # Step 3: Final verdict
    has_bottle = len(bottles) > 0
    has_cap = len(caps) > 0

    if has_bottle and has_cap:
        if cap_quality == "defective_cap":
            verdict = "Defective Cap"
        elif cap_quality == "no_cap":
            verdict = "Missing Cap"
        elif cap_quality == "good_cap":
            verdict = "Good Cap"
        else:
            verdict = "Cap Present"
    elif has_bottle:
        verdict = "Missing Cap"
    elif has_cap:
        if cap_quality == "defective_cap":
            verdict = "Defective Cap"
        elif cap_quality == "good_cap":
            verdict = "Good Cap"
        else:
            verdict = "Cap Present"
    else:
        verdict = "No Bottle"

    return {
        "verdict": verdict,
        "cap_quality": cap_quality,
        "cap_quality_conf": cap_quality_conf,
        "detections": det_result["detections"],
        "bottles": len(bottles),
        "caps_total": len(caps),
        "caps_p1": det_result["caps_p1"],
        "caps_p2": det_result["caps_p2"],
        "latency_ms": det_result["latency_ms"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw(img, result):
    """Draw bounding boxes with classifier verdict."""
    out = img.copy()

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        is_cap = "cap" in det["class"].lower()
        from_zoom = det.get("source") == "pass2_zoom"
        quality = det.get("quality", "")

        # Color based on quality
        if is_cap:
            if quality == "defective_cap":
                color = RED
            elif quality == "good_cap":
                color = GREEN
            elif quality == "no_cap":
                color = YELLOW
            else:
                color = CYAN if from_zoom else GREEN
        else:
            color = ORANGE

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label
        src = " [ZOOM]" if from_zoom else ""
        q_label = ""
        if quality:
            q_conf = det.get("quality_conf", 0)
            q_label = f" | {quality.replace('_', ' ')} {q_conf:.2f}"
        label = f"{det['class']} {det['conf']:.2f}{src}{q_label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Verdict banner
    v = result["verdict"]
    bc = {
        "Good Cap": GREEN, "Defective Cap": RED,
        "Cap Present": CYAN, "Missing Cap": YELLOW, "No Bottle": GRAY,
    }.get(v, GRAY)
    cv2.rectangle(out, (0, 0), (400, 45), bc, -1)
    cv2.putText(out, v.upper(), (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # Info bar
    info = f"{result['latency_ms']:.0f}ms | P1:{result['caps_p1']} P2:{result['caps_p2']}"
    q = result.get("cap_quality", "")
    if q: info += f" | {q}"
    cv2.putText(out, info, (10, out.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Pipeline: YOLO V3 Detection + Cap Classification"
    )
    parser.add_argument("--weights", default="runs/detect/bottle_cap_det_v2/weights/best.pt",
                        help="YOLO detector weights")
    parser.add_argument("--cls-weights", default="models/cap_classifier_best.pth",
                        help="MobileNetV3 classifier weights")
    parser.add_argument("--source", required=True, help="Image, folder, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--cap-conf", type=float, default=0.15, help="Cap confidence (pass 2)")
    parser.add_argument("--zoom", type=float, default=2.5, help="Zoom scale factor")
    parser.add_argument("--device", default="0", help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--show", action="store_true", help="Display results in window")
    args = parser.parse_args()

    save_zoomed_answer = input("Save zoomed bottle crops before pass-2 detection? (y/n): ").strip().lower()
    save_zoomed_images = save_zoomed_answer in ("y", "yes")
    zoomed_output_dir = Path("inference/zoomed_image_yolo") if save_zoomed_images else None

    # Load YOLO detector
    detector = YOLOv3Detector(
        weights=args.weights, device=args.device,
        det_conf=args.conf, cap_conf=args.cap_conf, zoom_scale=args.zoom,
        save_zoomed_images=save_zoomed_images,
        zoomed_output_dir=zoomed_output_dir,
    )

    # Load classifier
    assert Path(args.cls_weights).exists(), f"Classifier weights not found: {args.cls_weights}"
    classifier, cls_device = load_classifier(args.cls_weights, args.device)

    # Output directory
    out_dir = Path("inference/annotated/yolo_with_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  Hybrid Pipeline: YOLO V3 + Cap Classifier")
    print(f"{'='*55}")
    print(f"  YOLO weights : {args.weights}")
    print(f"  Cls weights  : {args.cls_weights}")
    print(f"  Classes      : {CLS_NAMES}")
    print(f"  Conf         : {args.conf} (pass1) / {args.cap_conf} (pass2)")
    print(f"  Zoom         : {args.zoom}x")
    print(f"  Output       : {out_dir}")
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
            result = hybrid_inspect(detector, classifier, cls_device, frame, source_name=f"webcam_{int(time.time() * 1000)}")
            frame = draw(frame, result)
            cv2.imshow("Hybrid Pipeline", frame)
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

    counts = {}
    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None: continue

        result = hybrid_inspect(detector, classifier, cls_device, img, source_name=img_path.stem)
        v = result["verdict"]
        counts[v] = counts.get(v, 0) + 1

        icon = {"Good Cap": "✅", "Defective Cap": "⚠️", "Cap Present": "✅",
                "Missing Cap": "❌", "No Bottle": "⬜"}.get(v, "?")
        q = result.get("cap_quality", "") or ""
        q_info = f" [{q}]" if q else ""
        zoom = f"zoom:{result['caps_p2']}" if result["caps_p2"] else "p1"
        print(f"  {icon} {v:<17} {img_path.name:<38} "
              f"{result['latency_ms']:>5.0f}ms  caps={result['caps_total']}({zoom}){q_info}")

        annotated = draw(img, result)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        if args.show:
            cv2.imshow("Hybrid Pipeline", annotated)
            cv2.waitKey(0)

    # Summary
    total = sum(counts.values())
    print(f"\n  {'─'*50}")
    print(f"  Total: {total}")
    for v, c in counts.items():
        if c > 0: print(f"  {v:<17}: {c}  ({100*c/total:.1f}%)")
    print(f"  Annotated → {out_dir}")
    print(f"  {'─'*50}\n")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
