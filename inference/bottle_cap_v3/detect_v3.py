"""
inference/bottle_cap_v3/detect_v3.py
======================================
Bottle-Guided 2-Pass Cap Detection + Classification (V3 Hybrid)

Full pipeline:
    Pass 1 (full frame)  → Detect bottles (large, easy)
    Pass 2 (smart zoom)  → Crop top 35% of each bottle → upscale → re-detect caps
    Pass 3 (classify)    → Crop each detected cap → MobileNetV3 → good/defective

Verdicts:
    ✅ Good Cap      — cap detected + classifier says good
    ⚠️ Defective Cap — cap detected + classifier says defective
    ❌ Missing Cap   — bottle found but no cap
    ⬜ No Bottle     — nothing detected

Usage:
    # Detection only (no classifier):
    python inference/bottle_cap_v3/detect_v3.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source path/to/images

    # Full hybrid (detection + classification):
    python inference/bottle_cap_v3/detect_v3.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source path/to/images

    # Webcam:
    python inference/bottle_cap_v3/detect_v3.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source 0 --show
"""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from ultralytics import YOLO

try:
    from inference.bottle_cap_v3.config import V3Config, PRESETS
except ImportError:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from inference.bottle_cap_v3.config import V3Config, PRESETS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
GREEN, RED, ORANGE, GRAY, CYAN = (0,200,0), (0,0,255), (0,140,255), (128,128,128), (255,255,0)
YELLOW = (0, 255, 255)

CLS_NAMES = ["good_cap", "defective_cap", "no_cap"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── CLASSIFIER LOADER ──

def load_classifier(weights_path: str, device: str):
    """Load trained MobileNetV3 cap quality classifier."""
    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn

    backbone = mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features

    class CapQualityClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.dropout = nn.Dropout(p=0.3)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, 3),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    dev = torch.device(device if device != "0" else "cuda:0")
    model = CapQualityClassifier()
    ckpt = torch.load(weights_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(dev).eval()
    return model, dev


def classify_cap_crop(classifier, device, crop_bgr: np.ndarray, size: int = 224):
    """Classify a cap crop as good_cap or defective_cap."""
    from PIL import Image
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Letterbox resize
    w, h = pil.size
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
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
        confidence = probs[0, cls_idx].item()

    return CLS_NAMES[cls_idx], confidence


class BottleCapV3:
    """2-pass bottle-guided cap detection + classification pipeline."""

    def __init__(self, cfg: V3Config, classifier=None, cls_device=None):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.classifier = classifier
        self.cls_device = cls_device
        self.class_names = self.model.names  # {0: 'bottle', 1: 'cap', ...}

    def _is_bottle(self, name: str) -> bool:
        return self.cfg.bottle_keyword in name.lower()

    def _is_cap(self, name: str) -> bool:
        return self.cfg.cap_keyword in name.lower()

    def _parse_boxes(self, results) -> list[dict]:
        dets = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = self.class_names.get(cls_id, str(cls_id))
            dets.append({
                "class": name, "conf": float(box.conf[0]),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                "source": "pass1",
            })
        return dets

    def _get_cap_crop_region(self, bottle_bbox, img_h, img_w):
        """Get the top region of a bottle where the cap lives."""
        x1, y1, x2, y2 = bottle_bbox
        bw, bh = x2 - x1, y2 - y1

        # Cap is in the top portion of the bottle
        cap_h = int(bh * self.cfg.cap_region_ratio)

        # Add context padding
        pad_w = int(bw * self.cfg.crop_pad)
        pad_h = int(cap_h * self.cfg.crop_pad)

        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(img_w, x2 + pad_w)
        cy2 = min(img_h, y1 + cap_h + pad_h)

        return cx1, cy1, cx2, cy2

    def _zoom_crop(self, crop: np.ndarray) -> np.ndarray:
        """Upscale crop to improve small-object detection."""
        h, w = crop.shape[:2]
        scale = self.cfg.zoom_scale
        new_w, new_h = int(w * scale), int(h * scale)

        # Clamp to max size
        max_px = self.cfg.max_zoom_px
        if max(new_w, new_h) > max_px:
            ratio = max_px / max(new_w, new_h)
            new_w, new_h = int(new_w * ratio), int(new_h * ratio)

        if new_w < self.cfg.min_crop_px or new_h < self.cfg.min_crop_px:
            return None
        return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def inspect(self, img: np.ndarray) -> dict:
        """Run full 2-pass inspection on a BGR image."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        cfg = self.cfg

        # ── PASS 1: Full-frame detection ──
        r1 = self.model.predict(img, conf=cfg.det_conf, iou=cfg.iou_thresh,
                                device=cfg.device, verbose=False)
        all_dets = self._parse_boxes(r1)

        bottles = [d for d in all_dets if self._is_bottle(d["class"])]
        caps_p1 = [d for d in all_dets if self._is_cap(d["class"])]

        # Check if pass 1 already found high-confidence caps
        best_cap_conf = max((c["conf"] for c in caps_p1), default=0.0)
        skip_zoom = best_cap_conf >= cfg.skip_zoom_if_cap_conf

        # ── PASS 2: Bottle-guided zoom for cap detection ──
        zoom_crops = []
        caps_p2 = []
        debug_crops = []  # For visual debugging

        if bottles and not skip_zoom:
            # Sort bottles by confidence, limit zoom targets
            sorted_bottles = sorted(bottles, key=lambda d: -d["conf"])[:cfg.max_zoom_targets]

            for i, bottle in enumerate(sorted_bottles):
                bx1, by1, bx2, by2 = bottle["bbox"]

                # Get cap region (top of bottle)
                cx1, cy1, cx2, cy2 = self._get_cap_crop_region(
                    bottle["bbox"], h, w
                )
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # Zoom (upscale)
                zoomed = self._zoom_crop(crop)
                if zoomed is None:
                    continue

                zoom_crops.append({
                    "origin": (cx1, cy1, cx2, cy2),
                    "crop_size": crop.shape[:2],
                    "zoom_size": zoomed.shape[:2],
                })

                # Run model on zoomed crop
                r2 = self.model.predict(zoomed, conf=cfg.cap_conf,
                                        iou=cfg.iou_thresh, device=cfg.device,
                                        verbose=False)

                # Save debug crops (the raw crop + zoomed + annotated)
                if cfg.save_crops:
                    zoomed_annotated = r2[0].plot()
                    debug_crops.append({
                        "idx": i,
                        "region": (cx1, cy1, cx2, cy2),
                        "raw_crop": crop.copy(),
                        "zoomed": zoomed.copy(),
                        "zoomed_annotated": zoomed_annotated,
                        "detections_in_zoom": len(r2[0].boxes),
                    })

                # Map detections back to original image coordinates
                zh, zw = zoomed.shape[:2]
                crop_h, crop_w = crop.shape[:2]
                scale_x = crop_w / zw
                scale_y = crop_h / zh

                for box in r2[0].boxes:
                    cls_id = int(box.cls[0])
                    name = self.class_names.get(cls_id, str(cls_id))
                    if not self._is_cap(name):
                        continue  # Only keep cap detections from zoom

                    zx1, zy1, zx2, zy2 = box.xyxy[0].tolist()
                    # Scale back to crop coords, then to original coords
                    ox1 = int(cx1 + zx1 * scale_x)
                    oy1 = int(cy1 + zy1 * scale_y)
                    ox2 = int(cx1 + zx2 * scale_x)
                    oy2 = int(cy1 + zy2 * scale_y)

                    caps_p2.append({
                        "class": name, "conf": float(box.conf[0]),
                        "bbox": [ox1, oy1, ox2, oy2],
                        "source": "pass2_zoom",
                    })

        # ── MERGE: Deduplicate caps from both passes ──
        all_caps = self._deduplicate_caps(caps_p1, caps_p2)
        all_dets_final = [d for d in all_dets if self._is_bottle(d["class"])] + all_caps

        # ── CLASSIFY each detected cap ──
        cap_quality = None
        cap_quality_conf = 0.0
        if all_caps and self.classifier is not None:
            for cap_det in all_caps:
                cx1, cy1, cx2, cy2 = cap_det["bbox"]
                # Add small padding for context
                pad_x = int((cx2 - cx1) * 0.1)
                pad_y = int((cy2 - cy1) * 0.1)
                cx1 = max(0, cx1 - pad_x)
                cy1 = max(0, cy1 - pad_y)
                cx2 = min(w, cx2 + pad_x)
                cy2 = min(h, cy2 + pad_y)

                cap_crop = img[cy1:cy2, cx1:cx2]
                if cap_crop.size == 0:
                    continue

                quality, conf = classify_cap_crop(
                    self.classifier, self.cls_device, cap_crop
                )
                cap_det["quality"] = quality
                cap_det["quality_conf"] = round(conf, 3)

                # Use worst quality among all caps for verdict
                if quality == "defective_cap":
                    cap_quality = "defective_cap"
                    cap_quality_conf = conf
                elif quality == "no_cap":
                    # Classifier says this isn't actually a cap (false positive)
                    if cap_quality is None:
                        cap_quality = "no_cap"
                        cap_quality_conf = conf
                elif cap_quality is None:
                    cap_quality = "good_cap"
                    cap_quality_conf = conf

        # ── VERDICT ──
        has_bottle = len(bottles) > 0
        has_cap = len(all_caps) > 0

        if has_bottle and has_cap:
            if cap_quality == "defective_cap":
                verdict = "Defective Cap"
            elif cap_quality == "no_cap":
                verdict = "Missing Cap"  # classifier overrides: not a real cap
            elif cap_quality == "good_cap":
                verdict = "Good Cap"
            else:
                verdict = "Cap Present"  # no classifier loaded
        elif has_bottle and not has_cap:
            verdict = "Missing Cap"
        elif has_cap and not has_bottle:
            if cap_quality == "defective_cap":
                verdict = "Defective Cap"
            elif cap_quality == "no_cap":
                verdict = "Missing Cap"
            elif cap_quality == "good_cap":
                verdict = "Good Cap"
            else:
                verdict = "Cap Present"
        else:
            verdict = "No Bottle"

        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "cap_quality": cap_quality,
            "cap_quality_conf": round(cap_quality_conf, 3),
            "bottles": len(bottles),
            "caps_pass1": len(caps_p1),
            "caps_pass2": len(caps_p2),
            "caps_total": len(all_caps),
            "zoom_skipped": skip_zoom,
            "zoom_crops": len(zoom_crops),
            "debug_crops": debug_crops,
            "detections": all_dets_final,
            "latency_ms": round(latency, 1),
        }

    def _deduplicate_caps(self, caps_p1, caps_p2):
        """Remove duplicate cap detections (keep highest confidence)."""
        if not caps_p1 and not caps_p2:
            return []
        if not caps_p2:
            return caps_p1
        if not caps_p1:
            return caps_p2

        # For each p2 cap, check IoU with p1 caps
        merged = list(caps_p1)
        for c2 in caps_p2:
            is_dup = False
            for c1 in caps_p1:
                if self._iou(c1["bbox"], c2["bbox"]) > 0.3:
                    is_dup = True
                    # Keep the higher confidence one
                    if c2["conf"] > c1["conf"]:
                        idx = merged.index(c1)
                        merged[idx] = c2
                    break
            if not is_dup:
                merged.append(c2)
        return merged

    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / max(a1 + a2 - inter, 1e-6)


# ── DRAWING ──

def draw_v3(img, result, show_debug=False):
    """Draw bounding boxes and verdict banner."""
    annotated = img.copy()
    verdict = result["verdict"]

    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        is_cap = "cap" in det["class"].lower()
        from_zoom = det.get("source") == "pass2_zoom"
        quality = det.get("quality", "")

        if is_cap:
            if quality == "defective_cap":
                color = RED
            elif quality == "good_cap":
                color = GREEN
            else:
                color = CYAN if from_zoom else GREEN
        else:
            color = ORANGE

        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        src = " [ZOOM]" if from_zoom else ""
        q_label = ""
        if quality:
            q_conf = det.get('quality_conf', 0)
            q_label = f" | {quality.replace('_',' ')} {q_conf:.2f}"
        label = f"{det['class']} {det['conf']:.2f}{src}{q_label}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Verdict banner
    colors = {
        "Good Cap": GREEN, "Defective Cap": RED,
        "Cap Present": CYAN, "Missing Cap": YELLOW, "No Bottle": GRAY,
    }
    bc = colors.get(verdict, GRAY)
    cv2.rectangle(annotated, (0,0), (400, 45), bc, -1)
    cv2.putText(annotated, verdict.upper(), (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # Info bar
    info = f"{result['latency_ms']:.0f}ms | P1:{result['caps_pass1']} P2:{result['caps_pass2']}"
    if result.get("zoom_skipped"):
        info += " | zoom:skip"
    cv2.putText(annotated, info, (10, annotated.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    return annotated


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser(
        description="V3 Bottle-Guided 2-Pass Cap Detection + Classification"
    )
    parser.add_argument("--weights", default="runs/detect/bottle_cap_det_v2/weights/best.pt")
    parser.add_argument("--cls-weights", default="",
                        help="Path to classifier .pth (optional — enables quality classification)")
    parser.add_argument("--source", required=True, help="Image, folder, or '0' for webcam")
    parser.add_argument("--preset", choices=["default","jetson","debug"], default="default")
    parser.add_argument("--conf", type=float, default=None, help="Override det_conf")
    parser.add_argument("--cap-conf", type=float, default=None, help="Override cap_conf")
    parser.add_argument("--zoom", type=float, default=None, help="Override zoom_scale")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=None, help="Override output dir")
    args = parser.parse_args()

    # Build config from preset + overrides
    cfg = PRESETS[args.preset]()
    cfg.weights = args.weights
    if args.conf: cfg.det_conf = args.conf
    if args.cap_conf: cfg.cap_conf = args.cap_conf
    if args.zoom: cfg.zoom_scale = args.zoom
    if args.device: cfg.device = args.device
    if args.output: cfg.output_dir = args.output
    cfg.show_window = args.show

    # Load classifier if provided
    classifier, cls_device = None, None
    if args.cls_weights and Path(args.cls_weights).exists():
        print(f"  Loading classifier: {args.cls_weights}")
        classifier, cls_device = load_classifier(
            args.cls_weights, cfg.device
        )
        print(f"  Classifier loaded → {CLS_NAMES}")

    print(f"\n{'='*55}")
    print("  Bottle Cap V3 — Hybrid Detection + Classification")
    print(f"{'='*55}")
    print(f"  Det weights : {cfg.weights}")
    print(f"  Cls weights : {args.cls_weights or 'NONE (detection only)'}")
    print(f"  Preset      : {args.preset}")
    print(f"  Det conf    : {cfg.det_conf}")
    print(f"  Cap conf    : {cfg.cap_conf} (pass 2)")
    print(f"  Zoom scale  : {cfg.zoom_scale}x")
    print(f"  Cap region  : top {cfg.cap_region_ratio*100:.0f}% of bottle")
    print()

    pipeline = BottleCapV3(cfg, classifier=classifier, cls_device=cls_device)
    output_dir = Path(cfg.output_dir)

    # ── Webcam ──
    if args.source in ("0", "webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Cannot open webcam"); return
        print("  Press 'q' to quit\n")
        while True:
            ret, frame = cap.read()
            if not ret: break
            result = pipeline.inspect(frame)
            frame = draw_v3(frame, result)
            cv2.imshow("Bottle Cap V3", frame)
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

    ann_dir = output_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        (output_dir / "crops").mkdir(parents=True, exist_ok=True)

    all_results = []
    counts = {}

    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None: continue

        result = pipeline.inspect(img)
        result["image"] = img_path.name
        all_results.append(result)
        v = result["verdict"]
        counts[v] = counts.get(v, 0) + 1

        icon = {"Good Cap":"✅", "Defective Cap":"⚠️", "Cap Present":"✅",
                "Missing Cap":"❌", "No Bottle":"⬜"}.get(v,"?")
        q = result.get('cap_quality', '')
        q_info = f" [{q}]" if q else ""
        zoom_info = f"zoom:{result['caps_pass2']}" if result['caps_pass2'] else "p1"
        print(f"  {icon} {v:<17} {img_path.name:<38} "
              f"{result['latency_ms']:>6.0f}ms  caps={result['caps_total']}({zoom_info}){q_info}")

        # Save annotated
        annotated = draw_v3(img, result, show_debug=(args.preset=="debug"))
        cv2.imwrite(str(ann_dir / img_path.name), annotated)

        # Save debug zoom crops
        if cfg.save_crops and result.get("debug_crops"):
            crops_dir = output_dir / "crops" / img_path.stem
            crops_dir.mkdir(parents=True, exist_ok=True)
            for dc in result["debug_crops"]:
                prefix = f"bottle{dc['idx']}"
                cv2.imwrite(str(crops_dir / f"{prefix}_1_raw_crop.jpg"), dc["raw_crop"])
                cv2.imwrite(str(crops_dir / f"{prefix}_2_zoomed.jpg"), dc["zoomed"])
                cv2.imwrite(str(crops_dir / f"{prefix}_3_zoomed_detected.jpg"), dc["zoomed_annotated"])
            # Also draw crop region on a debug copy
            debug_img = img.copy()
            for dc in result["debug_crops"]:
                cx1, cy1, cx2, cy2 = dc["region"]
                color = GREEN if dc["detections_in_zoom"] > 0 else RED
                cv2.rectangle(debug_img, (cx1, cy1), (cx2, cy2), color, 3)
                cv2.putText(debug_img, f"ZOOM REGION {dc['idx']} ({dc['detections_in_zoom']} dets)",
                            (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(str(crops_dir / "00_zoom_regions.jpg"), debug_img)

        if cfg.show_window:
            disp = annotated
            if disp.shape[1] > cfg.window_width:
                r = cfg.window_width / disp.shape[1]
                disp = cv2.resize(disp, (cfg.window_width, int(disp.shape[0]*r)))
            cv2.imshow("V3", disp)
            cv2.waitKey(0)

    # Summary
    total = len(all_results)
    avg_lat = sum(r["latency_ms"] for r in all_results) / max(total,1)
    zoom_used = sum(1 for r in all_results if r["caps_pass2"] > 0)
    print(f"\n  {'─'*50}")
    print(f"  Total     : {total}")
    for v, c in counts.items():
        if c > 0: print(f"  {v:<15}: {c}  ({100*c/total:.1f}%)")
    print(f"  Avg latency: {avg_lat:.0f}ms")
    print(f"  Zoom used  : {zoom_used}/{total} images")
    print(f"  {'─'*50}")

    if cfg.save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        jp = output_dir / "results.json"
        # Strip non-serializable debug crops before saving
        json_results = [{k: v for k, v in r.items() if k != "debug_crops"} for r in all_results]
        with open(jp, "w") as f: json.dump(json_results, f, indent=2)
        print(f"\n  Results   → {jp}")
    print(f"  Annotated → {ann_dir}\n")

    if cfg.show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
