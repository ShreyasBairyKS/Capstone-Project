"""
inference/only_yolo.py
=======================
YOLO-only bottle + cap detection with V3 smart zoom.

Solves the small-object cap detection problem using 2-pass bottle-guided zoom:
    Pass 1: Full-frame detection → finds bottles (large, easy)
    Pass 2: Crops top 35% of each bottle → zooms 2.5x → re-detects caps

Output verdicts:
    ✅ Cap Present   — bottle + cap both detected
    ❌ Missing Cap   — bottle detected, no cap
    ⬜ No Bottle     — nothing detected

Usage:
    python inference/only_yolo.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source path/to/images

    python inference/only_yolo.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source 0 --show
"""

import argparse, time
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Colors (BGR)
GREEN  = (0, 200, 0)
RED    = (0, 0, 255)
ORANGE = (0, 140, 255)
GRAY   = (128, 128, 128)
CYAN   = (255, 255, 0)


class YOLOv3Detector:
    """2-pass YOLO detector with bottle-guided zoom for small cap detection."""

    def __init__(self, weights, device="0", det_conf=0.25, cap_conf=0.15,
                 zoom_scale=2.5, cap_region_ratio=0.35, crop_pad=0.15,
                 max_zoom_targets=5, skip_zoom_if_cap_conf=0.70):
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

    def _is_bottle(self, name): return "bottle" in name.lower()
    def _is_cap(self, name): return "cap" in name.lower()

    def detect(self, img):
        """Run 2-pass detection. Returns verdict + list of detections."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]

        # ── PASS 1: Full frame ──
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

        # Skip zoom if pass1 already found high-confidence caps
        best_cap_conf = max((c["conf"] for c in caps_p1), default=0.0)
        skip_zoom = best_cap_conf >= self.skip_zoom_if_cap_conf

        # ── PASS 2: Bottle-guided zoom ──
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
                if crop.size == 0:
                    continue

                # Zoom (upscale)
                ch, cw = crop.shape[:2]
                new_w = int(cw * self.zoom_scale)
                new_h = int(ch * self.zoom_scale)
                new_w = min(new_w, 960)
                new_h = min(new_h, 960)
                if new_w < 32 or new_h < 32:
                    continue
                zoomed = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Detect on zoomed crop
                r2 = self.model.predict(zoomed, conf=self.cap_conf, iou=0.45,
                                        device=self.device, verbose=False)

                # Map back to original coordinates
                scale_x = cw / new_w
                scale_y = ch / new_h

                for box in r2[0].boxes:
                    cls_id = int(box.cls[0])
                    name = self.names.get(cls_id, str(cls_id))
                    if not self._is_cap(name):
                        continue
                    zx1, zy1, zx2, zy2 = box.xyxy[0].tolist()
                    ox1 = int(cx1 + zx1 * scale_x)
                    oy1 = int(cy1 + zy1 * scale_y)
                    ox2 = int(cx1 + zx2 * scale_x)
                    oy2 = int(cy1 + zy2 * scale_y)
                    caps_p2.append({
                        "class": name, "conf": float(box.conf[0]),
                        "bbox": [ox1, oy1, ox2, oy2],
                        "source": "pass2_zoom",
                    })

        # ── Deduplicate caps ──
        all_caps = self._dedup(caps_p1, caps_p2)
        all_dets = [d for d in dets if self._is_bottle(d["class"])] + all_caps

        # ── Verdict ──
        has_bottle = len(bottles) > 0
        has_cap = len(all_caps) > 0
        if has_bottle and has_cap:
            verdict = "Cap Present"
        elif has_bottle:
            verdict = "Missing Cap"
        elif has_cap:
            verdict = "Cap Present"
        else:
            verdict = "No Bottle"

        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "detections": all_dets,
            "bottles": len(bottles),
            "caps_p1": len(caps_p1),
            "caps_p2": len(caps_p2),
            "caps_total": len(all_caps),
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
            if not dup:
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


def draw(img, result):
    """Draw bounding boxes and verdict banner."""
    out = img.copy()
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        is_cap = "cap" in det["class"].lower()
        from_zoom = det["source"] == "pass2_zoom"

        color = (CYAN if from_zoom else GREEN) if is_cap else ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        src = " [ZOOM]" if from_zoom else ""
        label = f"{det['class']} {det['conf']:.2f}{src}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Verdict banner
    v = result["verdict"]
    bc = {"Cap Present": GREEN, "Missing Cap": RED, "No Bottle": GRAY}.get(v, GRAY)
    cv2.rectangle(out, (0, 0), (350, 45), bc, -1)
    cv2.putText(out, v.upper(), (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # Info
    info = f"{result['latency_ms']:.0f}ms | P1:{result['caps_p1']} P2:{result['caps_p2']}"
    cv2.putText(out, info, (10, out.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="YOLO-only detection with V3 smart zoom")
    parser.add_argument("--weights", default="runs/detect/bottle_cap_det_v2/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image, folder, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--cap-conf", type=float, default=0.15, help="Cap confidence (pass 2)")
    parser.add_argument("--zoom", type=float, default=2.5, help="Zoom scale factor")
    parser.add_argument("--device", default="0", help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--show", action="store_true", help="Display results in window")
    args = parser.parse_args()

    detector = YOLOv3Detector(
        weights=args.weights, device=args.device,
        det_conf=args.conf, cap_conf=args.cap_conf, zoom_scale=args.zoom,
    )

    # Output directory
    out_dir = Path("inference/annotated/yolo_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print("  YOLO-Only Detection (V3 Smart Zoom)")
    print(f"{'='*55}")
    print(f"  Weights  : {args.weights}")
    print(f"  Conf     : {args.conf} (pass1) / {args.cap_conf} (pass2)")
    print(f"  Zoom     : {args.zoom}x")
    print(f"  Output   : {out_dir}")
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
            result = detector.detect(frame)
            frame = draw(frame, result)
            cv2.imshow("YOLO V3", frame)
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

        result = detector.detect(img)
        v = result["verdict"]
        counts[v] = counts.get(v, 0) + 1

        icon = {"Cap Present": "✅", "Missing Cap": "❌", "No Bottle": "⬜"}.get(v, "?")
        zoom = f"zoom:{result['caps_p2']}" if result["caps_p2"] else "p1"
        print(f"  {icon} {v:<15} {img_path.name:<40} "
              f"{result['latency_ms']:>5.0f}ms  caps={result['caps_total']}({zoom})")

        annotated = draw(img, result)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        if args.show:
            cv2.imshow("YOLO V3", annotated)
            cv2.waitKey(0)

    # Summary
    total = sum(counts.values())
    print(f"\n  {'─'*50}")
    print(f"  Total: {total}")
    for v, c in counts.items():
        if c > 0: print(f"  {v:<15}: {c}  ({100*c/total:.1f}%)")
    print(f"  Annotated → {out_dir}")
    print(f"  {'─'*50}\n")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
