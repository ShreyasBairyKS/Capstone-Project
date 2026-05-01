"""
inference/hybrid_inference.py
==============================
Two-stage hybrid inference pipeline for bottle cap quality inspection.

Stage 1: YOLOv8 detection → find bottles and caps
Stage 2: MobileNetV3 classification → good_cap or defective_cap

Decision logic:
    - Bottle detected, no cap → "Missing Cap" ❌
    - Cap detected + classifier says good → "Good Cap" ✅
    - Cap detected + classifier says defective → "Defective Cap" ⚠️
    - No bottle detected → "No Bottle" (skip)

Usage:
    python inference/hybrid_inference.py \
        --det-weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --source path/to/images \
        --save

    # Single image:
    python inference/hybrid_inference.py \
        --det-weights best_det.pt \
        --cls-weights best_cls.pth \
        --source test.jpg \
        --save
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Classification class names (must match training order)
CLS_NAMES = ["good_cap", "defective_cap"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Verdict colors (BGR for OpenCV)
COLORS = {
    "Good Cap": (0, 200, 0),        # green
    "Defective Cap": (0, 0, 255),    # red
    "Missing Cap": (0, 140, 255),    # orange
    "No Bottle": (128, 128, 128),    # gray
}


# ─────────────────────────────────────────────────────────────────────────────
# LETTERBOX (aspect-ratio preserving)
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_crop(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio with gray padding."""
    h, w = crop.shape[:2]
    ratio = min(size / h, size / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, size - new_h - pad_top,
        pad_left, size - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """Rebuild MobileNetV3-Small architecture (must match training)."""
    from torchvision.models import mobilenet_v3_small

    backbone = mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features

    class CapQualityClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    return CapQualityClassifier()


def load_classifier(weights_path: str, device: torch.device) -> nn.Module:
    """Load trained classifier weights."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    num_classes = len(checkpoint.get("class_names", CLS_NAMES))

    model = build_classifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING FOR CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_cap_crop(crop_bgr: np.ndarray, imgsz: int = 224) -> torch.Tensor:
    """Preprocess a BGR cap crop for MobileNetV3 input."""
    # Letterbox resize
    crop_lb = letterbox_crop(crop_bgr, imgsz)
    # BGR → RGB → float → normalize
    rgb = cv2.cvtColor(crop_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class HybridInspector:
    """Two-stage bottle cap inspection pipeline."""

    def __init__(
        self,
        det_weights: str,
        cls_weights: str,
        device: str = "0",
        det_conf: float = 0.25,
        det_iou: float = 0.45,
        cls_imgsz: int = 224,
        cap_context_pad: float = 0.1,
        bottle_class: str = "bottle",
        cap_class: str = "cap",
    ):
        from ultralytics import YOLO

        self.device_str = device
        self.torch_device = torch.device(
            f"cuda:{device}" if device.isdigit() else device
        )

        # Stage 1: Detector
        self.detector = YOLO(det_weights)
        self.det_conf = det_conf
        self.det_iou = det_iou

        # Stage 2: Classifier
        self.classifier = load_classifier(cls_weights, self.torch_device)
        self.cls_imgsz = cls_imgsz

        self.cap_context_pad = cap_context_pad
        self.bottle_class = bottle_class
        self.cap_class = cap_class

    def inspect(self, img: np.ndarray) -> dict:
        """
        Run full inspection on a single BGR image.

        Returns dict with:
            verdict: "Good Cap" / "Defective Cap" / "Missing Cap" / "No Bottle"
            detections: list of detection dicts
            classification: classifier output (if cap found)
            latency_ms: total inference time
        """
        t0 = time.perf_counter()
        h, w = img.shape[:2]

        # Stage 1: Detection
        results = self.detector.predict(
            img, conf=self.det_conf, iou=self.det_iou,
            device=self.device_str, verbose=False,
        )

        bottles = []
        caps = []
        all_detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            det = {
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            }
            all_detections.append(det)

            if cls_name == self.bottle_class:
                bottles.append(det)
            elif cls_name == self.cap_class:
                caps.append(det)

        # Decision logic
        classification = None

        if not bottles and not caps:
            verdict = "No Bottle"
        elif bottles and not caps:
            verdict = "Missing Cap"
        elif caps:
            # Classify the highest-confidence cap
            best_cap = max(caps, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best_cap["bbox"]

            # Add context padding
            bw, bh = x2 - x1, y2 - y1
            pad_w = int(bw * self.cap_context_pad)
            pad_h = int(bh * self.cap_context_pad)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)

            cap_crop = img[cy1:cy2, cx1:cx2]
            if cap_crop.size > 0:
                classification = self._classify_cap(cap_crop)
                if classification["label"] == "good_cap":
                    verdict = "Good Cap"
                else:
                    verdict = "Defective Cap"
            else:
                verdict = "Missing Cap"
        else:
            verdict = "Missing Cap"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "detections": all_detections,
            "classification": classification,
            "n_bottles": len(bottles),
            "n_caps": len(caps),
            "latency_ms": round(latency_ms, 2),
        }

    @torch.no_grad()
    def _classify_cap(self, cap_crop_bgr: np.ndarray) -> dict:
        """Run MobileNetV3 classifier on a cap crop."""
        tensor = preprocess_cap_crop(cap_crop_bgr, self.cls_imgsz)
        tensor = tensor.to(self.torch_device)

        logits = self.classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

        return {
            "label": CLS_NAMES[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "probabilities": {
                name: round(probs[i].item(), 4)
                for i, name in enumerate(CLS_NAMES)
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_results(img: np.ndarray, result: dict) -> np.ndarray:
    """Draw bounding boxes and verdict on image."""
    annotated = img.copy()
    verdict = result["verdict"]
    color = COLORS.get(verdict, (255, 255, 255))

    # Draw all detection boxes
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        box_color = (0, 200, 0) if cls_name == "cap" else (255, 180, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw verdict banner
    banner_text = f"  {verdict}  "
    if result.get("classification"):
        banner_text += f"({result['classification']['confidence']:.0%})"

    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(annotated, (0, 0), (tw + 20, th + 20), color, -1)
    cv2.putText(annotated, banner_text, (10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Latency
    lat_text = f"{result['latency_ms']:.1f}ms"
    cv2.putText(annotated, lat_text, (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid bottle cap inspection: detection + classification"
    )
    parser.add_argument("--det-weights", required=True,
                        help="Path to YOLOv8 detector .pt weights")
    parser.add_argument("--cls-weights", required=True,
                        help="Path to classifier .pth weights")
    parser.add_argument("--source", required=True,
                        help="Image path or folder of images")
    parser.add_argument("--det-conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--det-iou", type=float, default=0.45,
                        help="Detection NMS IoU threshold")
    parser.add_argument("--cls-imgsz", type=int, default=224,
                        help="Classifier input size")
    parser.add_argument("--device", default="0",
                        help="Device: '0' GPU, 'cpu' CPU")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated images")
    parser.add_argument("--output", default="runs/hybrid_infer",
                        help="Output directory")
    parser.add_argument("--bottle-class", default="bottle",
                        help="Detection class name for bottle")
    parser.add_argument("--cap-class", default="cap",
                        help="Detection class name for cap")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output)
    assert source.exists(), f"Source not found: {source}"

    # Collect images
    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(
            p for p in source.glob("*.*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        print(f"No images found at: {source}")
        return

    print(f"\n{'='*60}")
    print("  Hybrid Bottle Cap Inspection — Inference")
    print(f"{'='*60}")
    print(f"  Detector  : {args.det_weights}")
    print(f"  Classifier: {args.cls_weights}")
    print(f"  Source     : {source}")
    print(f"  Images     : {len(image_paths)}")
    print(f"  Device     : {args.device}")
    print()

    inspector = HybridInspector(
        det_weights=args.det_weights,
        cls_weights=args.cls_weights,
        device=args.device,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        cls_imgsz=args.cls_imgsz,
        bottle_class=args.bottle_class,
        cap_class=args.cap_class,
    )

    all_results = []
    verdicts_count = {"Good Cap": 0, "Defective Cap": 0, "Missing Cap": 0, "No Bottle": 0}

    if args.save:
        (output_dir / "annotated").mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = inspector.inspect(img)
        result["image_path"] = str(img_path)
        all_results.append(result)

        v = result["verdict"]
        verdicts_count[v] = verdicts_count.get(v, 0) + 1

        icon = {"Good Cap": "✅", "Defective Cap": "⚠️", "Missing Cap": "❌", "No Bottle": "⬜"}.get(v, "?")
        print(f"  {icon} {v:<15}  {img_path.name:<40}  {result['latency_ms']:.1f}ms")

        if args.save:
            annotated = draw_results(img, result)
            cv2.imwrite(str(output_dir / "annotated" / img_path.name), annotated)

    # Summary
    total = len(all_results)
    avg_lat = sum(r["latency_ms"] for r in all_results) / max(total, 1)
    print(f"\n  ─────────────────────────────────────────")
    print(f"  Total inspected : {total}")
    for v, c in verdicts_count.items():
        if c > 0:
            print(f"  {v:<15} : {c}  ({100*c/total:.1f}%)")
    print(f"  Avg latency     : {avg_lat:.1f}ms")
    print(f"  ─────────────────────────────────────────")

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hybrid_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results → {json_path}")
    if args.save:
        print(f"  Annotated images → {output_dir / 'annotated'}")
    print()


if __name__ == "__main__":
    main()
