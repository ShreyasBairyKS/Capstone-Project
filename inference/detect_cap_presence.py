"""
inference/detect_cap_presence.py
=================================
Simple inference script for the 2-class bottle+cap detector.

Logic:
    - Bottle detected + cap detected  → "Cap Present ✅"
    - Bottle detected + NO cap        → "Missing Cap ❌"
    - Only cap detected (no bottle)   → "Cap Present ✅"
    - Nothing detected                → "No Bottle ⬜"

Usage:
    # Single image:
    python inference/detect_cap_presence.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source test.jpg

    # Folder:
    python inference/detect_cap_presence.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source path/to/images/ \
        --save

    # Webcam:
    python inference/detect_cap_presence.py \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --source 0 \
        --show
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Colors (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)
GRAY = (128, 128, 128)


def inspect_image(model, img_path, conf, iou, device):
    """Run detection and determine cap presence."""
    results = model.predict(
        str(img_path), conf=conf, iou=iou,
        device=device, verbose=False,
    )

    bottles = []
    caps = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names.get(cls_id, str(cls_id)).lower()
        c = float(box.conf[0])
        bbox = [int(v) for v in box.xyxy[0].tolist()]

        det = {"class": cls_name, "conf": round(c, 3), "bbox": bbox}

        if "cap" in cls_name:
            caps.append(det)
        elif "bottle" in cls_name:
            bottles.append(det)

    # Decision logic
    if bottles and caps:
        verdict = "Cap Present"
    elif bottles and not caps:
        verdict = "Missing Cap"
    elif caps and not bottles:
        verdict = "Cap Present"
    else:
        verdict = "No Bottle"

    return {
        "verdict": verdict,
        "bottles": len(bottles),
        "caps": len(caps),
        "detections": bottles + caps,
    }


def draw_result(img, result):
    """Draw boxes and verdict on image."""
    verdict = result["verdict"]

    # Draw detection boxes
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        is_cap = "cap" in det["class"]
        color = GREEN if is_cap else (255, 180, 0)
        label = f"{det['class']} {det['conf']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Verdict banner
    if verdict == "Cap Present":
        banner_color = GREEN
        icon = "CAP PRESENT"
    elif verdict == "Missing Cap":
        banner_color = RED
        icon = "MISSING CAP"
    else:
        banner_color = GRAY
        icon = "NO BOTTLE"

    cv2.rectangle(img, (0, 0), (300, 40), banner_color, -1)
    cv2.putText(img, icon, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def run_on_webcam(model, conf, iou, device):
    """Run live inference on webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("  Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily
        result = inspect_image(model, frame, conf, iou, device)
        # For webcam, run predict directly on frame
        results = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)

        bottles, caps = [], []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names.get(cls_id, str(cls_id)).lower()
            c = float(box.conf[0])
            bbox = [int(v) for v in box.xyxy[0].tolist()]
            det = {"class": cls_name, "conf": round(c, 3), "bbox": bbox}
            if "cap" in cls_name:
                caps.append(det)
            elif "bottle" in cls_name:
                bottles.append(det)

        if bottles and caps:
            verdict = "Cap Present"
        elif bottles and not caps:
            verdict = "Missing Cap"
        elif caps:
            verdict = "Cap Present"
        else:
            verdict = "No Bottle"

        result = {"verdict": verdict, "bottles": len(bottles),
                  "caps": len(caps), "detections": bottles + caps}

        frame = draw_result(frame, result)
        cv2.imshow("Bottle Cap Inspector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect bottle cap presence (2-class detector)"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to trained .pt weights")
    parser.add_argument("--source", required=True,
                        help="Image, folder, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", default="0",
                        help="Device: '0' GPU, 'cpu' CPU")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated images with bounding boxes (default: True)")
    parser.add_argument("--show", action="store_true",
                        help="Show results (webcam/display)")
    parser.add_argument("--output", default="runs/cap_inspection",
                        help="Output directory")
    args = parser.parse_args()

    weights = Path(args.weights)
    assert weights.exists(), f"Weights not found: {weights}"

    print(f"\n{'='*50}")
    print("  Bottle Cap Presence Inspector")
    print(f"{'='*50}")
    print(f"  Weights : {weights}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print()

    model = YOLO(str(weights))

    # Webcam
    if args.source == "0" or args.source == "webcam":
        run_on_webcam(model, args.conf, args.iou, args.device)
        return

    # Images
    source = Path(args.source)
    assert source.exists(), f"Source not found: {source}"

    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(
            p for p in source.glob("*.*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        print(f"  No images found at: {source}")
        return

    output_dir = Path(args.output)
    if args.save:
        (output_dir / "annotated").mkdir(parents=True, exist_ok=True)

    all_results = []
    counts = {"Cap Present": 0, "Missing Cap": 0, "No Bottle": 0}

    for img_path in image_paths:
        result = inspect_image(model, img_path, args.conf, args.iou, args.device)
        result["image"] = img_path.name
        all_results.append(result)
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1

        icon = {"Cap Present": "✅", "Missing Cap": "❌", "No Bottle": "⬜"}
        v = result["verdict"]
        print(f"  {icon.get(v, '?')} {v:<15}  {img_path.name:<40}  "
              f"bottles={result['bottles']}  caps={result['caps']}")

        if args.save:
            img = cv2.imread(str(img_path))
            annotated = draw_result(img, result)
            cv2.imwrite(str(output_dir / "annotated" / img_path.name), annotated)

    # Summary
    total = len(all_results)
    print(f"\n  {'─'*45}")
    print(f"  Total inspected : {total}")
    for v, c in counts.items():
        if c > 0:
            print(f"  {v:<15} : {c}  ({100*c/total:.1f}%)")
    print(f"  {'─'*45}")

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results → {json_path}")
    if args.save:
        print(f"  Annotated → {output_dir / 'annotated'}")
    print()


if __name__ == "__main__":
    main()
