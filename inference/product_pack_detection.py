"""
Product + tear detection inference script.

Pipeline:
    1) Detect product packs on full image
    2) Crop each product box
    3) Run tear detector on each crop
    4) Save final annotated image to inference/product_detect

Usage:
    python inference/product_pack_detection.py `
        --weights runs\Chips_pack_tear\product\product_detect_v1\weights\best.pt `
        --tear-weights runs\Chips_pack_tear\tear\tear_detect_v1_aggressive_fix\weights\best.pt `
        --source "C:\\Users\\PRO-LAB-4\\Desktop\\sddefault.jpg"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TEAR_WEIGHTS = "runs\\Chips_pack_tear\\tear\\tear_detect_v1_aggressive_fix\\weights\\best.pt"


def draw_detections(img, product_detections, tear_detections, latency_ms):
    """Draw product and tear boxes with confidence only."""
    out = img.copy()

    for det in product_detections:
        cls_name = det["class"]
        conf = det["conf"]
        x1, y1, x2, y2 = det["bbox"]
        tear_status = det.get("tear_status", "unknown")

        color = (0, 0, 255) if tear_status == "tear" else (0, 170, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for det in tear_detections:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), (255, 0, 255), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]

    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_on_image(model: YOLO, img, device: str, conf: float, iou: float):
    t0 = time.perf_counter()
    results = model.predict(img, conf=conf, iou=iou, device=device, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    detections = []
    names = results[0].names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class": cls_name,
                "conf": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections, round(latency_ms, 1)


def run_tear_on_products(
    tear_model: YOLO,
    image,
    product_detections,
    device: str,
    tear_conf: float,
    tear_iou: float,
):
    """Run tear detector on each product crop and map tear boxes back to full image."""
    h, w = image.shape[:2]
    all_tears = []

    for det in product_detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            det["tear_status"] = "no_tear"
            det["tear_conf"] = 0.0
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            det["tear_status"] = "no_tear"
            det["tear_conf"] = 0.0
            continue

        crop_tears, _ = run_on_image(tear_model, crop, device, tear_conf, tear_iou)
        if not crop_tears:
            det["tear_status"] = "no_tear"
            det["tear_conf"] = 0.0
            continue

        det["tear_status"] = "tear"
        det["tear_conf"] = max(t["conf"] for t in crop_tears)

        for t in crop_tears:
            tx1, ty1, tx2, ty2 = t["bbox"]
            all_tears.append(
                {
                    "class": t["class"],
                    "conf": t["conf"],
                    "bbox": [x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2],
                }
            )

    return all_tears


def main():
    parser = argparse.ArgumentParser(description="Product + tear detection inference")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--tear-weights", default=DEFAULT_TEAR_WEIGHTS, help="Path to tear YOLO .pt model")
    parser.add_argument("--source", required=True, help="Image path, folder path, or '0' webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--tear-conf", type=float, default=0.25, help="Tear detector confidence threshold")
    parser.add_argument("--tear-iou", type=float, default=0.45, help="Tear detector NMS IoU threshold")
    parser.add_argument("--device", default="0", help="Device: '0', 'cpu', 'cuda', etc")
    parser.add_argument("--show", action="store_true", help="Show results in OpenCV window")
    parser.add_argument("--output", default="inference/product_detect", help="Annotated output folder")
    args = parser.parse_args()

    product_weights = Path(args.weights)
    if not product_weights.exists():
        raise FileNotFoundError(f"Product model weights not found: {product_weights}")

    tear_weights = Path(args.tear_weights)
    if not tear_weights.exists():
        raise FileNotFoundError(f"Tear model weights not found: {tear_weights}")

    product_model = YOLO(str(product_weights))
    tear_model = YOLO(str(tear_weights))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Product + Tear Detection")
    print("=" * 60)
    print(f"Product weights : {product_weights}")
    print(f"Tear weights    : {tear_weights}")
    print(f"Source  : {args.source}")
    print(f"Output  : {output_dir}")
    print(f"Product conf/IoU: {args.conf}/{args.iou}")
    print(f"Tear conf/IoU   : {args.tear_conf}/{args.tear_iou}")

    if args.source in {"0", "webcam"}:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("\nPress 'q' to quit webcam mode.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            product_detections, product_latency = run_on_image(
                product_model, frame, args.device, args.conf, args.iou
            )
            tear_detections = run_tear_on_products(
                tear_model, frame, product_detections, args.device, args.tear_conf, args.tear_iou
            )
            annotated = draw_detections(frame, product_detections, tear_detections, product_latency)

            cv2.imshow("product_pack_detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    image_paths = collect_images(source)
    if not image_paths:
        print(f"No images found at: {source}")
        return

    total_product_detections = 0
    total_tear_detections = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        product_detections, latency_ms = run_on_image(product_model, image, args.device, args.conf, args.iou)
        tear_detections = run_tear_on_products(
            tear_model, image, product_detections, args.device, args.tear_conf, args.tear_iou
        )

        total_product_detections += len(product_detections)
        total_tear_detections += len(tear_detections)
        annotated = draw_detections(image, product_detections, tear_detections, latency_ms)

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)

        print(
            f"{image_path.name:<45} "
            f"products={len(product_detections):<3} tears={len(tear_detections):<3} "
            f"latency={latency_ms:>5.1f}ms"
        )

        if args.show:
            cv2.imshow("product_pack_detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    print("\n" + "-" * 60)
    print(f"Processed images : {len(image_paths)}")
    print(f"Total products   : {total_product_detections}")
    print(f"Total tears      : {total_tear_detections}")
    print(f"Saved annotated  : {output_dir}")


if __name__ == "__main__":
    main()
