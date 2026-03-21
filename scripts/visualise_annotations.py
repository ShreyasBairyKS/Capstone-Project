"""
scripts/visualise_annotations.py

Spot-check annotated images to verify bounding box labels are correct.

Usage:
    python scripts/visualise_annotations.py --data_dir data/annotated/train --n 5
    python scripts/visualise_annotations.py --save  # save to results/annotation_check/
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
    "improper_filling",
    "packaging_damage",
    "label_misalignment",
    "surface_contamination",
]

# BGR colours matching dashboard palette
CLASS_COLOURS = {
    "improper_filling": (235, 130, 59),       # Blue  (OpenCV BGR)
    "packaging_damage": (22, 145, 249),        # Orange
    "label_misalignment": (178, 91, 235),      # Purple
    "surface_contamination": (68, 68, 239),    # Red
}


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse a YOLO-format .txt annotation file into pixel-space bboxes."""
    detections = []
    if not label_path.exists():
        return detections

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, w, h = int(parts[0]), *map(float, parts[1:])
            x1 = int((x_c - w / 2) * img_w)
            y1 = int((y_c - h / 2) * img_h)
            x2 = int((x_c + w / 2) * img_w)
            y2 = int((y_c + h / 2) * img_h)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return detections


def draw_annotations(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and class labels on the image."""
    annotated = img.copy()
    for det in detections:
        colour = CLASS_COLOURS.get(det["class_name"], (0, 255, 0))
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label background
        label = det["class_name"].replace("_", " ").title()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise YOLO-format annotation labels.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/annotated/train"),
                        help="Directory containing images/ and labels/ subdirectories.")
    parser.add_argument("--n", type=int, default=5, help="Number of random images to show.")
    parser.add_argument("--save", action="store_true", help="Save annotated images instead of displaying.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/annotation_check"))
    args = parser.parse_args()

    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"

    if not images_dir.exists():
        print(f"[ERROR] images directory not found: {images_dir}")
        return

    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_paths:
        print(f"[ERROR] No images found in {images_dir}")
        return

    sample = random.sample(image_paths, min(args.n, len(image_paths)))

    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path}")
            continue

        h, w = img.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        detections = parse_yolo_label(label_path, w, h)
        annotated = draw_annotations(img, detections)

        n_boxes = len(detections)
        classes = [d["class_name"] for d in detections]
        print(f"  {img_path.name}: {n_boxes} annotation(s) → {classes}")

        if args.save:
            out_path = args.output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved → {out_path}")
        else:
            cv2.imshow(f"[{img_path.name}]  Press any key for next", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\nDone. Checked {len(sample)} image(s).")


if __name__ == "__main__":
    main()
