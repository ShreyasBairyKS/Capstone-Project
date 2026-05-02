r"""
bottle-cap-infer.py
===================
Run inference with the trained YOLOv11 beverage cap-quality model.

Supports:
    • Single image
    • Folder of images
    • Returns a structured result dict per image — ready to feed into
      VisionFood QAI's REMEDY triage engine

Run:
    python inference/bottle-cap-infer.py `
    --weights "runs/detect/bottle_cap_det_v2/weights/best.pt" `
    --source "C:/Users/PRO-LAB-4/Desktop/sample.jpg" `
      --conf 0.25 `
      --save
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_DEFECT_CLASSES = ""


# ─────────────────────────────────────────────────────────────────────────────
# RESULT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
# This is the dict structure returned per image.
# Feed this directly into your REMEDY triage severity scorer.
#
# {
#   "image_path": "path/to/img.jpg",
#   "is_defective": True,              ← binary decision for pass/fail
#   "defect_count": 3,
#   "detections": [
#     {
#       "class_id":   0,
#       "class_name": "defect",
#       "confidence": 0.87,
#       "bbox":       [x1, y1, x2, y2],   ← pixel coordinates
#       "bbox_norm":  [cx, cy, w, h],      ← YOLO normalised format
#       "area_frac":  0.043,               ← fraction of image area (for severity)
#     },
#     ...
#   ]
# }
# ─────────────────────────────────────────────────────────────────────────────


def parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def normalize_device(raw_device: str) -> str:
    """Normalize CLI device input to a torch/Ultralytics-compatible value."""
    token = str(raw_device).strip().lower()
    if token in {"cpu", "mps"}:
        return token
    if token.startswith("cuda:"):
        return token
    if token.isdigit():
        return f"cuda:{token}"
    return raw_device


def normalise_model_names(names_node) -> list[str]:
    if isinstance(names_node, dict):
        return [str(names_node[k]) for k in sorted(names_node, key=lambda x: int(x))]
    if isinstance(names_node, list):
        return [str(v) for v in names_node]
    return []


def auto_select_defect_classes(model_class_names: list[str]) -> set[str]:
    """Pick likely defect classes when the user does not provide --defect-classes."""
    if not model_class_names:
        return set()

    defect_keywords = ("defect", "no", "missing", "broken", "damage", "crack")
    good_keywords = ("good", "ok", "normal", "pass")

    selected = {
        name
        for name in model_class_names
        if any(k in name.lower() for k in defect_keywords)
        and not any(k in name.lower() for k in good_keywords)
    }

    if selected:
        return selected

    # Fallback: treat all classes as defect-relevant if no obvious mapping exists.
    return set(model_class_names)


def resolve_result_class_name(result, cls_id: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def parse_result(
    result,
    img_path: str,
    conf_thresh: float,
    defect_class_names: set[str],
    product_category: str,
    product_sub_type: str,
) -> dict:
    """
    Parse a single Ultralytics result object into the VisionFood QAI schema.
    """
    detections = []
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue

        cls_id   = int(box.cls[0])
        cls_name = resolve_result_class_name(result, cls_id)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy, bw, bh  = box.xywhn[0].tolist()   # normalised
        area_frac = bw * bh                         # fraction of image area
        is_cap_issue = cls_name in defect_class_names

        detections.append({
            "class_id":   cls_id,
            "class_name": cls_name,
            "is_cap_issue": is_cap_issue,
            "confidence": round(conf, 4),
            "bbox":       [round(v, 1) for v in [x1, y1, x2, y2]],
            "bbox_norm":  [round(v, 6) for v in [cx, cy, bw, bh]],
            "area_frac":  round(area_frac, 6),
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    defect_count = sum(1 for d in detections if d["is_cap_issue"])

    return {
        "image_path": str(img_path),
        "product_category": product_category,
        "product_sub_type": product_sub_type,
        "inspection_focus": "cap_integrity",
        "is_defective": defect_count > 0,
        "defect_count": defect_count,
        "all_detection_count": len(detections),
        "detections": detections,
    }


def run_inference(
    model: YOLO,
    source: Path,
    conf: float,
    iou: float,
    save: bool,
    output_dir: Path,
    defect_class_names: set[str],
    product_category: str,
    product_sub_type: str,
    device: str,
) -> list[dict]:
    """
    Run inference on a single image or a folder of images.
    Returns a list of result dicts in VisionFood QAI schema.
    """
    # Collect image paths
    if source.is_file():
        image_paths = [source]
    else:
        image_paths = sorted(
            p for p in source.glob("*.*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        print(f"  No images found at: {source}")
        return []

    print(f"\n  Running inference on {len(image_paths)} image(s)...")

    all_results = []
    defect_count = 0

    for img_path in image_paths:
        preds = model.predict(
            str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            save=save,
            project=str(output_dir),
            name="predictions",
            exist_ok=True,
        )

        result_dict = parse_result(
            preds[0],
            img_path,
            conf,
            defect_class_names,
            product_category,
            product_sub_type,
        )
        all_results.append(result_dict)

        # Live print per image
        label = "DEFECTIVE ❌" if result_dict["is_defective"] else "PASS      ✅"
        n_issue = result_dict["defect_count"]
        n_all = result_dict["all_detection_count"]
        top_conf = result_dict["detections"][0]["confidence"] if n_all > 0 else 0.0
        print(f"  [{label}]  {img_path.name:<40}  cap-issues={n_issue}  all-dets={n_all}  conf={top_conf:.2f}")

        if result_dict["is_defective"]:
            defect_count += 1

    # Summary
    total = len(all_results)
    print(f"\n  ─────────────────────────────────────")
    print(f"  Total inspected : {total}")
    print(f"  Defective       : {defect_count}  ({100*defect_count/total:.1f}%)")
    print(f"  Passed          : {total - defect_count}  ({100*(total-defect_count)/total:.1f}%)")
    print(f"  ─────────────────────────────────────")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Beverage cap-quality inference with YOLOv11")
    parser.add_argument("--weights", default="runs/detect/bottle_cap_defect/weights/best.pt",
                        help="Path to trained .pt weights")
    parser.add_argument("--source",  required=True,
                        help="Image path or folder path")
    parser.add_argument("--conf",    default=0.25, type=float,
                        help="Confidence threshold")
    parser.add_argument("--iou",     default=0.45, type=float,
                        help="NMS IoU threshold")
    parser.add_argument("--device",  default="0",
                        help="Device: '0' GPU, 'cpu' CPU")
    parser.add_argument("--save",    action="store_true",
                        help="Save annotated output images")
    parser.add_argument("--output",  default="runs/infer",
                        help="Output directory for results")
    parser.add_argument("--defect-classes", default=DEFAULT_DEFECT_CLASSES,
                        help="Comma-separated class names considered cap issues; leave empty for auto")
    parser.add_argument("--product-category", default="beverage",
                        choices=["beverage", "food"],
                        help="Inspection product category metadata")
    parser.add_argument("--product-sub-type", default="transparent_bottle",
                        choices=["transparent_bottle", "rigid_can", "flexible_wrapper", "rigid_box"],
                        help="Inspection product sub-type metadata")
    args = parser.parse_args()

    weights = Path(args.weights)
    source  = Path(args.source)
    output_dir = Path(args.output)

    device = normalize_device(args.device)
    
    assert weights.exists(), f"Weights not found: {weights}"
    assert source.exists(),  f"Source not found: {source}"

    print(f"\n{'='*55}")
    print("  VisionFood QAI — Bottle Defect Inference")
    print(f"{'='*55}")
    print(f"  Product : {args.product_category} / {args.product_sub_type}")
    print(f"  Weights : {weights}")
    print(f"  Source  : {source}")
    print(f"  Conf    : {args.conf}  |  IoU: {args.iou}")

    model = YOLO(str(weights))

    model_class_names = normalise_model_names(model.names)
    requested_defect_classes = set(parse_csv_list(args.defect_classes))
    if requested_defect_classes:
        unknown = sorted(requested_defect_classes - set(model_class_names))
        if unknown:
            raise ValueError(
                f"Unknown defect class names: {unknown}. Available classes: {model_class_names}"
            )
        defect_class_names = requested_defect_classes
    else:
        defect_class_names = auto_select_defect_classes(model_class_names)
        if not defect_class_names:
            raise ValueError("Could not infer defect classes from model labels")

    print(f"  Classes : {model_class_names}")
    print(f"  Defects : {sorted(defect_class_names)}")

    results = run_inference(
        model,
        source,
        args.conf,
        args.iou,
        args.save,
        output_dir,
        defect_class_names,
        args.product_category,
        args.product_sub_type,
        device,
    )

    # Save JSON results
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / "inference_results.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {json_out}")
    if args.save:
        print(f"  Annotated images → {output_dir}/predictions/")
    print()


if __name__ == "__main__":
    main()