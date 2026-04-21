"""
hard_negative_mining.py
=======================
Mine false positives and inject them into train split as hard negatives.

For this project, a hard negative means an image that is non-defective
for binary decisioning (typically goodCap) but predicted as a defect class
(defectCap/noCap).

Example:
    python hard_negative_mining.py \
      --weights runs/detect/bottle_cap_defect_quality/weights/best.pt \
      --fp-dir fp_analysis_conf045 \
      --data-dir dataset/Beverages/bottleDefect.v1-first.yolov11-cap \
      --conf 0.45 \
      --oversample 3 \
      --scan-all \
      --defect-classes defectCap,noCap \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WEIGHTS = "runs/detect/bottle_cap_defect_quality/weights/best.pt"
DEFAULT_FP_DIR = "fp_analysis_conf045"
DEFAULT_DATA_DIR = "dataset/Beverages/bottleDefect.v1-first.yolov11-cap"
DEFAULT_DEFECT_CLASSES = "defectCap,noCap"


def parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def normalize_names(names_node) -> list[str]:
    if isinstance(names_node, dict):
        return [str(names_node[k]) for k in sorted(names_node, key=lambda x: int(x))]
    if isinstance(names_node, list):
        return [str(v) for v in names_node]
    return []


def normalize_device(device: str) -> str:
    token = device.strip().lower()
    if token == "0":
        return "cuda:0"
    return device


def resolve_result_class_name(result, cls_id: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def read_label_class_ids(label_path: Path) -> list[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    class_ids: list[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                continue

    return class_ids


def is_non_defect_gt(label_path: Path, defect_class_ids: set[int]) -> bool:
    return not any(cls_id in defect_class_ids for cls_id in read_label_class_ids(label_path))


def is_predicted_as_defect(result, conf: float, defect_class_names: set[str]) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False

    for conf_raw, cls_raw in zip(result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        if float(conf_raw) < conf:
            continue
        cls_name = resolve_result_class_name(result, int(cls_raw))
        if cls_name in defect_class_names:
            return True

    return False


def find_split_dirs(data_dir: Path, split: str) -> tuple[Path, Path]:
    aliases = [split]
    if split == "val":
        aliases.append("valid")

    for alias in aliases:
        img_dir = data_dir / alias / "images"
        lbl_dir = data_dir / alias / "labels"
        if img_dir.exists() and lbl_dir.exists():
            return img_dir, lbl_dir

    # return default path even if missing so caller can handle uniformly
    alias = aliases[0]
    return data_dir / alias / "images", data_dir / alias / "labels"


def parse_fp_paths_from_dir(fp_dir: Path) -> list[Path]:
    candidates = [fp_dir / "fp_cases.json", fp_dir / "fp_analysis.json"]

    payload = None
    for file_path in candidates:
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            break

    if payload is None:
        return []

    if isinstance(payload, dict):
        cases = payload.get("cases", payload.get("fp_cases", []))
    else:
        cases = payload

    paths: list[Path] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        for key in ("image", "file", "image_path", "img_path", "path"):
            if key not in case:
                continue
            candidate = Path(case[key]).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.exists():
                paths.append(candidate)
                break

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def find_fp_by_inference(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    conf: float,
    iou: float,
    defect_class_ids: set[int],
    defect_class_names: set[str],
    device: str,
) -> list[Path]:
    if not image_dir.exists() or not label_dir.exists():
        return []

    image_paths = sorted(
        p for p in image_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    fp_paths: list[Path] = []
    for img_path in image_paths:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not is_non_defect_gt(lbl_path, defect_class_ids):
            continue

        result = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        if is_predicted_as_defect(result, conf, defect_class_names):
            fp_paths.append(img_path.resolve())

    return fp_paths


def append_unique_paths(base: list[Path], incoming: list[Path]) -> list[Path]:
    seen = {str(p.resolve()) for p in base}
    for p in incoming:
        key = str(p.resolve())
        if key in seen:
            continue
        base.append(p)
        seen.add(key)
    return base


def infer_label_path_from_image(image_path: Path) -> Path:
    # dataset/.../<split>/images/foo.jpg -> dataset/.../<split>/labels/foo.txt
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def copy_with_suffix(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def inject_hard_negatives(
    fp_image_paths: list[Path],
    train_images_dir: Path,
    train_labels_dir: Path,
    oversample: int,
) -> dict:
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    injected_pairs = 0
    skipped_existing = 0
    copied_without_label = 0

    for img_path in fp_image_paths:
        src_lbl = infer_label_path_from_image(img_path)
        lbl_exists = src_lbl.exists()

        for rep in range(oversample):
            stem = f"{img_path.stem}_hn{rep:02d}"
            dst_img = train_images_dir / f"{stem}{img_path.suffix}"
            dst_lbl = train_labels_dir / f"{stem}.txt"

            if dst_img.exists() or dst_lbl.exists():
                skipped_existing += 1
                continue

            copy_with_suffix(img_path, dst_img)
            if lbl_exists:
                copy_with_suffix(src_lbl, dst_lbl)
            else:
                dst_lbl.touch()
                copied_without_label += 1

            injected_pairs += 1

    return {
        "injected_pairs": injected_pairs,
        "skipped_existing": skipped_existing,
        "copied_without_label": copied_without_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard negative mining for cap defect confusion")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Model weights .pt")
    parser.add_argument("--fp-dir", default=DEFAULT_FP_DIR,
                        help="Directory containing fp_cases.json")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Dataset root containing train/valid/test folders")
    parser.add_argument("--conf", default=0.45, type=float,
                        help="Confidence threshold for mining")
    parser.add_argument("--iou", default=0.45, type=float,
                        help="NMS IoU threshold")
    parser.add_argument("--oversample", default=3, type=int,
                        help="Copies per mined hard negative")
    parser.add_argument("--scan-all", action="store_true",
                        help="Also mine additional FPs from train/val splits")
    parser.add_argument("--defect-classes", default=DEFAULT_DEFECT_CLASSES,
                        help="Comma-separated classes treated as defects")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0, 0, cpu")
    args = parser.parse_args()

    if args.oversample < 1:
        raise ValueError("--oversample must be >= 1")

    weights = Path(args.weights)
    fp_dir = Path(args.fp_dir)
    data_dir = Path(args.data_dir)
    device = normalize_device(args.device)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    model = YOLO(str(weights))
    model.to(device)
    class_names = normalize_names(model.names)

    defect_class_names = set(parse_csv_list(args.defect_classes))
    if not defect_class_names:
        raise ValueError("--defect-classes cannot be empty")
    unknown = sorted(defect_class_names - set(class_names))
    if unknown:
        raise ValueError(f"Unknown defect classes {unknown}. Available: {class_names}")
    defect_class_ids = {i for i, n in enumerate(class_names) if n in defect_class_names}

    test_img_dir, test_lbl_dir = find_split_dirs(data_dir, "test")
    train_img_dir, train_lbl_dir = find_split_dirs(data_dir, "train")

    print(f"\n{'='*60}")
    print("  VisionFood QAI — Hard Negative Mining")
    print(f"{'='*60}")
    print(f"  Weights      : {weights}")
    print(f"  FP dir       : {fp_dir}")
    print(f"  Data dir     : {data_dir}")
    print(f"  Defect class : {sorted(defect_class_names)}")
    print(f"  Conf / IoU   : {args.conf} / {args.iou}")
    print(f"  Oversample   : {args.oversample}")
    print(f"  Scan all     : {args.scan_all}")
    print(f"  Device       : {device}")
    print(f"{'='*60}\n")

    print("[1/3] Loading FP paths from fp-dir...")
    fp_paths = parse_fp_paths_from_dir(fp_dir)
    if fp_paths:
        print(f"  Loaded {len(fp_paths)} FP images from {fp_dir}")
    else:
        print("  No usable FP json found; running fallback scan on test split...")
        fp_paths = find_fp_by_inference(
            model=model,
            image_dir=test_img_dir,
            label_dir=test_lbl_dir,
            conf=args.conf,
            iou=args.iou,
            defect_class_ids=defect_class_ids,
            defect_class_names=defect_class_names,
            device=device,
        )
        print(f"  Fallback found {len(fp_paths)} FP images on test split")

    scan_details = {}
    if args.scan_all:
        print("\n[2/3] Scanning train/val for additional hard negatives...")
        for split in ("train", "val"):
            img_dir, lbl_dir = find_split_dirs(data_dir, split)
            mined = find_fp_by_inference(
                model=model,
                image_dir=img_dir,
                label_dir=lbl_dir,
                conf=args.conf,
                iou=args.iou,
                defect_class_ids=defect_class_ids,
                defect_class_names=defect_class_names,
                device=device,
            )
            before = len(fp_paths)
            fp_paths = append_unique_paths(fp_paths, mined)
            added = len(fp_paths) - before
            scan_details[split] = {
                "found": len(mined),
                "added_unique": added,
                "image_dir": str(img_dir),
            }
            print(f"  {split:<5} found={len(mined):>4} | added unique={added:>4}")
    else:
        print("\n[2/3] Skipping additional split scan (use --scan-all to enable)")

    if not fp_paths:
        print("\nNo hard negatives found. Nothing to inject.")
        return

    print("\n[3/3] Injecting hard negatives into training split...")
    inject_stats = inject_hard_negatives(
        fp_image_paths=fp_paths,
        train_images_dir=train_img_dir,
        train_labels_dir=train_lbl_dir,
        oversample=args.oversample,
    )

    total_train_images = len([p for p in train_img_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    total_train_labels = len(list(train_lbl_dir.glob("*.txt")))

    report = {
        "weights": str(weights),
        "fp_dir": str(fp_dir),
        "data_dir": str(data_dir),
        "device": device,
        "defect_classes": sorted(defect_class_names),
        "conf": args.conf,
        "iou": args.iou,
        "oversample": args.oversample,
        "scan_all": args.scan_all,
        "fp_images_mined": len(fp_paths),
        "injection": inject_stats,
        "scan_details": scan_details,
        "train_split_totals": {
            "images": total_train_images,
            "labels": total_train_labels,
            "images_dir": str(train_img_dir),
            "labels_dir": str(train_lbl_dir),
        },
    }

    out_dir = Path("runs/detect")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "hard_negative_mining_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "-" * 60)
    print("Hard negative mining complete")
    print("-" * 60)
    print(f"FP images mined          : {len(fp_paths)}")
    print(f"Injected pairs           : {inject_stats['injected_pairs']}")
    print(f"Skipped existing targets : {inject_stats['skipped_existing']}")
    print(f"Copied without label     : {inject_stats['copied_without_label']}")
    print(f"Train images total       : {total_train_images}")
    print(f"Train labels total       : {total_train_labels}")
    print(f"Report                   : {report_path}")
    print("-" * 60)


if __name__ == "__main__":
    main()