"""
training/train_can_defect.py
============================
Brand-agnostic YOLO11 training pipeline for structural can surface damage detection.

Key design goals:
1) Detect structural defects independent of can brand/color by using:
   - Grayscale channel for base signal
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for local texture
   - Canny edges for boundary + defect detection
2) Preserve structural information with moderate augmentation that maintains geometry.
3) Use staged fine-tuning: freeze first 10 layers for warmup, then unfreeze.

Usage:
    # Full run with structural features (RECOMMENDED)
    python training/train_can_defect.py --edge-channel

    # Prepare dataset only (structural channels)
    python training/train_can_defect.py --edge-channel --prepare-only
    
    # Legacy: grayscale only (not recommended)
    python training/train_can_defect.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import stat
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_SOURCE_DATASET = Path("dataset/Food/canned-food-surface-defect.v6-last-version.yolov11")
DEFAULT_PREPARED_DATASET = Path(
    "dataset/Food/canned-food-surface-defect.v6-last-version.yolov11_structural_v1"
)
DEFAULT_PROJECT = Path("inference/can_defect")
DEFAULT_RUN_NAME = "structural_v1"
DEFAULT_MODEL = "yolo11m.pt"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_WEIGHT_PRIOR = {
    "no defect": 1.0,
    "minor defect": 1.5,
    "major defect": 2.0,
    "critical defect": 3.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 can structural defect detector with grayscale-only signal."
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help=f"Path to downloaded Roboflow YOLO dataset (default: {DEFAULT_SOURCE_DATASET})",
    )
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=DEFAULT_PREPARED_DATASET,
        help=f"Path for generated grayscale structural dataset (default: {DEFAULT_PREPARED_DATASET})",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Starting YOLO checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="GPU index (e.g. 0) or cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--critical-oversample-factor",
        type=int,
        default=2,
        help="Duplicate train samples containing critical class by this factor.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=0,
        help="If >0, process only this many images per split (useful for fast smoke tests).",
    )
    parser.add_argument(
        "--edge-channel",
        action="store_true",
        help="Use structural channels [gray, CLAHE, Canny] instead of [gray, gray, gray]. "
        "Required for brand-agnostic defect detection. Recommended.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild prepared dataset even if it already exists.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip grayscale conversion and use existing prepared dataset.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset, do not train.")
    parser.add_argument(
        "--disable-custom-albu",
        action="store_true",
        help="Use Ultralytics default Albumentations instead of structural custom policy.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export after training.",
    )
    return parser.parse_args()


def _normalise_names(names_node: object) -> list[str]:
    if isinstance(names_node, dict):
        return [str(names_node[k]) for k in sorted(names_node, key=lambda x: int(x))]
    if isinstance(names_node, list):
        return [str(x) for x in names_node]
    raise ValueError("data.yaml 'names' must be a list or dict")


def _read_data_yaml(source_dataset: Path) -> tuple[Path, dict]:
    data_yaml = source_dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML in {data_yaml}")
    return data_yaml, cfg


def _resolve_images_dir(source_dataset: Path, source_data_yaml: Path, split_rel: str) -> Path:
    """Resolve split image dir from YOLO YAML with resilient fallback rules."""
    token = str(split_rel).replace("\\", "/")
    stripped = token.lstrip("./")
    deparented = token
    while deparented.startswith("../"):
        deparented = deparented[3:]

    candidates = [
        (source_data_yaml.parent / token).resolve(),
        (source_dataset / token).resolve(),
        (source_dataset / stripped).resolve(),
        (source_dataset / deparented).resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the most standard candidate for clearer error message upstream.
    return candidates[0]


def _safe_rmtree(path: Path, retries: int = 5, base_delay_sec: float = 0.2) -> None:
    """Robust recursive delete for Windows where file handles can linger briefly."""

    def _onerror(func, target_path, _exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay_sec * (attempt + 1))


def _gray_or_edge_3ch(image_bgr: np.ndarray, use_edge_channel: bool) -> np.ndarray:
    """
    Extract brand-agnostic structural features for defect detection.
    
    Channels:
    - Channel 1: Grayscale
    - Channel 2: CLAHE (local contrast enhancement for texture)
    - Channel 3: Canny edges (structural boundaries + defects)
    
    This approach is independent of brand color and focuses on surface geometry.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    if not use_edge_channel:
        # Standard: [gray, gray, gray]
        return np.stack([gray, gray, gray], axis=-1)
    
    # Enhanced structural features for brand-agnostic detection
    # Channel 2: Local contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)
    
    # Channel 3: Canny edge detection (structural boundaries)
    # Use adaptive thresholds based on image statistics
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny_edges = cv2.Canny(gray, lower, upper)
    
    # Stack: [grayscale, CLAHE contrast, Canny edges]
    return np.stack([gray, clahe_result, canny_edges], axis=-1)


def _iter_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])


def _has_class_id(label_path: Path, class_ids: set[int]) -> bool:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if cls_id in class_ids:
            return True
    return False


def prepare_structural_dataset(
    source_dataset: Path,
    prepared_dataset: Path,
    use_edge_channel: bool,
    critical_oversample_factor: int,
    force_rebuild: bool,
    max_images_per_split: int = 0,
) -> Path:
    source_dataset = source_dataset.resolve()
    prepared_dataset = prepared_dataset.resolve()

    source_data_yaml, source_cfg = _read_data_yaml(source_dataset)
    names = _normalise_names(source_cfg.get("names", []))
    if not names:
        raise ValueError("No class names found in source data.yaml")

    if force_rebuild and prepared_dataset.exists():
        _safe_rmtree(prepared_dataset)

    prepared_dataset.mkdir(parents=True, exist_ok=True)

    split_key_to_src = {"train": source_cfg.get("train"), "val": source_cfg.get("val")}
    if source_cfg.get("test"):
        split_key_to_src["test"] = source_cfg.get("test")

    split_key_to_dst_rel_images: dict[str, str] = {}
    critical_ids = {
        idx
        for idx, name in enumerate(names)
        if name.strip().lower() == "critical defect" or "critical" in name.strip().lower()
    }

    print("\n[INFO] Preparing structural dataset (grayscale conversion)...")
    for split_key, split_rel in split_key_to_src.items():
        if not split_rel:
            continue

        src_images_dir = _resolve_images_dir(source_dataset, source_data_yaml, str(split_rel))
        if not src_images_dir.exists():
            raise FileNotFoundError(f"{split_key} images path not found: {src_images_dir}")

        src_split_root = src_images_dir.parent
        src_labels_dir = src_split_root / "labels"
        if not src_labels_dir.exists():
            raise FileNotFoundError(f"{split_key} labels path not found: {src_labels_dir}")

        split_folder_name = src_split_root.name
        dst_images_dir = prepared_dataset / split_folder_name / "images"
        dst_labels_dir = prepared_dataset / split_folder_name / "labels"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        split_key_to_dst_rel_images[split_key] = f"{split_folder_name}/images"
        image_paths = _iter_images(src_images_dir)
        if max_images_per_split > 0:
            image_paths = image_paths[:max_images_per_split]
        print(f"[INFO] {split_key}: {len(image_paths)} images")

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            converted = _gray_or_edge_3ch(img, use_edge_channel)
            dst_img_path = dst_images_dir / image_path.name
            cv2.imwrite(str(dst_img_path), converted)

            src_label = src_labels_dir / f"{image_path.stem}.txt"
            dst_label = dst_labels_dir / src_label.name
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                dst_label.write_text("", encoding="utf-8")

    cls_pw = [CLASS_WEIGHT_PRIOR.get(name.strip().lower(), 1.0) for name in names]

    # Oversample critical defect samples by repeating image paths in train txt list.
    train_images_dir = prepared_dataset / split_key_to_dst_rel_images["train"]
    train_labels_dir = train_images_dir.parent / "labels"
    train_txt = prepared_dataset / "train_images_oversampled.txt"
    repeats = max(1, critical_oversample_factor)
    oversampled = 0
    total_entries = 0

    with train_txt.open("w", encoding="utf-8") as f:
        for image_path in _iter_images(train_images_dir):
            label_path = train_labels_dir / f"{image_path.stem}.txt"
            line = str(image_path.resolve())
            f.write(line + "\n")
            total_entries += 1

            if repeats > 1 and critical_ids and _has_class_id(label_path, critical_ids):
                for _ in range(repeats - 1):
                    f.write(line + "\n")
                    oversampled += 1
                    total_entries += 1

    prepared_cfg: dict[str, object] = {
        "train": train_txt.name,
        "val": split_key_to_dst_rel_images["val"],
        "nc": len(names),
        "names": names,
        "cls_pw": cls_pw,
    }
    if "test" in split_key_to_dst_rel_images:
        prepared_cfg["test"] = split_key_to_dst_rel_images["test"]

    prepared_data_yaml = prepared_dataset / "data.yaml"
    prepared_data_yaml.write_text(
        yaml.safe_dump(prepared_cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    print(f"[INFO] Saved prepared data.yaml: {prepared_data_yaml}")
    print(f"[INFO] cls_pw (aligned with names order): {cls_pw}")
    print(
        f"[INFO] train list entries: {total_entries} "
        f"(critical oversample duplicates added: {oversampled})"
    )
    return prepared_data_yaml


class StructuralAlbumentations:
    """Custom Albumentations policy to suppress appearance shortcuts.

    Defined at module scope so it can be imported/pickled by worker
    processes on Windows (spawn start method).
    """

    def __init__(self, p: float = 1.0) -> None:
        self.p = p
        self.transform = None
        self.contains_spatial = False

        try:
            import importlib

            A = importlib.import_module("albumentations")
            from ultralytics.utils.checks import check_version
            from ultralytics.utils import LOGGER

            check_version(A.__version__, "1.0.3", hard=True)

            transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    brightness_by_max=True,
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.3),
                A.Affine(shear={"x": (-5, 5), "y": (-5, 5)}, fit_output=False, p=0.3),
                A.CoarseDropout(
                    max_holes=2,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=128,
                    p=0.2,
                ),
            ]

            self.contains_spatial = True
            self.transform = A.Compose(
                transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
            )
            LOGGER.info(
                "albumentations: using structure-preserving policy "
                "(mild brightness/contrast, light blur/noise, CLAHE, minor shear, coarse dropout)"
            )
        except Exception as exc:
            from ultralytics.utils import LOGGER

            LOGGER.warning(f"albumentations: structural custom policy disabled ({exc})")

    def __call__(self, labels: dict) -> dict:
        if self.transform is None or random.random() > self.p:
            return labels

        cls = labels.get("cls")
        if cls is None:
            return labels

        if self.contains_spatial:
            if len(cls):
                image = labels["img"]
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*image.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                new = self.transform(image=image, bboxes=bboxes, class_labels=cls)
                if len(new["class_labels"]) > 0:
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                    labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]

        return labels


def install_structural_albumentations() -> None:
    """Patch Ultralytics Albumentations with our module-level policy."""
    from ultralytics.data import augment as ul_augment

    ul_augment.Albumentations = StructuralAlbumentations


def _build_train_kwargs(args: argparse.Namespace, data_yaml: Path, run_name: str) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "warmup_bias_lr": 0.01,
        "mosaic": 1.0,
        "mixup": 0.15,
        "degrees": 45.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 10.0,
        "perspective": 0.08,
        "flipud": 0.3,
        "fliplr": 0.5,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.4,
        "cos_lr": True,
        "label_smoothing": 0.05,
        "workers": args.workers,
        "project": str(args.project),
        "name": run_name,
        "save_period": 10,
        "seed": args.seed,
        "device": args.device,
        "exist_ok": True,
        "plots": True,
        "val": True,
    }


def train_two_stage(args: argparse.Namespace, prepared_data_yaml: Path) -> Path:
    from ultralytics import YOLO

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.disable_custom_albu:
        install_structural_albumentations()

    warmup_epochs = max(0, min(args.freeze_epochs, args.epochs))
    finetune_epochs = max(0, args.epochs - warmup_epochs)

    stage1_name = f"{args.name}_warmup"
    stage1_kwargs = _build_train_kwargs(args, prepared_data_yaml, stage1_name)
    stage1_kwargs["epochs"] = warmup_epochs
    stage1_kwargs["freeze"] = args.freeze_layers

    stage1_weights: Path | None = None
    if warmup_epochs > 0:
        print(
            "\n[INFO] Stage 1/2 warmup: "
            f"epochs={warmup_epochs}, freeze first {args.freeze_layers} layers"
        )
        model_warmup = YOLO(args.weights)
        model_warmup.train(**stage1_kwargs)

        stage1_dir = args.project / stage1_name / "weights"
        best = stage1_dir / "best.pt"
        last = stage1_dir / "last.pt"
        stage1_weights = best if best.exists() else last
        if stage1_weights is None or not stage1_weights.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found in {stage1_dir}")
    else:
        stage1_weights = Path(args.weights)

    if finetune_epochs <= 0:
        return stage1_weights

    print("\n[INFO] Stage 2/2 fine-tune: unfreeze all layers")
    stage2_kwargs = _build_train_kwargs(args, prepared_data_yaml, args.name)
    stage2_kwargs["epochs"] = finetune_epochs
    stage2_kwargs["freeze"] = 0

    model_full = YOLO(str(stage1_weights))
    model_full.train(**stage2_kwargs)

    final_weights_dir = args.project / args.name / "weights"
    best = final_weights_dir / "best.pt"
    last = final_weights_dir / "last.pt"
    final_weights = best if best.exists() else last
    if final_weights is None or not final_weights.exists():
        raise FileNotFoundError(f"Final checkpoint not found in {final_weights_dir}")
    return final_weights


def export_best_onnx(best_weights: Path, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(best_weights))
    export_result = model.export(format="onnx", imgsz=imgsz, simplify=True)
    exported = Path(str(export_result))
    if exported.exists():
        return exported

    fallback = best_weights.with_suffix(".onnx")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("ONNX export reported success but output file was not found.")


def main() -> None:
    args = parse_args()

    source_dataset = args.source_dataset.resolve()
    prepared_dataset = args.prepared_dataset.resolve()
    args.project = args.project.resolve()

    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset folder does not exist: {source_dataset}")

    if args.skip_prepare:
        prepared_data_yaml = prepared_dataset / "data.yaml"
        if not prepared_data_yaml.exists():
            raise FileNotFoundError(
                "--skip-prepare set but prepared data.yaml is missing: "
                f"{prepared_data_yaml}"
            )
    else:
        prepared_data_yaml = prepare_structural_dataset(
            source_dataset=source_dataset,
            prepared_dataset=prepared_dataset,
            use_edge_channel=args.edge_channel,
            critical_oversample_factor=args.critical_oversample_factor,
            force_rebuild=args.force_rebuild,
            max_images_per_split=args.max_images_per_split,
        )

    if args.prepare_only:
        print("\n[DONE] Dataset preparation completed (prepare-only mode).")
        print(f"Prepared YAML: {prepared_data_yaml}")
        return

    best_weights = train_two_stage(args, prepared_data_yaml)
    print(f"\n[INFO] Best checkpoint: {best_weights}")

    if not args.skip_export:
        onnx_path = export_best_onnx(best_weights=best_weights, imgsz=args.imgsz)
        print(f"[INFO] ONNX exported: {onnx_path}")

    final_run_dir = args.project / args.name
    print("\n[DONE] Training completed.")
    print("Expected artifacts:")
    print(f"  - {final_run_dir / 'weights' / 'best.pt'}")
    print(f"  - {final_run_dir / 'weights' / 'last.pt'}")
    print(f"  - {final_run_dir / 'results.csv'}")
    print(f"  - {final_run_dir / 'confusion_matrix.png'}")
    print(f"  - {final_run_dir / 'PR_curve.png'}")
    print(f"  - {final_run_dir / 'F1_curve.png'}")


if __name__ == "__main__":
    main()
