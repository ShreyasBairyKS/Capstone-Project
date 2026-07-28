"""
train_yolo_roi.py
─────────────────
Trains the YOLOv8 Object Detector to find ROI_1, ROI_2, ROI_3, and ROI_4.

Usage:
    python scripts/train_yolo_roi.py --data data/yolo_dataset/data.yaml
    python scripts/train_yolo_roi.py --data data/yolo_dataset/data.yaml --model yolov8s.pt --imgsz 1280
    python scripts/train_yolo_roi.py --resume --model models/rois/yolo/train/weights/last.pt
"""

import argparse
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 to localize ROI_1..ROI_4")
    parser.add_argument("--data", type=str, default="data/yolo_dataset_sliced/data.yaml",
                        help="Path to the data.yaml file")
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                        help="Base checkpoint (yolo11n/s/m/l/x.pt) or a run's last.pt to resume from")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for training (1024 is good for 1500px patches)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", type=str, default="models/rois/yolo", help="Save directory")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name under --project. Omit to auto-increment (train, train2, ...)")
    parser.add_argument("--device", type=str, default=None,
                        help="'0' for first GPU, 'cpu' to force CPU, '0,1' for multi-GPU. Auto-detected if omitted")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience, in epochs")
    parser.add_argument("--rect", action="store_true",
                        help="Rectangular training (keeps aspect ratio) — worth trying for non-square scans")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training; pass the interrupted run's last.pt as --model")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker processes")
    parser.add_argument("--cache", type=str, default=None, choices=["ram", "disk"],
                        help="Cache images for faster repeated runs (omit to disable)")
    return parser.parse_args()


def _extract_metrics(results):
    """Best-effort mAP extraction — Ultralytics has used a couple of different result shapes."""
    try:
        rd = results.results_dict
        return rd.get("metrics/mAP50(B)"), rd.get("metrics/mAP50-95(B)")
    except AttributeError:
        pass
    try:
        return results.box.map50, results.box.map
    except AttributeError:
        pass
    return None, None


def _load_dataset_config(data_path: Path):
    """Load the dataset YAML and make its root explicit for Ultralytics."""
    import yaml

    cfg = yaml.safe_load(data_path.read_text()) or {}
    cfg = dict(cfg)
    cfg["path"] = str(data_path.parent.resolve())
    return cfg


def _write_resolved_dataset_config(data_path: Path, cfg: dict) -> Path:
    """Write a resolved dataset YAML so Ultralytics can consume it as a path."""
    import yaml

    resolved_path = Path(tempfile.gettempdir()) / f"{data_path.stem}.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved_path


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"❌ data.yaml not found at: {data_path.resolve()}")

    try:
        cfg = _load_dataset_config(data_path)
        nc, names = cfg.get("nc"), cfg.get("names")
        print(f"Dataset: {nc} classes -> {names}")
        print(f"Dataset root: {cfg['path']}")
        if nc != 4:
            print(f"⚠️  Expected 4 ROI classes (roi_1..roi_4), found nc={nc}. "
                  f"Double-check before a full run — a class mismatch here breaks Stage 2 routing.")
    except ImportError:
        cfg = None

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    data_arg = args.data
    if cfg is not None:
        data_arg = str(_write_resolved_dataset_config(data_path, cfg))

    print(f"Training | data={args.data} | imgsz={args.imgsz} | epochs={args.epochs} | seed={args.seed}")

    train_kwargs = dict(
        data=data_arg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        exist_ok=False,      # never silently overwrite a previous run's results
        plots=True,
        seed=args.seed,
        patience=args.patience,
        rect=args.rect,
        workers=args.workers,
        resume=args.resume,
    )
    if args.name:
        train_kwargs["name"] = args.name
    if args.device:
        train_kwargs["device"] = args.device
    if args.cache:
        train_kwargs["cache"] = args.cache

    try:
        results = model.train(**train_kwargs)
    except Exception as e:
        sys.exit(f"❌ Training failed: {e}")

    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    save_dir = Path(save_dir) if save_dir else Path(args.project)
    best_path = save_dir / "weights" / "best.pt"
    map50, map5095 = _extract_metrics(results)

    print("\n✅ Training complete!")
    if map50 is not None:
        print(f"   mAP50:    {map50:.4f}")
    if map5095 is not None:
        print(f"   mAP50-95: {map5095:.4f}")
    print(f"   Best weights: {best_path if best_path.exists() else '⚠️ not found — check ' + str(save_dir)}")


if __name__ == "__main__":
    main()