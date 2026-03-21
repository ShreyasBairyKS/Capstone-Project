"""
training/train_yolov11.py — Fine-tune YOLOv11n on the VisionFood defect dataset.

Phase 1 deliverable.

Usage:
    python training/train_yolov11.py
    python training/train_yolov11.py --epochs 150 --batch 8

Output:
    runs/train/yolov11n_visionfood_v1/weights/best.pt
    models/yolov11n_best.pt            (copied from above)
    models/yolov11n_best.onnx          (FP16 ONNX export)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.config import settings
from core.logging import get_logger, setup_logging

setup_logging(log_format="text")
log = get_logger(__name__)

DATA_YAML = Path("data/annotated/data.yaml")
PRETRAINED_WEIGHTS = "yolov11n.pt"
OUTPUT_MODELS = Path("models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11n on VisionFood defect dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device index or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolov11n_visionfood_v1")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        log.error("data_yaml_not_found", path=str(DATA_YAML))
        print(f"\n[ERROR] Dataset not found at {DATA_YAML}")
        print("Complete Phase 0 dataset annotation first:")
        print("  1. Annotate images in Roboflow")
        print("  2. Export as YOLOv11 format")
        print("  3. Download to data/annotated/")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics_not_installed")
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    if args.wandb:
        try:
            import wandb
            wandb.init(project="visionfood-qai", name=args.name, config=vars(args))
        except ImportError:
            log.warning("wandb_not_installed", msg="Install wandb for experiment tracking")

    log.info("training_start", epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)

    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        mosaic=0.5,
        mixup=0.2,
        flipud=0.3,
        fliplr=0.5,
        degrees=15.0,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # Copy best checkpoint to models/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        OUTPUT_MODELS.mkdir(exist_ok=True)
        shutil.copy(best_pt, OUTPUT_MODELS / "yolov11n_best.pt")
        log.info("checkpoint_saved", path=str(OUTPUT_MODELS / "yolov11n_best.pt"))

        # ONNX export
        log.info("exporting_onnx")
        model_best = YOLO(str(OUTPUT_MODELS / "yolov11n_best.pt"))
        export_result = model_best.export(format="onnx", half=True, imgsz=args.imgsz, dynamic=False)
        onnx_path = Path(str(export_result)) if export_result else None
        if onnx_path and onnx_path.exists():
            log.info("onnx_exported", path=str(onnx_path))
        else:
            log.warning("onnx_export_location_unknown",
                        msg="Check Ultralytics export output directory.")
    else:
        log.error("best_checkpoint_not_found", expected=str(best_pt))

    # Print validation metrics summary
    metrics = model.val(data=str(DATA_YAML))
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    log.info("training_complete", mAP50=round(map50, 4), mAP50_95=round(map5095, 4))
    print(f"\n{'='*50}")
    print(f"  Training Complete")
    print(f"  mAP@50    : {map50:.4f}  (target ≥ 0.80)")
    print(f"  mAP@50-95 : {map5095:.4f}")
    print(f"  Status    : {'PASS ✓' if map50 >= 0.80 else 'NEEDS IMPROVEMENT ✗'}")
    print(f"{'='*50}\n")

    if args.wandb:
        try:
            import wandb
            wandb.log({"mAP50": map50, "mAP50_95": map5095})
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()



