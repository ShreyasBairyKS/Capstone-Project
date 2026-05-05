"""Train tear detection model (localize tears in packets, robust to color/brand)."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_augmented/tear_detect/data.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="tear_detect_v1_aggressive")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  TEAR DETECTION MODEL TRAINING")
    print(f"{'='*60}")
    print(f"  Dataset  : {args.data}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Device   : {args.device}")
    print(f"  Augment  : Aggressive (HSV, blur, noise, rotation)")
    print()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs/train/tear",
        name=args.name,
        patience=20,
        save=True,
        save_period=10,
        workers=args.workers,
        cache=False,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("  Training Complete")
    print(f"{'='*60}")
    print(f"  Best model: runs/train/tear/{args.name}/weights/best.pt")
    print()

if __name__ == "__main__":
    main()
