"""
train_yolo_roi.py
─────────────────
Trains the YOLOv8 Object Detector to find ROI_1, ROI_2, ROI_3, and ROI_4.

Usage:
    python scripts/train_yolo_roi.py --data data/yolo_dataset/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/yolo_dataset/data.yaml", 
                        help="Path to the data.yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", type=str, default="models/rois/yolo", help="Save directory")
    args = parser.parse_args()

    print(f"Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")  # Start from pretrained nano model for speed

    print(f"Starting training on {args.data}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name="train",
        exist_ok=True,
        plots=True,
    )
    
    print("\n✅ Training complete!")
    print(f"Best model saved to: {Path(args.project) / 'train' / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()
