"""
Test fill level classifier checkpoint on an image or folder of images.

Usage:
    python inference/test_fill_classifier.py --source path/to/image_or_folder

    python inference/test_fill_classifier.py \
        --weights models/fill_classifier_best.pth \
        --source data/fill_level/val \
        --save-csv runs/classify/fill_test/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["underfill", "normal", "overfill"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FILL_FLAG_COLORS = {
    "underfill": (0, 0, 255),
    "normal": (0, 180, 0),
    "overfill": (0, 165, 255),
}


def build_model(num_classes: int, dropout: float) -> nn.Module:
    """Build MobileNetV3-Small classifier head used during training."""
    backbone = mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features

    class FillLevelClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            return self.classifier(x)

    return FillLevelClassifier()


def letterbox_resize_pil(img: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio, then gray-pad to square."""
    w, h = img.size
    ratio = min(size / h, size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(img, (pad_left, pad_top))
    return canvas


def preprocess_image(image_path: Path, imgsz: int) -> torch.Tensor:
    """Load image and convert to normalized tensor."""
    img = Image.open(image_path).convert("RGB")
    img = letterbox_resize_pil(img, imgsz)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "0":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(weights_path: Path, device: torch.device):
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    dropout = float(args.get("dropout", 0.3))
    imgsz = int(args.get("imgsz", 224))

    model = build_model(num_classes=len(class_names), dropout=dropout)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names, imgsz


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]

    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@torch.no_grad()
def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(tensor.to(device))
    return torch.softmax(logits, dim=1)[0].cpu()


def save_predictions_csv(rows: list[dict], out_csv: Path, class_names: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["image", "predicted_class", "confidence"] + [f"prob_{c}" for c in class_names]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def maybe_save_annotated(image_path: Path, predicted: str, confidence: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        return

    fill_text = predicted.replace("_", " ").upper()
    label = f"FILL LEVEL: {fill_text}  |  CONF: {confidence:.2%}"
    color = FILL_FLAG_COLORS.get(predicted, (70, 70, 70))
    cv2.rectangle(image, (0, 0), (min(760, image.shape[1]), 54), color, -1)
    cv2.putText(
        image,
        label,
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / image_path.name), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test fill level classifier (.pth)")
    parser.add_argument("--weights", default="models/fill_classifier_best.pth", help="Path to .pth weights")
    parser.add_argument("--source", required=True, help="Image path or folder path")
    parser.add_argument("--device", default="0", help="'0', 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--imgsz", type=int, default=0, help="Override checkpoint input size if > 0")
    parser.add_argument("--topk", type=int, default=3, help="How many class probabilities to print")
    parser.add_argument("--save-csv", default="", help="Optional CSV path to save predictions")
    parser.add_argument("--save-annotated", action="store_true", help="Save label overlay images")
    parser.add_argument("--no-save-annotated", action="store_true", help="Disable annotated image saving")
    parser.add_argument("--annotated-dir", default="inference/fill_level_annotated", help="Output dir for annotated images")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    source = Path(args.source)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    device = resolve_device(args.device)
    model, class_names, ckpt_imgsz = load_checkpoint(weights_path, device)
    imgsz = args.imgsz if args.imgsz > 0 else ckpt_imgsz

    images = collect_images(source)
    if not images:
        print(f"No image files found in: {source}")
        return

    print("\n" + "=" * 60)
    print("Fill Classifier Test")
    print("=" * 60)
    print(f"Weights    : {weights_path}")
    print(f"Source     : {source}")
    print(f"Images     : {len(images)}")
    print(f"Device     : {device}")
    print(f"Input size : {imgsz}")
    print(f"Classes    : {class_names}")
    print()

    counts = {name: 0 for name in class_names}
    csv_rows: list[dict] = []
    topk = max(1, min(args.topk, len(class_names)))
    annotated_dir = Path(args.annotated_dir)
    should_save_annotated = True
    if args.no_save_annotated:
        should_save_annotated = False
    elif args.save_annotated:
        should_save_annotated = True

    for image_path in images:
        tensor = preprocess_image(image_path, imgsz)
        probs = predict(model, tensor, device)
        pred_idx = int(torch.argmax(probs).item())
        predicted = class_names[pred_idx]
        confidence = float(probs[pred_idx].item())
        counts[predicted] = counts.get(predicted, 0) + 1

        ranked = torch.argsort(probs, descending=True)[:topk].tolist()
        ranked_text = " | ".join(
            f"{class_names[i]}={float(probs[i]):.2%}" for i in ranked
        )
        print(f"{image_path.name:<45} -> {predicted:<10} ({confidence:.2%}) | {ranked_text}")

        row = {
            "image": str(image_path),
            "predicted_class": predicted,
            "confidence": f"{confidence:.6f}",
        }
        for i, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = f"{float(probs[i]):.6f}"
        csv_rows.append(row)

        if should_save_annotated:
            maybe_save_annotated(image_path, predicted, confidence, annotated_dir)

    total = sum(counts.values())
    print("\n" + "-" * 60)
    print(f"Total images: {total}")
    for class_name in class_names:
        c = counts.get(class_name, 0)
        pct = (100.0 * c / total) if total else 0.0
        print(f"{class_name:<10}: {c:>5} ({pct:5.1f}%)")

    if args.save_csv:
        out_csv = Path(args.save_csv)
        save_predictions_csv(csv_rows, out_csv, class_names)
        print(f"\nSaved CSV: {out_csv}")

    if should_save_annotated:
        print(f"Saved annotated images: {annotated_dir}")


if __name__ == "__main__":
    main()
