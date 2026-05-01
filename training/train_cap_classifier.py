"""
training/train_cap_classifier.py
================================
MobileNetV3-Small classifier for cap quality: good_cap vs defective_cap.

Stage 2 of the hybrid inspection pipeline.

Key features:
    • Focal Loss (γ=2.0) to penalize false negatives (defective→good)
    • Aspect-ratio-preserving letterbox resize (no distortion)
    • Class-weighted sampling for imbalanced datasets
    • Heavy augmentation: motion blur, glare, noise, rotation
    • MC Dropout for uncertainty quantification
    • Auto train/val split if only flat folders provided

Usage:
    python training/train_cap_classifier.py \
        --data-root data/caps \
        --epochs 50 \
        --batch 64 \
        --device 0

    # With auto-split from flat folders:
    python training/train_cap_classifier.py \
        --data-root data/caps \
        --auto-split \
        --val-ratio 0.2

Expected folder structure:
    data/caps/
      train/
        good_cap/      *.jpg
        defective_cap/ *.jpg
      val/
        good_cap/      *.jpg
        defective_cap/ *.jpg

    OR (with --auto-split):
    data/caps/
      good_cap/      *.jpg
      defective_cap/ *.jpg
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["good_cap", "defective_cap"]
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# FOCAL LOSS — penalizes false negatives for defective class
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary/multi-class classification.

    With γ=2.0, the loss down-weights easy (well-classified) examples and
    focuses training on hard examples — critical for catching defective caps
    that might otherwise be drowned out by the majority good_cap class.

    The α parameter additionally weights the defective class higher.
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal = alpha_t * focal

        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# LETTERBOX RESIZE — preserves aspect ratio
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_resize_pil(img, size: int):
    """Resize PIL image preserving aspect ratio with gray padding."""
    from PIL import Image
    w, h = img.size
    ratio = min(size / h, size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    new_img = Image.new("RGB", (size, size), (114, 114, 114))
    new_img.paste(img, (pad_left, pad_top))
    return new_img


class LetterboxResize:
    """Torchvision-compatible letterbox transform."""
    def __init__(self, size: int):
        self.size = size
    def __call__(self, img):
        return letterbox_resize_pil(img, self.size)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class MotionBlur:
    """Simulates motion blur from fast conveyor belt movement."""
    def __init__(self, kernel_size: int = 7, p: float = 0.3):
        self.kernel_size = kernel_size
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img)
        k = random.choice([self.kernel_size, self.kernel_size + 4])
        kernel = np.zeros((k, k))
        if random.random() > 0.5:
            kernel[k // 2, :] = 1.0 / k  # horizontal
        else:
            kernel[:, k // 2] = 1.0 / k  # vertical
        arr = cv2.filter2D(arr, -1, kernel)
        from PIL import Image
        return Image.fromarray(arr)


class GlareSim:
    """Simulates glare/specular reflection on cap surface."""
    def __init__(self, p: float = 0.2):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        r = random.randint(min(h, w) // 6, min(h, w) // 3)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = np.clip(1.0 - dist / r, 0, 1) ** 2
        intensity = random.uniform(50, 150)
        for c in range(3):
            arr[:, :, c] += mask * intensity
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(arr)


def get_transforms(split: str, imgsz: int):
    """Build transforms for train/val splits."""
    if split == "train":
        return T.Compose([
            LetterboxResize(imgsz),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
            MotionBlur(kernel_size=7, p=0.3),
            GlareSim(p=0.2),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return T.Compose([
        LetterboxResize(imgsz),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CapDataset(Dataset):
    """ImageFolder-style dataset for good_cap / defective_cap."""

    def __init__(self, root: Path, split: str, imgsz: int = 224):
        self.root = root / split
        self.transform = get_transforms(split, imgsz)
        self.samples: list[tuple[Path, int]] = []
        self._load()

    def _load(self):
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.glob("*.*")):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, cls_idx))

        if not self.samples:
            print(f"  ⚠️  No samples found in {self.root}")
            print(f"     Expected subfolders: {CLASS_NAMES}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# AUTO TRAIN/VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def auto_split(data_root: Path, val_ratio: float = 0.2, seed: int = 42):
    """Split flat good_cap/ defective_cap/ folders into train/ val/ subsets."""
    rng = random.Random(seed)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    for cls_name in CLASS_NAMES:
        src = data_root / cls_name
        if not src.is_dir():
            print(f"  [SKIP] {src} not found")
            continue

        images = sorted(p for p in src.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio))

        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for dst_split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            dst = data_root / dst_split / cls_name
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                shutil.copy2(img_path, dst / img_path.name)

        print(f"  {cls_name}: {len(train_imgs)} train, {len(val_imgs)} val")

    print(f"  Split complete → {train_dir}, {val_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """Build MobileNetV3-Small with MC Dropout head."""
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace classifier head with dropout + linear
    in_features = backbone.classifier[0].in_features

    class CapQualityClassifier(nn.Module):
        def __init__(self):
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
            x = self.classifier(x)
            return x

    return CapQualityClassifier()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = tn = 0  # for defective class (index 1)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)

        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        # Track defective-class metrics
        for p, g in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            if g == 1 and p == 1: tp += 1
            elif g == 0 and p == 1: fp += 1
            elif g == 1 and p == 0: fn += 1
            else: tn += 1

    acc = correct / max(total, 1)
    loss = total_loss / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return loss, acc, {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                       "recall": recall, "precision": precision, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 cap quality classifier (Stage 2)"
    )
    parser.add_argument("--data-root", default="data/caps",
                        help="Root folder with good_cap/ and defective_cap/ subfolders")
    parser.add_argument("--auto-split", action="store_true",
                        help="Auto-split flat folders into train/val")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio for --auto-split (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Initial learning rate (default: 3e-4)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout for MC Dropout (default: 0.3)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '0', 'cuda', 'cpu', '' for auto")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--run-name", default="cap_classifier_v1",
                        help="Run name for output directory")
    parser.add_argument("--export", action="store_true",
                        help="Export to ONNX after training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Data root not found: {data_root}"

    # Auto-split if requested
    if args.auto_split:
        print("\n  Auto-splitting dataset...")
        auto_split(data_root, args.val_ratio, args.seed)

    # Device
    if args.device:
        device = torch.device(args.device if args.device != "0" else "cuda:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dataset
    train_ds = CapDataset(data_root, "train", args.imgsz)
    val_ds = CapDataset(data_root, "val", args.imgsz)

    if len(train_ds) == 0:
        print("\n[ERROR] No training samples. Check folder structure.")
        print(f"  Expected: {data_root}/train/good_cap/ and {data_root}/train/defective_cap/")
        return

    train_counts = train_ds.class_counts()
    val_counts = val_ds.class_counts()

    print(f"\n{'='*60}")
    print("  Hybrid Pipeline — Stage 2: Cap Quality Classifier")
    print(f"{'='*60}")
    print(f"  Data root : {data_root}")
    print(f"  Classes   : {CLASS_NAMES}")
    print(f"  Train     : {train_counts}")
    print(f"  Val       : {val_counts}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  LR        : {args.lr}")
    print(f"  Focal γ   : {args.focal_gamma}")
    print(f"  Dropout   : {args.dropout}")
    print(f"{'='*60}\n")

    # Class-weighted sampling to handle imbalance
    class_counts_list = [train_counts[c] for c in CLASS_NAMES]
    total_samples = sum(class_counts_list)
    class_weights = [total_samples / max(c, 1) for c in class_counts_list]
    sample_weights = [class_weights[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(NUM_CLASSES, args.dropout).to(device)

    # Loss — Focal Loss with higher weight for defective class
    # alpha=[0.3, 0.7] means defective class gets 2.3× more loss weight
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        alpha=[0.3, 0.7],
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Output
    run_dir = Path("runs/classify") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_recall = 0.0
    best_weights = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"recall={val_metrics['recall']:.3f}  "
            f"FN={val_metrics['fn']}  ({elapsed:.1f}s)"
        )

        # Save best by RECALL (minimize false negatives)
        if val_metrics["recall"] > best_recall or (
            val_metrics["recall"] == best_recall and val_acc > best_recall
        ):
            best_recall = val_metrics["recall"]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "val_acc": val_acc,
                "val_recall": val_metrics["recall"],
                "val_metrics": val_metrics,
                "model_state_dict": best_weights,
                "class_names": CLASS_NAMES,
                "args": vars(args),
            }, run_dir / "best.pth")
            no_improve = 0
            print(f"  → Saved best (recall={best_recall:.3f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no recall improvement for {args.patience} epochs)")
                break

    # Save final model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / "cap_classifier_best.pth"
    shutil.copy2(run_dir / "best.pth", final_path)

    # Save training context
    context = {
        "pipeline_stage": "classification",
        "pipeline_version": "hybrid_v2",
        "class_names": CLASS_NAMES,
        "best_recall": best_recall,
        "model_architecture": "MobileNetV3-Small",
        "focal_loss_gamma": args.focal_gamma,
        "input_size": args.imgsz,
        "notes": "Optimized for recall (minimize false negatives on defective caps)",
    }
    with open(run_dir / "training_context.json", "w") as f:
        json.dump(context, f, indent=2)

    print(f"\n✅ Classifier training complete!")
    print(f"   Best recall : {best_recall:.4f}")
    print(f"   Weights     : {final_path}")
    print(f"   Run dir     : {run_dir}")

    # ONNX export
    if args.export and best_weights is not None:
        print("\n  Exporting to ONNX...")
        try:
            model.load_state_dict(best_weights)
            model.eval()
            dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
            onnx_path = models_dir / "cap_classifier_best.onnx"
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
            print(f"  ✓ ONNX → {onnx_path}")
        except Exception as e:
            print(f"  ⚠️  ONNX export failed: {e}")

    print(f"\n  → Next: run inference with hybrid_inference.py\n")


if __name__ == "__main__":
    main()
