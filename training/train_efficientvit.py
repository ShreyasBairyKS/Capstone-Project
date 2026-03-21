"""
training/train_efficientvit.py — Fine-tune EfficientViT-M5 on the VisionFood defect dataset.

Phase 2 deliverable — classifier for 4 defect classes.

The model is exported with MC Dropout active (training=True in ONNX export)
so that the UQ inspector can run Monte Carlo passes at inference time.

Usage:
    python training/train_efficientvit.py
    python training/train_efficientvit.py --epochs 40 --batch 32 --freeze-backbone 5

Output:
    runs/classify/efficientvit_visionfood_v1/best.pth    (best val-acc checkpoint)
    models/efficientvit_m5_best.pth                       (copied from above)
    models/efficientvit_m5_best.onnx                      (FP16 ONNX with dropout)
"""

from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import settings
from core.logging import get_logger, setup_logging

setup_logging(log_format="text")
log = get_logger(__name__)

DATA_ROOT = Path("data/annotated")
OUTPUT_MODELS = Path("models")
NUM_CLASSES = 4
DEFECT_CLASS_NAMES = [
    "improper_filling",
    "packaging_damage",
    "label_misalignment",
    "surface_contamination",
]

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientViT-M5 defect classifier on VisionFood dataset."
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--freeze-backbone", type=int, default=0,
                        help="Freeze backbone for this many epochs then unfreeze (default: 0)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cuda', 'cpu', or '' to auto-detect (default: '')")
    parser.add_argument("--project", type=str, default="runs/classify",
                        help="Parent directory for run outputs")
    parser.add_argument("--name", type=str, default="efficientvit_visionfood_v1",
                        help="Run subdirectory name")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability for MC Dropout head (default: 0.3)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Dataset — reuses YOLOv11 classification export structure:
#
#   data/annotated/
#     train/   (images sorted into class-named subfolders by Roboflow export)
#       improper_filling/
#       packaging_damage/
#       label_misalignment/
#       surface_contamination/
#     val/
#       ...
#     test/
#       ...
#
# If the dataset was exported in detection format (images + labels .txt), the
# helper `_build_classification_dataset` converts bounding boxes to crops.
# --------------------------------------------------------------------------- #

def _get_transforms(split: str, imgsz: int):
    """Return Albumentations-based transforms for train / val / test splits."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        HAS_ALBUMENTATIONS = True
    except ImportError:
        HAS_ALBUMENTATIONS = False

    if not HAS_ALBUMENTATIONS:
        # Minimal torchvision fallback
        from torchvision import transforms as T
        if split == "train":
            return T.Compose([
                T.Resize((imgsz, imgsz)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return T.Compose([
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if split == "train":
        return A.Compose([
            A.Resize(imgsz, imgsz),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(imgsz, imgsz),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class DefectClassificationDataset(torch.utils.data.Dataset):
    """
    ImageFolder-style dataset for the VisionFood defect classification task.

    Expects either:
      data/annotated/{split}/{class_name}/*.jpg   (classification format)
      data/annotated/{split}/images/*.jpg         (detection format — auto-crops)
    """

    def __init__(self, root: Path, split: str, imgsz: int = 224) -> None:
        self.root = root / split
        self.split = split
        self.transform = _get_transforms(split, imgsz)
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        # Classification layout: {split}/{class_name}/*.{jpg,png,jpeg}
        for cls_idx, cls_name in enumerate(DEFECT_CLASS_NAMES):
            cls_dir = self.root / cls_name
            if cls_dir.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
                    for img_path in cls_dir.glob(ext):
                        self.samples.append((img_path, cls_idx))

        if not self.samples:
            # Try detection layout: extract crops from images + label .txt
            images_dir = self.root / "images"
            labels_dir = self.root / "labels"
            if images_dir.is_dir() and labels_dir.is_dir():
                self.samples = self._collect_detection_crops(images_dir, labels_dir)

        log.info("dataset_loaded", split=self.split, n_samples=len(self.samples))
        if not self.samples:
            log.warning(
                "no_samples_found",
                path=str(self.root),
                hint="Ensure data/annotated/ has classification or detection layout.",
            )

    def _collect_detection_crops(
        self, images_dir: Path, labels_dir: Path
    ) -> list[tuple[Path, int]]:
        """Yield (image_path, label_path) pairs from YOLOv11 detection layout."""
        import cv2, numpy as np

        samples = []
        for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
            label_path = labels_dir / img_path.with_suffix(".txt").name
            if not label_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                # Save crop to a temp path so DataLoader can use __getitem__
                crop_dir = self.root / "_crops" / DEFECT_CLASS_NAMES[cls_id % NUM_CLASSES]
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{img_path.stem}_crop_{len(samples)}.jpg"
                if not crop_path.exists():
                    cv2.imwrite(str(crop_path), crop)
                samples.append((crop_path, cls_id % NUM_CLASSES))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                raise OSError(f"Could not read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            # Fallback to Pillow
            from PIL import Image
            import numpy as np
            img = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if hasattr(self.transform, "__call__"):
            try:
                # Albumentations
                transformed = self.transform(image=img)
                img_tensor = transformed["image"]
            except TypeError:
                # Torchvision
                from PIL import Image
                img_pil = Image.fromarray(img)
                img_tensor = self.transform(img_pil)
        return img_tensor.float(), label


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """Build EfficientViT-M5 with a dropout + linear head for MC Dropout UQ."""
    import timm

    backbone = timm.create_model(
        "efficientvit_m5",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )
    in_features = backbone.num_features
    log.info("backbone_loaded", model="efficientvit_m5", in_features=in_features)

    class EfficientViTDefectClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(p=dropout)
            self.head = nn.Linear(in_features, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            feats = self.dropout(feats)
            return self.head(feats)

    return EfficientViTDefectClassifier()


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

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
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    # Check dataset
    if not (DATA_ROOT / "train").exists():
        log.error("dataset_not_found", path=str(DATA_ROOT))
        print(f"\n[ERROR] Dataset not found at {DATA_ROOT}/train/")
        print("Complete Phase 0 dataset annotation first.")
        print("Expected layout:")
        print("  data/annotated/train/{class_name}/*.jpg")
        print("  data/annotated/val/{class_name}/*.jpg")
        return

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("training_device", device=str(device))

    # W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="visionfood-qai", name=args.name, config=vars(args))
        except Exception as e:
            log.warning("wandb_init_failed", error=str(e))

    # Data
    train_ds = DefectClassificationDataset(DATA_ROOT, "train", args.imgsz)
    val_ds = DefectClassificationDataset(DATA_ROOT, "val", args.imgsz)

    if len(train_ds) == 0:
        print("\n[ERROR] No training samples found. Check dataset layout.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(num_classes=NUM_CLASSES, dropout=args.dropout).to(device)

    # Optionally freeze backbone for warm-up phase
    if args.freeze_backbone > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        log.info("backbone_frozen", epochs=args.freeze_backbone)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Output directory
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_weights = None

    print(f"\nTraining EfficientViT-M5 for {args.epochs} epochs on {len(train_ds)} samples")
    print(f"Val set: {len(val_ds)} samples | Device: {device}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Unfreeze backbone after warm-up
        if args.freeze_backbone > 0 and epoch == args.freeze_backbone + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer to include backbone params
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
            log.info("backbone_unfrozen", epoch=epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s)"
        )
        log.info(
            "epoch_complete",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
        )

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
            })

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            ckpt_path = run_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "model_state_dict": best_weights,
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info("checkpoint_saved", epoch=epoch, val_acc=round(val_acc, 4), path=str(ckpt_path))

    print(f"\nBest val accuracy: {best_acc:.4f}")

    # Copy best checkpoint to models/
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    final_pth = OUTPUT_MODELS / "efficientvit_m5_best.pth"
    shutil.copy2(run_dir / "best.pth", final_pth)
    print(f"Checkpoint → {final_pth}")

    # ONNX export
    print("\nExporting to ONNX (FP16)…")
    try:
        from export.export_onnx import export_efficientvit, validate_onnx
        onnx_path = export_efficientvit(
            weights=final_pth,
            output_dir=OUTPUT_MODELS,
            opset=17,
            fp16=True,
        )
        validate_onnx(onnx_path, (1, 3, 224, 224))
    except Exception as e:
        log.warning("onnx_export_failed", error=str(e))
        print(f"  [WARNING] ONNX export failed: {e}")
        print("  Run manually: python export/export_onnx.py --model efficientvit")

    if wandb_run:
        wandb_run.finish()

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
