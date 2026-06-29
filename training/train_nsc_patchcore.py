"""
training/train_nsc_patchcore.py
================================
Stage 3b — PatchCore unsupervised anomaly detection training for NSC
cylindrical tube rotation-unwrap images.

Trains on GOOD-only patches extracted from the label band. BAD patches are
used exclusively for validation / threshold calibration.

Architecture:
    WideResNet-50-2 (pretrained ImageNet) -> feature extraction (layer2 + layer3)
    -> adaptive average pooling -> feature concatenation -> coreset subsampling
    -> k-NN memory bank for anomaly scoring

Memory budget (A500 8GB VRAM):
    - Backbone feature extraction: ~2GB
    - Memory bank: 1-2GB depending on coreset ratio
    - Batch size: 16 (conservative)
    - AMP (FP16) enabled for feature extraction

Usage:
    # Full training (after running prepare_nsc_dataset.py)
    python training/train_nsc_patchcore.py

    # Custom parameters for A500
    python training/train_nsc_patchcore.py --batch 16 --coreset-ratio 0.10 --device cuda:0

    # Smoke test with small coreset
    python training/train_nsc_patchcore.py --coreset-ratio 0.01 --max-patches 100

    # CPU-only (slower but works anywhere)
    python training/train_nsc_patchcore.py --device cpu --batch 8

Output:
    models/nsc_patchcore_membank.pt     — memory bank + coreset + config
    models/nsc_patchcore_backbone.onnx  — backbone feature extractor (ONNX)
    runs/nsc_patchcore/                 — validation report, heatmaps, metrics
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_DATA_ROOT = Path("dataset/NSC_prepared")
DEFAULT_OUTPUT = Path("runs/nsc_patchcore")
DEFAULT_MODELS = Path("models")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Simple dataset for loading pre-extracted patches from disk.

    Expected directory structure:
        root/
            good_train/  *.png
            good_val/    *.png
            bad_val/     *.png
    """

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int = 384,
        max_patches: int = 0,
    ) -> None:
        self.root = root / split
        self.image_size = image_size
        self.paths: list[Path] = []

        if not self.root.exists():
            print(f"  [WARN]  Patch directory not found: {self.root}")
            return

        self.paths = sorted(
            [p for p in self.root.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        )

        if max_patches > 0:
            self.paths = self.paths[:max_patches]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Return a blank tensor on read failure
            return torch.zeros(3, self.image_size, self.image_size), path.name

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Normalize to ImageNet stats
        img = img.astype(np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = (img[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor, path.name


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor (WideResNet-50-2)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Extract intermediate features from WideResNet-50-2 for PatchCore.

    Uses layer2 and layer3 outputs, then adaptive-pools and concatenates them
    to a fixed spatial resolution for consistent patch-level features.
    """

    def __init__(self, target_size: int = 24) -> None:
        super().__init__()
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval()

        # Extract layers up to layer3
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.target_size = target_size

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate layer2 + layer3 features.

        Returns: (B, C2+C3, target_size, target_size) feature maps
        """
        x = self.layer0(x)
        x = self.layer1(x)

        feat2 = self.layer2(x)   # (B, 512, H/8, W/8) for WRN50-2 -> (B, 1024, H/8, W/8)
        feat3 = self.layer3(feat2)  # (B, 1024, H/16, W/16) -> (B, 2048, H/16, W/16)

        # Adaptive pool to common spatial resolution
        feat2 = F.adaptive_avg_pool2d(feat2, self.target_size)
        feat3 = F.adaptive_avg_pool2d(feat3, self.target_size)

        # Concatenate along channel dimension
        features = torch.cat([feat2, feat3], dim=1)

        return features


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore Memory Bank
# ─────────────────────────────────────────────────────────────────────────────

class PatchCoreMemoryBank:
    """
    PatchCore memory bank with greedy coreset subsampling.

    Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection"
    (CVPR 2022)
    """

    def __init__(
        self,
        coreset_ratio: float = 0.10,
        k_nearest: int = 9,
        device: torch.device | str = "cpu",
    ) -> None:
        self.coreset_ratio = coreset_ratio
        self.k_nearest = k_nearest
        self.device = torch.device(device) if isinstance(device, str) else device

        self.memory_bank: torch.Tensor | None = None
        self.is_fitted = False

    def fit(self, features: torch.Tensor) -> None:
        """
        Build the memory bank from extracted features.

        Args:
            features: (N, C) feature vectors from good-only patches
        """
        print(f"\n  [MEMBANK] Building memory bank from {features.shape[0]} features "
              f"(dim={features.shape[1]})...")

        n_total = features.shape[0]
        n_coreset = max(1, int(n_total * self.coreset_ratio))

        print(f"  [MEMBANK] Coreset subsampling: {n_total} -> {n_coreset} "
              f"({self.coreset_ratio:.0%})")

        # Greedy coreset selection
        coreset_indices = self._greedy_coreset(features, n_coreset)

        self.memory_bank = features[coreset_indices].to(self.device)
        self.is_fitted = True

        print(f"  [MEMBANK] Memory bank: {self.memory_bank.shape} "
              f"({self.memory_bank.nbytes / 1e6:.1f} MB)")

    def _greedy_coreset(self, features: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Greedy coreset subsampling: iteratively select the point farthest from
        the current coreset. This maximizes coverage of the feature space.
        """
        n = features.shape[0]

        if n_select >= n:
            return torch.arange(n)

        # Work on CPU for large matrices to avoid GPU OOM
        feats = features.cpu()

        # Start with a random seed point
        selected = [random.randint(0, n - 1)]
        min_distances = torch.cdist(
            feats[selected[0]].unsqueeze(0), feats
        ).squeeze(0)

        for i in range(1, n_select):
            # Select point with maximum minimum distance to current coreset
            idx = torch.argmax(min_distances).item()
            selected.append(idx)

            # Update minimum distances
            new_distances = torch.cdist(
                feats[idx].unsqueeze(0), feats
            ).squeeze(0)
            min_distances = torch.minimum(min_distances, new_distances)

            if (i + 1) % 500 == 0:
                print(f"    Coreset selection: {i + 1}/{n_select}")

        return torch.tensor(selected, dtype=torch.long)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for a batch of features using k-NN distance.

        Args:
            features: (N, C) feature vectors to score

        Returns:
            scores: (N,) anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Memory bank not fitted. Call fit() first.")

        # Compute pairwise distances to memory bank
        # Process in chunks to avoid OOM
        chunk_size = 256
        all_scores = []

        for i in range(0, features.shape[0], chunk_size):
            chunk = features[i : i + chunk_size].to(self.device)
            distances = torch.cdist(chunk, self.memory_bank)
            # k-NN: take mean of k nearest distances
            topk_distances, _ = torch.topk(distances, self.k_nearest, dim=1, largest=False)
            scores = topk_distances.mean(dim=1)
            all_scores.append(scores.cpu())

        return torch.cat(all_scores)

    def save(self, path: Path, config: dict | None = None) -> None:
        """Save memory bank to disk."""
        save_dict = {
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "coreset_ratio": self.coreset_ratio,
            "k_nearest": self.k_nearest,
            "is_fitted": self.is_fitted,
            "config": config or {},
        }
        torch.save(save_dict, path)
        print(f"  [MEMBANK] Saved -> {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "PatchCoreMemoryBank":
        """Load memory bank from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls(
            coreset_ratio=data["coreset_ratio"],
            k_nearest=data["k_nearest"],
            device=device,
        )
        bank.memory_bank = data["memory_bank"].to(device)
        bank.is_fitted = data["is_fitted"]
        return bank


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """
    Extract features from all patches in a DataLoader.

    Returns:
        features: (N * spatial * spatial, C) flattened feature vectors
        names: list of patch filenames
    """
    model.eval()
    all_features = []
    all_names = []

    for batch_idx, (imgs, names) in enumerate(loader):
        imgs = imgs.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            feats = model(imgs)

        # Reshape: (B, C, H, W) -> (B*H*W, C)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)

        all_features.append(feats.cpu().float())
        all_names.extend(names)

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(loader)}, "
                  f"features so far: {sum(f.shape[0] for f in all_features)}")

    return torch.cat(all_features, dim=0), all_names


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(
    good_scores: np.ndarray,
    bad_scores: np.ndarray,
    target_recall: float = 0.99,
) -> dict:
    """
    Find the anomaly threshold that achieves target recall on bad samples.

    For FN=0 priority: we want to catch (nearly) all defects, accepting some
    false positives on good samples.

    Returns:
        dict with threshold, recall, precision, f1, auroc
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    # Combine scores and labels
    scores = np.concatenate([good_scores, bad_scores])
    labels = np.concatenate([
        np.zeros(len(good_scores)),   # 0 = good (normal)
        np.ones(len(bad_scores)),     # 1 = bad (anomaly)
    ])

    # AUROC
    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = 0.0

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # Find threshold achieving target recall
    # precision_recall_curve returns recall in descending order
    best_thresh = None
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= target_recall:
            f1 = 2 * p * r / max(p + r, 1e-10)
            if best_thresh is None or f1 > best_f1:
                best_thresh = float(t)
                best_f1 = f1
                best_precision = float(p)
                best_recall = float(r)

    # Fallback: if no threshold achieves target recall, use the lowest threshold
    if best_thresh is None:
        # Use a threshold below the minimum bad score
        if len(bad_scores) > 0:
            best_thresh = float(np.min(bad_scores) * 0.95)
        else:
            best_thresh = float(np.percentile(good_scores, 95))
        best_recall = 1.0
        best_precision = float(np.mean(good_scores > best_thresh))
        # Precision calculation: TP / (TP + FP)
        tp = np.sum(bad_scores >= best_thresh)
        fp = np.sum(good_scores >= best_thresh)
        best_precision = float(tp / max(tp + fp, 1))
        best_f1 = 2 * best_precision * best_recall / max(best_precision + best_recall, 1e-10)

    return {
        "threshold": best_thresh,
        "recall": best_recall,
        "precision": best_precision,
        "f1": best_f1,
        "auroc": float(auroc),
        "target_recall": target_recall,
        "good_score_mean": float(np.mean(good_scores)),
        "good_score_std": float(np.std(good_scores)),
        "good_score_p95": float(np.percentile(good_scores, 95)),
        "good_score_p99": float(np.percentile(good_scores, 99)),
        "bad_score_mean": float(np.mean(bad_scores)) if len(bad_scores) > 0 else None,
        "bad_score_min": float(np.min(bad_scores)) if len(bad_scores) > 0 else None,
        "bad_score_max": float(np.max(bad_scores)) if len(bad_scores) > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────

def export_backbone_onnx(
    model: FeatureExtractor,
    output_path: Path,
    image_size: int = 384,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Export the backbone feature extractor to ONNX for edge deployment."""
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  [ONNX] Backbone exported -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detector on NSC tube unwrap patches (Stage 3b)"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=f"Prepared dataset root with patches/ subdirectory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for training results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS,
        help=f"Directory for saved model files (default: {DEFAULT_MODELS})",
    )
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for feature extraction (default: 16, safe for A500 8GB)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--image-size", type=int, default=384,
                        help="Patch resize before backbone (default: 384)")
    parser.add_argument("--coreset-ratio", type=float, default=0.10,
                        help="Fraction of features to keep in memory bank (default: 0.10)")
    parser.add_argument("--k-nearest", type=int, default=9,
                        help="k for k-NN scoring (default: 9)")
    parser.add_argument("--target-recall", type=float, default=0.99,
                        help="Target recall for threshold calibration (default: 0.99)")
    parser.add_argument("--feature-dim", type=int, default=24,
                        help="Spatial dimension for feature adaptive pooling (default: 24)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cuda:0', 'cpu', '' for auto (default: '')")
    parser.add_argument("--max-patches", type=int, default=0,
                        help="Max patches per split for smoke testing (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX backbone export")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save per-patch anomaly score heatmaps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    patches_root = args.data_root / "patches"

    print(f"\n{'=' * 70}")
    print("  NSC PatchCore Anomaly Detection Training (Stage 3b)")
    print(f"{'=' * 70}")
    print(f"  Data root     : {args.data_root}")
    print(f"  Patches root  : {patches_root}")
    print(f"  Device        : {device}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Image size    : {args.image_size}")
    print(f"  Feature dim   : {args.feature_dim}")
    print(f"  Coreset ratio : {args.coreset_ratio}")
    print(f"  k-NN k        : {args.k_nearest}")
    print(f"  Target recall : {args.target_recall}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  GPU           : {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"{'=' * 70}")

    # ── Step 1: Load datasets ──
    print("\n[STEP 1] Loading patch datasets...")
    train_ds = PatchDataset(patches_root, "good_train", args.image_size, args.max_patches)
    val_good_ds = PatchDataset(patches_root, "good_val", args.image_size, args.max_patches)
    val_bad_ds = PatchDataset(patches_root, "bad_val", args.image_size, args.max_patches)

    print(f"  Train (good-only) : {len(train_ds)} patches")
    print(f"  Val (good)        : {len(val_good_ds)} patches")
    print(f"  Val (bad)         : {len(val_bad_ds)} patches")

    if len(train_ds) == 0:
        print("\n[ERROR] No training patches found.")
        print(f"  Expected: {patches_root}/good_train/*.png")
        print("  Run scripts/prepare_nsc_dataset.py first.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_good_loader = DataLoader(
        val_good_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    ) if len(val_good_ds) > 0 else None

    val_bad_loader = DataLoader(
        val_bad_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    ) if len(val_bad_ds) > 0 else None

    # ── Step 2: Build feature extractor ──
    print("\n[STEP 2] Building WideResNet-50-2 feature extractor...")
    feature_extractor = FeatureExtractor(target_size=args.feature_dim).to(device)
    feature_extractor.eval()

    total_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M (frozen)")

    # ── Step 3: Extract training features ──
    print("\n[STEP 3] Extracting features from training patches...")
    t0 = time.time()
    use_amp = device.type == "cuda"

    train_features, train_names = extract_features(
        feature_extractor, train_loader, device, use_amp=use_amp
    )
    train_time = time.time() - t0

    print(f"  Training features: {train_features.shape} "
          f"({train_features.nbytes / 1e6:.1f} MB)")
    print(f"  Extraction time: {train_time:.1f}s")

    # ── Step 4: Build memory bank ──
    print("\n[STEP 4] Building PatchCore memory bank...")
    t0 = time.time()
    memory_bank = PatchCoreMemoryBank(
        coreset_ratio=args.coreset_ratio,
        k_nearest=args.k_nearest,
        device="cpu",  # Keep on CPU to save GPU memory for scoring
    )
    memory_bank.fit(train_features)
    bank_time = time.time() - t0
    print(f"  Memory bank build time: {bank_time:.1f}s")

    # ── Step 5: Validate ──
    print("\n[STEP 5] Scoring validation patches...")
    t0 = time.time()

    good_patch_scores = None
    bad_patch_scores = None

    if val_good_loader is not None:
        print("  Scoring good validation patches...")
        val_good_features, val_good_names = extract_features(
            feature_extractor, val_good_loader, device, use_amp=use_amp
        )
        good_raw_scores = memory_bank.score(val_good_features)
        # Aggregate per-patch: reshape from (N*H*W) back to per-patch
        spatial_size = args.feature_dim * args.feature_dim
        n_patches = len(val_good_ds)
        if good_raw_scores.shape[0] >= n_patches * spatial_size:
            good_patch_scores = good_raw_scores.reshape(n_patches, spatial_size).max(dim=1).values.numpy()
        else:
            # Fallback: take mean
            good_patch_scores = good_raw_scores.numpy()
        print(f"  Good val scores: mean={np.mean(good_patch_scores):.4f}, "
              f"std={np.std(good_patch_scores):.4f}, "
              f"p95={np.percentile(good_patch_scores, 95):.4f}")

    if val_bad_loader is not None:
        print("  Scoring bad validation patches...")
        val_bad_features, val_bad_names = extract_features(
            feature_extractor, val_bad_loader, device, use_amp=use_amp
        )
        bad_raw_scores = memory_bank.score(val_bad_features)
        n_patches_bad = len(val_bad_ds)
        spatial_size = args.feature_dim * args.feature_dim
        if bad_raw_scores.shape[0] >= n_patches_bad * spatial_size:
            bad_patch_scores = bad_raw_scores.reshape(n_patches_bad, spatial_size).max(dim=1).values.numpy()
        else:
            bad_patch_scores = bad_raw_scores.numpy()
        print(f"  Bad val scores:  mean={np.mean(bad_patch_scores):.4f}, "
              f"std={np.std(bad_patch_scores):.4f}, "
              f"min={np.min(bad_patch_scores):.4f}")

    val_time = time.time() - t0
    print(f"  Validation time: {val_time:.1f}s")

    # ── Step 6: Threshold calibration ──
    threshold_result = None
    if good_patch_scores is not None and bad_patch_scores is not None:
        print("\n[STEP 6] Calibrating anomaly threshold...")
        threshold_result = calibrate_threshold(
            good_patch_scores, bad_patch_scores,
            target_recall=args.target_recall,
        )
        print(f"  AUROC     : {threshold_result['auroc']:.4f}")
        print(f"  Threshold : {threshold_result['threshold']:.4f}")
        print(f"  Recall    : {threshold_result['recall']:.4f}")
        print(f"  Precision : {threshold_result['precision']:.4f}")
        print(f"  F1        : {threshold_result['f1']:.4f}")
        print(f"  Good p95  : {threshold_result['good_score_p95']:.4f}")
        print(f"  Good p99  : {threshold_result['good_score_p99']:.4f}")
    elif good_patch_scores is not None:
        print("\n[STEP 6] No bad patches available — setting threshold from good distribution...")
        threshold_result = {
            "threshold": float(np.percentile(good_patch_scores, 99)),
            "recall": None,
            "precision": None,
            "f1": None,
            "auroc": None,
            "target_recall": args.target_recall,
            "good_score_mean": float(np.mean(good_patch_scores)),
            "good_score_std": float(np.std(good_patch_scores)),
            "good_score_p95": float(np.percentile(good_patch_scores, 95)),
            "good_score_p99": float(np.percentile(good_patch_scores, 99)),
        }
        print(f"  Threshold (p99): {threshold_result['threshold']:.4f}")
    else:
        print("\n[STEP 6] No validation data — skipping threshold calibration.")

    # ── Step 7: Save artifacts ──
    print("\n[STEP 7] Saving artifacts...")

    # Save memory bank
    membank_config = {
        "backbone": "wide_resnet50_2",
        "feature_layers": ["layer2", "layer3"],
        "feature_dim": args.feature_dim,
        "image_size": args.image_size,
        "coreset_ratio": args.coreset_ratio,
        "k_nearest": args.k_nearest,
        "n_train_patches": len(train_ds),
        "n_train_features": int(train_features.shape[0]),
        "threshold": threshold_result["threshold"] if threshold_result else None,
        "auroc": threshold_result["auroc"] if threshold_result else None,
    }
    membank_path = args.models_dir / "nsc_patchcore_membank.pt"
    memory_bank.save(membank_path, config=membank_config)

    # Save config YAML
    config_path = args.models_dir / "nsc_patchcore_config.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(membank_config, f, sort_keys=False)
    print(f"  Config -> {config_path}")

    # Save validation report
    report = {
        "training": {
            "n_train_patches": len(train_ds),
            "n_train_features": int(train_features.shape[0]),
            "feature_extraction_time_s": round(train_time, 1),
            "memory_bank_build_time_s": round(bank_time, 1),
            "validation_time_s": round(val_time, 1),
            "device": str(device),
        },
        "validation": {
            "n_good_val": len(val_good_ds) if val_good_ds else 0,
            "n_bad_val": len(val_bad_ds) if val_bad_ds else 0,
        },
        "threshold_calibration": threshold_result,
        "hyperparameters": {
            "backbone": "wide_resnet50_2",
            "feature_dim": args.feature_dim,
            "image_size": args.image_size,
            "coreset_ratio": args.coreset_ratio,
            "k_nearest": args.k_nearest,
            "batch_size": args.batch,
            "seed": args.seed,
        },
    }
    report_path = output_dir / "patchcore_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report -> {report_path}")

    # ONNX export
    if not args.skip_export:
        print("\n[STEP 8] Exporting backbone to ONNX...")
        try:
            onnx_path = args.models_dir / "nsc_patchcore_backbone.onnx"
            export_backbone_onnx(
                feature_extractor, onnx_path,
                image_size=args.image_size,
                device=torch.device("cpu"),
            )
        except Exception as e:
            print(f"  [WARN]  ONNX export failed: {e}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  PatchCore Training Complete")
    print(f"{'=' * 70}")
    print(f"  Memory bank   : {membank_path}")
    print(f"  Config        : {config_path}")
    print(f"  Report        : {report_path}")
    if threshold_result:
        print(f"  Threshold     : {threshold_result['threshold']:.4f}")
        if threshold_result.get("auroc"):
            print(f"  AUROC         : {threshold_result['auroc']:.4f}")
        if threshold_result.get("recall"):
            print(f"  Recall        : {threshold_result['recall']:.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
