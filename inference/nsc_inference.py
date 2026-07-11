# -*- coding: utf-8 -*-
"""
inference/nsc_inference.py
==========================
NSC Cylindrical Tube Anomaly Inference — GOOD / BAD classifier.

Uses the trained PatchCore pipeline:
    1. Load input image (BMP, PNG, JPG, etc.)
    2. Preprocess: resize patches → ImageNet normalize
    3. Extract features via ONNX backbone (WideResNet-50-2, layer2+layer3)
    4. Score each spatial location against the memory bank (k-NN distance)
    5. Aggregate patch scores → image-level score
    6. Compare against calibrated threshold → GOOD or BAD

Model files (in models/):
    nsc_patchcore_backbone.onnx   — ONNX feature extractor
                                    Input:  (N, 3, 384, 384) ImageNet-normalized
                                    Output: (N, 1536, 8, 8)  feature maps
    nsc_patchcore_membank.pt      — Memory bank
                                    Shape:  (967, 1536) coreset vectors
                                    Threshold: 2.7921
    nsc_patchcore_config.yaml     — Config (feature_dim, image_size, threshold, k)

Usage:
    # Single image
    python inference/nsc_inference.py path/to/image.bmp

    # Single image with heatmap saved
    python inference/nsc_inference.py path/to/image.bmp --save-heatmap

    # Custom threshold
    python inference/nsc_inference.py path/to/image.bmp --threshold 2.5

    # Batch — all images in a folder
    python inference/nsc_inference.py path/to/folder/ --batch

    # GPU inference
    python inference/nsc_inference.py path/to/image.bmp --device cuda

    # Custom models directory
    python inference/nsc_inference.py path/to/image.bmp --models-dir /path/to/models

Output:
    Console: GOOD / BAD result + anomaly score
    Optional: heatmap overlay image (--save-heatmap)
    Optional: JSON result file (--save-json)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match training configuration
# ─────────────────────────────────────────────────────────────────────────────

# WideResNet-50-2 layer2+layer3 feature channels (as produced by the ONNX model)
FEATURE_CHANNELS = 1536   # 512 (layer2) + 1024 (layer3)
SPATIAL_DIM      = 8      # adaptive pool target size (8×8 = 64 locations)
IMAGE_SIZE       = 384    # input patch size (pixels)
PATCH_STRIDE     = 192    # 50% overlap between patches

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_BACKBONE   = "nsc_patchcore_backbone.onnx"
DEFAULT_MEMBANK    = "nsc_patchcore_membank.pt"
DEFAULT_CONFIG     = "nsc_patchcore_config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime Session
# ─────────────────────────────────────────────────────────────────────────────

def load_onnx_session(onnx_path: Path, device: str = "cpu"):
    """Load the ONNX backbone feature extractor session."""
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options,
                                   providers=providers)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Memory Bank
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBank:
    """
    PatchCore k-NN memory bank for anomaly scoring.

    Loaded from the .pt file saved during training.
    Memory bank shape: (967, 1536) — 967 coreset vectors × 1536 feature dims.
    """

    def __init__(self, bank: np.ndarray, k: int = 9) -> None:
        self.bank = bank      # (N_coreset, 1536) float32
        self.k    = k

    @classmethod
    def load(cls, membank_path: Path) -> "MemoryBank":
        data = torch.load(str(membank_path), map_location="cpu", weights_only=False)
        bank = data["memory_bank"].numpy().astype(np.float32)
        k    = data.get("k_nearest", 9)
        print(f"  [MEMBANK] Loaded: {bank.shape[0]} vectors × {bank.shape[1]} dims  "
              f"(k={k})")
        return cls(bank, k)

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute k-NN anomaly score for each feature vector.

        Args:
            features: (N, 1536) array of feature vectors

        Returns:
            scores: (N,) anomaly scores — higher = more anomalous
        """
        chunk_size  = 256
        all_scores  = []

        for i in range(0, len(features), chunk_size):
            chunk = features[i : i + chunk_size]          # (C, 1536)
            # Pairwise L2 distances: (C, N_coreset)
            diffs = chunk[:, None, :] - self.bank[None, :, :]  # (C, N_c, 1536)
            dists = np.sqrt((diffs ** 2).sum(axis=-1))          # (C, N_c)
            # k nearest neighbours
            k = min(self.k, dists.shape[1])
            idx = np.argpartition(dists, k, axis=1)[:, :k]
            topk = np.take_along_axis(dists, idx, axis=1)
            scores = topk.mean(axis=1)                           # (C,)
            all_scores.append(scores)

        return np.concatenate(all_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_patch(patch_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Resize and normalize a single BGR patch to a model-ready float32 array.

    Returns: (1, 3, size, size) float32 NCHW array, ImageNet-normalized.
    """
    patch = cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.astype(np.float32) / 255.0

    # ImageNet normalization per channel
    patch = (patch - IMAGENET_MEAN) / IMAGENET_STD   # (H, W, 3)
    patch = patch.transpose(2, 0, 1)[np.newaxis]     # (1, 3, H, W)
    return np.ascontiguousarray(patch, dtype=np.float32)


def extract_patches_sliding(
    image_bgr: np.ndarray,
    patch_size: int = IMAGE_SIZE,
    stride: int = PATCH_STRIDE,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Extract overlapping patches from the full image using a sliding window.

    Returns: list of (y_start, x_start, patch_bgr)
    """
    h, w = image_bgr.shape[:2]
    patches = []

    if h < patch_size or w < patch_size:
        # Image smaller than patch — resize the whole image to patch size
        patches.append((0, 0, cv2.resize(image_bgr, (patch_size, patch_size))))
        return patches

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append((y, x, image_bgr[y:y + patch_size, x:x + patch_size]))

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Core Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_image(
    image_bgr: np.ndarray,
    session,
    membank: MemoryBank,
    threshold: float,
    patch_size: int = IMAGE_SIZE,
    stride: int = PATCH_STRIDE,
    spatial_dim: int = SPATIAL_DIM,
    batch_size: int = 8,
) -> dict:
    """
    Run PatchCore inference on one image.

    Returns a dict with:
        is_bad       (bool)   — True if anomaly detected
        label        (str)    — 'BAD' or 'GOOD'
        score        (float)  — maximum patch anomaly score
        threshold    (float)  — decision threshold
        patch_scores (list)   — per-patch max anomaly scores
        heatmap_info (dict)   — position + score for heatmap construction
    """
    patches = extract_patches_sliding(image_bgr, patch_size, stride)
    if not patches:
        return {
            "is_bad": False, "label": "GOOD",
            "score": 0.0, "threshold": threshold,
            "patch_scores": [], "heatmap_info": [],
        }

    # Batch feature extraction through ONNX backbone
    input_name  = session.get_inputs()[0].name   # "image"
    output_name = session.get_outputs()[0].name  # "features"

    all_features = []  # will be (N_patches, 1536) after processing

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i : i + batch_size]
        # Stack preprocessed patches: (B, 3, 384, 384)
        batch_input = np.concatenate(
            [preprocess_patch(p_bgr, patch_size) for _, _, p_bgr in batch_patches],
            axis=0,
        )
        # ONNX forward: (B, 1536, 8, 8)
        feat_maps = session.run([output_name], {input_name: batch_input})[0]

        B, C, H, W = feat_maps.shape
        # Flatten spatial: (B, C, H, W) → (B×H×W, C) = (B×64, 1536)
        feat_flat = feat_maps.transpose(0, 2, 3, 1).reshape(-1, C)
        all_features.append(feat_flat)

    all_features = np.concatenate(all_features, axis=0)  # (N_patches × 64, 1536)

    # Score all feature vectors against the memory bank
    raw_scores = membank.score(all_features)              # (N_patches × 64,)

    # Aggregate: per-spatial-location → per-patch (max score)
    spatial_locs = spatial_dim * spatial_dim              # 8×8 = 64
    n_patches    = len(patches)

    if raw_scores.shape[0] >= n_patches * spatial_locs:
        patch_scores = (
            raw_scores.reshape(n_patches, spatial_locs)
            .max(axis=1)
        )                                                 # (N_patches,)
    else:
        # Fallback if shapes don't perfectly align
        patch_scores = raw_scores.reshape(n_patches, -1).max(axis=1)

    # Image-level score = worst patch score (maximum anomaly)
    image_score = float(patch_scores.max())
    is_bad      = image_score > threshold

    # Build heatmap info for optional visualization
    heatmap_info = [
        {"y": y, "x": x, "score": float(s)}
        for (y, x, _), s in zip(patches, patch_scores)
    ]

    return {
        "is_bad":       is_bad,
        "label":        "BAD" if is_bad else "GOOD",
        "score":        image_score,
        "threshold":    threshold,
        "patch_scores": patch_scores.tolist(),
        "heatmap_info": heatmap_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap Visualization
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap_overlay(
    image_bgr: np.ndarray,
    heatmap_info: list[dict],
    patch_size: int = IMAGE_SIZE,
    threshold: float = 2.79,
) -> np.ndarray:
    """
    Build a color heatmap overlay on the image showing anomaly scores.

    Hot colors (red/yellow) = high anomaly score.
    Cool colors (blue/green) = normal.
    """
    h, w = image_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    count   = np.zeros((h, w), dtype=np.float32)

    for info in heatmap_info:
        y, x, score = info["y"], info["x"], info["score"]
        y2 = min(y + patch_size, h)
        x2 = min(x + patch_size, w)
        heatmap[y:y2, x:x2] += score
        count[y:y2, x:x2]   += 1

    heatmap = np.where(count > 0, heatmap / count, 0)

    # Normalize to [0, 255] with threshold as the reference minimum
    score_max = heatmap.max()
    if score_max > threshold:
        norm = np.clip((heatmap - threshold) / (score_max - threshold + 1e-6), 0, 1)
    else:
        norm = np.clip(heatmap / (threshold + 1e-6), 0, 1)

    heatmap_u8    = (norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # Blend with original image
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap_color, 0.45, 0)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Result Display
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict, image_name: str, elapsed: float) -> None:
    """Print a formatted result banner to the console."""
    label     = result["label"]
    score     = result["score"]
    threshold = result["threshold"]
    margin    = score - threshold
    n_patches = len(result["patch_scores"])

    if label == "BAD":
        banner_char = "[X]"
        verdict     = "ANOMALY DETECTED - BAD"
    else:
        banner_char = "[OK]"
        verdict     = "NO ANOMALY - GOOD"

    print(f"\n{'=' * 56}")
    print(f"  {banner_char}  {verdict}")
    print(f"{'=' * 56}")
    print(f"  Image     : {image_name}")
    print(f"  Score     : {score:.4f}  (threshold: {threshold:.4f})")
    print(f"  Margin    : {margin:+.4f}  ({'above' if margin > 0 else 'below'} threshold)")
    print(f"  Patches   : {n_patches}")
    print(f"  Inference : {elapsed * 1000:.1f} ms")
    print(f"{'=' * 56}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NSC Cylindrical Tube Anomaly Inference\n"
            "Classifies an image as GOOD or BAD using PatchCore + ONNX backbone.\n\n"
            "Models used:\n"
            "  nsc_patchcore_backbone.onnx  — ONNX feature extractor\n"
            "  nsc_patchcore_membank.pt     — k-NN memory bank (threshold=2.7921)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to an image file (BMP/PNG/JPG) or a folder of images (use with --batch)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
        help=f"Directory containing model files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--backbone", type=str, default=DEFAULT_BACKBONE,
        help=f"ONNX backbone filename inside --models-dir (default: {DEFAULT_BACKBONE})",
    )
    parser.add_argument(
        "--membank", type=str, default=DEFAULT_MEMBANK,
        help=f"Memory bank filename inside --models-dir (default: {DEFAULT_MEMBANK})",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            "Override anomaly threshold (default: from config YAML or 2.7921).\n"
            "Lower = more sensitive (more BAD), Higher = less sensitive (more GOOD)."
        ),
    )
    parser.add_argument(
        "--patch-size", type=int, default=IMAGE_SIZE,
        help=f"Patch size for sliding window (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--stride", type=int, default=PATCH_STRIDE,
        help=f"Sliding window stride (default: {PATCH_STRIDE}, = 50%% overlap)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Patches per ONNX forward pass (default: 8)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Inference device for ONNX Runtime (default: cpu)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all images in a folder (input must be a directory)",
    )
    parser.add_argument(
        "--save-heatmap", action="store_true",
        help="Save anomaly heatmap overlay image alongside the input",
    )
    parser.add_argument(
        "--heatmap-dir", type=Path, default=Path("runs/nsc_inference"),
        help="Output directory for heatmaps (default: runs/nsc_inference/)",
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save inference result as a JSON file in --heatmap-dir",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve model paths ─────────────────────────────────────────────────
    backbone_path = args.models_dir / args.backbone
    membank_path  = args.models_dir / args.membank
    config_path   = args.models_dir / DEFAULT_CONFIG

    for p, name in [(backbone_path, "ONNX backbone"), (membank_path, "Memory bank")]:
        if not p.exists():
            print(f"[ERROR] {name} not found: {p}")
            print(f"  Expected at: {p.resolve()}")
            return

    # ── Load config for threshold and feature params ─────────────────────────
    threshold = args.threshold
    spatial_dim = SPATIAL_DIM

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if threshold is None:
            threshold = float(cfg.get("threshold", 2.7921))
        spatial_dim = int(cfg.get("feature_dim", SPATIAL_DIM))
        img_size    = int(cfg.get("image_size",  args.patch_size))
        print(f"  [CONFIG] Loaded: threshold={threshold:.4f}, "
              f"spatial_dim={spatial_dim}, image_size={img_size}")
    else:
        print(f"  [CONFIG] Config not found at {config_path}, using defaults.")
        if threshold is None:
            threshold = 2.7921

    # -- Load models ----------------------------------------------------------
    print(f"\n{'=' * 56}")
    print("  NSC PatchCore Inference")
    print(f"{'=' * 56}")
    print(f"  ONNX backbone : {backbone_path.name}  ({backbone_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Memory bank   : {membank_path.name}")
    print(f"  Threshold     : {threshold:.4f}")
    print(f"  Patch size    : {args.patch_size}x{args.patch_size}")
    print(f"  Stride        : {args.stride}")
    print(f"  Device        : {args.device}")

    t_load = time.time()
    session = load_onnx_session(backbone_path, device=args.device)
    membank = MemoryBank.load(membank_path)
    print(f"  Load time     : {(time.time() - t_load) * 1000:.0f} ms")

    # ── Collect image paths ──────────────────────────────────────────────────
    if args.batch:
        if not args.input.is_dir():
            print(f"[ERROR] --batch requires a directory, got: {args.input}")
            return
        image_paths = sorted([
            p for p in args.input.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ])
        print(f"\n  Batch mode: {len(image_paths)} images found in {args.input}")
    else:
        if not args.input.exists():
            print(f"[ERROR] Input not found: {args.input}")
            return
        image_paths = [args.input]

    if not image_paths:
        print("[ERROR] No images found.")
        return

    # ── Output directory ─────────────────────────────────────────────────────
    if args.save_heatmap or args.save_json:
        args.heatmap_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────────
    all_results = []
    n_good = n_bad = 0

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  [WARN]  Cannot read image: {img_path}")
            continue

        t_start = time.time()
        result  = infer_image(
            image_bgr      = image_bgr,
            session        = session,
            membank        = membank,
            threshold      = threshold,
            patch_size     = args.patch_size,
            stride         = args.stride,
            spatial_dim    = spatial_dim,
            batch_size     = args.batch_size,
        )
        elapsed = time.time() - t_start

        result["image_path"] = str(img_path)
        all_results.append(result)

        if result["is_bad"]:
            n_bad += 1
        else:
            n_good += 1

        # Print result
        print_result(result, img_path.name, elapsed)

        # Save heatmap
        if args.save_heatmap:
            overlay = build_heatmap_overlay(
                image_bgr, result["heatmap_info"],
                patch_size=args.patch_size, threshold=threshold,
            )
            label_str = result["label"]
            stem      = img_path.stem
            out_path  = args.heatmap_dir / f"{stem}_{label_str}_heatmap.png"
            cv2.imwrite(str(out_path), overlay)
            print(f"  Heatmap saved → {out_path}")

        # Save JSON for single-image
        if args.save_json and not args.batch:
            json_result = {k: v for k, v in result.items() if k != "heatmap_info"}
            json_path   = args.heatmap_dir / f"{img_path.stem}_result.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, indent=2)
            print(f"  JSON saved    → {json_path}")

    # ── Batch summary ────────────────────────────────────────────────────────
    if len(image_paths) > 1:
        print(f"\n{'=' * 56}")
        print(f"  BATCH SUMMARY  ({len(all_results)} images)")
        print(f"{'=' * 56}")
        print(f"  GOOD  : {n_good:>4}  ({100 * n_good / max(len(all_results), 1):.1f}%)")
        print(f"  BAD   : {n_bad:>4}  ({100 * n_bad  / max(len(all_results), 1):.1f}%)")
        print(f"{'=' * 56}")

        if args.save_json:
            batch_json = args.heatmap_dir / "batch_results.json"
            summaries  = [
                {k: v for k, v in r.items() if k != "heatmap_info"}
                for r in all_results
            ]
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump({"n_good": n_good, "n_bad": n_bad, "results": summaries},
                          f, indent=2)
            print(f"  Batch JSON → {batch_json}")


if __name__ == "__main__":
    main()
