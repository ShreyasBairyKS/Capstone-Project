"""
training/train_detector.py
==========================
YOLOv8 training script for 2-class bottle/cap detection.

This is Stage 1 of the hybrid inspection pipeline.
Detects only TWO classes:
    0: bottle
    1: cap

The detector does NOT classify defects — that is handled by the
Stage 2 classifier (train_cap_classifier.py).

Run:
    python training/train_detector.py \
        --data configs/detection_data.yaml \
        --model yolov8s \
        --epochs 150 \
        --profile a5000-balanced

    # Resume from checkpoint:
    python training/train_detector.py \
        --model runs/detect/bottle_cap_det_v2/weights/last.pt \
        --data configs/detection_data.yaml \
        --epochs 150 \
        --profile a5000-balanced

After training, the best weights will be at:
    runs/detect/bottle_cap_det_v2/weights/best.pt
"""

import argparse
import json
import os
import ctypes
from pathlib import Path

import yaml
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# MODEL VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
# yolov8n  →  Nano    — fastest, edge devices
# yolov8s  →  Small   — best balance for 2-class detection  ← RECOMMENDED
# yolov8m  →  Medium  — more accuracy, GPU required
# yolov8l  →  Large   — high accuracy, GPU required
# yolov8x  →  XLarge  — max accuracy, GPU required
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

EXPECTED_CLASSES = ["bottle", "cap"]

BASELINE_DEFAULTS = {
    "imgsz": 640,
    "batch": 16,
    "workers": 8,
    "cache": False,
    "amp": True,
    "cos_lr": False,
    "multi_scale": False,
    "close_mosaic": 15,
    "deterministic": False,
}

A5000_PROFILE_PRESETS = {
    "a5000-fast": {
        "imgsz": 640,
        "workers": 20,
        "cache": "ram",
        "amp": True,
        "cos_lr": False,
        "multi_scale": False,
        "close_mosaic": 15,
        "deterministic": False,
        "batch_by_model": {
            "yolov8n": 128,
            "yolov8s": 96,
            "yolov8m": 48,
            "yolov8l": 32,
            "yolov8x": 20,
        },
    },
    "a5000-balanced": {
        "imgsz": 736,
        "workers": 16,
        "cache": "ram",
        "amp": True,
        "cos_lr": True,
        "multi_scale": True,
        "close_mosaic": 12,
        "deterministic": False,
        "batch_by_model": {
            "yolov8n": 96,
            "yolov8s": 64,
            "yolov8m": 36,
            "yolov8l": 20,
            "yolov8x": 14,
        },
    },
    "a5000-quality": {
        "imgsz": 896,
        "workers": 12,
        "cache": "ram",
        "amp": True,
        "cos_lr": True,
        "multi_scale": True,
        "close_mosaic": 8,
        "deterministic": True,
        "batch_by_model": {
            "yolov8n": 64,
            "yolov8s": 40,
            "yolov8m": 24,
            "yolov8l": 14,
            "yolov8x": 10,
        },
    },
}

PROFILE_CHOICES = ["auto", "baseline", "a5000-fast", "a5000-balanced", "a5000-quality"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _infer_profile_model_key(model_arg: str) -> str:
    """Map a model argument to a known variant key for batch-size lookup."""
    if model_arg in SUPPORTED_MODELS:
        return model_arg

    token = Path(model_arg).stem.lower()
    for candidate in SUPPORTED_MODELS:
        if candidate in token:
            return candidate

    return "yolov8s"


def _resolve_model_weights(model_arg: str) -> tuple[str, str]:
    """Resolve model argument to (weights_path, human_label)."""
    if model_arg in SUPPORTED_MODELS:
        return f"{model_arg}.pt", model_arg

    candidate = Path(model_arg).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if candidate.suffix.lower() != ".pt":
        raise ValueError(
            f"Unsupported --model value: {model_arg}. "
            "Use a YOLOv8 variant (yolov8n/s/m/l/x) or a .pt checkpoint path."
        )

    assert candidate.exists(), f"Model checkpoint not found: {candidate}"
    return str(candidate), candidate.name


def _get_system_ram_gb() -> float | None:
    """Detect system RAM in GB."""
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)

        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 ** 3)
    except Exception:
        return None
    return None


def _parse_device_index(device: str) -> int:
    token = str(device).split(",")[0].strip().lower()
    if token in {"cpu", "mps"}:
        return -1
    if token.startswith("cuda:"):
        token = token.split(":", 1)[1]
    if token.isdigit():
        return int(token)
    return 0


def _get_gpu_info(device: str) -> dict:
    info = {"available": False, "name": None, "vram_gb": None, "device_index": None}
    try:
        import torch
        if str(device).lower() == "cpu" or not torch.cuda.is_available():
            return info
        idx = _parse_device_index(device)
        if idx < 0:
            return info
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "available": True,
            "name": props.name,
            "vram_gb": props.total_memory / (1024 ** 3),
            "device_index": idx,
        })
    except Exception:
        return info
    return info


def _resolve_cache_setting(cache_arg: str, default_cache):
    if cache_arg == "auto":
        return default_cache
    if cache_arg == "none":
        return False
    return cache_arg


def _select_training_profile(profile_arg: str, gpu_info: dict, ram_gb: float | None) -> str:
    if profile_arg != "auto":
        return profile_arg
    gpu_name = (gpu_info.get("name") or "").upper()
    if "A5000" in gpu_name and (ram_gb is None or ram_gb >= 64):
        return "a5000-balanced"
    return "baseline"


def _resolve_runtime_settings(args, training_profile: str) -> dict:
    runtime = dict(BASELINE_DEFAULTS)
    profile_model_key = _infer_profile_model_key(args.model)

    if training_profile in A5000_PROFILE_PRESETS:
        preset = A5000_PROFILE_PRESETS[training_profile]
        runtime.update({
            "imgsz": preset["imgsz"],
            "workers": preset["workers"],
            "cache": preset["cache"],
            "amp": preset["amp"],
            "cos_lr": preset["cos_lr"],
            "multi_scale": preset["multi_scale"],
            "close_mosaic": preset["close_mosaic"],
            "deterministic": preset["deterministic"],
            "batch": preset["batch_by_model"][profile_model_key],
        })

    if args.imgsz is not None:
        runtime["imgsz"] = args.imgsz
    if args.batch is not None:
        runtime["batch"] = args.batch
    if args.workers is not None:
        runtime["workers"] = args.workers

    runtime["cache"] = _resolve_cache_setting(args.cache, runtime["cache"])
    if args.amp is not None:
        runtime["amp"] = args.amp

    return runtime


def _resolve_data_yaml(path_str: str) -> Path:
    data_yaml = Path(path_str).expanduser()
    if not data_yaml.is_absolute():
        data_yaml = Path.cwd() / data_yaml
    data_yaml = data_yaml.resolve()
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"
    return data_yaml


def _load_dataset_info(data_yaml: Path) -> dict:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names_node = cfg.get("names", [])
    if isinstance(names_node, dict):
        class_names = [names_node[k] for k in sorted(names_node, key=lambda x: int(x))]
    else:
        class_names = list(names_node)

    return {
        "class_names": class_names,
        "n_classes": len(class_names),
        "raw": cfg,
    }


def _validate_classes(class_names: list[str]) -> None:
    """Warn if dataset classes don't match expected 2-class setup."""
    if set(class_names) != set(EXPECTED_CLASSES):
        print(f"\n  ⚠️  WARNING: Expected classes {EXPECTED_CLASSES} but got {class_names}")
        print(f"     The detector should only detect 'bottle' and 'cap'.")
        print(f"     Do NOT include defect classes — those go in the classifier.\n")


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def build_hyperparams(args, data_yaml: Path, runtime: dict) -> dict:
    """
    Returns the Ultralytics training hyperparameter dict.

    Key decisions for 2-class (bottle + cap) detection:
    ─────────────────────────────────────────────────────────────────────────
    optimizer=AdamW  Better for smaller datasets than SGD.
    lr0=0.001        Conservative LR for fine-tuning.
    mosaic=0.8       High mosaic for spatial diversity.
    mixup=0.15       Mild mixup for regularization.
    degrees=15       Bottles can tilt on conveyor belt.
    hsv_h/s/v        Aggressive color jitter to prevent color overfitting.
    translate=0.15   Simulates position variation on belt.
    scale=0.5        Multi-scale robustness.
    patience=25      Generous early stopping for 2-class problem.
    ─────────────────────────────────────────────────────────────────────────
    """
    return {
        "data":          str(data_yaml),
        "epochs":        args.epochs,
        "imgsz":         runtime["imgsz"],
        "batch":         runtime["batch"],
        "workers":       runtime["workers"],
        "cache":         runtime["cache"],
        "amp":           runtime["amp"],
        "device":        args.device,
        "project":       "runs/detect",
        "name":          args.run_name,
        "exist_ok":      True,

        # Optimiser
        "optimizer":     args.optimizer,
        "lr0":           args.lr0,
        "lrf":           args.lrf,
        "momentum":      0.937,
        "weight_decay":  0.0005,
        "warmup_epochs": 5,
        "warmup_bias_lr": 0.1,

        # Loss weights
        "box":           7.5,
        "cls":           args.cls,
        "dfl":           1.5,

        # Online augmentation — aggressive to prevent single-bottle overfitting
        "mosaic":        0.8,
        "mixup":         0.15,
        "degrees":       15.0,
        "flipud":        0.2,
        "fliplr":        0.5,
        "hsv_h":         0.02,     # wider hue range than default
        "hsv_s":         0.75,     # aggressive saturation jitter
        "hsv_v":         0.5,      # aggressive value jitter
        "translate":     0.15,
        "scale":         0.5,
        "shear":         3.0,
        "perspective":   0.0001,
        "erasing":       0.1,      # random erasing for robustness

        # Training control
        "cos_lr":        runtime["cos_lr"],
        "multi_scale":   runtime["multi_scale"],
        "close_mosaic":  runtime["close_mosaic"],
        "deterministic": runtime["deterministic"],
        "patience":      args.patience,
        "save_period":   10,
        "val":           True,
        "plots":         True,
        "verbose":       True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    data_yaml = _resolve_data_yaml(args.data)
    model_weights, model_label = _resolve_model_weights(args.model)
    dataset_info = _load_dataset_info(data_yaml)
    ram_gb = _get_system_ram_gb()
    gpu_info = _get_gpu_info(args.device)
    training_profile = _select_training_profile(args.profile, gpu_info, ram_gb)
    runtime = _resolve_runtime_settings(args, training_profile)

    _validate_classes(dataset_info["class_names"])

    print(f"\n{'='*60}")
    print("  Hybrid Pipeline — Stage 1: Bottle/Cap Detection Training")
    print(f"{'='*60}")
    print(f"  Profile : {training_profile}")
    if gpu_info["available"]:
        print(f"  GPU     : {gpu_info['name']} ({gpu_info['vram_gb']:.1f} GB VRAM)")
    else:
        print("  GPU     : not detected (running with provided --device setting)")
    if ram_gb is not None:
        print(f"  RAM     : {ram_gb:.1f} GB")
    print(f"  Model   : {model_label}")
    print(f"  Data    : {data_yaml}")
    print(f"  Classes : {dataset_info['class_names']}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSize : {runtime['imgsz']}")
    print(f"  Batch   : {runtime['batch']}")
    print(f"  Workers : {runtime['workers']}")
    print(f"  Cache   : {runtime['cache'] if runtime['cache'] else 'off'}")
    print(f"  AMP     : {runtime['amp']}")
    print(f"  Device  : {args.device}")
    print(f"  RunName : {args.run_name}")
    print(f"{'='*60}\n")

    for split_name in ("train", "val", "test"):
        split_value = dataset_info["raw"].get(split_name)
        if split_value:
            split_path = Path(split_value)
            if not split_path.is_absolute():
                split_path = (data_yaml.parent / split_value).resolve()
            print(f"  {split_name:>5} split → {split_path}")
    print()

    # ── Load pretrained YOLOv8 ─────────────────────────────────────────────
    print(f"[1/3] Loading pretrained weights: {model_weights}")
    model = YOLO(model_weights)

    # ── Train ──────────────────────────────────────────────────────────────
    print("[2/3] Starting training...")
    params = build_hyperparams(args, data_yaml, runtime)
    results = model.train(**params)

    # ── Post-training ──────────────────────────────────────────────────────
    run_dir = Path("runs/detect") / args.run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    context_path = run_dir / "training_context.json"
    context_payload = {
        "pipeline_stage": "detection",
        "pipeline_version": "hybrid_v2",
        "training_profile": training_profile,
        "hardware": {
            "gpu": gpu_info["name"],
            "gpu_vram_gb": gpu_info["vram_gb"],
            "ram_gb": ram_gb,
        },
        "runtime": runtime,
        "data_yaml": str(data_yaml),
        "class_names": dataset_info["class_names"],
        "notes": "2-class detector for bottle + cap only. No defect classes.",
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with context_path.open("w", encoding="utf-8") as f:
        json.dump(context_payload, f, indent=2)

    if best_weights.exists() and args.export:
        print("\n[3/3] Exporting best model to ONNX...")
        export_model = YOLO(str(best_weights))
        exported = export_model.export(
            format="onnx", imgsz=runtime["imgsz"], simplify=True
        )
        onnx_path = (
            Path(exported) if isinstance(exported, (str, Path))
            else (best_weights.parent / "best.onnx")
        )
        if not onnx_path.exists():
            fallback = sorted(best_weights.parent.glob("*.onnx"))
            onnx_path = fallback[0] if fallback else onnx_path
        print(f"  ✓ ONNX model → {onnx_path}")

    print("\n✅ Detection training complete!")
    print(f"   Best weights : {best_weights}")
    print(f"   Last weights : {last_weights}")
    print(f"   Context file : {context_path}")
    print(f"\n   → Next: use cap_crop_extractor.py to build classification dataset")
    print(f"   → Then: train classifier with train_cap_classifier.py\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for 2-class bottle/cap detection (Stage 1 of hybrid pipeline)"
    )
    parser.add_argument("--data", default="configs/detection_data.yaml",
                        help="Path to data.yaml (must have classes: bottle, cap)")
    parser.add_argument("--model", default="yolov8s",
                        help="YOLOv8 variant (yolov8n/s/m/l/x) or .pt checkpoint path")
    parser.add_argument("--epochs", default=150, type=int,
                        help="Number of training epochs (default: 150)")
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD", "Adam"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", default=0.001, type=float,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--cls", default=1.0, type=float,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--lrf", default=0.01, type=float,
                        help="Final LR fraction for scheduler (default: 0.01)")
    parser.add_argument("--patience", default=25, type=int,
                        help="Early stopping patience epochs (default: 25)")
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES,
                        help="Hardware profile: auto/baseline/a5000-fast/a5000-balanced/a5000-quality")
    parser.add_argument("--imgsz", default=None, type=int,
                        help="Input image size override")
    parser.add_argument("--batch", default=None, type=int,
                        help="Batch size override")
    parser.add_argument("--workers", default=None, type=int,
                        help="Data loader workers override")
    parser.add_argument("--cache", default="auto", choices=["auto", "none", "ram", "disk"],
                        help="Dataset cache mode")
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Force mixed precision training on")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Force mixed precision training off")
    parser.set_defaults(amp=None)
    parser.add_argument("--device", default="0",
                        help="Device: '0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--run-name", default="bottle_cap_det_v2",
                        help="Output run name under runs/detect/")
    parser.add_argument("--export", action="store_true",
                        help="Export best model to ONNX after training")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
