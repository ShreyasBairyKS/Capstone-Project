"""
bottle-cap-train.py
===================
YOLOv11 training script for VisionFood QAI — beverage cap-quality detection.

Covers:
    • Model variant selection (nano → xlarge)
    • Hyperparameter configuration
    • Class-weighted loss for imbalanced defect data
    • Early stopping & best-model checkpointing
    • Automatic export to ONNX for edge deployment

Run:
    python training/bottle-cap-train.py --data dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml --model yolo11m --epochs 120 --profile a5000-balanced

After training, the best weights will be at:
    runs/detect/bottle_cap_defect/weights/best.pt
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# MODEL VARIANTS — choose based on your hardware target
# ─────────────────────────────────────────────────────────────────────────────
# yolo11n  →  Nano    — fastest, lowest accuracy  (edge RPi / microcontrollers)
# yolo11s  →  Small   — best starting point        ← RECOMMENDED for your case
# yolo11m  →  Medium  — more accuracy, needs GPU
# yolo11l  →  Large   — high accuracy, GPU required
# yolo11x  →  XLarge  — max accuracy, GPU required
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
SUPPORTED_PRODUCT_CATEGORIES = ["beverage", "food"]
SUPPORTED_PRODUCT_SUBTYPES = [
    "transparent_bottle",
    "rigid_can",
    "flexible_wrapper",
    "rigid_box",
]
DEFAULT_DATA_YAML = "dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml"
PROFILE_CHOICES = ["auto", "baseline", "a5000-fast", "a5000-balanced", "a5000-quality"]

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
            "yolo11n": 96,
            "yolo11s": 64,
            "yolo11m": 40,
            "yolo11l": 24,
            "yolo11x": 16,
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
            "yolo11n": 72,
            "yolo11s": 48,
            "yolo11m": 28,
            "yolo11l": 16,
            "yolo11x": 10,
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
            "yolo11n": 48,
            "yolo11s": 32,
            "yolo11m": 20,
            "yolo11l": 12,
            "yolo11x": 8,
        },
    },
}


def _infer_profile_model_key(model_arg: str) -> str:
    if model_arg in SUPPORTED_MODELS:
        return model_arg

    token = Path(model_arg).stem.lower()
    for candidate in SUPPORTED_MODELS:
        if candidate in token:
            return candidate

    return "yolo11s"


def _resolve_model_weights(model_arg: str) -> tuple[str, str]:
    if model_arg in SUPPORTED_MODELS:
        return f"{model_arg}.pt", model_arg

    candidate = Path(model_arg).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()

    if candidate.suffix.lower() != ".pt":
        raise ValueError(
            f"Unsupported --model value: {model_arg}. "
            "Use a YOLO variant (yolo11n/s/m/l/x) or a .pt checkpoint path."
        )

    assert candidate.exists(), f"Model checkpoint not found: {candidate}"
    return str(candidate), candidate.name


def _get_system_ram_gb() -> float | None:
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
    info = {
        "available": False,
        "name": None,
        "vram_gb": None,
        "device_index": None,
    }

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
    if "A5000" in gpu_name and (ram_gb is None or ram_gb >= 96):
        return "a5000-balanced"
    return "baseline"


def _resolve_runtime_settings(args, training_profile: str) -> dict:
    runtime = dict(BASELINE_DEFAULTS)
    profile_model_key = args.profile_model or _infer_profile_model_key(args.model)

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


def _resolve_split_path(data_yaml: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (data_yaml.parent / split_value).resolve()
    return split_path


def build_hyperparams(args, data_yaml: Path, runtime: dict) -> dict:
    """
    Returns the Ultralytics training hyperparameter dict.

    Key decisions explained:
    ─────────────────────────────────────────────────────────────────────────
    imgsz=640       Standard YOLO input resolution. Good for bottle images.
                    Use 320 only if you're on a severely constrained edge device.

    batch=16        Adjust based on your GPU VRAM:
                        8 GB  → batch=16
                        4 GB  → batch=8
                        CPU   → batch=4

    optimizer=AdamW Outperforms SGD on small/imbalanced defect datasets.
                    SGD works better when you have 10k+ images.

    lr0=0.001       Starting LR for AdamW. Lower than YOLO's default (0.01)
                    because defect datasets are small — too high LR = overfitting.

    lrf=0.01        Final LR fraction. Cosine schedule decays to lr0 × lrf.

    weight_decay    L2 regularisation. Helps with small datasets.

    warmup_epochs   Lets the model stabilise before full LR kicks in.
                    Important when fine-tuning a pretrained backbone.

    mosaic=0.5      Mosaic augmentation: pastes 4 images together.
                    Helps with scale variation in defects. Set to 0.0 if
                    your defect labels are very small (can cause label noise).

    mixup=0.1       Mixes two images with a random weight.
                    Improves generalisation on small defect datasets.

    degrees=10      Random rotation ±10°. Bottles on conveyor can tilt slightly.

    flipud=0.2      Vertical flip — useful if bottles can be upside down.
    fliplr=0.5      Horizontal flip — standard for most defect types.

    cls=1.0         Classification loss weight. Increase if False Negatives
                    (missed defects) are worse for you than False Positives.

    box=7.5         Bounding box regression loss weight. Default is fine.

    patience=20     Early stopping: stops if no improvement for 20 epochs.
                    Prevents overfitting on small datasets.

    save_period=10  Save checkpoint every 10 epochs as backup.
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

        # Online augmentation (applied during training, not on disk)
        "mosaic":        0.5,
        "mixup":         0.1,
        "degrees":       10.0,
        "flipud":        0.2,
        "fliplr":        0.5,
        "hsv_h":         0.015,
        "hsv_s":         0.7,
        "hsv_v":         0.4,
        "translate":     0.1,
        "scale":         0.5,
        "shear":         2.0,
        "perspective":   0.0,

        # Training control
        "cos_lr":        runtime["cos_lr"],
        "multi_scale":   runtime["multi_scale"],
        "close_mosaic":  runtime["close_mosaic"],
        "deterministic": runtime["deterministic"],
        "patience":      args.patience,
        "save_period":   10,
        "val":           True,
        "plots":         True,   # saves confusion matrix, PR curve, etc.
        "verbose":       True,
    }


def train(args):
    data_yaml = _resolve_data_yaml(args.data)
    model_weights, model_label = _resolve_model_weights(args.model)
    dataset_info = _load_dataset_info(data_yaml)
    ram_gb = _get_system_ram_gb()
    gpu_info = _get_gpu_info(args.device)
    training_profile = _select_training_profile(args.profile, gpu_info, ram_gb)
    runtime = _resolve_runtime_settings(args, training_profile)

    print(f"\n{'='*55}")
    print("  VisionFood QAI — YOLOv11 Cap-Quality Training")
    print(f"{'='*55}")
    print(f"  Profile : {training_profile}")
    if gpu_info["available"]:
        print(f"  GPU     : {gpu_info['name']} ({gpu_info['vram_gb']:.1f} GB VRAM)")
    else:
        print("  GPU     : not detected (running with provided --device setting)")
    if ram_gb is not None:
        print(f"  RAM     : {ram_gb:.1f} GB")
    print(f"  Product : {args.product_category} / {args.product_sub_type}")
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
    print(f"{'='*55}\n")

    for split_name in ("train", "val", "test"):
        split_value = dataset_info["raw"].get(split_name)
        if split_value:
            split_path = _resolve_split_path(data_yaml, split_value)
            print(f"  {split_name:>5} split → {split_path}")
    print()

    # ── Load pretrained YOLOv11 ───────────────────────────────────────────────
    # Ultralytics downloads the COCO-pretrained weights automatically
    # on first run. We fine-tune from this pretrained backbone.
    print(f"[1/3] Loading pretrained weights: {model_weights}")
    model = YOLO(model_weights)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[2/3] Starting training...")
    params = build_hyperparams(args, data_yaml, runtime)
    results = model.train(**params)

    # ── Post-training: export for edge deployment ──────────────────────────────
    run_dir = Path("runs/detect") / args.run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    context_path = run_dir / "training_context.json"
    context_payload = {
        "training_profile": training_profile,
        "hardware": {
            "gpu": gpu_info["name"],
            "gpu_vram_gb": gpu_info["vram_gb"],
            "ram_gb": ram_gb,
        },
        "runtime": runtime,
        "product_category": args.product_category,
        "product_sub_type": args.product_sub_type,
        "data_yaml": str(data_yaml),
        "class_names": dataset_info["class_names"],
        "defect_classes_default": ["defectCap", "noCap"],
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with context_path.open("w", encoding="utf-8") as f:
        json.dump(context_payload, f, indent=2)

    if best_weights.exists() and args.export:
        print("\n[3/3] Exporting best model to ONNX...")
        export_model = YOLO(str(best_weights))

        # ONNX — universal format, runs on CPU/GPU/NPU across platforms
        exported = export_model.export(format="onnx", imgsz=runtime["imgsz"], simplify=True)
        onnx_path = Path(exported) if isinstance(exported, (str, Path)) else (best_weights.parent / "best.onnx")
        if not onnx_path.exists():
            fallback = sorted(best_weights.parent.glob("*.onnx"))
            onnx_path = fallback[0] if fallback else onnx_path

        assert onnx_path.exists(), "ONNX export completed but .onnx file was not found"
        print(f"  ✓ ONNX model → {onnx_path}")

        # TensorRT — use this if deploying on Jetson or NVIDIA GPU
        # Uncomment the line below when you decide on Jetson deployment:
        # export_model.export(format="engine", imgsz=args.imgsz, half=True)

    print("\n✅ Training complete!")
    print(f"   Best weights : {best_weights}")
    print(f"   Last weights : {last_weights}")
    print(f"   Training plots  : {run_dir}")
    print(f"   Context file    : {context_path}")
    print(f"\n   → Next step: run evaluate.py to validate on test set.\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for beverage cap-quality detection")
    parser.add_argument("--data",    default=DEFAULT_DATA_YAML,
                        help="Path to data.yaml")
    parser.add_argument("--model",   default="yolo11s",
                        help="YOLOv11 variant (yolo11n/s/m/l/x) or .pt checkpoint path")
    parser.add_argument("--profile-model", default=None, choices=SUPPORTED_MODELS,
                        help="Profile batch lookup model key when --model is a .pt checkpoint")
    parser.add_argument("--epochs",  default=100, type=int,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD", "Adam"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", default=0.001, type=float,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--cls", default=1.0, type=float,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--lrf", default=0.01, type=float,
                        help="Final LR fraction for scheduler (default: 0.01)")
    parser.add_argument("--patience", default=20, type=int,
                        help="Early stopping patience epochs (default: 20)")
    parser.add_argument("--profile", default="auto", choices=PROFILE_CHOICES,
                        help="Hardware profile: auto/baseline/a5000-fast/a5000-balanced/a5000-quality")
    parser.add_argument("--imgsz",   default=None, type=int,
                        help="Input image size override. If omitted, chosen from --profile")
    parser.add_argument("--batch",   default=None, type=int,
                        help="Batch size override. If omitted, chosen from --profile")
    parser.add_argument("--workers", default=None, type=int,
                        help="Data loader workers override. If omitted, chosen from --profile")
    parser.add_argument("--cache", default="auto", choices=["auto", "none", "ram", "disk"],
                        help="Dataset cache mode. 'auto' follows profile recommendation")
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Force mixed precision training on")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Force mixed precision training off")
    parser.set_defaults(amp=None)
    parser.add_argument("--device",  default="0",
                        help="Device: '0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--run-name", default="bottle_cap_defect",
                        help="Output run name under runs/detect/")
    parser.add_argument("--product-category", default="beverage",
                        choices=SUPPORTED_PRODUCT_CATEGORIES,
                        help="Active product category for this training run")
    parser.add_argument("--product-sub-type", default="transparent_bottle",
                        choices=SUPPORTED_PRODUCT_SUBTYPES,
                        help="Active product sub-type for this training run")
    parser.add_argument("--export",  action="store_true",
                        help="Export best model to ONNX after training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()