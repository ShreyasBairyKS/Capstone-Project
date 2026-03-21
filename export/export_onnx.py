"""
export/export_onnx.py — Export trained PyTorch models to ONNX FP16 format.

Supports:
  - YOLOv11n  : exported via Ultralytics built-in ONNX export (FP16)
  - EfficientViT-M5 : exported via torch.onnx.export with training=True
                       so MC Dropout remains active in the ONNX graph

Usage:
    # Export both models
    python export/export_onnx.py

    # Export only YOLOv11
    python export/export_onnx.py --model yolo

    # Export only EfficientViT
    python export/export_onnx.py --model efficientvit

    # Custom paths
    python export/export_onnx.py \\
        --yolo-weights runs/train/yolov11n_visionfood_v1/weights/best.pt \\
        --efficientvit-weights models/efficientvit_m5_best.pth \\
        --output-dir models/

Output:
    models/yolov11n_best.onnx          (FP16, opset 17)
    models/efficientvit_m5_best.onnx   (FP16, opset 17, training=True dropout)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.logging import get_logger, setup_logging

setup_logging(log_format="text")
log = get_logger(__name__)

YOLO_WEIGHTS_DEFAULT = Path("runs/train/yolov11n_visionfood_v1/weights/best.pt")
EFFICIENTVIT_WEIGHTS_DEFAULT = Path("models/efficientvit_m5_best.pth")
OUTPUT_DIR_DEFAULT = Path("models")

NUM_CLASSES = 4
DEFECT_CLASS_NAMES = [
    "improper_filling",
    "packaging_damage",
    "label_misalignment",
    "surface_contamination",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export VisionFood QAI models to ONNX FP16."
    )
    parser.add_argument(
        "--model",
        choices=["yolo", "efficientvit", "both"],
        default="both",
        help="Which model(s) to export (default: both)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=YOLO_WEIGHTS_DEFAULT,
        help=f"Path to YOLOv11 .pt weights (default: {YOLO_WEIGHTS_DEFAULT})",
    )
    parser.add_argument(
        "--efficientvit-weights",
        type=Path,
        default=EFFICIENTVIT_WEIGHTS_DEFAULT,
        help=f"Path to EfficientViT .pth weights (default: {EFFICIENTVIT_WEIGHTS_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Directory to write ONNX models into (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Export in FP16 precision (default: True)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# YOLOv11 export (via Ultralytics)
# --------------------------------------------------------------------------- #

def export_yolov11(
    weights: Path,
    output_dir: Path,
    opset: int = 17,
    fp16: bool = True,
) -> Path:
    """
    Export YOLOv11 .pt → .onnx using Ultralytics built-in export.

    Args:
        weights:    Path to best.pt from training
        output_dir: Where to copy the resulting .onnx
        opset:      ONNX opset version
        fp16:       Export with FP16 flag (Ultralytics handles internally)

    Returns:
        Path to the ONNX file in output_dir
    """
    log.info("yolov11_export_start", weights=str(weights), opset=opset, fp16=fp16)

    if not weights.exists():
        raise FileNotFoundError(
            f"YOLOv11 weights not found at: {weights}\n"
            "Run training first: python training/train_yolov11.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed — run: pip install ultralytics")

    model = YOLO(str(weights))

    # Ultralytics exports alongside the weights file; we copy to output_dir
    export_result = model.export(
        format="onnx",
        imgsz=640,
        opset=opset,
        half=fp16,
        simplify=True,
        dynamic=False,
    )

    # Ultralytics returns the path to the exported ONNX file
    exported_path = Path(str(export_result))
    if not exported_path.exists():
        # Fallback: look for .onnx adjacent to weights
        exported_path = weights.with_suffix(".onnx")

    if not exported_path.exists():
        raise RuntimeError(
            f"Expected ONNX at {exported_path} but it was not created. "
            "Check Ultralytics export output above."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "yolov11n_best.onnx"
    shutil.copy2(exported_path, dest)

    log.info("yolov11_export_complete", output=str(dest), size_mb=round(dest.stat().st_size / 1e6, 2))
    print(f"\n[OK] YOLOv11 ONNX → {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# EfficientViT-M5 export (via torch.onnx)
# --------------------------------------------------------------------------- #

def _build_efficientvit_model(num_classes: int = NUM_CLASSES) -> "torch.nn.Module":
    """Load EfficientViT-M5 from timm with the capstone head."""
    import torch
    import torch.nn as nn
    import timm

    backbone = timm.create_model(
        "efficientvit_m5",
        pretrained=False,
        num_classes=0,          # Remove classifier head — we add our own
        global_pool="avg",
    )
    in_features = backbone.num_features

    class EfficientViTDefectClassifier(nn.Module):
        def __init__(self, backbone: nn.Module, num_classes: int, p_drop: float = 0.3) -> None:
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(p=p_drop)
            self.head = nn.Linear(in_features, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feats = self.backbone(x)
            feats = self.dropout(feats)
            return self.head(feats)

    model = EfficientViTDefectClassifier(backbone, num_classes=num_classes)
    return model


def export_efficientvit(
    weights: Path,
    output_dir: Path,
    opset: int = 17,
    fp16: bool = True,
) -> Path:
    """
    Export EfficientViT-M5 .pth → .onnx with dropout layers active.

    The key requirement is exporting with training=True so that nn.Dropout
    stays stochastic at ONNX Runtime inference time — enabling MC Dropout UQ.

    Args:
        weights:    Path to .pth checkpoint (state_dict)
        output_dir: Where to write the .onnx
        opset:      ONNX opset version
        fp16:       Cast model to FP16 before export

    Returns:
        Path to the ONNX file
    """
    log.info("efficientvit_export_start", weights=str(weights), opset=opset, fp16=fp16)

    if not weights.exists():
        raise FileNotFoundError(
            f"EfficientViT weights not found at: {weights}\n"
            "Run training first: python training/train_efficientvit.py"
        )

    try:
        import torch
        import timm  # noqa: F401 — verify installed
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Run: pip install torch timm")

    model = _build_efficientvit_model(num_classes=NUM_CLASSES)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    # Support both raw state_dict and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)

    # Put entire model in eval mode first (BatchNorm uses running stats).
    # Then selectively re-enable only Dropout layers so they remain stochastic
    # during MC Dropout UQ passes at inference time.
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    if fp16:
        model = model.half()

    dummy_input = torch.zeros(1, 3, 224, 224, dtype=torch.float16 if fp16 else torch.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "efficientvit_m5_best.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(dest),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        training=torch.onnx.TrainingMode.TRAINING,   # Keeps Dropout stochastic
        do_constant_folding=False,                    # Must be False for MC Dropout
        export_params=True,
    )

    log.info(
        "efficientvit_export_complete",
        output=str(dest),
        size_mb=round(dest.stat().st_size / 1e6, 2),
    )
    print(f"\n[OK] EfficientViT ONNX → {dest}  ({dest.stat().st_size / 1e6:.2f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# Validation: quick sanity-check the exported ONNX models
# --------------------------------------------------------------------------- #

def validate_onnx(onnx_path: Path, input_shape: tuple) -> None:
    """Run a dummy forward pass through OnnxRuntime to verify the export."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        log.warning("onnxruntime_not_installed_skipping_validation")
        return

    input_name = "images" if "yolo" in onnx_path.name else "input"
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    dummy = np.zeros(input_shape, dtype=np.float32)
    outputs = session.run(None, {input_name: dummy})
    log.info("onnx_validation_ok", model=onnx_path.name, output_shapes=[o.shape for o in outputs])
    print(f"  Validation pass: input {input_shape} → outputs {[o.shape for o in outputs]}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("yolo", "both"):
        yolo_out = export_yolov11(
            weights=args.yolo_weights,
            output_dir=args.output_dir,
            opset=args.opset,
            fp16=args.fp16,
        )
        print("  Validating YOLOv11 ONNX…")
        validate_onnx(yolo_out, (1, 3, 640, 640))

    if args.model in ("efficientvit", "both"):
        eff_out = export_efficientvit(
            weights=args.efficientvit_weights,
            output_dir=args.output_dir,
            opset=args.opset,
            fp16=args.fp16,
        )
        print("  Validating EfficientViT ONNX…")
        validate_onnx(eff_out, (1, 3, 224, 224))

    print("\n✓ Export complete. Models saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
