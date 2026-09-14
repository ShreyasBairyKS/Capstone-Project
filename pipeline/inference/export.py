"""
Model Export Engine for YOLOv11m-seg Bottle Cap Defect Detection.
Exports master PyTorch checkpoint (best.pt) to:
1. ONNX (.onnx) - Portable, high-performance cross-platform graph.
2. TensorRT (.engine) - Hardware-compiled FP16 engine tailored for target NVIDIA GPUs.

Usage:
  python -m pipeline.inference.export --format onnx
  python -m pipeline.inference.export --format engine --half
"""
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics is required for model export. Please install via: pip install ultralytics")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("pipeline.inference.export")

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent.parent
    / "YOLOv11seg-X model"
    / "YOLO_TrainingScripts"
    / "student_yolo11m_distilled"
    / "weights"
    / "best.pt"
)


def export_model(
    checkpoint_path: Optional[Path] = None,
    export_format: str = "onnx",
    half: bool = False,
    dynamic: bool = True,
    imgsz: int = 640
) -> Path:
    """
    Exports PyTorch model checkpoint to ONNX or TensorRT engine.

    Args:
        checkpoint_path: Path to master best.pt checkpoint.
        export_format: Target format ('onnx' or 'engine').
        half: Enable FP16 half-precision optimization (recommended for TensorRT).
        dynamic: Enable dynamic input batch/resolution dimensions.
        imgsz: Target square input image resolution (default 640).

    Returns:
        Path to the exported model file.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"Master checkpoint not found at: {ckpt}")

    export_format = export_format.lower().strip()
    if export_format not in ("onnx", "engine", "openvino"):
        raise ValueError(f"Unsupported format '{export_format}'. Supported: onnx, engine, openvino")

    logger.info("=" * 65)
    logger.info(f"STARTING MODEL EXPORT: {ckpt.name} -> {export_format.upper()}")
    logger.info(f"Source Checkpoint : {ckpt}")
    logger.info(f"Export Format     : {export_format.upper()}")
    logger.info(f"FP16 Half-Prec    : {half}")
    logger.info(f"Dynamic Dimensions: {dynamic}")
    logger.info(f"Input Resolution  : {imgsz}x{imgsz}")
    logger.info("=" * 65)

    # TensorRT hardware check
    if export_format == "engine":
        if not torch.cuda.is_available():
            logger.warning(
                "NVIDIA CUDA GPU not detected on current host. TensorRT (.engine) compilation "
                "requires an active NVIDIA GPU with CUDA and TensorRT drivers installed. "
                "Compile on the target GPU machine or inside the NVIDIA Docker container."
            )

    logger.info(f"Loading master PyTorch checkpoint: {ckpt}...")
    model = YOLO(str(ckpt))

    export_kwargs = {
        "format": export_format,
        "dynamic": dynamic,
        "imgsz": imgsz
    }

    if export_format == "onnx":
        export_kwargs["simplify"] = True
        export_kwargs["opset"] = 12

    if half and (torch.cuda.is_available() or export_format == "engine"):
        export_kwargs["half"] = True

    logger.info(f"Compiling graph with parameters: {export_kwargs}...")
    output_path_str = model.export(**export_kwargs)
    output_path = Path(output_path_str)

    logger.info("=" * 65)
    logger.info(f"EXPORT COMPLETED SUCCESSFULLY!")
    logger.info(f"Generated File: {output_path}")
    logger.info(f"File Size     : {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info("=" * 65)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11m-seg bottle cap model to ONNX or TensorRT.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt checkpoint")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "openvino"], help="Target format")
    parser.add_argument("--half", action="store_true", help="Enable FP16 half-precision")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (default 640)")
    args = parser.parse_args()

    export_model(
        checkpoint_path=Path(args.weights) if args.weights else None,
        export_format=args.format,
        half=args.half,
        imgsz=args.imgsz
    )


if __name__ == "__main__":
    main()
