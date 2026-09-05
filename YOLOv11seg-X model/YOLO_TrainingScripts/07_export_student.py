"""
Script 07: Export Student Model for Jetson Orin Deployment

Exports yolo11m-seg student to:
  1. TensorRT FP16 (.engine) -- PRIMARY: Maximum speed on Jetson Orin GPU
  2. ONNX (.onnx)            -- BACKUP:  CPU fallback / validation tool

Jetson Orin Notes:
  - Run this script ON the Jetson Orin (not on training PC)
    because TensorRT engines are device-specific.
  - OR cross-compile using trtexec on an x86 machine with
    the Jetson SDK installed (not recommended — run on device).
  - FP16 is the sweet spot on Orin: ~2x speedup vs FP32 with minimal accuracy drop.
  - INT8 is possible but requires a calibration dataset (add --int8 flag below).

Expected Jetson Orin Inference Speed:
  - FP16 TensorRT: ~2-4ms per image at 640x640
  - Compared to Teacher on A5000: ~18ms
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import shutil

# ─── Paths ────────────────────────────────────────────────────────────────────
STUDENT_WEIGHTS = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\student_yolo11m_distilled\weights\best.pt"
EXPORT_DIR      = r"D:\Yolo Dataset\YOLO_TrainingScripts\exported_models"
CALIB_DATA      = r"D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\data.yaml"  # For INT8 calibration


def export_student(weights_path, export_dir, use_int8=False):
    weights_path = Path(weights_path)
    export_dir   = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"[Error] Weights not found: {weights_path}")
        print("Run 05_train_student_distill.py first.")
        return

    model = YOLO(str(weights_path))

    print("=" * 65)
    print("  STUDENT MODEL EXPORT FOR JETSON ORIN")
    print(f"  Model   : yolo11m-seg (feature-KD distilled)")
    print(f"  Source  : {weights_path.name}")
    print(f"  Output  : {export_dir}")
    print("=" * 65)

    # ── 1. Export to ONNX ────────────────────────────────────────────────────
    print("\n[1/2] Exporting to ONNX (backup / validation format)...")
    onnx_result = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,    # Fixed shape for Jetson TensorRT compatibility
        simplify=True,    # Graph simplification via onnxsim
        opset=17,
        half=False,       # ONNX stays FP32; TensorRT handles FP16 conversion
        device="cpu",
    )

    # Copy ONNX to export dir
    if onnx_result is None:
        print("  [Error] ONNX export returned None. Check ultralytics version.")
    else:
        onnx_src = Path(onnx_result)
        onnx_dst = export_dir / onnx_src.name
        shutil.copy2(str(onnx_src), str(onnx_dst))
        print(f"  ONNX saved   : {onnx_dst}")

    # ── 2. Export to TensorRT FP16 (Jetson Orin primary) ─────────────────────
    print(f"\n[2/2] Exporting to TensorRT FP16 (Jetson Orin primary format)...")
    print("  NOTE: If running on x86 training PC, this builds an x86 engine.")
    print("  For Jetson-specific engine: copy best.pt to Jetson and run this script there.")

    try:
        trt_result = model.export(
            format="engine",       # TensorRT .engine
            imgsz=640,
            half=True,             # FP16 -- optimal for Jetson Orin GPU
            device=0,
            workspace=4,           # GB of GPU workspace for TRT optimization
            int8=use_int8,
            data=CALIB_DATA if use_int8 else None,
            simplify=True,
        )
        if trt_result is None:
            print("  [Error] TensorRT export returned None. Run on Jetson or ensure TRT is installed.")
        else:
            trt_src = Path(trt_result)
            trt_dst = export_dir / trt_src.name
            shutil.copy2(str(trt_src), str(trt_dst))
            print(f"  TensorRT FP16 saved : {trt_dst}")

    except Exception as e:
        print(f"  [Warn] TensorRT export failed: {e}")
        print("  This is expected if CUDA/TRT is not installed on this machine.")
        print("  Copy best.pt to Jetson Orin and run this script there.")

    # ── Deployment Summary ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DEPLOYMENT PACKAGE SUMMARY")
    print("─" * 65)
    for f in sorted(export_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<45} {size_mb:>6.1f} MB")

    print("\n" + "─" * 65)
    print("  JETSON ORIN DEPLOYMENT INSTRUCTIONS")
    print("─" * 65)
    print("  1. Transfer the .engine file (or best.pt) to the Jetson Orin")
    print("  2. On the Jetson, install: pip install ultralytics")
    print("  3. Run inference:")
    print()
    print("     from ultralytics import YOLO")
    print("     model = YOLO('best.engine')")
    print("     results = model('image.jpg', imgsz=640)")
    print()
    print("  4. If using best.pt on Jetson, export to TRT there:")
    print("     model.export(format='engine', half=True, imgsz=640)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w",  "--weights",  default=STUDENT_WEIGHTS)
    parser.add_argument("-o",  "--output",   default=EXPORT_DIR)
    parser.add_argument("--int8", action="store_true",
                        help="Also export INT8 engine (requires calibration data)")
    args = parser.parse_args()
    export_student(args.weights, args.output, use_int8=args.int8)
