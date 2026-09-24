r"""Validate four ROI ONNX exports before submitting them.

Run from PowerShell, for example:

    python validate_roi_onnx.py --models-dir "E:\P-25 Vision Food ai\models\rois"

Install the optional validation packages on the validation computer first:

    python -m pip install onnx onnxruntime

The script checks file presence, SHA256 uniqueness, ONNX graph validity, and
one real ONNX Runtime inference using a zero-valued input tensor. It does not
alter any model files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any


ROI_NAMES = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dimension(value: Any, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def validate_onnx(path: Path) -> tuple[bool, str]:
    try:
        import onnx
    except ImportError:
        return False, "onnx is not installed"

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:
        return False, f"ONNX graph check failed: {exc}"

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return False, "onnxruntime is not installed"

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            return False, f"expected 1 input, found {len(inputs)}"

        input_info = inputs[0]
        shape = [
            dimension(value, fallback)
            for value, fallback in zip(input_info.shape, (1, 3, 256, 256))
        ]
        if len(shape) != 4:
            return False, f"expected a 4D image input, found shape {input_info.shape}"

        input_array = np.zeros(shape, dtype=np.float32)
        outputs = session.run(None, {input_info.name: input_array})
        if not outputs:
            return False, "inference returned no outputs"
        if not all(np.isfinite(output).all() for output in outputs if hasattr(output, "dtype")):
            return False, "inference returned NaN or infinity"

        output_names = ", ".join(output.name for output in session.get_outputs())
        return True, f"runtime OK; input {shape}; outputs [{output_names}]"
    except Exception as exc:
        return False, f"ONNX Runtime inference failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/rois"),
        help="folder containing ROI_1 through ROI_4",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser()
    print(f"Checking: {models_dir.resolve()}")
    print()

    failures = 0
    hashes: dict[str, list[str]] = {}
    model_paths: dict[str, Path] = {}

    for roi_name in ROI_NAMES:
        weights_dir = models_dir / roi_name / "weights"
        onnx_path = weights_dir / "model.onnx"
        ckpt_path = weights_dir / "best.ckpt"
        print(roi_name)

        if not onnx_path.is_file():
            print("  FAIL: model.onnx is missing")
            failures += 1
            continue

        file_hash = sha256(onnx_path)
        hashes.setdefault(file_hash, []).append(roi_name)
        model_paths[roi_name] = onnx_path
        print(f"  file: {onnx_path}")
        print(f"  size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  sha256: {file_hash}")
        print(f"  checkpoint: {'present' if ckpt_path.is_file() else 'MISSING'}")

        valid, message = validate_onnx(onnx_path)
        print(f"  {'PASS' if valid else 'FAIL'}: {message}")
        if not valid:
            failures += 1
        print()

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    if duplicate_groups:
        failures += len(duplicate_groups)
        print("DUPLICATE CONTENT:")
        for group in duplicate_groups:
            print(f"  FAIL: {', '.join(group)} have identical SHA256 hashes")
        print()

    missing = [roi_name for roi_name in ROI_NAMES if roi_name not in model_paths]
    print("FINAL VERDICT")
    if missing:
        print(f"  FAIL: missing ONNX files for {', '.join(missing)}")
    if failures:
        print("  DO NOT submit the ONNX files as verified.")
        print("  Fix the reported failures and run this script again.")
        return 1

    print("  PASS: all four ONNX files are present, different, valid, and executable.")
    print("  The ONNX files are ready to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())