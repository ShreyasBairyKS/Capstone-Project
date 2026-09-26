"""Export and verify the four grayscale EfficientAD checkpoints as ONNX models.

The script expects checkpoints such as:

    models/gray_scale_rois/ROI_1/weights/last.ckpt
    models/gray_scale_rois/ROI_2/weights/last.ckpt
    models/gray_scale_rois/ROI_3/weights/last.ckpt
    models/gray_scale_rois/ROI_4/weights/last.ckpt

It exports each checkpoint to the matching ``weights/model.onnx`` file. The
ONNX interface is deliberately fixed for LabVIEW:

    input:  image       float32 [1, 3, 256, 256], RGB, pixels in [0, 1]
    output: anomaly_map float32 [1, 1, 256, 256]

The model receives three identical grayscale RGB channels. Do not add external
ImageNet normalization in LabVIEW: the checkpoint model's own preprocessing is
part of the exported computation.

Run this on the PC that has the original training environment and the four
``.ckpt`` files. It does not retrain any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROI_NAMES = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")
INPUT_SHAPE = (1, 3, 256, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(weights_dir: Path) -> Path | None:
    """Prefer the best checkpoint, then last, then the newest checkpoint."""
    for name in ("best.ckpt", "last.ckpt"):
        candidate = weights_dir / name
        if candidate.is_file():
            return candidate
    candidates = sorted(weights_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def extract_anomaly_map(output: Any, torch_module: Any) -> Any:
    """Extract the spatial map without reducing it to an image score."""
    if isinstance(output, dict) and "anomaly_map" in output:
        anomaly_map = output["anomaly_map"]
    elif hasattr(output, "anomaly_map"):
        anomaly_map = output.anomaly_map
    elif isinstance(output, torch_module.Tensor):
        anomaly_map = output
    else:
        keys = list(output.keys()) if isinstance(output, dict) else None
        raise RuntimeError(f"EfficientAD output has no anomaly_map (keys: {keys})")

    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    if anomaly_map.ndim != 4:
        raise RuntimeError(f"Expected a 4D anomaly map, received shape {tuple(anomaly_map.shape)}")
    return anomaly_map


def model_forward(model: Any, image: Any, torch_module: Any) -> Any:
    """Support the direct and dictionary forward forms used by anomalib releases."""
    try:
        return extract_anomaly_map(model(image), torch_module)
    except Exception as direct_error:
        try:
            return extract_anomaly_map(model({"image": image}), torch_module)
        except Exception as dict_error:
            raise RuntimeError(
                "Could not run checkpoint using either model(image) or model({'image': image}). "
                f"Direct error: {direct_error}; dictionary error: {dict_error}"
            ) from dict_error


class AnomalyMapExportWrapper:  # instantiated dynamically after torch is imported
    """Placeholder used only to keep the torch-dependent class below readable."""


@dataclass
class ExportResult:
    roi: str
    checkpoint: str
    onnx_path: str
    onnx_sha256: str
    pytorch_map_shape: list[int]
    onnx_map_shape: list[int]
    max_abs_difference: float
    mean_abs_difference: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=project_root / "models" / "gray_scale_rois",
        help="folder containing ROI_N/weights/*.ckpt and receiving model.onnx",
    )
    parser.add_argument(
        "--roi",
        choices=ROI_NAMES,
        action="append",
        help="export only this ROI; repeat the option to export more than one",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing weights/model.onnx")
    return parser.parse_args()


def import_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"Missing export dependency: {exc}. Install torch, onnx, onnxruntime, and the same anomalib environment "
            "used for training."
        ) from exc

    try:
        from anomalib.models import EfficientAd
    except ImportError:
        try:
            from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
        except ImportError as exc:
            raise SystemExit(
                "Cannot import anomalib EfficientAd. Run this on the training PC with the original anomalib environment."
            ) from exc
    return np, onnx, ort, torch, EfficientAd


def save_metadata(roi_dir: Path, result: ExportResult, opset: int) -> None:
    metadata_path = roi_dir / "model_meta.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot update invalid metadata JSON: {metadata_path} ({exc})") from exc

    metadata["onnx_export"] = {
        "checkpoint": result.checkpoint,
        "onnx_path": result.onnx_path,
        "onnx_sha256": result.onnx_sha256,
        "opset": opset,
        "input": {"name": "image", "dtype": "float32", "shape": list(INPUT_SHAPE), "layout": "NCHW"},
        "output": {"name": "anomaly_map", "dtype": "float32", "shape": result.onnx_map_shape, "layout": "NCHW"},
        "verification": {
            "pytorch_map_shape": result.pytorch_map_shape,
            "max_abs_difference": result.max_abs_difference,
            "mean_abs_difference": result.mean_abs_difference,
        },
        "exported_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.opset < 11:
        raise SystemExit("Use an ONNX opset of 11 or newer; 17 is required for the existing LabVIEW contract.")

    np, onnx, ort, torch, EfficientAd = import_dependencies()
    selected_rois = tuple(args.roi) if args.roi else ROI_NAMES
    models_root = args.models_root.resolve()
    if not models_root.is_dir():
        raise SystemExit(f"Models root does not exist: {models_root}")

    missing: list[str] = []
    export_jobs: list[tuple[str, Path, Path, Path]] = []
    for roi_name in selected_rois:
        roi_dir = models_root / roi_name
        weights_dir = roi_dir / "weights"
        checkpoint_path = find_checkpoint(weights_dir)
        output_path = weights_dir / "model.onnx"
        if checkpoint_path is None:
            missing.append(f"{roi_name}: no .ckpt found under {weights_dir}")
        elif output_path.exists() and not args.overwrite:
            missing.append(f"{roi_name}: {output_path} already exists (use --overwrite to replace it)")
        else:
            export_jobs.append((roi_name, roi_dir, checkpoint_path, output_path))
    if missing:
        print("No models were exported:")
        for message in missing:
            print(f"  - {message}")
        return 2

    class ExportWrapper(torch.nn.Module):
        def __init__(self, checkpoint_model: Any) -> None:
            super().__init__()
            self.checkpoint_model = checkpoint_model

        def forward(self, image: Any) -> Any:
            return model_forward(self.checkpoint_model, image, torch)

    results: list[ExportResult] = []
    for roi_name, roi_dir, checkpoint_path, output_path in export_jobs:
        print(f"\n[{roi_name}] Loading {checkpoint_path.name}")
        try:
            model = EfficientAd.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"[{roi_name}] Could not load checkpoint. Use the same anomalib version used for training. {exc}"
            ) from exc
        model.eval().cpu()
        wrapper = ExportWrapper(model).eval().cpu()
        example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
        with torch.no_grad():
            pytorch_map = wrapper(example).detach().cpu().numpy().astype(np.float32, copy=False)
        if tuple(pytorch_map.shape) != (1, 1, 256, 256):
            raise RuntimeError(f"[{roi_name}] Expected PyTorch output [1,1,256,256], got {list(pytorch_map.shape)}")

        temporary_path = output_path.with_suffix(".partial.onnx")
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"[{roi_name}] Exporting ONNX opset {args.opset}")
        try:
            torch.onnx.export(
                wrapper,
                example,
                str(temporary_path),
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["anomaly_map"],
                dynamic_axes=None,
            )
            onnx.checker.check_model(onnx.load(str(temporary_path)))
            session = ort.InferenceSession(str(temporary_path), providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            if input_info.name != "image" or output_info.name != "anomaly_map":
                raise RuntimeError(f"Unexpected ONNX names: input={input_info.name}, output={output_info.name}")
            onnx_map = session.run(["anomaly_map"], {"image": example.numpy()})[0].astype(np.float32, copy=False)
            if tuple(onnx_map.shape) != (1, 1, 256, 256):
                raise RuntimeError(f"Unexpected ONNX output shape: {list(onnx_map.shape)}")
            difference = np.abs(pytorch_map - onnx_map)
            max_abs_difference = float(difference.max())
            mean_abs_difference = float(difference.mean())
            if not np.allclose(pytorch_map, onnx_map, rtol=1e-4, atol=1e-4):
                raise RuntimeError(
                    f"ONNX output differs from PyTorch beyond tolerance: max={max_abs_difference}, mean={mean_abs_difference}"
                )
            if output_path.exists():
                output_path.unlink()
            temporary_path.replace(output_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        result = ExportResult(
            roi=roi_name,
            checkpoint=str(checkpoint_path.relative_to(models_root)),
            onnx_path=str(output_path.relative_to(models_root)),
            onnx_sha256=sha256(output_path),
            pytorch_map_shape=list(pytorch_map.shape),
            onnx_map_shape=list(onnx_map.shape),
            max_abs_difference=max_abs_difference,
            mean_abs_difference=mean_abs_difference,
        )
        save_metadata(roi_dir, result, args.opset)
        results.append(result)
        print(
            f"[{roi_name}] PASS: {output_path} | SHA-256 {result.onnx_sha256} | "
            f"max map delta {result.max_abs_difference:.8f}"
        )

    manifest_path = models_root / "grayscale_onnx_export_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_type": "EfficientAD grayscale ROI checkpoints",
                "input": {"name": "image", "dtype": "float32", "shape": list(INPUT_SHAPE), "layout": "NCHW"},
                "output": {"name": "anomaly_map", "dtype": "float32", "shape": [1, 1, 256, 256], "layout": "NCHW"},
                "opset": args.opset,
                "exports": [asdict(result) for result in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nPASS: exported and verified {len(results)} grayscale ONNX model(s).")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no completed ONNX model was replaced.", file=sys.stderr)
        raise SystemExit(130)
