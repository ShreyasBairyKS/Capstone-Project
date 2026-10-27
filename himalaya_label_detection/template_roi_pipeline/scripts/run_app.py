r"""
run_app.py
-----------------------------------------------------------------------------
Start the upload-and-inspect web app:

    python himalaya_label_detection/template_roi_pipeline/scripts/run_app.py
    -> open http://127.0.0.1:8000

Models and input mode default to what config/roi_onnx_thresholds.json was calibrated for
(currently the color-trained models in models/rois). Once the grayscale models are exported and
thresholds re-calibrated, point it at them:

    python .../run_app.py --models-root models/gray_scale_rois --input-mode gray

--good / --bad add the NSC sample scans to a "try a sample" picker (found automatically when the
NSC folder sits next to the repo).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # himalaya_label_detection/template_roi_pipeline
REPO_ROOT = PKG_ROOT.parents[1]                         # Capstone-Project (holds models/)
sys.path.insert(0, str(PKG_ROOT.parent))

from template_roi_pipeline.inspector import TemplateROIInspector  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", type=Path, default=PKG_ROOT / "config")
    p.add_argument("--thresholds", type=Path, help="default: <config-dir>/roi_onnx_thresholds.json")
    p.add_argument("--models-root", type=Path, help="default: the models_root recorded in the thresholds file")
    p.add_argument("--input-mode", choices=("gray", "color"),
                   help="default: the input_mode the thresholds were fitted for")
    p.add_argument("--good", type=Path, default=REPO_ROOT.parent / "NSC" / "NSC GOOD IMAGES")
    p.add_argument("--bad", type=Path, default=REPO_ROOT.parent / "NSC" / "NSC BAD IMAGES")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def collect_samples(label: str, folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    files = sorted((f for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda f: (len(f.stem), f.stem))
    return {f"{label}/{f.name}": f for f in files}


def main() -> int:
    args = parse_args()
    thr_path = args.thresholds or args.config_dir / "roi_onnx_thresholds.json"
    if not thr_path.is_file():
        raise SystemExit(f"{thr_path} not found. Calibrate with scripts/run_template_inference.py --calibrate.")
    doc = json.loads(thr_path.read_text(encoding="utf-8"))

    fitted = doc.get("input_mode", "gray")
    mode = args.input_mode or fitted
    if mode != fitted:
        raise SystemExit(f"{thr_path.name} was calibrated for input_mode='{fitted}', not '{mode}'. "
                         "Thresholds are mode-specific: re-run run_template_inference.py --calibrate in this mode.")
    models_root = args.models_root or REPO_ROOT / doc.get("models_root", "models/rois")
    stat = doc.get("score_stat", "mean")

    print(f"Loading models from {models_root}  (input_mode={mode}, score_stat={stat}) ...")
    inspector = TemplateROIInspector.from_config(
        args.config_dir, models_root, thresholds_file=thr_path, input_mode=mode, score_stat=stat)

    import uvicorn
    from template_roi_pipeline.app.server import create_app

    samples = {**collect_samples("good", args.good), **collect_samples("bad", args.bad)}
    app = create_app(inspector, doc, str(models_root), samples)
    print(f"{len(samples)} sample scans available. Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
