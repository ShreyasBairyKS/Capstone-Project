# Template-matching ROI pipeline

```
full scan (1504 x 8000, BMP)
   -> grayscale
   -> ROITemplateMatcher   (roi_template_matcher.py)
        4 full-ROI templates, coarse-to-fine matchTemplate, best occurrence per ROI
   -> rotate each crop 90 deg clockwise   (classifiers were trained on upright crops, see scripts/crop_roi.py)
   -> per-ROI ONNX EfficientAD classifier  (inspector.py)
   -> ROI PASS/FAIL (score > threshold)  -> label PASS/FAIL
```

Final verdict: **FAIL** if any ROI score exceeds its threshold, or any ROI could not be located
(match score < 0.45; `--allow-missing` disables the second rule). Otherwise PASS.

Overlap rule: ROIs never overlap on the label, but a low-contrast template can score ~0.57 on a neighbouring ROI's
area (ROI_4's template on ROI_3's text panel), above the 0.45 cutoff. If two found ROIs overlap by more than 10%,
the lower-scoring one is re-searched with the other's area excluded, so an absent/damaged ROI becomes MISSING
instead of silently landing on a neighbour. On the 120 NSC scans this rule never fires (0 positions change).

## Layout

This folder is self-contained and does not touch the existing `src/`, `scripts/` or `config/` of
`himalaya_label_detection`. It is a regular Python package (`template_roi_pipeline`); put
`himalaya_label_detection/` on `sys.path` (the scripts and tests do this themselves).

| File | Role |
|------|------|
| `roi_template_matcher.py` | localisation (~120 ms / scan) |
| `inspector.py` | ONNX classifiers + aggregation (`TemplateROIInspector`) |
| `__init__.py` | public API re-exports |
| `tests/test_template_roi_pipeline.py` | model-free tests (~2 s) |
| `scripts/build_roi_templates.py` | rebuild `config/roi_templates/*.png` from one clean scan |
| `scripts/run_template_inference.py` | inspect scans, evaluate, calibrate |
| `config/roi_templates/roi_N_template.png` | full-resolution ROI templates (from `NSC GOOD IMAGES/1.bmp`) |
| `config/roi_onnx_thresholds.json` | thresholds + the `input_mode` / `score_stat` they were fitted for |

## Usage

Needs `opencv-python`, `numpy`, `onnxruntime`. Run from the repo root (`Capstone-Project/`):

```bash
# inspect scans
python himalaya_label_detection/template_roi_pipeline/scripts/run_template_inference.py --input-mode color \
    --images "path/to/scan.bmp" --save-annotated results/annotated

# labelled evaluation / threshold calibration (writes template_roi_pipeline/config/roi_onnx_thresholds.json)
python himalaya_label_detection/template_roi_pipeline/scripts/run_template_inference.py --input-mode color \
    --good "path/to/GOOD" --bad "path/to/BAD" --calibrate

# unit tests (from himalaya_label_detection/)
python -m pytest template_roi_pipeline/tests -q
```

From Python (with `himalaya_label_detection/` on `sys.path`):

```python
import cv2
from template_roi_pipeline import TemplateROIInspector

inspector = TemplateROIInspector.from_config(
    "himalaya_label_detection/template_roi_pipeline/config",
    models_root="models/rois",        # models/gray_scale_rois once the grayscale models are exported
    input_mode="color",               # "gray" (default) with the grayscale models
)
result = inspector.inspect(cv2.imread("scan.bmp"))     # cv2.IMREAD_GRAYSCALE in gray mode
print(result.verdict, result.fail_reasons)
```

## Grayscale vs color  (read this)

The classifiers must be fed the kind of image they were trained on.

* `models/rois/*/weights/model.onnx` (the ONNX files committed in this repo) are **color-trained**.
  With grayscale input they flag every crop (mean anomaly map ~0.5 on a clean crop vs ~0 in color).
  Use `--input-mode color` with them.
* The **grayscale-trained** models are not committed. Export them with
  `scripts/export_grayscale_roi_onnx.py`, then run with `--input-mode gray` (the default) and
  `--models-root models/gray_scale_rois`, and re-run `--calibrate` (thresholds are mode-specific; the
  CLI refuses to use thresholds fitted for a different mode/statistic).

## Current status of the committed thresholds

`config/roi_onnx_thresholds.json` is **provisional**: color mode, `mean` score, fitted on 16 good + 16 bad
scans (all of which were in the models' training data). Bad scans are only labelled at image level, so the
fit optimises the aggregated image verdict. A split-half check inside that subset gave 15/16 bad caught and
3/16 good flagged. Re-calibrate on the full set and validate on held-out scans before relying on it.

ROI_2 and ROI_4 show weak separation in this data (only 14 / 26 of the 42 bad scans have a defect in
ROI_2 / ROI_4, so image-level labels dilute the signal).

## What has and has not been tested

* Tested: localisation on all 120 NSC scans (all 4 ROIs found every time), the unit tests, the CLI
  (inspect and `--calibrate`) and the Python API on real scans in color mode.
* Not tested: gray mode with grayscale-trained models (not available yet), the full 120 scans through the
  classifiers, and any held-out data.

## Performance

On a 16-core CPU the ONNX models take ~7 s per crop (~25 s per scan); they were trained on a GPU.
`ONNXROIClassifier(..., intra_op_threads=N)` caps threads per session.
