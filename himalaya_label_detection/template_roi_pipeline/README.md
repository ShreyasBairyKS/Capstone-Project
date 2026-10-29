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
| `explain.py` | heatmap overlays, hot regions, counterfactual scores, plain-language reasons |
| `app/server.py`, `app/static/index.html` | upload web app (FastAPI + one HTML page, no build step) |
| `__init__.py` | public API re-exports |
| `tests/` | model-free tests: pipeline, explainability and the web app (~4 s) |
| `scripts/run_app.py` | start the web app |
| `scripts/build_roi_templates.py` | rebuild `config/roi_templates/*.png` from one clean scan |
| `scripts/run_template_inference.py` | inspect scans, evaluate, calibrate (command line) |
| `config/roi_templates/roi_N_template.png` | full-resolution ROI templates (from `NSC GOOD IMAGES/1.bmp`) |
| `config/roi_onnx_thresholds.json` | thresholds + the `input_mode` / `score_stat` they were fitted for |
| `requirements.txt` | what the pipeline and the app need |

## Web app: upload a scan, get a verdict with evidence

```bash
cd Capstone-Project          # the paths below are relative to the repo root
pip install -r himalaya_label_detection/template_roi_pipeline/requirements.txt
python himalaya_label_detection/template_roi_pipeline/scripts/run_app.py      # then open http://127.0.0.1:8000
```

Keep that terminal window open: closing it stops the server. Open the address in a browser; do not open
`app/static/index.html` directly from the disk (the page then has no server behind it, and says so).
If the page reports "Cannot reach the inspection server", the server is not running (or has been closed):
start it again and reload.

Upload a full scan (or pick one of the NSC samples). The page shows live progress (read, locate, one step per
ROI, explain), then:

* the **verdict** (PASS / FAIL) with a one-line reason, and a JSON report you can download;
* the **scan turned upright with the four ROIs boxed**: green pass, red fail, amber not located;
* one card per ROI (failures first) with the score against its limit on a gauge, the crop, and two heat views;
* a **plain-language explanation** per ROI.

Models and input mode default to whatever `config/roi_onnx_thresholds.json` was calibrated for (today: the
color-trained models in `models/rois`). The app refuses `--input-mode` values that don't match the thresholds.
Once grayscale models are exported and thresholds recalibrated:
`run_app.py --models-root models/gray_scale_rois --input-mode gray`.

The selected retrained ROI_4 model is stored as `models/gray_scale_rois/new_roi4`.
Use its dedicated grayscale profile so the pipeline loads that folder for ROI_4
and uses its calibrated threshold:

`run_app.py --config-dir himalaya_label_detection/template_roi_pipeline/config --thresholds himalaya_label_detection/template_roi_pipeline/config/roi_onnx_thresholds_gray_new_roi4.json`

Runs are queued one at a time (the models already use every core); an identical file uploaded again is served
from cache. On this CPU a scan takes about 25 s.

### How the explanation works

The ROI score is the **mean of the model's anomaly map**, and the ROI fails when it is above its limit.
So every pixel of the map either pulls the score up or not, and the explanation is built on that:

* **Above limit** view: only pixels scored above the ROI's limit are tinted, brighter = further above. The
  colour range follows the map but never shrinks below 0.25, so noise on a clean crop stays faint.
* **Hot regions** (failing ROIs only): up to 3 connected areas, strongest first, boxed and numbered. For each
  one the page states how much it **adds to the score** and what the ROI would **score without it**, replacing
  the region by the crop's median. If the score without it is under the limit, that region alone causes the
  rejection; if not, the anomaly is spread out ("not confined to one place").
* **Full map** view: the whole map min-max normalised. It always highlights a hottest spot, even on a clean
  crop, so it is for exploring, not judging.
* Boxes are drawn only on failing ROIs; a passing ROI shows its gauge and the one-line summary.

Example (NSC bad scan 2): ROI_2 scores 0.2464 against a limit of 0.2282; region 1 (centre, 2.5% of the area)
adds 0.1874 to that, and without it ROI_2 would score 0.0590, so that region causes the rejection.

Heatmaps show where the model reacted; they do not prove a defect is physically there. They are the place
to look first.

## Usage (command line)

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

* Tested: localisation on all 120 NSC scans (all 4 ROIs found every time), the unit tests (30), the CLI
  (inspect and `--calibrate`) and the Python API on real scans in color mode, and the web app in a browser on
  a good scan (PASS) and a bad scan (FAIL, ROI_2) plus unreadable / wrong-size / missing-ROI inputs, desktop,
  phone width, light and dark.
* Not tested: gray mode with grayscale-trained models (not available yet), the full 120 scans through the
  classifiers, the heatmaps on ROI_1 / ROI_3 failures (the scans examined had defects only in ROI_2), and
  any held-out data.

## Performance

On a 16-core CPU the ONNX models take ~7 s per crop (~25 s per scan); they were trained on a GPU.
`ONNXROIClassifier(..., intra_op_threads=N)` caps threads per session.
