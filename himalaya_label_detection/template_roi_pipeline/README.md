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

Upload a full scan (drop it anywhere on the page, or pick one of the NSC samples). The page shows live progress
(read, locate, one step per ROI with its own timer, a time-left estimate, explain), then:

* the **verdict** (PASS / FAIL) with the reasons, and a JSON report you can download. For the NSC samples it also
  says whether the verdict agrees with the folder the scan came from (good / bad), which makes a false alarm or a
  missed defect obvious;
* four **summary tiles** (one per ROI: score, limit, margin). Hovering a tile or a box on the overview highlights
  the other, and a click jumps to that ROI's check;
* the **scan turned upright with the four ROIs boxed**: green pass, red fail, amber not located;
* one card per ROI (failures first) with the score against its limit on a meter, the model that judged it, and
  four views of the crop: **Original**, **Above limit**, **Full map** and **Compare** (drag a handle between the
  original and the heat view). Hovering a crop opens a magnifier; clicking opens it full screen, where the arrow
  keys switch view and a click toggles actual size;
* a **plain-language explanation** per ROI, and for failures the hot regions with what each adds to the score;
* a "Models, limits and method" panel with every ROI's model, score, limit, margin and position.

Each result has its own address (`#/scan/<id>`), so a reload or the browser's back button returns to it while the
server is running; recent results are listed on the start page. The theme follows the system and can be switched
with the button in the top bar. The page has no external dependencies (no CDN, fonts or build step).

Models, input mode and limits come from a **profile**: a thresholds file in `config/`. The default,
`config/roi_onnx_thresholds.json`, is the grayscale set in `models/gray_scale_rois`: ROI_1 to ROI_3 are the models
from the LabVIEW handoff, **ROI_4 is the new model `new_roi4`**. Each ROI's card shows which model judged it.
A profile can point single ROIs at another model folder:

```json
"model_dirs": { "ROI_4": "new_roi4" }     // folder under "models_root" (default: the ROI's own name, e.g. ROI_4)
```

Other profiles in `config/`, selected with `run_app.py --thresholds <file>`:

| File | Models |
|------|--------|
| `roi_onnx_thresholds.json` (default) | grayscale: ROI_1-3 from the LabVIEW handoff, ROI_4 `new_roi4` |
| `roi_onnx_thresholds_color.json` | the previous set: the color-trained models in `models/rois` |
| `roi_onnx_thresholds_gray_new_roi4.json` | same models as the default (as committed with `new_roi4`) |

The app refuses `--input-mode` values that don't match the profile.

Runs are queued one at a time (the models already use every core); an identical file uploaded again is served
from cache. On this CPU a scan takes about 35 s (more while other jobs use it). The crops and heat views are sent
up to 1600 px wide (`explain.DISPLAY_WIDTH`), close to the widest crop's native size, so the magnifier shows real
detail.

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
* **Compare** view: the original crop on the left of a draggable handle, the Above-limit view on the right.
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
* The **grayscale-trained** set is in `models/gray_scale_rois` (`ROI_1..3` from the LabVIEW handoff, `new_roi4`
  for ROI_4); the default profile runs it with `--input-mode gray`. Thresholds are mode-specific; the CLI refuses
  to use thresholds fitted for a different mode/statistic. `--calibrate` keeps a profile's `model_dirs`, so it
  fits the limits for the models that profile actually uses.
* The crops must be upright (rotated 90 degrees clockwise from the raw scan, as `inspector.py` does). On good scan 1
  the grayscale set scores the raw, sideways crops +0.6 to +2.2 and the upright ones -0.04 to +0.17.

## Current status of the committed thresholds

`config/roi_onnx_thresholds.json` is **provisional**: grayscale set, `mean` score, limits fitted on the crops this
pipeline cuts from 40 NSC scans (26 good, 14 bad, interleaved), all of which were in the models' training data.
Bad scans are only labelled at image level, so the fit optimises the aggregated image verdict
(`fit_thresholds` in `scripts/run_template_inference.py`). In-sample: 14/14 bad caught, 0/26 good rejected. A
split-half check (fit on half, test on the other half) is much weaker: 8 of 14 bad caught, 1 of 26 good rejected.

The limits that came with the models (the LabVIEW handoff's for ROI_1-3, the trainers' for ROI_4; kept in the file
as `trainer_thresholds_not_used`) reject every good scan tested through this pipeline (26/26): the models score
these crops higher than their calibration range. The handoff's own sample crops, scored upright, also land above
them. So the supplied limits and the models do not match under this preprocessing; ask whoever calibrated them
which checkpoints and crops were used.

ROI_4 (`new_roi4`) shows no separation on these 40 scans: good 0.008-0.079 (median 0.033), bad 0.010-0.073
(median 0.032), so its fitted limit sits above every bad score and ROI_4 never rejects a label. ROI_1-3 carry the
verdict (of the 14 bad scans: ROI_1 flags 8, ROI_3 flags 8, ROI_2 flags 3). Only ~26 of the 42 bad scans have a
defect in ROI_4, which dilutes image-level labels but does not explain a bad median equal to the good median.
Re-calibrate on the full set and validate on held-out scans before relying on any of this.

## What has and has not been tested

* Tested: localisation on all 120 NSC scans (all 4 ROIs found every time), the unit tests (35), the grayscale set
  (handoff ROI_1-3 + `new_roi4`) on 40 NSC scans through the pipeline's own crops, and the web app in a browser
  on those models. Earlier, in color mode: the CLI (inspect and `--calibrate`), the Python API, and the web app on
  a good and a bad scan plus unreadable / wrong-size / missing-ROI inputs, desktop, phone width, light and dark.
* Not tested: the full 120 scans through the grayscale classifiers (40 were scored), any held-out data, the CLI
  `--calibrate` with the grayscale profile, and the handoff's own checks: its `checksums.sha256`, README provenance
  note and test vectors describe the earlier color models and fail against the models now in that folder (the
  manifest and each `model_meta.json` do match the current files).

## Performance

On a 16-core CPU the ONNX models take ~6-9 s per crop (~35 s per scan); they were trained on a GPU.
`ONNXROIClassifier(..., intra_op_threads=N)` caps threads per session.
