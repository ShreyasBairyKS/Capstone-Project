# VisionFood QAI Bottle Cap Process Knowledge Base

This document is the complete practical reference for your current bottle-cap defect pipeline.

It explains:
- End-to-end workflow from data to deployment checks
- What each script does in this repository
- How to run and validate the model (using `.venv` only)
- How confidence score and operating threshold are selected
- How to decide when the model is "finished" for a milestone

---

## 1) What This Model Does

The model performs object detection for bottle cap quality classes:
- `defectCap`
- `goodCap`
- `noCap`

For pass/fail business logic, defect classes are:
- `defectCap`
- `noCap`

Non-defect class:
- `goodCap`

So image-level defective decision is:
- defective if any detection belongs to `{defectCap, noCap}` above confidence threshold

---

## 2) Repository Scripts You Actually Use

Current active scripts in this repo for the bottle-cap pipeline:

- Training:
  - `training/bottle-cap-train.py`
- Evaluation:
  - `Evaluate/bottle-cap-evaluate.py`
- Inference:
  - `inference/bottle-cap-infer.py`
- Confidence threshold sweep:
  - `threshold_sweep.py`
- Color diversity augmentation:
  - `color_diversity_aug.py`

Note:
- `train.py` and `evaluate.py` (root-level names) are not the active entrypoints in this workspace.

---

## 3) Current Dataset Layout (Augmented)

Current training config points to:
- `data_augmented/data.yaml`

Expected layout:
- `data_augmented/images/train`
- `data_augmented/images/val`
- `data_augmented/images/test`
- `data_augmented/labels/train`
- `data_augmented/labels/val`
- `data_augmented/labels/test`

Current class order (important for IDs):
- class `0` = `defectCap`
- class `1` = `goodCap`
- class `2` = `noCap`

---

## 4) End-to-End Process (Operational)

### Step A: Prepare/augment train data

Use color augmentation to improve robustness across bottle/cap color variants.

### Step B: Train from pretrained YOLO weights

Use `yolo11s` for good quality-speed tradeoff.

### Step C: Evaluate at image-level and bbox-level

Use `Evaluate/bottle-cap-evaluate.py` for:
- YOLO metrics: precision, recall, mAP50, mAP50-95
- Binary image-level metrics: TP, FP, TN, FN, F1, accuracy
- Speed: avg latency, FPS

### Step D: Sweep confidence threshold

Use `threshold_sweep.py` to test operating points over confidence range and choose deployment threshold.

### Step E: Run inference smoke/acceptance tests

Use `inference/bottle-cap-infer.py` on:
- held-out test set
- then real unseen production-like images

---

## 5) .venv-Only Execution Standard

Run all commands with:
- `.\.venv\Scripts\python.exe`

This guarantees consistent packages, CUDA compatibility, and reproducible outcomes.

---

## 6) Commands You Need Most

### 6.1 Train

```powershell
Set-Location "E:\P-25 Vision Food ai"
.\.venv\Scripts\python.exe training/bottle-cap-train.py \
  --data ./data_augmented/data.yaml \
  --model yolo11s \
  --epochs 100 \
  --batch 16 \
  --lr0 0.001 \
  --cls 1.5 \
  --run-name bottle_cap_defect_coloraug_scratch \
  --device 0
```

### 6.2 Evaluate

```powershell
Set-Location "E:\P-25 Vision Food ai"
.\.venv\Scripts\python.exe Evaluate/bottle-cap-evaluate.py \
  --weights runs/detect/bottle_cap_defect_coloraug_scratch/weights/best.pt \
  --data data_augmented/data.yaml \
  --conf 0.40 \
  --iou 0.45 \
  --device cuda:0 \
  --defect-classes defectCap,noCap
```

### 6.3 Threshold Sweep

```powershell
Set-Location "E:\P-25 Vision Food ai"
.\.venv\Scripts\python.exe threshold_sweep.py \
  --weights runs/detect/bottle_cap_defect_coloraug_scratch/weights/best.pt \
  --test-images data_augmented/images/test \
  --test-labels data_augmented/labels/test \
  --defect-classes defectCap,noCap \
  --device cuda:0 \
  --out runs/detect/threshold_sweep_coloraug_latest.json
```

### 6.4 Inference Smoke Test

```powershell
Set-Location "E:\P-25 Vision Food ai"
.\.venv\Scripts\python.exe inference/bottle-cap-infer.py \
  --weights runs/detect/bottle_cap_defect_coloraug_scratch/weights/best.pt \
  --source data_augmented/images/test \
  --conf 0.40 \
  --iou 0.45 \
  --device cuda:0 \
  --defect-classes defectCap,noCap \
  --save \
  --output runs/infer/coloraug_check
```

---

## 7) How Confidence Score Works (Core Knowledge)

Each predicted box has a confidence score in `[0, 1]`.

Interpretation:
- higher confidence = model is more certain that detection/class is correct
- lower confidence = model is less certain

At inference, you set a threshold `conf`:
- keep detections with confidence `>= conf`
- drop detections with confidence `< conf`

This threshold changes business outcomes:
- Increasing `conf` usually reduces false positives, but can increase false negatives.
- Decreasing `conf` usually increases recall, but can increase false positives.

Mathematically at image-level binary decision:
- precision $= \frac{TP}{TP+FP}$
- recall $= \frac{TP}{TP+FN}$
- F1 $= \frac{2 \cdot precision \cdot recall}{precision+recall}$
- accuracy $= \frac{TP+TN}{TP+TN+FP+FN}$

So threshold selection is a controlled tradeoff problem, not a fixed universal value.

---

## 8) How Threshold Is Selected in This Project

Selection workflow:
1. Sweep confidence from `0.15` to `0.80` with `threshold_sweep.py`.
2. Compute TP/FP/TN/FN and precision/recall/F1/accuracy at each threshold.
3. Choose threshold based on business objective.

Current objective update (your latest):
- FN can be tolerated for now
- prioritize strong overall F1 and accuracy

From current sweep file:
- `runs/detect/threshold_sweep_coloraug_latest.json`

Observed best plateau:
- `conf = 0.15` to `0.40`
- TP=33, FP=0, TN=22, FN=1
- F1 ≈ 0.9851
- Accuracy ≈ 0.9821

Compared to `conf = 0.45`:
- TP=31, FP=0, TN=22, FN=3
- F1 ≈ 0.9538
- Accuracy ≈ 0.9464

Practical deployment choice right now:
- Use `conf = 0.40`

Reason:
- same best F1/accuracy as lower plateau values
- keeps stricter filtering than 0.15/0.20 while preserving top metrics on this test set

---

## 9) Current Model Status Summary

From latest eval report:
- `runs/detect/bottle_cap_eval/eval_report.json`

At `conf = 0.45`:
- YOLO precision: `0.9855`
- YOLO recall: `0.9015`
- mAP50: `0.9471`
- mAP50-95: `0.6738`
- Binary TP/FP/TN/FN: `31/0/22/3`
- Binary F1: `0.9538`
- Binary accuracy: `0.9464`
- Speed: `36.3 ms`, `27.5 FPS`

Implication:
- model quality is strong
- threshold choice matters significantly for final binary metrics

---

## 10) Is It Finished?

For your updated milestone (good accuracy + good F1, and FN tolerated):
- Yes, this is good for pilot usage.

For strict production hard constraints with FN close to zero on real line data:
- still do one more validation cycle on truly unseen production captures.

---

## 11) Acceptance Checklist Before Real Deployment

1. Run inference smoke test on held-out test set.
2. Run evaluation and confirm accuracy/F1 targets.
3. Run threshold sweep and pick fixed operating threshold.
4. Test on fresh production-line images (lighting, motion blur, camera angle shifts).
5. Confirm latency/FPS on the intended hardware.
6. Freeze config (weights path, conf threshold, defect class set).

---

## 12) Recommended Frozen Config (Current)

- Weights:
  - `runs/detect/bottle_cap_defect_coloraug_scratch/weights/best.pt`
- Defect classes:
  - `defectCap,noCap`
- Confidence threshold:
  - `0.40`
- IoU:
  - `0.45`
- Runtime Python:
  - `.\.venv\Scripts\python.exe`

---

## 13) Troubleshooting Quick Notes

- If `cv2` import fails:
  - install in `.venv`: `python -m pip install opencv-python==4.10.0.84`
- If Ultralytics resolves dataset to wrong base path:
  - keep absolute `path:` in YAML (already done in `data_augmented/data.yaml`)
- If CUDA not detected in `.venv`:
  - verify torch CUDA build in `.venv`

---

## 14) What To Re-Run For Reporting

After any retrain, always produce these:
1. `runs/detect/bottle_cap_eval/eval_report.json`
2. `runs/detect/threshold_sweep_*.json`
3. `runs/infer/*/inference_results.json`

These three files together give complete quality + threshold + behavior evidence.
