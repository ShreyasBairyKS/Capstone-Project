# Bottle Cap Guide (RTX A5000 + 128GB RAM)

## Scope
This guide is for Collaborator A bottle-cap work only, using:
- `training/bottle-cap-train.py`
- `Evaluate/bottle-cap-evaluate.py`
- `inference/bottle-cap-infer.py`

It documents a hardware-optimized workflow for a workstation with NVIDIA RTX A5000 and 128GB system RAM.

## Dataset Assumption
This guide assumes the Roboflow dataset at:
- `dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml`

Class names are expected to be:
- `defectCap`
- `goodCap`
- `noCap`

Binary cap-fail defaults are:
- defective: `defectCap,noCap`
- non-defective: `goodCap`

## Hardware-Aware Training Profiles
`training/bottle-cap-train.py` supports profile-based runtime tuning through `--profile`.

Profiles:
- `auto`: selects `a5000-balanced` when A5000 is detected, otherwise baseline
- `baseline`: conservative defaults for general environments
- `a5000-fast`: maximize throughput and iteration speed
- `a5000-balanced`: best default for quality/speed tradeoff
- `a5000-quality`: higher-resolution, quality-focused training

A5000 profile behavior includes tuned values for:
- image size (`imgsz`)
- batch size (model-dependent)
- workers
- RAM caching
- AMP mixed precision
- scheduler/augmentation settings (`cos_lr`, `multi_scale`, `close_mosaic`)

Approximate A5000 default batches by model:

| Profile | yolo11n | yolo11s | yolo11m | yolo11l | yolo11x |
|---|---:|---:|---:|---:|---:|
| a5000-fast | 96 | 64 | 40 | 24 | 16 |
| a5000-balanced | 72 | 48 | 28 | 16 | 10 |
| a5000-quality | 48 | 32 | 20 | 12 | 8 |

You can override any profile choice with explicit flags like `--batch`, `--imgsz`, `--workers`, `--cache`, `--amp`, `--no-amp`.

## Recommended Commands

### 1) Balanced training (recommended)
```bash
python training/bottle-cap-train.py \
  --data dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml \
  --model yolo11m \
  --epochs 120 \
  --profile a5000-balanced \
  --device 0 \
  --run-name bottle_cap_defect \
  --product-category beverage \
  --product-sub-type transparent_bottle \
  --export
```

### 2) Faster iteration training
```bash
python training/bottle-cap-train.py \
  --data dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml \
  --model yolo11s \
  --epochs 80 \
  --profile a5000-fast \
  --device 0 \
  --run-name bottle_cap_defect_fast
```

### 3) Quality-first training
```bash
python training/bottle-cap-train.py \
  --data dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml \
  --model yolo11l \
  --epochs 140 \
  --profile a5000-quality \
  --device 0 \
  --run-name bottle_cap_defect_quality \
  --export
```

## Evaluation
```bash
python Evaluate/bottle-cap-evaluate.py \
  --weights runs/detect/bottle_cap_defect/weights/best.pt \
  --data dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml \
  --conf 0.25 \
  --iou 0.45 \
  --defect-classes defectCap,noCap \
  --product-category beverage \
  --product-sub-type transparent_bottle
```

Outputs:
- plots and metrics under `runs/detect/bottle_cap_eval/`
- report JSON at `runs/detect/bottle_cap_eval/eval_report.json`

## Inference
```bash
python inference/bottle-cap-infer.py \
  --weights runs/detect/bottle_cap_defect/weights/best.pt \
  --source dataset/Beverages/bottleDefect.v1-first.yolov11-cap/test/images \
  --conf 0.25 \
  --iou 0.45 \
  --defect-classes defectCap,noCap \
  --product-category beverage \
  --product-sub-type transparent_bottle \
  --save
```

Outputs:
- JSON: `runs/infer/inference_results.json`
- optional annotated images: `runs/infer/predictions/`

## Troubleshooting

### CUDA out-of-memory
Try one or more:
- lower `--batch`
- use `--model yolo11s` or `--model yolo11m`
- switch to `--profile a5000-fast`
- lower `--imgsz` (for example `--imgsz 640`)

### Disk or RAM pressure while caching
- set `--cache disk` or `--cache none`
- lower worker count with `--workers`

### Validation looks unstable across runs
- use `--profile a5000-quality`
- keep `--epochs` high enough (100+)
- tune confidence/IoU in evaluation (`--conf`, `--iou`)

## Artifacts to Track
Each training run writes:
- weights: `runs/detect/<run-name>/weights/`
- training plots: `runs/detect/<run-name>/`
- context: `runs/detect/<run-name>/training_context.json`

Keep the `training_context.json` and evaluation report together for reproducibility and collaborator handoff.
