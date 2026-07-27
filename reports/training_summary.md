# Training Summary

## YOLO ROI detector
- Run directory: runs/detect/water_surface_v1
- Weights: runs/detect/water_surface_v1/weights/last.pt
- Training epochs: 110
- Batch size: 16
- Image size: 1280
- Notes: results were logged in runs/detect/water_surface_v1/results.csv

## ROI anomaly classifiers
- Checkpoints saved under models/rois/ROI_1..ROI_4/weights/
- Thresholds saved in himalaya_label_detection/config/roi_thresholds.json
- Calibrated thresholds:
  - ROI_1: 0.082487
  - ROI_2: 0.086158
  - ROI_3: 0.05
  - ROI_4: 0.113014
- Training script: himalaya_label_detection/scripts/train_classifiers.py
