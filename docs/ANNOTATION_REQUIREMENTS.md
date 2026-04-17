# Annotation Requirements — VisionFood QAI Pipeline V2

> **Audience:** Data labellers, dataset engineers, and MLOps team members preparing
> training and validation sets for the VisionFood QAI inspection pipeline.

---

## 1. Camera Setup

| Parameter | Requirement |
|-----------|-------------|
| Resolution | Minimum **1280 × 720 px** (Full HD 1920 × 1080 px preferred) |
| Frame rate | 30 fps minimum; 60 fps preferred for fast conveyor lines |
| Lighting | Diffuse LED ring light — avoid specular highlights on transparent containers |
| Background | Neutral matte surface (grey or black); no patterns behind product |
| Distance | Camera-to-product distance must be consistent within ± 5 mm across a batch |
| Angle | Top-down or 30 ° front-angle — document the exact angle in the dataset README |
| Calibration | Capture an ArUco calibration board image before each annotation session |

---

## 2. General Annotation Rules

1. **Tool:** Use [Label Studio](https://labelstud.io/) or [CVAT](https://cvat.ai/) with the
   `Polygon` or `Rectangle` tool. Export in **YOLO format** (`.txt` sidecar files).
2. **Class IDs** must match `configs/class_map.yaml` exactly — do not use free-text labels.
3. **Bounding box tightness:** Boxes must be tight to the visible defect boundary,
   with ≤ 5 px margin.
4. **Occlusion:** Annotate defects that are ≥ 50 % visible. Fully occluded defects
   must be skipped.
5. **Minimum defect area:** Do not annotate defects whose bounding box is less than
   **0.5 % of the total image area** — they are below pipeline detection resolution.
6. **Difficult flag:** Mark ambiguous annotations with the `difficult=1` flag.
   These are excluded from mAP calculation but kept for training.
7. **Train / Val / Test split:** 70 / 15 / 15. Split is performed **per product SKU**
   to ensure all SKUs are represented in every subset.
8. **File naming:** `{sku_id}_{session_date}_{frame_index:06d}.jpg`
   e.g., `bottle_250ml_20260315_000123.jpg`.
9. **Metadata sidecar:** Each session produces a `session_meta.json` with:
   `sku_id`, `capture_date`, `operator`, `camera_model`, `lighting_config`, `conveyor_speed_mps`.

---

## 3. Per-Sub-Type Annotation Instructions

### 3.1 `flexible_wrapper` (Pouches — `lays_50g`, `pouch_100g`)

| Class | Annotation guidance |
|-------|---------------------|
| `surface_tear` | Annotate any visible perforation, puncture, or linear tear in the film. Include the full extent of the tear even if the edges curl back. |
| `surface_smudge` | Annotate ink smears, grease marks, or moisture staining on the outer film surface. Minimum size: 1 % of product area. |
| `label_misalignment` | Annotate the entire label region when the label is visibly skewed (> 3 °) or offset (> 5 mm from nominal centre). |
| `improper_filling` | Annotate when the pouch appears visibly under-filled (excessive headspace) or over-filled (bulging seals). |
| `packaging_damage` | Annotate crushed or crease-damaged corners and seal failures. |

**Seal annotation note:** Mark the seal region separately when a seal failure
is detectable. Use the `packaging_damage` class with a tight polygon around
the failed seal segment.

---

### 3.2 `rigid_can` (Cans — `can_330ml`, `can_food_400g`)

| Class | Annotation guidance |
|-------|---------------------|
| `packaging_damage` | Annotate dents (any visible deformation of the cylindrical body or lid), cracks, and sharp rim deformations. Even a small dent on the lid seam qualifies — see `can_food_400g` risk weight `0.90`. |
| `surface_contamination` | Annotate rust spots, corrosion blisters, or external liquid contamination. |
| `label_misalignment` | For printed-on labels: annotate when text/graphic registration is off by more than 2 mm. For adhesive labels: annotate any visible bubble, peel, or skew. |
| `surface_smudge` | Annotate oil, fingerprints, or production-line grime covering > 2 % of the label area. |

**Fill-level note:** `fill_level_detectable: false` for all `rigid_can` SKUs.
Do **not** annotate fill-level defect classes for cans; the pipeline will skip
that sub-pipeline entirely.

---

### 3.3 `rigid_box` (Cardboard — `cardboard_box_generic`)

| Class | Annotation guidance |
|-------|---------------------|
| `packaging_damage` | Annotate corner crush, edge crease, and panel deformation. A "crush" is any inward deformation visible from the camera angle. |
| `surface_smudge` | Annotate moisture staining, water marks, or ink transfer on the outer surface. |
| `label_misalignment` | Annotate when the product label/sticker is peeling, bubbling, or shifted by more than 3 mm from its nominal position. |

**Corner annotation tip:** Use a polygon (not rectangle) for corner-crush damage,
as the damage boundary is rarely axis-aligned.

---

### 3.4 `transparent_bottle` (Bottles — `bottle_250ml`, `transparent_bottle_500ml`)

| Class | Annotation guidance |
|-------|---------------------|
| `surface_contamination` | Annotate visible particulate matter, biofilm, or liquid splash on the outer bottle surface. Any contamination on the neck/cap region is critical. |
| `improper_filling` | Annotate when the liquid meniscus is outside the `fill_level_min_ratio`–`fill_level_max_ratio` window defined in the SKU profile. Draw the bbox around the headspace or overfill bulge. |
| `cap_fitting_anomaly` | Annotate when the cap is visibly misaligned (tilted > 5 °), cross-threaded, or absent. |
| `label_misalignment` | Annotate the full label panel when shifted, wrinkled, or peeling. |
| `packaging_damage` | Annotate chips, cracks, or scratches on the glass/PET body. |

**HSV calibration note:** When adding a new transparent bottle SKU, capture
10 reference frames of a correctly filled, uncontaminated bottle under standard
lighting. Use the `tools/calibrate_hsv.py` script to derive `expected_bottle_hsv_centre`.

---

## 4. Dataset Usage Guidance

### 4.1 Delivering datasets to the pipeline

1. Place annotated images in `data/datasets/{sku_id}/images/{split}/`.
2. Place YOLO `.txt` label files in `data/datasets/{sku_id}/labels/{split}/`.
3. Create a `data/datasets/{sku_id}/dataset.yaml` following the YOLOv11 format:
   ```yaml
   path: data/datasets/{sku_id}
   train: images/train
   val:   images/val
   test:  images/test
   nc: <number of classes>
   names: [<class names in class_map order>]
   ```

### 4.2 Quality checks before delivery

Run the following before handing off a dataset:

```bash
python tools/validate_annotations.py --dataset data/datasets/{sku_id}
```

This script checks:
- All image files have a matching label file.
- No bounding boxes exceed image bounds.
- Class IDs are within the expected range.
- No duplicate frame indices within a session.

### 4.3 Class imbalance guidelines

| Defect class | Minimum instances per split |
|--------------|-----------------------------|
| `surface_contamination` | 200 train / 50 val |
| `packaging_damage` | 200 train / 50 val |
| `improper_filling` | 150 train / 35 val |
| `label_misalignment` | 100 train / 25 val |
| `surface_tear` | 150 train / 35 val |
| `surface_smudge` | 100 train / 25 val |
| `cap_fitting_anomaly` | 100 train / 25 val |

If a class falls below the minimum, apply **offline augmentation**
(random crop, brightness jitter, horizontal flip) before training.
Document augmentation steps in `session_meta.json`.

### 4.4 Versioning

Dataset versions follow `YYYY.MM.PATCH` (e.g., `2026.04.1`).
Record the version in `session_meta.json` and tag the commit in the
`data/` submodule with the same string.
