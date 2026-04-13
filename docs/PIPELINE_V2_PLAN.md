# VisionFood QAI — Pipeline V2 Implementation Plan

**Date:** April 13, 2026  
**Approach:** Agent-executable prompts, parallel phases preferred  
**Imaging method:** Standard camera (RGB frame) — no laser, no structured light  
**Key change from V1:** Product-type-aware pipeline with dedicated inspection modules per category

---

## Architecture Overview

The pipeline now branches at the product type level before any model runs. A single camera captures the product frame; the SKU profile (already a YAML file per product) carries the product type and sub-type, which determines which inspection modules are activated for that frame.

```
Camera Frame
    │
    ▼
Product Type Router   ← reads SKU profile
    │
    ├──── FOOD ──────────────────────────────────────────────────┐
    │     ├── Subtype: Flexible Wrapper (Lays-type)              │
    │     ├── Subtype: Rigid Can                                 │
    │     └── Subtype: Carton                                    │
    │     Common checks: Surface anomaly, Label anomaly + OCR dates, Multi-barcode QR
    │                                                            │
    └──── BEVERAGE ──────────────────────────────────────────────┘
          └── Subtype: Transparent Bottle
          Checks: Fill level, Cap fitting, Surface contamination
```

The REMEDY engine, severity scoring, and verdict logic remain unchanged downstream — only the defect classes and modules feeding into them change per product type.

---

## Phase 0 — Schema and Profile Foundation
**Can start immediately. All tasks in this phase are parallel.**

### Task 0-A: Extend Product Type Schema (`core/schemas.py`)

Prompt the agent:

> In `core/schemas.py`, add two new enumerations: `ProductCategory` with values `food` and `beverage`, and `ProductSubType` with values `flexible_wrapper`, `rigid_can`, `carton`, and `transparent_bottle`. Extend the existing `DefectClass` enum to include the following new defect classes without removing existing ones: `fill_level_low`, `fill_level_high`, `cap_fitting_anomaly`, `surface_tear`, `surface_smudge`, `label_date_mismatch`, `label_barcode_mismatch`. Update `DEFECT_CLASS_NAMES` to reflect all classes. Add `ProductCategory` and `ProductSubType` as optional fields on `InspectionResult` so the type is carried through to the database and dashboard.

### Task 0-B: Extend SKU Profile Schema (`remedy/sku_profile_manager.py`)

Prompt the agent:

> Extend the SKU profile YAML structure and the `sku_profile_manager.py` loader to support two new required fields: `product_category` (already present in existing YAMLs as `food` or `beverage`) and `product_sub_type` (one of `flexible_wrapper`, `rigid_can`, `carton`, `transparent_bottle`). Also add an optional `ocr_date_fields` section per profile that lists which date fields to verify (e.g. `mfg_date`, `expiry_date`) and the expected date format string (e.g. `DD/MM/YYYY`). Add a `barcode_verification` section with `target_type` (either `QRCODE` or `CODE128` etc.) and `expected_value_field` pointing to the product registration field to compare against.

### Task 0-C: Update Existing SKU YAMLs

Prompt the agent:

> Update all three existing SKU profile YAML files (`bottle_250ml.yaml`, `can_330ml.yaml`, `pouch_100g.yaml`) in `configs/sku_profiles/` to include the new fields introduced in Task 0-B. For `bottle_250ml.yaml` and `can_330ml.yaml` set `product_sub_type: transparent_bottle` and `product_sub_type: rigid_can` respectively. For `pouch_100g.yaml` set `product_sub_type: flexible_wrapper`. Add placeholder `ocr_date_fields` and `barcode_verification` sections to all three. Also create two new SKU profile YAMLs: `carton_1l.yaml` for a standard 1-litre food carton with `product_sub_type: carton` and appropriate risk weights.

### Task 0-D: Extend `InspectRequest` in `api/routers/inspection.py`

Prompt the agent:

> Add an optional `product_sub_type` field (string, max 32 chars) to the `InspectRequest` Pydantic model in `api/routers/inspection.py`. If not provided, it should be resolved from the SKU profile at inspection time by calling the `SKUProfileManager`. Pass the resolved sub-type into the pipeline's `inspect()` call.

---

## Phase 1 — Parallel Inspection Modules
**Phase 0 must complete first. All four blocks within Phase 1 are fully parallel.**

---

### Block 1-A: Multi-Barcode QR Decoder Module (replaces single-barcode pyzbar call)

Prompt the agent:

> Create a new file `inference/modules/barcode_verifier.py`. Implement a `BarcodeVerifier` class whose `verify(frame, sku_profile)` method does the following:
>
> 1. Run `pyzbar.decode()` on the frame converted to a PIL RGB image.
> 2. If multiple barcodes are detected, log a warning with the count and list all types found (e.g. `QRCODE`, `CODE128`, `EAN13`).
> 3. Filter barcodes to only those whose `barcode.type` matches the `target_type` specified in the SKU profile's `barcode_verification` section. If no target type is configured, default to `QRCODE`.
> 4. If no barcode of the target type is found after filtering, return a `LabelQRStatus` with `qr_detected=False` and append `"expected_type_not_found"` to `label_anomaly_types`.
> 5. If the target barcode is found, decode its string value and compare it against the expected value fetched from the product record in MongoDB (field specified by `expected_value_field` in the profile). Use an in-memory cache keyed by SKU with a TTL read from `QR_CACHE_TTL_SEC` in config to avoid per-frame database hits.
> 6. Return a fully populated `LabelQRStatus` — including `qr_decoded`, `qr_expected`, `qr_matched`, and `label_anomaly_types`.
> 7. Add unit tests in `tests/unit/test_barcode_verifier.py` covering: single QR match, single QR mismatch, multiple barcodes with correct type filter, multiple barcodes where target type absent, and no barcodes at all.

---

### Block 1-B: Label OCR & Date Verification Module

Prompt the agent:

> Create a new file `inference/modules/label_ocr_verifier.py`. Implement a `LabelOCRVerifier` class whose `verify(frame, sku_profile)` method does the following:
>
> 1. Accept the full camera frame and the SKU profile. If the profile has no `ocr_date_fields` configured, return immediately with an empty result — do not run OCR.
> 2. Use the bounding box or region hint from the SKU profile (add a `label_region` field in Phase 0-B: normalised `x1, y1, x2, y2`) to crop the label area from the frame before passing to OCR. If no region hint exists, use the full frame.
> 3. Pre-process the crop: convert to grayscale, apply adaptive thresholding (not global), and upscale to at least 300 DPI equivalent using bicubic interpolation before passing to Tesseract. This improves OCR accuracy on camera-captured text without a laser scanner.
> 4. Run `pytesseract.image_to_string()` with `lang='eng'` and `config='--psm 6'` (assume a uniform block of text).
> 5. For each date field listed in `ocr_date_fields`, apply a regex to the raw OCR output to extract the date value. Support at minimum: `DD/MM/YYYY`, `MM/YYYY`, and `DDMMMYYYY` formats.
> 6. Compare each extracted date against the expected value stored in the product record in MongoDB (fetched via the same SKU cache used by the barcode verifier). If extraction fails because the regex finds no match, treat it as an anomaly (`ocr_extraction_failed`). If the value is found but doesn't match, treat as `label_date_mismatch`.
> 7. Return a new `LabelTextStatus` Pydantic model (add to `core/schemas.py`) containing: `dates_verified: bool`, `fields: dict[str, str]` (field name → extracted value), `anomaly_types: list[str]`.
> 8. Add unit tests in `tests/unit/test_label_ocr_verifier.py` covering: clean label image with correct dates, correct QR but wrong printed date (to prove OCR catches what QR alone misses), and images where Tesseract fails to extract a date.

---

### Block 1-C: Food Surface Anomaly Module (Wrappers, Cans, Cartons)

Prompt the agent:

> Create a new file `inference/modules/food_surface_inspector.py`. Implement a `FoodSurfaceInspector` class whose `inspect(frame, product_sub_type)` method returns a list of `Detection` objects representing surface anomalies found. The method must behave differently per sub-type:
>
> **For `flexible_wrapper` (Lays-type):**  
> Surface anomalies to detect are tears, punctures, and smudges. The wrapping material is thin reflective plastic — the dominant visual cues are irregular specular highlights (tears catch light differently), dark smear regions (smudges), and irregular edge geometry. Implement a primarily classical CV approach first: use Canny edge detection to find irregular edge breakpoints (potential tears), and blob detection on a lightness-normalised frame to find smudge patches. Represent each anomaly as a bounding box with class `surface_tear` or `surface_smudge` and a confidence score based on contour area relative to the product bounding box area. Document clearly in the code where a trained anomaly detection model should be plugged in later to replace the classical CV fallback.
>
> **For `rigid_can`:**  
> Surface anomalies are dents, scratches, and label deformation. Use edge magnitude analysis on the cylindrical surface region (centre band of the frame, excluding top and bottom). Flag areas where horizontal edge density is significantly higher than background (dent = deformation of the smooth cylinder wall). Return detections with class `packaging_damage`.
>
> **For `carton`:**  
> Detect corner crush, crease lines, and moisture staining. Cartons have flat rectangular faces — use corner detection and quadrilateral fitting. If the fitted quad deviates significantly from a rectangle, flag as `packaging_damage`. For staining, detect brownish discolouration in HSV space.
>
> All three sub-types share: skip processing if `product_sub_type` does not match `food` category; return an empty list rather than raising. Add unit tests in `tests/unit/test_food_surface_inspector.py` with mock frames for each sub-type using solid-colour patches to verify the module runs without exception and returns the correct structure, even if no anomalies are detected.

---

### Block 1-D: Beverage Inspection Module (Transparent Bottle)

Prompt the agent:

> Create a new file `inference/modules/beverage_inspector.py`. Implement a `BeverageInspector` class whose `inspect(frame, sku_profile)` method returns a list of `Detection` objects. It must implement three checks:
>
> **Check 1 — Fill Level:**  
> This applies only to transparent bottles where the liquid level is visible through the packaging.  
> Approach: identify the bottle region using the YOLOv11 detection bounding box passed in (do not re-detect; accept detections as a parameter). Within the bottle region, convert to HSV and isolate the liquid-air interface by looking for the sharp horizontal transition between the coloured liquid region and the clear empty headspace above it. Compute the fill ratio as `(liquid height) / (total bottle interior height)`. Compare against an acceptable range loaded from the SKU profile (add `fill_level_min_ratio` and `fill_level_max_ratio` to the profile). If outside range, return a `Detection` with class `fill_level_low` or `fill_level_high` and a confidence score derived from how far the ratio deviates from the acceptable boundary.  
> **Important caveat to document in code:** this method relies on liquid-air contrast being visible. Opaque, frosted, or highly labelled bottles will have zero contrast — in that case, return `fill_level_undetectable=True` in the result and do not flag a defect. The SKU profile should have a `fill_level_detectable: bool` flag to short-circuit this check for non-transparent products.
>
> **Check 2 — Cap Fitting Anomaly:**  
> Crop the top region of the bottle (top 15% of the bounding box height). Convert to grayscale. Compute the horizontal symmetry score of the cap region — a properly fitted cap is symmetric about the vertical centre axis. A cross-correlation symmetry score below a configurable threshold (add `cap_symmetry_threshold` to SKU profile) should be flagged as `cap_fitting_anomaly`. Alternatively, check whether the cap has uniform circular edge geometry using Hough Circle Transform. Document both approaches and implement the simpler (Hough) first.
>
> **Check 3 — Surface Contamination:**  
> On the bottle body region (excluding cap and base crops), look for non-uniform brightly coloured or dark patches inconsistent with the expected bottle colour profile. The expected colour profile (dominant HSV range) should be derivable from the good samples in the dataset. At runtime, compute the percentage of pixels deviating from the expected range. If it exceeds `surface_contamination_threshold` from the SKU profile, flag as `surface_contamination` with the fraction as confidence.
>
> Add unit tests in `tests/unit/test_beverage_inspector.py` for each check using synthetic frames: a gradient image for fill level, symmetric/asymmetric cap crop for cap fitting, and a clean vs stained body for surface contamination.

---

## Phase 2 — Pipeline Router Integration
**Depends on Phase 0 and Phase 1 completing. Tasks 2-A and 2-B are parallel.**

### Task 2-A: Product Type Router in Pipeline

Prompt the agent:

> Modify `inference/pipeline.py` to add a `_route_inspection(frame, sku, product_sub_type, detections)` method. This method:
>
> 1. Reads the SKU profile to determine `product_category` and `product_sub_type` if not supplied by the caller.
> 2. If `product_category == 'food'`: instantiate and call `FoodSurfaceInspector.inspect()` and `LabelOCRVerifier.verify()` in addition to the existing detection. Merge the returned defect detections into the main `detections` list before passing to UQ and REMEDY.
> 3. If `product_category == 'beverage'` and `product_sub_type == 'transparent_bottle'`: instantiate and call `BeverageInspector.inspect()`, passing the YOLO detections for the bottle bounding box. Merge returned detections.
> 4. Always call `BarcodeVerifier.verify()` for all product types — this replaces the raw pyzbar call currently planned in V1.
> 5. Do not change the UQ, verdict, or REMEDY stages — they receive the merged detection list and run unchanged.
> 6. Add a `label_text: Optional[LabelTextStatus]` field to `InspectionResult` in `core/schemas.py` to carry the OCR date verification result alongside the existing `label_qr` field.

### Task 2-B: Update Verdict Logic

Prompt the agent:

> Update the `_apply_verdict_logic()` method in `inference/pipeline.py` to handle two new verdict triggers:
>
> 1. If `label_text` is present and `label_text.anomaly_types` contains `label_date_mismatch`, apply the same verdict weight as a QR mismatch (configurable: add `DATE_MISMATCH_VERDICT` to `EdgeConfig`, defaulting to `ESCALATE`). Note: QR mismatch should still take priority over date mismatch.
> 2. For beverage products, if `fill_level_low` or `fill_level_high` is in the detection list, treat them as a `FAIL` verdict regardless of confidence, because fill level deviation is a compliance issue not subject to uncertainty (it either is or isn't in range). Add a config flag `FILL_LEVEL_HARD_FAIL: bool = True` to make this overridable.
> 3. For `cap_fitting_anomaly`: apply standard confidence-gated verdict logic (same as other defects). Cap anomalies are mechanical and visible — no hard overrides needed.
> 4. Document any new config flags added in a comment block at the top of `core/config.py` in the appropriate section.

---

## Phase 3 — SKU Profile Expansion & Dataset Annotations
**Parallel with Phase 2. Independent of pipeline code changes.**

### Task 3-A: Create New SKU Profiles

Prompt the agent:

> Create the following new SKU profile YAML files in `configs/sku_profiles/`. For each, include all existing fields (class risk overrides, rejection area thresholds, remedy stations, max attempts) plus the new fields defined in Phase 0-B (product_sub_type, ocr_date_fields, barcode_verification, fill_level settings, cap_symmetry_threshold, surface_contamination_threshold, label_region):
>
> - `lays_50g.yaml` — flexible wrapper, 50g snack pouch, food category
> - `can_food_400g.yaml` — rigid can, 400g food product (e.g. beans), food category
> - `carton_1l.yaml` — carton, 1-litre juice/milk, food category  
> - `transparent_bottle_500ml.yaml` — transparent PET bottle, 500ml beverage
>
> Use conservative risk weights for food categories and tighter rejection thresholds for cans (seal failure risk). For the transparent bottle, set `fill_level_detectable: true`, `fill_level_min_ratio: 0.88`, `fill_level_max_ratio: 0.97`, and `cap_symmetry_threshold: 0.75`.

### Task 3-B: Dataset Annotation Requirements Document

Prompt the agent:

> Create `docs/ANNOTATION_REQUIREMENTS.md` (documentation only, no code). This document must specify for a human annotator or external annotation team exactly what to annotate per product sub-type from the existing datasets in `DATASETS/` and any new data captured from the production camera:
>
> For **flexible_wrapper**: bounding boxes for tears (polygon preferred), smudges (bounding box). At least 200 positive examples per class. Include images at different lighting conditions (overhead, side-lit, direct flash) since the camera setup is fixed position.
>
> For **rigid_can**: bounding boxes for dents (mark the deformed region on the cylinder surface), scratches, label misalignment. Do not annotate printed text as a defect.
>
> For **carton**: corner crush (polygon around the corner), crease lines (line segment annotations), moisture staining (bounding box with estimated severity note).
>
> For **transparent_bottle**: annotate fill level as a horizontal line coordinate (y-pixel of liquid surface), cap region as bounding box (top 15% of bottle), contamination patches as bounding boxes. For fill level, note the fill ratio computed from the annotation along with the bottle exterior top/bottom y coordinates.
>
> For all types: include 30% "good" (no defect) examples per class for balance. Note camera position requirements (270–400mm working distance, diffuse backlight for fill level visibility on bottles).

---

## Phase 4 — Integration Tests & Latency Validation
**Depends on Phase 2 completing. Tasks are parallel.**

### Task 4-A: End-to-End Product Type Pipeline Tests

Prompt the agent:

> Create `tests/integration/test_product_type_pipeline.py`. Write integration tests that cover the following scenarios using synthetic numpy frames (no live camera required):
>
> 1. Food wrapper product: pipeline routes to `FoodSurfaceInspector` and `LabelOCRVerifier`, result contains `label_text` field.
> 2. Food can product: pipeline routes to `FoodSurfaceInspector`, surface anomaly detections present in result.
> 3. Beverage transparent bottle: pipeline routes to `BeverageInspector`, result contains fill level result and cap result.
> 4. Mismatched barcode (correct type targeted): verdict is FAIL with escalated=True.
> 5. Multiple barcodes in frame: warning is logged, correct type is selected, wrong types are ignored.
> 6. Correct QR but wrong printed expiry date: verdict is ESCALATE, `label_text.anomaly_types` contains `label_date_mismatch`.
> 7. Transparent bottle with undetectable fill level (opaque mock): no fill level defect flagged, `fill_level_undetectable=True` in result.
>
> All tests must assert that `result.latency_ms < 300`. Use `unittest.mock` to stub the MongoDB SKU cache lookups so tests run without a live database.

### Task 4-B: Per-Module Latency Benchmarks

Prompt the agent:

> Create `tests/benchmarks/test_module_latency.py` using `pytest-benchmark`. Write benchmark tests for each new module in isolation:
>
> - `BarcodeVerifier.verify()` on a 1920×1080 frame — expected: under 10ms
> - `LabelOCRVerifier.verify()` on a cropped label region (approx 400×200px after crop) — expected: under 800ms on CPU (Tesseract is CPU-bound)
> - `FoodSurfaceInspector.inspect()` per sub-type — expected: under 30ms each (classical CV only)
> - `BeverageInspector.inspect()` — expected: under 25ms (classical CV + Hough)
>
> If any benchmark exceeds its budget, document the bottleneck in a comment and flag it as a known limitation requiring a trained model replacement in a future sprint. The latency budget for the full pipeline end-to-end remains 300ms total; OCR is the only module allowed to exceed 100ms individually because it runs only when a label region is configured and the SKU profile opts in.

---

## Phase 5 — Configuration & Deployment Readiness
**Depends on Phase 4. Final validation before integration with cloud deployment.**

### Task 5-A: Config Completeness Check

Prompt the agent:

> Audit `core/config.py` and ensure all new configuration flags introduced by Phase 1 and Phase 2 tasks are present with sensible defaults. Specifically verify these are in `EdgeConfig`: `QR_VERIFICATION_ENABLED`, `QR_CACHE_TTL_SEC`, `DATE_MISMATCH_VERDICT`, `FILL_LEVEL_HARD_FAIL`, `OCR_ENABLED`, `OCR_MIN_CONFIDENCE`. Add any that are missing. Update the `.env` example or documentation with the new keys.

### Task 5-B: Updated Database Schema for New Defect Classes

Prompt the agent:

> Update the MongoDB inspection document schema and any relevant Pydantic validators in `database/models.py` to allow the new defect class names (`fill_level_low`, `fill_level_high`, `cap_fitting_anomaly`, `surface_tear`, `surface_smudge`, `label_date_mismatch`, `label_barcode_mismatch`) as valid values in the `class_name` field of embedded defect sub-documents. Add `label_text` as an optional embedded object field on the inspection document schema. Add a compound MongoDB index on `(sku, defects.class_name, timestamp)` to support filtered analytics queries by defect type and product.

### Task 5-C: Dashboard Type Display Update

Prompt the agent:

> Update `dashboard/src/types.ts` to add the new defect class names to the type union or string literals used for `Detection.class_name`. Add a `label_text: LabelTextStatus | null` field to `InspectionResult` matching the schema added in Phase 2-A, where `LabelTextStatus` contains `dates_verified: boolean`, `fields: Record<string, string>`, and `anomaly_types: string[]`. Update any UI components that render defect class names to display human-readable labels for the new classes (e.g. `fill_level_low` → `"Underfill"`, `cap_fitting_anomaly` → `"Cap Loose/Misfit"`).

---

## Dependency Map (Visual)

```
Phase 0 (all parallel)
  0-A, 0-B, 0-C, 0-D
       │
       ▼
Phase 1 (all parallel, depends on Phase 0)
  1-A  1-B  1-C  1-D
       │
       ▼
Phase 2 (2-A and 2-B parallel, depends on Phase 1)
  ┌────┴─────┐
  2-A       2-B
  └────┬─────┘
       │               Phase 3-A, 3-B (parallel, independent of Phase 2)
       │               can start alongside Phase 1 or Phase 2
       ▼
Phase 4 (4-A and 4-B parallel, depends on Phase 2)
  ┌────┴─────┐
  4-A       4-B
  └────┬─────┘
       ▼
Phase 5 (5-A, 5-B, 5-C parallel, depends on Phase 4)
  ┌─────┬────┴────┐
  5-A  5-B       5-C
```

---

## Defect Class × Product Type Mapping

| Defect Class | Flexible Wrapper | Rigid Can | Carton | Transparent Bottle |
|---|:---:|:---:|:---:|:---:|
| `label_misalignment` | ✅ | ✅ | ✅ | ✅ |
| `label_barcode_mismatch` | ✅ | ✅ | ✅ | ✅ |
| `label_date_mismatch` | ✅ | ✅ | ✅ | ✅ |
| `surface_tear` | ✅ | — | — | — |
| `surface_smudge` | ✅ | — | ✅ | — |
| `packaging_damage` | — | ✅ | ✅ | — |
| `surface_contamination` | ✅ | ✅ | ✅ | ✅ |
| `fill_level_low` | — | — | — | ✅ |
| `fill_level_high` | — | — | — | ✅ |
| `cap_fitting_anomaly` | — | — | — | ✅ |
| `improper_filling` | ✅ (weight) | ✅ (volume) | ✅ (volume) | — |

---

## Key Design Decisions Documented

**Why camera OCR for date verification, not a laser scanner:**  
Laser line scanners (structured light) give precise 3D surface profiles but are expensive, require controlled positioning, and are overkill for flat label text. A standard RGB camera at controlled working distance (270–400mm) with diffuse illumination and adaptive thresholding as preprocessing is sufficient for reading printed date codes at industry QC speeds. The preprocessing pipeline in `LabelOCRVerifier` (crop → grayscale → adaptive threshold → upscale → Tesseract --psm 6) is specifically tuned for this setup.

**Why not EasyOCR for date verification:**  
Tesseract with preprocessing is faster on CPU (<800ms on a crop vs 2–3s for EasyOCR on the full frame), and date fields are short, regular-format strings — exactly where Tesseract is reliable. EasyOCR's advantage is on curved, handwritten, or highly stylised text, none of which apply to machine-printed date codes.

**Why fill level uses classical CV rather than a model:**  
For transparent bottles, the liquid-air interface is a sharp, high-contrast horizontal transition that can be reliably localised with HSV thresholding + horizontal edge detection — no training data needed. A trained model's only advantage would be handling ambiguous cases (bubbles, condensation); the SKU profile's `fill_level_detectable` flag already handles opaque containers by disabling the check.

**Why QR mismatch still has higher priority than date mismatch:**  
A QR mismatch means the entire product identity is wrong — the wrong product is on the line. A date mismatch means the right product has a wrong batch sticker. QR mismatch is therefore the more severe condition and results in FAIL + escalated, while date mismatch defaults to ESCALATE (which can be downgraded to REVIEW via config).

**Why cap fitting anomaly does not hard-fail:**  
Cap anomalies can be caused by camera angle variation, condensation on the cap, or reflective cap material causing false positives. Running with confidence-gated logic (same as other defects, going through UQ) is safer than hard-failing and triggering many false rejects during camera setup calibration. Once the model is well-calibrated and false positive rates are known, a hard-fail flag can be added to the SKU profile.
