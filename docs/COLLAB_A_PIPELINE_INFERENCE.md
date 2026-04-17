# VisionFood QAI — Pipeline V2: Collaborator A Implementation Plan

## Role: Pipeline & Inference Engineer

**Date:** April 14, 2026
**Approach:** Agent-executable prompts, parallel phases where possible
**Imaging method:** Standard RGB camera — no laser, no structured light
**Replaces:** `PIPELINE_V2_PLAN.md` (this split plan supersedes it for implementation)

---

## Conveyor Belt Execution Context

All modules you build run inside a continuous camera loop on a moving production line:

- Products arrive at irregular intervals (~1–3 seconds apart at normal line speed)
- The camera captures frames at 30fps continuously
- A product is in the optimal inspection position for approximately 3–10 frames before it exits
- Motion blur increases with belt speed — frame selection must be timing-sensitive
- **A single product SKU runs the full shift** — the operator sets product type once at run start (via Collaborator B's UI). No per-frame product classification is needed.

All inspection modules you write receive a single **pre-selected frame** from the `ConveyorFrameSelector`. Modules do not handle frame selection, queue management, or camera I/O — those belong to the preprocessor.

---

## File Ownership — Collaborator A

You own the following files exclusively. **Collaborator B will not touch these.**

| File | Action |
|---|---|
| `core/schemas.py` | Extend — new enums, defect classes, status models |
| `core/config.py` | Extend — new config flags |
| `remedy/sku_profile_manager.py` | Extend — loader for new YAML fields |
| `remedy/severity_scorer.py` | Extend — liquid vs solid severity differentiation |
| `inference/preprocessor.py` | Extend — conveyor belt frame selector class |
| `inference/pipeline.py` | Extend — product router, verdict logic |
| `inference/modules/__init__.py` | Create |
| `inference/modules/barcode_verifier.py` | Create |
| `inference/modules/label_ocr_verifier.py` | Create |
| `inference/modules/food_surface_inspector.py` | Create |
| `inference/modules/beverage_inspector.py` | Create |
| `tests/unit/test_frame_selector.py` | Create |
| `tests/unit/test_barcode_verifier.py` | Create |
| `tests/unit/test_label_ocr_verifier.py` | Create |
| `tests/unit/test_food_surface_inspector.py` | Create |
| `tests/unit/test_beverage_inspector.py` | Create |
| `tests/unit/test_severity_scorer_liquid.py` | Create |
| `tests/benchmarks/test_module_latency.py` | Create |

---

## Interface Contract with Collaborator B

### What Collaborator B needs FROM you (A → B):

| Deliverable | Needed by B phase | Delivery trigger |
|---|---|---|
| `core/schemas.py` updated with `ProductCategory`, `ProductSubType`, `ContainerContents`, `LabelQRStatus`, `LabelTextStatus`, new `DefectClass` values, new `InspectionResult` fields **including `annotated_image_b64` and `gradient_weights`** | B Phase 1-B (types.ts sync) | Your Phase 0-A complete |
| `remedy/sku_profile_manager.py` loading `container_contents`, `product_sub_type`, and new YAML fields | B Phase 1-C (inspection router) | Your Phase 0-C complete |
| `inference/pipeline.py` — `inspect()` signature accepting `product_sub_type` and `container_contents` | B Phase 3-B (E2E tests) | Your Phase 2-A complete |

### What you need FROM Collaborator B (B → A):

| Deliverable | Needed by your phase | How to stub locally before it arrives |
|---|---|---|
| `configs/sku_profiles/*.yaml` with `product_sub_type`, `container_contents`, `barcode_verification`, `ocr_date_fields`, `fill_level_*` fields populated | Your Phase 1 testing | Create `tests/fixtures/test_sku_profile.yaml` with the same structure — use it in all unit tests |
| `database/repositories/product_repository.py` exposing `get_expected_qr(sku)` and `get_expected_dates(sku)` async callables | Your Phase 1-A and 1-B | Use `unittest.mock.AsyncMock` returning hardcoded values in unit tests |

**Architectural rule:** Never import from `api/`, `database/`, or `dashboard/` in any `inference/` code. All database lookups needed in inference modules must be injected as async callables in constructors. This keeps the inference layer testable without a live database.

---

## Phase 0 — Foundation
**All five tasks are fully parallel. No dependencies between them. Start all immediately.**

---

### Task 0-A: Schema Extensions (`core/schemas.py`)

Prompt the agent:

> In `core/schemas.py`, make the following additions. Do not modify or remove any existing fields or classes — all changes are additive to maintain backward compatibility:
>
> 1. Add a `ProductCategory` enum with values `food` and `beverage`.
> 2. Add a `ProductSubType` enum with values:
>    - `flexible_wrapper` — thin foil or plastic pouches (snack bags, sachets)
>    - `rigid_can` — aluminium or steel cylindrical containers
>    - `rigid_box` — any cuboidal cardboard enclosure regardless of size or product type (cereal boxes, biscuit boxes, pasta packaging, tea boxes — not limited to milk cartons)
>    - `transparent_bottle` — PET or glass bottles where liquid level is visible through the packaging
> 3. Add a `ContainerContents` enum with values `liquid`, `solid`, and `powder`. This is used by the severity scorer — do not confuse it with product type. A rigid can can hold either liquid (beer) or solid (chickpeas); the contents type drives damage severity independently.
> 4. Extend the existing `DefectClass` enum to add the following values without removing any: `fill_level_low`, `fill_level_high`, `cap_fitting_anomaly`, `surface_tear`, `surface_smudge`, `label_date_mismatch`, `label_barcode_mismatch`.
> 5. If `DEFECT_CLASS_NAMES` (or any defect class display name mapping) exists in the file, add entries for all seven new classes.
> 6. Add a `LabelQRStatus` Pydantic model with fields: `qr_detected: bool`, `qr_decoded: Optional[str] = None`, `qr_expected: Optional[str] = None`, `qr_matched: bool = False`, `label_anomaly_types: list[str] = []`. If this class already exists as a TypeScript-only type, this is the Python backend equivalent.
> 7. Add a `LabelTextStatus` Pydantic model with fields: `dates_verified: bool`, `fields: dict[str, str]` (field name → extracted value), `anomaly_types: list[str]`.
> 8. Add the following optional fields to `InspectionResult`, all defaulting to `None` for backward compatibility: `product_category: Optional[ProductCategory]`, `product_sub_type: Optional[ProductSubType]`, `container_contents: Optional[ContainerContents]`, `label_qr: Optional[LabelQRStatus]`, `label_text: Optional[LabelTextStatus]`.
> 9. Add `annotated_image_b64: Optional[str] = None` to `InspectionResult`. This field holds the JPEG-encoded base64 image with OpenCV bounding-box overlays drawn for all detected defects. It is populated in `inference/pipeline.py` (Task 2-A) after all inspection modules complete, before building the result object. This is a **critical output** — Collaborator B's API stores it in MongoDB and the frontend displays it directly. If the frame received by `inspect()` is `None` (no frame emitted by `ConveyorFrameSelector`), set `annotated_image_b64 = None`. If detections list is empty, still encode and return the clean frame so the operator can see the inspected product.
> 10. Add `gradient_weights: Optional[list[float]] = None` to `InspectionResult`. These are the Grad-CAM channel weights (αk — global-average-pooled gradients of the top predicted class score with respect to the last convolutional feature map channels). For ONNX-only edge deployments these remain `None`. When the pipeline runs in PyTorch mode (full model with `register_backward_hook`), populate this field: it is a compact 1-D float vector (one value per feature map channel, e.g. 512 values for ResNet-18's last conv layer). Collaborator B's server stores these in MongoDB for future server-side XAI heatmap reconstruction without requiring the edge to resend full feature maps.

---

### Task 0-B: Config Extensions (`core/config.py`)

Prompt the agent:

> In `core/config.py`, add the following new fields to `EdgeConfig`. All must have sensible production defaults and be readable from environment variables. Group them under clearly labelled comment blocks:
>
> **`# --- Conveyor Belt Frame Trigger ---`**
> - `BELT_TRIGGER_MODE: str = "software"` — `software` uses bbox IOU stability; `hardware` is reserved for future photoelectric sensor integration
> - `BELT_STABILITY_IOU_THRESHOLD: float = 0.92` — minimum bbox IOU between consecutive frames to count as stable
> - `BELT_MIN_STABLE_FRAMES: int = 3` — consecutive stable frames required before triggering inspection
> - `BELT_DEBOUNCE_FRAMES: int = 15` — frames to suppress after a trigger to avoid re-inspecting the same product as it exits
> - `BELT_TRIGGER_ROI: list[float] = [0.25, 0.1, 0.75, 0.9]` — normalised [x1, y1, x2, y2] of the region within which a product centroid must fall to be eligible for inspection
>
> **`# --- Label Verification ---`**
> - `QR_VERIFICATION_ENABLED: bool = True`
> - `QR_CACHE_TTL_SEC: int = 300`
> - `OCR_ENABLED: bool = True`
> - `OCR_MIN_CONFIDENCE: float = 0.6`
>
> **`# --- Verdict Overrides ---`**
> - `DATE_MISMATCH_VERDICT: str = "ESCALATE"` — verdict applied when OCR finds a printed date that differs from the registered expected value
> - `FILL_LEVEL_HARD_FAIL: bool = True` — if True, any fill level deviation forces FAIL regardless of UQ confidence; fill level is a measurement not a probability

---

### Task 0-C: SKU Profile Manager Extensions (`remedy/sku_profile_manager.py`)

Prompt the agent:

> Extend `remedy/sku_profile_manager.py` to load the following new YAML fields when reading a SKU profile. All new fields must be optional with documented defaults so existing profiles without them continue to load without error:
>
> - `product_sub_type: str` — one of `flexible_wrapper`, `rigid_can`, `rigid_box`, `transparent_bottle`. Default: `flexible_wrapper`
> - `container_contents: str` — one of `liquid`, `solid`, `powder`. Default: `solid`. This is independent of packaging shape — a rigid can may hold liquid or solid contents
> - `label_region: list[float] | None` — normalised [x1, y1, x2, y2] of the label area on the product face, used to crop before OCR. Default: `None` (use full frame)
> - `barcode_verification.target_type: str` — barcode type to target. Default: `QRCODE`
> - `barcode_verification.expected_value_field: str` — field name on the product document to compare decoded value against. Default: `qr_code`
> - `ocr_date_fields: list[dict]` — list of `{name: str, format: str}` dicts. Default: `[]` (OCR skipped if empty)
> - `fill_level_detectable: bool` — whether fill level analysis applies (requires transparent packaging with visible liquid). Default: `False`
> - `fill_level_min_ratio: float` — minimum acceptable fill ratio. Default: `0.85`
> - `fill_level_max_ratio: float` — maximum acceptable fill ratio. Default: `0.98`
> - `cap_symmetry_threshold: float` — minimum symmetry score for cap check. Default: `0.70`
> - `surface_contamination_threshold: float` — fraction of body pixels deviating from expected HSV that triggers contamination flag. Default: `0.05`
> - `expected_bottle_hsv_centre: list[int] | None` — [H, S, V] of the expected clean-bottle colour for contamination baseline. Default: `None` (module uses statistical mode from frame if not provided)
>
> Expose all loaded values as attributes on the existing `SKUProfile` dataclass or equivalent. Do not change the existing public method signatures of `SKUProfileManager`.

---

### Task 0-D: Severity Scorer — Liquid vs Solid (`remedy/severity_scorer.py`)

Prompt the agent:

> Modify `remedy/severity_scorer.py` to apply different base risk weights for `packaging_damage` depending on `container_contents` from the SKU profile. The rationale is that structural packaging damage has different real-world consequences depending on what is inside:
>
> - `liquid`: `packaging_damage` base risk → `0.95`. Any structural breach on a liquid container causes immediate spillage and line contamination.
> - `powder`: `packaging_damage` base risk → `0.80`. Powder escapement and caking are serious but slightly less time-critical than liquid spill.
> - `solid`: `packaging_damage` base risk → `0.60`. Structural damage to solid food packaging affects presentation and shelf life but does not cause immediate contamination.
>
> Implementation requirements:
> 1. Add an optional `container_contents: Optional[ContainerContents]` parameter to `SeverityScorer.score()` (or store it at construction time if you refactor). Default to `ContainerContents.solid` when not provided to preserve existing behaviour.
> 2. Look up the container-type multiplier in a dict before applying any SKU profile `class_risk_overrides`. The profile overrides still take final precedence — the container type multiplier is the new default, overridable per-profile.
> 3. All existing unit tests in `tests/unit/test_pipeline_verdict.py` must continue to pass unchanged.
> 4. Document the multiplier dict and rationale in a comment block in the file.

---

### Task 0-E: Conveyor Belt Frame Selector (`inference/preprocessor.py`)

Prompt the agent:

> Add a `ConveyorFrameSelector` class to `inference/preprocessor.py`. This class handles continuous frame ingestion from the camera loop on a moving conveyor belt and emits ready-to-inspect frames.
>
> **Constructor:** Accept `config: EdgeConfig`. Read `BELT_TRIGGER_ROI`, `BELT_STABILITY_IOU_THRESHOLD`, `BELT_MIN_STABLE_FRAMES`, `BELT_DEBOUNCE_FRAMES`, and `BELT_TRIGGER_MODE` from config at init time.
>
> **`push_frame(frame: np.ndarray, bbox: Optional[list[float]]) -> Optional[np.ndarray]`:**
> Called on every camera frame. `bbox` is a normalised [x1, y1, x2, y2] from the YOLO presence detection (or `None` if no product detected).
>
> Logic:
> 1. If `bbox` is `None` (no product), reset stability counter and best-candidate buffer. Return `None`.
> 2. Compute the centroid of `bbox`. If the centroid falls outside `BELT_TRIGGER_ROI`, reset and return `None`.
> 3. Compute IOU between the current `bbox` and the previous frame's `bbox`. If IOU >= `BELT_STABILITY_IOU_THRESHOLD`, increment a stability counter and update the best-candidate frame (keep the frame with the highest consecutive stability count seen so far). Otherwise reset the counter.
> 4. If stability counter reaches `BELT_MIN_STABLE_FRAMES`: emit the current frame (return it), activate debounce mode. During debounce (`BELT_DEBOUNCE_FRAMES` frames remaining), return `None` unconditionally regardless of bbox state. When debounce expires, reset all state.
> 5. **Fallback for fast belts:** If a product's bbox exits `BELT_TRIGGER_ROI` before stability was ever achieved (stability counter never reached `BELT_MIN_STABLE_FRAMES`), but a best-candidate frame exists: emit the best-candidate frame and log a `WARNING` including the SKU context (`"belt speed exceeded stability threshold — inspecting best available frame"`). This prevents products from passing uninspected at high line speeds.
>
> **Design note to document in code:** In future, replace software-based triggering with `BELT_TRIGGER_MODE = "hardware"` where a photoelectric sensor on the belt sends a GPIO/serial interrupt directly to `push_frame`. The software implementation here is the production fallback for lines without sensor infrastructure.
>
> Write unit tests in `tests/unit/test_frame_selector.py` covering: normal product entering and stabilising → frame emitted once; debounce preventing double-emit from same product; product moves too fast (never stable) → best-candidate fallback emitted with warning logged; product bbox centroid outside ROI → no frame emitted.

---

## Phase 1 — Inspection Modules
**All four blocks are fully parallel. Phase 0 must be complete first.**

---

### Block 1-A: Multi-Barcode QR Decoder (`inference/modules/barcode_verifier.py`)

Prompt the agent:

> Create `inference/modules/barcode_verifier.py`. Implement a `BarcodeVerifier` class.
>
> **Constructor:** Accept `get_product_qr: Callable[[str], Awaitable[Optional[str]]]` — an async callable that accepts a SKU string and returns the expected QR string from the database, or `None` if none registered. This is injected to keep the module database-agnostic and unit-testable.
>
> **`async verify(frame: np.ndarray, sku: str, sku_profile: SKUProfile) -> LabelQRStatus`:**
>
> 1. Convert the frame to a PIL RGB image. Run `pyzbar.decode()` on it.
> 2. **Multi-barcode handling:** If `len(decoded_barcodes) > 1`, log a `WARNING` that includes: SKU, number of barcodes found, and a list of their types (e.g. `['QRCODE', 'EAN13']`). Do NOT raise — having both a QR code and a linear retail barcode on the same label is valid and common.
> 3. Filter the decoded list to only barcodes whose `barcode.type` matches `sku_profile.barcode_verification.target_type` (default `QRCODE`).
> 4. If no barcode of the target type survives the filter: return `LabelQRStatus(qr_detected=False, qr_matched=False, label_anomaly_types=["expected_type_not_found"])`.
> 5. Decode the matched barcode data bytes as UTF-8. Call `await get_product_qr(sku)` to get the expected value. Cache the response in a module-level async TTL cache keyed by SKU using TTL from `QR_CACHE_TTL_SEC` config — avoid a database hit on every frame.
> 6. If `expected_qr` is `None` (product not registered with a QR value), return `LabelQRStatus(qr_detected=True, qr_decoded=decoded, qr_expected=None, qr_matched=False, label_anomaly_types=["qr_not_registered"])`.
> 7. Compare decoded vs expected. Return fully populated `LabelQRStatus`.
>
> Write unit tests in `tests/unit/test_barcode_verifier.py`. Use `unittest.mock.AsyncMock` for `get_product_qr`. Cover: single QR match, QR mismatch, frame with QR + EAN13 where QR is found correctly, frame with only EAN13 where QRCODE is target (not found), frame with no barcodes.

---

### Block 1-B: Label OCR Date Verifier (`inference/modules/label_ocr_verifier.py`)

Prompt the agent:

> Create `inference/modules/label_ocr_verifier.py`. Implement a `LabelOCRVerifier` class.
>
> **Constructor:** Accept `get_product_dates: Callable[[str], Awaitable[dict[str, str]]]` — an async callable returning `{field_name: expected_date_string}` for the given SKU (e.g. `{"expiry_date": "06/2026"}`).
>
> **`async verify(frame: np.ndarray, sku: str, sku_profile: SKUProfile) -> Optional[LabelTextStatus]`:**
>
> 1. If `sku_profile.ocr_date_fields` is empty or `OCR_ENABLED` config flag is `False`, return `None` immediately — no OCR overhead.
> 2. Crop the frame to `sku_profile.label_region` (denormalise coordinates to pixel values using frame dimensions). If `label_region` is `None`, use the full frame but log a `DEBUG` warning that OCR accuracy may be lower without a label region hint.
> 3. Pre-process the cropped region before passing to Tesseract:
>    - Convert to grayscale
>    - Apply `cv2.adaptiveThreshold` (Gaussian, block size 21, C=10)
>    - Upscale to minimum 2× using `cv2.INTER_CUBIC` (achieves ~300 DPI equivalent at typical 300mm working distance)
>    - Apply a mild sharpening kernel (3×3 with centre weight 5, edges -1)
> 4. Run `pytesseract.image_to_string(processed, lang='eng', config='--psm 6 --oem 3')`.
> 5. For each entry in `sku_profile.ocr_date_fields`, apply format-specific regex to the OCR output string. Support at minimum: `DD/MM/YYYY`, `MM/YYYY`, `DDMMMYYYY` (e.g. `15JAN2025`), `YYYY-MM-DD`. If regex finds no match in the OCR text: record anomaly `ocr_extraction_failed` for that field. If a value is extracted but differs from the registered expected value: record `label_date_mismatch` for that field.
> 6. Return `LabelTextStatus(dates_verified=all_matched, fields=extracted_dict, anomaly_types=anomaly_list)`.
>
> Write unit tests in `tests/unit/test_label_ocr_verifier.py`. Mock `get_product_dates` with `AsyncMock`. Mandatory test names:
> - `test_correct_dates_verified` — all dates extracted and match registered values
> - `test_qr_passes_but_date_mismatch` — correct QR would pass, but printed date is wrong (proves OCR catches what QR alone cannot)
> - `test_ocr_extraction_fails_no_match` — Tesseract output contains no recognisable date pattern

---

### Block 1-C: Food Surface Inspector (`inference/modules/food_surface_inspector.py`)

Prompt the agent:

> Create `inference/modules/food_surface_inspector.py`. Implement a `FoodSurfaceInspector` class.
>
> **`inspect(frame: np.ndarray, product_sub_type: ProductSubType, product_bbox: Optional[list[float]]) -> list[Detection]`:**
>
> If `product_bbox` is provided, crop the frame to the product region (denormalise coordinates) before running any detection. This reduces false positives from background.
>
> Run the appropriate branch based on `product_sub_type`:
>
> **`flexible_wrapper` branch (foil/plastic pouches):**
> Target defects: `surface_tear` (physical breach) and `surface_smudge` (contamination mark).
> - Convert to LAB colour space. Run Canny edge detection on the L channel with thresholds tuned for thin reflective plastic (lower: 30, upper: 100). Find contours. Flag contour segments that: exceed a minimum arc length (configurable, default 15 pixels), have a perpendicular orientation relative to the dominant product axis (tears typically run against the wrapper grain). Return `Detection(class_name="surface_tear", confidence=contour_area/product_area)`.
> - Convert to HSV. Detect regions where V < 40 (abnormally dark) or S > 185 (abnormally saturated) that are not explained by the product's printed design. Use morphological opening to remove noise before measuring blob area. Return `Detection(class_name="surface_smudge", confidence=blob_area/product_area)`.
> - **Comment in code:** `# Classical CV baseline for thin reflective packaging. Replace with a trained PatchCore or EfficientAD anomaly model once annotated dataset reaches 200+ positives per class (see docs/ANNOTATION_REQUIREMENTS.md).`
>
> **`rigid_can` branch (cylindrical aluminium/steel):**
> Target defects: dents on the cylindrical body surface.
> - Use only the centre 60% of the cropped image height (exclude top/bottom 20% — these contain the lid and base curves which have valid high-edge-density geometry). Convert to grayscale. Compute horizontal Sobel gradient magnitude. Compute column-wise average gradient magnitude. A localised column-band where the average gradient exceeds 3× the global mean for the body strip indicates a surface discontinuity (dent or deep scratch). Flag as `Detection(class_name="packaging_damage", confidence=peak_gradient/global_mean_gradient, capped at 1.0)`.
>
> **`rigid_box` branch (cuboidal cardboard boxes — any size/product type):**
> Target defects: corner crush and moisture staining. This branch handles all rectangular cardboard packaging — cereal boxes, biscuit boxes, pasta packaging, tea boxes — not only milk cartons.
> - Corner crush: apply Shi-Tomasi corner detection on the full crop. Fit the four strongest corner candidates to a convex quadrilateral. If the fitted quad's aspect ratio deviates more than 15% from the expected bounding box aspect ratio, flag as `Detection(class_name="packaging_damage", confidence=deviation_ratio)`.
> - Moisture staining: convert to HSV. Compute a mask of pixels with H in [10, 30] (brownish), S > 60, V in [40, 180]. If the masked pixel fraction exceeds 2% of the product area, flag as `Detection(class_name="surface_smudge", confidence=stain_fraction)`.
>
> All branches: return an empty list (not an exception) for invalid frames or insufficient product area. Write unit tests in `tests/unit/test_food_surface_inspector.py` using solid-colour synthetic numpy frames for each sub-type. Tests must confirm: each branch runs without exceptions, and the returned list contains `Detection` objects with valid `class_name` and a confidence float in [0, 1].

---

### Block 1-D: Beverage Inspector (`inference/modules/beverage_inspector.py`)

Prompt the agent:

> Create `inference/modules/beverage_inspector.py`. Implement a `BeverageInspector` class.
>
> **`inspect(frame: np.ndarray, sku_profile: SKUProfile, bottle_bbox: Optional[list[float]]) -> tuple[list[Detection], bool]`:**
>
> Returns `(detections, fill_level_undetectable)`. If `bottle_bbox` is provided, crop the frame to the bottle region first.
>
> **Check 1 — Fill Level** (only if `sku_profile.fill_level_detectable == True`):
>
> The goal is to find the liquid-air interface in a transparent bottle body — the sharp horizontal transition between coloured liquid below and clear empty headspace above.
>
> - Divide the cropped bottle into three vertical zones by height fraction: cap (top 12%), body (12–88%), base (88–100%).
> - Within the body zone, compute per-row mean HSV saturation. Liquid regions (coloured) have higher saturation than the clear headspace above. Find the uppermost row where per-row saturation exceeds 40% of the body-wide mean saturation — this is the liquid surface row.
> - If no clear transition is found (standard deviation of per-row saturation across the body zone < 15), set `fill_level_undetectable = True` and return without flagging any defect. Opaque, frosted, or fully labelled bottles will hit this path — do not penalise them.
> - Otherwise: `fill_ratio = (body_height - (liquid_surface_row - body_top_row)) / body_height`. If `fill_ratio < fill_level_min_ratio`: return `Detection(class_name="fill_level_low", confidence=min_ratio - fill_ratio)`. If `fill_ratio > fill_level_max_ratio`: return `Detection(class_name="fill_level_high", confidence=fill_ratio - max_ratio)`.
> - **Comment in code:** `# confidence here is the ratio deviation, not a model probability. Fill level is a measurement; confidence = how far out of range, not classification certainty.`
>
> **Check 2 — Cap Fitting:**
>
> - Crop the top 15% of the bottle bounding box height (cap region). Convert to grayscale.
> - Apply `cv2.HoughCircles` (HOUGH_GRADIENT) to detect the cap rim. Use the bounding box width to bound the expected radius range (min: 20% of bbox width, max: 55% of bbox width).
> - If Hough finds no circle: flag `cap_fitting_anomaly` with fixed confidence 0.70 (cap absence is binary — either detectable or not).
> - If Hough finds a circle: check (a) that the circle centre x-coordinate is within 10% of the cap region's midpoint x (symmetry), and (b) that the radius is within ±20% of `sku_profile` expected cap radius (derive from bbox width as above). If either check fails: flag `cap_fitting_anomaly`.
> - If multiple Hough circles are returned: use the one with the strongest accumulator vote (first in the returned array).
>
> **Check 3 — Surface Contamination:**
>
> - Within the body zone (12–88% height), convert to HSV. If `sku_profile.expected_bottle_hsv_centre` is set, use it as the clean-bottle colour centre [H, S, V] with tolerance H±15, S±30, V±40. If not set, compute the mode HSV value from the body zone pixels and use that as the expected centre.
> - Compute the fraction of body pixels that fall outside the tolerance range. If this fraction exceeds `sku_profile.surface_contamination_threshold`: flag `Detection(class_name="surface_contamination", confidence=deviation_fraction)`.
>
> Write unit tests in `tests/unit/test_beverage_inspector.py` with synthetic numpy frames: a vertical gradient for fill level testing (known pixels above and below threshold), a symmetric vs horizontally-shifted cap crop for cap fitting, and a uniform vs patched body frame for contamination. All tests must assert the return type is `tuple[list[Detection], bool]`.

---

## Phase 2 — Pipeline Integration
**Depends on Phase 0 and Phase 1 completing. Tasks 2-A and 2-B are parallel.**

---

### Task 2-A: Product Router in Pipeline (`inference/pipeline.py`)

Prompt the agent:

> Modify `inference/pipeline.py` to integrate the new inspection modules. Make the following changes:
>
> **1. Update `inspect()` signature:** Add optional parameters `product_sub_type: Optional[str] = None` and `container_contents: Optional[str] = None`. These are passed in by the API layer from the active production run (Collaborator B populates them). If not provided, resolve them from the SKU profile via `SKUProfileManager`.
>
> **2. Frame guard:** At the start of `inspect()`, if the received frame is `None`, return `None` immediately. The caller (capture loop) may pass `None` when `ConveyorFrameSelector` returns no frame.
>
> **3. Add `_route_to_modules(frame, sku, sku_profile, yolo_detections)` async method:**
>
> - Always call `await BarcodeVerifier.verify(frame, sku, sku_profile)` — for all product types. Store as `label_qr`.
> - If `sku_profile.ocr_date_fields` is non-empty: call `await LabelOCRVerifier.verify(frame, sku, sku_profile)`. Store as `label_text`.
> - If `sku_profile.product_category == "food"`: call `FoodSurfaceInspector.inspect(frame, sku_profile.product_sub_type, primary_yolo_bbox)`. Merge returned detections into `yolo_detections`.
> - If `sku_profile.product_sub_type == "transparent_bottle"`: call `BeverageInspector.inspect(frame, sku_profile, primary_yolo_bbox)`. Merge returned detections. Capture `fill_level_undetectable` flag.
> - Return `(merged_detections, label_qr, label_text, fill_level_undetectable)`.
>
> **4. Instantiate modules** at pipeline construction time (not per-call) to avoid repeated initialisation overhead. Inject the DB callables for `BarcodeVerifier` and `LabelOCRVerifier` from the FastAPI dependency injection context (pass them in through the pipeline constructor).
>
> **5. Populate the new `InspectionResult` fields** from the module outputs: `label_qr`, `label_text`, `product_category`, `product_sub_type`, `container_contents`.
>
> **6. Render annotated image (`annotated_image_b64`):** After `_route_to_modules()` returns and `merged_detections` is final, draw bounding-box overlays on a copy of the original frame using OpenCV:
> - For each detection in `merged_detections`: draw a filled semi-transparent rectangle for the bbox background, then draw the full bbox border using `cv2.rectangle()`. Use a distinct colour per defect class (define a `DEFECT_CLASS_COLOURS: dict[str, tuple[int, int, int]]` palette at module level — BGR format). Overlay class name + confidence percentage using `cv2.putText()` with a small drop shadow for legibility.
> - Encode the annotated frame as JPEG using `cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])`.
> - Base64-encode the JPEG bytes and store in `InspectionResult.annotated_image_b64`.
> - If the input frame is `None`, set `annotated_image_b64 = None`. If `merged_detections` is empty, encode the clean frame and store it anyway — the operator should see what the model inspected even when no defects are found.
> - **This field is required by Collaborator B** — it stores it in MongoDB `InspectionMedia` and the frontend fetches it for display. Do not skip this step.

---

### Task 2-B: Verdict Logic Update (`inference/pipeline.py` — `_apply_verdict_logic`)

Prompt the agent:

> Update `_apply_verdict_logic()` in `inference/pipeline.py` to handle new verdict conditions. All existing rules remain — these additions follow a defined priority order. Document the priority order as a comment above the verdict logic block:
>
> **Priority order (highest to lowest):**
> `QR_MISMATCH > FILL_LEVEL (hard_fail) > DATE_MISMATCH > standard confidence-gated defects`
>
> **New rule 1 — QR barcode mismatch (highest priority):**
> If `label_qr` is not None, `label_qr.qr_detected == True`, and `label_qr.qr_matched == False`: force verdict to `FAIL` with `escalated=True`. Skip all lower-priority checks. Rationale: a QR mismatch means the wrong product is on the line — entire batch identity is wrong.
>
> **New rule 2 — Fill level deviation:**
> If any detection in the merged list has class `fill_level_low` or `fill_level_high`, AND `fill_level_undetectable` is `False`, AND config `FILL_LEVEL_HARD_FAIL` is `True`: force verdict to `FAIL` with `escalated=False`. Rationale: fill level is a measured compliance parameter — it either meets spec or it doesn't.
>
> **New rule 3 — Date mismatch:**
> If `label_text` is not None and `"label_date_mismatch"` is in `label_text.anomaly_types`: apply the verdict from `DATE_MISMATCH_VERDICT` config (default `ESCALATE`). Only applies if QR mismatch has not already forced a FAIL.
>
> **New rule 4 — Cap fitting anomaly:**
> No override — process through standard UQ confidence-gated logic like all other defect classes.
>
> **New rule 5 — Severity scorer update:**
> Pass `container_contents` from the SKU profile into `SeverityScorer.score()` so liquid/solid severity differentiation applies automatically.

---

## Phase 3 — Unit Tests
**Parallel with Phase 2. Most tests can be written alongside module development in Phase 1.**

### Task 3-A: Severity Scorer Liquid/Solid Tests

Prompt the agent:

> In `tests/unit/test_severity_scorer_liquid.py`, write tests verifying:
>
> 1. A `packaging_damage` detection on `container_contents=liquid` produces a higher severity grade than the same detection at the same confidence on `container_contents=solid`.
> 2. `container_contents=powder` falls between `liquid` and `solid` for `packaging_damage` severity.
> 3. The existing tests in `tests/unit/test_pipeline_verdict.py` all still pass — the default `container_contents=solid` preserves old behaviour.
> 4. A `class_risk_override` in the SKU profile for `packaging_damage` correctly overrides the container-type default.

---

## Phase 4 — Latency Benchmarks
**Depends on Phase 2 completing.**

### Task 4-A: Module Latency Benchmarks

Prompt the agent:

> Create `tests/benchmarks/test_module_latency.py` using `pytest-benchmark`. Benchmark each new module in isolation using synthetic numpy frames (1920×1080 for full-frame tests, 400×200 crops where noted). Mock all DB calls with `AsyncMock`.
>
> Targets and rationale:
> - `ConveyorFrameSelector.push_frame()` — target: **<2ms**. Called on every camera frame at 30fps; must be near-zero overhead.
> - `BarcodeVerifier.verify()` on 1920×1080 — target: **<10ms**. pyzbar is a C library wrapper; Python overhead only.
> - `LabelOCRVerifier.verify()` on a 400×200 label crop — target: **<800ms** on CPU. Tesseract is CPU-bound. This is acceptable because OCR only runs when `ocr_date_fields` is non-empty in the SKU profile (opt-in per product).
> - `FoodSurfaceInspector.inspect()` for each of `flexible_wrapper`, `rigid_can`, `rigid_box` — target: **<35ms each** (classical CV only, no model inference).
> - `BeverageInspector.inspect()` — target: **<30ms** (classical CV + Hough).
>
> Total pipeline latency budget: 300ms end-to-end. OCR is the only module with a budget >100ms and is justified by its opt-in gating.
>
> If any benchmark exceeds its target, add: `# LATENCY BUDGET EXCEEDED: this classical CV fallback should be replaced with a trained ONNX model. See docs/ANNOTATION_REQUIREMENTS.md for dataset requirements.`

---

## Dependency Map

```
Phase 0 (all parallel — start immediately)
  0-A  0-B  0-C  0-D  0-E
       │
       ▼
Phase 1 (all parallel — depends on all of Phase 0)
  1-A  1-B  1-C  1-D
       │
       ▼
Phase 2 (2-A and 2-B parallel — depends on Phase 1)
  ┌────┴─────┐
  2-A       2-B
  └────┬─────┘
       │
       ├──── Phase 3-A runs in parallel with Phase 2
       │     (only needs Phase 0-D to have run)
       ▼
Phase 4 (depends on Phase 2)
  4-A
```

---

## Key Design Decisions

**Why product type is NOT classified per frame:**
A production line runs one SKU continuously per shift. The product type is set once by the operator at run start (via Collaborator B's UI). Per-frame classification would add model overhead, introduce misclassification risk, and solve a problem that does not exist in this deployment context. The SKU profile carries the type; the pipeline reads it.

**Why async callables are injected into modules rather than importing DB code:**
All inference modules that need database values (`BarcodeVerifier`, `LabelOCRVerifier`) accept the lookup as an injected callable. This makes the `inference/` package independently testable with no database dependency. At runtime, the FastAPI application injects the real repository methods. In tests, `AsyncMock` provides fake values.

**Why the conveyor belt selector uses bbox IOU rather than frame differencing:**
Simple frame differencing detects the belt texture moving under the product as motion, causing false stability readings. IOU between consecutive product bounding boxes (from the YOLO presence detection) specifically measures whether the product itself has stabilised — high IOU means the product has slowed into the inspection window. This gives a much cleaner trigger.

**Why `rigid_box` is the correct sub-type name (not `carton`):**
`carton` implies a milk or juice carton specifically. The system applies the same inspection logic to any cuboidal cardboard enclosure — cereal boxes, snack boxes, pasta packaging, medicine boxes. `rigid_box` is unambiguous and inclusive of all these forms.

**Why fill level detection uses classical CV for transparent bottles:**
The liquid-air interface in a transparent bottle is a sharp, high-contrast horizontal transition (coloured liquid → clear headspace). This is reliably localised with HSV row-wise saturation analysis — no training data needed. The `fill_level_undetectable` guard handles opaque or labelled bottles by disabling the check cleanly. A trained model would add latency and training data overhead for a signal that classical CV handles adequately.
