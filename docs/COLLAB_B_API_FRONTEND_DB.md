# VisionFood QAI — Pipeline V2: Collaborator B Implementation Plan

## Role: API, Database & Frontend Engineer

**Date:** April 14, 2026
**Approach:** Agent-executable prompts, parallel phases where possible
**Replaces:** `PIPELINE_V2_PLAN.md` (this split plan supersedes it for implementation)

---

## Context for This Role

The pipeline now requires knowing the product type before inspection begins. Rather than classifying it per frame (which would need another model), the operator sets the product type **once at the start of a production run** through a form you will build. From that point, every inspection on that line automatically uses the correct sub-type profile.

This is architecturally sound because:
- A single conveyor line runs **one product SKU continuously** per shift — no mixed products
- The operator knows what they are running before they start the belt
- Setting type per run eliminates a whole class of misclassification errors and simplifies the model

Your work makes this operator interaction possible, persists the configuration, and ensures the inference engine always knows what it is inspecting.

---

## File Ownership — Collaborator B

You own the following files exclusively. **Collaborator A will not touch these.**

| File | Action |
|---|---|
| `database/models.py` | Extend — add `Product` and `ProductionRun` MongoDB document schemas |
| `database/repositories/product_repository.py` | Create |
| `database/repositories/production_run_repository.py` | Create |
| `api/routers/products.py` | Create |
| `api/routers/runs.py` | Create |
| `api/routers/inspection.py` | Extend — resolve sub-type from active run |
| `api/main.py` | Extend — register new routers |
| `dashboard/src/types.ts` | Extend — sync with A's schema additions |
| `dashboard/src/components/ProductRegistration.tsx` | Create |
| `dashboard/src/components/RunSetup.tsx` | Create |
| `configs/sku_profiles/*.yaml` | Create and update all YAML files |
| `tests/integration/test_products_api.py` | Create |
| `tests/integration/test_product_type_pipeline.py` | Create |
| `docs/ANNOTATION_REQUIREMENTS.md` | Create |

---

## Interface Contract with Collaborator A

### What Collaborator A delivers to you (A → B):

| Deliverable | Needed by your phase | How to proceed before it arrives |
|---|---|---|
| `core/schemas.py` — `ProductCategory`, `ProductSubType`, `ContainerContents`, `LabelQRStatus`, `LabelTextStatus`, updated `InspectionResult` | Your Phase 1-B (types.ts) | Draft `types.ts` with `// TODO: sync with Collab A schemas` placeholder comments on the new types; fill in once A delivers |
| `remedy/sku_profile_manager.py` loading `product_sub_type`, `container_contents`, and all new YAML fields | Your Phase 1-C (inspection router) | Stub the profile loader call with a mock in tests |
| `inference/pipeline.py` `inspect()` accepting `product_sub_type` and `container_contents` kwargs | Your Phase 3-B (E2E pipeline tests) | Use `unittest.mock.patch` on the pipeline in early test drafts |

### What you deliver to Collaborator A (B → A):

| Deliverable | Required by A's phase | Priority |
|---|---|---|
| `configs/sku_profiles/*.yaml` updated with new fields | A Phase 1 module testing | **Deliver first** — A needs this to test modules locally |
| `database/repositories/product_repository.py` with `get_expected_qr(sku)` and `get_expected_dates(sku)` async methods | A Phase 1-A and 1-B (injected DB callables) | Complete Phase 0-C before A starts Phase 1 |

---

## Phase 0 — Foundation
**All four tasks are fully parallel. Start all immediately. No external dependencies.**

---

### Task 0-A: Update All SKU Profile YAMLs (`configs/sku_profiles/`)

Prompt the agent:

> Update the three existing SKU profile YAML files in `configs/sku_profiles/` to include new fields that Collaborator A's profile manager will load. Do not modify any existing fields. Add the following sections to each:
>
> **`bottle_250ml.yaml`:**
> Add: `product_sub_type: transparent_bottle`, `container_contents: liquid`, `fill_level_detectable: true`, `fill_level_min_ratio: 0.88`, `fill_level_max_ratio: 0.97`, `cap_symmetry_threshold: 0.75`, `expected_bottle_hsv_centre: [105, 50, 200]`, `surface_contamination_threshold: 0.04`, `label_region: [0.10, 0.15, 0.90, 0.65]`, `barcode_verification: {target_type: QRCODE, expected_value_field: qr_code}`, `ocr_date_fields: [{name: expiry_date, format: "MM/YYYY"}]`.
>
> **`can_330ml.yaml`:**
> Add: `product_sub_type: rigid_can`, `container_contents: liquid`, `fill_level_detectable: false`, `surface_contamination_threshold: 0.05`, `label_region: [0.05, 0.20, 0.95, 0.80]`, `barcode_verification: {target_type: QRCODE, expected_value_field: qr_code}`, `ocr_date_fields: [{name: mfg_date, format: "DDMMMYYYY"}, {name: expiry_date, format: "DDMMMYYYY"}]`.
>
> **`pouch_100g.yaml`:**
> Add: `product_sub_type: flexible_wrapper`, `container_contents: solid`, `fill_level_detectable: false`, `surface_contamination_threshold: 0.06`, `label_region: [0.05, 0.05, 0.95, 0.60]`, `barcode_verification: {target_type: QRCODE, expected_value_field: qr_code}`, `ocr_date_fields: [{name: mfg_date, format: "DD/MM/YYYY"}, {name: best_before, format: "DD/MM/YYYY"}]`.
>
> Also create four new SKU profile YAML files:
>
> **`lays_50g.yaml`:** Snack pouch. `product_sub_type: flexible_wrapper`, `container_contents: solid`, `product_category: food`. Risk weights: `surface_tear: 0.85` (physical breach of snack pouch = food safety concern), `surface_smudge: 0.55`, `label_misalignment: 0.70`. `rejection_area_threshold: 0.02` (tight — 2% of product area). Two OCR date fields: `mfg_date` and `best_before` in `DD/MM/YYYY` format. `barcode_verification: {target_type: QRCODE}`.
>
> **`can_food_400g.yaml`:** Canned food product. `product_sub_type: rigid_can`, `container_contents: solid`, `product_category: food`. Risk weights: `packaging_damage: 0.90` (even for solid contents, a dented can lid seal is a food safety risk), `label_misalignment: 0.60`. `max_remediation_attempts: 1` (aluminium — irreversible deformation). OCR: `expiry_date` in `DDMMMYYYY` format.
>
> **`cardboard_box_generic.yaml`:** Any cuboidal cardboard enclosure. `product_sub_type: rigid_box`, `container_contents: solid`, `product_category: food`. Risk weights: `packaging_damage: 0.65` (corner crush/crease — affects shelf life and presentation for solid contents), `surface_smudge: 0.50` (moisture staining), `label_misalignment: 0.60`. `rejection_area_threshold: 0.05`. Add YAML comment: `# Default profile for any rectangular cardboard box — cereal, biscuits, pasta, tea, confectionery. Override class_risk_overrides for product-specific thresholds.`
>
> **`transparent_bottle_500ml.yaml`:** PET beverage bottle. `product_sub_type: transparent_bottle`, `container_contents: liquid`, `product_category: beverage`. `fill_level_detectable: true`, `fill_level_min_ratio: 0.88`, `fill_level_max_ratio: 0.97`, `cap_symmetry_threshold: 0.75`, `expected_bottle_hsv_centre: [105, 50, 200]` (placeholder — calibrate per actual product colour), `surface_contamination_threshold: 0.04`. `max_remediation_attempts: 2`. OCR: `expiry_date` in `MM/YYYY` format.

---

### Task 0-B: MongoDB Document Schemas (`database/models.py`)

Prompt the agent:

> In `database/models.py`, add the following MongoDB document schema definitions using the same Motor-compatible Pydantic pattern already present in the file (with `model_config = ConfigDict(populate_by_name=True)` and `id: Optional[PyObjectId] = Field(alias="_id")`). Do not modify existing document models:
>
> **`Product` document:**
> Fields: `sku: str` (unique index), `name: str` (max 80 chars), `description: Optional[str] = None` (max 300 chars — free-text operator notes, no inspection behaviour), `product_category: str` (one of `food`/`beverage`), `product_sub_type: str`, `container_contents: str`, `qr_code: Optional[str] = None` (expected decoded QR value for label verification), `expected_dates: dict[str, str] = {}` (field name → expected date string, e.g. `{"expiry_date": "06/2026"}`), `sku_profile_name: str` (file stem of the YAML in `configs/sku_profiles/`), `created_at: datetime`, `updated_at: datetime`, `__v: int = 0`.
>
> Add `ProductCreate` Pydantic model for `POST /products` requests — exclude `_id`, `created_at`, `updated_at`, `__v` (set server-side). Add `ProductUpdate` for `PATCH` — all fields optional.
>
> **`ProductionRun` document:**
> Fields: `run_id: str` (UUID, unique), `sku: str`, `product_id: PyObjectId` (reference to Product `_id`), `started_at: datetime`, `ended_at: Optional[datetime] = None`, `status: str` (one of `active`, `completed`, `aborted`), `operator_id: str`, `inspection_count: int = 0`, `defect_count: int = 0`, `__v: int = 0`.
>
> Add `RunCreate` model for `POST /runs` — fields: `sku: str`, `operator_id: str`.
>
> Add a function `create_product_run_indexes(db: AsyncIOMotorDatabase)` that creates: `Product.sku` unique index; compound `(ProductionRun.sku, ProductionRun.status)` index; `ProductionRun.started_at` descending index. Call this function from `database/session.py` startup alongside any existing index creation.

---

### Task 0-C: Product & Run Repositories (`database/repositories/`)

Prompt the agent:

> Create `database/repositories/product_repository.py`. Implement a `ProductRepository` class. All methods are async. Constructor accepts `db: AsyncIOMotorDatabase`.
>
> Methods:
> - `async create_product(data: ProductCreate) -> Product` — insert document, set `created_at = utcnow()`, `updated_at = utcnow()`, `__v = 0`. Let `DuplicateKeyError` propagate (caught at API layer as 409).
> - `async get_product_by_sku(sku: str) -> Optional[Product]` — `findOne({"sku": sku})`. Return `None` if not found.
> - `async get_expected_qr(sku: str) -> Optional[str]` — returns `product.qr_code` or `None`. **This method is the async callable injected into `BarcodeVerifier` by Collaborator A.**
> - `async get_expected_dates(sku: str) -> dict[str, str]` — returns `product.expected_dates` or `{}`. **This method is the async callable injected into `LabelOCRVerifier` by Collaborator A.**
> - `async update_product(sku: str, data: ProductUpdate, current_version: int) -> Optional[Product]` — `findOneAndUpdate` with filter `{"sku": sku, "__v": current_version}`. If no document matches (version mismatch or not found), return `None`. Set `updated_at = utcnow()`, `$inc: {__v: 1}`.
> - `async list_products(skip: int = 0, limit: int = 50) -> list[Product]` — paginated list, sorted by `created_at` descending.
> - `async list_sku_profile_names() -> list[str]` — scans `configs/sku_profiles/` directory and returns all `.yaml` file stems (without extension). Used by the frontend dropdown and the API validation for `sku_profile_name`.
>
> Create `database/repositories/production_run_repository.py`. Implement `ProductionRunRepository`:
> - `async start_run(data: RunCreate, product_id: PyObjectId) -> ProductionRun` — creates run with `run_id=uuid4()`, `status="active"`, `started_at=utcnow()`.
> - `async get_active_run_for_sku(sku: str) -> Optional[ProductionRun]` — `findOne({"sku": sku, "status": "active"})`.
> - `async get_any_active_run() -> Optional[ProductionRun]` — `findOne({"status": "active"})`.
> - `async end_run(run_id: str, final_status: str) -> Optional[ProductionRun]` — updates `status`, sets `ended_at = utcnow()`.
> - `async increment_counts(run_id: str, defect_found: bool)` — atomic `$inc: {"inspection_count": 1, "defect_count": (1 if defect_found else 0)}`.

---

### Task 0-D: Dataset Annotation Requirements (`docs/ANNOTATION_REQUIREMENTS.md`)

Prompt the agent:

> Create `docs/ANNOTATION_REQUIREMENTS.md`. This is a documentation-only file written as instructions for a human annotator or external annotation team. No code. Structure it as follows:
>
> **Camera and Setup Requirements (all product types):**
> - Working distance: 270–400mm from product surface to camera lens
> - Illumination: diffuse front lighting preferred to minimise specular highlights on smooth surfaces. For transparent bottles, add optional diffuse back-illumination to improve liquid-air interface contrast for fill level annotation.
> - Resolution: 1080p minimum (1920×1080)
>
> **General Annotation Rules (all product types):**
> - Include 30% "no defect" images per product sub-type for class balance
> - Minimum 200 annotated positive examples per defect class per product sub-type
> - Each annotation must include: product sub-type tag, defect class label, severity estimate (1=minor, 2=moderate, 3=critical)
> - Do not annotate normal printing variation, intended design features, or shadows as defects
>
> **Flexible Wrappers (foil/plastic pouches — snack bags, sachets):**
> - `surface_tear`: Polygon annotation preferred over bounding box. Mark only the torn/punctured region. Include: fresh tears (bright specular edge), aged tears (dark contaminated edge), pinhole punctures. Collect at 3 different lighting angles per sample since reflective plastic changes appearance dramatically.
> - `surface_smudge`: Bounding box. Include ink smear, dirt transfer, fingerprint marks. Do not annotate moisture condensation.
>
> **Rigid Cans (aluminium or steel cylinders):**
> - `packaging_damage` (dent): Bounding box around the deformed region on the cylinder wall. Do not annotate the lid seam or base ring. Only inspect the centre 60% of can height (top/bottom lids are excluded from machine inspection). Annotate dents ≥3mm depth (smaller are below detection threshold).
> - `label_misalignment`: Bounding box on the label edge. Only annotate when misalignment >5mm.
> - Note: fill level detection does not apply to cans (`fill_level_detectable: false`).
>
> **Rigid Boxes (cuboidal cardboard — any product type):**
> - **Important:** This category covers ALL rectangular cardboard packaging regardless of what is inside — cereal boxes, biscuit boxes, pasta packaging, tea boxes, medicine boxes, confectionery. Not restricted to milk cartons.
> - `packaging_damage` (corner crush): Polygon around the crushed corner. Include boxes of at least 3 different sizes in the dataset. Annotate only when the box corner is visibly deformed, not just scuffed.
> - `surface_smudge` (moisture staining): Bounding box. Mark brownish staining only — H in [10, 30] in HSV. Do not annotate normal printing colour variation.
> - `label_misalignment`: Bounding box on label edge, >5mm deviation.
>
> **Transparent Bottles (PET or glass, liquid contents visible):**
> - Fill level: Do NOT use a bounding box. Annotate as a horizontal line at the liquid surface (y-pixel coordinate of the interface). Also record: bottle exterior top y-coordinate, bottle exterior bottom y-coordinate, and the computed fill ratio `= (bottle_interior_bottom - liquid_surface_y) / bottle_interior_height`. Include images with correct fill, underfill, and overfill examples.
> - Cap region: Bounding box of the cap area (top 15% of bottle height). Annotate as `cap_fitting_anomaly` only for: visibly tilted cap, loose cap (off-centre gap visible), or missing cap entirely. Do not annotate normal cap colour variation.
> - `surface_contamination`: Bounding boxes on the bottle body (exclude cap and base 12%). Mark only foreign deposits — do not annotate label area as contamination.
>
> **Using Existing DATASETS:**
> The `DATASETS/mvtec_anomaly_detection/` folder contains: `bottle/`, `can/`, `capsule/` sub-folders. These can be used as supplementary examples after re-labelling to the project's defect class taxonomy. Key mappings: MVTec `broken_large`/`broken_small` on bottles → `packaging_damage`; MVTec `contamination` → `surface_contamination`; MVTec `poke` on capsules is not relevant. The `DATASETS/can/` folder and `DATASETS/juice bottles/` folders should be reviewed for any pre-annotated images before commissioning new labelling work.

---

## Phase 1 — API Layer
**All tasks parallel. Depends on Phase 0. Also requires Collaborator A's Phase 0-A (`core/schemas.py`) for enum validation.**

---

### Task 1-A: Product Registration API (`api/routers/products.py`)

Prompt the agent:

> Create `api/routers/products.py` as a FastAPI `APIRouter`. Use `Depends(get_db)` to inject the Motor database. Apply role guards using the existing `require_role` dependency pattern from `api/dependencies.py`.
>
> Endpoints:
>
> - `POST /products` (requires `operator` role or higher): Accept `ProductCreate`. Before inserting:
>   1. Validate `sku` against regex `^[A-Z0-9_-]{3,32}$` — return 422 if invalid. Validation should be on the Pydantic model itself via `@field_validator`.
>   2. Validate that `sku_profile_name` corresponds to an actual file in `configs/sku_profiles/` (call `ProductRepository.list_sku_profile_names()`) — return 422 with a clear message if not found.
>   3. Validate `product_category` and `product_sub_type` match valid enum values from `core/schemas.py`. Return 422 for invalid values.
>   4. Strip leading/trailing whitespace from all string fields.
>   5. Call `ProductRepository.create_product()`. Return `201 Created`. Return `409 Conflict` on `DuplicateKeyError` with message `"A product with SKU '{sku}' already exists"`.
>
> - `GET /products` (requires `viewer` role): Paginated list. Query params: `skip: int = 0`, `limit: int = 50` (max 100 enforced). Return `200` with list of `Product` documents.
>
> - `GET /products/{sku}` (requires `viewer` role): Return single product or `404`.
>
> - `PATCH /products/{sku}` (requires `supervisor` role): Accept `ProductUpdate` with a required `__v: int` field (optimistic concurrency). Call `ProductRepository.update_product(sku, data, current_version=data.__v)`. If returned `None`, check if product exists — if yes, return `409` (version mismatch); if no, return `404`.
>
> - `GET /products/sku-profiles` (requires `viewer` role): Return list of available SKU profile names from `ProductRepository.list_sku_profile_names()`. Used by the frontend dropdown.

---

### Task 1-B: Production Run API (`api/routers/runs.py`)

Prompt the agent:

> Create `api/routers/runs.py` as a FastAPI `APIRouter`. Use `Depends(get_db)` for Motor injection. Apply role guards.
>
> Endpoints:
>
> - `POST /runs` (requires `operator` role): Accept `RunCreate` (fields: `sku: str`, `operator_id: str`). Before creating:
>   1. Verify the product with the given SKU exists — return `404` if not.
>   2. Check that no other `active` run exists for the same SKU via `ProductionRunRepository.get_active_run_for_sku(sku)` — return `409 Conflict` with message `"A production run for SKU '{sku}' is already active"` if one exists.
>   3. Create the run. Return `201 Created` with the full `ProductionRun` document including `run_id`.
>
> - `GET /runs/active` (requires `viewer` role): Return the currently active run (if any) from any SKU. Returns `200` with `ProductionRun` or `null`. Used by the dashboard run status bar.
>
> - `GET /runs/active/{sku}` (requires `viewer` role): Return the active run for a specific SKU, or `null`. Used by the capture loop before starting inspection.
>
> - `PATCH /runs/{run_id}/end` (requires `supervisor` role): Accept body `{status: "completed" | "aborted"}`. Validate status value. Call `ProductionRunRepository.end_run(...)`. Return `200` with updated `ProductionRun` or `404` if run not found.

---

### Task 1-C: Extend Inspection Router (`api/routers/inspection.py`)

Prompt the agent:

> Modify the existing `api/routers/inspection.py` to resolve product sub-type and container contents from the active production run. Make only these changes:
>
> 1. Add optional fields `product_sub_type: Optional[str] = None` (max 32 chars) and `container_contents: Optional[str] = None` to the `InspectRequest` Pydantic model.
> 2. In the `POST /inspect` endpoint handler, after validating the request and before calling the pipeline: if either field is `None`, call `ProductionRunRepository.get_active_run_for_sku(request.sku)` to get the active run. Then load the SKU profile via `SKUProfileManager.load(active_run.sku)` and read the missing field(s) from it.
> 3. If no active run exists, log a `WARNING` with the SKU and proceed with `None` values (pipeline uses its own defaults). Do not reject the inspection request.
> 4. Pass the resolved `product_sub_type` and `container_contents` as keyword arguments into the existing `pipeline.inspect(...)` call.
> 5. Do not change the response schema — the new fields are populated inside the pipeline and returned in the existing `InspectionResult` response.

---

### Task 1-D: Register New Routers (`api/main.py`)

Prompt the agent:

> In `api/main.py`, register the two new routers from `api/routers/products.py` and `api/routers/runs.py`. Add them using `app.include_router()` with prefix `/api/v1` and tags `["products"]` and `["production-runs"]` respectively. Place registrations alongside the existing router includes. Also call `create_product_run_indexes(db)` inside the lifespan startup block alongside any existing startup tasks.

---

## Phase 2 — Frontend Components
**Depends on Phase 1. Also requires Collaborator A's Phase 0-A for enum values in types.ts. Tasks are parallel.**

---

### Task 2-A: TypeScript Type Sync (`dashboard/src/types.ts`)

Prompt the agent:

> Update `dashboard/src/types.ts` to match the new backend schemas. Make only additive changes — do not remove or rename anything already present:
>
> 1. Add `type ProductCategory = "food" | "beverage"`.
> 2. Add `type ProductSubType = "flexible_wrapper" | "rigid_can" | "rigid_box" | "transparent_bottle"`. Note: `rigid_box` is the correct term for all cuboidal cardboard boxes regardless of product type.
> 3. Add `type ContainerContents = "liquid" | "solid" | "powder"`.
> 4. Extend the `DefectClass` type (or equivalent string union) to include: `"fill_level_low"`, `"fill_level_high"`, `"cap_fitting_anomaly"`, `"surface_tear"`, `"surface_smudge"`, `"label_date_mismatch"`, `"label_barcode_mismatch"`.
> 5. Add `LabelQRStatus` interface if not already present: `{ qr_detected: boolean; qr_decoded: string | null; qr_expected: string | null; qr_matched: boolean; label_anomaly_types: string[] }`.
> 6. Add `LabelTextStatus` interface: `{ dates_verified: boolean; fields: Record<string, string>; anomaly_types: string[] }`.
> 7. Add to `InspectionResult`: `label_text: LabelTextStatus | null`, `product_category: ProductCategory | null`, `product_sub_type: ProductSubType | null`, `container_contents: ContainerContents | null`.
> 8. Add `Product` interface: `{ sku: string; name: string; description: string | null; product_category: ProductCategory; product_sub_type: ProductSubType; container_contents: ContainerContents; qr_code: string | null; expected_dates: Record<string, string>; sku_profile_name: string; __v: number }`.
> 9. Add `ProductionRun` interface: `{ run_id: string; sku: string; started_at: string; ended_at: string | null; status: "active" | "completed" | "aborted"; operator_id: string; inspection_count: number; defect_count: number }`.
> 10. Add a `DEFECT_CLASS_LABELS` constant of type `Record<string, string>` mapping every defect class to its display label. New entries: `fill_level_low → "Underfill"`, `fill_level_high → "Overfill"`, `cap_fitting_anomaly → "Cap Loose / Misaligned"`, `surface_tear → "Surface Tear"`, `surface_smudge → "Smudge / Stain"`, `label_date_mismatch → "Printed Date Mismatch"`, `label_barcode_mismatch → "Barcode Mismatch"`.
> 11. Add `PRODUCT_SUBTYPE_LABELS: Record<ProductSubType, string>` with values: `flexible_wrapper → "Flexible Wrapper (foil/plastic pouch)"`, `rigid_can → "Rigid Can (aluminium/steel)"`, `rigid_box → "Rigid Box (any cuboidal cardboard box)"`, `transparent_bottle → "Transparent Bottle (PET/glass)"`.

---

### Task 2-B: Product Registration Form (`dashboard/src/components/ProductRegistration.tsx`)

Prompt the agent:

> Create `dashboard/src/components/ProductRegistration.tsx`. This form allows operators to register a new product SKU. Place it as a panel in the Settings tab (accessible to `supervisor` and `admin` roles — use the existing role guard pattern from `App.tsx`).
>
> **Form fields (in order):**
>
> 1. **SKU** (text input, required): Validate on blur against `^[A-Z0-9_-]{3,32}$`. Transform input to uppercase automatically. Show inline error with the regex rule if invalid.
>
> 2. **Product Name** (text input, required, max 80 chars).
>
> 3. **Product Description** (textarea, optional, max 300 chars): Operator notes — e.g. "Lays Classic 50g, green crinkle pouch". This is metadata only; it does not affect inspection behaviour. Include a helper text: `"Optional. Describe the product for reference. This does not affect inspection settings."`.
>
> 4. **Product Category** (dropdown, required): Options: `Food`, `Beverage`. On change, reset sub-type and container contents fields to empty to prevent invalid combinations.
>
> 5. **Product Sub-Type** (dropdown, required): Disabled until category is selected. Filter options based on category:
>    - Food: `Flexible Wrapper (foil/plastic pouches)`, `Rigid Can (aluminium/steel)`, `Rigid Box (any cuboidal cardboard box)`
>    - Beverage: `Transparent Bottle (PET/glass)`
>    - Use `PRODUCT_SUBTYPE_LABELS` from `types.ts` for display text and the enum value (`flexible_wrapper` etc.) as the internal value.
>
> 6. **Container Contents** (dropdown, required): Options: `Liquid`, `Solid`, `Powder`. Auto-select and lock based on sub-type selection:
>    - `transparent_bottle` → lock to `Liquid` (cannot be overridden)
>    - `rigid_can` when category is `beverage` → pre-select `Liquid` (can override to `Solid` for food cans)
>    - All others → pre-select `Solid`, allow override to `Powder`
>    - Include tooltip: `"Container contents affect defect severity scoring — liquid containers receive higher damage severity weights."`
>
> 7. **SKU Profile** (dropdown, required): Populated by `GET /api/v1/products/sku-profiles`. Shows profile filename without `.yaml` extension. Add a tooltip showing the full filename on hover. Show a loading state while fetching.
>
> 8. **Expected QR Code Value** (text input, optional): The exact string encoded in the product's QR code. Used for barcode verification on the inspection line.
>
> 9. **Date Fields** (dynamic list, optional): A repeatable row group. Each row has:
>    - Field name (select): `mfg_date`, `expiry_date`, `best_before`, `packed_date`
>    - Expected value (text input): e.g. `"06/2026"`
>    - Expected format (select): `DD/MM/YYYY`, `MM/YYYY`, `DDMMMYYYY`, `YYYY-MM-DD`
>    - "Remove" button for each row. "Add Date Field" button below the list.
>
> **Submission:** Call `POST /api/v1/products`. On `201`: show success toast, reset form. On `409`: show error toast `"A product with this SKU already exists"`. On `422`: map field errors from the response to the appropriate form fields and show inline. Use the existing toast notification pattern from the dashboard.
>
> **Style:** Use existing Tailwind utility classes. Follow the card/form panel layout used elsewhere in the dashboard. Do not introduce new CSS classes or external component libraries.

---

### Task 2-C: Production Run Setup Component (`dashboard/src/components/RunSetup.tsx`)

Prompt the agent:

> Create `dashboard/src/components/RunSetup.tsx`. This component sits at the top of the main Inspections tab and is visible to `operator` role and above. It displays the current production run status and lets operators start and supervisors end a run.
>
> **When no run is active:**
> Show a compact banner: `"No active production run"` with a `"Start Run"` button (requires `operator` role — hide entirely for `viewer` role). On click, open a modal with:
> - SKU selector (searchable dropdown, populated from `GET /api/v1/products`, shows `SKU — Product Name`)
> - Operator name field (pre-filled from logged-in user if available, otherwise text input)
> - `"Start"` button → calls `POST /api/v1/runs`. On `201`: close modal, update run state. On `409`: show error `"A run for this SKU is already active"`.
>
> **When a run is active:**
> Show a persistent info bar at the top of the page with:
> - Product name and SKU (bold)
> - Sub-type badge using `PRODUCT_SUBTYPE_LABELS` (e.g. chip reading "Rigid Box")
> - Container contents badge (e.g. "Liquid" / "Solid")
> - Duration since run started (formatted as `Xh Ym`)
> - Inspection count and defect count as `Inspected: 1240 | Defects: 3`
> - `"End Run"` button: requires `supervisor` role — show it but disable with tooltip `"Supervisor required to end runs"` for `operator` role. On click, show confirmation dialog with two options: `Completed` and `Aborted`. Call `PATCH /api/v1/runs/{run_id}/end`. On success, clear run state.
>
> **State management:** Hoist the active run to the App-level React context or existing Zustand store so the live inspection feed and other tabs can show the product name alongside the SKU.
>
> **Auto-refresh:** Poll `GET /api/v1/runs/active` every 10 seconds to keep the display in sync if another operator starts or ends a run from a different browser session.

---

## Phase 3 — Integration Tests
**Depends on Phase 2. Also requires Collaborator A's Phase 2 for E2E pipeline tests. Tasks are parallel.**

---

### Task 3-A: Product and Run API Tests (`tests/integration/test_products_api.py`)

Prompt the agent:

> Create `tests/integration/test_products_api.py`. Use the existing FastAPI `TestClient` or `AsyncClient` pattern from `tests/integration/test_api.py`. Override the `get_db` FastAPI dependency with an in-memory MongoDB mock (use `mongomock-motor` or an AsyncMock fixture). Cover:
>
> 1. `POST /api/v1/products` success → 201 with correct fields including `__v: 0`.
> 2. `POST /api/v1/products` duplicate SKU → 409.
> 3. `POST /api/v1/products` lowercase SKU (fails regex) → 422 with field error on `sku`.
> 4. `POST /api/v1/products` non-existent `sku_profile_name` → 422 with message identifying the bad field.
> 5. `POST /api/v1/products` invalid `product_sub_type` value → 422.
> 6. `PATCH /api/v1/products/{sku}` with stale `__v` (0 where current is 1) → 409.
> 7. `POST /api/v1/runs` success → 201 with `run_id` UUID.
> 8. `POST /api/v1/runs` when a run is already active for that SKU → 409.
> 9. `GET /api/v1/runs/active` after starting a run → returns the run.
> 10. `PATCH /api/v1/runs/{run_id}/end` with `status: "completed"` → 200, `ended_at` is set, `status` is `"completed"`. Subsequent `POST /api/v1/runs` for same SKU → 201 (no conflict).

---

### Task 3-B: End-to-End Pipeline Routing Tests (`tests/integration/test_product_type_pipeline.py`)

Prompt the agent:

> Create `tests/integration/test_product_type_pipeline.py`. These tests exercise the full path from API request through to inspection result. Use `unittest.mock` to stub camera I/O and DB calls. Import Collaborator A's inference modules directly and verify routing behaviour.
>
> Cover:
> 1. Active run for a `rigid_box` SKU → `POST /inspect` handler resolves sub-type from run → pipeline confirms `FoodSurfaceInspector.inspect()` is called with `rigid_box`.
> 2. Active run for `transparent_bottle` → pipeline confirms `BeverageInspector.inspect()` is called.
> 3. No active run → inspection proceeds with default sub-type, `WARNING` is logged.
> 4. Correct QR, wrong printed expiry date → `result.label_text.anomaly_types` contains `"label_date_mismatch"`, verdict is `ESCALATE`.
> 5. Multiple barcodes in frame (QR + EAN13 on the same label) → only QR is evaluated, EAN13 is ignored, `WARNING` is logged containing both barcode types.
> 6. Transparent bottle underfill (fill ratio below `fill_level_min_ratio`) → `fill_level_low` in detections, verdict is `FAIL`.
> 7. Transparent bottle where fill level is undetectable (opaque packaging mock) → `fill_level_undetectable=True`, no fill level defect raised.
> 8. All scenarios: `result.latency_ms < 300`.
>
> Use `unittest.mock.patch` on `inference.pipeline.BarcodeVerifier.verify`, `inference.pipeline.LabelOCRVerifier.verify`, `inference.pipeline.FoodSurfaceInspector.inspect`, and `inference.pipeline.BeverageInspector.inspect` to control outputs without real camera or model inference.

---

## Phase 4 — API Latency Benchmarks
**Parallel with Phase 3. Depends only on Phase 2.**

### Task 4-A: API Endpoint Benchmarks

Prompt the agent:

> Add API endpoint benchmarks to `tests/benchmarks/test_module_latency.py` (the same file Collaborator A creates — coordinate to ensure both sets of benchmarks are in this file without overwriting each other's content). Use `pytest-benchmark` with FastAPI `TestClient`. Mock all database calls.
>
> Targets:
> - `POST /api/v1/products` — target: **<50ms** (mocked DB write)
> - `GET /api/v1/runs/active` — target: **<20ms** (mocked DB read)
> - `POST /api/v1/inspect` overhead from sub-type resolution (active run lookup + profile load) above base latency — target: **<50ms additional overhead**
>
> If any benchmark exceeds its target, document the bottleneck in a comment.

---

## Dependency Map

```
Phase 0 (all parallel — start immediately, no external dependencies)
  0-A (YAMLs)   0-B (DB schema)   0-C (repositories)   0-D (annotation docs)
      │               │                  │
      │               └──────────────────┘
      │                       │
      ▼                       ▼
Phase 1 (parallel — depends on Phase 0; 1-A/1-B/1-C also need Collab A Phase 0-A for enums)
  1-A (products API)   1-B (runs API)   1-C (inspection router)   1-D (register routers)
              │
              ▼
Phase 2 (parallel — depends on Phase 1; 2-A/2-B/2-C also need Collab A Phase 0-A)
  2-A (types.ts)   2-B (ProductRegistration.tsx)   2-C (RunSetup.tsx)
              │
              ▼
Phase 3 (parallel — depends on Phase 2; 3-B also needs Collab A Phase 2)
  3-A (product API tests)   3-B (pipeline E2E tests)
              │
              ▼
Phase 4 (depends on Phase 3)
  4-A (API latency benchmarks — coordinate with Collab A for shared benchmark file)
```

---

## Coordination Points with Collaborator A

This table summarises the cross-collaborator synchronisation events in order:

| Step | Action | Who waits | Who delivers |
|---|---|---|---|
| 1 | Finish Phase 0-A (YAML files) | Collab A (needs YAMLs for Phase 1 testing) | Collab B delivers |
| 2 | Finish Phase 0-C (repositories) | Collab A (needs `get_expected_qr` for module injection) | Collab B delivers |
| 3 | Finish Phase 0-A (schemas) | Collab B (needs enum values for `types.ts` and API validation) | Collab A delivers |
| 4 | Finish Phase 2-A (pipeline router) | Collab B (needs updated `inspect()` signature for E2E tests) | Collab A delivers |
| 5 | Write to shared benchmark file | Both | Coordinate — split into two functions in the same file |

**No shared file writes otherwise.** If uncertain, message your collaborator before touching any file not in your ownership table.

---

## Key Design Decisions

**Why product type is set at run start, not per frame:**
A single conveyor line runs one product SKU continuously per shift. There are no mixed products on the belt segment being inspected. Classifying product type per frame would add model overhead, introduce misclassification errors on transition frames, and solve a non-existent problem. The operator configures it once; the pipeline reads it from the active run on every frame.

**Why `rigid_box` is the precise sub-type name (not `carton`):**
`carton` is commonly understood as a liquid carton (milk, juice). This system's cuboidal cardboard category covers all rectangular cardboard packaging regardless of product — cereal, biscuits, pasta, tea, confectionery, medicines. `rigid_box` is unambiguous. The field label in the UI uses descriptive language ("Rigid Box (any cuboidal cardboard box)") for operator clarity while `rigid_box` is the internal value throughout the codebase.

**Why the form uses cascading dropdowns (category → sub-type → contents):**
A flat dropdown listing all sub-types would allow operators to select invalid combinations (e.g. `transparent_bottle` under `food`). Cascading dropdowns constrain choices to valid states and reduce form errors before they reach the API. Pre-selecting and locking `container_contents` for deterministic combinations (transparent bottle → liquid) eliminates a common misconfiguration.

**Why `container_contents` is a separate field from `product_sub_type`:**
A rigid can may hold liquid (beer, soft drink) or solid (chickpeas, tomato paste). The severity of packaging damage differs based on what is inside — a breached liquid container causes immediate line contamination; a breached solid container affects shelf life only. Separating contents type from packaging shape gives the severity scorer the information it needs without requiring product-type-specific logic at every decision point.

**Why the run setup polls every 10 seconds:**
WebSocket would be more efficient but adds connection management complexity for a low-frequency event (runs start and end rarely). 10-second polling has negligible load impact and eliminates a whole class of connection reliability issues for the embedded production environment where network conditions may be unstable.
