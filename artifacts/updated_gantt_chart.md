# VisionFood QAI — Updated Gantt Chart and Progress Record

**Status date:** 27 October 2026
**Calendar convention:** `active` = work performed; `done` = evidenced in repository/PDF; `crit` = dependency-critical.
**Baseline:** Phase 1 plan (March–May 2026), which envisaged a broad food-and-beverage platform with a Phase-1 review in May.
**Actual basis:** Git commits across all branches from 21 March to 26 September 2026, the initial review, the paper-review response, audited dataset/model artefacts, and Himalaya branch records.

```mermaid
gantt
    title VisionFood QAI — planned vs actual (Mar–Oct 2026)
    dateFormat  YYYY-MM-DD
    axisFormat  %b

    section Phase 1 baseline plan
    Scope, architecture, literature review          :done, p1, 2026-03-21, 2026-04-10
    Platform/API/dashboard foundation               :done, p2, after p1, 2026-05-05
    Bottle-cap detector and classifier baseline     :done, p3, 2026-04-21, 2026-05-05
    Fill-level module and integrated demo           :done, p4, 2026-05-03, 2026-05-15
    Broad SKU, REMEDY, XAI and edge validation      :crit, p5, 2026-04-13, 2026-05-22
    Phase-1 report/review                           :milestone, done, p6, 2026-05-22, 0d

    section Actual implementation and adjustment
    Bottleneck review and evidence-gap identification :done, a1, 2026-05-23, 2026-06-20
    Himalaya label: PatchCore/ROI feasibility         :done, a2, 2026-06-20, 2026-07-21
    Himalaya ROI redesign, slicing and YOLO11m        :done, crit, a3, 2026-07-22, 2026-07-28
    EfficientAD training, grayscale experiment        :done, a4, 2026-07-26, 2026-08-07
    Paper review response and scope correction        :done, a5, 2026-08-22, 2026-08-22
    Cap model retraining, thresholds and evaluation   :done, crit, a6, 2026-09-04, 2026-09-16
    Distilled YOLO11m-seg and dashboard pipeline      :done, crit, a7, 2026-09-05, 2026-09-16
    Himalaya ONNX export and equivalence checks       :done, a8, 2026-09-18, 2026-09-26
    Final evidence consolidation and thesis artifacts :active, a9, 2026-10-01, 2026-10-31
```

## Planned versus actual progress

| ID | Workstream | Phase-1 plan | Actual progress / evidence | Status at 27 Oct | Dependency |
|---|---|---|---|---|---|
| P1 | Scope, architecture and literature survey | 21 Mar–10 Apr | Scaffold and documentation committed 21 Mar; Phase-1 review completed | Complete | — |
| P2 | API, database and dashboard foundation | 11 Apr–5 May | API/database work 17 Apr; frontend branch and live inspection integration 2–15 May | Complete prototype | P1 |
| P3 | Bottle-cap detection and first cap classifier | 21 Apr–5 May | Detector experiments, two-pass crop pipeline and MobileNet baseline completed Apr–May | Superseded/improved | P1, cap dataset |
| P4 | Fill-level estimator | 3–15 May | Dataset prep, YOLO fill model and frontend/backend integration recorded 3–15 May | Complete prototype | P1, fill data |
| P5 | Broad product coverage, REMEDY, XAI and edge validation | By 22 May | Designed/partly implemented but not validated end-to-end; reviewers found scope overclaiming | Descoped from paper claims | P2–P4 |
| A1 | Cap-classifier recovery and prototype artefacts | Not in Phase-1 baseline | Checkpoints and ONNX exports retained; inspected export exposes three outputs, but no traceable held-out benchmark package is available | Prototype; reproducible benchmark deferred | P3, expanded dataset |
| A2 | Edge export/profiling | Planned but unmeasured | ONNX artefacts exist, but no traceable timing environment/output record is available | Deferred pending target-hardware protocol | A1 |
| A2a | Distilled cap inspection and visual dashboard | Not in Phase-1 baseline | September branch: YOLO11m-seg student artefact, Docker pipeline, FastAPI/WebSocket service, visual overlays, multi-camera dashboard, KPI and audit-replay UI | Complete prototype; model result is validation evidence | Cap dataset; platform foundation |
| A3 | Himalaya label detection | Not in Phase-1 research scope | Separate four-ROI pipeline: SAHI + YOLO11m localisation and EfficientAD anomaly models; export package and ROI-4 limitation preserved | Prototype; performance table requires traceable evaluation package | New label dataset |
| A4 | ONNX validation for label models | Not in Phase-1 baseline | ROI 1–4 ONNX export manifest; map differences approximately 0.99e-6–5.72e-6 | Complete | A3 |
| A5 | Final report/thesis and paper evidence | May review deliverable | In progress; research paper is restricted to bottle-cap and fill-level experiments; Himalaya is thesis/report only | Active | A1–A4 |

## Dependency table

| Successor | Must follow | Why |
|---|---|---|
| Bottle-cap detector evaluation | dataset/annotation package | Labels and split protocol must exist before metrics are defensible. |
| Cap classifier benchmark | cap crops, non-overlapping splits, imbalance plan | Test metrics and class-aware measures depend on them. |
| ONNX/INT8 profiling | selected classifier | Export only follows model selection and frozen preprocessing. |
| Fill-level validation | water-surface model and fill-ratio thresholds | The decision cannot be evaluated without localisation and SKU rule. |
| Himalaya EfficientAD models | ROI localisation/crops | Each anomaly model consumes a stable ROI crop. |
| Himalaya ONNX equivalence | trained EfficientAD checkpoints | Export validation compares against frozen PyTorch outputs. |
| Dashboard visualisation | inference/API output schema and audit records | Live feed, KPI charts, manual inspection, and defect replay depend on stable inspection events. |
| Integrated demonstration | validated module outputs and API/dashboard | A demo is meaningful only after module interfaces are stable. |
| Final report/paper | evaluation protocol, evidence and scope decision | Prevents claims that cannot be supported by experiments. |

## Risks, mitigations and adjustment evidence

| Risk / issue | Impact | Mitigation and adjustment | Evidence |
|---|---|---|---|
| Scope exceeded available validated data | Claims could not be defended | Reframed research scope to the three bottle modules; broad packaging/system claims moved to future work | Conference reviews; `resubmission_plan.md` |
| Validation metrics used as final metrics; possible leakage | Inflated performance and rejection risk | Explicitly label current model values as validation-only; require group-aware held-out protocol, frozen checkpoint, scripts and outputs before publication | Reviewer response plan; audited dataset/model inventory |
| Cap-classifier benchmark provenance is incomplete | Accuracy and latency claims cannot be defended | Retain checkpoint/export as prototype; remove comparative benchmark from final report/paper and define a reproducible evaluation work package | Audited three-output ONNX interface; final evidence-bound documents |
| Small caps missed at full-image scale | False negatives | Bottle-guided two-pass zoom; later hard-negative/threshold work | Apr–May pipeline commits; Sep threshold/evaluation commits |
| Himalaya ROI 4 produced many false positives | Unreliable all-ROI label verdict | Record ROI 4 as a limitation; prioritise re-annotation/retraining rather than overclaiming | `himalaya_label_detection/PROJECT_CONTEXT.md` |
| Standard inference shrank 8000px label detail | ROI detection failures | Applied SAHI slicing with YOLO11m ROI detector | Jul branch commits and architecture record |
| Windows/CUDA/ONNX runtime incompatibilities | Delayed reproducible inference | Added compatibility fixes, CPU fallback and direct PyTorch validation; later verified ONNX numerical agreement | Jul–Sep commits; grayscale ONNX manifest |
| Dashboard visuals could be mistaken for validated operational performance | Overstated system maturity | Classify the dashboard as a functional prototype; report features and integration evidence without inferring factory accuracy or operator benefit | May frontend/integration commits; September Docker/WebSocket dashboard pipeline |

## Submission notes

- Show this chart as the **updated schedule**, not as a claim that all Phase-1 plans were delivered exactly as first proposed.
- In the final report, explicitly label the Himalaya stream as an additional thesis/report module. Do not use it to support the research-paper contribution or merge its metrics with bottle results.
- Before submission, replace the `active` final-consolidation bar with actual completion dates, have the three owners confirm accountability mapping, record a target venue/submission status, and attach an approved-tool similarity report at or below 10% (excluding references and standard phrases).
