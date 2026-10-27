# VisionFood QAI — Final Work Breakdown Structure

**Version:** Final-phase update, 27 October 2026
**Planning basis:** Phase 1 review plan; Git history (21 March–26 September 2026); final cap-classifier experiment study; Himalaya-label branch documentation.

## Scope statement

The WBS covers the complete final capstone deliverable: an evidence-bounded visual-inspection prototype and its evidence package. Its four inspection deliverables are bottle-cap detection, cap-classifier prototype work, fill-level estimation, and Himalaya-label defect detection. It also includes an operator dashboard for visualisation, live monitoring, manual inspection, and audit review. The label stream is a **report/thesis deliverable only** and is not part of the research-paper claims. Broad multi-package inspection, production-line deployment, and unmeasured REMEDY/XAI claims are excluded unless supported by final evidence.

The lowest-level items are work packages. Each is mutually exclusive, has one accountable owner, and is estimated at **8–80 hours**, satisfying the 8/80 rule. Estimates are effort hours, not elapsed calendar time.

| WBS | Deliverable / work package | Owner | Effort (h) | Acceptance evidence |
|---|---|---:|---:|---|
| **1.0** | **VisionFood QAI final capstone deliverable** | Ullas N | — | Accepted integrated prototype and evidence package |
| **1.1** | **Project governance and scope-control package** | Ullas N | — | Approved scope, schedule, risk and traceability records |
| 1.1.1 | Phase-1 baseline and final-scope record | Ullas N | 16 | Baseline-to-actual scope table and approved exclusions |
| 1.1.2 | Experiment and version-control record | Shreyas Bairy K S | 16 | Commit/branch map, dataset manifests and reproducibility notes |
| 1.1.3 | Risk, issue and change-control record | Ullas N | 12 | Dated risk register with mitigation evidence |
| **1.2** | **Bottle-cap detection module** | Sushanth S | — | Defect-localisation model and evaluation evidence |
| 1.2.1 | Cap-detection dataset and annotation set | Shreyas Bairy K S | 40 | Class taxonomy, split manifest, labels and QA record |
| 1.2.2 | YOLO detector training artifact | Sushanth S | 48 | Reproducible configuration, weights and training curves |
| 1.2.3 | Two-pass cap-localisation pipeline | Sushanth S | 32 | Full-frame/bottle-guided inference implementation and test outputs |
| 1.2.4 | Detector threshold and error-analysis report | Sushanth S | 24 | Threshold sweep, confusion/error cases, and student-model validation record |
| 1.2.5 | Distilled YOLO11m-seg cap-inspection artifact | Sushanth S | 64 | Six-class teacher--student model, validation curves, masks, and ONNX/TensorRT export path |
| **1.3** | **Cap-quality classifier prototype and reproducibility module** | Sushanth S | — | Checkpoint/export artefacts and a future-test protocol |
| 1.3.1 | Cap-crop provenance and class-map package | Shreyas Bairy K S | 48 | Source-data manifest, class map and group-aware split plan |
| 1.3.2 | Classifier checkpoint and export package | Sushanth S | 40 | Saved model, ONNX interface record and preprocessing specification |
| 1.3.3 | Reproducible held-out evaluation package | Sushanth S | 64 | Frozen checkpoint, test script, confusion matrix and class-aware metrics |
| 1.3.4 | Classifier error-analysis package | Sushanth S | 32 | Reviewed failure cases and acceptance/rejection decision record |
| 1.3.5 | Target-hardware timing package | Sushanth S | 32 | Reproducible timing protocol, environment record and measured outputs |
| 1.3.6 | Saliency and interpretation package | Shreyas Bairy K S | 24 | Classifier saliency examples with limitations statement |
| **1.4** | **Fill-level estimation module** | Sushanth S | — | Water-surface detector, fill-ratio decision and validation evidence |
| 1.4.1 | Fill-level dataset and surface annotations | Shreyas Bairy K S | 32 | Bottle/surface labels, split manifest and annotation QA |
| 1.4.2 | Water-surface detection artifact | Sushanth S | 40 | YOLO training configuration, weights and evaluation outputs |
| 1.4.3 | Fill-ratio computation and SKU threshold package | Sushanth S | 24 | Formula implementation, underfill/normal/overfill cases |
| 1.4.4 | Fill-level validation and error-case report | Shreyas Bairy K S | 24 | Reproduced validation output, boundary cases and limitations |
| **1.5** | **Himalaya-label defect-detection module (thesis/report only)** | Shreyas Bairy K S | — | Four-ROI anomaly-inspection prototype; excluded from paper claims |
| 1.5.1 | Himalaya image, ROI and annotation package | Shreyas Bairy K S | 48 | Source inventory, ROI definitions and defect annotations |
| 1.5.2 | SAHI-enabled YOLO ROI-localisation artifact | Sushanth S | 40 | Sliced-data configuration, YOLO11m weights and ROI localisation outputs |
| 1.5.3 | EfficientAD ROI grayscale anomaly models | Shreyas Bairy K S | 64 | Checkpoints, thresholds, heatmaps and evaluation-package requirements |
| 1.5.4 | ROI 4 remediation decision package | Shreyas Bairy K S | 24 | Re-annotation issue record, limitations statement and retraining plan |
| 1.5.5 | ONNX export and equivalence-validation package | Sushanth S | 24 | Export manifest, SHA-256 values and PyTorch/ONNX difference checks |
| **1.6** | **Integrated application and deployment package** | Ullas N | — | Demonstrable inspection workflow with traceable outputs |
| 1.6.1 | Inference orchestration and decision interface | Ullas N | 40 | API contract, module routing and PASS/FAIL/REVIEW output schema |
| 1.6.2 | Backend persistence and audit package | Ullas N | 40 | API routes, inspection records, authentication/audit tests |
| 1.6.3 | Operator dashboard and visualisation package | Ullas N | 48 | Live multi-camera feed, KPI cards/charts, manual upload, inspection history, defect replay, settings and report UI |
| 1.6.4 | Containerisation and reproducible runtime package | Ullas N | 24 | Docker/runtime configuration and setup guide |
| 1.6.5 | End-to-end demonstration and latency evidence | Ullas N | 32 | Demo scenarios, timing record and known-limitations log |
| **1.7** | **Final evaluation and documentation package** | Ullas N | — | Honest, traceable final artefacts |
| 1.7.1 | Test protocol and metrics package | Shreyas Bairy K S | 32 | Split protocol, metrics definitions and leakage controls |
| 1.7.2 | Results, comparison and limitations package | Sushanth S | 40 | Consolidated test table, ablations, failure analysis and scope statement |
| 1.7.3 | Final thesis/report evidence package | Ullas N | 48 | Architecture, all four modules, screenshots and appendices |
| 1.7.4 | Research-paper evidence and submission package | Sushanth S | 40 | Paper-only scope, traceability index, target-venue record and similarity report |
| 1.7.5 | Final presentation and demonstration package | Ullas N | 24 | Slide deck, scripted demo and artifact index |

## WBS integrity checks

- **100% rule:** Sections 1.1–1.7 collectively contain every in-scope final deliverable. No item outside the stated scope is included.
- **No overlap:** 1.2 localises cap defects; 1.3 classifies cap quality; 1.4 measures fill level; 1.5 inspects Himalaya labels. Data/evidence ownership is assigned once per module.
- **Scope adjustment:** Dashboard/API and deployment are documented as prototype integration. Claims of unverified multi-package production capability, production-line performance, or Himalaya results in the research paper are not included.
