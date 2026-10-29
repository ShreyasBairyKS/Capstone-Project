# Response to Reviewers — CSITSS 2026, Paper 1204

We thank the reviewers. The revised paper is narrower in scope: it covers only the two stations that are built and measured, cap inspection and fill level. The cap model is now a YOLO11m-seg student distilled from a YOLO11x-seg teacher. Every number was measured on held-out test splits inside the deployment container. The scripts and raw JSON outputs are in `evidence/`.

## Reviewer #2

| # | Comment | Change |
|---|---|---|
| 1 | Abstract sells unevaluated components (SKU routing, REMEDY, XAI, dashboard, MongoDB). | Removed. The paper covers only Station 1 (cap) and Station 2 (fill level), and states that other packaging types, label checks, and UI features are out of scope (Sec. I). |
| 2 | Own targets (≥93% pass/fail accuracy, <100 ms GPU) never reported. | Image-level PASS/REJECT accuracy with false-accept/false-reject rates is in Table V. Network, stage, and closed-loop latency are in Tables VIII–IX. Both are listed against the targets in Table X. |
| 3 | Summary table cherry-picked. | Table X lists every target, including the misses: bottle and meniscus AP@0.5:0.95, meniscus P/R, inference-stage p95, fill-service p99, and CPU latency. |
| 4 | YOLO + MobileNetV3 fusion asserted, not evaluated. | Removed. Station 1 is a single distilled segmentation model plus a documented gate (Sec. IV). |
| 5 | Validation-only metrics, best-epoch selection on the same set. | All headline results are on the untouched test split (354 images), for both FP32 and the deployed FP16 engine. The model was also checked on images of our own bottles outside the dataset. |
| 6 | Four packaging types claimed, only bottles tested. | Scope is now transparent PET bottles only. |
| 7 | Irrelevant ref [18] (equivariant networks). | Removed. |
| 8 | "16 studies" vs 17 refs. | The count claim was removed together with the survey section. |
| 9 | mAP target anchored to Liu et al., who report no mAP. | Removed. Targets are reported as project targets, without that justification. |
| 10 | No dataset size, balance, or annotation details. | Table I gives per-split, per-class counts and source photographs. Sec. V-A gives the fill datasets. Augmentation and single-annotator labelling are stated. |

## Reviewer #3

| Comment | Change |
|---|---|
| Near-perfect mAP needs justification. | The dataset is fully characterised (Table I, capture resolution, object-size distribution). Per-class mask/box AP for FP32 and FP16 is in Table IV, and the teacher–student comparison on the same split is in Table II. Product-level error rates with exact confidence bounds are in Table V, and a threshold-margin analysis is in Fig. 4. |
| Image/class/split sizes, test vs validation, per-class AP, confusion. | Tables I, II, IV, and V. Top-1 results: 354/354 on test, 710/712 on validation (both errors misplaced→open, still REJECT). |
| Split by session/batch. | A capture-session-grouped held-out benchmark is identified as the next evaluation step (Sec. VIII). |
| Multi-product claims are architectural only. | Removed. |
| Literature claims (4%, "no prior system") need evidence. | Removed. The related work now describes cited works without novelty claims. |
| No system-level PASS/FAIL, false-accept/false-reject rates. | Table V. Fig. 4 maps thresholds to review load. |
| No code or reproducibility package; annotation protocol. | Scripts and outputs are in `evidence/`. Single-annotator labelling is stated. |

## Reviewer #4

| Comment | Change |
|---|---|
| MobileNetV3-Small at 86.5% accuracy. | Replaced by the distilled YOLO11m-seg (Tables II and IV). Table III keeps the earlier classifier benchmark only as model-selection evidence. |
| No FPS, ms, or power figures. | Per-stage latency for PyTorch, ONNX Runtime, and TensorRT, plus closed-loop stress tests (Tables VIII–IX). Power was not measured, and the paper does not claim it. |
| How MC-Dropout uncertainty is handled. | MC Dropout is not part of the deployed system. Uncertainty is handled by the gate's UNCERTAIN band (Sec. IV-E, Alg. 1), which routes to review without ejection. |
| Dataset scale and acquisition conditions. | Given in Sec. IV-A. |
