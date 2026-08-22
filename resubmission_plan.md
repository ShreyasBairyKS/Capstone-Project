# VisionFood QAI: Resubmission Plan

## Background

**Paper:** VisionFood QAI: A Deep Learning Based Quality Inspection System for Food and Beverage Products  
**Submitted to:** CSITSS 2026 (Track-4: Data Science, Analytics, and Intelligent Decision Systems)  
**Outcome:** Rejected — all 3 reviewers rated "Borderline" with High confidence  
**Reviewers:** #2, #3, #4

The paper was rejected not because the core idea is bad — all three reviewers praised the topic relevance, literature survey, and engineering effort. The rejections were unanimous around **experimental rigor gaps**, **overclaiming scope vs. what was built**, and **missing dataset transparency**.

---

## Summary of All Reviewer Concerns

### 🔴 Critical Issues (Will block acceptance anywhere)

| # | Issue | Reviewers |
|---|-------|-----------|
| C1 | **No held-out test set** — all metrics are validation-set numbers, cherry-picked from best epoch | #2, #3 |
| C2 | **Dataset not described** — no image counts, class splits, train/val/test sizes, annotation process | #2, #3, #4 |
| C3 | **Potential data leakage** — no confirmation that augmented/adjacent frames didn't bleed across splits | #3 |
| C4 | **System-level PASS/FAIL accuracy never reported** — paper claims ≥93% target but never measures it | #2, #3 |
| C5 | **Latency never measured** — paper claims <100ms on GPU target but zero FPS/ms benchmarks reported | #2, #4 |
| C6 | **Abstract/title overclaims** — dashboard, FastAPI, React, MongoDB, XAI, SKU routing all described as done but never evaluated | #2 |

### 🟠 Major Issues (Will weaken acceptance)

| # | Issue | Reviewers |
|---|-------|-----------|
| M1 | **Table 13 cherry-picks metrics** — precision misses (e.g. cap detector 0.869 vs 0.93 target) hidden from summary | #2 |
| M2 | **No fusion logic described** — YOLO + MobileNetV3 both predict same cap classes; no explanation of what wins on disagreement | #2 |
| M3 | **Multi-packaging claims not validated** — cans, wrappers, boxes never tested; only bottle pipeline evaluated | #2, #3 |
| M4 | **MobileNetV3 86.5% accuracy too low for industrial use** — ~13.5% error rate on high-speed line is risky | #4 |
| M5 | **Literature claims lack verifiable methodology** — "4% of studies are beverage-focused" and "no prior system integrates XAI + remediation" need reproducible search protocol | #3 |
| M6 | **Table 5 mAP target anchored to Liu et al. but Liu et al. never reports mAP** — circular justification | #2 |
| M7 | **No per-class AP, confusion matrices** — aggregate metrics hide per-class weaknesses | #3 |
| M8 | **Monte Carlo Dropout triage logic never explained** — how are uncertain predictions handled in practice? | #4 |

### 🟡 Minor Issues (Easy to fix)

| # | Issue | Reviewers |
|---|-------|-----------|
| S1 | Reference [18] ("Implicit bias of linear equivariant networks", ICML 2022) is never cited and irrelevant — remove it | #2 |
| S2 | Abstract says "16 recent studies" but Tables 1–4 list 17 references — fix the count | #2 |

---

## Proposed Resubmission Strategy

> [!IMPORTANT]
> The core system is solid. The rejection is almost entirely about **experimental gap**, not concept gap. This is fixable with targeted experiments and honest scoping of claims.

### Phase 1: Restructure Claims (Week 1)

**Goal:** Align abstract/introduction/title with what was actually built and tested.

1. **Rewrite abstract** — remove future-tense claims for dashboard, XAI, SKU routing. Clearly state: *"Three CV submodules are implemented and evaluated: (1) cap defect detection, (2) cap quality classification, (3) fill-level estimation."*
2. **Add a "System Scope" section** — explicitly acknowledge that multi-packaging support (cans, wrappers, boxes) is an architectural design for future work.
3. **Reframe the paper** — instead of "VisionFood QAI: A Full System", position it as *"VisionFood QAI: A Deep Learning Pipeline for Bottle Cap and Fill-Level Quality Inspection with Uncertainty Quantification"* — accurate and still strong.
4. **Fix Table 13** — add all metrics (including precision misses). Honest reporting is a strength, not a weakness (Reviewer #2 explicitly said so).
5. **Fix reference [18]** — remove it.
6. **Fix abstract count** — "16" → "17" or re-count carefully.
7. **Fix Table 5** — don't anchor the mAP target to Liu et al. if Liu et al. doesn't report mAP. Use a different justification.

---

### Phase 2: Add Missing Experiments (Week 2–3)

**Goal:** Fill every critical experimental gap identified by reviewers.

#### 2a. Create a proper train/val/test split
- Split data **by production run or physical bottle instance**, NOT randomly by image.
- Document: total images, images per class, unique products/bottles, split sizes.
- Confirm no augmented images bleed across splits.

#### 2b. Run final evaluation on a **held-out test set**
- Re-run all three modules on the withheld test set.
- Report test-set numbers as the primary result (validation numbers can go in ablation).

#### 2c. Report full metrics per module
- Per-class AP, precision, recall, F1, confusion matrix for cap detector (YOLOv11s).
- Per-class precision, recall, F1, confusion matrix for cap classifier (MobileNetV3).
- Per-class metrics for water-surface detector (YOLOv8).

#### 2d. Measure system-level PASS/FAIL accuracy
- Define a product-level evaluation: given a bottle, does the system correctly classify it as PASS/FAIL?
- Report false-accept rate (FAR) and false-reject rate (FRR) — these are the metrics industrial QC actually uses.

#### 2e. Measure latency
- Run inference on GPU (and ideally edge device if available).
- Report: FPS, per-image inference time (ms), memory usage.
- If ONNX/TensorRT is benchmarked, include those numbers.

#### 2f. Describe fusion logic
- Explicitly document: when YOLO and MobileNetV3 disagree on cap class, what is the decision rule?
- If there's no fusion yet, state it clearly and treat module outputs as separate.

#### 2g. Describe MC Dropout inference
- How many forward passes? What threshold triggers a "low confidence" flag? How does the triage workflow change for uncertain predictions?

---

### Phase 3: Improve MobileNetV3 Accuracy (Optional but Recommended)

**Goal:** Close the gap between 86.5% achieved and the industrial requirement.

Options to try:
- [ ] Data augmentation tuning (more aggressive if dataset is small).
- [ ] Try EfficientNet-B0 or MobileNetV2 as alternative classifiers.
- [ ] Ensemble MobileNetV3 predictions with YOLO class scores.
- [ ] Re-examine class balance — if defective-cap images are rare, SMOTE or oversampling may help.
- [ ] Fine-tune Focal Loss gamma parameter.

---

### Phase 4: Strengthen Literature Review (Week 1)

**Goal:** Make literature claims reproducible and defensible.

- Add a PRISMA-style or explicit search protocol: databases searched (Scopus, IEEE Xplore, Google Scholar), keywords, date range, inclusion/exclusion criteria.
- Back the "4% beverage studies" claim with a clearly countable subset of the search results.
- Tone down "no prior system integrates XAI + remediation" to "to the best of our knowledge based on [search]..."

---

## Target Journal Recommendations

> [!NOTE]
> Since all reviewers rated "Borderline" at a **conference**, a **journal** submission gives you more space to address the shortcomings thoroughly. Journals also allow longer experimental sections.

| Journal | Fit | Impact Factor | Notes |
|---------|-----|--------------|-------|
| **IEEE Access** | ⭐⭐⭐⭐⭐ | ~3.4 | Open access, fast review, applied engineering work, no page limit — perfect for adding all missing experiments and detailed dataset section |
| **Expert Systems with Applications (Elsevier)** | ⭐⭐⭐⭐ | ~8.5 | Strong for AI + industry applications; XAI/remediation angle fits well |
| **Computers & Electronics in Agriculture** | ⭐⭐⭐ | ~8.3 | Food QC focus, but agriculture-skewed; beverage framing may not fit |
| **Journal of Food Engineering (Elsevier)** | ⭐⭐⭐ | ~5.3 | Food processing focus, but lower CS depth expected |
| **Engineering Applications of AI (Elsevier)** | ⭐⭐⭐⭐ | ~8.0 | Good for applied ML systems; edge AI angle is a plus |

> [!TIP]
> **Recommended first target: IEEE Access.** It's open access, has fast turnaround (~4–6 weeks first decision), and gives you unlimited space to properly document dataset, experiments, and ablations. The "applied systems" positioning is exactly what IEEE Access values.

---

## Resubmission Checklist

### Must-Have Before Submitting Anywhere
- [ ] Rewrite abstract — scope accurately to what's built
- [ ] Add dataset section: image counts, class balance, split sizes, annotation protocol
- [ ] Hold-out test set created (split by product instance, not image)
- [ ] All final metrics reported on **test set**, not validation best-epoch
- [ ] Full metrics table (no cherry-picking) with confusion matrices
- [ ] System-level PASS/FAIL FAR/FRR reported
- [ ] Latency benchmarks (FPS + ms on target hardware)
- [ ] Fusion logic for YOLO + MobileNetV3 cap class disagreement documented
- [ ] MC Dropout triage logic documented (pass count, threshold, triage action)
- [ ] Multi-packaging scope clearly marked as future work
- [ ] Reference [18] removed
- [ ] Literature count fixed (16 vs 17)
- [ ] Table 5 mAP target justification fixed
- [ ] Literature search methodology added (PRISMA or equivalent)

### Nice-to-Have
- [ ] Improved MobileNetV3 accuracy (target ≥90%)
- [ ] Edge device latency (Jetson Nano / Raspberry Pi)
- [ ] At least one dataset made partially public (even a small sample)
- [ ] Screenshots of dashboard (even if not fully evaluated)

---

## Timeline Estimate

| Week | Work |
|------|------|
| 1 | Restructure paper: abstract, scope, Table 13, minor fixes |
| 2 | Dataset documentation + proper train/val/test split |
| 3 | Re-run experiments on test set, collect full metrics, measure latency |
| 4 | Model improvements (MobileNetV3), fusion logic, MC Dropout documentation |
| 5 | Final paper write-up, formatting for IEEE Access |
| 6 | Submission |
