# Citation Audit Report: Phase 2 College Report (`phase2_report/main.tex`)

- Date: 2026-11-10
- Mode: ARS `academic-paper` citation-check (citation_compliance_agent)
- Scope: `main.tex` plus the 8 `\input` chapters and `chapters/references.tex`
- Style detected: numbered `[N]`, IEEE-like, manual `thebibliography` in the `report` class
- What was done: report only. **No changes made in this pass.** The six reference fixes shared with the research paper were applied earlier the same day and are counted below as already fixed.
- How metadata was checked: Crossref API (title and DOI lookups, plus abstracts where Crossref has them), PubMed Central, and publisher pages.
- Not checked: Retraction Watch; the Roboflow dataset pages (HTTP 403); full texts behind paywalls (IEEE: Baokaew, Kavitha, Subramanian, Krishnan).

---

## Summary

| Metric | Result |
|---|---|
| Citation instances in text | 91 |
| Distinct cited keys | 53 |
| Reference list entries | 53 |
| Orphan in-text citations | **0** |
| Orphan references | **0** |
| Duplicate `\bibitem` keys | 0 |
| Numbering vs order of first citation | **All 53 in order** |
| Undefined citations in `main.log` | 0 |
| References with wrong metadata (still open) | 0 wrong; **4 incomplete** (§3) |
| References with wrong metadata (already fixed today) | 6 (`chatchapol2024`, `xiong2024`, `farhangi2024`, `liu2025`, `liakos2025`, `banus2021`) + `jacob2018` venue |
| Claim–source problems | **1 overstated** (§1.1), 3 weak or unverified (§2) |
| Self-citation ratio | 0% |
| Sources from 2022–2026 | 21/53 (40%) |
| Sources older than 10 years | 7: `cohen1960`, `clopper1934`, `lewis1995`, `brunelli2009`, `lin2014coco`, `hinton2015`, `romero2015`. All are seminal or method sources, so they are acceptable. |
| DOIs given | 0/53 (consistent across the list; see §4) |

**Bottom line:** the cross-referencing is clean. Every source the report cites exists, and Crossref matched every reference it indexes. After today's fixes, no reference has wrong bibliographic data. What is left is **one sentence in Chapter 1 that attributes a claim to Liakos *et al.* that the review does not make**, two weaker attributions, and a few incomplete entries.

---

## 1. Claim–source problems (fix)

### 1.1 `ch1_introduction.tex:12`: Liakos is cited for something it doesn't say
> "…yet published systems rarely report what matters to a plant: the rate at which defective products are passed, the rate at which good products are discarded, and the end-to-end latency of the deployed system~\cite{liakos2025}."

The Liakos *et al.* review (*Foods* 14(19):3424) does **not** say that studies rarely report these quantities. In its section 2.7 it *recommends* reporting false-reject and false-accept rates, alarm frequency, and throughput for industrial deployment, and it lists latency per item as a packaging metric. Its stated open problems are domain-coverage bias, data scarcity, under-explored unsupervised methods, and integration with legacy systems. The "rarely report" part is the authors' own reading of the literature, which Chapter 2 (§2.1) argues study by study.

**Suggested rewording:**
> …yet a recent review recommends that industrial systems report false-accept and false-reject rates and throughput~\cite{liakos2025}, and most published inspectors still report validation accuracy alone (Section~\ref{sec:lit-food}).

This keeps the point and attaches to Liakos only what Liakos says.

---

## 2. Weak or unverified attributions (author check)

| # | Location | Claim | Finding |
|---|---|---|---|
| 2.1 | `ch1:12` | "inspectors examining reflective plastic or glossy print lose consistency over a shift~\cite{banus2021}" | Banús's abstract says packages "are still evaluated by human operators" and that automation improves efficiency. It does not mention fatigue or loss of consistency over a shift, and it is about thermoformed trays, not reflective plastic or glossy print. **Weak.** Soften to "…and manual inspection limits efficiency and coverage~\cite{banus2021}", or cite a human-factors source for the fatigue claim. |
| 2.2 | `ch2:12`, Table 2.1 | Baokaew *et al.*: "CNN on an embedded GPU", "classify the whole image and do not localise the liquid surface" | The IEEE abstract mentions a camera module, image processing, and conveyor-speed control, and reports 98.89% level accuracy and 100% lid-seal accuracy. **"Embedded GPU" and "no localisation" are not in the abstract.** Check them against the full text, or drop "embedded GPU". |
| 2.3 | `ch2:14` | Kavitha *et al.* "coupled YOLO detection to Arduino-driven sorting" | The paper exists (GCAT 2024, DOI 10.1109/GCAT62922.2024.10924069), but neither Crossref nor a web search shows its abstract, so **"Arduino" is unverified**. Watch for the similarly titled 2022 *Sustainability* paper by Park *et al.*, which uses a single-board computer; make sure the detail isn't taken from that one. |
| 2.4 | `ch2:14` | Nguyen *et al.* "integrated YOLOv8 with a PLC… only Nguyen *et al.* measure the latency of the full loop" | The abstract confirms YOLOv8, multi-threading, and above 90% accuracy. **"PLC" and "full-loop latency" are not in the abstract**; check the full text. |
| 2.5 | `ch2:10`, Table 2.1 | Farhangi *et al.*: "edge detection, and pattern matching" | The abstract says "distance determination and pattern matching", and confirms the CMOS camera and 95.6% (100/95/92%). "Edge detection" is likely but not stated; "distance measures and pattern matching" matches the source exactly. **Minor.** |

Verified as supported by abstract or full text:
- **Domain papers:** Liu (CBAM + ASFF, 7,400 images, accuracy 0.96, recall 0.94), Banús (FAR and FRR on the line; "classification only"), Liakos (beverages a small minority; data scarcity and legacy integration), Ribeiro (pharmaceutical bottles, multi-view, incremental learning), Cao (RFID/AIoT, so the "sensor-heavy" grouping is correct), Truong (title confirms defects and contamination).
- **Distillation:** Wang (full-map imitation hurts), Chen (hint + logit + bounded regression for two-stage detectors), Zagoruyko (attention transfer), Hinton, Romero.
- **Detection:** YOLACT, Mask R-CNN.
- **Anomaly detection:** PatchCore, EfficientAD, MVTec AD, SAHI, StyleGAN2-ADA.
- **Statistics:** Clopper–Pearson. The formula 1 − 0.025^{1/n} is correct for zero events; n=153 gives 2.4%.

### 2.6 `ch6_results.tex:32`: COCO size rule
"every instance is 'large' by the COCO definition ($\geq96^2$ pixels)". COCO defines large as area **> 96²**. Change `\geq` to `>`, the same fix already made in the paper.

### 2.7 Distillation section: no conflict with the project directive, but a notation clash
`ch5_implementation.tex:65–77` presents the general form (logit + box + feature) with citations, then says "We therefore set α=β=0 and distil features only". That matches the directive that the loss is feature-only and the logit term is not used. **Notation:** α is both the logit-term weight in Eq. (5.x, distill-general) and the ramped feature weight α(t) in Eq. (kd). Consider renaming the general-form weights (e.g., λ_cls, λ_box, λ_feat) so that "α=β=0" can't be read as switching off the feature ramp.

---

## 3. Incomplete reference entries (minor, auto-correctable)

| Key | Missing | Verified value (Crossref) |
|---|---|---|
| `kavitha2024` | Author names are garbled: "K. K. S, M. C. G, and R. T. V" | **K. S. Kavitha, C. G. Mamatha, and T. V. Rohini**; GCAT 2024, pp. 1–6 |
| `cao2025` | Article number | *Electronics*, vol. 15, no. 1, **art. no. 106**, 2025 (issued 25 Dec 2025) |
| `ribeiro2025` | Issue and article number | *J. Imaging*, vol. 11, **no. 10, art. no. 350**, 2025 |
| `subramanian2025` | Pages | ICCRTEE 2025, **pp. 1–6** |
| `truong2024` | Uses "p." for an article number | write `art. no. 100758` |
| `fillds1`, `fillds2` | No year | add the year shown on each Roboflow page (not verifiable here) |
| `jocher2023`, `jocher2024`, `onnxruntime` | No version | ver. 8.0.0 / 11.0.0 / the ORT version used |

**Kavitha entry, corrected:**
```latex
\bibitem{kavitha2024}
K. S. Kavitha, C. G. Mamatha, and T. V. Rohini,
``Deep learning based product defect detection for sustainable smart manufacturing,''
in \emph{Proc. 5th IEEE Global Conf. Advancement in Technology (GCAT)}, 2024, pp.~1--6.
```

---

## 4. Consistency notes (advisory)

- **No DOIs anywhere.** This is consistent within the report, and acceptable for a college thesis unless the BMSCE template asks for them. If DOIs are wanted, the 19 listed in the paper's audit apply to the shared references, and the report-only references have these:

| Key | DOI |
|---|---|
| subramanian2025 | 10.1109/ICCRTEE64519.2025.11052928 |
| nguyen2025 | 10.1142/S2196888825400032 |
| kavitha2024 | 10.1109/GCAT62922.2024.10924069 |
| krishnan2024 | 10.1109/TQCEBT59414.2024.10545305 |
| cao2025 | 10.3390/electronics15010106 |
| du2025 | 10.1016/j.procs.2025.05.031 |
| truong2024 | 10.1016/j.crfs.2024.100758 |
| ribeiro2025 | 10.3390/jimaging11100350 |
| redmon2016 | 10.1109/CVPR.2016.91 |
| he2017maskrcnn | 10.1109/ICCV.2017.322 |
| bergmann2019 | 10.1109/CVPR.2019.00982 |
| roth2022 | 10.1109/CVPR52688.2022.01392 |
| batzner2024 | 10.1109/WACV57701.2024.00020 |
| akcay2022 | 10.1109/ICIP46576.2022.9897283 |
| akyon2022 | 10.1109/ICIP46576.2022.9897990 |

- **Page ranges** follow the CVF open-access versions (e.g., MVTec AD 9592–9600, PatchCore 14318–14328, EfficientAD 128–138, Mask R-CNN 2961–2969). This is consistent within the report. They differ from the IEEE Xplore pages, so if DOIs are added, switch to the Xplore pages as was done in the paper.
- **The paper and the report now differ** on page ranges and DOIs for the shared references. That is fine for separate documents.
- **`akcay2022`** (Anomalib, ICIP 2022) is cited for "reference implementations of both" PatchCore and EfficientAD. EfficientAD (2023/24) was added to the library after the 2022 paper. That is acceptable when citing the library, but "the Anomalib library~\cite{akcay2022}" reads more accurately than suggesting the 2022 paper describes EfficientAD.
- **`micikevicius2018`** (ch2 §2.2.3): "stores weights and activations in 16-bit…". The method keeps an **FP32 master copy** of the weights. Consider "computes in 16-bit floating point while keeping an FP32 master copy of the weights and numerically sensitive operations in 32 bits".

---

## 5. Action checklist (priority order)

1. [ ] Reword the `ch1:12` Liakos sentence (§1.1).
2. [ ] Soften or re-source the `ch1:12` inspector-fatigue claim on Banús (§2.1).
3. [ ] Check the full texts for Baokaew ("embedded GPU", "no localisation"), Kavitha ("Arduino"), and Nguyen ("PLC", "full-loop latency"); drop whatever cannot be confirmed (§2.2–2.4).
4. [ ] Fix the `kavitha2024` author names; complete `cao2025`, `ribeiro2025`, `subramanian2025`, `truong2024` (§3).
5. [ ] `ch6:32`: change `\geq96^2` to `>96^2`.
6. [ ] Optional: rename the general-form KD weights to avoid the α clash (§2.7); clarify the Anomalib and mixed-precision wording (§4); add DOIs and versions.
7. [ ] Confirm the Roboflow dataset years by hand; run Retraction Watch screening by hand.

### Applied (2026-11-10)
Items 1–6 have been applied. Where full texts could not be checked (§2.2–2.4), the unconfirmed details were **removed**, leaving only what the abstracts support:
- **ch1:12.** The Liakos sentence now says what the review says, plus the authors' own observation, with a pointer to §2.1. The Banús fatigue clause was replaced with "many packaging lines still rely on human operators for this step, which limits both efficiency and coverage".
- **ch2.**
  - Farhangi: "distance measures, pattern matching", in the text and Table 2.1.
  - Baokaew: "camera, image processing, and a CNN, with conveyor-speed control… a level-detection accuracy rather than a measured fill ratio". In Table 2.1 the method is now "Image processing, CNN" and the gap "No measured fill ratio". The gap-table row is now "Pass/fail fill checking".
  - Nguyen: "multi-threaded data processing for real-time surface inspection", and "of them only Nguyen *et al.* address real-time operation of the full system". Table 2.1 now says "YOLOv8, multi-threading".
  - Kavitha: "applied deep-learning defect detection to smart manufacturing".
  - Mixed-precision wording now mentions the FP32 master weights; the Anomalib sentence now says "in its current releases".
- **ch5.** The general-form KD weights were renamed to $w_{\text{cls}}, w_{\text{box}}, w_{\text{feat}}$, and the text now says "$w_{\text{cls}}=w_{\text{box}}=0$ … with the feature weight $\alpha(t)$". The loss is unchanged and still feature-only.
- **ch6:32.** COCO rule changed to `>96^2`.
- **references.tex.** Kavitha author names fixed; Cao art. no. 106; Ribeiro no. 10, art. no. 350; Subramanian pp. 1–6; Truong art. no. 100758; versions added for YOLOv8, YOLO11, and ONNX Runtime.

Verification: 91 citations, 53 distinct, 53 entries, 0 orphans, all in order. pdflatex (TinyTeX) builds 80 pages, the same as before, with no errors and no undefined citations or references.

Still open: Roboflow dataset years and Retraction Watch screening (item 7), both by hand. DOIs were not added, to stay consistent with the rest of the report.

### Sources consulted
- Crossref REST API (`api.crossref.org/works`) for every reference except the ICLR/NeurIPS/ICML papers, arXiv, the books, and the web sources
- [Liakos et al., PMC12523314](https://pmc.ncbi.nlm.nih.gov/articles/PMC12523314/)
- [Liu et al., PMC12021230](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12021230/)
- [Banús et al., PMC8576017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8576017/)
- [Baokaew et al., IEEE Xplore 10773275](https://ieeexplore.ieee.org/document/10773275/)
- [Park et al. 2022 (similar title to Kavitha), MDPI Sustainability 14(5):2697](https://mdpi.com/2071-1050/14/5/2697)
