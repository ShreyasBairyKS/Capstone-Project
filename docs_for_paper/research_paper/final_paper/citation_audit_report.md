# Citation Audit Report: `main.tex` (VisionQAI, IEEEtran)

- Date: 2026-11-10
- Mode: ARS `academic-paper` citation-check (citation_compliance_agent)
- Style detected: IEEE, numbered `[N]`, manual `thebibliography`
- What was done: no auto-corrections were applied to `main.tex`. Every fix is listed below for the authors to approve.
- How metadata was checked: Crossref API lookups by DOI and title, plus PubMed Central, IEEE Xplore, and publisher pages through web search.
- Not checked: Retraction Watch screening, and the three Roboflow dataset pages (they returned HTTP 403 to automated fetch).

---

## Summary

| Metric | Count |
|---|---|
| Distinct cited keys in text | 32 |
| Reference list entries | 32 |
| Orphan in-text citations (no reference) | 0 |
| Orphan references (never cited) | 0 |
| Numbering out of order of first citation | 2 entries misplaced (shifts [13]–[16] and [31]–[32]) |
| References with wrong bibliographic data (title, venue, authors, or pages) | 5 (`chatchapol2024`, `xiong2024`, `liakos2025`, `farhangi2024`, `liu2025`) |
| Claim–source conflicts | 1 major (`banus2021` contradicts a claim in Related Work), 3 unverified |
| Uncited factual claims | 2 (opening sentence of the Introduction; Clopper–Pearson) |
| Missing DOIs (sources that have one) | 25 of 32 (no entry gives a DOI) |
| Self-citation ratio | 0% |
| Sources from the last 5 years (2022–2026) | 9/32 (28%) |
| Sources older than 10 years | 4 (`cohen1960`, `he2016`, `hinton2015`, `romero2015`), all seminal, so acceptable |

**Bottom line:** the cross-referencing is clean (no orphans). But **5 of the 7 domain references have incorrect metadata**. One of them (`chatchapol2024`) uses the authors' given names as surnames, which also makes the in-text "Chatchapol *et al.*" wrong. One (`xiong2024`) gives the wrong title and the wrong journal. A Related Work claim is contradicted by its own cited source (`banus2021`). These are the problems a reviewer is most likely to catch. For CSITSS Paper 1204, a resubmission, they should be fixed first.

---

## 1. Critical: wrong bibliographic data

### 1.1 `chatchapol2024`: author names inverted, title and pages wrong
| Field | In `main.tex` | Verified (Crossref, DOI 10.23919/ICCAS63016.2024.10773275) |
|---|---|---|
| Authors | B. Chatchapol, W. Autanan, W. Anan | **Chatchapol Baokaew, Autanan Wannachai, Anan Wongjan**, i.e., C. Baokaew, A. Wannachai, A. Wongjan |
| Title | "…using image processing and CNN" | "…Using Image Processing **(CNN)**" |
| Pages | pp. 1–6 (placeholder) | **pp. 1434–1439** |

Each Thai given name was treated as a surname. The in-text narrative at line 54, "Chatchapol \textit{et al.}", must become **"Baokaew \textit{et al.}"**. The citation key itself can stay as it is.

**Corrected:**
```latex
\bibitem{chatchapol2024} C. Baokaew, A. Wannachai, and A. Wongjan, ``Liquid filling quality control system for beverage industry using image processing (CNN),'' in \textit{Proc. 24th Int. Conf. Control, Automation and Systems (ICCAS)}, Jeju, Korea, 2024, pp. 1434--1439, doi: 10.23919/ICCAS63016.2024.10773275.
```

### 1.2 `xiong2024`: wrong title and wrong journal
| Field | In `main.tex` | Verified (Crossref, DOI 10.1007/s10696-023-09523-9) |
|---|---|---|
| Title | "Computer vision-based quality control for food packaging using data-centric AI" | **"Designing a computer-vision-based artifact for automated quality control: a case study in the food industry"** |
| Venue | *Annals of Operations Research* | ***Flexible Services and Manufacturing Journal***, vol. 36, no. 4, pp. 1422–1449 |

No paper by these authors in *Annals of Operations Research* could be found. The cited title looks like a paraphrase of the real paper's topic. The claim it supports, "data quality matters as much as model choice", fits the real paper's data-centric AI framing, so the claim can stay once the reference is fixed.

**Corrected:**
```latex
\bibitem{xiong2024} F. Xiong, N. Kühl, and M. Stauder, ``Designing a computer-vision-based artifact for automated quality control: A case study in the food industry,'' \textit{Flexible Services and Manufacturing J.}, vol. 36, no. 4, pp. 1422--1449, 2024, doi: 10.1007/s10696-023-09523-9.
```

### 1.3 `liakos2025`: wrong title
| Field | In `main.tex` | Verified (DOI 10.3390/foods14193424) |
|---|---|---|
| Title | "Machine learning for food quality control: A systematic review" | **"Machine learning for quality control in the food industry: A review"** |
| Issue/article | missing | no. 19, art. no. 3424 |

The claim it supports is verified: beverages make up 4% of the reviewed studies (1 of 25), so "a small minority" is accurate.

```latex
\bibitem{liakos2025} K. G. Liakos, V. Athanasiadis, E. Bozinou, and S. I. Lalas, ``Machine learning for quality control in the food industry: A review,'' \textit{Foods}, vol. 14, no. 19, art. no. 3424, 2025, doi: 10.3390/foods14193424.
```

### 1.4 `farhangi2024`: wrong journal name, no volume or pages
| Field | In `main.tex` | Verified (DOI 10.37256/ccds.5220244756) |
|---|---|---|
| Journal | *Cluster Computing and Data Science* | ***Cloud Computing and Data Science*** (Universal Wiser Publisher) |
| Vol/pages | missing | pp. 242–254, published 25 Jun 2024. The vol. 5, no. 2 is read from the DOI pattern and should be confirmed on the article page. |

```latex
\bibitem{farhangi2024} O. Farhangi, E. Sheidaee, and A. Kisalaei, ``Machine vision for detecting defects in liquid bottles: An industrial application for food and packaging sector,'' \textit{Cloud Computing and Data Science}, vol. 5, no. 2, pp. 242--254, 2024, doi: 10.37256/ccds.5220244756.
```

### 1.5 `liu2025`: incomplete
Authors, title, and year are verified. The issue and article number are missing: vol. 20, no. 4, **e0321971**, DOI 10.1371/journal.pone.0321971.

```latex
\bibitem{liu2025} G. Liu, S. Zhang, L. Wang, X. Li, and G. Li, ``Research on mechanical automatic food packaging defect detection model based on improved YOLOv5 algorithm,'' \textit{PLOS ONE}, vol. 20, no. 4, art. no. e0321971, 2025, doi: 10.1371/journal.pone.0321971.
```

---

## 2. Critical: a claim is contradicted by its own source

**Line 54 (Related Work):** "*None of these works reports product-level false-accept and false-reject rates or end-to-end deployment latency.*"

`banus2021` (Banús *et al.*, *Sci. Rep.* 11:21887) **does report product-level error rates**. For its best model (DenseNet161) it gives false-positive rates of 0.03–0.30%, false-negative rates of 0–0.07%, and a production rejection rate of 0.64–5.09%. It was also validated inline on a real production line at production cadence. The sentence is therefore false for one of the five works it lists, and any reviewer who opens [1] will see this.

The same weakness affects **line 42 (Introduction)**: "*published food-packaging inspectors often report only validation-set accuracy, without product-level error rates…*" cites `liu2025` and `chatchapol2024`. That sentence is defensible for those two papers, but it sits next to `banus2021`, which counters it.

**Suggested rewording (Related Work, line 54):**
> Banús *et al.*~\cite{banus2021} compare CNN backbones for thermoformed food packages and report inline false-positive and false-negative rates on a production line, but for sealing defects on trays rather than bottle closures or fill level. Apart from~\cite{banus2021}, these works report image-level accuracy without product-level error rates, and none reports end-to-end deployment latency for a multi-view bottle station.

The authors need to decide the exact framing. The correction above keeps the paper's point, that no prior work gives product-level FA/FR plus latency *for bottles*, without misrepresenting [1].

---

## 3. Numbering out of order of appearance (IEEE requires first-citation order)

`thebibliography` numbers entries in the order they are listed, not the order they are cited. Two entries are misplaced:

| Key | First cited (text order) | Current number | Fix |
|---|---|---|---|
| `chicco2020` | 13th (Related Work, right after `cohen1960`) | **[32]** | Move the `\bibitem` to directly after `cohen1960` |
| `jacob2018` | 31st (Section VI, after `tensorrt`) | **[16]** | Move the `\bibitem` to directly after `tensorrt` |

As a result, the text currently jumps [12] → [32] → [13], and [16] first appears after [30]. Once both entries are moved, all 32 numbers run in order of first citation. This was verified with a script that parses `\cite` order against `\bibitem` order.

---

## 4. Claim support: unverified or weak (flagged for author review)

| # | Location | Claim | Status |
|---|---|---|---|
| 4.1 | L42 | "Closure and fill-level defects are the two most frequent visual rejects on beverage bottling lines." | **No citation.** It is a strong empirical claim. Add a source, or soften to "are among the most common visual rejects". |
| 4.2 | L42 | "Manual inspection does not scale to line rates of several bottles per second~\cite{banus2021}" | **Weak.** Banús covers thermoformed trays and says inline 100% inspection must keep up with production cadence. It does not give a bottles-per-second rate. Acceptable as general support, but consider "does not scale to inline 100% inspection". |
| 4.3 | L54 | Baokaew *et al.* classify fill levels "with a CNN on an embedded GPU but do not localise the liquid surface" | **Unverified.** The abstract mentions a camera module, image processing, and conveyor-speed control, and reports 98.89% liquid-level accuracy and 100% lid-sealing accuracy. Neither "embedded GPU" nor "no localisation" could be confirmed from the abstract. Also note: the paper checks **capping** too, which is close to your Station 1. Check the full text. |
| 4.4 | L54 | Farhangi *et al.* use "rule-based edge detection and pattern matching" | **Unverified** (full text blocked). The abstract shows the paper also covers **cap and seal-ring defects and label placement**, so it is the closest prior work to your two-station system. Reviewers may expect it to be discussed more directly. |
| 4.5 | L54 | Liu *et al.* "detect five packaging defects with an attention-augmented YOLOv5" | Partly verified: improved YOLOv5 with ASFF fusion and CBAM comparisons. The count of five defect classes was not confirmed. |
| 4.6 | L118 | Every instance is "large" by COCO ($\geq96^2$ px) | COCO defines *large* as area **> 96²**. Change `\geq` to `>`. |
| 4.7 | L284 | FP16 inference cited to `micikevicius2018` | That paper is about mixed-precision **training**. It works as FP16 background, but `tensorrt` is the real support for FP16 inference. Acceptable as is. |
| 4.8 | L319 | Clopper–Pearson exact bounds | **No citation.** Optional but conventional: C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," *Biometrika*, vol. 26, no. 4, pp. 404–413, 1934, doi: 10.1093/biomet/26.4.404. |

Verified as supported: `liakos2025` (beverages 4%), `xiong2024` (data-centric AI, once the reference is fixed), `hinton2015`, `romero2015`, `wang2019`, `bolya2019`, `cohen1960`, `chicco2020`, `buda2018`, `cui2019`, `lin2017`, `zhang2018`, `yun2019`, and the classifier backbones in Table III.

---

## 5. IEEE format issues (minor, auto-correctable)

| # | Entry/area | Issue | Fix |
|---|---|---|---|
| 5.1 | All journal/conference entries | **No DOIs.** IEEE expects them. 25 of 32 entries have one. | Add `doi: …` (list in §6) |
| 5.2 | `banus2021` | Missing article number | add `art. no. 21887` |
| 5.3 | `jacob2018` | Venue named "IEEE Conf." | 2018 proceedings are **"IEEE/CVF Conf. Computer Vision and Pattern Recognition"**. Also, `he2016` is correctly "IEEE Conf." |
| 5.4 | `capdata`, `filldata1`, `filldata2` | Dataset entries: no creator for capdata, **no year** for filldata1/2, no access date | IEEE dataset form: `Owner, "Title," Roboflow Universe, ver. N, Year. [Online]. Available: URL (accessed Mon. DD, YYYY).` The page owners could not be verified because of 403. Confirm workspace names, versions, and CC BY 4.0 by hand. |
| 5.5 | `yolo11`, `yolov8`, `tensorrt`, `onnxruntime`, datasets | Web sources lack "Accessed:" dates | add an access date, e.g., `Accessed: Nov. 10, 2026` |
| 5.6 | `yolo11`, `yolov8`, `onnxruntime` | No version | IEEE software form: add `ver. 11.0.0`, `ver. 8.0.0`, and the ORT version you used (1.31.1 per §VI) |
| 5.7 | All URLs | Typed as plain text, with `\_` escapes in `capdata` | Load `\usepackage{url}` (or `hyperref`) and wrap URLs in `\url{…}` so they break and hyphenate correctly |
| 5.8 | `hinton2015` | arXiv only | Fine. Optionally add "presented at NIPS Deep Learning Workshop, 2014". |
| 5.9 | Page ranges | CVF open-access pages are used (e.g., YOLACT 9157–9166, ShuffleNet V2 116–131). IEEE Xplore/LNCS pages differ (9156–9165; 122–138). | Pick one source consistently. If you add DOIs, use the DOI-resolving (Xplore/LNCS) pages. |
| 5.10 | `\begin{thebibliography}{00}` | Fine (two-digit labels) | none |

---

## 6. DOIs to add (all checked against Crossref)

| Key | DOI |
|---|---|
| banus2021 | 10.1038/s41598-021-01254-x |
| liu2025 | 10.1371/journal.pone.0321971 |
| chatchapol2024 | 10.23919/ICCAS63016.2024.10773275 |
| liakos2025 | 10.3390/foods14193424 |
| farhangi2024 | 10.37256/ccds.5220244756 |
| xiong2024 | 10.1007/s10696-023-09523-9 |
| bolya2019 | 10.1109/ICCV.2019.00925 |
| buda2018 | 10.1016/j.neunet.2018.07.011 |
| cui2019 | 10.1109/CVPR.2019.00949 |
| lin2017 | 10.1109/ICCV.2017.324 |
| cohen1960 | 10.1177/001316446002000104 |
| yun2019 | 10.1109/ICCV.2019.00612 |
| jacob2018 | 10.1109/CVPR.2018.00286 |
| wang2019 | 10.1109/CVPR.2019.00507 |
| coco | 10.1007/978-3-319-10602-1_48 |
| howard2019 | 10.1109/ICCV.2019.00140 |
| ma2018 | 10.1007/978-3-030-01264-9_8 |
| he2016 | 10.1109/CVPR.2016.90 |
| chicco2020 | 10.1186/s12864-019-6413-7 |

The ICLR and ICML papers (`zhang2018`, `romero2015`, `micikevicius2018`, `tan2021`, `touvron2021`) and the arXiv preprint `hinton2015` have no publisher DOI. Leaving them without one is correct.

---

## 7. Coverage observations (advisory, not errors)

- **The distillation related work stops at 2019** (`hinton2015`, `romero2015`, `wang2019`). Your loss is MSE on channel-wise L2-normalised features. That is close to recent detector-KD work that normalises features before imitation, such as Cao *et al.*, PKD (NeurIPS 2022), and Yang *et al.*, Focal and Global KD (CVPR 2022). Citing one or two recent detector or segmentation KD papers would preempt a "not novel / not positioned" comment. The KD claims themselves are untouched, per the authors' directive.
- **No citation for YOLO11's C3k2/SPPF/C2PSA modules** beyond the Ultralytics repository. That is acceptable, since no paper exists for YOLO11.
- Only 28% of sources are from the last 5 years. For an applied CV venue this is fine, because the older sources are foundational.

---

## 8. Action checklist (priority order)

1. [ ] Fix `chatchapol2024` authors, title, and pages, and change "Chatchapol *et al.*" to "Baokaew *et al.*" (L54).
2. [ ] Replace `xiong2024` with the real title and journal (*Flexible Services and Manufacturing J.*).
3. [ ] Rewrite the "None of these works reports product-level…" sentence (L54) so it no longer contradicts `banus2021`.
4. [ ] Fix the `liakos2025` title, the `farhangi2024` journal name and pages, and the `liu2025` issue and article number.
5. [ ] Move `chicco2020` after `cohen1960` and `jacob2018` after `tensorrt` to restore IEEE order.
6. [ ] Add a citation for, or soften, the opening sentence (L42). Optionally cite Clopper–Pearson.
7. [ ] Check the Baokaew and Farhangi full texts against the L54 descriptions (embedded GPU, no surface localisation, edge detection).
8. [ ] Add DOIs (§6), access dates, and software versions; wrap URLs in `\url{}`; change `\geq96^2` to `>96^2`.
9. [ ] Confirm the Roboflow dataset owners, versions, and years by hand (automated fetch was blocked).
10. [ ] Run Retraction Watch screening by hand (not performed here).

### Applied to `main.tex` (2026-11-10)
Items 1–6 and 8 have been applied, except access dates. Those were left out because the dataset pages could not be opened, and writing a date the authors never accessed them would be false. The changes:
- `chatchapol2024`, `xiong2024`, `liakos2025`, `farhangi2024`, `liu2025` corrected; "Baokaew *et al.*" now used in the text
- The `banus2021` sentence rewritten
- `chicco2020` and `jacob2018` reordered
- DOIs added to 19 entries
- `jacob2018` venue changed to IEEE/CVF
- Pages switched to the DOI (Xplore/LNCS) ranges
- Software versions added
- `\url{}` and `\usepackage{url}` added
- COCO `>96^2`
- Opening sentence softened to "among the most common"
- Clopper–Pearson cited as new reference [33]

Verification: the cross-check script reports 33 citations and 33 entries, 0 orphans, all in order. TinyTeX pdflatex builds cleanly (no errors, undefined citations, or overfull boxes), but the output is **8 pages** against 7 before. References [26]–[33] spill onto page 8, and they still spill without DOIs (6 entries).

The authors accepted the 8-page length (2026-11-10).

Still open: items 7, 9, 10.

### Sources consulted
- Crossref REST API (`api.crossref.org/works/<doi>`) for every DOI in §6 and for the title search that found the ICCAS paper
- [Liu et al., PMC12021230](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12021230/)
- [Liakos et al., PMC12523314](https://pmc.ncbi.nlm.nih.gov/articles/PMC12523314/)
- [Banús et al., PMC8576017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8576017/)
- [Baokaew et al., IEEE Xplore 10773275](https://ieeexplore.ieee.org/document/10773275/)
- [Farhangi et al., Wiserpub CCDS 4756](https://ojs.wiserpub.com/index.php/CCDS/article/view/4756)
- [Xiong et al., Springer s10696-023-09523-9](https://link.springer.com/article/10.1007/s10696-023-09523-9)
