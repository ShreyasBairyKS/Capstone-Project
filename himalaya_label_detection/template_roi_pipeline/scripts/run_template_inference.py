r"""
run_template_inference.py
-----------------------------------------------------------------------------
Run the template-matching ROI pipeline on full NSC scans:

    scan -> grayscale -> template matcher (4 ROIs) -> per-ROI ONNX classifier
         -> aggregated PASS / FAIL

Examples
    # Inspect scans (files and/or folders of .bmp)
    python himalaya_label_detection/template_roi_pipeline/scripts/run_template_inference.py \
        --images "../NSC/NSC GOOD IMAGES/1.bmp" "../NSC/NSC BAD IMAGES" \
        --save-annotated results/annotated --json-out results/template_run.json

    # Labelled evaluation (image-level confusion matrix)
    python himalaya_label_detection/template_roi_pipeline/scripts/run_template_inference.py \
        --good "../NSC/NSC GOOD IMAGES" --bad "../NSC/NSC BAD IMAGES"

    # Calibrate thresholds from labelled scans -> config/roi_onnx_thresholds.json
    python himalaya_label_detection/template_roi_pipeline/scripts/run_template_inference.py \
        --good "../NSC/NSC GOOD IMAGES" --bad "../NSC/NSC BAD IMAGES" --calibrate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]          # himalaya_label_detection/template_roi_pipeline
REPO_ROOT = PKG_ROOT.parents[1]                         # Capstone-Project (holds models/)
sys.path.insert(0, str(PKG_ROOT.parent))

from template_roi_pipeline.roi_template_matcher import ROI_NAMES, load_gray  # noqa: E402
from template_roi_pipeline.inspector import (  # noqa: E402
    InspectionResult,
    ONNXROIClassifier,
    TemplateROIInspector,
)

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GREEN, RED, AMBER = (0, 180, 0), (0, 0, 220), (0, 165, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="*", type=Path, default=[], help="scan files / folders to inspect (unlabelled)")
    p.add_argument("--good", type=Path, help="folder of known-good scans (labelled evaluation / calibration)")
    p.add_argument("--bad", type=Path, help="folder of known-bad scans (labelled evaluation / calibration)")
    p.add_argument("--config-dir", type=Path, default=PKG_ROOT / "config")
    p.add_argument("--models-root", type=Path, default=REPO_ROOT / "models" / "rois",
                   help="folder with ROI_N/weights/model.onnx (use models/gray_scale_rois for the grayscale-trained set)")
    p.add_argument("--input-mode", choices=("gray", "color"), default="gray",
                   help="what the classifiers are fed. gray = grayscale-trained models (default); "
                        "color = the color-trained models currently in models/rois")
    p.add_argument("--score-stat", choices=ONNXROIClassifier.SCORE_STATS, default="mean",
                   help="statistic of the anomaly map used as the ROI score (mean = deployment contract)")
    p.add_argument("--thresholds", type=Path, default=None, help="default: <config-dir>/roi_onnx_thresholds.json")
    p.add_argument("--calibrate", action="store_true", help="fit per-ROI thresholds on --good/--bad and write them")
    p.add_argument("--target-recall", type=float, default=None,
                   help="calibrate for this image-level recall instead of max F1 (e.g. 0.95)")
    p.add_argument("--min-match-score", type=float, default=None, help="template score needed to accept a ROI")
    p.add_argument("--allow-missing", action="store_true", help="do not fail the label when a ROI is not located")
    p.add_argument("--json-out", type=Path, help="write per-image results as JSON")
    p.add_argument("--csv-out", type=Path, help="write per-image ROI scores as CSV")
    p.add_argument("--save-annotated", type=Path, help="folder for downscaled annotated overview images")
    return p.parse_args()


def collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += [f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return sorted(out, key=lambda f: (f.parent.name, int(f.stem) if f.stem.isdigit() else 0, f.name))


def annotate(gray: np.ndarray, result: InspectionResult, scale: float = 0.2) -> np.ndarray:
    canvas = cv2.cvtColor(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                          cv2.COLOR_GRAY2BGR)
    for name, v in result.rois.items():
        color = {"PASS": GREEN, "FAIL": RED, "MISSING": AMBER}[v.status]
        x1, y1, x2, y2 = (int(c * scale) for c in v.match.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {v.status}" + ("" if v.score is None else f" {v.score:.3f}")
        cv2.putText(canvas, label, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    banner = GREEN if result.passed else RED
    cv2.rectangle(canvas, (0, 0), (110, 28), banner, -1)
    cv2.putText(canvas, result.verdict, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def print_result(path: Path, r: InspectionResult, label: str | None) -> None:
    cells = "  ".join(
        f"{n[-1]}:{v.status[0]}{'' if v.score is None else f'{v.score:.3f}'}" for n, v in r.rois.items()
    )
    tag = f" [{label}]" if label else ""
    print(f"{path.name:>10}{tag:<7} {r.verdict}  {cells}  ({r.latency_ms:.0f} ms)")


def build_inspector(args: argparse.Namespace, profile: dict) -> TemplateROIInspector:
    """profile = the parsed thresholds file: besides the thresholds it names per-ROI model folders / input modes."""
    return TemplateROIInspector.from_profile(
        profile, args.config_dir, args.models_root, args.min_match_score,
        fail_on_missing=not args.allow_missing, input_mode=args.input_mode, score_stat=args.score_stat)


def load_scan(path: Path, color: bool) -> np.ndarray:
    """BGR scan when any ROI takes color crops, else a gray scan (the matcher converts to gray itself)."""
    if not color:
        return load_gray(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def fit_thresholds(scores: dict[str, tuple[np.ndarray, np.ndarray]], target_recall: float | None) -> dict[str, float]:
    """
    Per-ROI thresholds so the AGGREGATED image verdict (FAIL if any ROI > its
    threshold) is best on the labelled scans.

    Bad scans are only known bad at image level (a defect may be in any one ROI),
    so thresholds are fitted jointly by coordinate descent on image-level F1
    (or, with target_recall, the highest thresholds keeping that recall).
    """
    good = np.stack([scores[n][0] for n in ROI_NAMES], axis=1)  # [n_good, 4]
    bad = np.stack([scores[n][1] for n in ROI_NAMES], axis=1)   # [n_bad, 4]

    def metrics(th: np.ndarray) -> tuple[float, float, int, int]:
        tp = int(np.any(bad > th, axis=1).sum())
        fp = int(np.any(good > th, axis=1).sum())
        rec = tp / len(bad)
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return f1, rec, tp, fp

    # candidate thresholds per ROI: midpoints between all observed scores
    cands = []
    for i in range(4):
        v = np.unique(np.concatenate([good[:, i], bad[:, i]]))
        cands.append(np.concatenate([[v[0] - 1e-9], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-9]]))

    th = np.array([c[-1] for c in cands])  # start: everything passes
    for _ in range(6):
        changed = False
        for i in range(4):
            best_key, best_val = None, th[i]
            for c in cands[i]:
                trial = th.copy(); trial[i] = c
                f1, rec, tp, fp = metrics(trial)
                if target_recall is not None:
                    key = (rec >= target_recall, -fp, c)          # meet recall, fewest FP, then loosest
                else:
                    key = (f1, c)                                  # max F1, then loosest
                if best_key is None or key > best_key:
                    best_key, best_val = key, c
            if best_val != th[i]:
                th[i], changed = best_val, True
        if not changed:
            break
    return {n: float(th[i]) for i, n in enumerate(ROI_NAMES)}


def main() -> int:
    args = parse_args()
    labelled = [(p, "good") for p in collect([args.good])] if args.good else []
    labelled += [(p, "bad") for p in collect([args.bad])] if args.bad else []
    jobs = labelled + [(p, None) for p in collect(args.images)]
    if not jobs:
        raise SystemExit("Nothing to do: pass --images and/or --good/--bad")
    if args.calibrate and not (args.good and args.bad):
        raise SystemExit("--calibrate needs both --good and --bad")

    thr_path = args.thresholds or args.config_dir / "roi_onnx_thresholds.json"
    # An existing file may name per-ROI model folders / input modes ("model_dirs", "input_modes"); calibrating
    # keeps them, so the thresholds are fitted for the models that profile actually uses.
    old_doc = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.is_file() else None
    if args.calibrate:
        profile = {**(old_doc or {}), "thresholds": {n: float("inf") for n in ROI_NAMES}}
    else:
        if old_doc is None:
            raise SystemExit(f"{thr_path} not found. Run once with --calibrate on labelled scans.")
        profile = old_doc
        fitted = (profile.get("input_mode"), profile.get("score_stat"))
        if fitted != (args.input_mode, args.score_stat):
            raise SystemExit(f"{thr_path} was fitted for input_mode/score_stat={fitted}, but this run uses "
                             f"({args.input_mode}, {args.score_stat}). Re-run with --calibrate or pass --thresholds.")

    inspector = build_inspector(args, profile)
    if args.save_annotated:
        args.save_annotated.mkdir(parents=True, exist_ok=True)

    records, csv_rows = [], []
    for path, label in jobs:
        scan = load_scan(path, inspector.input_mode == "color")
        result = inspector.inspect(scan, str(path))
        gray = scan if scan.ndim == 2 else cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        records.append((path, label, result))
        if not args.calibrate:
            print_result(path, result, label)
        row = {"image": path.name, "label": label or "", "verdict": result.verdict}
        for n, v in result.rois.items():
            row[f"{n}_score"] = "" if v.score is None else f"{v.score:.6f}"
            row[f"{n}_status"] = v.status
            row[f"{n}_match"] = f"{v.match.score:.4f}"
        csv_rows.append(row)
        if args.save_annotated and not args.calibrate:
            cv2.imwrite(str(args.save_annotated / f"{path.parent.name}_{path.stem}.png"), annotate(gray, result))

    if args.calibrate:
        scores = {
            n: (
                np.array([r.rois[n].score for _, l, r in records if l == "good" and r.rois[n].score is not None]),
                np.array([r.rois[n].score for _, l, r in records if l == "bad" and r.rois[n].score is not None]),
            )
            for n in ROI_NAMES
        }
        for n in ROI_NAMES:
            g, b = scores[n]
            print(f"{n}: good n={len(g)} median={np.median(g):.4f} max={g.max():.4f} | "
                  f"bad n={len(b)} median={np.median(b):.4f} max={b.max():.4f}")
        thresholds = fit_thresholds(scores, args.target_recall)
        doc = {
            "thresholds": thresholds,
            "input_mode": args.input_mode,
            "score_stat": args.score_stat,
            "decision_rule": f"ROI FAIL when {args.score_stat}(anomaly_map) > threshold; label FAIL if any ROI FAILs or is MISSING",
            "models_root": str(args.models_root),
            "selection": f"image-level target recall {args.target_recall}" if args.target_recall else "image-level max F1",
            "n_good": sum(1 for _, l in labelled if l == "good"),
            "n_bad": sum(1 for _, l in labelled if l == "bad"),
            "warning": "Fitted on the same scans the models were trained on; confirm on a held-out set.",
            **{k: v for k, v in (old_doc or {}).items() if k in ("model_dirs", "input_modes")},
        }
        thr_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote thresholds -> {thr_path}")
        for n, t in thresholds.items():
            print(f"  {n}: {t:.6f}")
        for _, _, r in records:  # re-judge with the fitted thresholds for the summary below
            for n, v in r.rois.items():
                if v.score is not None:
                    v.status = "FAIL" if v.score > thresholds[n] else "PASS"
            r.fail_reasons = [n for n, v in r.rois.items() if v.status != "PASS"]
            r.verdict = "FAIL" if r.fail_reasons else "PASS"

    lab = [(l, r) for _, l, r in records if l]
    if lab:
        tp = sum(1 for l, r in lab if l == "bad" and not r.passed)
        fn = sum(1 for l, r in lab if l == "bad" and r.passed)
        fp = sum(1 for l, r in lab if l == "good" and not r.passed)
        tn = sum(1 for l, r in lab if l == "good" and r.passed)
        print(f"\nImage-level (FAIL = defect):  TP={tp} FN={fn} FP={fp} TN={tn}  "
              f"recall={tp / max(1, tp + fn):.3f}  precision={tp / max(1, tp + fp):.3f}  "
              f"specificity={tn / max(1, tn + fp):.3f}")
        missing = sum(1 for _, r in lab for v in r.rois.values() if v.status == "MISSING")
        print(f"ROIs not located: {missing}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([{**r.to_dict(), "label": l} for _, l, r in records], indent=2), encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
