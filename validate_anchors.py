"""
validate_anchors.py
────────────────────────────────────────────────────────────────────────────
Validates that template matching correctly finds all 3 ROI regions in
EVERY good and bad image before training begins.

Run from project root:
    python validate_anchors.py

Pass rate must be 100% for good images before proceeding.
"""

import cv2
import sys
import numpy as np
from pathlib import Path

# Make sure the src package is importable
sys.path.insert(0, str(Path(__file__).parent / "himalaya_label_detection"))
from src.preprocessing.anchor import TemplateAnchorFinder

CONFIG_DIR = Path("himalaya_label_detection/config")
GOOD_DIR   = Path("dataset/NSC/NSC GOOD IMAGES")
BAD_DIR    = Path("dataset/NSC/NSC BAD IMAGES")

# ── Init finder ───────────────────────────────────────────────────────────────
finder = TemplateAnchorFinder(CONFIG_DIR)

if not finder.templates:
    print("ERROR: No template patches found in config/")
    print("Run:  python himalaya_label_detection/scripts/save_templates.py  first")
    sys.exit(1)

print(f"Templates loaded: {list(finder.templates.keys())}")
print(f"ROI heights: {finder.roi_heights}")
print()


def validate_folder(folder: Path, label: str):
    files = sorted(folder.glob("*.bmp"))
    if not files:
        print(f"  [SKIP] {folder} — no .bmp files found")
        return 0, 0, 0

    ok = 0; partial = 0; fail = 0
    all_scores: dict[str, list] = {k: [] for k in finder.templates}

    print(f"{'='*70}")
    print(f"  {label.upper()}  ({len(files)} images)")
    print(f"{'='*70}")

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ERROR  {f.name}: cannot read file")
            fail += 1
            continue

        anchors = finder.find_all(img)
        n_found = sum(1 for a in anchors.values() if a.found)

        scores_str = "  ".join(
            f"{k.replace('ROI_','')}={'✓' if a.found else '✗'} {a.score:.2f} y={a.y if a.found else '—':>4}"
            for k, a in anchors.items()
        )

        if n_found == len(anchors):
            status = "OK  "
            ok += 1
        elif n_found > 0:
            status = "PART"
            partial += 1
        else:
            status = "FAIL"
            fail += 1

        print(f"  {status} {f.name:10s}  {scores_str}")

        for k, a in anchors.items():
            all_scores[k].append(a.score)

    print()
    print(f"  Result: {ok} OK | {partial} PARTIAL | {fail} FAIL  of {len(files)}")

    if all_scores:
        for k, sc in all_scores.items():
            print(f"  {k}: score min={min(sc):.3f}  max={max(sc):.3f}  mean={np.mean(sc):.3f}")

    print()
    return ok, partial, fail


# ── Validate ──────────────────────────────────────────────────────────────────
g_ok, g_part, g_fail = validate_folder(GOOD_DIR, "GOOD IMAGES")
b_ok, b_part, b_fail = validate_folder(BAD_DIR,  "BAD IMAGES")

print("="*70)
print("  FINAL VERDICT")
print("="*70)
total = g_ok + g_part + g_fail + b_ok + b_part + b_fail
passed = g_ok + b_ok
print(f"  Good images: {g_ok}/{g_ok+g_part+g_fail} fully found")
print(f"  Bad  images: {b_ok}/{b_ok+b_part+b_fail} fully found")

if g_fail == 0 and g_part == 0:
    print()
    print("  ✅ ALL GOOD IMAGES PASS — safe to proceed with training and inference")
elif g_fail == 0:
    print()
    print("  ⚠️  Some good images only found PARTIAL ROIs — review PART rows above")
    print("     Consider lowering TemplateAnchorFinder.MATCH_THRESHOLD to 0.45")
else:
    print()
    print("  ❌ Some good images FAILED — do NOT proceed until fixed")
    print("     Action: update template patches using a clearer reference image")
    print("     Re-run: python himalaya_label_detection/scripts/save_templates.py")
