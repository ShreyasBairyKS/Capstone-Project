"""
Model-free tests for the template-matching ROI pipeline.

Run from himalaya_label_detection/:   python -m pytest template_roi_pipeline/tests -q

A synthetic 1504 x 8000 "scan" is built by pasting the real ROI templates into a
textured background at known positions, so localisation accuracy, the
complete-occurrence rule, the missing-ROI rule and the crop orientation handed to
the classifiers can all be checked without ONNX models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]          # himalaya_label_detection/template_roi_pipeline
sys.path.insert(0, str(PKG_ROOT.parent))

from template_roi_pipeline.inspector import ONNXROIClassifier, TemplateROIInspector  # noqa: E402
from template_roi_pipeline.roi_template_matcher import ROI_NAMES, ROITemplateMatcher  # noqa: E402

CONFIG = PKG_ROOT / "config"
H, W = 8000, 1504

# (x, y) of each ROI in the synthetic scan; ROI heights/widths come from the templates.
LAYOUT = {"ROI_1": (255, 300), "ROI_4": (259, 2000), "ROI_3": (311, 2900), "ROI_2": (276, 5000)}


@pytest.fixture(scope="module")
def matcher() -> ROITemplateMatcher:
    return ROITemplateMatcher.from_config(CONFIG)


def make_scan(matcher: ROITemplateMatcher, layout=LAYOUT, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scan = cv2.GaussianBlur(rng.integers(0, 60, (H, W), dtype=np.uint8), (0, 0), 3)
    for name, (x, y) in layout.items():
        t = matcher.templates[name]
        scan[y:y + t.shape[0], x:x + t.shape[1]] = t
    return scan


def test_locates_every_roi_within_a_pixel(matcher):
    matches = matcher.locate(make_scan(matcher))
    for name, (x, y) in LAYOUT.items():
        m = matches[name]
        assert m.found and m.score > 0.95, (name, m)
        assert abs(m.x - x) <= 1 and abs(m.y - y) <= 1, (name, m, (x, y))


def test_second_occurrence_is_not_a_problem(matcher):
    # ROI_1 appears twice (label wraps the tube): either complete copy is acceptable, a single one is returned.
    scan = make_scan(matcher)
    t = matcher.templates["ROI_1"]
    scan[6400:6400 + t.shape[0], 400:400 + t.shape[1]] = t      # second, equally good copy lower down
    m = matcher.locate(scan)["ROI_1"]
    first, second = (255, 300), (400, 6400)
    assert m.found
    assert any(abs(m.x - x) <= 1 and abs(m.y - y) <= 1 for x, y in (first, second)), m


def test_roi_cut_by_the_image_edge_is_never_chosen(matcher):
    # Only a truncated copy of ROI_3 is present (hanging off the bottom edge): it must be reported not found.
    t = matcher.templates["ROI_3"]
    rng = np.random.default_rng(1)
    scan = cv2.GaussianBlur(rng.integers(0, 60, (H, W), dtype=np.uint8), (0, 0), 3)
    keep = 700
    scan[H - keep:H, 311:311 + t.shape[1]] = t[:keep]
    assert not matcher.locate(scan)["ROI_3"].found


class RecordingClassifier:
    """Stands in for ONNXROIClassifier: returns a preset score and remembers the crop it was given."""

    def __init__(self, score: float):
        self._score, self.seen = score, None

    def score(self, crop):
        self.seen = crop
        return self._score, np.zeros((256, 256), np.float32)


def make_inspector(matcher, scores, thresholds=None, **kw):
    clf = {n: RecordingClassifier(scores[n]) for n in ROI_NAMES}
    thr = thresholds or {n: 1.0 for n in ROI_NAMES}
    return TemplateROIInspector(matcher, clf, thr, **kw), clf


def test_all_pass_when_scores_below_thresholds(matcher):
    insp, _ = make_inspector(matcher, {n: 0.5 for n in ROI_NAMES})
    r = insp.inspect(make_scan(matcher))
    assert r.verdict == "PASS" and not r.fail_reasons
    assert all(v.status == "PASS" for v in r.rois.values())


def test_one_failing_roi_fails_the_label(matcher):
    insp, _ = make_inspector(matcher, {"ROI_1": 0.5, "ROI_2": 0.5, "ROI_3": 2.0, "ROI_4": 0.5})
    r = insp.inspect(make_scan(matcher))
    assert r.verdict == "FAIL"
    assert [n for n, v in r.rois.items() if v.status == "FAIL"] == ["ROI_3"]
    assert "ROI_3" in r.fail_reasons[0]


def test_score_equal_to_threshold_passes(matcher):
    insp, _ = make_inspector(matcher, {n: 1.0 for n in ROI_NAMES})  # FAIL only when score > threshold
    assert insp.inspect(make_scan(matcher)).verdict == "PASS"


def test_missing_roi_fails_unless_allowed(matcher):
    layout = {k: v for k, v in LAYOUT.items() if k != "ROI_4"}
    scan = make_scan(matcher, layout)
    insp, clf = make_inspector(matcher, {n: 0.5 for n in ROI_NAMES})
    r = insp.inspect(scan)
    assert r.rois["ROI_4"].status == "MISSING" and r.verdict == "FAIL"
    assert clf["ROI_4"].seen is None  # classifier is never run on an unlocated ROI

    insp2, _ = make_inspector(matcher, {n: 0.5 for n in ROI_NAMES}, fail_on_missing=False)
    assert insp2.inspect(scan).verdict == "PASS"


def test_classifier_receives_crop_rotated_clockwise(matcher):
    insp, clf = make_inspector(matcher, {n: 0.5 for n in ROI_NAMES})
    insp.inspect(make_scan(matcher))
    for name in ROI_NAMES:
        t = matcher.templates[name]
        seen = clf[name].seen
        assert seen.shape == (t.shape[1], t.shape[0])  # (w, h): rotated 90 degrees
        assert np.array_equal(seen, cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE))


def test_color_mode_passes_bgr_crops_and_requires_bgr(matcher):
    insp, clf = make_inspector(matcher, {n: 0.5 for n in ROI_NAMES}, input_mode="color")
    gray = make_scan(matcher)
    with pytest.raises(ValueError):
        insp.inspect(gray)
    insp.inspect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert clf["ROI_1"].seen.ndim == 3 and clf["ROI_1"].seen.shape[2] == 3


def test_preprocess_contract():
    gray = np.random.default_rng(0).integers(0, 256, (645, 1502), dtype=np.uint8)
    x = ONNXROIClassifier.preprocess(gray)
    assert x.shape == (1, 3, 256, 256) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert np.array_equal(x[0, 0], x[0, 1]) and np.array_equal(x[0, 1], x[0, 2])  # gray replicated to RGB
    bgr = np.zeros((300, 300, 3), np.uint8); bgr[..., 0] = 255                      # pure blue in BGR order
    y = ONNXROIClassifier.preprocess(bgr)
    assert y[0, 2].min() == 1.0 and y[0, 0].max() == 0.0                            # blue ends up in the RGB B channel
