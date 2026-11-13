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
import shutil
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("threshold", [-0.02557061705738306, 0.05])
@pytest.mark.parametrize("offset,expected,reason", [
    (-0.000001, "PASS", None), (0.0, "FAIL", "near_threshold"),
    (0.0017, "FAIL", "near_threshold"), (0.005, "FAIL", "near_threshold"),
    (0.006, "FAIL", "above_threshold"),
])
def test_rejection_margin_applies_at_the_cutoff_and_for_negative_thresholds(matcher, threshold, offset, expected, reason):
    margin = 0.005
    scores = {n: threshold - 0.01 for n in ROI_NAMES}
    scores["ROI_1"] = threshold if offset == margin else threshold - margin + offset
    inspector, _ = make_inspector(matcher, scores, thresholds={n: threshold for n in ROI_NAMES}, rejection_margin=margin)
    result = inspector.inspect(make_scan(matcher))
    roi = result.rois["ROI_1"]
    assert result.verdict == expected and roi.status == expected and roi.rejection_reason == reason
    assert roi.threshold == threshold and roi.decision_threshold == pytest.approx(threshold - margin)
    assert all(v.status == "PASS" for n, v in result.rois.items() if n != "ROI_1")
    exported = roi.to_dict()
    assert exported["rejection_margin"] == margin and exported["rejection_reason"] == reason
    assert exported["decision_threshold"] == pytest.approx(threshold - margin, abs=0.00000051)
    if reason == "near_threshold":
        assert "rejection margin" in result.fail_reasons[0] and "> threshold" not in result.fail_reasons[0]


@pytest.mark.parametrize("margin", [-0.001, float("nan"), float("inf"), float("-inf")])
def test_invalid_rejection_margin_is_refused_before_loading_models(matcher, margin):
    with pytest.raises(ValueError, match="rejection margin"):
        make_inspector(matcher, {n: 0.0 for n in ROI_NAMES}, rejection_margin=margin)
    with pytest.raises(ValueError, match="rejection margin"):
        TemplateROIInspector.from_profile({"thresholds": {n: 0.1 for n in ROI_NAMES}, "rejection_margin": margin}, CONFIG, "unused")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_model_scores_never_pass(matcher, value):
    scores = {n: 0.5 for n in ROI_NAMES}
    scores["ROI_1"] = value
    inspector, _ = make_inspector(matcher, scores)
    with pytest.raises(ValueError, match="invalid score"):
        inspector.inspect(make_scan(matcher))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_profiles_reject_non_finite_limits_before_loading_models(value):
    profile = {"thresholds": {n: value for n in ROI_NAMES}}
    with pytest.raises(ValueError, match="non-finite threshold"):
        TemplateROIInspector.from_profile(profile, CONFIG, "unused")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_direct_inspectors_reject_non_finite_thresholds(matcher, value):
    with pytest.raises(ValueError, match="finite number"):
        make_inspector(matcher, {n: 0.0 for n in ROI_NAMES}, thresholds={n: value for n in ROI_NAMES})


@pytest.mark.parametrize("settings", [{"input_mode": "color"}, {"score_stat": "max"}])
def test_profiles_reject_mismatched_preprocessing(settings):
    profile = {"thresholds": {n: 0.1 for n in ROI_NAMES}, **settings}
    with pytest.raises(ValueError, match="must match"):
        TemplateROIInspector.from_profile(profile, CONFIG, "unused", input_mode="gray", score_stat="mean")


@pytest.mark.parametrize("output", [np.zeros((1, 1, 64, 64), np.float32),
                                    np.full((1, 1, 256, 256), np.nan, np.float32)])
def test_onnx_output_contract_is_checked(output):
    class Session:
        def run(self, *_):
            return [output]

    classifier = object.__new__(ONNXROIClassifier)
    classifier.session, classifier.roi_name, classifier.score_stat = Session(), "ROI_1", "mean"
    with pytest.raises(ValueError, match="invalid anomaly map"):
        classifier.score(np.zeros((20, 30), np.uint8))


def test_templates_load_from_a_unicode_windows_path(tmp_path):
    config = tmp_path / "inspection \u6d4b\u8bd5"
    shutil.copytree(CONFIG / "roi_templates", config / "roi_templates")
    matcher = ROITemplateMatcher.from_config(config)
    assert set(matcher.templates) == set(ROI_NAMES)
    assert all(not m.found for m in matcher.locate(np.zeros((1, 1), np.uint8)).values())


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


class _FakeONNX:
    """Stands in for ONNXROIClassifier when a profile is loaded: remembers where its model was read from."""

    def __init__(self, roi_name, model_path, score_stat="mean"):
        self.roi_name, self.model_path = roi_name, Path(model_path)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_profile_can_point_one_roi_at_another_model_set(tmp_path, monkeypatch):
    import template_roi_pipeline.inspector as inspector_module
    monkeypatch.setattr(inspector_module, "ONNXROIClassifier", _FakeONNX)
    for name in ROI_NAMES:
        _touch(tmp_path / "rois" / name / "weights" / "model.onnx")
    _touch(tmp_path / "gray_scale_rois" / "new_roi4" / "weights" / "model.onnx")
    profile = {"thresholds": {n: 0.1 for n in ROI_NAMES}, "model_dirs": {"ROI_4": "../gray_scale_rois/new_roi4"}, "rejection_margin": 0.005}

    insp = TemplateROIInspector.from_profile(profile, CONFIG, tmp_path / "rois", input_mode="gray")
    assert insp.rejection_margin == 0.005 and insp.thresholds == profile["thresholds"]

    assert insp.classifiers["ROI_2"].model_path == tmp_path / "rois" / "ROI_2" / "weights" / "model.onnx"
    assert insp.classifiers["ROI_4"].model_path.resolve() == (
        tmp_path / "gray_scale_rois" / "new_roi4" / "weights" / "model.onnx").resolve()

    profile["model_dirs"] = {"ROI_4": "nowhere"}
    with pytest.raises(FileNotFoundError, match="nowhere"):
        TemplateROIInspector.from_profile(profile, CONFIG, tmp_path / "rois", input_mode="gray")


@pytest.mark.parametrize("profile", ["roi_onnx_thresholds.json", "roi_onnx_thresholds_color.json"])
def test_shipped_profiles_resolve_to_models_that_exist(profile):
    import json
    repo_root = PKG_ROOT.parents[1]
    doc = json.loads((CONFIG / profile).read_text(encoding="utf-8"))
    assert set(doc["thresholds"]) == set(ROI_NAMES)
    for name in ROI_NAMES:
        path = repo_root / doc["models_root"] / doc.get("model_dirs", {}).get(name, name) / "weights" / "model.onnx"
        assert path.is_file(), path


def test_default_profile_is_the_grayscale_set_with_the_new_roi4():
    import json
    doc = json.loads((CONFIG / "roi_onnx_thresholds.json").read_text(encoding="utf-8"))
    assert doc["input_mode"] == "gray" and doc["models_root"] == "models/gray_scale_rois"
    assert doc["model_dirs"] == {"ROI_4": "new_roi4"}
    assert doc["rejection_margin"] == 0.005
    assert doc["thresholds"] == {"ROI_1": -0.02557061705738306, "ROI_2": 0.22597697377204895,
                                 "ROI_3": 0.04930554889142513, "ROI_4": 0.0789654339419136}


def test_calibration_uses_finite_limits_and_exports_the_fitted_verdicts(tmp_path, monkeypatch):
    import csv
    import json
    from template_roi_pipeline.inspector import InspectionResult, ROIVerdict
    from template_roi_pipeline.roi_template_matcher import ROIMatch
    from template_roi_pipeline.scripts import run_template_inference as cli

    args = SimpleNamespace(good=tmp_path / "good", bad=tmp_path / "bad", images=[], calibrate=True,
                           thresholds=tmp_path / "limits.json", config_dir=CONFIG, models_root=tmp_path / "models",
                           input_mode="gray", score_stat="mean", save_annotated=None, target_recall=None,
                           json_out=tmp_path / "results.json", csv_out=tmp_path / "results.csv")
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "collect", lambda paths: [p / "scan.bmp" for p in paths])
    monkeypatch.setattr(cli, "load_scan", lambda *_: np.zeros((10, 10), np.uint8))

    class Inspector:
        input_mode = "gray"

        def inspect(self, _, path):
            rois = {n: ROIVerdict(n, "PASS", 0.2, 1.0, ROIMatch(n, 0, 0, 1, 1, 1.0, True)) for n in ROI_NAMES}
            return InspectionResult("PASS", rois, [], 1.0, path)

    def build(_, profile):
        assert all(np.isfinite(v) for v in profile["thresholds"].values())
        return Inspector()

    monkeypatch.setattr(cli, "build_inspector", build)
    monkeypatch.setattr(cli, "fit_thresholds", lambda *_: {n: 0.1 for n in ROI_NAMES})
    assert cli.main() == 0
    results = json.loads(args.json_out.read_text(encoding="utf-8"))
    assert all(r["verdict"] == "FAIL" and all(v["threshold"] == 0.1 and v["status"] == "FAIL" for v in r["rois"].values()) for r in results)
    with args.csv_out.open(newline="", encoding="utf-8") as stream:
        assert all(row["verdict"] == "FAIL" and row["ROI_1_status"] == "FAIL" for row in csv.DictReader(stream))


def test_preprocess_contract():
    gray = np.random.default_rng(0).integers(0, 256, (645, 1502), dtype=np.uint8)
    x = ONNXROIClassifier.preprocess(gray)
    assert x.shape == (1, 3, 256, 256) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert np.array_equal(x[0, 0], x[0, 1]) and np.array_equal(x[0, 1], x[0, 2])  # gray replicated to RGB
    bgr = np.zeros((300, 300, 3), np.uint8); bgr[..., 0] = 255                      # pure blue in BGR order
    y = ONNXROIClassifier.preprocess(bgr)
    assert y[0, 2].min() == 1.0 and y[0, 0].max() == 0.0                            # blue ends up in the RGB B channel
