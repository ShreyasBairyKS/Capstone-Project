"""
Model-free tests for the explainability helpers and the upload web app.

Run from himalaya_label_detection/:   python -m pytest template_roi_pipeline/tests -q

The app is exercised through FastAPI's TestClient with stand-in classifiers (fixed score + a known
blob in the anomaly map), so the whole upload -> locate -> score -> explain -> JSON/images path runs
without ONNX models.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from template_roi_pipeline import explain  # noqa: E402
from template_roi_pipeline.app.server import create_app  # noqa: E402
from template_roi_pipeline.inspector import ROIVerdict, TemplateROIInspector  # noqa: E402
from template_roi_pipeline.roi_template_matcher import ROI_NAMES, ROIMatch, ROITemplateMatcher  # noqa: E402
from test_template_roi_pipeline import make_scan  # noqa: E402

CONFIG = PKG_ROOT / "config"
THRESHOLD = 0.05


def blob_map(value: float = 4.0, box=(100, 140, 100, 140)) -> np.ndarray:
    """256x256 anomaly map: zero everywhere except one square at `value`."""
    y0, y1, x0, x1 = box
    amap = np.zeros((256, 256), np.float32)
    amap[y0:y1, x0:x1] = value
    return amap


def verdict(status: str, score: float | None, amap=None, crop=None, found=True) -> ROIVerdict:
    match = ROIMatch("ROI_2", 10, 20, 100, 60, 0.9 if found else 0.2, found)
    return ROIVerdict("ROI_2", status, score, THRESHOLD, match, amap, crop)


# ── explain: geometry ─────────────────────────────────────────────────────────

def test_upright_to_scan_inverts_the_clockwise_rotation():
    # Put a marker in a crop (scan orientation), rotate it the way the inspector does, find the marker in the
    # upright crop, and map its box back: it must land on the marker's real position in the crop.
    w, h = 200, 500
    crop = np.zeros((h, w), np.uint8)
    crop[300:340, 30:90] = 255                                        # y 300..340, x 30..90
    upright = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    ys, xs = np.where(upright > 0)
    uh, uw = upright.shape
    box = (xs.min() / uw, ys.min() / uh, (xs.max() + 1) / uw, (ys.max() + 1) / uh)
    x0, y0, x1, y1 = explain.upright_to_scan(box, (0, 0, w, h))
    assert (round(x0), round(y0), round(x1), round(y1)) == (30, 300, 90, 340)
    ox, oy, *_ = explain.upright_to_scan(box, (1000, 5000, w, h))      # offset by the ROI position
    assert (round(ox), round(oy)) == (1030, 5300)


@pytest.mark.parametrize("cx,cy,word", [(0.1, 0.1, "top-left"), (0.9, 0.9, "bottom-right"), (0.5, 0.5, "centre"),
                                        (0.5, 0.1, "top"), (0.9, 0.5, "right")])
def test_position_words(cx, cy, word):
    assert explain.describe_position(cx, cy) == word


# ── explain: regions and counterfactual ───────────────────────────────────────

def test_blob_is_found_with_its_counterfactual():
    amap = blob_map()
    (region,) = explain.find_hot_regions(amap, THRESHOLD)
    lo, hi = 100 / 256, 140 / 256                                      # the blob's extent, as a fraction of the map
    assert (region.x0, region.y0, region.x1, region.y1) == pytest.approx((lo, lo, hi, hi), abs=0.02)
    assert region.where == "centre"
    # score = mean(map); removing the blob leaves a flat zero map
    assert region.accounts_for == pytest.approx(float(amap.mean()), rel=1e-3)
    assert region.score_without == pytest.approx(0.0, abs=1e-6)


def test_clean_map_has_no_regions():
    rng = np.random.default_rng(0)
    assert explain.find_hot_regions(rng.normal(0, 0.01, (256, 256)).astype(np.float32), THRESHOLD) == []


def test_two_blobs_are_two_regions_strongest_first():
    amap = blob_map(box=(20, 50, 20, 50))
    amap[180:230, 150:230] = 6.0
    regions = explain.find_hot_regions(amap, THRESHOLD)
    assert [r.index for r in regions] == [1, 2]
    assert regions[0].accounts_for > regions[1].accounts_for
    assert regions[0].where == "bottom-right" and regions[1].where == "top-left"


# ── explain: text ─────────────────────────────────────────────────────────────

def test_fail_text_names_the_region_that_causes_it():
    amap = blob_map()
    v = verdict("FAIL", float(amap.mean()), amap)
    info = explain.explain_roi(v, explain.find_hot_regions(amap, THRESHOLD))
    assert "above the limit" in info["summary"] and "causes the rejection" in info["summary"]
    assert info["margin"] == pytest.approx(v.score - THRESHOLD, abs=1e-6)
    gauge = info["gauge"]
    assert gauge["lo"] < THRESHOLD < gauge["hi"] and (gauge["lo"] + gauge["hi"]) / 2 == pytest.approx(THRESHOLD, abs=1e-5)


def test_widespread_anomaly_is_not_blamed_on_one_region():
    amap = np.full((256, 256), 1.0, np.float32)
    amap[100:120, 100:120] = 9.0                                       # small spike on top of a uniformly high map
    v = verdict("FAIL", float(amap.mean()), amap)
    info = explain.explain_roi(v, explain.find_hot_regions(amap, THRESHOLD))
    assert "not confined to one place" in info["summary"]


def test_pass_and_missing_text():
    clean = np.zeros((256, 256), np.float32)
    info = explain.explain_roi(verdict("PASS", 0.0, clean), [])
    assert "below the limit" in info["summary"] and info["margin"] < 0
    missing = explain.explain_roi(verdict("MISSING", None, found=False), [])
    assert "could not be located" in missing["summary"] and missing["score"] is None and missing["gauge"] is None


# ── explain: rendering ────────────────────────────────────────────────────────

def test_overlay_only_tints_where_the_map_is_above_the_threshold():
    crop = np.full((120, 300), 128, np.uint8)
    amap = blob_map()
    out = explain.render_overlay(crop, amap, THRESHOLD, "above", [], width=300)
    base = explain.render_crop(crop, 300)
    changed = np.any(out != base, axis=2)
    assert out.shape == base.shape and changed.any()
    assert not changed[:30, :].any()                                   # far from the blob (rows ~ 0.39..0.55 of height)
    assert not np.any(explain.render_overlay(crop, np.zeros((256, 256), np.float32), THRESHOLD, "above", [],
                                             width=300) != base)       # a clean map leaves the crop untouched


def test_overview_boxes_cover_the_rois_and_skip_missing():
    matcher = ROITemplateMatcher.from_config(CONFIG)
    scan = cv2.cvtColor(make_scan(matcher), cv2.COLOR_GRAY2BGR)
    rois = {n: ROIVerdict(n, "PASS", 0.0, 1.0, m) for n, m in matcher.locate(cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)).items()}
    rois["ROI_4"] = ROIVerdict("ROI_4", "MISSING", None, 1.0, ROIMatch("ROI_4", 0, 0, 634, 687, -1.0, False))
    from template_roi_pipeline.inspector import InspectionResult
    img, boxes = explain.render_overview(scan, InspectionResult("FAIL", rois, ["x"], 1.0), {})
    assert set(boxes) == {"ROI_1", "ROI_2", "ROI_3"}                   # no click target for a ROI that was not found
    assert all(0 <= v <= 1 for b in boxes.values() for v in b) and img.shape[0] < img.shape[1]   # upright strip is wide


# ── app ───────────────────────────────────────────────────────────────────────

class FakeClassifier:
    def __init__(self, amap: np.ndarray):
        self.amap = amap

    def score(self, crop):
        return float(self.amap.mean()), self.amap


@pytest.fixture(scope="module")
def client():
    matcher = ROITemplateMatcher.from_config(CONFIG)
    classifiers = {n: FakeClassifier(blob_map() if n == "ROI_2" else np.zeros((256, 256), np.float32)) for n in ROI_NAMES}
    inspector = TemplateROIInspector(matcher, classifiers, {n: THRESHOLD for n in ROI_NAMES}, input_mode="color")
    app = create_app(inspector, {"status": "PROVISIONAL: test", "score_stat": "mean"}, "models/test", samples={})
    return TestClient(app)


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def wait_done(client: TestClient, job_id: str, timeout: float = 60) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def scan_png() -> bytes:
    matcher = ROITemplateMatcher.from_config(CONFIG)
    return encode(cv2.cvtColor(make_scan(matcher), cv2.COLOR_GRAY2BGR))


def test_status_and_index(client):
    s = client.get("/api/status").json()
    assert s["input_mode"] == "color" and s["expected_size"] == [1504, 8000] and s["threshold_status"].startswith("PROVISIONAL")
    page = client.get("/")
    assert page.status_code == 200 and "Himalaya" in page.text
    assert isinstance(s["jobs"], list)                                 # ids of the results still in memory


def test_upload_runs_the_whole_pipeline(client, scan_png):
    job_id = client.post("/api/jobs", files={"file": ("scan.png", scan_png, "image/png")}).json()["job_id"]
    job = wait_done(client, job_id)
    assert job["state"] == "done" and job["progress"] == 1.0 and set(job["steps"].values()) == {"done"}
    assert job_id in client.get("/api/status").json()["jobs"]            # the page lists only results the server still has
    r = job["result"]
    assert r["verdict"] == "FAIL" and r["filename"] == "scan.png" and r["image_size"] == [1504, 8000]
    assert [n for n, i in r["rois"].items() if i["status"] == "FAIL"] == ["ROI_2"]
    assert "ROI_2" in r["headline"] and not r["warnings"]
    roi2 = r["rois"]["ROI_2"]
    assert roi2["regions"] and roi2["regions"][0]["where"] == "centre" and roi2["images"]
    assert set(r["overview_boxes"]) == set(ROI_NAMES)

    for rel in [r["overview"], *roi2["images"].values()]:               # every referenced image can be fetched
        resp = client.get(f"/api/jobs/{job_id}/{rel}")
        assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR) is not None
    report = client.get(f"/api/jobs/{job_id}/report.json")
    assert report.status_code == 200 and report.json()["verdict"] == "FAIL"
    assert 'filename="scan_report.json"' in report.headers["content-disposition"]


def test_grayscale_mode_explains_single_channel_crops_and_names_each_model(scan_png):
    # input_mode="gray": every crop the classifiers (and the heat views) see is single-channel, and ROI_4 comes from
    # its own model folder; the page data must say which model judged each ROI
    from pathlib import Path

    class WithModel(FakeClassifier):
        def __init__(self, amap, model_path):
            super().__init__(amap)
            self.model_path = Path(model_path)

    matcher = ROITemplateMatcher.from_config(CONFIG)
    flat = np.zeros((256, 256), np.float32)
    classifiers = {n: WithModel(blob_map() if n == "ROI_4" else flat, f"models/gray_scale_rois/{n}/weights/model.onnx") for n in ROI_NAMES}
    classifiers["ROI_4"].model_path = Path("models/gray_scale_rois/new_roi4/weights/model.onnx")
    inspector = TemplateROIInspector(matcher, classifiers, {n: THRESHOLD for n in ROI_NAMES}, input_mode="gray")
    gray_client = TestClient(create_app(inspector, {"status": "PROVISIONAL: test"}, "models/gray_scale_rois", samples={}))

    s = gray_client.get("/api/status").json()
    assert s["input_mode"] == "gray" and s["models"] == {"ROI_1": "ROI_1", "ROI_2": "ROI_2", "ROI_3": "ROI_3", "ROI_4": "new_roi4"}

    job = wait_done(gray_client, gray_client.post("/api/jobs", files={"file": ("scan.png", scan_png)}).json()["job_id"])
    assert job["state"] == "done", job["error"]
    rois = job["result"]["rois"]
    assert [n for n, i in rois.items() if i["status"] == "FAIL"] == ["ROI_4"]
    assert rois["ROI_4"]["model"] == "new_roi4" and rois["ROI_1"]["model"] == "ROI_1"
    assert rois["ROI_4"]["regions"] and rois["ROI_4"]["images"]
    for rel in rois["ROI_4"]["images"].values():
        resp = gray_client.get(f"/api/jobs/{job['id']}/{rel}")
        assert resp.status_code == 200 and cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR) is not None


def test_identical_upload_is_served_from_cache_under_its_own_name(client, scan_png):
    first = wait_done(client, client.post("/api/jobs", files={"file": ("one.png", scan_png)}).json()["job_id"])
    second_id = client.post("/api/jobs", files={"file": ("two.png", scan_png)}).json()["job_id"]
    second = client.get(f"/api/jobs/{second_id}").json()               # already done: no waiting
    assert second_id != first["id"] and second["state"] == "done"
    assert second["result"]["filename"] == "two.png" and second["result"]["verdict"] == first["result"]["verdict"]
    assert client.get(f"/api/jobs/{second_id}/{second['result']['overview']}").status_code == 200


def test_unreadable_file_is_reported_not_crashed(client):
    job = wait_done(client, client.post("/api/jobs", files={"file": ("notes.bmp", b"not an image")}).json()["job_id"])
    assert job["state"] == "error" and "could not be read" in job["error"]
    ok = wait_done(client, client.post("/api/jobs", files={"file": ("x.png", encode(np.zeros((50, 50, 3), np.uint8)))}).json()["job_id"])
    assert ok["state"] == "done"                                       # the worker survived the previous failure


def test_wrong_size_and_rotated_scans_get_warnings(client):
    small = wait_done(client, client.post("/api/jobs", files={"file": ("s.png", encode(np.zeros((600, 800, 3), np.uint8)))}).json()["job_id"])
    assert "not the usual" in small["result"]["warnings"][0] and small["result"]["verdict"] == "FAIL"
    assert set(i["status"] for i in small["result"]["rois"].values()) == {"MISSING"}
    rotated = wait_done(client, client.post("/api/jobs", files={"file": ("r.png", encode(np.zeros((1504, 8000, 3), np.uint8)))}).json()["job_id"])
    assert "rotated" in rotated["result"]["warnings"][0]


def test_bad_requests(client):
    assert client.post("/api/jobs", files={"file": ("e.bmp", b"")}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/images/overview.jpg").status_code == 404
    assert client.post("/api/jobs/sample", json={"name": "../../etc/passwd"}).status_code == 404
