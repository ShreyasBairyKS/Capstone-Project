"""
Model-free tests for what the desktop app adds: saved results, folder runs, the host/header guards, the
launcher's start-up and path rules.

Run from himalaya_label_detection/:   python -m pytest template_roi_pipeline/tests -q
"""

from __future__ import annotations

import json
import csv
import io
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from template_roi_pipeline.app import server  # noqa: E402
from template_roi_pipeline.app.server import Batch, create_app  # noqa: E402
from template_roi_pipeline.app.history import HistoryStore  # noqa: E402
from template_roi_pipeline.desktop import launcher, paths  # noqa: E402
from template_roi_pipeline.inspector import TemplateROIInspector  # noqa: E402
from template_roi_pipeline.roi_template_matcher import ROI_NAMES, ROITemplateMatcher  # noqa: E402
from test_explain_and_app import FakeClassifier, blob_map, encode, wait_done  # noqa: E402
from test_template_roi_pipeline import make_scan  # noqa: E402

CONFIG = PKG_ROOT / "config"
UI = {"X-Requested-With": "himalaya"}                      # what the page sends with every request


class SlowClassifier(FakeClassifier):
    """Scores only once the test opens the gate, so a folder run can be stopped mid-way."""

    def __init__(self, amap, gate: threading.Event):
        super().__init__(amap)
        self.gate = gate

    def score(self, crop):
        self.gate.wait(30)
        return super().score(crop)


def make_inspector(gate: threading.Event | None = None, limit: float = 0.05, rejection_margin: float = 0.0) -> TemplateROIInspector:
    matcher = ROITemplateMatcher.from_config(CONFIG)
    flat, blob = np.zeros((256, 256), np.float32), blob_map()
    classifiers = {n: (SlowClassifier(flat, gate) if gate else FakeClassifier(blob if n == "ROI_2" else flat)) for n in ROI_NAMES}
    return TemplateROIInspector(matcher, classifiers, {n: limit for n in ROI_NAMES}, input_mode="color", rejection_margin=rejection_margin)


def make_app(tmp_path: Path, inspector=None, **kw):
    kw.setdefault("history_dir", tmp_path / "history")
    return create_app(inspector or make_inspector(), {"status": "PROVISIONAL: test", "score_stat": "mean"}, "models/test", **kw)


@pytest.fixture(scope="module")
def scans() -> list[bytes]:
    """Three different scans (one corner pixel apart), so none of them is answered from another's cache."""
    base = make_scan(ROITemplateMatcher.from_config(CONFIG))
    out = []
    for i in range(3):
        img = base.copy()
        img[0, 0] = 10 + i
        out.append(encode(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    return out


def post_scan(client, name: str, data: bytes) -> str:
    return client.post("/api/jobs", files={"file": (name, data)}).json()["job_id"]


def wait_batch(client, batch_id: str, timeout: float = 120) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        b = client.get(f"/api/batches/{batch_id}").json()
        if b["state"] != "running":
            return b
        time.sleep(0.2)
    raise AssertionError("folder run did not finish")


def tiny_png(value: int) -> bytes:
    return encode(np.full((40, 40, 3), value, np.uint8))


# ── saved results ─────────────────────────────────────────────────────────────

def test_results_survive_a_restart_and_identical_files_are_answered_from_them(tmp_path, scans):
    first = TestClient(make_app(tmp_path))
    job_id = post_scan(first, "one.png", scans[0])
    done = wait_done(first, job_id)
    assert done["state"] == "done"
    saved = tmp_path / "history" / "jobs" / job_id
    assert (saved / "job.json").is_file() and (saved / "images" / "overview.jpg").is_file()

    # a new app on the same folder: the scan is listed, its page data and images load from disk
    second = TestClient(make_app(tmp_path))
    listed = second.get("/api/history").json()
    assert listed["persistent"] and [s["id"] for s in listed["scans"]] == [job_id]
    assert second.get(f"/api/jobs/{job_id}").json()["result"]["verdict"] == done["result"]["verdict"]
    assert second.get(f"/api/jobs/{job_id}/{done['result']['overview']}").status_code == 200

    # the same bytes again: instant, under its own name, with images of its own (deleting one does not hurt the other)
    again_id = post_scan(second, "two.png", scans[0])
    again = second.get(f"/api/jobs/{again_id}").json()
    assert again["state"] == "done" and again["result"]["filename"] == "two.png" and again_id != job_id
    assert second.get(f"/api/jobs/{again_id}/{again['result']['overview']}").status_code == 200
    assert (tmp_path / "history" / "jobs" / again_id / "images" / "overview.jpg").is_file()


def test_a_changed_limit_does_not_reuse_an_old_answer(tmp_path, scans):
    old = TestClient(make_app(tmp_path))
    wait_done(old, post_scan(old, "one.png", scans[0]))
    new = TestClient(make_app(tmp_path, inspector=make_inspector(limit=0.5)))     # same models, another limit
    job_id = post_scan(new, "one.png", scans[0])
    assert new.get(f"/api/jobs/{job_id}").json()["state"] != "done"               # recomputed, not served from disk
    assert wait_done(new, job_id)["result"]["rois"]["ROI_2"]["threshold"] == 0.5


def test_changed_rejection_margin_recomputes_old_pass_and_matches_json_folder_csv_and_restart(tmp_path, scans):
    old = TestClient(make_app(tmp_path, inspector=make_inspector(limit=0.1)))
    old_id = post_scan(old, "original.png", scans[0])
    original = wait_done(old, old_id)["result"]
    assert original["verdict"] == "PASS" and original["rejection_margin"] == 0

    new = TestClient(make_app(tmp_path, inspector=make_inspector(limit=0.1, rejection_margin=0.005)))
    new_id = post_scan(new, "rechecked.png", scans[0])
    result = wait_done(new, new_id)
    assert not result["cached"] and result["result"]["verdict"] == "FAIL"
    r = result["result"]
    assert r["setup_id"] != original["setup_id"] and r["rejection_margin"] == 0.005
    assert [n for n, v in r["rois"].items() if v["status"] == "FAIL"] == ["ROI_2"]
    roi = r["rois"]["ROI_2"]
    assert roi["threshold"] == 0.1 and roi["decision_threshold"] == 0.095 and roi["margin"] < 0
    assert roi["rejection_reason"] == "near_threshold" and "rejection margin" in roi["summary"]
    assert new.get(f"/api/jobs/{new_id}/report.json").json() == r
    for endpoint in ("/api/status", "/api/about"):
        status = new.get(endpoint).json()
        assert status["rejection_margin"] == 0.005 and status["thresholds"] == {n: 0.1 for n in ROI_NAMES}
        assert all(t == pytest.approx(0.095) for t in status["decision_thresholds"].values())

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "borderline.png").write_bytes(scans[0])
    batch_id = new.post("/api/batches", json={"folder": str(folder)}, headers=UI).json()["batch_id"]
    batch = wait_batch(new, batch_id)
    assert batch["verdict"] == "FAIL" and batch["failed"] == 1
    assert batch["rows"][0]["rois"]["ROI_2"]["rejection_reason"] == "near_threshold"
    rows = list(csv.DictReader(io.StringIO(new.get(f"/api/batches/{batch_id}/report.csv").content.decode("utf-8-sig"))))
    assert rows[0]["verdict"] == "FAIL" and rows[0]["ROI_2_status"] == "FAIL"
    assert rows[0]["ROI_2_limit"] == "0.1" and rows[0]["ROI_2_cutoff"] == "0.095"
    assert rows[0]["ROI_2_rejection_reason"] == "near_threshold" and rows[0]["rejection_margin"] == "0.005"

    restarted = TestClient(make_app(tmp_path, inspector=make_inspector(limit=0.1, rejection_margin=0.005)))
    again_id = post_scan(restarted, "again.png", scans[0])
    again = wait_done(restarted, again_id)
    assert again["cached"] and again["result"]["rois"] == r["rois"]
    assert restarted.get(f"/api/jobs/{old_id}").json()["result"] == original  # historical evidence is not rewritten
    assert restarted.get(f"/api/batches/{batch_id}/report.csv").content == new.get(f"/api/batches/{batch_id}/report.csv").content


@pytest.mark.parametrize("change", ["template", "matching_limit"])
def test_changed_template_or_matcher_does_not_reuse_saved_verdicts(tmp_path, scans, change):
    old = TestClient(make_app(tmp_path))
    wait_done(old, post_scan(old, "one.png", scans[0]))
    inspector = make_inspector()
    if change == "template":
        inspector.matcher.templates["ROI_1"] = np.fliplr(inspector.matcher.templates["ROI_1"]).copy()
    else:
        inspector.matcher.min_score = 0.46
    new = TestClient(make_app(tmp_path, inspector=inspector))
    job_id = post_scan(new, "one.png", scans[0])
    assert not new.get(f"/api/jobs/{job_id}").json()["cached"]
    assert wait_done(new, job_id)["state"] == "done"


def test_changed_model_bytes_invalidate_cache_even_with_the_same_size_and_timestamp(tmp_path, scans):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"abcd")
    stat = model.stat()

    def inspector():
        result = make_inspector()
        for classifier in result.classifiers.values():
            classifier.model_path = model
        return result

    old = TestClient(make_app(tmp_path, inspector=inspector()))
    wait_done(old, post_scan(old, "one.png", scans[0]))
    model.write_bytes(b"efgh")
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    new = TestClient(make_app(tmp_path, inspector=inspector()))
    job_id = post_scan(new, "one.png", scans[0])
    assert not new.get(f"/api/jobs/{job_id}").json()["cached"]
    assert wait_done(new, job_id)["state"] == "done"


def test_only_the_newest_results_are_kept_on_disk(tmp_path, scans):
    client = TestClient(make_app(tmp_path, max_history=2))
    ids = [post_scan(client, f"s{i}.png", scans[i]) for i in range(3)]
    wait_done(client, ids[2])                                               # one worker, in order: the others are done too
    on_disk = sorted(p.name for p in (tmp_path / "history" / "jobs").iterdir())
    assert on_disk == sorted(ids[1:])
    assert client.get(f"/api/jobs/{ids[0]}").status_code == 404
    assert [s["id"] for s in client.get("/api/history").json()["scans"]] == [ids[2], ids[1]]    # newest first


def test_clear_history_needs_the_page_header_and_removes_everything(tmp_path, scans):
    client = TestClient(make_app(tmp_path))
    job_id = post_scan(client, "one.png", scans[0])
    wait_done(client, job_id)
    assert client.delete("/api/history").status_code == 403
    assert client.delete("/api/history", headers=UI).json()["removed"] == 1
    assert client.get("/api/history").json()["scans"] == [] and not (tmp_path / "history" / "jobs" / job_id).exists()


def test_clear_history_waits_for_a_result_write_and_it_cannot_reappear(tmp_path, monkeypatch):
    writing, release, clearing, cleared = (threading.Event() for _ in range(4))
    original = HistoryStore.save_job

    def slow_save(store, *args):
        writing.set()
        release.wait(5)
        original(store, *args)

    monkeypatch.setattr(HistoryStore, "save_job", slow_save)
    client = TestClient(make_app(tmp_path))
    post_scan(client, "scan.png", tiny_png(30))
    assert writing.wait(5)

    def clear():
        clearing.set()
        client.delete("/api/history", headers=UI)
        cleared.set()

    thread = threading.Thread(target=clear)
    thread.start()
    try:
        assert clearing.wait(5)
        assert not cleared.wait(0.1)
    finally:
        release.set()
        thread.join(10)
    assert cleared.is_set() and not list((tmp_path / "history" / "jobs").iterdir())
    assert TestClient(make_app(tmp_path)).get("/api/history").json()["scans"] == []


def test_a_torn_write_is_cleaned_up_at_start(tmp_path):
    (tmp_path / "history" / "jobs" / "deadbeef0000" / "images").mkdir(parents=True)        # images but no job.json
    app = make_app(tmp_path)
    assert TestClient(app).get("/api/history").json()["scans"] == []
    assert not (tmp_path / "history" / "jobs" / "deadbeef0000").exists()


# ── folder runs ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("state,rows,expected", [
    ("done", [("done", "PASS"), ("done", "PASS")], "PASS"),
    ("done", [("error", None), ("error", None)], None),
    ("done", [("done", "PASS"), ("error", None)], None),
    ("cancelled", [("done", "PASS"), ("skipped", None)], None),
    ("interrupted", [("skipped", None), ("skipped", None)], None),
    ("running", [("done", "PASS"), ("running", None)], None),
    ("done", [("done", "FAIL"), ("error", None)], "FAIL"),
    ("done", [], None),
])
def test_folder_verdict_passes_only_when_every_image_was_inspected_and_passed(tmp_path, state, rows, expected):
    batch = Batch(tmp_path, [tmp_path / f"{i}.png" for i in range(len(rows))])
    batch.state = state
    for row, (row_state, verdict) in zip(batch.rows, rows):
        row.update(state=row_state, verdict=verdict)
    assert batch.summary()["verdict"] == expected

def test_folder_run_inspects_every_image_and_exports_a_csv(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "a.png").write_bytes(tiny_png(30))
    (folder / "b.png").write_bytes(tiny_png(60))
    (folder / "b_again.png").write_bytes(tiny_png(60))                      # same bytes: answered from cache
    (folder / "broken.png").write_bytes(b"not an image")
    (folder / "notes.txt").write_text("ignored")
    client = TestClient(make_app(tmp_path))

    assert client.post("/api/batches", json={"folder": str(folder)}).status_code == 403          # needs the page header
    batch_id = client.post("/api/batches", json={"folder": str(folder)}, headers=UI).json()["batch_id"]
    b = wait_batch(client, batch_id)

    assert b["state"] == "done" and b["total"] == 4 and b["done"] == 4 and b["errors"] == 1
    rows = {r["file"]: r for r in b["rows"]}
    assert set(rows) == {"a.png", "b.png", "b_again.png", "broken.png"}
    assert rows["a.png"]["verdict"] == "FAIL" and rows["a.png"]["rois"]["ROI_1"]["status"] == "MISSING"   # 40x40: no region found
    assert rows["broken.png"]["state"] == "error" and "could not be read" in rows["broken.png"]["error"]
    assert all(rows[n]["has_detail"] for n in ("a.png", "b.png", "b_again.png"))
    assert client.get(f"/api/jobs/{rows['a.png']['job_id']}").json()["batch_id"] == batch_id

    csv_text = client.get(f"/api/batches/{batch_id}/report.csv").content.decode("utf-8-sig")
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("file,verdict,reasons,seconds,error,ROI_1_status,ROI_1_score,ROI_1_limit")
    assert len(lines) == 5 and any(l.startswith("broken.png,error") for l in lines)

    listed = client.get("/api/history").json()
    assert listed["scans"] == [] and [x["id"] for x in listed["batches"]] == [batch_id]         # folder items stay out of the scan list


def test_folder_run_can_be_stopped_and_survives_a_restart(tmp_path, scans):
    folder = tmp_path / "scans"
    folder.mkdir()
    for i, data in enumerate(scans):
        (folder / f"s{i}.png").write_bytes(data)
    gate = threading.Event()
    client = TestClient(make_app(tmp_path, inspector=make_inspector(gate)))
    batch_id = client.post("/api/batches", json={"folder": str(folder)}, headers=UI).json()["batch_id"]

    assert client.post("/api/batches", json={"folder": str(folder)}, headers=UI).status_code == 409   # one run at a time
    assert client.post(f"/api/batches/{batch_id}/cancel").status_code == 403
    assert client.post(f"/api/batches/{batch_id}/cancel", headers=UI).status_code == 200
    gate.set()                                                                  # the scan already in the worker finishes
    b = wait_batch(client, batch_id)
    assert b["state"] == "cancelled"
    assert [r["state"] for r in b["rows"]] == ["done", "skipped", "skipped"]

    again = TestClient(make_app(tmp_path))                                      # the run is still listed after a restart
    assert again.get(f"/api/batches/{batch_id}").json()["state"] == "cancelled"


def test_a_run_that_was_cut_short_by_a_restart_is_marked_interrupted(tmp_path):
    (tmp_path / "history" / "batches").mkdir(parents=True)
    saved = {"id": "abc123", "folder": str(tmp_path), "name": "x", "state": "running", "created": 1.0, "total": 1, "done": 0,
             "passed": 0, "failed": 0, "errors": 0,
             "rows": [{"file": "a.png", "state": "running", "job_id": None, "verdict": None, "reasons": [], "seconds": None, "rois": {}, "error": None}]}
    (tmp_path / "history" / "batches" / "abc123.json").write_text(json.dumps(saved))
    client = TestClient(make_app(tmp_path))
    interrupted = client.get("/api/batches/abc123").json()
    assert interrupted["state"] == "interrupted" and interrupted["verdict"] is None
    assert interrupted["rows"][0]["state"] == "skipped"


def test_folder_run_continues_in_memory_when_saving_fails(tmp_path, monkeypatch):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "a.png").write_bytes(tiny_png(30))
    (folder / "b.png").write_bytes(tiny_png(60))
    client = TestClient(make_app(tmp_path))

    def fail_save(*_):
        raise OSError("Disk full")

    monkeypatch.setattr(HistoryStore, "save_batch", fail_save)
    response = client.post("/api/batches", json={"folder": str(folder)}, headers=UI)
    assert response.status_code == 200
    batch = wait_batch(client, response.json()["batch_id"])
    assert batch["state"] == "done" and batch["done"] == 2 and "could not be saved" in batch["warning"]
    assert not client.app.state.is_busy()
    assert client.post("/api/batches", json={"folder": str(folder)}, headers=UI).status_code == 200


def test_folder_size_limit_and_csv_formula_safety(tmp_path, monkeypatch):
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "=formula.png").write_bytes(b"not an image")
    (folder / "oversized.bmp").write_bytes(b"x" * 33)
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 32)
    client = TestClient(make_app(tmp_path))
    batch_id = client.post("/api/batches", json={"folder": str(folder)}, headers=UI).json()["batch_id"]
    batch = wait_batch(client, batch_id)
    assert batch["errors"] == 2 and batch["verdict"] is None
    assert "larger than 120 MB" in batch["rows"][1]["error"]
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/batches/{batch_id}/report.csv").content.decode("utf-8-sig"))))
    assert rows[0]["file"] == "'=formula.png"
    assert rows[0]["threshold_status"] == "PROVISIONAL: test" and rows[0]["setup_id"]


def test_upload_queue_is_bounded_and_recovers(tmp_path, scans):
    gate = threading.Event()
    client = TestClient(make_app(tmp_path, inspector=make_inspector(gate)))
    try:
        ids = [post_scan(client, f"s{i}.png", scans[i]) for i in range(3)]
        refused = client.post("/api/jobs", files={"file": ("extra.png", scans[0])})
        assert refused.status_code == 429
    finally:
        gate.set()
    assert all(wait_done(client, job_id)["state"] == "done" for job_id in ids)


def test_upload_byte_and_decoded_pixel_limits(tmp_path, monkeypatch):
    client = TestClient(make_app(tmp_path))
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 32)
    assert client.post("/api/jobs", files={"file": ("large.bmp", b"x" * 33)}).status_code == 413
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(server, "MAX_IMAGE_PIXELS", 100)
    job = wait_done(client, post_scan(client, "large.png", tiny_png(30)))
    assert job["state"] == "error" and "pixel limit" in job["error"]


def test_json_report_preserves_setup_and_provisional_warning(tmp_path):
    app = create_app(make_inspector(), {"score_stat": "mean", "status": "PROVISIONAL: test",
                                      "warning": "Not validated on unseen scans."}, "models/test",
                     app_info={"version": "9.9", "profile": "p.json"})
    client = TestClient(app)
    job_id = post_scan(client, "scan.png", tiny_png(30))
    wait_done(client, job_id)
    report = client.get(f"/api/jobs/{job_id}/report.json").json()
    assert report["job_id"] == job_id and report["completed_at"]
    assert report["profile"] == "p.json" and report["app_version"] == "9.9"
    assert report["score_stat"] == "mean" and report["setup_id"] and set(report["model_hashes"]) == set(ROI_NAMES)
    assert report["threshold_warning"] == "Not validated on unseen scans."


def test_folder_run_rejects_a_missing_or_empty_folder(tmp_path):
    client = TestClient(make_app(tmp_path))
    assert client.post("/api/batches", json={"folder": str(tmp_path / "nope")}, headers=UI).status_code == 404
    empty = tmp_path / "empty"
    empty.mkdir()
    r = client.post("/api/batches", json={"folder": str(empty)}, headers=UI)
    assert r.status_code == 400 and "no bmp" in r.json()["detail"].lower()


# ── guards ────────────────────────────────────────────────────────────────────

def test_requests_with_a_foreign_host_header_are_refused(tmp_path):
    foreign = TestClient(make_app(tmp_path, allowed_hosts=["127.0.0.1"]))        # TestClient says Host: testserver
    assert foreign.get("/api/status").status_code == 400
    local = TestClient(make_app(tmp_path, allowed_hosts=["testserver"]))
    assert local.get("/api/status").status_code == 200


def test_open_folder_only_opens_folders_the_launcher_named(tmp_path):
    opened = []
    client = TestClient(make_app(tmp_path, folders={"home": tmp_path}, open_folder=opened.append))
    assert client.post("/api/open-folder", json={"key": "home"}).status_code == 403
    assert client.post("/api/open-folder", json={"key": "home"}, headers=UI).status_code == 200 and opened == [tmp_path]
    assert client.post("/api/open-folder", json={"key": "../../Windows"}, headers=UI).status_code == 404
    assert opened == [tmp_path]
    plain = TestClient(make_app(tmp_path, folders={"home": tmp_path}))             # no opener (a browser session)
    assert plain.post("/api/open-folder", json={"key": "home"}, headers=UI).status_code == 404


def test_about_reports_models_folders_and_saved_results(tmp_path, scans):
    client = TestClient(make_app(tmp_path, app_info={"version": "9.9", "frozen": True, "profile": "p.json"},
                                 folders={"home": tmp_path}, open_folder=lambda p: None))
    wait_done(client, post_scan(client, "one.png", scans[0]))
    about = client.get("/api/about").json()
    assert about["version"] == "9.9" and about["folders"] == {"home": str(tmp_path)} and about["can_open_folders"]
    assert set(about["models"]) == set(ROI_NAMES) and about["history"]["saved"] and about["history"]["count"] == 1
    assert about["history"]["bytes"] > 0
    assert client.get("/api/status").json()["desktop"] is True


# ── launcher ──────────────────────────────────────────────────────────────────

def test_packaged_app_looks_for_models_and_config_next_to_the_exe(tmp_path, monkeypatch):
    monkeypatch.delenv("HIMALAYA_HOME", raising=False)
    monkeypatch.setattr(paths, "FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Himalaya" / "Himalaya.exe"))
    (tmp_path / "Himalaya").mkdir()
    home = paths.app_home()
    assert home == (tmp_path / "Himalaya").resolve()
    assert paths.config_dir(home) == paths.PKG_ROOT / "config"                    # no templates shipped there: fall back
    (home / "config" / "roi_templates").mkdir(parents=True)
    assert paths.config_dir(home) == home / "config"                              # shipped next to the exe: used


def test_home_and_data_folders_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("HIMALAYA_HOME", str(tmp_path / "h"))
    monkeypatch.setenv("HIMALAYA_DATA_DIR", str(tmp_path / "d"))
    assert paths.app_home() == (tmp_path / "h").resolve() and paths.data_dir() == (tmp_path / "d").resolve()
    assert paths.data_dir(tmp_path / "x") == (tmp_path / "x").resolve()           # an explicit argument wins


def test_error_page_shows_the_message_without_running_it(tmp_path):
    page = launcher.error_html("missing <script>alert(1)</script> folder", tmp_path / "app.log")
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_missing_webview2_uses_the_browser_not_a_blank_legacy_window(tmp_path, monkeypatch):
    import webview

    class Event:
        def __iadd__(self, _):
            return self

    destroyed = []
    window = SimpleNamespace(events=SimpleNamespace(closing=Event()), destroy=lambda: destroyed.append(True))
    monkeypatch.setattr(webview, "create_window", lambda *a, **k: window)
    monkeypatch.setattr(webview, "renderer", "mshtml")
    monkeypatch.setattr(webview, "start", lambda boot, **k: boot())
    fallback = []
    monkeypatch.setattr(launcher, "run_browser", lambda *a, **k: fallback.append(k["reason"]) or 7)
    rt = SimpleNamespace(data=tmp_path)
    assert launcher.run_window(rt, tmp_path / "app.log") == 7
    assert destroyed and "WebView2" in fallback[0]


def test_missing_models_or_limits_give_a_message_not_a_traceback(tmp_path):
    args = launcher.parse_args(["--home", str(tmp_path)])                         # no models/ in there
    with pytest.raises(launcher.AppError, match="models folder is missing"):
        launcher.Runtime(args, tmp_path / "data").load()
    args = launcher.parse_args(["--thresholds", str(tmp_path / "none.json")])
    with pytest.raises(launcher.AppError, match="limits file is missing"):
        launcher.Runtime(args, tmp_path / "data").load()


def test_launcher_serves_the_app_on_a_free_port_and_stops(tmp_path, monkeypatch):
    (tmp_path / "home" / "models" / "gray_scale_rois").mkdir(parents=True)
    monkeypatch.setattr(TemplateROIInspector, "from_config", classmethod(lambda cls, *a, **k: make_inspector()))
    args = launcher.parse_args(["--home", str(tmp_path / "home")])
    rt = launcher.Runtime(args, tmp_path / "data")
    url = rt.start()
    try:
        assert url.startswith("http://127.0.0.1:") and not url.endswith(":0/")
        with urllib.request.urlopen(url + "api/status", timeout=10) as r:
            status = json.loads(r.read())
        assert status["desktop"] and status["persistent"] and status["samples"] == []
        with urllib.request.urlopen(url, timeout=10) as r:
            assert b"Himalaya" in r.read()
        assert not rt.busy()
    finally:
        rt.stop()
    assert (tmp_path / "data" / "history").is_dir()
