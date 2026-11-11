"""
server.py
─────────────────────────────────────────────────────────────────────────────
Web app for the template-matching ROI pipeline: upload a full scan, watch it
being cropped and scored, read the verdict with the evidence behind it.

    browser  --upload-->  POST /api/jobs
                          background worker: decode -> locate 4 ROIs -> score each -> explain
             <--poll----  GET  /api/jobs/{id}      (progress, then the full result)
             <--images--  GET  /api/jobs/{id}/images/{key}.jpg
             <--report--  GET  /api/jobs/{id}/report.json
             <--list----  GET  /api/history        (finished scans, newest first, and folder runs)

    folder run           POST /api/batches {folder}      one scan after another, from a folder on this machine
                         GET  /api/batches/{id}          (rows fill in as scans finish)
                         GET  /api/batches/{id}/report.csv
                         POST /api/batches/{id}/cancel

One worker thread runs the inspections one at a time (the ONNX models already use every core),
so a second upload waits in "queued". A folder run feeds the worker one file at a time, so a scan
uploaded meanwhile waits for at most one folder item. An identical file inspected again with the same
models and limits is answered from cache.

Results are kept in memory (newest MAX_JOBS), or, when a history_dir is given (the desktop app),
on disk (newest max_history) so they survive a restart. The desktop app also passes:
  app_info / folders / open_folder   for the About dialog and its "open folder" buttons
  allowed_hosts                      requests whose Host header is not on the list are refused (DNS rebinding)
Routes that act on this machine (folder runs, opening folders, clearing results) additionally require
the X-Requested-With header the page sends, which another web page cannot add to a cross-site request.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .. import explain
from ..inspector import TemplateROIInspector
from ..roi_template_matcher import ROI_NAMES
from .history import HistoryStore

log = logging.getLogger("himalaya")
STATIC = Path(__file__).parent / "static"
MAX_JOBS = 12                          # results kept in memory when nothing is saved to disk
MAX_UPLOAD_BYTES = 120 * 1024 * 1024
MAX_BATCH_FILES = 5000
EXPECTED_SIZE = (1504, 8000)          # (width, height) of an NSC scan
STEPS = ["decode", "locate", *ROI_NAMES, "explain"]
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class SampleRequest(BaseModel):
    name: str


class BatchRequest(BaseModel):
    folder: str


class FolderRequest(BaseModel):
    key: str


class Job:
    def __init__(self, filename: str):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.state = "queued"                       # queued | running | done | error
        self.steps = {s: "pending" for s in STEPS}  # pending | active | done
        self.started = time.time()
        self.finished = 0.0
        self.elapsed = 0.0
        self.result: Optional[dict] = None
        self.images: Dict[str, bytes] = {}          # empty once the images are on disk
        self.on_disk = False
        self.error: Optional[str] = None
        self.digest = ""                            # content hash + setup id: what makes a cached answer valid
        self.batch_id: Optional[str] = None
        self.batch: Optional["Batch"] = None
        self.row = -1
        self.cancelled = False
        self.cached = False                         # answered from an earlier identical inspection

    def set_step(self, step: str, status: str) -> None:
        self.steps[step] = status

    def public(self) -> dict:
        elapsed = self.elapsed if self.state in ("done", "error") else time.time() - self.started
        done = sum(s == "done" for s in self.steps.values())
        return {
            "id": self.id, "filename": self.filename, "state": self.state, "steps": self.steps,
            "progress": round(done / len(STEPS), 3), "elapsed_s": round(elapsed, 1),
            "error": self.error, "result": self.result, "batch_id": self.batch_id,
        }

    def meta(self) -> dict:
        """What is written to disk for a finished job (the images go next to it)."""
        return {"id": self.id, "filename": self.filename, "digest": self.digest, "finished": self.finished,
                "elapsed": self.elapsed, "batch_id": self.batch_id, "result": self.result}

    @classmethod
    def from_meta(cls, meta: dict) -> "Job":
        job = cls(meta["filename"])
        job.id, job.digest, job.finished = meta["id"], meta.get("digest", ""), meta.get("finished", 0.0)
        job.elapsed, job.batch_id, job.result = meta.get("elapsed", 0.0), meta.get("batch_id"), meta["result"]
        job.state, job.steps, job.on_disk = "done", dict.fromkeys(STEPS, "done"), True
        return job


class Batch:
    """One run over the images of a folder."""

    def __init__(self, folder: Path, files: List[Path]):
        self.id = uuid.uuid4().hex[:12]
        self.folder = str(folder)
        self.files = files                          # not saved: a restarted app cannot resume a run
        self.created = time.time()
        self.state = "running"                      # running | done | cancelled | interrupted
        self.cancelled = False
        self.next = 0
        self.rows = [{"file": f.name, "state": "pending", "job_id": None, "verdict": None, "reasons": [],
                      "seconds": None, "rois": {}, "error": None} for f in files]

    def counts(self) -> dict:
        done = [r for r in self.rows if r["state"] in ("done", "error")]
        return {"total": len(self.rows), "done": len(done),
                "passed": sum(r["verdict"] == "PASS" for r in self.rows),
                "failed": sum(r["verdict"] == "FAIL" for r in self.rows),
                "errors": sum(r["state"] == "error" for r in self.rows)}

    def summary(self) -> dict:
        return {"id": self.id, "folder": self.folder, "name": Path(self.folder).name or self.folder,
                "state": self.state, "created": self.created, **self.counts()}

    def to_dict(self) -> dict:
        return {**self.summary(), "rows": self.rows}

    @classmethod
    def from_dict(cls, d: dict) -> "Batch":
        b = cls(Path(d["folder"]), [])
        b.id, b.created, b.rows = d["id"], d.get("created", 0.0), d["rows"]
        b.state = "interrupted" if d["state"] == "running" else d["state"]
        return b


def _jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def create_app(
    inspector: TemplateROIInspector,
    threshold_doc: dict,
    models_root: str,
    samples: Optional[Dict[str, Path]] = None,
    *,
    history_dir: Optional[Path] = None,
    max_history: int = 100,
    app_info: Optional[dict] = None,
    folders: Optional[Dict[str, Path]] = None,
    open_folder: Optional[Callable[[Path], None]] = None,
    allowed_hosts: Optional[List[str]] = None,
) -> FastAPI:
    app = FastAPI(title="Himalaya label inspection")
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    jobs: "OrderedDict[str, Job]" = OrderedDict()
    batches: "OrderedDict[str, Batch]" = OrderedDict()
    cache: Dict[str, str] = {}                      # content hash + setup id -> job id
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=1)
    samples = samples or {}
    folders = folders or {}
    store = HistoryStore(history_dir) if history_dir else None
    limit_jobs = max_history if store else MAX_JOBS
    color = inspector.input_mode == "color"
    # the folder holding weights/model.onnx, e.g. "new_roi4"; shown so it is visible which model judged which ROI
    model_names = {n: getattr(c, "model_path", Path(n) / "weights" / "model.onnx").parent.parent.name
                   for n, c in inspector.classifiers.items()}

    def setup_id() -> str:
        """Identifies the models, limits and input mode: a cached answer is only valid for the setup that made it."""
        parts: dict = {"mode": inspector.input_mode, "stat": threshold_doc.get("score_stat", "mean"),
                       "limits": inspector.thresholds}
        for n, c in inspector.classifiers.items():
            path = getattr(c, "model_path", None)
            try:
                st = Path(path).stat()
                parts[n] = [str(path), st.st_size, int(st.st_mtime)]
            except (TypeError, OSError):
                parts[n] = str(path)
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]

    setup = setup_id()

    # ── Bookkeeping ───────────────────────────────────────────────────────────

    def trim() -> None:
        """Drop the oldest finished results beyond the limit (unfinished ones are never dropped)."""
        with lock:
            for old_id in list(jobs):
                if len(jobs) <= limit_jobs:
                    break
                old = jobs[old_id]
                if old.state in ("queued", "running"):
                    continue
                del jobs[old_id]
                for h, jid in list(cache.items()):
                    if jid == old_id:
                        del cache[h]
                if store:
                    store.delete_job(old_id)

    def remember(job: Job) -> None:
        with lock:
            jobs[job.id] = job
            trim()

    def persist(job: Job) -> None:
        if store is None or job.state != "done":
            return
        store.save_job(job.id, job.meta(), job.images)
        job.on_disk = True
        job.images = {}                                          # served from disk from now on

    if store:
        saved = store.load_jobs()                               # oldest first
        for meta in saved[:-limit_jobs] if len(saved) > limit_jobs else []:
            store.delete_job(meta["id"])                        # the limit may have been lowered since they were saved
        for meta in saved[-limit_jobs:]:
            job = Job.from_meta(meta)
            jobs[job.id] = job
            if job.digest.endswith(setup):
                cache[job.digest] = job.id
        for d in store.load_batches():
            batch = Batch.from_dict(d)
            batches[batch.id] = batch

    def save_batch(batch: Batch) -> None:
        if store:
            store.save_batch(batch.to_dict())

    # ── Worker ────────────────────────────────────────────────────────────────

    def run(job: Job, data: bytes, digest: str) -> None:
        job.state = "running"
        t0 = job.started = time.time()                          # the clock excludes time spent queued
        try:
            if job.batch is not None and job.batch.cancelled:
                job.cancelled = True
                raise ValueError("Cancelled before it started.")
            job.set_step("decode", "active")
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("The file could not be read as an image (expected a BMP, PNG or JPEG scan).")
            h, w = image.shape[:2]
            warnings = []
            if (w, h) == EXPECTED_SIZE[::-1]:
                warnings.append(f"The image is {w}x{h}: it looks rotated by 90 degrees. Scans are expected to be "
                                f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} (width x height); ROI matching will probably fail.")
            elif (w, h) != EXPECTED_SIZE:
                warnings.append(f"The image is {w}x{h}, not the usual {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}. "
                                "Templates are matched at their original size, so results may be unreliable.")
            job.set_step("decode", "done")

            def progress(event: str, roi: Optional[str]) -> None:
                if event == "locating":
                    job.set_step("locate", "active")
                elif event == "located":
                    job.set_step("locate", "done")
                elif event == "scoring":
                    job.set_step(roi, "active")
                elif event == "scored":
                    job.set_step(roi, "done")

            result = inspector.inspect(image, image_path=job.filename, progress=progress)
            for name, v in result.rois.items():                 # MISSING ROIs never reach "scoring"
                job.set_step(name, "done")

            job.set_step("explain", "active")
            job.result, job.images = build_result(job, image, result, warnings)
            job.set_step("explain", "done")
            job.finished = time.time()
            job.elapsed = job.finished - t0
            job.state = "done"
            try:
                persist(job)
            except OSError:
                log.exception("Could not save the result of %s to disk; keeping it in memory", job.filename)
            with lock:
                cache[digest] = job.id
            trim()                                              # jobs queued together are only counted once they finish
        except Exception as exc:                                # surfaced to the UI, never crashes the worker
            # ValueErrors are our own readable messages (bad file, wrong colour depth); anything else keeps its type
            job.state, job.error = "error", str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
        finally:
            job.elapsed = time.time() - t0
            if job.batch is not None:                           # a folder run moves on whatever happened here
                try:
                    batch_record(job.batch, job.row, job)
                except Exception:
                    log.exception("Could not record %s in its folder run", job.filename)
                try:
                    advance(job.batch)
                except Exception:
                    log.exception("Folder run %s could not continue", job.batch.id)

    def build_result(job: Job, image: np.ndarray, result, warnings: list) -> tuple[dict, Dict[str, bytes]]:
        images: Dict[str, bytes] = {}
        regions = {n: (explain.find_hot_regions(v.anomaly_map, v.threshold) if v.anomaly_map is not None else [])
                   for n, v in result.rois.items()}
        # Boxes are drawn only on failing ROIs: on a passing crop they would read as alarms the verdict denies.
        boxed = {n: (r if result.rois[n].status == "FAIL" else []) for n, r in regions.items()}
        rois = {}
        for name, v in result.rois.items():
            info = explain.explain_roi(v, regions[name])
            info["model"] = model_names[name]
            if v.crop is not None:
                alpha = 0.85 if v.status == "FAIL" else 0.5
                images[f"{name}_crop"] = _jpeg(explain.render_crop(v.crop))
                images[f"{name}_above"] = _jpeg(explain.render_overlay(
                    v.crop, v.anomaly_map, v.threshold, "above", boxed[name], alpha=alpha))
                images[f"{name}_full"] = _jpeg(explain.render_overlay(
                    v.crop, v.anomaly_map, v.threshold, "full", boxed[name]))
                info["images"] = {k: f"images/{name}_{k}.jpg" for k in ("crop", "above", "full")}
            else:
                info["images"] = None
            rois[name] = info
        overview, overview_boxes = explain.render_overview(image, result, boxed)
        images["overview"] = _jpeg(overview, 85)
        verdict = explain.explain_label(result, rois)
        h, w = image.shape[:2]
        out = {
            "filename": job.filename,
            "image_size": [w, h],
            "input_mode": inspector.input_mode,
            "verdict": result.verdict,
            "headline": verdict["headline"],
            "reasons": verdict["reasons"],
            "latency_ms": round(result.latency_ms, 1),
            "warnings": warnings,
            "overview": "images/overview.jpg",              # image paths are relative to /api/jobs/{id}/
            "overview_boxes": overview_boxes,
            "rois": rois,
        }
        return out, images

    def submit(filename: str, data: bytes, batch: Optional[Batch] = None, row: int = -1) -> Job:
        digest = hashlib.sha1(data).hexdigest() + setup
        with lock:
            hit = jobs.get(cache.get(digest, ""))
        if hit is not None and hit.state == "done":
            # same bytes, same models and limits: nothing to recompute, but this request keeps its own name
            clone = Job(filename)
            clone.state, clone.steps, clone.digest = "done", dict.fromkeys(STEPS, "done"), digest
            clone.cached = True
            clone.result = {**hit.result, "filename": filename}
            clone.finished = time.time()
            clone.batch, clone.batch_id, clone.row = batch, batch.id if batch else None, row
            try:
                if store:
                    store.clone_job(hit.id, clone.id, clone.meta())
                    clone.on_disk = True
                else:
                    clone.images = hit.images
            except OSError:
                hit = None                                      # its images are gone: compute it again below
            if hit is not None:
                remember(clone)
                return clone
        job = Job(filename)
        job.digest, job.batch, job.batch_id, job.row = digest, batch, batch.id if batch else None, row
        remember(job)
        pool.submit(run, job, data, digest)
        return job

    # ── Folder runs ───────────────────────────────────────────────────────────

    def batch_record(batch: Batch, i: int, job: Job) -> None:
        with lock:
            row = batch.rows[i]
            if job.state == "done":
                r = job.result
                row.update(state="done", job_id=job.id, verdict=r["verdict"], reasons=r["reasons"],
                           seconds=round(r["latency_ms"] / 1000, 1),
                           rois={n: {"status": v["status"], "score": v["score"], "limit": v["threshold"]}
                                 for n, v in r["rois"].items()})
            else:
                row.update(state="skipped" if job.cancelled else "error", error=job.error)
        save_batch(batch)

    def advance(batch: Batch) -> None:
        """Hand the worker the next file. Answers from cache finish at once, so keep going until one has to be computed."""
        while True:
            with lock:
                if batch.cancelled or batch.next >= len(batch.files):
                    batch.state = "cancelled" if batch.cancelled else "done"
                    for r in batch.rows:
                        if r["state"] == "pending":
                            r["state"] = "skipped"
                    save_batch(batch)
                    return
                i = batch.next
                batch.next += 1
                batch.rows[i]["state"] = "running"
            try:
                data = batch.files[i].read_bytes()
            except OSError as exc:
                with lock:
                    batch.rows[i].update(state="error", error=f"Could not read the file: {exc.strerror or exc}")
                continue
            job = submit(batch.files[i].name, data, batch=batch, row=i)
            if not job.cached:
                return                                          # the worker calls advance() when it finishes
            batch_record(batch, i, job)

    # ── Routes ────────────────────────────────────────────────────────────────

    def require_ui(request: Request) -> None:
        if request.headers.get("x-requested-with") != "himalaya":
            raise HTTPException(403, "This action is only available from the app's own page.")

    def busy() -> bool:
        with lock:
            return any(j.state in ("queued", "running") for j in jobs.values()) or \
                any(b.state == "running" for b in batches.values())

    app.state.is_busy = busy

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        with lock:
            ids = list(jobs)
        return {
            "input_mode": inspector.input_mode,
            "models": model_names,
            "models_root": models_root,
            "score_stat": threshold_doc.get("score_stat", "mean"),
            "thresholds": inspector.thresholds,
            "threshold_status": threshold_doc.get("status", ""),
            "threshold_warning": threshold_doc.get("warning", ""),
            "expected_size": list(EXPECTED_SIZE),
            "samples": [{"name": n, "label": n.split("/")[0]} for n in samples],
            "jobs": ids,                                        # ids still held: lets a page skip stale links
            "persistent": store is not None,
            "desktop": app_info is not None,
            "version": (app_info or {}).get("version"),
        }

    @app.get("/api/about")
    def about():
        with lock:
            n_jobs = sum(j.state == "done" for j in jobs.values())
        return {
            **(app_info or {}),
            "folders": {k: str(v) for k, v in folders.items()},
            "can_open_folders": open_folder is not None,
            "input_mode": inspector.input_mode,
            "score_stat": threshold_doc.get("score_stat", "mean"),
            "models": model_names,
            "thresholds": inspector.thresholds,
            "threshold_status": threshold_doc.get("status", ""),
            "threshold_warning": threshold_doc.get("warning", ""),
            "history": {"saved": store is not None, "count": n_jobs, "limit": limit_jobs,
                        "bytes": store.size_bytes() if store else 0},
        }

    @app.get("/api/history")
    def history():
        with lock:
            scans = [{"id": j.id, "filename": j.filename, "verdict": j.result["verdict"], "finished": j.finished,
                      "latency_ms": j.result["latency_ms"]}
                     for j in reversed(jobs.values()) if j.state == "done" and j.result and j.batch_id is None]
            runs = [b.summary() for b in reversed(batches.values())]
        return {"scans": scans, "batches": runs, "persistent": store is not None}

    @app.delete("/api/history")
    def clear_history(request: Request):
        require_ui(request)
        with lock:
            keep_jobs = {j.id for j in jobs.values() if j.state in ("queued", "running")}
            keep_batches = {b.id for b in batches.values() if b.state == "running"}
            removed = len(jobs) - len(keep_jobs)
            for jid in [j for j in jobs if j not in keep_jobs]:
                del jobs[jid]
            for bid in [b for b in batches if b not in keep_batches]:
                del batches[bid]
            cache.clear()
            if store:
                store.clear(keep_jobs, keep_batches)
        return {"removed": removed}

    @app.post("/api/jobs")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The file is larger than 120 MB.")
        return {"job_id": submit(file.filename or "upload", data).id}

    @app.post("/api/jobs/sample")
    def sample(req: SampleRequest):
        path = samples.get(req.name)                            # only names we listed ourselves, no path joining
        if path is None:
            raise HTTPException(404, "Unknown sample.")
        return {"job_id": submit(req.name, path.read_bytes()).id}

    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown or expired job.")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return JSONResponse(get_job(job_id).public(), headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/images/{key}.jpg")
    def job_image(job_id: str, key: str):
        job = get_job(job_id)
        data = job.images.get(key)
        if data is None and job.on_disk and store:
            data = store.read_image(job.id, key)
        if data is None:
            raise HTTPException(404, "No such image.")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/jobs/{job_id}/report.json")
    def job_report(job_id: str):
        job = get_job(job_id)
        if job.result is None:
            raise HTTPException(409, "The job has not finished.")
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(job.filename).stem) or "scan"   # filename comes from the client
        return Response(json.dumps(job.result, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_report.json"'})

    # ── Folder-run routes ─────────────────────────────────────────────────────

    def get_batch(batch_id: str) -> Batch:
        batch = batches.get(batch_id)
        if batch is None:
            raise HTTPException(404, "Unknown or expired folder run.")
        return batch

    @app.post("/api/batches")
    def create_batch(req: BatchRequest, request: Request):
        require_ui(request)
        folder = Path(req.folder).expanduser()
        if not folder.is_dir():
            raise HTTPException(404, "That folder does not exist.")
        try:
            files = sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
                           key=lambda f: (len(f.stem), f.stem.lower()))
        except OSError as exc:
            raise HTTPException(400, f"That folder cannot be read: {exc.strerror or exc}")
        if not files:
            raise HTTPException(400, "There are no BMP, PNG, JPEG or TIFF images in that folder.")
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(400, f"That folder has {len(files)} images; the limit is {MAX_BATCH_FILES} per run.")
        with lock:
            if any(b.state == "running" for b in batches.values()):
                raise HTTPException(409, "A folder is already being inspected. Stop it or wait for it to finish.")
            batch = Batch(folder, files)
            batches[batch.id] = batch
            while len(batches) > 20:                            # keep the list short; the CSV is the lasting output
                old_id, old = next(iter(batches.items()))
                if old.state == "running":
                    break
                del batches[old_id]
                if store:
                    store.delete_batch(old_id)
        save_batch(batch)
        advance(batch)
        return {"batch_id": batch.id}

    @app.get("/api/batches/{batch_id}")
    def batch_status(batch_id: str):
        batch = get_batch(batch_id)
        with lock:
            data = batch.summary()
            # a row's details are gone once its job was dropped from the history
            data["rows"] = [{**r, "has_detail": r["job_id"] in jobs} for r in batch.rows]
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.post("/api/batches/{batch_id}/cancel")
    def batch_cancel(batch_id: str, request: Request):
        require_ui(request)
        batch = get_batch(batch_id)
        with lock:
            batch.cancelled = True
            if batch.state == "running" and not any(
                    j.batch_id == batch.id and j.state in ("queued", "running") for j in jobs.values()):
                advance(batch)                                  # nothing in flight: close the run now
        return {"ok": True}

    @app.get("/api/batches/{batch_id}/report.csv")
    def batch_report(batch_id: str):
        batch = get_batch(batch_id)
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["file", "verdict", "reasons", "seconds", "error",
                    *[f"{n}_{k}" for n in ROI_NAMES for k in ("status", "score", "limit")]])
        with lock:
            for r in batch.rows:
                cells = []
                for n in ROI_NAMES:
                    v = r["rois"].get(n, {})
                    cells += [v.get("status", ""), "" if v.get("score") is None else v["score"],
                              "" if v.get("limit") is None else v["limit"]]
                w.writerow([r["file"], r["verdict"] or r["state"], "; ".join(r["reasons"]), r["seconds"] or "",
                            r["error"] or "", *cells])
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(batch.folder).name) or "folder"
        return Response("﻿" + out.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_results.csv"'})

    # ── Desktop-only ──────────────────────────────────────────────────────────

    @app.post("/api/open-folder")
    def open_folder_route(req: FolderRequest, request: Request):
        require_ui(request)
        path = folders.get(req.key)                             # only the folders the launcher named; no client path
        if open_folder is None or path is None:
            raise HTTPException(404, "That folder cannot be opened from here.")
        open_folder(Path(path))
        return {"ok": True}

    return app
