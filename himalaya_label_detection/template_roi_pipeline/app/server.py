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

One worker thread runs the inspections one at a time (the ONNX models already use every core),
so a second upload waits in "queued". Finished jobs are kept in memory (newest MAX_JOBS), and an
identical file uploaded again is answered from cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from .. import explain
from ..inspector import TemplateROIInspector
from ..roi_template_matcher import ROI_NAMES

STATIC = Path(__file__).parent / "static"
MAX_JOBS = 12
MAX_UPLOAD_BYTES = 120 * 1024 * 1024
EXPECTED_SIZE = (1504, 8000)          # (width, height) of an NSC scan
STEPS = ["decode", "locate", *ROI_NAMES, "explain"]


class SampleRequest(BaseModel):
    name: str


class Job:
    def __init__(self, filename: str):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.state = "queued"                       # queued | running | done | error
        self.steps = {s: "pending" for s in STEPS}  # pending | active | done
        self.started = time.time()
        self.elapsed = 0.0
        self.result: Optional[dict] = None
        self.images: Dict[str, bytes] = {}
        self.error: Optional[str] = None

    def set_step(self, step: str, status: str) -> None:
        self.steps[step] = status

    def public(self) -> dict:
        elapsed = self.elapsed if self.state in ("done", "error") else time.time() - self.started
        done = sum(s == "done" for s in self.steps.values())
        return {
            "id": self.id, "filename": self.filename, "state": self.state, "steps": self.steps,
            "progress": round(done / len(STEPS), 3), "elapsed_s": round(elapsed, 1),
            "error": self.error, "result": self.result,
        }


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
) -> FastAPI:
    app = FastAPI(title="Himalaya label inspection")
    jobs: "OrderedDict[str, Job]" = OrderedDict()
    cache: Dict[str, str] = {}                      # content hash -> job id
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)
    samples = samples or {}
    color = inspector.input_mode == "color"
    # the folder holding weights/model.onnx, e.g. "new_roi4"; shown so it is visible which model judged which ROI
    model_names = {n: getattr(c, "model_path", Path(n) / "weights" / "model.onnx").parent.parent.name
                   for n, c in inspector.classifiers.items()}

    def remember(job: Job) -> None:
        with lock:
            jobs[job.id] = job
            while len(jobs) > MAX_JOBS:
                old_id, _ = jobs.popitem(last=False)
                for h, jid in list(cache.items()):
                    if jid == old_id:
                        del cache[h]

    # ── Worker ────────────────────────────────────────────────────────────────

    def run(job: Job, data: bytes, digest: str) -> None:
        job.state = "running"
        t0 = job.started = time.time()                          # the clock excludes time spent queued
        try:
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
            job.state = "done"
            with lock:
                cache[digest] = job.id
        except Exception as exc:                                # surfaced to the UI, never crashes the worker
            # ValueErrors are our own readable messages (bad file, wrong colour depth); anything else keeps its type
            job.state, job.error = "error", str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
        finally:
            job.elapsed = time.time() - t0

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

    def submit(filename: str, data: bytes) -> Job:
        digest = hashlib.sha1(data).hexdigest() + inspector.input_mode
        with lock:
            hit = jobs.get(cache.get(digest, ""))
        if hit is not None and hit.state == "done":
            # same bytes, same models: nothing to recompute, but this request keeps its own name
            clone = Job(filename)
            clone.state, clone.steps, clone.images = "done", dict.fromkeys(STEPS, "done"), hit.images
            clone.result = {**hit.result, "filename": filename}
            remember(clone)
            return clone
        job = Job(filename)
        remember(job)
        pool.submit(run, job, data, digest)
        return job

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
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
            "jobs": list(jobs),                                 # ids still in memory: the page's "recent results" skips the rest
        }

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
        data = get_job(job_id).images.get(key)
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

    return app
