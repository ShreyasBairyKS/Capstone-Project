"""
history.py
─────────────────────────────────────────────────────────────────────────────
Finished inspections kept on disk, so they survive a restart of the app.

    <root>/jobs/<job id>/job.json          result + metadata (written last: a folder without it is a torn write)
    <root>/jobs/<job id>/images/<key>.jpg  the crops, heat views and overview of that scan
    <root>/batches/<batch id>.json         one folder run (its rows point at job ids)

The server decides what to keep (newest N); this module only reads and writes files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_SAFE = re.compile(r"^[A-Za-z0-9_]+$")          # job ids and image keys: never part of a path unless they match


class HistoryStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.batches_dir = self.root / "batches"
        for d in (self.jobs_dir, self.batches_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── jobs ──────────────────────────────────────────────────────────────────

    def _job_dir(self, job_id: str) -> Path:
        if not _SAFE.match(job_id):
            raise ValueError(f"bad job id {job_id!r}")
        return self.jobs_dir / job_id

    def save_job(self, job_id: str, meta: dict, images: Dict[str, bytes]) -> None:
        d = self._job_dir(job_id)
        (d / "images").mkdir(parents=True, exist_ok=True)
        for key, data in images.items():
            if _SAFE.match(key):
                (d / "images" / f"{key}.jpg").write_bytes(data)
        _write_json(d / "job.json", meta)

    def clone_job(self, src_id: str, dst_id: str, meta: dict) -> None:
        """Give a new job its own copy of another job's images (same file inspected again)."""
        shutil.copytree(self._job_dir(src_id) / "images", self._job_dir(dst_id) / "images")
        _write_json(self._job_dir(dst_id) / "job.json", meta)

    def read_image(self, job_id: str, key: str) -> Optional[bytes]:
        if not (_SAFE.match(job_id) and _SAFE.match(key)):
            return None
        try:
            return (self.jobs_dir / job_id / "images" / f"{key}.jpg").read_bytes()
        except OSError:
            return None

    def load_jobs(self) -> List[dict]:
        """Every complete job, oldest first. Folders without a readable job.json are leftovers and are removed."""
        found = []
        for d in self.jobs_dir.iterdir():
            if not d.is_dir():
                continue
            try:
                found.append(json.loads((d / "job.json").read_text(encoding="utf-8")))
            except (OSError, ValueError):
                shutil.rmtree(d, ignore_errors=True)
        return sorted(found, key=lambda m: m.get("finished", 0))

    def delete_job(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    # ── batches ───────────────────────────────────────────────────────────────

    def save_batch(self, batch: dict) -> None:
        if _SAFE.match(batch["id"]):
            _write_json(self.batches_dir / f"{batch['id']}.json", batch)

    def load_batches(self) -> List[dict]:
        out = []
        for f in self.batches_dir.glob("*.json"):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                f.unlink(missing_ok=True)
        return sorted(out, key=lambda b: b.get("created", 0))

    def delete_batch(self, batch_id: str) -> None:
        if _SAFE.match(batch_id):
            (self.batches_dir / f"{batch_id}.json").unlink(missing_ok=True)

    # ── housekeeping ──────────────────────────────────────────────────────────

    def clear(self, keep_jobs: Iterable[str] = (), keep_batches: Iterable[str] = ()) -> None:
        keep_jobs, keep_batches = set(keep_jobs), set(keep_batches)
        for d in self.jobs_dir.iterdir():
            if d.is_dir() and d.name not in keep_jobs:
                shutil.rmtree(d, ignore_errors=True)
        for f in self.batches_dir.glob("*.json"):
            if f.stem not in keep_batches:
                f.unlink(missing_ok=True)

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)
