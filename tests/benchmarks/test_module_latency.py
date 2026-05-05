"""
tests/benchmarks/test_module_latency.py — API and pipeline latency benchmarks.

Benchmark functions (pytest-benchmark):
  bench_post_products      — POST /api/v1/products           target < 50 ms
  bench_get_active_run     — GET  /api/v1/runs/active        target < 20 ms
  bench_subtype_resolution — Sub-type resolution overhead    target < 50 ms above base inspect

Run with:
    pytest tests/benchmarks/test_module_latency.py --benchmark-only -v

Notes:
  - Motor DB is mocked with AsyncMock (in-memory) so benchmarks are network-free.
  - Pipeline inference is mocked — these benchmarks measure API routing overhead only.
  - Collaborator A will add their per-module inference benchmarks to this same file.
    Write new benchmark functions; do NOT remove or rename existing ones.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap environment before any app imports
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "visionfood_bench")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_jpeg_b64() -> str:
    import cv2
    img = np.ones((240, 320, 3), dtype=np.uint8) * 180
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


def _make_profile_dir() -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "bottle_bench.yaml").write_text("sku_id: bottle_bench\n")
    return d


FAKE_PRODUCT = {
    "_id": "000000000000000000000001",
    "sku": "bench_sku_001",
    "name": "Benchmark Product",
    "product_category": "beverage",
    "product_sub_type": "transparent_bottle",
    "container_contents": "liquid",
    "sku_profile_name": "bottle_bench",
    "qr_code": "QR-BENCH-001",
    "expected_dates": [],
    "created_at": "2026-04-16T00:00:00Z",
    "updated_at": "2026-04-16T00:00:00Z",
    "__v": 0,
}

FAKE_RUN = {
    "_id": "000000000000000000000010",
    "run_id": "bench-run-0001",
    "sku": "bench_sku_001",
    "product_id": "000000000000000000000001",
    "started_at": "2026-04-16T07:00:00Z",
    "ended_at": None,
    "status": "active",
    "operator_id": "bench_op",
    "inspection_count": 0,
    "defect_count": 0,
}

HEADERS = {"X-API-Key": "test-key", "X-User-Role": "admin"}
IMAGE_B64 = _tiny_jpeg_b64()

MINIMAL_RESULT = {
    "inspection_id": "bench-insp-01",
    "product_id": None,
    "sku": "bench_sku_001",
    "timestamp": "2026-04-16T08:00:00Z",
    "verdict": "PASS",
    "escalated": False,
    "detections": [],
    "uq_result": None,
    "severity_result": None,
    "remediation_action": None,
    "annotated_image_b64": None,
    "label_qr": None,
    "label_text": None,
    "product_category": None,
    "product_sub_type": None,
    "container_contents": None,
    "latency_ms": 10.0,
    "device_id": "edge_bench",
}


def _mock_motor_db():
    db = MagicMock()
    products = MagicMock()
    products.find_one = AsyncMock(return_value={**FAKE_PRODUCT})
    products.insert_one = AsyncMock(return_value=MagicMock(inserted_id="000000000000000000000099"))
    products.create_index = AsyncMock()

    runs = MagicMock()
    runs.find_one = AsyncMock(return_value={**FAKE_RUN})
    runs.update_one = AsyncMock()
    runs.create_index = AsyncMock()

    def _col(name):
        return products if name == "products" else runs

    db.__getitem__ = MagicMock(side_effect=_col)
    return db


def _build_test_client(profile_dir, motor_db):
    result_obj = MagicMock()
    for k, v in MINIMAL_RESULT.items():
        setattr(result_obj, k, v)

    with patch("core.config.settings.SKU_PROFILES_DIR", profile_dir):
        with patch("inference.pipeline.EdgeInferencePipeline.load_models", return_value=None):
            with patch("inference.pipeline.EdgeInferencePipeline.inspect",
                       return_value=result_obj):
                with patch("database.session.init_motor", return_value=None):
                    with patch("database.session.close_motor", return_value=None):
                        with patch("database.session.create_tables", return_value=None):
                            from api.main import app
                            from database.session import get_motor_db
                            from fastapi.testclient import TestClient

                            async def _override():
                                yield motor_db

                            app.dependency_overrides[get_motor_db] = _override
                            return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Session-scoped shared state (avoids re-building app on every benchmark call)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bench_client():
    profile_dir = _make_profile_dir()
    motor_db = _mock_motor_db()
    return _build_test_client(profile_dir, motor_db)


# ---------------------------------------------------------------------------
# Benchmark: POST /api/v1/products  (target < 50 ms)
# ---------------------------------------------------------------------------

def test_bench_post_products(benchmark, bench_client):
    """
    Benchmark POST /api/v1/products.
    Target: mean latency < 50 ms (mocked Motor insert).
    """
    import random
    import string

    def _post():
        # Use a unique SKU each call to avoid 409 conflicts
        sku = "bench_" + "".join(random.choices(string.ascii_lowercase, k=8))
        resp = bench_client.post(
            "/api/v1/products",
            json={
                "sku": sku,
                "name": "Benchmark Product",
                "product_category": "beverage",
                "product_sub_type": "transparent_bottle",
                "container_contents": "liquid",
                "sku_profile_name": "bottle_bench",
            },
            headers=HEADERS,
        )
        return resp.status_code

    result = benchmark(_post)
    # benchmark.stats["mean"] is in seconds
    mean_ms = benchmark.stats["mean"] * 1000
    assert mean_ms < 50, f"POST /products mean latency {mean_ms:.1f} ms exceeds 50 ms target"


# ---------------------------------------------------------------------------
# Benchmark: GET /api/v1/runs/active  (target < 20 ms)
# ---------------------------------------------------------------------------

def test_bench_get_active_run(benchmark, bench_client):
    """
    Benchmark GET /api/v1/runs/active.
    Target: mean latency < 20 ms (mocked Motor find_one).
    """
    def _get():
        resp = bench_client.get("/api/v1/runs/active", headers=HEADERS)
        return resp.status_code

    benchmark(_get)
    mean_ms = benchmark.stats["mean"] * 1000
    assert mean_ms < 20, f"GET /runs/active mean latency {mean_ms:.1f} ms exceeds 20 ms target"


# ---------------------------------------------------------------------------
# Benchmark: Sub-type resolution overhead  (target < 50 ms above base)
# ---------------------------------------------------------------------------

def test_bench_subtype_resolution_overhead(benchmark, bench_client):
    """
    Measures the overhead added by sub-type resolution (Motor run + product lookup)
    on top of a base inspection call where sub-type is explicitly provided.

    Target: resolution overhead < 50 ms.
    """
    # Baseline: explicit sub-type provided, no DB resolution
    N = 10

    import time

    t0 = time.perf_counter()
    for _ in range(N):
        bench_client.post(
            "/inspections",
            json={
                "image_b64": IMAGE_B64,
                "sku": "bench_sku_001",
                "product_sub_type": "transparent_bottle",
                "container_contents": "liquid",
            },
            headers=HEADERS,
        )
    baseline_ms = (time.perf_counter() - t0) * 1000 / N

    # Resolved: no sub-type — Motor lookup triggered
    def _resolved_inspect():
        bench_client.post(
            "/inspections",
            json={"image_b64": IMAGE_B64, "sku": "bench_sku_001"},
            headers=HEADERS,
        )

    benchmark(_resolved_inspect)
    resolved_ms = benchmark.stats["mean"] * 1000
    overhead_ms = resolved_ms - baseline_ms

    assert overhead_ms < 50, (
        f"Sub-type resolution overhead {overhead_ms:.1f} ms exceeds 50 ms target. "
        f"Baseline: {baseline_ms:.1f} ms, Resolved: {resolved_ms:.1f} ms."
    )
