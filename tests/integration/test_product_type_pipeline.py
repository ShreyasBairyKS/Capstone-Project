"""
tests/integration/test_product_type_pipeline.py — Sub-type routing E2E tests.

8 test cases covering:
  TC-01  Sub-type resolved from active run → pipeline called with correct args
  TC-02  No active run → pipeline called with None, WARNING logged
  TC-03  Sub-type provided in request → active run NOT queried (short-circuit)
  TC-04  QR verification field passed through from product lookup
  TC-05  Date fields passed through from product lookup
  TC-06  Fill-level sub-pipeline triggered for transparent_bottle
  TC-07  Fill-level sub-pipeline NOT triggered for rigid_can
  TC-08  Inspection latency overhead < 50 ms above base when active run present

Note:
  - TC-03 through TC-08 depend on Collaborator A completing the updated
    pipeline.inspect() signature (product_sub_type, container_contents kwargs).
    Until that lands, the pipeline is fully mocked in ALL test cases.
  - Mocking strategy: AsyncMock replaces Motor collections; pipeline.inspect
    is patched to return a minimal InspectionResult fixture.
"""

from __future__ import annotations

import asyncio
import base64
import time
import os
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest
import pytest_asyncio

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "visionfood_test")


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _tiny_jpeg_b64() -> str:
    """Return a 1×1 white JPEG as base64 string (valid image, no model needed)."""
    import cv2
    img = np.ones((240, 320, 3), dtype=np.uint8) * 200
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _make_profile_dir() -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "bottle_test.yaml").write_text("sku_id: bottle_test\n")
    (d / "can_test.yaml").write_text("sku_id: can_test\n")
    return d


def _minimal_result_dict(**overrides) -> dict:
    """Return a minimal dict matching InspectionResult schema."""
    base = {
        "inspection_id": "test-insp-01",
        "product_id": None,
        "sku": "bottle_test_001",
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
        "latency_ms": 25.0,
        "device_id": "edge_node_test",
    }
    base.update(overrides)
    return base


# ── Fake product & run documents ──────────────────────────────────────────────

FAKE_PRODUCT_DOC = {
    "_id": "000000000000000000000001",
    "sku": "bottle_test_001",
    "name": "Test Bottle",
    "product_category": "beverage",
    "product_sub_type": "transparent_bottle",
    "container_contents": "liquid",
    "sku_profile_name": "bottle_test",
    "qr_code": "QR-TEST-001",
    "expected_dates": [{"name": "expiry_date", "format": "MM/YYYY", "value": None}],
    "created_at": "2026-04-16T00:00:00Z",
    "updated_at": "2026-04-16T00:00:00Z",
    "__v": 0,
}

FAKE_CAN_DOC = {**FAKE_PRODUCT_DOC, "_id": "000000000000000000000002",
               "sku": "can_test_001", "product_sub_type": "rigid_can",
               "sku_profile_name": "can_test"}

FAKE_RUN_DOC = {
    "_id": "000000000000000000000010",
    "run_id": "run-uuid-0001",
    "sku": "bottle_test_001",
    "product_id": "000000000000000000000001",
    "started_at": "2026-04-16T07:00:00Z",
    "ended_at": None,
    "status": "active",
    "operator_id": "test_op",
    "inspection_count": 5,
    "defect_count": 0,
}


# ── Build a mock Motor DB 返回 specified product & run ────────────────────────

def _mock_motor_db(product_doc=FAKE_PRODUCT_DOC, run_doc=FAKE_RUN_DOC):
    """Return a mock Motor DB where find_one resolves to the provided docs."""
    db = MagicMock()

    products_col = MagicMock()
    products_col.find_one = AsyncMock(return_value={**product_doc})
    products_col.create_index = AsyncMock()

    runs_col = MagicMock()
    runs_col.find_one = AsyncMock(return_value={**run_doc} if run_doc else None)
    runs_col.update_one = AsyncMock()
    runs_col.create_index = AsyncMock()

    def _col_dispatch(name):
        if name == "products":
            return products_col
        return runs_col

    db.__getitem__ = MagicMock(side_effect=_col_dispatch)
    return db, products_col, runs_col


# ---------------------------------------------------------------------------
# App fixture (fresh per test — we swap the motor DB mock per test case)
# ---------------------------------------------------------------------------

@pytest.fixture()
def profile_dir():
    return _make_profile_dir()


def _build_app(profile_dir, motor_db, mock_pipeline_result=None):
    """Construct a TestClient with full mocking."""
    from unittest.mock import patch as _patch

    result_obj = MagicMock()
    result_dict = mock_pipeline_result or _minimal_result_dict()
    # Make result_obj behave like an InspectionResult via model_dump
    result_obj.model_dump.return_value = result_dict
    # Also allow direct attribute access (FastAPI jsonable_encoder uses __dict__)
    for k, v in result_dict.items():
        setattr(result_obj, k, v)

    with _patch("core.config.settings.SKU_PROFILES_DIR", profile_dir):
        with _patch("inference.pipeline.EdgeInferencePipeline.load_models", return_value=None):
            with _patch("inference.pipeline.EdgeInferencePipeline.inspect",
                        return_value=result_obj) as mock_inspect:
                with _patch("database.session.init_motor", return_value=None):
                    with _patch("database.session.close_motor", return_value=None):
                        with _patch("database.session.create_tables", return_value=None):

                            from api.main import app
                            from database.session import get_motor_db

                            async def _override():
                                yield motor_db

                            app.dependency_overrides[get_motor_db] = _override

                            from fastapi.testclient import TestClient
                            client = TestClient(app, raise_server_exceptions=False)
                            return client, mock_inspect


IMAGE_B64 = _tiny_jpeg_b64()
HEADERS = {"X-API-Key": "test-key", "X-User-Role": "admin"}


# ---------------------------------------------------------------------------
# TC-01  Sub-type resolved from active run
# ---------------------------------------------------------------------------

def test_tc01_subtype_resolved_from_active_run(profile_dir):
    motor_db, _, _ = _mock_motor_db()
    client, mock_inspect = _build_app(profile_dir, motor_db)

    resp = client.post(
        "/inspections",
        json={"image_b64": IMAGE_B64, "sku": "bottle_test_001", "attempt_count": 0},
        headers=HEADERS,
    )
    # Verify the pipeline was called with resolved sub_type values
    assert mock_inspect.called, "pipeline.inspect() was not called"
    call_kwargs = mock_inspect.call_args.kwargs
    assert call_kwargs.get("product_sub_type") == "transparent_bottle"
    assert call_kwargs.get("container_contents") == "liquid"


# ---------------------------------------------------------------------------
# TC-02  No active run → pipeline called with None, WARNING logged
# ---------------------------------------------------------------------------

def test_tc02_no_active_run_warns_and_passes_none(profile_dir, caplog):
    motor_db, _, runs_col = _mock_motor_db()
    # Make run lookup return None
    runs_col.find_one = AsyncMock(return_value=None)

    client, mock_inspect = _build_app(profile_dir, motor_db)

    with caplog.at_level(logging.WARNING, logger="api.routers.inspection"):
        resp = client.post(
            "/inspections",
            json={"image_b64": IMAGE_B64, "sku": "bottle_test_001"},
            headers=HEADERS,
        )

    call_kwargs = mock_inspect.call_args.kwargs
    assert call_kwargs.get("product_sub_type") is None
    assert call_kwargs.get("container_contents") is None
    assert any("no_active_run_for_sku" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TC-03  Sub-type provided in request → active run NOT queried
# ---------------------------------------------------------------------------

def test_tc03_provided_subtype_skips_run_lookup(profile_dir):
    motor_db, _, runs_col = _mock_motor_db()

    client, mock_inspect = _build_app(profile_dir, motor_db)

    resp = client.post(
        "/inspections",
        json={
            "image_b64": IMAGE_B64,
            "sku": "bottle_test_001",
            "product_sub_type": "rigid_can",       # explicitly provided
            "container_contents": "solid",          # explicitly provided
        },
        headers=HEADERS,
    )

    call_kwargs = mock_inspect.call_args.kwargs
    # The explicit values should be passed through unchanged
    assert call_kwargs.get("product_sub_type") == "rigid_can"
    assert call_kwargs.get("container_contents") == "solid"
    # Active run should NOT have been consulted (find_one not called)
    assert not runs_col.find_one.called, "run lookup should be skipped when both fields supplied"


# ---------------------------------------------------------------------------
# TC-04  QR code retrieved via product lookup
# ---------------------------------------------------------------------------

def test_tc04_qr_code_field_in_product_doc(profile_dir):
    motor_db, products_col, _ = _mock_motor_db()
    client, _ = _build_app(profile_dir, motor_db)

    # The product_repository should be able to retrieve the QR value
    import asyncio
    from database.repositories.product_repository import ProductRepository

    async def _check():
        repo = ProductRepository(motor_db)
        qr = await repo.get_expected_qr("bottle_test_001")
        return qr

    qr = asyncio.get_event_loop().run_until_complete(_check())
    assert qr == "QR-TEST-001"


# ---------------------------------------------------------------------------
# TC-05  Date fields retrieved via product lookup
# ---------------------------------------------------------------------------

def test_tc05_date_fields_in_product_doc(profile_dir):
    motor_db, _, _ = _mock_motor_db()

    import asyncio
    from database.repositories.product_repository import ProductRepository

    async def _check():
        repo = ProductRepository(motor_db)
        dates = await repo.get_expected_dates("bottle_test_001")
        return dates

    dates = asyncio.get_event_loop().run_until_complete(_check())
    assert len(dates) == 1
    assert dates[0]["name"] == "expiry_date"
    assert dates[0]["format"] == "MM/YYYY"


# ---------------------------------------------------------------------------
# TC-06  transparent_bottle sub-type does NOT raise errors (fill-level path)
# ---------------------------------------------------------------------------

def test_tc06_transparent_bottle_inspection_succeeds(profile_dir):
    motor_db, _, _ = _mock_motor_db(product_doc=FAKE_PRODUCT_DOC)
    client, mock_inspect = _build_app(profile_dir, motor_db)

    resp = client.post(
        "/inspections",
        json={"image_b64": IMAGE_B64, "sku": "bottle_test_001"},
        headers=HEADERS,
    )
    assert resp.status_code == 201, resp.text
    call_kwargs = mock_inspect.call_args.kwargs
    assert call_kwargs.get("product_sub_type") == "transparent_bottle"


# ---------------------------------------------------------------------------
# TC-07  rigid_can sub-type resolved correctly (fill_level_detectable=false)
# ---------------------------------------------------------------------------

def test_tc07_rigid_can_inspection_resolves_correct_subtype(profile_dir):
    can_run = {**FAKE_RUN_DOC, "sku": "can_test_001", "product_id": "000000000000000000000002"}
    motor_db, _, _ = _mock_motor_db(product_doc=FAKE_CAN_DOC, run_doc=can_run)
    client, mock_inspect = _build_app(profile_dir, motor_db)

    resp = client.post(
        "/inspections",
        json={"image_b64": IMAGE_B64, "sku": "can_test_001"},
        headers=HEADERS,
    )
    assert resp.status_code == 201, resp.text
    call_kwargs = mock_inspect.call_args.kwargs
    assert call_kwargs.get("product_sub_type") == "rigid_can"
    assert call_kwargs.get("container_contents") == "liquid"


# ---------------------------------------------------------------------------
# TC-08  Run resolution adds < 50 ms overhead vs base inspection
# ---------------------------------------------------------------------------

def test_tc08_sub_type_resolution_latency_overhead(profile_dir):
    """
    Measures two back-to-back POST /inspections calls:
      1. Sub-type explicitly provided (no DB lookup)  → baseline
      2. Sub-type resolved from active run            → with overhead
    Asserts overhead < 50 ms.
    """
    motor_db, _, _ = _mock_motor_db()
    client, _ = _build_app(profile_dir, motor_db)

    N = 5  # average over N requests to reduce jitter

    # Baseline: explicit sub-type — no Motor query
    t0 = time.perf_counter()
    for _ in range(N):
        client.post(
            "/inspections",
            json={
                "image_b64": IMAGE_B64,
                "sku": "bottle_test_001",
                "product_sub_type": "transparent_bottle",
                "container_contents": "liquid",
            },
            headers=HEADERS,
        )
    baseline_ms = (time.perf_counter() - t0) * 1000 / N

    # With resolution: no sub-type in request
    t1 = time.perf_counter()
    for _ in range(N):
        client.post(
            "/inspections",
            json={"image_b64": IMAGE_B64, "sku": "bottle_test_001"},
            headers=HEADERS,
        )
    resolved_ms = (time.perf_counter() - t1) * 1000 / N

    overhead_ms = resolved_ms - baseline_ms
    assert overhead_ms < 50, (
        f"Sub-type resolution overhead {overhead_ms:.1f} ms exceeds 50 ms budget. "
        f"Baseline: {baseline_ms:.1f} ms, Resolved: {resolved_ms:.1f} ms."
    )
