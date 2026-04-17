"""
tests/integration/test_products_api.py — Products & Runs API integration tests.

10 test cases covering:
  TC-01  POST /products              → 201 (happy path)
  TC-02  POST /products              → 409 (duplicate SKU)
  TC-03  POST /products              → 422 (invalid SKU format)
  TC-04  POST /products              → 422 (unknown product_category)
  TC-05  POST /products              → 422 (sku_profile_name not on disk)
  TC-06  GET  /products              → 200 (list)
  TC-07  GET  /products/{sku}        → 200 (found)
  TC-08  GET  /products/{sku}        → 404 (not found)
  TC-09  PATCH /products/{sku}       → 409 (stale __v)
  TC-10  POST + GET runs active/end  → full run lifecycle

Mock strategy:
  - Motor DB is replaced with mongomock_motor (in-memory MongoDB).
  - SQLite :memory: is used for the SQLAlchemy layer (set in conftest.py).
  - The SKU profile directory is patched to a temp dir containing one YAML.
  - `get_motor_db` dependency is overridden on the FastAPI app.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ---------------------------------------------------------------------------
# Env vars must be set before importing app (conftest.py handles DATABASE_URL
# and API_KEY; we add MONGO_URL here as a no-op value — Motor will be mocked)
# ---------------------------------------------------------------------------
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "visionfood_test")

try:
    import mongomock_motor  # type: ignore[import]
    HAS_MONGOMOCK = True
except ImportError:
    HAS_MONGOMOCK = False

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile_dir() -> Path:
    """Create a temp dir with a single YAML SKU profile for testing."""
    d = Path(tempfile.mkdtemp())
    (d / "bottle_test.yaml").write_text("sku_id: bottle_test\n")
    return d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profile_dir() -> Path:
    return _make_profile_dir()


@pytest.fixture(scope="module")
def mongo_client():
    """Return an in-memory mongomock Motor client, or None if unavailable."""
    if not HAS_MONGOMOCK:
        pytest.skip("mongomock-motor not installed — skipping Motor integration tests")
    client = mongomock_motor.AsyncMongoMockClient()
    return client


@pytest.fixture(scope="module")
def mongo_db(mongo_client):
    return mongo_client["visionfood_test"]


@pytest.fixture(scope="module")
def app(profile_dir, mongo_db):
    """Build the FastAPI app with Motor and pipeline mocked."""
    # Patch the SKU profile dir in settings before importing app
    with patch("core.config.settings.SKU_PROFILES_DIR", profile_dir):
        # Patch pipeline loading so tests don't need ONNX models
        with patch("inference.pipeline.EdgeInferencePipeline.load_models", return_value=None):
            with patch("database.session.init_motor", return_value=None):
                with patch("database.session.close_motor", return_value=None):
                    from api.main import app as _app

                    # Override Motor DB dependency to use in-memory mock
                    from database.session import get_motor_db

                    async def _override_motor_db() -> AsyncGenerator:
                        yield mongo_db

                    _app.dependency_overrides[get_motor_db] = _override_motor_db
                    yield _app


@pytest_asyncio.fixture(scope="module")
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-API-Key": "test-key", "X-User-Role": "admin"},
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PRODUCT = {
    "sku": "bottle_test_001",
    "name": "Test Bottle 250ml",
    "description": "Integration test product",
    "product_category": "beverage",
    "product_sub_type": "transparent_bottle",
    "container_contents": "liquid",
    "sku_profile_name": "bottle_test",
    "qr_code": "QR-TEST-001",
    "expected_dates": [
        {"name": "expiry_date", "format": "MM/YYYY"},
    ],
}


# ---------------------------------------------------------------------------
# TC-01  POST /products → 201 happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc01_create_product_happy_path(client: AsyncClient):
    resp = await client.post("/api/v1/products", json=VALID_PRODUCT)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["sku"] == VALID_PRODUCT["sku"]
    assert body["product_category"] == "beverage"
    assert body["__v"] == 0
    assert "created_at" in body


# ---------------------------------------------------------------------------
# TC-02  POST /products → 409 duplicate SKU
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc02_duplicate_sku_returns_409(client: AsyncClient):
    resp = await client.post("/api/v1/products", json=VALID_PRODUCT)
    assert resp.status_code == 409, resp.text
    assert "already exists" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# TC-03  POST /products → 422 invalid SKU format
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc03_invalid_sku_format(client: AsyncClient):
    bad = {**VALID_PRODUCT, "sku": "HAS SPACES AND CAPS!"}
    resp = await client.post("/api/v1/products", json=bad)
    assert resp.status_code == 422, resp.text
    assert "sku" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# TC-04  POST /products → 422 invalid product_category
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc04_invalid_category(client: AsyncClient):
    bad = {**VALID_PRODUCT, "sku": "valid_sku_001", "product_category": "invalid_cat"}
    resp = await client.post("/api/v1/products", json=bad)
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# TC-05  POST /products → 422 sku_profile_name not on disk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc05_missing_sku_profile(client: AsyncClient):
    bad = {**VALID_PRODUCT, "sku": "valid_sku_002", "sku_profile_name": "nonexistent_profile"}
    resp = await client.post("/api/v1/products", json=bad)
    assert resp.status_code == 422, resp.text
    assert "nonexistent_profile" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# TC-06  GET /products → 200 and list contains our product
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc06_list_products(client: AsyncClient):
    resp = await client.get("/api/v1/products")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    skus = [p["sku"] for p in body]
    assert VALID_PRODUCT["sku"] in skus


# ---------------------------------------------------------------------------
# TC-07  GET /products/{sku} → 200
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc07_get_product_by_sku(client: AsyncClient):
    resp = await client.get(f"/api/v1/products/{VALID_PRODUCT['sku']}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["sku"] == VALID_PRODUCT["sku"]


# ---------------------------------------------------------------------------
# TC-08  GET /products/{sku} → 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc08_get_product_not_found(client: AsyncClient):
    resp = await client.get("/api/v1/products/sku_that_does_not_exist")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# TC-09  PATCH /products/{sku} → 409 stale __v
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc09_stale_version_returns_409(client: AsyncClient):
    # Fetch current version
    get_resp = await client.get(f"/api/v1/products/{VALID_PRODUCT['sku']}")
    current_v = get_resp.json()["__v"]

    # First patch — should succeed
    patch_resp = await client.patch(
        f"/api/v1/products/{VALID_PRODUCT['sku']}",
        json={"name": "Updated Name", "version": current_v},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    # Second patch with stale version — should fail with 409
    stale_resp = await client.patch(
        f"/api/v1/products/{VALID_PRODUCT['sku']}",
        json={"name": "Another Update", "version": current_v},  # still old __v
    )
    assert stale_resp.status_code == 409, stale_resp.text
    assert "stale" in stale_resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# TC-10  Full run lifecycle: start → get active → end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tc10_full_run_lifecycle(client: AsyncClient):
    # Start a run for the product we created in TC-01
    start_resp = await client.post("/api/v1/runs", json={"sku": VALID_PRODUCT["sku"]})
    assert start_resp.status_code == 201, start_resp.text
    run = start_resp.json()
    run_id = run["run_id"]
    assert run["status"] == "active"
    assert run["sku"] == VALID_PRODUCT["sku"]

    # Duplicate start → 409
    dup_resp = await client.post("/api/v1/runs", json={"sku": VALID_PRODUCT["sku"]})
    assert dup_resp.status_code == 409, dup_resp.text

    # Get active run (any SKU)
    active_resp = await client.get("/api/v1/runs/active")
    assert active_resp.status_code == 200, active_resp.text
    active = active_resp.json()
    assert active is not None
    assert active["run_id"] == run_id

    # Get active run for specific SKU
    sku_active_resp = await client.get(f"/api/v1/runs/active/{VALID_PRODUCT['sku']}")
    assert sku_active_resp.status_code == 200, sku_active_resp.text
    assert sku_active_resp.json()["run_id"] == run_id

    # End the run
    end_resp = await client.patch(f"/api/v1/runs/{run_id}/end", json={"status": "completed"})
    assert end_resp.status_code == 200, end_resp.text
    ended = end_resp.json()
    assert ended["status"] == "completed"
    assert ended["ended_at"] is not None

    # Active run should now be None
    after_resp = await client.get(f"/api/v1/runs/active/{VALID_PRODUCT['sku']}")
    assert after_resp.status_code == 200
    assert after_resp.json() is None
