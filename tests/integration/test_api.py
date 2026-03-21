"""
tests/integration/test_api.py — Integration tests for the FastAPI backend.

These tests run against the real FastAPI app using httpx.AsyncClient
with an in-memory SQLite database (no Redis required — WebSocket tests
are excluded here and covered in e2e/).

Run with:
    pytest tests/integration/ -v

Requirements: pip install httpx pytest-anyio
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import uuid

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Override database to in-memory SQLite before importing the app
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("API_KEY", "test-key")

from api.main import app
from api.dependencies import get_pipeline
from core.schemas import InspectionResult, Verdict
from database.session import create_tables


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create tables in the in-memory test database once per session."""
    create_tables()


@pytest.fixture
def api_headers():
    return {"X-API-Key": "test-key"}


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _tiny_jpeg_b64() -> str:
    """Return a base64-encoded 4×4 solid-colour JPEG for tests."""
    import cv2
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:] = (120, 60, 30)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #

class TestHealthCheck:

    @pytest.mark.anyio
    async def test_health_returns_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

class TestAuthentication:

    @pytest.mark.anyio
    async def test_missing_api_key_returns_403(self, client: AsyncClient):
        resp = await client.get("/inspections")
        assert resp.status_code in (401, 403)

    @pytest.mark.anyio
    async def test_wrong_api_key_returns_403(self, client: AsyncClient):
        resp = await client.get("/inspections", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code in (401, 403)

    @pytest.mark.anyio
    async def test_valid_api_key_passes(self, client: AsyncClient, api_headers):
        resp = await client.get("/inspections", headers=api_headers)
        # Should return 200 (possibly empty list), not 401/403
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Inspection submission
# --------------------------------------------------------------------------- #

class TestInspectionEndpoint:

    @pytest.mark.anyio
    async def test_submit_inspection_pass(self, client: AsyncClient, api_headers):
        """POST /inspections with a valid image returns 201 and an InspectionResult."""
        mock_result = InspectionResult(
            inspection_id=str(uuid.uuid4()),  # unique per [asyncio]/[trio] run
            product_id="P999",
            sku="bottle_250ml",
            timestamp=datetime.utcnow(),
            verdict=Verdict.PASS,
            escalated=False,
            detections=[],
            uq_result=None,
            severity_result=None,
            remediation_action=None,
            latency_ms=45.2,
            device_id="edge_node_test",
        )

        pipeline_mock = MagicMock()
        pipeline_mock.inspect.return_value = mock_result
        app.dependency_overrides[get_pipeline] = lambda: pipeline_mock

        try:
            resp = await client.post(
                "/inspections",
                json={
                    "image_b64": _tiny_jpeg_b64(),
                    "product_id": "P999",
                    "sku": "bottle_250ml",
                    "attempt_count": 0,
                },
                headers=api_headers,
            )
        finally:
            app.dependency_overrides.pop(get_pipeline, None)

        assert resp.status_code == 201
        data = resp.json()
        assert data["verdict"] == "PASS"
        assert "inspection_id" in data

    @pytest.mark.anyio
    async def test_invalid_base64_returns_422(self, client: AsyncClient, api_headers):
        dummy = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: dummy
        try:
            resp = await client.post(
                "/inspections",
                json={"image_b64": "this-is-not-base64!!!"},
                headers=api_headers,
            )
        finally:
            app.dependency_overrides.pop(get_pipeline, None)
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_invalid_image_bytes_returns_422(self, client: AsyncClient, api_headers):
        """Valid base64 but not a valid image."""
        garbage = base64.b64encode(b"not an image").decode()
        dummy = MagicMock()
        app.dependency_overrides[get_pipeline] = lambda: dummy
        try:
            resp = await client.post(
                "/inspections",
                json={"image_b64": garbage},
                headers=api_headers,
            )
        finally:
            app.dependency_overrides.pop(get_pipeline, None)
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# List inspections
# --------------------------------------------------------------------------- #

class TestListInspections:

    @pytest.mark.anyio
    async def test_list_returns_list(self, client: AsyncClient, api_headers):
        resp = await client.get("/inspections", headers=api_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.anyio
    async def test_limit_parameter_respected(self, client: AsyncClient, api_headers):
        resp = await client.get("/inspections?limit=5", headers=api_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    @pytest.mark.anyio
    async def test_invalid_verdict_filter_returns_422(self, client: AsyncClient, api_headers):
        resp = await client.get("/inspections?verdict=UNKNOWN", headers=api_headers)
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Analytics endpoints
# --------------------------------------------------------------------------- #

class TestAnalyticsEndpoints:

    @pytest.mark.anyio
    async def test_summary_returns_dict(self, client: AsyncClient, api_headers):
        resp = await client.get("/analytics/summary", headers=api_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_inspections" in data
        assert "defect_rate" in data

    @pytest.mark.anyio
    async def test_pareto_returns_list(self, client: AsyncClient, api_headers):
        resp = await client.get("/analytics/defect-pareto", headers=api_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.anyio
    async def test_severity_distribution_returns_list(self, client: AsyncClient, api_headers):
        resp = await client.get("/analytics/severity-distribution", headers=api_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.anyio
    async def test_summary_invalid_hours_returns_422(self, client: AsyncClient, api_headers):
        resp = await client.get("/analytics/summary?hours=0", headers=api_headers)
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Verdict override
# --------------------------------------------------------------------------- #

class TestVerdictOverride:

    @pytest.mark.anyio
    async def test_override_nonexistent_inspection_returns_404(
        self, client: AsyncClient, api_headers
    ):
        resp = await client.patch(
            "/inspections/nonexistent-id/verdict",
            json={"new_verdict": "PASS", "reason": "Manual override by operator"},
            headers=api_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_override_invalid_verdict_returns_422(
        self, client: AsyncClient, api_headers
    ):
        resp = await client.patch(
            "/inspections/some-id/verdict",
            json={"new_verdict": "INVALID", "reason": "test"},
            headers=api_headers,
        )
        assert resp.status_code == 422
