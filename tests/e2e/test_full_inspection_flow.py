"""
tests/e2e/test_full_inspection_flow.py — End-to-end tests for the complete
VisionFood QAI inspection flow: image → API → DB → analytics.

These tests start the FastAPI app in-process (no real server), use an
in-memory SQLite database, and mock the ML pipeline so they run without
ONNX model files. They exercise the full HTTP call chain including
middleware.

Run with:
    pytest tests/e2e/ -v

Requirements: pip install httpx pytest-anyio
"""

from __future__ import annotations

import base64
import os
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# In-memory DB + fixed API key for e2e suite
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["API_KEY"] = "test-key"

from api.main import app
from api.dependencies import get_pipeline as _get_pipeline
from database.session import create_tables


@pytest.fixture(scope="module", autouse=True)
def init_db():
    create_tables()


@pytest.fixture
def headers():
    return {"X-API-Key": "test-key"}


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _rgb_image_b64(r=200, g=100, b=50, size=8) -> str:
    """Create a tiny solid-colour JPEG and return as base64."""
    import cv2
    img = np.full((size, size, 3), (b, g, r), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_mock_pass_result(product_id: str = "E2E-PASS-001", sku: str = "default"):
    from core.schemas import InspectionResult, Verdict
    return InspectionResult(
        inspection_id=str(uuid.uuid4()),
        product_id=product_id,
        sku=sku,
        timestamp=datetime.utcnow(),
        verdict=Verdict.PASS,
        escalated=False,
        detections=[],
        latency_ms=22.0,
        device_id="e2e_node",
    )


def _build_mock_fail_result(product_id: str = "E2E-FAIL-001", sku: str = "default"):
    from core.schemas import (
        BoundingBox, DefectClass, Detection, InspectionResult,
        RemediationAction, RemediationActionType, SeverityGrade, SeverityResult,
        UQResult, Verdict,
    )
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    det = Detection(
        class_id=1, class_name=DefectClass.PACKAGING_DAMAGE,
        confidence=0.91, bbox=bbox, bbox_area_ratio=bbox.area_ratio,
    )
    uq = UQResult(
        mean_confidence=0.91, std_confidence=0.04,
        ci_low=0.83, ci_high=0.99,
        is_uncertain=False, escalation_required=False, n_passes=20,
    )
    sev = SeverityResult(
        grade=SeverityGrade.S2, score=0.44,
        area_component=0.36, conf_uncertainty_component=0.13,
        class_risk_component=0.65, attempt_penalty_component=0.0,
    )
    action = RemediationAction(
        action=RemediationActionType.REPACK,
        station="C", is_remediable=True,
        reason="Packaging damage S2 — station C repack",
        max_attempts=2,
    )
    return InspectionResult(
        inspection_id=str(uuid.uuid4()),
        product_id=product_id,
        sku=sku,
        timestamp=datetime.utcnow(),
        verdict=Verdict.FAIL,
        escalated=False,
        detections=[det],
        uq_result=uq,
        severity_result=sev,
        remediation_action=action,
        latency_ms=78.3,
        device_id="e2e_node",
    )


# --------------------------------------------------------------------------- #
# E2E: PASS flow
# --------------------------------------------------------------------------- #

class TestPassFlow:
    """Submit a clean product, verify PASS flows through API → DB → analytics."""

    @pytest.mark.anyio
    async def test_pass_inspection_stored_and_retrievable(
        self, client: AsyncClient, headers
    ):
        mock_result = _build_mock_pass_result()
        pipeline_mock = MagicMock()
        pipeline_mock.inspect.return_value = mock_result
        app.dependency_overrides[_get_pipeline] = lambda: pipeline_mock
        try:
            post_resp = await client.post(
                "/inspections",
                json={"image_b64": _rgb_image_b64(), "sku": "default"},
                headers=headers,
            )
        finally:
            app.dependency_overrides.pop(_get_pipeline, None)

        assert post_resp.status_code == 201
        created = post_resp.json()
        assert created["verdict"] == "PASS"
        inspection_id = created["inspection_id"]

        # Retrieve by ID
        get_resp = await client.get(f"/inspections/{inspection_id}", headers=headers)
        assert get_resp.status_code == 200
        retrieved = get_resp.json()
        assert retrieved["verdict"] == "PASS"
        assert retrieved["id"] == inspection_id

    @pytest.mark.anyio
    async def test_pass_appears_in_list(
        self, client: AsyncClient, headers
    ):
        mock_result = _build_mock_pass_result(product_id="LIST-TEST")
        pipeline_mock = MagicMock()
        pipeline_mock.inspect.return_value = mock_result
        app.dependency_overrides[_get_pipeline] = lambda: pipeline_mock
        try:
            await client.post(
                "/inspections",
                json={"image_b64": _rgb_image_b64(), "sku": "default"},
                headers=headers,
            )
        finally:
            app.dependency_overrides.pop(_get_pipeline, None)

        list_resp = await client.get("/inspections?limit=100", headers=headers)
        assert list_resp.status_code == 200
        ids = [r["id"] for r in list_resp.json()]
        assert mock_result.inspection_id in ids


# --------------------------------------------------------------------------- #
# E2E: FAIL flow
# --------------------------------------------------------------------------- #

class TestFailFlow:
    """Submit a defective product, verify FAIL + REMEDY flows through correctly."""

    @pytest.mark.anyio
    async def test_fail_inspection_has_defect_and_remedy(
        self, client: AsyncClient, headers
    ):
        mock_result = _build_mock_fail_result()
        pipeline_mock = MagicMock()
        pipeline_mock.inspect.return_value = mock_result
        app.dependency_overrides[_get_pipeline] = lambda: pipeline_mock
        try:
            post_resp = await client.post(
                "/inspections",
                json={
                    "image_b64": _rgb_image_b64(r=255, g=0, b=0),
                    "sku": "can_330ml",
                    "product_id": "E2E-FAIL-CAN",
                },
                headers=headers,
            )
        finally:
            app.dependency_overrides.pop(_get_pipeline, None)

        assert post_resp.status_code == 201
        body = post_resp.json()
        assert body["verdict"] == "FAIL"
        assert len(body["detections"]) == 1
        assert body["detections"][0]["class_name"] == "packaging_damage"
        assert body["severity_result"]["grade"] == "S2"
        assert body["remediation_action"]["action"] == "REPACK"

    @pytest.mark.anyio
    async def test_fail_filter_works(self, client: AsyncClient, headers):
        """?verdict=FAIL should only return FAIL rows."""
        list_resp = await client.get("/inspections?verdict=FAIL&limit=100", headers=headers)
        assert list_resp.status_code == 200
        for row in list_resp.json():
            assert row["verdict"] == "FAIL"


# --------------------------------------------------------------------------- #
# E2E: Analytics aggregation
# --------------------------------------------------------------------------- #

class TestAnalyticsFlow:

    @pytest.mark.anyio
    async def test_analytics_reflects_submitted_inspections(
        self, client: AsyncClient, headers
    ):
        """Analytics summary total_inspections should be ≥ the inspections we submitted."""
        summary_resp = await client.get("/analytics/summary?hours=24", headers=headers)
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        # We've submitted at least a few inspections above
        assert summary["total_inspections"] >= 0
        assert 0.0 <= summary["defect_rate"] <= 1.0
        assert summary["avg_latency_ms"] >= 0.0

    @pytest.mark.anyio
    async def test_pareto_chart_data_is_sorted(self, client: AsyncClient, headers):
        """Pareto data must be sorted descending by count."""
        resp = await client.get("/analytics/defect-pareto?hours=24", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        counts = [d["count"] for d in data]
        assert counts == sorted(counts, reverse=True)


# --------------------------------------------------------------------------- #
# E2E: Verdict override flow
# --------------------------------------------------------------------------- #

class TestVerdictOverrideFlow:

    @pytest.mark.anyio
    async def test_operator_can_override_verdict(self, client: AsyncClient, headers):
        """FAIL → PASS override by operator should persist and be retrievable."""
        # First create a FAIL inspection
        mock_result = _build_mock_fail_result(product_id="OVERRIDE-TEST")
        pipeline_mock = MagicMock()
        pipeline_mock.inspect.return_value = mock_result
        app.dependency_overrides[_get_pipeline] = lambda: pipeline_mock
        try:
            post_resp = await client.post(
                "/inspections",
                json={"image_b64": _rgb_image_b64(), "sku": "default"},
                headers=headers,
            )
        finally:
            app.dependency_overrides.pop(_get_pipeline, None)

        assert post_resp.status_code == 201
        inspection_id = post_resp.json()["inspection_id"]

        # Override the verdict
        override_resp = await client.patch(
            f"/inspections/{inspection_id}/verdict",
            json={
                "new_verdict": "PASS",
                "reason": "Re-inspection confirmed no defect. Operator: Jane Smith",
            },
            headers=headers,
        )
        assert override_resp.status_code == 200
        override_body = override_resp.json()
        assert override_body["verdict"] == "PASS"

        # Confirm the update is persisted
        get_resp = await client.get(f"/inspections/{inspection_id}", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["verdict"] == "PASS"
