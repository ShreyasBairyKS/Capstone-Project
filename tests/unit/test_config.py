"""tests/unit/test_config.py — Phase 0 acceptance test for core/config.py"""

import os
from pathlib import Path

import pytest

from core.config import EdgeConfig, settings


class TestEdgeConfigDefaults:
    """Verify all required fields have sensible defaults."""

    def test_tier_default(self):
        cfg = EdgeConfig()
        assert cfg.TIER == "edge"

    def test_device_id_default(self):
        cfg = EdgeConfig()
        assert cfg.DEVICE_ID == "edge_node_01"

    def test_model_paths_are_path_objects(self):
        cfg = EdgeConfig()
        assert isinstance(cfg.YOLOV11_ONNX_PATH, Path)
        assert isinstance(cfg.EFFICIENTVIT_ONNX_PATH, Path)

    def test_confidence_threshold_range(self):
        cfg = EdgeConfig()
        assert 0.0 < cfg.YOLOV11_CONF_THRESHOLD < 1.0
        assert 0.0 < cfg.YOLOV11_IOU_THRESHOLD < 1.0

    def test_verdict_thresholds_ordered(self):
        """AUTO_PASS > ESCALATE > HUMAN_REVIEW prevents logical gaps."""
        cfg = EdgeConfig()
        assert cfg.AUTO_PASS_THRESHOLD > cfg.ESCALATE_THRESHOLD
        assert cfg.ESCALATE_THRESHOLD > cfg.HUMAN_REVIEW_THRESHOLD

    def test_uq_passes_positive(self):
        cfg = EdgeConfig()
        assert cfg.UQ_N_PASSES > 0

    def test_remedy_enabled_by_default(self):
        cfg = EdgeConfig()
        assert cfg.REMEDY_ENABLED is True

    def test_database_url_default_is_sqlite(self):
        cfg = EdgeConfig()
        assert "sqlite" in cfg.DATABASE_URL

    def test_module_singleton_is_edge_config(self):
        assert isinstance(settings, EdgeConfig)


class TestEdgeConfigEnvOverride:
    """Verify that environment variables override defaults correctly."""

    def test_tier_override(self, monkeypatch):
        monkeypatch.setenv("TIER", "fog")
        cfg = EdgeConfig()
        assert cfg.TIER == "fog"

    def test_device_id_override(self, monkeypatch):
        monkeypatch.setenv("DEVICE_ID", "test_node_99")
        cfg = EdgeConfig()
        assert cfg.DEVICE_ID == "test_node_99"

    def test_conf_threshold_override(self, monkeypatch):
        monkeypatch.setenv("YOLOV11_CONF_THRESHOLD", "0.55")
        cfg = EdgeConfig()
        assert abs(cfg.YOLOV11_CONF_THRESHOLD - 0.55) < 1e-9

    def test_remedy_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("REMEDY_ENABLED", "false")
        cfg = EdgeConfig()
        assert cfg.REMEDY_ENABLED is False

    def test_api_key_override(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-secret-key-abc123")
        cfg = EdgeConfig()
        assert cfg.API_KEY == "test-secret-key-abc123"


class TestEdgeConfigValidation:
    """Verify Pydantic rejects invalid tier values."""

    def test_invalid_tier_raises(self, monkeypatch):
        monkeypatch.setenv("TIER", "invalid_tier")
        with pytest.raises(Exception):
            EdgeConfig()
