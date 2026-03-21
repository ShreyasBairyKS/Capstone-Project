"""
core/config.py — Environment-driven configuration for VisionFood QAI.

All runtime parameters are loaded from environment variables or a .env file.
No secrets or hardcoded paths in code.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class EdgeConfig(BaseSettings):
    """
    Configuration for the Edge inference tier.
    All values can be overridden via environment variables or a .env file.
    """

    # ------------------------------------------------------------------ #
    # Deployment identity
    # ------------------------------------------------------------------ #
    TIER: Literal["edge", "fog", "cloud"] = "edge"
    DEVICE_ID: str = "edge_node_01"

    # ------------------------------------------------------------------ #
    # Model paths
    # ------------------------------------------------------------------ #
    YOLOV11_ONNX_PATH: Path = Path("models/yolov11n_best.onnx")
    EFFICIENTVIT_ONNX_PATH: Path = Path("models/efficientvit_m5_best.onnx")

    # ------------------------------------------------------------------ #
    # Inference thresholds
    # ------------------------------------------------------------------ #
    YOLOV11_CONF_THRESHOLD: float = 0.40
    YOLOV11_IOU_THRESHOLD: float = 0.45
    AUTO_PASS_THRESHOLD: float = 0.85      # ≥ this AND not uncertain → PASS
    ESCALATE_THRESHOLD: float = 0.60       # < this → fog escalation attempt
    HUMAN_REVIEW_THRESHOLD: float = 0.45   # < this → human review queue

    # ------------------------------------------------------------------ #
    # Camera / capture
    # ------------------------------------------------------------------ #
    CAMERA_INDEX: int = 0
    CAMERA_MODE: Literal["software", "hardware"] = "software"
    CAPTURE_FPS: float = 2.0               # Frames per second in software trigger mode

    # ------------------------------------------------------------------ #
    # UQ (MC Dropout)
    # ------------------------------------------------------------------ #
    UQ_N_PASSES: int = 20
    UQ_UNCERTAINTY_THRESHOLD: float = 0.15  # std dev above this → uncertain

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        default="sqlite:///./visionfood_dev.db",
        description="SQLAlchemy sync database URL. Set in .env for production.",
    )

    # ------------------------------------------------------------------ #
    # Redis
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_LIVE_STREAM: str = "inspections:live"
    REDIS_STREAM_MAX_LEN: int = 1000

    # ------------------------------------------------------------------ #
    # API security
    # ------------------------------------------------------------------ #
    API_KEY: str = Field(
        default="dev-insecure-key",
        description="Override in production — generate with: python -c \"import secrets; print(secrets.token_hex(32))\"",
    )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"
    AUDIT_LOG_PATH: str = "logs/audit.jsonl"

    # ------------------------------------------------------------------ #
    # REMEDY engine
    # ------------------------------------------------------------------ #
    REMEDY_ENABLED: bool = True

    # SKU profile directory
    SKU_PROFILES_DIR: Path = Path("configs/sku_profiles")

    # ------------------------------------------------------------------ #
    # Experiment tracking (optional — can be None in offline mode)
    # ------------------------------------------------------------------ #
    WANDB_API_KEY: str | None = Field(default=None, description="Set in .env")
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Module-level singleton — import this anywhere with:
#   from core.config import settings
settings = EdgeConfig()
