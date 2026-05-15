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
    YOLO_FILL_DETECTOR_WEIGHTS: Path = Path("runs/detect/bottle_cap_det_v2/weights/best.pt")
    YOLO_FILL_WATER_WEIGHTS: Path = Path("runs/detect/water_surface_v1/weights/best.pt")
    YOLO_FILL_CAP_CLASSIFIER_WEIGHTS: Path = Path("models/cap_classifier_best.pth")
    YOLO_FILL_DEVICE: str = "cpu"

    # ------------------------------------------------------------------ #
    # Inference thresholds
    # ------------------------------------------------------------------ #
    YOLOV11_CONF_THRESHOLD: float = 0.40
    YOLOV11_IOU_THRESHOLD: float = 0.45
    YOLO_FILL_CONF_THRESHOLD: float = 0.25
    YOLO_FILL_ZOOM_SCALE: float = 2.5
    CONFIRMED_DEFECT_THRESHOLD: float = 0.85  # ≥ this AND not uncertain → FAIL (confirmed defect)
    ESCALATE_THRESHOLD: float = 0.60            # < this → fog escalation attempt
    HUMAN_REVIEW_THRESHOLD: float = 0.45        # < this → human review queue

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
    UQ_ESCALATION_CONF_THRESHOLD: float = 0.60  # mean conf below this → escalation

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = Field(
        default="sqlite:///./visionfood_dev.db",
        description="SQLAlchemy sync database URL. Set in .env for production.",
    )

    # ------------------------------------------------------------------ #
    # MongoDB (Motor async)
    # ------------------------------------------------------------------ #
    MONGO_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URL for Motor async client.",
    )
    MONGO_DB_NAME: str = Field(
        default="visionfood",
        description="MongoDB database name used by Motor.",
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
    REMEDY_MAX_ATTEMPTS: int = 2

    # Severity scorer weights (must sum to 1.0)
    SEVERITY_W_AREA: float = 0.35
    SEVERITY_W_CONF_UQ: float = 0.15
    SEVERITY_W_CLASS_RISK: float = 0.40
    SEVERITY_W_ATTEMPT: float = 0.10

    # Severity grade thresholds
    SEVERITY_THRESHOLD_S1: float = 0.30
    SEVERITY_THRESHOLD_S2: float = 0.55
    SEVERITY_THRESHOLD_S3: float = 0.80

    # Normalisation divisors for severity components
    SEVERITY_AREA_CAP: float = 0.25
    SEVERITY_CONF_UQ_CAP: float = 0.30

    # SKU profile directory
    SKU_PROFILES_DIR: Path = Path("configs/sku_profiles")

    # ------------------------------------------------------------------ #
    # Experiment tracking (optional — can be None in offline mode)
    # ------------------------------------------------------------------ #
    WANDB_API_KEY: str | None = Field(default=None, description="Set in .env")
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # ------------------------------------------------------------------ #
    # Application metadata
    # ------------------------------------------------------------------ #
    APP_VERSION: str = "0.2.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Module-level singleton — import this anywhere with:
#   from core.config import settings
settings = EdgeConfig()
