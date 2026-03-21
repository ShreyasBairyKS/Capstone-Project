"""VisionFood QAI — Core Module

Shared configuration, schemas, and logging utilities used across all tiers.
"""
from core.config import EdgeConfig
from core.logging import get_logger, setup_logging

__all__ = ["EdgeConfig", "get_logger", "setup_logging"]
