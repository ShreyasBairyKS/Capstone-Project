"""
Shared Station 2 configuration helpers.
Single source of truth for fill thresholds so the quality gate, inference engine,
visualizer ruler, and /api/fill/health all agree with VISIONQAI_FILL_* env settings.
"""
import os


def fill_threshold_from_env(name: str, default: float) -> float:
    """Reads a float threshold from the environment, falling back to `default` on missing/bad input."""
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default
