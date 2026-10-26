"""
Inspection rules and threshold configuration for Fill Level Quality Gate.
"""
from pydantic import BaseModel, Field

from pipeline.fill_level.config import fill_threshold_from_env


class FillRulesConfig(BaseModel):
    """
    Industrial fill level quality control policy configuration.
    Defines nominal volume limits, underfill/overfill rejection boundaries,
    and gray-zone / review parameters.
    """
    underfill_threshold: float = Field(
        default_factory=lambda: fill_threshold_from_env("VISIONQAI_FILL_UNDERFILL_THRESHOLD", 0.40),
        ge=0.10,
        le=0.60,
        description="Minimum fill ratio required to avoid underfill rejection (e.g. 0.40 = 40% full)"
    )

    overfill_threshold: float = Field(
        default_factory=lambda: fill_threshold_from_env("VISIONQAI_FILL_OVERFILL_THRESHOLD", 0.85),
        ge=0.60,
        le=0.98,
        description="Maximum fill ratio allowed before overfill rejection (e.g. 0.85 = 85% full)"
    )

    borderline_band: float = Field(
        default=0.02,
        ge=0.00,
        le=0.05,
        description="Width of gray-zone band around rejection thresholds (e.g. 0.02 = 2%)"
    )

    borderline_conf_floor: float = Field(
        default=0.50,
        ge=0.20,
        le=0.80,
        description="Meniscus detection confidence below which borderline cases route to UNCERTAIN"
    )

    def is_underfill(self, fill_ratio: float) -> bool:
        return fill_ratio < self.underfill_threshold

    def is_overfill(self, fill_ratio: float) -> bool:
        return fill_ratio > self.overfill_threshold

    def is_nominal(self, fill_ratio: float) -> bool:
        return self.underfill_threshold <= fill_ratio <= self.overfill_threshold

    def is_borderline(self, fill_ratio: float) -> bool:
        """Checks if fill ratio falls into the gray-zone margin near thresholds."""
        near_under = abs(fill_ratio - self.underfill_threshold) <= self.borderline_band
        near_over = abs(fill_ratio - self.overfill_threshold) <= self.borderline_band
        return near_under or near_over
