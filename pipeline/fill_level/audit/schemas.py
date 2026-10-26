"""
Data schemas for Part 4: Dedicated SQLite DB & Audit Storage (Station 2).
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from pipeline.env_loader import REPO_ROOT, env_path


class FillAuditConfig(BaseModel):
    """Configuration for Fill Level Station storage and SQLite database."""
    db_path: Path = Field(
        default_factory=lambda: env_path("VISIONQAI_AUDIT_DIR", REPO_ROOT / "audit") / "fill_level.db",
        description="Path to dedicated SQLite database file"
    )
    base_dir: Path = Field(
        default_factory=lambda: env_path("VISIONQAI_AUDIT_DIR", REPO_ROOT / "audit") / "fill_level",
        description="Root folder for Station 2 artifacts"
    )
    bottles_subfolder: str = Field(
        default="bottles",
        description="Subfolder where isolated bottle bundles are stored"
    )
    save_raw_image: bool = Field(
        default=True,
        description="Persist clean, unburned raw JPEG image for model training"
    )
    async_saving: bool = Field(
        default=False,
        description="Enable background saving queue (False for predictable testing)"
    )


class FillKPISummary(BaseModel):
    """Aggregated manufacturing KPI metrics for Fill Level Inspection Station."""
    total_inspected: int = Field(default=0, description="Total bottles processed")
    pass_count: int = Field(default=0, description="Compliant nominal bottles")
    underfill_count: int = Field(default=0, description="Underfill defects")
    overfill_count: int = Field(default=0, description="Overfill defects")
    dry_count: int = Field(default=0, description="Bottles without liquid detected")
    uncertain_count: int = Field(default=0, description="Borderline review cases")
    pass_rate: float = Field(default=0.0, description="Percentage of compliant bottles (0-100%)")
    mean_fill_ratio: float = Field(default=0.0, description="Average liquid fill ratio across nominal bottles")
    mean_latency_ms: float = Field(default=0.0, description="Average end-to-end cycle time in ms")


class FillInspectionFilter(BaseModel):
    """Query filters for paginated audit log retrieval."""
    limit: int = Field(default=50, ge=1, le=50000)
    offset: int = Field(default=0, ge=0)
    verdict: Optional[str] = Field(default=None, description="'PASS', 'REJECT', 'UNCERTAIN'")
    fill_status: Optional[str] = Field(default=None, description="'normal', 'underfill', 'overfill', 'no_water'")
    search: Optional[str] = Field(default=None, description="Substring search on bottle_id")
    start_epoch: Optional[float] = Field(default=None)
    end_epoch: Optional[float] = Field(default=None)


class FillInspectionRecord(BaseModel):
    """Persisted database record representing a single bottle inspection."""
    id: int
    bottle_id: str
    timestamp: str
    epoch_time: float
    global_verdict: str
    fill_status: str
    fill_ratio: Optional[float] = None
    water_y: Optional[int] = None
    bottle_h: Optional[int] = None
    water_conf: Optional[float] = None
    bottle_conf: Optional[float] = None
    is_compliant: bool
    trigger_actuator: bool
    decision_reason: str
    total_latency_ms: float
    bundle_dir: str
