"""
Data schemas for Part 5: Logging, Defect Audit & Telemetry.
Defines models for audit configuration, DB records, filters, and KPI rollups.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    """Configuration options for the audit logging and bundling subsystem."""
    base_dir: str = Field(default="audit", description="Base directory for audit logs and bundles")
    db_filename: str = Field(default="inspections.db", description="SQLite database filename")
    bottles_subfolder: str = Field(default="bottles", description="Subfolder name for isolated bottle bundles")
    save_pass_images: bool = Field(default=True, description="Whether to persist raw frames for compliant PASS bottles")
    save_defect_images: bool = Field(default=True, description="Whether to persist raw frames for REJECT/UNCERTAIN bottles")
    save_yolo_labels: bool = Field(default=True, description="Whether to generate YOLO format polygon label files")
    jpeg_quality: int = Field(default=95, ge=50, le=100, description="Compression quality for archived raw frames")
    async_saving: bool = Field(default=True, description="Whether to offload disk I/O to a background thread pool")
    thread_pool_workers: int = Field(default=2, description="Number of worker threads for background file I/O")


class BottleAuditRecord(BaseModel):
    """Structured inspection event record stored in the database."""
    id: Optional[int] = Field(None, description="Auto-incremented primary key in SQLite")
    bottle_id: str = Field(..., description="Unique inspection session ID for the bottle")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    epoch_time: float = Field(..., description="UNIX epoch timestamp for fast indexed range filtering")
    global_verdict: str = Field(..., description="Final decision verdict (PASS, REJECT, UNCERTAIN, INSPECTION_FAILED)")
    is_compliant: bool = Field(..., description="True if PASS, False otherwise")
    trigger_actuator: bool = Field(..., description="Whether physical rejection actuator was fired")
    primary_defect: Optional[str] = Field(None, description="Primary defect identified (e.g. damaged_cap)")
    primary_defect_confidence: Optional[float] = Field(None, description="Confidence of primary defect")
    flagged_cameras: List[str] = Field(default_factory=list, description="List of camera IDs that observed defect/issue")
    decision_reason: str = Field(..., description="Human-readable decision explanation")
    total_latency_ms: float = Field(..., description="Total pipeline latency (ms)")
    camera_count: int = Field(..., description="Total camera views processed for this bottle")
    bundle_dir: Optional[str] = Field(None, description="Relative path to isolated bottle bundle directory")


class KPISummary(BaseModel):
    """Aggregated manufacturing telemetry and quality metrics."""
    total_inspected: int = Field(..., description="Total bottles processed through inspection")
    passed: int = Field(..., description="Total compliant bottles passed")
    rejected: int = Field(..., description="Total defective bottles rejected")
    uncertain: int = Field(..., description="Total bottles routed for review")
    failed: int = Field(..., description="Total missing/failed inspection cycles")
    pass_rate_pct: float = Field(..., description="Overall manufacturing compliance yield percentage (0-100%)")
    reject_rate_pct: float = Field(..., description="Defect rejection percentage (0-100%)")
    defect_breakdown: Dict[str, int] = Field(default_factory=dict, description="Tally of defects by category")
    average_latency_ms: float = Field(..., description="Mean pipeline cycle latency across recorded items")


class InspectionFilter(BaseModel):
    """Query parameters for filtering historical inspection records."""
    verdict: Optional[str] = Field(None, description="Filter by verdict (PASS, REJECT, UNCERTAIN, INSPECTION_FAILED)")
    defect_type: Optional[str] = Field(None, description="Filter by specific defect class name")
    start_epoch: Optional[float] = Field(None, description="Start UNIX epoch time")
    end_epoch: Optional[float] = Field(None, description="End UNIX epoch time")
    limit: int = Field(default=50, ge=1, le=1000, description="Max records to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
