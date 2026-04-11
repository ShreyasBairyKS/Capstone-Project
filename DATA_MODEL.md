# VisionFood QAI — Data Model

---

## Overview

The system uses **SQLite** in development and **PostgreSQL** in production. The schema is managed with Alembic migrations. All ORM models live in `database/models.py`. All API-layer data structures are Pydantic schemas in `core/schemas.py`.

---

## Database Tables

### Entity Relationship Diagram

```
┌──────────────────┐        ┌───────────────────────┐
│   inspections    │  1:N   │        defects        │
│──────────────────│───────►│───────────────────────│
│ id (PK)          │        │ id (PK)               │
│ product_id       │        │ inspection_id (FK)    │
│ timestamp        │        │ class_name            │
│ verdict          │        │ confidence            │
│ overall_severity │        │ bbox_x1, y1, x2, y2   │
│ model_version_id │        │ area_fraction         │
│ inference_ms     │        │ severity_grade        │
│ escalated        │        │ remedy_action         │
│ operator_note    │        │ attempt_count         │
└──────────────────┘        └───────────────────────┘
         │
         │ 1:1
         ▼
┌──────────────────────┐
│  remediation_actions  │
│──────────────────────│
│ id (PK)              │
│ inspection_id (FK)   │
│ action_type          │
│ station              │
│ outcome              │
│ re_inspection_id(FK) │
│ created_at           │
└──────────────────────┘

┌──────────────────┐        ┌───────────────────────┐
│  model_versions  │        │    quality_reports    │
│──────────────────│        │───────────────────────│
│ id (PK)          │        │ id (PK)               │
│ name             │        │ report_type           │
│ architecture     │        │ period_start          │
│ stage            │        │ period_end            │
│ map50            │        │ total_inspected       │
│ map50_95         │        │ pass_count            │
│ f1_score         │        │ fail_count            │
│ latency_cpu_ms   │        │ defect_rate_pct       │
│ latency_gpu_ms   │        │ remedy_save_rate_pct  │
│ onnx_path        │        │ pdf_path              │
│ is_active        │        │ generated_at          │
│ trained_at       │        │ generated_by          │
│ wandb_run_id     │        └───────────────────────┘
└──────────────────┘
```

---

## SQLAlchemy ORM Models (`database/models.py`)

```python
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, ForeignKey, Enum, Text
)
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime
import uuid
import enum


class Base(DeclarativeBase):
    pass


class VerdictEnum(str, enum.Enum):
    PASS    = "PASS"
    FAIL    = "FAIL"
    ESCALATE = "ESCALATE"
    REVIEW  = "REVIEW"


class SeverityGradeEnum(str, enum.Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


class ModelStageEnum(str, enum.Enum):
    CANDIDATE  = "candidate"
    SHADOW     = "shadow"
    STAGING    = "staging"
    PRODUCTION = "production"
    STANDBY    = "standby"
    ARCHIVED   = "archived"


class Inspection(Base):
    __tablename__ = "inspections"

    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id       = Column(String(64), nullable=False, index=True)
    timestamp        = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    verdict          = Column(Enum(VerdictEnum), nullable=False, index=True)
    overall_severity = Column(Enum(SeverityGradeEnum), nullable=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    inference_ms     = Column(Float, nullable=False)
    escalated        = Column(Boolean, default=False)
    operator_note    = Column(Text, nullable=True)   # Manual override reason
    image_path       = Column(String(512), nullable=True)

    defects             = relationship("Defect", back_populates="inspection", cascade="all, delete-orphan")
    remediation_action  = relationship("RemediationAction", back_populates="inspection", uselist=False)
    model_version       = relationship("ModelVersion", back_populates="inspections")


class Defect(Base):
    __tablename__ = "defects"

    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=False, index=True)
    class_name    = Column(String(64), nullable=False, index=True)
    confidence    = Column(Float, nullable=False)
    bbox_x1       = Column(Float, nullable=False)   # Normalised 0–1
    bbox_y1       = Column(Float, nullable=False)
    bbox_x2       = Column(Float, nullable=False)
    bbox_y2       = Column(Float, nullable=False)
    area_fraction = Column(Float, nullable=False)
    severity_grade = Column(Enum(SeverityGradeEnum), nullable=True)
    remedy_action  = Column(String(32), nullable=True)  # RELABEL/REFILL/REPACK/CLEAN/REJECT
    attempt_count  = Column(Integer, default=0)

    # UQ fields
    uq_mean       = Column(Float, nullable=True)
    uq_std        = Column(Float, nullable=True)
    uq_ci_low     = Column(Float, nullable=True)
    uq_ci_high    = Column(Float, nullable=True)

    inspection    = relationship("Inspection", back_populates="defects")


class RemediationAction(Base):
    __tablename__ = "remediation_actions"

    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    inspection_id    = Column(String(36), ForeignKey("inspections.id"), nullable=False, index=True)
    action_type      = Column(String(32), nullable=False)  # RELABEL/REFILL/REPACK/CLEAN
    station          = Column(String(8), nullable=True)    # A/B/C
    outcome          = Column(String(32), nullable=False)  # PASS/FAIL/PENDING
    re_inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    inspection       = relationship("Inspection", back_populates="remediation_action",
                                    foreign_keys=[inspection_id])


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name           = Column(String(64), nullable=False, unique=True)   # e.g. yolov11n_v1.2.0
    architecture   = Column(String(64), nullable=False)                # yolov11n / efficientvit_m5
    stage          = Column(Enum(ModelStageEnum), nullable=False, default=ModelStageEnum.CANDIDATE)
    map50          = Column(Float, nullable=True)
    map50_95       = Column(Float, nullable=True)
    f1_score       = Column(Float, nullable=True)
    latency_cpu_ms = Column(Float, nullable=True)
    latency_gpu_ms = Column(Float, nullable=True)
    onnx_path      = Column(String(512), nullable=True)
    dataset_hash   = Column(String(64), nullable=True)   # SHA-256 of training dataset
    is_active      = Column(Boolean, default=False)
    trained_at     = Column(DateTime, nullable=True)
    wandb_run_id   = Column(String(128), nullable=True)

    inspections    = relationship("Inspection", back_populates="model_version")


class QualityReport(Base):
    __tablename__ = "quality_reports"

    id                  = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_type         = Column(String(16), nullable=False)  # shift / daily / weekly
    period_start        = Column(DateTime, nullable=False)
    period_end          = Column(DateTime, nullable=False)
    total_inspected     = Column(Integer, nullable=False)
    pass_count          = Column(Integer, nullable=False)
    fail_count          = Column(Integer, nullable=False)
    remediated_count    = Column(Integer, nullable=False, default=0)
    defect_rate_pct     = Column(Float, nullable=False)
    remedy_save_rate_pct = Column(Float, nullable=True)
    pdf_path            = Column(String(512), nullable=True)
    status              = Column(String(16), default="pending")  # pending/complete/failed
    generated_at        = Column(DateTime, default=datetime.utcnow)
    generated_by        = Column(String(64), nullable=True)
```

---

## Pydantic Schemas (`core/schemas.py`)

These are used for API request/response validation and serialisation.

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class VerdictEnum(str, Enum):
    PASS     = "PASS"
    FAIL     = "FAIL"
    ESCALATE = "ESCALATE"
    REVIEW   = "REVIEW"


class SeverityGradeEnum(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


# ── Detection ─────────────────────────────────────────────────────

class DetectionSchema(BaseModel):
    class_name:    str
    confidence:    float = Field(ge=0.0, le=1.0)
    bbox:          tuple[float, float, float, float]  # x1, y1, x2, y2 normalised
    area_fraction: float = Field(ge=0.0, le=1.0)

    severity_grade: Optional[SeverityGradeEnum] = None
    remedy_action:  Optional[str] = None

    uq_mean:    Optional[float] = None
    uq_std:     Optional[float] = None
    uq_ci_low:  Optional[float] = None
    uq_ci_high: Optional[float] = None


# ── Inspection ─────────────────────────────────────────────────────

class InspectionResultSchema(BaseModel):
    product_id:       str
    timestamp:        datetime
    verdict:          VerdictEnum
    detections:       List[DetectionSchema]
    overall_severity: Optional[SeverityGradeEnum] = None
    inference_ms:     float
    model_version:    str
    escalated:        bool = False
    remedy_action:    Optional[str] = None

    model_config = {"from_attributes": True}


class InspectionListItemSchema(BaseModel):
    id:               str
    product_id:       str
    timestamp:        datetime
    verdict:          VerdictEnum
    overall_severity: Optional[SeverityGradeEnum]
    inference_ms:     float
    defect_classes:   List[str]

    model_config = {"from_attributes": True}


# ── Analytics ─────────────────────────────────────────────────────

class AnalyticsSummarySchema(BaseModel):
    total_inspected:      int
    pass_count:           int
    fail_count:           int
    pass_rate_pct:        float
    defect_rate_pct:      float
    remedy_save_rate_pct: float
    avg_inference_ms:     float
    period_start:         datetime
    period_end:           datetime


class DefectRatePointSchema(BaseModel):
    timestamp:  datetime
    defect_rate: float
    class_name: Optional[str] = None   # None = overall


class ParetoItemSchema(BaseModel):
    class_name:  str
    count:       int
    pct_of_total: float
    cumulative_pct: float


# ── Reports ───────────────────────────────────────────────────────

class ReportGenerateRequestSchema(BaseModel):
    report_type:  str = Field(pattern="^(shift|daily|weekly)$")
    period_start: datetime
    period_end:   datetime
    generated_by: Optional[str] = None


class ReportStatusSchema(BaseModel):
    id:           str
    status:       str
    pdf_url:      Optional[str] = None
    generated_at: Optional[datetime] = None


# ── Models ────────────────────────────────────────────────────────

class ModelVersionSchema(BaseModel):
    id:             str
    name:           str
    architecture:   str
    stage:          str
    map50:          Optional[float]
    map50_95:       Optional[float]
    f1_score:       Optional[float]
    latency_cpu_ms: Optional[float]
    latency_gpu_ms: Optional[float]
    is_active:      bool
    trained_at:     Optional[datetime]

    model_config = {"from_attributes": True}
```

---

## SQLite vs PostgreSQL Configuration

```python
# .env (development)
DATABASE_URL=sqlite:///./visionfood_dev.db

# .env.production
DATABASE_URL=postgresql://qai_user:${DB_PASSWORD}@db:5432/visionfood
```

Switch is handled entirely by `DATABASE_URL` in `core/config.py`. No code changes required.

---

## Key Indices

| Table | Column | Reason |
|-------|--------|--------|
| `inspections` | `timestamp` | Date-range analytics queries |
| `inspections` | `verdict` | Filter by pass/fail |
| `inspections` | `product_id` | Product traceability lookup |
| `defects` | `class_name` | Pareto aggregation |
| `defects` | `inspection_id` | Join to parent inspection |

---

## Alembic Migration Commands

```bash
# Initial setup (run once)
alembic init database/migrations

# Create migration after model change
alembic revision --autogenerate -m "add_uq_fields_to_defect"

# Apply migrations
alembic upgrade head

# Rollback one revision
alembic downgrade -1

# View migration history
alembic history --verbose
```

---

## Seed Data (Development)

`scripts/seed_db.py` inserts synthetic inspection records for dashboard development:
- 500 inspections across the last 7 days
- Realistic verdict distribution: 78% PASS, 15% FAIL, 5% ESCALATE, 2% REVIEW
- 4 defect classes with proportional frequency (label misalignment most common)
- Sample model version record with `is_active=True`

---

## Phase 7–9 Pydantic Schemas (Planned)

These schemas support the new production endpoints added in Phases 7–9. They will be defined in `core/schemas.py` alongside existing schemas.

### Batch Inspection (Phase 7)

```python
class BatchInspectionRequest(BaseModel):
    """POST /inspections/batch — run multiple images in one call."""
    product_id: str
    images: List[str]  # base64-encoded JPEG/PNG frames


class BatchInspectionResponse(BaseModel):
    """Response envelope for batch inspection."""
    batch_id: str
    results: List[InspectionResultSchema]
    total_ms: float
    avg_ms: float
```

### Health & Readiness Probes (Phase 7)

```python
class ReadinessResponse(BaseModel):
    """GET /readiness — Kubernetes readiness probe."""
    status: str          # "ready" | "not_ready"
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float
```

### Drift Detection (Phase 8)

```python
class DriftReport(BaseModel):
    """GET /analytics/drift — feature distribution drift."""
    window_start: datetime
    window_end: datetime
    kl_divergence: float
    drift_detected: bool
    threshold: float
    class_distribution: dict[str, float]       # current window
    baseline_distribution: dict[str, float]    # reference window
```

### Explainability — Grad-CAM++ (Phase 8)

```python
class ExplainabilityResult(BaseModel):
    """Returned when ?explain=true on POST /inspections."""
    heatmap_base64: str        # PNG heatmap overlay, base64-encoded
    top_regions: List[dict]    # [{"bbox": [x1,y1,x2,y2], "attribution": 0.87}, ...]
    method: str = "gradcam++"
```

> **Note:** The database schema (5 tables) does not change in Phases 7–9. All new endpoints operate on transient data (metrics counters, in-memory drift baselines, on-the-fly heatmaps) or reuse the existing `inspections` / `defects` tables.
