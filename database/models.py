"""
database/models.py — SQLAlchemy ORM models for VisionFood QAI.

Tables:
  - inspections        : One row per product inspection
  - defects            : One row per detected defect (FK→inspections)
  - remediation_actions: One row per triage decision (FK→inspections)
  - model_versions     : Tracks trained model artefacts and active version
  - quality_reports    : Generated PDF report metadata

Implemented in Phase 4.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String(64), nullable=True, index=True)
    sku = Column(String(64), nullable=False, default="default")
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    verdict = Column(String(16), nullable=False, index=True)   # PASS/FAIL/ESCALATE/REVIEW
    escalated = Column(Boolean, default=False)
    latency_ms = Column(Float, nullable=True)
    device_id = Column(String(64), nullable=False)
    attempt_count = Column(Integer, default=0)

    defects = relationship("Defect", back_populates="inspection", cascade="all, delete-orphan")
    remediation_action = relationship("RemediationAction", back_populates="inspection", uselist=False, cascade="all, delete-orphan")


class Defect(Base):
    __tablename__ = "defects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String(36), ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True)
    class_name = Column(String(32), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    bbox_area_ratio = Column(Float, default=0.0)
    severity_grade = Column(String(4), nullable=True)
    severity_score = Column(Float, nullable=True)
    uq_mean = Column(Float, nullable=True)
    uq_std = Column(Float, nullable=True)

    inspection = relationship("Inspection", back_populates="defects")


class RemediationAction(Base):
    __tablename__ = "remediation_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String(36), ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, unique=True)
    action = Column(String(16), nullable=False)          # RELABEL / REFILL / REPACK / CLEAN / REJECT / PASS
    station = Column(String(4), nullable=True)           # A / B / C
    is_remediable = Column(Boolean, nullable=False)
    reason = Column(Text, nullable=False)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)

    inspection = relationship("Inspection", back_populates="remediation_action")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    version_tag = Column(String(32), nullable=False, unique=True)
    detector_path = Column(String(256), nullable=False)
    classifier_path = Column(String(256), nullable=False)
    map50 = Column(Float, nullable=True)
    top1_accuracy = Column(Float, nullable=True)
    trained_at = Column(DateTime, nullable=True)
    deployed_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False, index=True)
    rollback_to = Column(String(32), nullable=True)  # Previous version tag


class QualityReport(Base):
    __tablename__ = "quality_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False, default="VisionFood QAI Quality Report")
    report_type = Column(String(16), nullable=True)      # shift / daily / weekly
    from_dt = Column(DateTime, nullable=False)
    to_dt = Column(DateTime, nullable=False)
    total_inspections = Column(Integer, default=0)
    pass_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)
    defect_rate = Column(Float, default=0.0)
    remedy_save_rate_pct = Column(Float, default=0.0)
    status = Column(String(16), default="pending")       # pending / generating / complete / failed
    pdf_path = Column(String(256), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    generated_by = Column(String(128), nullable=False, default="system")
