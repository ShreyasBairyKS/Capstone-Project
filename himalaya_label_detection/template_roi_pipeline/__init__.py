"""Template-matching ROI pipeline for Himalaya NSC label scans (see README.md)."""

from .inspector import (
    InspectionResult,
    ONNXROIClassifier,
    ROIVerdict,
    TemplateROIInspector,
)
from .roi_template_matcher import ROI_NAMES, ROIMatch, ROITemplateMatcher

__all__ = [
    "ROI_NAMES",
    "ROIMatch",
    "ROITemplateMatcher",
    "ONNXROIClassifier",
    "ROIVerdict",
    "InspectionResult",
    "TemplateROIInspector",
]
