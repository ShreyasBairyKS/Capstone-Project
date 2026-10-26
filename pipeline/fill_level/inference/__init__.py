"""
Inference module for Fill Level Inspection Station.
"""
from .schemas import BottleBox, MeniscusBox, SingleImageFillResult
from .engine import FillLevelInferenceEngine

__all__ = [
    "BottleBox",
    "MeniscusBox",
    "SingleImageFillResult",
    "FillLevelInferenceEngine",
]
