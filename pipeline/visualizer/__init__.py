"""
Part 4: Visual Overlay & Annotation Engine
Provides semi-transparent mask blending, bounding boxes, defect confidence pills,
top status HUD banners, and dynamic on-demand replay rendering from Part 5 bundles.
"""
from pipeline.visualizer.schemas import VisualizerConfig, DEFAULT_BGR_PALETTE
from pipeline.visualizer.renderer import VisualOverlayRenderer

__all__ = [
    "VisualizerConfig",
    "DEFAULT_BGR_PALETTE",
    "VisualOverlayRenderer"
]
