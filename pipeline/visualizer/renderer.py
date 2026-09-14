"""
Visual Overlay Renderer for Part 4: Visual Overlay & Annotation Engine.
Provides:
1. Semi-transparent polygon mask rendering (alpha blend + antialiased borders).
2. Color-coded bounding boxes and defect confidence pills.
3. Top-edge high-visibility industrial status HUD banners.
4. Unified multi-camera side-by-side composite grid rendering.
5. Dynamic On-Demand Replay rendering directly from Part 5 raw storage bundles.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np

from pipeline.visualizer.schemas import VisualizerConfig
from pipeline.inference.schemas import Detection, CameraResult
from pipeline.decision.schemas import (
    GlobalInspectionResult,
    ViewVerdict,
    InspectionVerdict
)

logger = logging.getLogger("pipeline.visualizer.renderer")


class VisualOverlayRenderer:
    """
    Renders visual overlays, transparent segmentation masks, confidence pills,
    and industrial HUD telemetry banners for live streams and dynamic replay.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def draw_mask(
        self,
        image: np.ndarray,
        polygon: List[List[float]],
        color: Tuple[int, int, int],
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Draws a semi-transparent colored polygon mask over the target contour.
        Underlying bottle cap texture and defect ridges remain visible.
        """
        if not polygon or len(polygon) < 3:
            return image

        a = alpha if alpha is not None else self.config.mask_alpha
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

        # Create overlay layer for transparent blend
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, a, image, 1.0 - a, 0, image)

        # Draw sharp outer contour line for clear boundary definition
        if self.config.mask_border_thickness > 0:
            cv2.polylines(
                image,
                [pts],
                isClosed=True,
                color=color,
                thickness=self.config.mask_border_thickness,
                lineType=cv2.LINE_AA
            )
        return image

    def draw_box_and_pill(
        self,
        image: np.ndarray,
        box: List[float],
        class_name: str,
        confidence: Optional[float],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draws a crisp bounding box and high-contrast defect label pill.
        """
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw bounding box
        if self.config.show_boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=self.config.box_thickness,
                lineType=cv2.LINE_AA
            )

        # Format label text
        if self.config.show_confidence and confidence is not None:
            text = f"{class_name}: {confidence * 100:.1f}%"
        else:
            text = class_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = self.config.font_thickness

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pill_pad = 4

        # Position pill above box; if too close to frame top, place inside box
        pill_y1 = max(0, y1 - text_h - (pill_pad * 2))
        pill_y2 = pill_y1 + text_h + (pill_pad * 2)
        pill_x1 = max(0, x1)
        pill_x2 = min(image.shape[1], pill_x1 + text_w + (pill_pad * 2))

        # Draw filled pill background
        cv2.rectangle(
            image,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            color,
            thickness=-1
        )
        # Draw pill border
        cv2.rectangle(
            image,
            (pill_x1, pill_y1),
            (pill_x2, pill_y2),
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

        # Render white text inside pill
        text_origin = (pill_x1 + pill_pad, pill_y2 - pill_pad - 1)
        cv2.putText(
            image,
            text,
            text_origin,
            font,
            font_scale,
            (255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA
        )
        return image

    def draw_hud_banner(
        self,
        image: np.ndarray,
        verdict: str,
        defect_name: Optional[str] = None,
        latency_ms: Optional[float] = None,
        bottle_id: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders a top-edge industrial HUD banner with verdict status and telemetry.
        """
        h, w = image.shape[:2]
        banner_h = max(36, int(h * 0.05))  # Adaptive banner height
        banner_h = min(banner_h, 50)

        # Draw translucent dark background bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # Verdict badge color & text
        badge_color = self.config.get_color(verdict)
        if verdict == "REJECT" and defect_name:
            verdict_text = f"REJECT: {defect_name.upper()}"
        else:
            verdict_text = verdict.upper()

        badge_str = f" [{verdict_text}] "
        font = cv2.FONT_HERSHEY_SIMPLEX
        badge_scale = max(0.5, font_scale := banner_h / 65.0)

        # Draw left verdict badge
        (bw, bh), _ = cv2.getTextSize(badge_str, font, badge_scale, 2)
        badge_rect_w = bw + 16
        cv2.rectangle(image, (8, 4), (8 + badge_rect_w, banner_h - 4), badge_color, -1)
        cv2.putText(
            image,
            badge_str,
            (16, int(banner_h * 0.7)),
            font,
            badge_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw right telemetry metadata
        meta_items = []
        if bottle_id:
            meta_items.append(f"ID: {bottle_id}")
        if camera_id:
            meta_items.append(f"Cam: {camera_id}")
        if latency_ms is not None:
            meta_items.append(f"{latency_ms:.1f}ms")

        if meta_items:
            meta_str = " | ".join(meta_items)
            meta_scale = badge_scale * 0.85
            (mw, mh), _ = cv2.getTextSize(meta_str, font, meta_scale, 1)
            cv2.putText(
                image,
                meta_str,
                (max(8 + badge_rect_w + 10, w - mw - 16), int(banner_h * 0.68)),
                font,
                meta_scale,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        return image

    def render_view(
        self,
        frame: np.ndarray,
        view_data: Union[CameraResult, ViewVerdict, Dict[str, Any]],
        global_result: Optional[GlobalInspectionResult] = None,
        camera_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders a single camera view with transparent masks, bounding boxes,
        confidence pills, and HUD status banner.

        Args:
            frame: Raw BGR input frame (will not be modified in-place; a copy is returned).
            view_data: CameraResult, ViewVerdict, or raw dictionary of detections.
            global_result: Optional global inspection result for overall verdict context.
            camera_id: Optional camera identifier string.

        Returns:
            Annotated BGR frame copy.
        """
        annotated = frame.copy()

        # Parse detections and view verdict
        detections = []
        verdict_str = "PASS"
        primary_defect = None
        cam_str = camera_id

        if isinstance(view_data, ViewVerdict):
            detections = view_data.detections
            verdict_str = view_data.verdict.value if hasattr(view_data.verdict, "value") else str(view_data.verdict)
            primary_defect = view_data.defect_name
            cam_str = cam_str or view_data.camera_id
        elif isinstance(view_data, CameraResult):
            detections = view_data.detections
            cam_str = cam_str or view_data.camera_id
        elif isinstance(view_data, dict):
            # Parse from JSON dictionary (e.g. from inspection_data.json)
            raw_dets = view_data.get("detections", [])
            for d in raw_dets:
                if isinstance(d, Detection):
                    detections.append(d)
                elif isinstance(d, dict):
                    detections.append(Detection(**d))
            verdict_str = view_data.get("verdict", "PASS")
            primary_defect = view_data.get("defect_name")
            cam_str = cam_str or view_data.get("camera_id")

        if global_result:
            verdict_str = global_result.global_verdict.value if hasattr(global_result.global_verdict, "value") else str(global_result.global_verdict)
            primary_defect = global_result.primary_defect or primary_defect

        # 1. Render Semi-Transparent Masks
        if self.config.show_masks:
            for det in detections:
                color = self.config.get_color(det.class_name)
                if det.mask and len(det.mask) >= 3:
                    self.draw_mask(annotated, det.mask, color)

        # 2. Render Bounding Boxes & Confidence Pills
        for det in detections:
            color = self.config.get_color(det.class_name)
            self.draw_box_and_pill(
                annotated,
                box=det.box,
                class_name=det.class_name,
                confidence=det.confidence,
                color=color
            )

        # 3. Render Top HUD Banner
        if self.config.show_hud_banner:
            lat = global_result.total_latency_ms if global_result else None
            bid = global_result.bottle_id if global_result else None
            self.draw_hud_banner(
                annotated,
                verdict=verdict_str,
                defect_name=primary_defect,
                latency_ms=lat,
                bottle_id=bid,
                camera_id=cam_str
            )

        # 4. Camera ID Tag in bottom-left corner
        if self.config.show_camera_label and cam_str:
            tag_text = f"CAM: {cam_str.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(tag_text, font, 0.5, 1)
            cv2.rectangle(annotated, (6, annotated.shape[0] - th - 12), (tw + 16, annotated.shape[0] - 6), (30, 30, 30), -1)
            cv2.putText(
                annotated,
                tag_text,
                (10, annotated.shape[0] - 10),
                font,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated

    def render_multi_view(
        self,
        frames: Dict[str, np.ndarray],
        global_result: GlobalInspectionResult
    ) -> np.ndarray:
        """
        Renders all camera views and stitches them side-by-side into a unified
        composite multi-camera inspection review image.
        """
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        rendered_views = []
        target_h = None

        for cam_id, frame in frames.items():
            view_verdict = global_result.per_view_results.get(cam_id)
            view_ann = self.render_view(
                frame=frame,
                view_data=view_verdict if view_verdict else {},
                global_result=global_result,
                camera_id=cam_id
            )
            if target_h is None:
                target_h = view_ann.shape[0]
            elif view_ann.shape[0] != target_h:
                # Resize to common height
                scale = target_h / view_ann.shape[0]
                new_w = int(view_ann.shape[1] * scale)
                view_ann = cv2.resize(view_ann, (new_w, target_h), interpolation=cv2.INTER_AREA)

            rendered_views.append(view_ann)

        # Horizontally stack views
        composite = np.hstack(rendered_views)

        # Enforce max width constraint if specified
        if composite.shape[1] > self.config.side_by_side_max_width:
            scale = self.config.side_by_side_max_width / float(composite.shape[1])
            composite = cv2.resize(
                composite,
                (self.config.side_by_side_max_width, int(composite.shape[0] * scale)),
                interpolation=cv2.INTER_AREA
            )

        return composite

    def render_from_bundle(
        self,
        bundle_dir: Union[str, Path],
        camera_id: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Dynamic On-Demand Review Renderer.
        Loads raw, clean frames from a Part 5 bottle bundle directory, parses
        inspection_data.json, and renders dynamic overlays on-the-fly.

        Args:
            bundle_dir: Path to the bottle bundle folder (e.g. audit/bottles/BOTTLE_xxx).
            camera_id: If specified, returns single annotated frame for this camera.
                       If None, returns dictionary of camera_id -> annotated frame.

        Returns:
            Annotated BGR frame or dictionary of camera views.
        """
        b_path = Path(bundle_dir)
        meta_file = b_path / "inspection_data.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"inspection_data.json not found in {bundle_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        global_verdict = meta.get("global_verdict", "PASS")
        primary_defect = meta.get("primary_defect")
        bottle_id = meta.get("bottle_id")
        latency = meta.get("total_latency_ms")
        per_view_data = meta.get("per_view_results", {})

        manifest_images = meta.get("bundle_manifest", {}).get("images", {})
        results = {}

        for cam, img_name in manifest_images.items():
            if camera_id and cam != camera_id:
                continue

            raw_path = b_path / img_name
            if not raw_path.exists():
                logger.warning(f"Raw frame {raw_path} missing in bundle.")
                continue

            raw_frame = cv2.imread(str(raw_path))
            if raw_frame is None:
                continue

            view_data = per_view_data.get(cam, {})
            ann_frame = self.render_view(
                frame=raw_frame,
                view_data=view_data,
                global_result=None,
                camera_id=cam
            )
            # Apply overall bottle banner
            ann_frame = self.draw_hud_banner(
                ann_frame,
                verdict=global_verdict,
                defect_name=primary_defect,
                latency_ms=latency,
                bottle_id=bottle_id,
                camera_id=cam
            )
            results[cam] = ann_frame

        if camera_id:
            return results.get(camera_id, np.zeros((480, 640, 3), dtype=np.uint8))
        return results
