"""
scripts/capture_loop.py — Live camera inspection loop for VisionFood QAI.

Captures frames from a webcam (or video file), submits each to the running
FastAPI backend via HTTP, and overlays real-time results on screen.

Two modes:
  - API mode  (default): sends frames to http://localhost:8000/inspections via REST
  - Local mode (--local): runs the inference pipeline in-process (no server needed)

Usage:
    # Start the API first (one terminal):
    uvicorn api.main:app --reload

    # Then run the capture loop (another terminal):
    python scripts/capture_loop.py

    # Local mode (no server needed, slower startup):
    python scripts/capture_loop.py --local

    # Use a video file instead of webcam:
    python scripts/capture_loop.py --source path/to/video.mp4

    # Slow down capture rate:
    python scripts/capture_loop.py --fps 1.0
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from core.config import settings
from core.logging import get_logger, setup_logging

setup_logging(log_format="text")
log = get_logger(__name__)

# Verdict → BGR colour for overlay
VERDICT_COLOURS = {
    "PASS":    (34,  197, 94),    # Green
    "FAIL":    (239, 68,  68),    # Red
    "ESCALATE": (249, 115, 22),  # Orange
    "REVIEW":  (234, 179, 8),     # Yellow
}

DEFECT_BOX_COLOUR = (59, 130, 246)   # Blue
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionFood QAI — live camera inspection loop."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (0, 1, …) or path to video file (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target inspection frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="VisionFood QAI API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=settings.API_KEY,
        help="API key for authentication (reads from .env by default)",
    )
    parser.add_argument(
        "--sku",
        type=str,
        default="default",
        help="SKU identifier sent with each inspection (default: default)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run inference pipeline in-process instead of calling the API",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save flagged (FAIL/ESCALATE) frames to this directory",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress OpenCV window — useful for headless servers",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Frame → base64 helper
# --------------------------------------------------------------------------- #

def frame_to_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR numpy array as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# --------------------------------------------------------------------------- #
# API mode
# --------------------------------------------------------------------------- #

def inspect_via_api(
    frame: np.ndarray,
    api_url: str,
    api_key: str,
    sku: str,
    product_counter: int,
) -> dict | None:
    """Submit frame to FastAPI backend and return the JSON result dict."""
    try:
        import requests
    except ImportError:
        log.error("requests_not_installed", hint="pip install requests")
        return None

    payload = {
        "image_b64": frame_to_b64(frame),
        "product_id": f"LIVE-{product_counter:06d}",
        "sku": sku,
        "attempt_count": 0,
    }
    try:
        resp = requests.post(
            f"{api_url}/inspections",
            json=payload,
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("api_request_failed", error=str(e))
        return None


# --------------------------------------------------------------------------- #
# Local pipeline mode
# --------------------------------------------------------------------------- #

_LOCAL_PIPELINE = None


def _get_local_pipeline():
    """Lazy initialise the local inference pipeline (loads ONNX models once)."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        from inference.pipeline import EdgeInferencePipeline
        log.info("loading_local_pipeline")
        print("Loading ONNX models…", flush=True)
        _LOCAL_PIPELINE = EdgeInferencePipeline()
        try:
            _LOCAL_PIPELINE.load_models()
            log.info("local_pipeline_ready")
        except Exception as e:
            log.warning("local_pipeline_model_load_failed", error=str(e))
            print(f"  [WARN] Models not found: {e}")
            print("  Running pipeline without ONNX models (verdict logic only).")
    return _LOCAL_PIPELINE


def inspect_locally(
    frame: np.ndarray,
    sku: str,
    product_counter: int,
) -> dict | None:
    """Run inference in-process and return a dict compatible with API response shape."""
    pipeline = _get_local_pipeline()
    try:
        result = pipeline.inspect(
            frame=frame,
            product_id=f"LIVE-{product_counter:06d}",
            sku=sku,
        )
        return result.model_dump()
    except Exception as e:
        log.warning("local_pipeline_error", error=str(e))
        return None


# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #

def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Draw inspection result overlay on a copy of the frame.

    Renders:
     - Verdict banner (top bar with colour-coded background)
     - Bounding boxes for each detected defect
     - Latency and confidence text
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    verdict = result.get("verdict", "REVIEW")
    colour = VERDICT_COLOURS.get(verdict, (200, 200, 200))
    latency = result.get("latency_ms", 0.0)

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, 50), colour, -1)
    # Verdict text
    cv2.putText(overlay, f"  {verdict}", (8, 35), FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    # Latency
    cv2.putText(overlay, f"{latency:.0f}ms", (w - 110, 35), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Defect bounding boxes
    for det in result.get("detections", []):
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0) * w)
        y1 = int(bbox.get("y1", 0) * h) + 50   # offset below banner
        x2 = int(bbox.get("x2", 0) * w)
        y2 = int(bbox.get("y2", 0) * h) + 50

        conf = det.get("confidence", 0.0)
        name = det.get("class_name", "defect")
        label = f"{name} {conf:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), DEFECT_BOX_COLOUR, 2)
        label_y = max(y1 - 8, 60)
        cv2.putText(overlay, label, (x1, label_y), FONT, 0.55, DEFECT_BOX_COLOUR, 1, cv2.LINE_AA)

    # UQ info (if present)
    uq = result.get("uq_result")
    if uq:
        uq_text = f"UQ: mean={uq.get('mean_confidence', 0):.2f}  std={uq.get('std_confidence', 0):.2f}"
        uncertain_flag = "  UNCERTAIN" if uq.get("is_uncertain") else ""
        cv2.putText(
            overlay, uq_text + uncertain_flag,
            (8, h - 12), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA
        )

    # Severity + remediation action
    sev = result.get("severity_result")
    action = result.get("remediation_action")
    if sev and action:
        sev_text = f"Severity: {sev.get('grade', '?')}  →  {action.get('action', '?')}"
        cv2.putText(
            overlay, sev_text,
            (8, h - 32), FONT, 0.55, (255, 200, 50), 1, cv2.LINE_AA
        )

    return overlay


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    # Open capture source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    log.info("capture_loop_start", source=args.source, fps=args.fps, mode="local" if args.local else "api")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = 1.0 / max(args.fps, 0.1)
    product_counter = 0
    last_result: dict | None = None
    last_inspect_t = 0.0

    print("\nVisionFood QAI — Live Inspection")
    print(f"  Source : {args.source}")
    print(f"  Mode   : {'local' if args.local else 'API → ' + args.api_url}")
    print(f"  FPS    : {args.fps}")
    print("  Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.info("capture_ended")
            break

        now = time.perf_counter()
        if now - last_inspect_t >= frame_interval:
            last_inspect_t = now
            product_counter += 1

            if args.local:
                result = inspect_locally(frame, args.sku, product_counter)
            else:
                result = inspect_via_api(frame, args.api_url, args.api_key, args.sku, product_counter)

            if result:
                last_result = result
                verdict = result.get("verdict", "?")
                latency = result.get("latency_ms", 0.0)
                n_dets = len(result.get("detections", []))
                print(f"  [{product_counter:06d}]  verdict={verdict:<9}  dets={n_dets}  latency={latency:.0f}ms")

                # Save flagged frames
                if args.save_dir and verdict in ("FAIL", "ESCALATE"):
                    fname = args.save_dir / f"{product_counter:06d}_{verdict}.jpg"
                    cv2.imwrite(str(fname), frame)

        # Draw overlay using most recent result
        display_frame = draw_overlay(frame, last_result) if last_result else frame.copy()

        if not args.no_display:
            cv2.imshow("VisionFood QAI — Live Inspection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or Esc
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    log.info("capture_loop_stopped", total_frames=product_counter)
    print(f"\nInspected {product_counter} products. Exiting.")


if __name__ == "__main__":
    main()
