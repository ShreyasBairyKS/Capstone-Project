"""
Threaded Camera Worker for Zero-Lag Asynchronous Frame Acquisition.
Prevents OpenCV buffer bloat by maintaining only the latest frame (queue_size=1).
"""
import time
import threading
import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from .schemas import CameraSourceConfig

logger = logging.getLogger("pipeline.ingestion.camera")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ThreadedCameraWorker:
    """
    Dedicated worker thread for a single physical camera or video stream.
    Features:
      - Non-blocking continuous capture thread.
      - Zero-buffer lag: Keeps only the most recent frame.
      - Zero-copy read support for ultra-high-resolution streams (4K/12MP).
      - Disconnection resilience: Auto-reconnects if stream drops.
    """
    def __init__(self, config: CameraSourceConfig):
        self.config = config
        self.camera_id = config.camera_id
        self.source = config.source_path

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._is_new_frame: bool = False

        self._start_capture()

    def _open_capture(self) -> bool:
        """Initializes cv2.VideoCapture with optimal low-latency flags."""
        logger.info(f"[{self.camera_id}] Opening source: {self.source}...")
        
        # Handle integer device index vs string stream path
        src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            logger.error(f"[{self.camera_id}] Failed to open source: {self.source}")
            return False

        # Set buffer size to 1 to eliminate queue latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.config.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self._cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[{self.camera_id}] Connected successfully! Res: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        return True

    def _start_capture(self):
        """Spawns the background capture thread."""
        if self._open_capture():
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name=f"Worker-{self.camera_id}", daemon=True)
            self._thread.start()
        else:
            logger.warning(f"[{self.camera_id}] Initialization deferred (will retry in loop).")
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name=f"Worker-{self.camera_id}", daemon=True)
            self._thread.start()

    def _capture_loop(self):
        """Continuous background loop reading frames without blocking."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(2.0)
                if self._running:
                    self._open_capture()
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                now = time.perf_counter()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_count += 1
                    self._is_new_frame = True
            else:
                if self.config.source_type == "video":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    logger.warning(f"[{self.camera_id}] Frame grab failed. Re-syncing...")
                    time.sleep(0.01)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info(f"[{self.camera_id}] Capture loop terminated.")

    def read_latest(self, copy: bool = False) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieves the latest available frame.
        Use copy=False for zero-copy reference read (<0.1ms even for 4K/12MP frames).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            
            frame = self._latest_frame.copy() if copy else self._latest_frame
            ts = self._latest_timestamp
            is_new = self._is_new_frame
            self._is_new_frame = False
            return is_new, frame, ts

    def stop(self):
        """Terminates worker thread and releases hardware resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
