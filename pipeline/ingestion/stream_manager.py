"""
Stream Manager for Multi-Camera Synchronization and Ingestion.
Coordinates parallel camera workers and produces synchronized frame bundles.
"""
import time
from datetime import datetime, timezone
import logging
from typing import List, Dict, Optional, Union, Generator
from pathlib import Path
import cv2
import numpy as np

from .schemas import CameraSourceConfig, SynchronizedFrameBundle
from .threaded_camera import ThreadedCameraWorker

logger = logging.getLogger("pipeline.ingestion.manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StreamManager:
    """
    Manages multi-camera array ingestion.
    Supports:
      1. Live physical cameras (USB3 / GenICam / RTSP).
      2. Threaded non-blocking capture with zero buffer lag.
      3. Offline synchronized image pair / video simulation feeder.
    """
    def __init__(self, camera_configs: Optional[List[CameraSourceConfig]] = None):
        self.camera_configs = camera_configs or []
        self.workers: Dict[str, ThreadedCameraWorker] = {}
        self._bottle_counter = 0

        for cfg in self.camera_configs:
            self.add_camera(cfg)

    def add_camera(self, config: CameraSourceConfig):
        """Adds and launches a threaded camera worker."""
        if config.camera_id in self.workers:
            logger.warning(f"Camera '{config.camera_id}' is already registered. Replacing...")
            self.workers[config.camera_id].stop()

        worker = ThreadedCameraWorker(config)
        self.workers[config.camera_id] = worker
        logger.info(f"Registered camera worker: '{config.camera_id}' (Source: {config.source_path})")

    def acquire_synchronized_bundle(
        self,
        timeout_sec: float = 1.0,
        bottle_id: Optional[str] = None,
    ) -> Optional[SynchronizedFrameBundle]:
        """
        Polls all active camera workers to assemble a synchronized frame bundle.
        Returns None if any camera has not produced a frame within timeout.
        """
        if not self.workers:
            raise RuntimeError("No camera workers are configured in StreamManager.")

        start_time = time.perf_counter()
        frames: Dict[str, np.ndarray] = {}

        while time.perf_counter() - start_time < timeout_sec:
            frames.clear()
            for cid, worker in self.workers.items():
                _, frame, _ = worker.read_latest()
                if frame is not None:
                    frames[cid] = frame

            # If all cameras have delivered at least one frame, package bundle
            if len(frames) == len(self.workers):
                self._bottle_counter += 1
                b_id = bottle_id or f"BOTTLE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{self._bottle_counter:04d}"
                ts = datetime.now(timezone.utc).isoformat()
                return SynchronizedFrameBundle(
                    bottle_id=b_id,
                    timestamp=ts,
                    frames=frames,
                    metadata={"source": "live_camera_array", "counter": self._bottle_counter},
                )
            time.sleep(0.005)

        logger.warning(f"Timeout waiting for synchronized frames. Received: {list(frames.keys())}/{list(self.workers.keys())}")
        return None

    @staticmethod
    def create_offline_bundle(
        camera_images: Dict[str, Union[str, Path, np.ndarray]],
        bottle_id: Optional[str] = None,
    ) -> SynchronizedFrameBundle:
        """
        Directly constructs a SynchronizedFrameBundle from offline test images/arrays.
        Useful for repeatable benchmarking and edge case regression testing.
        """
        loaded_frames: Dict[str, np.ndarray] = {}
        for cid, img_ref in camera_images.items():
            if isinstance(img_ref, (str, Path)):
                img = cv2.imread(str(img_ref))
                if img is None:
                    raise FileNotFoundError(f"Failed to load image for '{cid}' at: {img_ref}")
                loaded_frames[cid] = img
            elif isinstance(img_ref, np.ndarray):
                loaded_frames[cid] = img_ref
            else:
                raise TypeError(f"Unsupported frame type for '{cid}': {type(img_ref)}")

        b_id = bottle_id or f"BOTTLE_TEST_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:21]}"
        ts = datetime.now(timezone.utc).isoformat()
        return SynchronizedFrameBundle(
            bottle_id=b_id,
            timestamp=ts,
            frames=loaded_frames,
            metadata={"source": "offline_bundle"},
        )

    def stop(self):
        """Stops all background camera threads."""
        logger.info(f"Stopping all {len(self.workers)} camera workers...")
        for cid, worker in self.workers.items():
            worker.stop()
        self.workers.clear()
        logger.info("All camera workers stopped.")
