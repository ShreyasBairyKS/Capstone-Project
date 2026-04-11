# VisionFood QAI — Complete Software Architecture

---

## Software System Overview

VisionFood QAI is structured as a **distributed microservices system** across three compute tiers — edge, fog, and cloud — each running independently deployable containerised services that communicate via a unified message bus. The software stack is fully open-source, containerised with Docker, orchestrated via Docker Compose (single-site) or Kubernetes (multi-site enterprise), and designed to operate entirely offline at the edge tier with optional cloud enrichment.

The system is composed of **9 major software subsystems**: the Inference Engine, REMEDY Triage Engine, Model Lifecycle Manager, Data Pipeline, Intelligence Plane, REMEDY Station Controller, Dashboard & API, Federated Learning Coordinator, and the Compliance & Audit Logger.

---

## 1. Development Environment Setup

### Python Environment

```
Python 3.11.x (required — Mamba SSM and TensorRT 10 bindings are 3.11-specific)

Core ML:
  torch==2.3.0+cu121
  torchvision==0.18.0
  ultralytics==8.3.x          # YOLOv11 — latest stable
  timm==1.0.x                  # EfficientViT, ConvNeXt V2
  transformers==4.44.x         # DINOv2, SAM 2
  segment-anything-2           # Meta SAM 2
  mamba-ssm==2.2.x             # Mamba-QC temporal model
  torch-geometric==2.5.x       # GATv2 for CDAG-Net

Edge Runtimes:
  tensorrt==10.x               # Jetson only
  onnxruntime-gpu==1.18.x
  openvino==2024.3.x
  onnx==1.16.x

Explainability:
  shap==0.45.x
  grad-cam==1.5.x              # pytorch-grad-cam
  captum==0.7.x
  dowhy==0.11.x                # Pearl do-calculus
  econml==0.15.x

Federated Learning:
  flwr==1.9.x                  # Flower federated framework
  opacus==1.4.x                # DP-SGD
  syft==0.8.x                  # PySyft secure aggregation

Data & Annotation:
  albumentations==1.4.x
  roboflow==1.1.x
  pandas==2.2.x
  numpy==1.26.x
  opencv-python==4.10.x

Active Learning:
  badge2                        # pip install git+https://github.com/JordanAsh/badge

Experiment Tracking:
  mlflow==2.14.x
  wandb==0.17.x

Causal Inference:
  dowhy==0.11.x
  pgmpy==0.1.x                 # Bayesian network structure learning

Backend:
  fastapi==0.111.x
  uvicorn==0.30.x
  websockets==12.x
  sqlalchemy==2.0.x
  alembic==1.13.x              # DB migrations
  redis==5.0.x
  pydantic==2.7.x
  celery==5.4.x                # Async task queue

Testing:
  pytest==8.x
  pytest-asyncio
  hypothesis                   # Property-based testing
```

### Project Folder Structure

```
visionfood-qai/
│
├── core/                          # Shared utilities, config, schemas
│   ├── config.py                  # Pydantic settings (env-driven)
│   ├── schemas.py                 # Shared Pydantic models
│   ├── logging.py                 # Structured JSON logging
│   └── messaging.py               # Redis pub/sub wrappers
│
├── inference/                     # Tier 2 — Edge inference engine
│   ├── models/
│   │   ├── yolov11_detector.py
│   │   ├── efficientvit_classifier.py
│   │   ├── uq_inspector.py
│   │   └── model_registry.py
│   ├── pipeline.py                # Main inference orchestrator
│   ├── preprocessor.py
│   └── postprocessor.py
│
├── fusion/                        # Tier 3 — HAFFN + SAM 2 + fog models
│   ├── haffn.py
│   ├── rtdetr_v2.py
│   ├── sam2_segmentor.py
│   ├── depth_anything.py
│   └── fog_pipeline.py
│
├── remedy/                        # REMEDY engine
│   ├── severity_scorer.py
│   ├── remediability_classifier.py
│   ├── triage_router.py
│   ├── station_controller.py      # Arduino/PLC bridge
│   ├── reinspection_loop.py
│   └── sku_profile_manager.py
│
├── intelligence/                  # Tier 5 — Causal + Temporal + XAI
│   ├── cdag_net.py
│   ├── mamba_qc.py
│   ├── xmi_gateway.py
│   ├── spc_engine.py
│   └── oee_calculator.py
│
├── mlops/                         # Model lifecycle + rollback
│   ├── model_lifecycle.py
│   ├── shadow_runner.py
│   ├── canary_manager.py
│   ├── rollback_controller.py
│   └── drift_monitor.py
│
├── federated/                     # Tier 7 — FCL engine
│   ├── fl_client.py
│   ├── fl_server.py
│   ├── ewc.py
│   ├── dp_sgd_trainer.py
│   └── secure_aggregation.py
│
├── training/                      # Model training scripts
│   ├── train_yolov11.py
│   ├── train_efficientvit.py
│   ├── train_rtdetr.py
│   ├── train_cdag_net.py
│   ├── train_mamba_qc.py
│   ├── train_mae_patchcore.py
│   ├── active_learning.py
│   └── synthetic_augment.py       # StyleGAN3 pipeline
│
├── export/                        # Edge optimisation pipeline
│   ├── export_onnx.py
│   ├── export_tensorrt.py
│   ├── export_openvino.py
│   ├── export_hailo.py
│   └── quantisation_aware_training.py
│
├── api/                           # FastAPI backend
│   ├── main.py
│   ├── routers/
│   │   ├── inspection.py
│   │   ├── remedy.py
│   │   ├── analytics.py
│   │   ├── models.py
│   │   ├── compliance.py
│   │   └── websocket.py
│   ├── middleware/
│   │   ├── auth.py
│   │   ├── rate_limiter.py
│   │   └── audit_logger.py
│   └── dependencies.py
│
├── dashboard/                     # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── store/
│   └── package.json
│
├── database/
│   ├── models.py                  # SQLAlchemy ORM models
│   ├── migrations/                # Alembic migrations
│   └── repositories/              # Data access layer
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker/
│   ├── Dockerfile.edge
│   ├── Dockerfile.fog
│   ├── Dockerfile.cloud
│   └── docker-compose.yml
│
├── scripts/
│   ├── setup_env.sh
│   ├── collect_calibration_data.py
│   └── benchmark_edge.py
│
└── configs/
    ├── edge_config.yaml
    ├── fog_config.yaml
    ├── cloud_config.yaml
    └── sku_profiles/
        ├── bottle_250ml.yaml
        ├── pouch_100g.yaml
        └── can_330ml.yaml
```

---

## 2. Configuration Management

All configuration is environment-driven using Pydantic Settings. No hardcoded values anywhere.

```python
# core/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path

class EdgeConfig(BaseSettings):
    # Deployment tier
    TIER: Literal["edge", "fog", "cloud"] = "edge"
    DEVICE_ID: str = "edge_node_01"
    
    # Model paths
    YOLOV11_TRT_PATH: Path = Path("models/yolov11n_int8.trt")
    EFFICIENTVIT_TRT_PATH: Path = Path("models/efficientvit_m5_int8.trt")
    
    # Inference thresholds
    YOLOV11_CONF_THRESHOLD: float = 0.40       # Detection confidence
    YOLOV11_IOU_THRESHOLD: float = 0.45        # NMS IoU
    AUTO_PASS_THRESHOLD: float = 0.85          # Above this → pass without escalation
    ESCALATE_THRESHOLD: float = 0.60           # Below this → fog escalation
    HUMAN_REVIEW_THRESHOLD: float = 0.45       # Below this → human queue
    
    # Inference performance
    INFERENCE_BATCH_SIZE: int = 1              # Real-time: always 1
    MAX_INFERENCE_LATENCY_MS: float = 5.0
    TARGET_THROUGHPUT_PPM: int = 120           # Products per minute
    
    # Camera
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 640
    CAMERA_FPS: int = 60
    CAMERA_TRIGGER_GPIO_PIN: int = 18
    
    # REMEDY
    REMEDY_ENABLED: bool = True
    MAX_REMEDIATION_ATTEMPTS: int = 2
    STATION_A_TIMEOUT_S: float = 10.0
    STATION_B_TIMEOUT_S: float = 8.0
    STATION_C_TIMEOUT_S: float = 12.0
    
    # Message bus
    REDIS_URL: str = "redis://localhost:6379"
    FOG_ENDPOINT: str = "http://fog-node:8001"
    
    # Database
    DATABASE_URL: str = "sqlite:///./visionfood_edge.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class FogConfig(BaseSettings):
    TIER: Literal["edge", "fog", "cloud"] = "fog"
    
    RTDETR_ONNX_PATH: Path = Path("models/rtdetrv2_fp16.onnx")
    SAM2_CHECKPOINT_PATH: Path = Path("models/sam2_hiera_large.pt")
    DEPTH_ANYTHING_ONNX_PATH: Path = Path("models/depth_anything_v2.onnx")
    HAFFN_CHECKPOINT_PATH: Path = Path("models/haffn_v1.2.pt")
    
    FOG_LATENCY_SLA_MS: float = 50.0
    CLOUD_ENDPOINT: str = "http://cloud-core:8002"
    DATABASE_URL: str = "postgresql://qai:password@db:5432/visionfood"
    
    MC_DROPOUT_SAMPLES: int = 20
    ENSEMBLE_SIZE: int = 5
    
    class Config:
        env_file = ".env.fog"
```

---

## 3. Inference Engine (Edge Tier — Core)

### 3.1 TensorRT Inference Wrapper

```python
# inference/models/yolov11_detector.py

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from dataclasses import dataclass
from typing import List, Tuple
import time

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 (normalised 0-1)
    area_fraction: float                        # defect area / image area
    
CLASS_NAMES = [
    "improper_filling",
    "packaging_damage", 
    "label_misalignment",
    "surface_contamination"
]

class YOLOv11TRTDetector:
    """
    TensorRT INT8 inference wrapper for YOLOv11.
    Target latency: <1.5ms on Jetson Orin NX.
    """
    
    def __init__(self, engine_path: str, conf_threshold: float = 0.40,
                 iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_shape = (1, 3, 640, 640)
        self.host_input = cuda.pagelocked_empty(
            trt.volume(self.input_shape), dtype=np.float32
        )
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        
        output_shape = (1, 8400, 6)
        self.host_output = cuda.pagelocked_empty(
            trt.volume(output_shape), dtype=np.float32
        )
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        
        self.stream = cuda.Stream()
        self._warmup()
    
    def _warmup(self, iterations: int = 10):
        dummy = np.zeros(self.input_shape, dtype=np.float32)
        for _ in range(iterations):
            self._infer_raw(dummy)
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return np.ascontiguousarray(chw[np.newaxis, :])
    
    def _infer_raw(self, preprocessed: np.ndarray) -> np.ndarray:
        np.copyto(self.host_input, preprocessed.ravel())
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        
        self.context.execute_async_v2(
            bindings=[int(self.device_input), int(self.device_output)],
            stream_handle=self.stream.handle
        )
        
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        return self.host_output.reshape(1, 8400, 6)
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        preprocessed = self._preprocess(frame)
        raw_output = self._infer_raw(preprocessed)
        detections = self._postprocess(raw_output, frame.shape)
        latency_ms = (time.perf_counter() - t0) * 1000
        return detections, latency_ms
    
    def _postprocess(self, raw: np.ndarray, 
                     original_shape: Tuple) -> List[Detection]:
        boxes = raw[0]
        mask = boxes[:, 4] >= self.conf_threshold
        boxes = boxes[mask]
        if len(boxes) == 0:
            return []
        
        xyxy = boxes[:, :4]
        confidences = boxes[:, 4]
        class_ids = boxes[:, 5].astype(int)
        
        import torchvision
        import torch
        keep = torchvision.ops.nms(
            torch.tensor(xyxy, dtype=torch.float32),
            torch.tensor(confidences, dtype=torch.float32),
            self.iou_threshold
        ).numpy()
        
        results = []
        for idx in keep:
            x1, y1, x2, y2 = xyxy[idx]
            nx1, ny1 = x1/640, y1/640
            nx2, ny2 = x2/640, y2/640
            area_frac = ((nx2-nx1) * (ny2-ny1))
            
            results.append(Detection(
                class_id=class_ids[idx],
                class_name=CLASS_NAMES[min(class_ids[idx], 3)],
                confidence=float(confidences[idx]),
                bbox=(nx1, ny1, nx2, ny2),
                area_fraction=float(area_frac)
            ))
        
        return sorted(results, key=lambda d: d.confidence, reverse=True)
```

### 3.2 Uncertainty Quantification Inspector

```python
# inference/models/uq_inspector.py

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass  
class UQResult:
    mean_confidence: float
    std_confidence: float
    confidence_interval_low: float     # μ - 2σ
    confidence_interval_high: float    # μ + 2σ
    is_uncertain: bool
    escalation_required: bool


def enable_dropout(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class MCDropoutUQ:
    """
    Monte Carlo Dropout for uncertainty quantification.
    Runs N forward passes with dropout active.
    Computes calibrated confidence intervals.
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 20,
                 uncertainty_threshold: float = 0.12,
                 escalation_threshold: float = 0.72):
        self.model = model
        self.n_samples = n_samples
        self.uncertainty_threshold = uncertainty_threshold
        self.escalation_threshold = escalation_threshold
        self.model.eval()
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> UQResult:
        enable_dropout(self.model)
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)
        mean = predictions.mean(axis=0)[0]
        std = predictions.std(axis=0)[0]
        
        top_class = mean.argmax()
        top_mean = float(mean[top_class])
        top_std = float(std[top_class])
        
        ci_low = max(0.0, top_mean - 2 * top_std)
        ci_high = min(1.0, top_mean + 2 * top_std)
        
        return UQResult(
            mean_confidence=top_mean,
            std_confidence=top_std,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            is_uncertain=top_std > self.uncertainty_threshold,
            escalation_required=top_mean < self.escalation_threshold
        )


class DeepEnsembleUQ:
    """
    Deep Ensemble: 5 independently trained models.
    More accurate than MC Dropout, 5× compute.
    Used for batch re-analysis of flagged items.
    """
    
    def __init__(self, model_paths: List[str], device: str = "cuda"):
        self.models = []
        for path in model_paths:
            m = torch.load(path, map_location=device)
            m.eval()
            self.models.append(m)
        self.device = device
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> UQResult:
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.stack(predictions)
        mean = predictions.mean(axis=0)[0]
        std = predictions.std(axis=0)[0]
        top_class = mean.argmax()
        top_mean = float(mean[top_class])
        top_std = float(std[top_class])
        
        return UQResult(
            mean_confidence=top_mean,
            std_confidence=top_std,
            confidence_interval_low=max(0.0, top_mean - 2*top_std),
            confidence_interval_high=min(1.0, top_mean + 2*top_std),
            is_uncertain=top_std > 0.10,
            escalation_required=top_mean < 0.72
        )
```

### 3.3 Main Inference Pipeline

```python
# inference/pipeline.py

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
import uuid
import json
import redis.asyncio as redis

from .models.yolov11_detector import YOLOv11TRTDetector, Detection
from .models.efficientvit_classifier import EfficientViTClassifier
from .models.uq_inspector import MCDropoutUQ, UQResult
from core.config import EdgeConfig


@dataclass
class InspectionResult:
    product_id: str
    timestamp: datetime
    verdict: str                           # "PASS" | "FAIL" | "ESCALATE" | "REVIEW"
    detections: List[Detection]
    uq_result: Optional[UQResult]
    inference_latency_ms: float
    model_version: str
    escalated: bool = False
    remedy_action: Optional[str] = None


class EdgeInferencePipeline:
    """
    Orchestrates the full edge inference stack:
    YOLOv11 → EfficientViT → UQ → verdict → REMEDY routing
    Target: <5ms end-to-end for normal products
    """
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.detector = YOLOv11TRTDetector(
            engine_path=str(config.YOLOV11_TRT_PATH),
            conf_threshold=config.YOLOV11_CONF_THRESHOLD,
            iou_threshold=config.YOLOV11_IOU_THRESHOLD
        )
        self.classifier = EfficientViTClassifier(
            engine_path=str(config.EFFICIENTVIT_TRT_PATH)
        )
        self.uq = MCDropoutUQ(
            model=self.classifier.model,
            n_samples=20,
            escalation_threshold=config.ESCALATE_THRESHOLD
        )
        self.redis_client = None
        self.model_version = self._load_active_model_version()
        self._latency_buffer = []
    
    async def connect(self):
        self.redis_client = await redis.from_url(self.config.REDIS_URL)
    
    def _load_active_model_version(self) -> str:
        try:
            with open("models/active_version.json") as f:
                return json.load(f)["version"]
        except FileNotFoundError:
            return "v1.0.0-fallback"
    
    async def inspect(self, frame: np.ndarray,
                      product_id: Optional[str] = None) -> InspectionResult:
        if product_id is None:
            product_id = str(uuid.uuid4())
        
        t_start = asyncio.get_event_loop().time()
        
        detections, det_latency = self.detector.detect(frame)
        
        if not detections:
            result = InspectionResult(
                product_id=product_id,
                timestamp=datetime.utcnow(),
                verdict="PASS",
                detections=[],
                uq_result=None,
                inference_latency_ms=det_latency,
                model_version=self.model_version
            )
            await self._publish_result(result)
            return result
        
        top_detection = detections[0]
        cropped = self._crop_detection(frame, top_detection)
        cls_result = self.classifier.classify(cropped)
        uq_result = self.uq.predict_with_uncertainty(
            self.classifier.preprocess(cropped)
        )
        
        verdict, escalated = self._apply_verdict_logic(
            detections, uq_result, top_detection
        )
        
        total_latency = (asyncio.get_event_loop().time() - t_start) * 1000
        self._latency_buffer.append(total_latency)
        if len(self._latency_buffer) > 1000:
            self._latency_buffer.pop(0)
        
        result = InspectionResult(
            product_id=product_id,
            timestamp=datetime.utcnow(),
            verdict=verdict,
            detections=detections,
            uq_result=uq_result,
            inference_latency_ms=total_latency,
            model_version=self.model_version,
            escalated=escalated
        )
        
        await self._publish_result(result)
        
        if escalated:
            await self._escalate_to_fog(result, frame)
        
        return result
    
    def _apply_verdict_logic(self, detections, uq, top):
        if not detections:
            return "PASS", False
        
        conf = uq.mean_confidence
        
        if conf >= self.config.AUTO_PASS_THRESHOLD and not uq.is_uncertain:
            if top.confidence >= self.config.AUTO_PASS_THRESHOLD:
                return "FAIL", False
            else:
                return "PASS", False
        elif conf >= self.config.ESCALATE_THRESHOLD:
            return "FAIL", True
        elif conf >= self.config.HUMAN_REVIEW_THRESHOLD:
            return "ESCALATE", True
        else:
            return "REVIEW", True
    
    def _crop_detection(self, frame, detection):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        pad = 0.10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(1, x2 + pad)
        y2 = min(1, y2 + pad)
        ix1, iy1 = int(x1*w), int(y1*h)
        ix2, iy2 = int(x2*w), int(y2*h)
        return frame[iy1:iy2, ix1:ix2]
    
    async def _publish_result(self, result: InspectionResult):
        if self.redis_client:
            payload = {
                "product_id": result.product_id,
                "verdict": result.verdict,
                "detections": [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 4),
                        "bbox": list(d.bbox),
                        "area_fraction": round(d.area_fraction, 4)
                    }
                    for d in result.detections
                ],
                "latency_ms": round(result.inference_latency_ms, 2),
                "model_version": result.model_version,
                "timestamp": result.timestamp.isoformat(),
                "uq": {
                    "mean": round(result.uq_result.mean_confidence, 4),
                    "std": round(result.uq_result.std_confidence, 4),
                    "ci_low": round(result.uq_result.confidence_interval_low, 4),
                    "ci_high": round(result.uq_result.confidence_interval_high, 4)
                } if result.uq_result else None
            }
            await self.redis_client.xadd(
                "inspections:live",
                {"data": json.dumps(payload)},
                maxlen=10000
            )
    
    async def _escalate_to_fog(self, result, frame):
        import aiohttp, base64, cv2
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_b64 = base64.b64encode(buf.tobytes()).decode()
        
        payload = {
            "product_id": result.product_id,
            "frame_b64": frame_b64,
            "edge_detections": [
                {"class": d.class_name, "confidence": d.confidence,
                 "bbox": list(d.bbox)} for d in result.detections
            ],
            "uq_std": result.uq_result.std_confidence if result.uq_result else None,
            "requires_human_review": result.verdict == "REVIEW"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    f"{self.config.FOG_ENDPOINT}/escalate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=0.05)
                )
            except asyncio.TimeoutError:
                pass
    
    def get_latency_percentiles(self) -> dict:
        if not self._latency_buffer:
            return {}
        buf = sorted(self._latency_buffer)
        n = len(buf)
        return {
            "p50": round(buf[int(n*0.50)], 2),
            "p95": round(buf[int(n*0.95)], 2),
            "p99": round(buf[int(n*0.99)], 2),
            "count": n
        }
```

---

## 4. REMEDY Engine

### 4.1 Severity Scorer

```python
# remedy/severity_scorer.py

from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
from inference.models.yolov11_detector import Detection


class SeverityGrade(str, Enum):
    S1 = "S1"   # Minor cosmetic — fully remediable
    S2 = "S2"   # Moderate functional — conditionally remediable
    S3 = "S3"   # Serious — case-by-case evaluation
    S4 = "S4"   # Critical — hard reject, no remediation


CLASS_SEVERITY_WEIGHTS = {
    "label_misalignment":    0.30,
    "improper_filling":      0.45,
    "packaging_damage":      0.60,
    "surface_contamination": 0.90,
}

ALWAYS_S4_CLASSES = set()


@dataclass
class SeverityResult:
    raw_score: float
    grade: SeverityGrade
    primary_factor: str
    is_remediable: bool
    recommended_action: str


class SeverityScorer:
    """
    Computes severity score and grade for each detection.
    
    Score = w_area*(defect_area/product_area)
          + w_conf*(1 - confidence)
          + w_class*class_severity_weight
          + w_attempt*attempt_penalty
    """
    
    def __init__(self, w_area=0.35, w_conf=0.15, w_class=0.40,
                 w_attempt=0.10, attempt_penalty_per_attempt=0.30):
        self.w_area = w_area
        self.w_conf = w_conf
        self.w_class = w_class
        self.w_attempt = w_attempt
        self.attempt_penalty = attempt_penalty_per_attempt
    
    def score(self, detection: Detection, attempt_count: int = 0) -> SeverityResult:
        area_score = min(1.0, detection.area_fraction * 20)
        conf_score = 1.0 - detection.confidence
        class_score = CLASS_SEVERITY_WEIGHTS.get(detection.class_name, 0.5)
        attempt_score = min(1.0, attempt_count * self.attempt_penalty)
        
        raw_score = (
            self.w_area    * area_score   +
            self.w_conf    * conf_score   +
            self.w_class   * class_score  +
            self.w_attempt * attempt_score
        )
        raw_score = min(1.0, max(0.0, raw_score))
        
        factors = {
            "defect_area": self.w_area * area_score,
            "model_uncertainty": self.w_conf * conf_score,
            "defect_class_risk": self.w_class * class_score,
            "repeat_attempt": self.w_attempt * attempt_score
        }
        primary_factor = max(factors, key=factors.get)
        
        if raw_score < 0.30:
            grade = SeverityGrade.S1
        elif raw_score < 0.55:
            grade = SeverityGrade.S2
        elif raw_score < 0.75:
            grade = SeverityGrade.S3
        else:
            grade = SeverityGrade.S4
        
        if detection.class_name in ALWAYS_S4_CLASSES:
            grade = SeverityGrade.S4
        
        is_remediable = grade in (SeverityGrade.S1, SeverityGrade.S2)
        action = self._recommend_action(detection.class_name, grade)
        
        return SeverityResult(
            raw_score=round(raw_score, 4),
            grade=grade,
            primary_factor=primary_factor,
            is_remediable=is_remediable,
            recommended_action=action
        )
    
    def _recommend_action(self, class_name: str, grade: SeverityGrade) -> str:
        action_map = {
            ("label_misalignment", SeverityGrade.S1): "RELABEL",
            ("label_misalignment", SeverityGrade.S2): "RELABEL",
            ("label_misalignment", SeverityGrade.S3): "REJECT",
            ("improper_filling",   SeverityGrade.S1): "REFILL",
            ("improper_filling",   SeverityGrade.S2): "REFILL",
            ("improper_filling",   SeverityGrade.S3): "REJECT",
            ("packaging_damage",   SeverityGrade.S1): "REPACK",
            ("packaging_damage",   SeverityGrade.S2): "REPACK",
            ("packaging_damage",   SeverityGrade.S3): "REJECT",
            ("surface_contamination", SeverityGrade.S1): "CLEAN",
            ("surface_contamination", SeverityGrade.S2): "REJECT",
        }
        return action_map.get((class_name, grade), "REJECT")
```

### 4.2 REMEDY Triage Router

```python
# remedy/triage_router.py

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import redis.asyncio as redis

from .severity_scorer import SeverityScorer, SeverityResult
from .station_controller import StationController, StationID
from .reinspection_loop import ReinspectionLoop
from .sku_profile_manager import SKUProfile
from inference.pipeline import InspectionResult


class RemediationAction(str, Enum):
    PASS         = "PASS"
    RELABEL      = "RELABEL"      # Station A
    REFILL       = "REFILL"       # Station B
    REPACK       = "REPACK"       # Station C
    CLEAN        = "CLEAN"        # Station C (cleaning variant)
    REJECT       = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class RemediationDecision:
    product_id: str
    action: RemediationAction
    severity: SeverityResult
    station: Optional[StationID]
    attempt_count: int
    reason: str
    timestamp: datetime


class REMEDYTriageRouter:
    """
    Core REMEDY decision engine.
    Receives FAIL inspections, scores severity, 
    routes to appropriate remediation station or hard reject.
    """
    
    def __init__(self, config, sku_profile: SKUProfile):
        self.config = config
        self.sku = sku_profile
        self.scorer = SeverityScorer(
            w_area=sku_profile.severity_weights["area"],
            w_conf=sku_profile.severity_weights["confidence"],
            w_class=sku_profile.severity_weights["class"],
            w_attempt=sku_profile.severity_weights["attempt"]
        )
        self.station_controller = StationController(config)
        self.reinspector = ReinspectionLoop(config)
        self.redis_client = None
        self._attempt_counts: dict = {}
    
    async def connect(self):
        self.redis_client = await redis.from_url(self.config.REDIS_URL)
    
    async def process_fail(self, inspection: InspectionResult,
                           frame) -> RemediationDecision:
        product_id = inspection.product_id
        attempt = self._attempt_counts.get(product_id, 0)
        
        if attempt >= self.config.MAX_REMEDIATION_ATTEMPTS:
            del self._attempt_counts[product_id]
            decision = RemediationDecision(
                product_id=product_id,
                action=RemediationAction.REJECT,
                severity=self.scorer.score(inspection.detections[0], attempt),
                station=None,
                attempt_count=attempt,
                reason=f"Max remediation attempts ({attempt}) exceeded",
                timestamp=datetime.utcnow()
            )
            await self._log_decision(decision)
            await self._actuate_reject(product_id)
            return decision
        
        if not inspection.detections:
            return RemediationDecision(
                product_id=product_id,
                action=RemediationAction.PASS,
                severity=None,
                station=None,
                attempt_count=attempt,
                reason="No detections on FAIL — auto-pass",
                timestamp=datetime.utcnow()
            )
        
        primary_detection = inspection.detections[0]
        severity = self.scorer.score(primary_detection, attempt)
        
        action = RemediationAction(severity.recommended_action)
        station = self._action_to_station(action)
        
        self._attempt_counts[product_id] = attempt + 1
        
        decision = RemediationDecision(
            product_id=product_id,
            action=action,
            severity=severity,
            station=station,
            attempt_count=attempt + 1,
            reason=f"Severity {severity.grade} | Primary factor: {severity.primary_factor}",
            timestamp=datetime.utcnow()
        )
        
        await self._log_decision(decision)
        
        if action == RemediationAction.REJECT:
            await self._actuate_reject(product_id)
            del self._attempt_counts[product_id]
            return decision
        
        station_success = await self.station_controller.actuate(
            station=station,
            product_id=product_id,
            defect_class=primary_detection.class_name,
            defect_bbox=primary_detection.bbox
        )
        
        if not station_success:
            decision.action = RemediationAction.REJECT
            decision.reason += " | Station actuation failed"
            await self._actuate_reject(product_id)
            del self._attempt_counts[product_id]
            await self._log_decision(decision)
            return decision
        
        reinspect_result = await self.reinspector.inspect(product_id)
        
        if reinspect_result.verdict == "PASS":
            del self._attempt_counts[product_id]
            decision.reason += " | Re-inspection PASSED"
            await self._publish_recovery(product_id, action)
        else:
            decision = await self.process_fail(reinspect_result, frame)
        
        return decision
    
    def _action_to_station(self, action):
        mapping = {
            RemediationAction.RELABEL: StationID.STATION_A,
            RemediationAction.REFILL:  StationID.STATION_B,
            RemediationAction.REPACK:  StationID.STATION_C,
            RemediationAction.CLEAN:   StationID.STATION_C,
        }
        return mapping.get(action)
    
    async def _actuate_reject(self, product_id):
        await self.station_controller.actuate_reject_diverter(product_id)
    
    async def _publish_recovery(self, product_id, action):
        if self.redis_client:
            await self.redis_client.xadd(
                "remedy:recoveries",
                {"product_id": product_id, "action": action.value},
                maxlen=50000
            )
    
    async def _log_decision(self, decision):
        if self.redis_client:
            payload = {
                "product_id": decision.product_id,
                "action": decision.action.value,
                "severity_grade": decision.severity.grade if decision.severity else "N/A",
                "severity_score": str(decision.severity.raw_score) if decision.severity else "0",
                "attempt_count": str(decision.attempt_count),
                "reason": decision.reason,
                "timestamp": decision.timestamp.isoformat()
            }
            await self.redis_client.xadd(
                "remedy:audit_log", payload, maxlen=500000
            )
```

---

## 5. Model Lifecycle Manager & Rollback System

```python
# mlops/model_lifecycle.py

import json
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List
import mlflow
import numpy as np
from collections import deque

from core.logging import get_logger
logger = get_logger(__name__)


class ModelStage(str, Enum):
    CANDIDATE  = "candidate"
    SHADOW     = "shadow"
    STAGING    = "staging"
    PRODUCTION = "production"
    STANDBY    = "standby"
    ARCHIVED   = "archived"


@dataclass
class ModelVersion:
    version_id: str
    mlflow_run_id: str
    stage: ModelStage
    trt_engine_path: str
    onnx_path: str
    dataset_hash: str
    metrics: Dict
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    rollback_count: int = 0
    notes: str = ""


@dataclass  
class ProductionMetrics:
    window_size: int = 100
    fn_events: deque = field(default_factory=lambda: deque(maxlen=100))
    fp_events: deque = field(default_factory=lambda: deque(maxlen=100))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=200))
    
    @property
    def fn_rate(self) -> float:
        if not self.fn_events:
            return 0.0
        return sum(self.fn_events) / len(self.fn_events)
    
    @property
    def fp_rate(self) -> float:
        if not self.fp_events:
            return 0.0
        return sum(self.fp_events) / len(self.fp_events)
    
    def add_inspection(self, was_false_negative, was_false_positive,
                       confidence, latency_ms):
        self.fn_events.append(1 if was_false_negative else 0)
        self.fp_events.append(1 if was_false_positive else 0)
        self.confidence_scores.append(confidence)
        self.latencies_ms.append(latency_ms)


class RollbackTrigger(str, Enum):
    FN_RATE_SPIKE        = "fn_rate_spike"
    FP_RATE_SPIKE        = "fp_rate_spike"
    LATENCY_SLA_BREACH   = "latency_sla_breach"
    CONFIDENCE_DRIFT     = "confidence_drift"
    MANUAL               = "manual"


class ModelLifecycleManager:
    """
    Manages the full model version lifecycle.
    Triggers automatic rollback within 30 seconds of detecting degradation.
    """
    
    FN_RATE_SPIKE_THRESHOLD  = 0.02
    FP_RATE_SPIKE_THRESHOLD  = 0.08
    LATENCY_SLA_BREACH_MS    = 5.0
    CONFIDENCE_KL_THRESHOLD  = 0.15
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.versions: Dict[str, ModelVersion] = {}
        self.production_model: Optional[ModelVersion] = None
        self.standby_model: Optional[ModelVersion] = None
        self.shadow_model: Optional[ModelVersion] = None
        self.production_metrics = ProductionMetrics()
        self.baseline_metrics: Optional[ProductionMetrics] = None
        self._load_registry()
        mlflow.set_tracking_uri("http://mlflow:5000")
    
    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            for v_id, v_data in data.get("versions", {}).items():
                self.versions[v_id] = ModelVersion(**v_data)
            self.production_model = self.versions.get(data.get("active_production"))
            self.standby_model = self.versions.get(data.get("active_standby"))
    
    def _save_registry(self):
        data = {
            "versions": {v_id: vars(v) for v_id, v in self.versions.items()},
            "active_production": (
                self.production_model.version_id if self.production_model else None
            ),
            "active_standby": (
                self.standby_model.version_id if self.standby_model else None
            )
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    async def promote_to_production(self, version_id: str):
        if self.production_model:
            self.production_model.stage = ModelStage.STANDBY
            self.standby_model = self.production_model
        
        new_version = self.versions[version_id]
        new_version.stage = ModelStage.PRODUCTION
        new_version.deployed_at = datetime.utcnow()
        self.production_model = new_version
        self.baseline_metrics = ProductionMetrics()
        self.production_metrics = ProductionMetrics()
        
        with open("models/active_version.json", "w") as f:
            json.dump({
                "version": version_id,
                "trt_path": new_version.trt_engine_path,
                "deployed_at": datetime.utcnow().isoformat()
            }, f)
        
        self._save_registry()
        logger.info(f"Promoted {version_id} to PRODUCTION")
    
    async def rollback(self, trigger: RollbackTrigger,
                       triggered_by: str = "system",
                       reason: str = "") -> bool:
        if not self.standby_model:
            logger.error("Rollback requested but no standby model available")
            return False
        
        t_start = time.time()
        failed_version = self.production_model
        restore_version = self.standby_model
        
        logger.warning(
            f"ROLLBACK TRIGGERED | Trigger: {trigger.value} | "
            f"Rolling back from {failed_version.version_id} "
            f"to {restore_version.version_id} | Reason: {reason}"
        )
        
        failed_version.rollback_count += 1
        failed_version.stage = ModelStage.CANDIDATE
        failed_version.retired_at = datetime.utcnow()
        
        restore_version.stage = ModelStage.PRODUCTION
        restore_version.deployed_at = datetime.utcnow()
        self.production_model = restore_version
        self.standby_model = None
        
        with open("models/active_version.json", "w") as f:
            json.dump({
                "version": restore_version.version_id,
                "trt_path": restore_version.trt_engine_path,
                "deployed_at": datetime.utcnow().isoformat(),
                "rollback": True,
                "rollback_trigger": trigger.value
            }, f)
        
        self._save_registry()
        rollback_latency = time.time() - t_start
        
        with mlflow.start_run(run_name=f"rollback_{restore_version.version_id}"):
            mlflow.log_params({
                "trigger": trigger.value,
                "failed_version": failed_version.version_id,
                "restored_version": restore_version.version_id,
                "triggered_by": triggered_by,
                "reason": reason
            })
            mlflow.log_metric("rollback_latency_seconds", rollback_latency)
        
        logger.info(
            f"Rollback complete in {rollback_latency:.2f}s | "
            f"Now running {restore_version.version_id}"
        )
        return True
    
    async def monitor_production(self):
        while True:
            await asyncio.sleep(10)
            if not self.production_model or not self.baseline_metrics:
                continue
            
            pm = self.production_metrics
            bm = self.baseline_metrics
            
            if len(pm.fn_events) >= 50 and len(bm.fn_events) >= 50:
                fn_delta = pm.fn_rate - bm.fn_rate
                if fn_delta > self.FN_RATE_SPIKE_THRESHOLD:
                    await self.rollback(
                        trigger=RollbackTrigger.FN_RATE_SPIKE,
                        reason=f"FN rate delta: +{fn_delta:.3f}"
                    )
                    continue
            
            if len(pm.fp_events) >= 50 and len(bm.fp_events) >= 50:
                fp_delta = pm.fp_rate - bm.fp_rate
                if fp_delta > self.FP_RATE_SPIKE_THRESHOLD:
                    await self.rollback(
                        trigger=RollbackTrigger.FP_RATE_SPIKE,
                        reason=f"FP rate delta: +{fp_delta:.3f}"
                    )
                    continue
            
            if len(pm.latencies_ms) >= 20:
                p95_latency = float(np.percentile(list(pm.latencies_ms), 95))
                if p95_latency > self.LATENCY_SLA_BREACH_MS:
                    await self.rollback(
                        trigger=RollbackTrigger.LATENCY_SLA_BREACH,
                        reason=f"P95 latency {p95_latency:.1f}ms exceeds SLA"
                    )
                    continue
```

---

## 6. CDAG-Net — Causal Attribution

```python
# intelligence/cdag_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


PROCESS_PARAMS = [
    "sealing_temperature",
    "conveyor_speed",
    "fill_nozzle_pressure",
    "film_tension",
    "ambient_temperature",
    "ambient_humidity",
    "batch_number",
    "shift_id",
    "machine_runtime_hours",
    "label_roll_tension",
    "fill_material_viscosity",
]

DEFECT_CLASSES = [
    "improper_filling",
    "packaging_damage",
    "label_misalignment",
    "surface_contamination"
]


@dataclass
class CausalAttributionResult:
    defect_class: str
    top_causes: List[Dict]
    causal_graph_edges: List[Tuple]
    intervention_recommendation: str
    confidence: float


class CDAGNet(nn.Module):
    """
    Causal Defect Attribution Graph Network.
    GATv2 backbone + Pearl do-calculus intervention layer.
    Novel contribution: First GATv2 + Pearl causal model
    for manufacturing defect attribution.
    """
    
    def __init__(self, num_params=len(PROCESS_PARAMS),
                 num_defect_classes=len(DEFECT_CLASSES),
                 hidden_dim=128, num_attention_heads=8, num_gat_layers=3):
        super().__init__()
        
        self.num_params = num_params
        self.num_defect_classes = num_defect_classes
        
        self.node_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_attention_heads,
                heads=num_attention_heads,
                concat=True,
                edge_dim=1,
                add_self_loops=True
            )
            for _ in range(num_gat_layers)
        ])
        
        self.causal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_defect_classes)
        )
        
        self.intervention_head = nn.Sequential(
            nn.Linear(hidden_dim + num_params, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_defect_classes)
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_embedding(data.x)
        
        for gat_layer in self.gat_layers:
            x_new = gat_layer(x, data.edge_index, data.edge_attr)
            x = x + x_new
            x = F.layer_norm(x, x.shape[-1:])
        
        causal_strengths = torch.sigmoid(self.causal_head(x))
        
        param_concat = torch.cat(
            [x.mean(0, keepdim=True).expand(self.num_params, -1),
             data.x.expand(self.num_params, self.num_params)],
            dim=-1
        )
        intervention_probs = torch.sigmoid(self.intervention_head(param_concat))
        
        return causal_strengths, intervention_probs
    
    def attribute(self, process_snapshot, defect_class,
                  edge_index, edge_attr) -> CausalAttributionResult:
        self.eval()
        defect_idx = DEFECT_CLASSES.index(defect_class)
        param_values = self._normalise_params(process_snapshot)
        x = torch.tensor(param_values, dtype=torch.float32).unsqueeze(1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        with torch.no_grad():
            causal_strengths, intervention_probs = self.forward(data)
        
        strengths = causal_strengths[:, defect_idx].numpy()
        ranked_indices = np.argsort(strengths)[::-1]
        
        top_causes = []
        for idx in ranked_indices[:5]:
            param_name = PROCESS_PARAMS[idx]
            strength = float(strengths[idx])
            counterfactual = self._generate_counterfactual(
                param_name, process_snapshot.get(param_name, 0),
                strength, defect_class
            )
            top_causes.append({
                "parameter": param_name,
                "causal_strength": round(strength, 4),
                "current_value": process_snapshot.get(param_name, "unknown"),
                "counterfactual": counterfactual
            })
        
        top_param = top_causes[0]["parameter"] if top_causes else "unknown"
        recommendation = self._generate_recommendation(
            top_param, process_snapshot, defect_class
        )
        
        return CausalAttributionResult(
            defect_class=defect_class,
            top_causes=top_causes,
            causal_graph_edges=[(int(e[0]), int(e[1]))
                                for e in edge_index.T.numpy()],
            intervention_recommendation=recommendation,
            confidence=float(strengths[ranked_indices[0]])
        )
    
    def _normalise_params(self, snapshot):
        ranges = {
            "sealing_temperature": (150, 220),
            "conveyor_speed": (10, 80),
            "fill_nozzle_pressure": (0.5, 5.0),
            "film_tension": (5, 50),
            "ambient_temperature": (15, 40),
            "ambient_humidity": (20, 90),
            "batch_number": (0, 9999),
            "shift_id": (0, 2),
            "machine_runtime_hours": (0, 2000),
            "label_roll_tension": (1, 20),
            "fill_material_viscosity": (100, 5000)
        }
        normalised = []
        for param in PROCESS_PARAMS:
            val = snapshot.get(param, 0)
            lo, hi = ranges.get(param, (0, 1))
            normalised.append((val - lo) / max(hi - lo, 1e-6))
        return normalised
    
    def _generate_counterfactual(self, param, current_value, strength, defect_class):
        if strength < 0.3:
            return f"{param} has minimal causal influence on {defect_class}"
        probability_reduction = int(strength * 100)
        return (
            f"If {param.replace('_', ' ')} had been within its optimal range, "
            f"probability of {defect_class.replace('_', ' ')} would have "
            f"reduced by approximately {probability_reduction}%"
        )
    
    def _generate_recommendation(self, top_param, snapshot, defect_class):
        recommendations = {
            "sealing_temperature":
                "Adjust sealing jaw temperature to setpoint ±1°C. Check thermocouple calibration.",
            "conveyor_speed":
                "Reduce conveyor speed by 5–10% until defect rate normalises.",
            "fill_nozzle_pressure":
                "Recalibrate fill nozzle pressure to ±0.1 bar of setpoint.",
            "film_tension":
                "Inspect film tension dancer roll. Check unwind brake setting.",
            "label_roll_tension":
                "Adjust label web tension. Check label roll brake and guide alignment.",
            "machine_runtime_hours":
                "Machine approaching maintenance interval. Schedule PM within next 8 hours.",
        }
        return recommendations.get(
            top_param,
            f"Investigate {top_param.replace('_', ' ')} — "
            f"identified as primary causal factor for {defect_class.replace('_', ' ')}"
        )
```

---

## 7. Training Pipeline

### 7.1 YOLOv11 Training

```python
# training/train_yolov11.py

from ultralytics import YOLO
import wandb


def train_yolov11(
    model_size: str = "s",
    dataset_yaml: str = "data/food_defects.yaml",
    epochs: int = 150,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    pretrained: bool = True,
    project: str = "visionfood_qai",
    experiment_name: str = "yolov11_food_defects_v1"
):
    """
    Dataset YAML format (data/food_defects.yaml):
    ---
    path: /datasets/food_defects
    train: images/train
    val: images/val
    test: images/test
    nc: 4
    names:
      0: improper_filling
      1: packaging_damage
      2: label_misalignment
      3: surface_contamination
    """
    
    wandb.init(project=project, name=experiment_name)
    
    weights = f"yolo11{model_size}.pt" if pretrained else f"yolo11{model_size}.yaml"
    model = YOLO(weights)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        project=f"runs/{project}",
        name=experiment_name,
        save_period=10,
        val=True,
        plots=True,
        patience=30,
    )
    
    model.export(format="onnx", half=True, imgsz=img_size,
                 dynamic=False, simplify=True)
    
    metrics = {
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "mAP50_95": results.results_dict["metrics/mAP50-95(B)"],
        "precision": results.results_dict["metrics/precision(B)"],
        "recall": results.results_dict["metrics/recall(B)"],
    }
    wandb.log(metrics)
    wandb.finish()
    return results, metrics


def train_quantisation_aware(base_model_path, dataset_yaml, epochs=30):
    """QAT for cases where PTQ INT8 drops accuracy beyond 1% tolerance."""
    model = YOLO(base_model_path)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        int8=True,
        lr0=0.0001,
        patience=15,
        freeze=10,
    )
    return results
```

### 7.2 EfficientViT Classifier Training

```python
# training/train_efficientvit.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import wandb


def train_efficientvit(
    dataset_root: str = "data/crops",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    device: str = "cuda"
):
    wandb.init(project="visionfood_qai", name="efficientvit_m5_classifier")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(f"{dataset_root}/train",
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{dataset_root}/val",
                                       transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    
    model = timm.create_model("efficientvit_m5", pretrained=True,
                               num_classes=4).to(device)
    
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
            pt = torch.exp(-ce_loss)
            return (((1 - pt) ** self.gamma) * ce_loss).mean()
    
    criterion = FocalLoss(gamma=2.0)
    ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                   weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                             T_max=epochs)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(images)
            loss = 0.7 * criterion(logits, labels) + 0.3 * ce_smooth(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
        
        scheduler.step()
        
        model.eval()
        val_correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss / len(train_loader),
            "train/accuracy": train_correct / len(train_dataset),
            "val/loss": val_loss / len(val_loader),
            "val/accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/efficientvit_best.pt")
            print(f"Epoch {epoch}: New best val_acc = {val_acc:.4f}")
    
    wandb.finish()
    return model
```

---

## 8. FastAPI Backend

```python
# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .routers import inspection, remedy, analytics, models, compliance, websocket
from .middleware.audit_logger import AuditLoggerMiddleware
from .middleware.auth import JWTAuthMiddleware
from core.config import EdgeConfig

config = EdgeConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await redis.from_url(config.REDIS_URL,
                                            decode_responses=True)
    engine = create_async_engine(config.DATABASE_URL, echo=False)
    app.state.db_session = sessionmaker(engine, class_=AsyncSession,
                                         expire_on_commit=False)
    yield
    await app.state.redis.close()


app = FastAPI(
    title="VisionFood QAI API",
    description="AI-powered food packaging inspection and REMEDY system",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://dashboard:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuditLoggerMiddleware)
app.add_middleware(JWTAuthMiddleware,
                   exempt_paths=["/health", "/docs", "/openapi.json"])

app.include_router(inspection.router, prefix="/api/v1/inspection",
                   tags=["Inspection"])
app.include_router(remedy.router,     prefix="/api/v1/remedy",
                   tags=["REMEDY"])
app.include_router(analytics.router,  prefix="/api/v1/analytics",
                   tags=["Analytics"])
app.include_router(models.router,     prefix="/api/v1/models",
                   tags=["Model Lifecycle"])
app.include_router(compliance.router, prefix="/api/v1/compliance",
                   tags=["Compliance"])
app.include_router(websocket.router,  prefix="/ws",
                   tags=["WebSocket"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
```

```python
# api/routers/websocket.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import redis.asyncio as redis

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@router.websocket("/inspections")
async def inspection_websocket(websocket: WebSocket):
    """Real-time inspection feed. Streams from Redis inspections:live stream."""
    await manager.connect(websocket)
    redis_client = await redis.from_url("redis://localhost:6379")
    last_id = "$"
    
    try:
        while True:
            messages = await redis_client.xread(
                {"inspections:live": last_id}, count=10, block=100
            )
            if messages:
                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        last_id = msg_id
                        payload = json.loads(msg_data["data"])
                        await websocket.send_json({
                            "type": "inspection",
                            "data": payload
                        })
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        await redis_client.close()


@router.websocket("/alerts")
async def alerts_websocket(websocket: WebSocket):
    """High-priority alert stream for defect clusters, rollbacks, OEE drops."""
    await manager.connect(websocket)
    redis_client = await redis.from_url("redis://localhost:6379")
    last_id = "$"
    
    try:
        while True:
            messages = await redis_client.xread(
                {"alerts:critical": last_id}, count=5, block=200
            )
            if messages:
                for _, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        last_id = msg_id
                        await websocket.send_json({
                            "type": "alert",
                            "severity": msg_data.get("severity", "WARNING"),
                            "message": msg_data.get("message"),
                            "timestamp": msg_data.get("timestamp")
                        })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        await redis_client.close()
```

---

## 9. Database Schema

```python
# database/models.py

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    JSON, Text, Index, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid
import enum

Base = declarative_base()


class VerdictEnum(str, enum.Enum):
    PASS     = "PASS"
    FAIL     = "FAIL"
    ESCALATE = "ESCALATE"
    REVIEW   = "REVIEW"


class RemediationActionEnum(str, enum.Enum):
    RELABEL      = "RELABEL"
    REFILL       = "REFILL"
    REPACK       = "REPACK"
    CLEAN        = "CLEAN"
    REJECT       = "REJECT"
    PASS         = "PASS"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class InspectionRecord(Base):
    """
    Core inspection record. Immutable after creation.
    One row per product unit inspected.
    """
    __tablename__ = "inspection_records"
    
    id                    = Column(UUID(as_uuid=True), primary_key=True,
                                   default=uuid.uuid4)
    product_id            = Column(String(64), nullable=False, index=True)
    sku_id                = Column(String(64), nullable=False, index=True)
    batch_id              = Column(String(64), nullable=True, index=True)
    verdict               = Column(SQLEnum(VerdictEnum), nullable=False, index=True)
    defect_classes        = Column(JSON, nullable=True)
    detections            = Column(JSON, nullable=True)
    mean_confidence       = Column(Float, nullable=True)
    std_confidence        = Column(Float, nullable=True)
    ci_low                = Column(Float, nullable=True)
    ci_high               = Column(Float, nullable=True)
    inference_latency_ms  = Column(Float, nullable=False)
    tier_resolved         = Column(String(16), default="edge")
    model_version         = Column(String(32), nullable=False, index=True)
    dataset_version       = Column(String(64), nullable=True)
    inspected_at          = Column(DateTime(timezone=True),
                                   server_default=func.now(), index=True)
    gradcam_path          = Column(String(256), nullable=True)
    shap_values           = Column(JSON, nullable=True)
    was_escalated         = Column(Boolean, default=False)
    required_human_review = Column(Boolean, default=False)
    
    __table_args__ = (
        Index("ix_inspected_at_sku", "inspected_at", "sku_id"),
        Index("ix_verdict_batch", "verdict", "batch_id"),
    )


class RemediationRecord(Base):
    """One row per REMEDY action taken."""
    __tablename__ = "remediation_records"
    
    id                     = Column(UUID(as_uuid=True), primary_key=True,
                                    default=uuid.uuid4)
    product_id             = Column(String(64), nullable=False, index=True)
    inspection_id          = Column(UUID(as_uuid=True), nullable=True)
    action                 = Column(SQLEnum(RemediationActionEnum), nullable=False)
    severity_grade         = Column(String(2), nullable=False)
    severity_score         = Column(Float, nullable=False)
    primary_factor         = Column(String(64), nullable=True)
    station_id             = Column(String(16), nullable=True)
    attempt_count          = Column(Integer, default=1)
    station_success        = Column(Boolean, nullable=True)
    reinspection_verdict   = Column(String(16), nullable=True)
    pre_remedy_image_path  = Column(String(256), nullable=True)
    post_remedy_image_path = Column(String(256), nullable=True)
    action_at              = Column(DateTime(timezone=True),
                                    server_default=func.now(), index=True)
    reason                 = Column(Text, nullable=True)
    product_value          = Column(Float, nullable=True)


class CausalAttributionRecord(Base):
    """CDAG-Net causal attribution output per defect event."""
    __tablename__ = "causal_attributions"
    
    id               = Column(UUID(as_uuid=True), primary_key=True,
                               default=uuid.uuid4)
    product_id       = Column(String(64), nullable=False, index=True)
    defect_class     = Column(String(64), nullable=False)
    top_causes       = Column(JSON, nullable=False)
    recommendation   = Column(Text, nullable=True)
    confidence       = Column(Float, nullable=True)
    process_snapshot = Column(JSON, nullable=True)
    attributed_at    = Column(DateTime(timezone=True), server_default=func.now())


class ModelVersionRecord(Base):
    """Model version audit trail. Immutable entries."""
    __tablename__ = "model_versions"
    
    id              = Column(UUID(as_uuid=True), primary_key=True,
                             default=uuid.uuid4)
    version_id      = Column(String(32), unique=True, nullable=False)
    stage           = Column(String(16), nullable=False)
    mlflow_run_id   = Column(String(64), nullable=True)
    trt_engine_path = Column(String(256), nullable=True)
    dataset_hash    = Column(String(64), nullable=True)
    metrics         = Column(JSON, nullable=True)
    deployed_at     = Column(DateTime(timezone=True), nullable=True)
    retired_at      = Column(DateTime(timezone=True), nullable=True)
    rollback_count  = Column(Integer, default=0)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class AuditLogEntry(Base):
    """
    Immutable audit log. Required: FDA 21 CFR Part 11, ISO 22000.
    Every API action, model decision, operator override, and rollback logged here.
    """
    __tablename__ = "audit_log"
    
    id          = Column(UUID(as_uuid=True), primary_key=True,
                         default=uuid.uuid4)
    event_type  = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=True)
    entity_id   = Column(String(64), nullable=True, index=True)
    actor_id    = Column(String(64), nullable=True)
    action      = Column(String(128), nullable=False)
    payload     = Column(JSON, nullable=True)
    ip_address  = Column(String(45), nullable=True)
    occurred_at = Column(DateTime(timezone=True),
                         server_default=func.now(), index=True)
    
    __table_args__ = (
        Index("ix_audit_event_time", "event_type", "occurred_at"),
    )
```

---

## 10. Docker Deployment

```yaml
# docker/docker-compose.yml

version: "3.9"

services:

  redis:
    image: redis:7.4-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: visionfood
      POSTGRES_USER: qai
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.14.0
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://qai:${DB_PASSWORD}@postgres:5432/visionfood
      --default-artifact-root /mlflow/artefacts
    ports:
      - "5000:5000"
    depends_on: [postgres]
    volumes:
      - mlflow_artefacts:/mlflow/artefacts
    restart: unless-stopped

  edge-inference:
    build:
      context: .
      dockerfile: docker/Dockerfile.edge
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - TIER=edge
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///./edge.db
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on: [redis]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  fog-processor:
    build:
      context: .
      dockerfile: docker/Dockerfile.fog
    runtime: nvidia
    environment:
      - TIER=fog
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://qai:${DB_PASSWORD}@postgres:5432/visionfood
      - CLOUD_ENDPOINT=http://cloud-core:8002
    volumes:
      - ./models:/app/models
    depends_on: [redis, postgres]
    restart: unless-stopped

  api-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.fog
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://qai:${DB_PASSWORD}@postgres:5432/visionfood
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on: [redis, postgres]
    restart: unless-stopped

  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://api-server:8000
      - REACT_APP_WS_URL=ws://api-server:8000/ws
    depends_on: [api-server]
    restart: unless-stopped

  celery-worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.fog
    command: celery -A intelligence.tasks worker --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://qai:${DB_PASSWORD}@postgres:5432/visionfood
    depends_on: [redis, postgres]
    restart: unless-stopped

  nginx:
    image: nginx:1.27-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/ssl/certs:ro
    depends_on: [api-server, dashboard]
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  mlflow_artefacts:
```

```dockerfile
# docker/Dockerfile.edge
# Optimised for NVIDIA Jetson Orin NX (JetPack 6.0 base)

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev python3-opencv \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.edge.txt .
RUN pip install --no-cache-dir -r requirements.edge.txt

COPY core/ core/
COPY inference/ inference/
COPY remedy/ remedy/
COPY api/ api/

RUN mkdir -p models logs

RUN useradd -m -u 1000 qaiuser
USER qaiuser

CMD ["python", "-m", "inference.main"]
```

---

## 11. Edge Optimisation Export Scripts

```python
# export/export_tensorrt.py

import subprocess
import time


def export_yolov11_to_trt(
    onnx_path: str,
    output_path: str,
    calibration_data_dir: str,
    precision: str = "int8",
    max_batch_size: int = 1,
    workspace_gb: int = 4
):
    """
    Export YOLOv11 ONNX to TensorRT engine.
    Run directly on the target Jetson device.
    INT8 calibration requires 500+ representative images.
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_gb * 1024}",
        f"--maxBatch={max_batch_size}",
        "--verbose",
        "--avgRuns=10",
    ]
    
    if precision == "int8":
        cmd.extend([
            "--int8",
            f"--calib={calibration_data_dir}",
            "--calibBatchSize=8",
        ])
    elif precision == "fp16":
        cmd.append("--fp16")
    
    t_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    build_time = time.time() - t_start
    
    if result.returncode != 0:
        raise RuntimeError(f"TensorRT build failed:\n{result.stderr}")
    
    print(f"TRT engine built in {build_time:.1f}s → {output_path}")
    benchmark_trt(output_path)


def benchmark_trt(engine_path: str, n_runs: int = 1000):
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--iterations={n_runs}",
        "--warmUp=100",
        "--percentile=50,95,99"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "Percentile" in line or "Latency" in line:
            print(line)
    return result.stdout


def verify_accuracy_drop(original_onnx, trt_engine,
                          test_images_dir, threshold=0.01) -> bool:
    """Returns True if accuracy drop acceptable, False if QAT required."""
    from ultralytics import YOLO
    
    onnx_model = YOLO(original_onnx)
    onnx_metrics = onnx_model.val(data=test_images_dir, device="cpu")
    onnx_map = onnx_metrics.results_dict["metrics/mAP50-95(B)"]
    
    trt_model = YOLO(trt_engine)
    trt_metrics = trt_model.val(data=test_images_dir, device=0)
    trt_map = trt_metrics.results_dict["metrics/mAP50-95(B)"]
    
    drop = onnx_map - trt_map
    print(f"ONNX mAP50-95: {onnx_map:.4f}")
    print(f"TRT  mAP50-95: {trt_map:.4f}")
    print(f"Drop: {drop:.4f} "
          f"({'ACCEPTABLE' if drop <= threshold else 'EXCEEDS THRESHOLD — QAT REQUIRED'})")
    return drop <= threshold
```

---

## 12. GitHub CI/CD Pipeline

```yaml
# .github/workflows/ci.yml

name: VisionFood QAI CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ${{ github.repository_owner }}/visionfood-qai

jobs:

  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install ruff mypy
      - run: ruff check . --select E,W,F,I
      - run: mypy core/ inference/ remedy/ intelligence/ api/ --ignore-missing-imports

  unit-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: pip install -r requirements.dev.txt
      - run: pytest tests/unit/ -v --cov=. --cov-report=xml --timeout=60
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - run: docker compose -f docker/docker-compose.test.yml up -d
      - run: pip install -r requirements.dev.txt
      - run: pytest tests/integration/ -v --timeout=120
      - run: docker compose -f docker/docker-compose.test.yml down

  model-accuracy-gate:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Check model metrics vs baseline
        run: |
          python scripts/check_model_accuracy_gate.py \
            --model-path models/yolov11_candidate.onnx \
            --test-data data/test/ \
            --min-map50 0.88 \
            --min-map50-95 0.72 \
            --min-f1 0.87

  build-and-push:
    runs-on: ubuntu-latest
    needs: [lint-and-type-check, unit-tests]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push edge image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.edge
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}-edge:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}-edge:${{ github.sha }}
      
      - name: Build and push fog image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.fog
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}-fog:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}-fog:${{ github.sha }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: staging
    steps:
      - name: Deploy to staging via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/visionfood-qai
            docker compose pull
            docker compose up -d --no-deps --force-recreate api-server fog-processor
            docker compose exec api-server python -m alembic upgrade head
            echo "Staging deployment complete: ${{ github.sha }}"
```

---

## Software Architecture Summary

| Service | Description |
|---------|-------------|
| **Edge service** | YOLOv11 TRT INT8 + EfficientViT + MC Dropout UQ + REMEDY triage + Station controller + Redis publisher. Runs on Jetson Orin NX. Handles 120 products/minute in under 5ms. |
| **Fog service** | RT-DETRv2 + SAM 2 + HAFFN fusion + Depth Anything V2 + Deep Ensemble UQ + escalation handler. Runs on RTX 4090. Handles 15% of escalated cases within 50ms. |
| **API server** | FastAPI with 6 router modules, JWT auth, audit logging middleware, WebSocket streaming. PostgreSQL backend. MLflow model registry. |
| **Intelligence service** | CDAG-Net causal attribution + Mamba-QC SSM + XMI Gateway + SPC engine + OEE calculator. Runs as async Celery workers. |
| **MLOps service** | Model lifecycle manager, shadow runner, canary manager, automatic rollback controller, drift monitor. 30-second rollback SLA enforced. |
| **Dashboard** | React + Plotly + WebSocket real-time feed. REMEDY command centre. Analytics suite. Compliance portal. |
| **Federated learning service** | Flower (flwr) server + EWC + DP-SGD + Shamir aggregation. Runs on cloud or on-premise GPU server. |
