# VisionFood QAI — Inspection Pipeline Optimization & Model Training Plan

**Date:** April 13, 2026  
**Focus:** pyzbar label verification integration + model training for production-grade inference  
**Preprocessing → Training → Validation → Deployment**

---

## 0. Pyzbar Clarification

**Does pyzbar require training?**

No. `pyzbar` is a **pre-built barcode/QR decoder library** — it's a C wrapper around the ZBar library. It uses hardcoded algorithms (not neural networks) to:
1. Detect QR/barcode patterns in an image
2. Decode the bit patterns to extract the string value

It's equivalent to a standard library — you just call `decode(image)` and get back strings. No ML, no weights, no training. It's like `json.loads()` but for barcodes.

---

## 1. Revised Inspection Pipeline Architecture

### 1.1 Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Raw Frame from Camera (BGR)                       │
│                         resolution: 1920x1080                        │
│                         FPS: 2 (software trigger)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼ (PARALLEL - 80ms)                   ▼ (PARALLEL - 3ms)
    ┌──────────────────────┐            ┌──────────────────────┐
    │   YOLOv11 Detection  │            │  pyzbar QR Decode    │
    │   - Load ONNX model  │            │  - Locate QR pattern │
    │   - Infer objects    │            │  - Decode bit matrix │
    │   - Filter by conf   │            │  - Return string(s)  │
    │   - Return bboxes    │            │  - Compare with DB   │
    └──────────┬───────────┘            └──────────┬───────────┘
               │                                   │
               │ detections[]             qr_status: LabelQRStatus
               │                                   │
               └───────────────┬───────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │ Decision: Defects Found?   │
                 └─────────────┬──────────────┘
                        Yes   │   No
                             ▼
          ┌──────────────────────────────────────┐
          │ Extract Crop (primary detection)      │
          │ bbox.x1, y1, x2, y2 → crop region    │
          └──────────────┬───────────────────────┘
                         ▼
          ┌──────────────────────────────────────┐
          │ EfficientViT MC Dropout UQ (~60ms)    │
          │ - Load ONNX model (dropout enabled)  │
          │ - Forward pass 20x with perturbation │
          │ - Compute mean_conf, std_conf        │
          │ - Return UQResult                    │
          └──────────────┬───────────────────────┘
                         │
                    uq_result
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ┌───────────────────┐       ┌──────────────────┐
    │ Apply Verdict     │       │ (if no UQ,       │
    │ Logic             │       │  skip this       │
    │ - YOLOv11 conf    │       │  segment)        │
    │ - UQ uncertainty  │       │                  │
    │ - QR mismatch     │       │                  │
    │ - Return Verdict  │       │                  │
    └────────┬──────────┘       └──────────────────┘
             │
    ┌────────▼─────────────────┐
    │ Verdict Decision Tree     │
    │                           │
    │ NO DEFECTS:               │
    │   QR mismatch? → FAIL     │
    │   else → PASS             │
    │                           │
    │ DEFECTS EXIST:            │
    │   conf ≥ 0.85 AND         │
    │   NOT uncertain?          │
    │     → FAIL (confirmed)    │
    │   conf ≥ ESCALATE_THR?    │
    │     → FAIL + escalated    │
    │   conf ≥ REVIEW_THR?      │
    │     → ESCALATE            │
    │   else → REVIEW           │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │ REMEDY Engine (if FAIL/ESCALATE)  │
    │ - SeverityScorer                  │
    │ - TriageRouter                    │
    │ - Station assignment              │
    └────────┬───────────────────────────┘
             │
    ┌────────▼──────────────────────────────┐
    │ InspectionResult Output               │
    │ {                                     │
    │   inspection_id, verdict, detections,│
    │   uq_result, severity, remediation,  │
    │   label_qr, latency_ms, device_id   │
    │ }                                     │
    └───────────────────────────────────────┘
                      │
                      ▼
            Write to MongoDB
            Publish to Redis Stream
            Fan out via WebSocket
```

### 1.2 Latency Budget Breakdown

| Stage | Model | Time | Budget | Notes |
|---|---|---|---|---|
| YOLOv11 | Detection | 70–90ms | 80ms | ONNX quantized |
| pyzbar | QR decode | 2–5ms | 10ms | C library, no ML |
| EfficientViT UQ | Classification | 50–70ms | 60ms | 20x forward passes |
| Verdict logic | N/A | <5ms | 5ms | Python decision tree |
| REMEDY | Severity + Triage | 10–20ms | 20ms | Scoring + routing |
| **Total** | | | **~200ms** | Comfortably below 500ms industry standard |

---

## 2. Stage 1: Integrate pyzbar into Pipeline (Week 1)

### 2.1 Changes to `core/schemas.py`

Add `LabelQRStatus` and update `InspectionResult`:

```python
# core/schemas.py additions

class LabelQRStatus(BaseModel):
    """Label and QR verification result."""
    qr_detected: bool
    qr_decoded: Optional[str] = None         # what was actually scanned
    qr_expected: Optional[str] = None        # what should be there for this SKU
    qr_matched: Optional[bool] = None        # does decoded == expected?
    label_anomaly_types: list[str] = Field(default_factory=list)

# Update InspectionResult:
class InspectionResult(BaseModel):
    ...existing fields...
    label_qr: Optional[LabelQRStatus] = None  # ← add this
```

### 2.2 Changes to `inference/pipeline.py`

Add pyzbar integration to the `inspect()` method:

```python
# inference/pipeline.py

def inspect(
    self,
    frame: np.ndarray,
    product_id: Optional[str] = None,
    sku: str = "default",
    attempt_count: int = 0,
) -> InspectionResult:
    t0 = time.perf_counter()
    inspection_id = str(uuid.uuid4())

    # PARALLEL: Detection + QR decode
    detections: list[Detection] = self._run_detection(frame)
    label_qr: LabelQRStatus = self._check_label_qr(frame, sku)

    # UQ only if defects found
    uq_result: Optional[UQResult] = None
    if detections:
        uq_result = self._run_uq(frame, detections)

    # Verdict logic: incorporate QR mismatch
    verdict, escalated = self._apply_verdict_logic(
        detections, uq_result, label_qr
    )

    # REMEDY if needed
    severity_result = None
    remediation_action = None
    if verdict in (Verdict.FAIL, Verdict.ESCALATE) and self.config.REMEDY_ENABLED:
        severity_result, remediation_action = self._run_remedy(
            detections, uq_result, sku, attempt_count
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = InspectionResult(
        inspection_id=inspection_id,
        product_id=product_id,
        sku=sku,
        timestamp=datetime.utcnow(),
        verdict=verdict,
        escalated=escalated,
        detections=detections,
        uq_result=uq_result,
        severity_result=severity_result,
        remediation_action=remediation_action,
        label_qr=label_qr,  # ← include QR result
        latency_ms=round(latency_ms, 2),
        device_id=self.config.DEVICE_ID,
    )
    return result

def _check_label_qr(self, frame: np.ndarray, sku: str) -> LabelQRStatus:
    """Decode QR from frame and verify against expected value from DB."""
    from pyzbar.pyzbar import decode as pyzbar_decode
    from PIL import Image
    import asyncio

    # Convert BGR to RGB for PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    try:
        barcodes = pyzbar_decode(pil_img)
    except Exception as exc:
        log.warning("qr_decode_error", error=str(exc))
        return LabelQRStatus(qr_detected=False)

    if not barcodes:
        return LabelQRStatus(
            qr_detected=False,
            qr_expected=self._get_expected_qr_sync(sku),
            label_anomaly_types=["qr_not_found"],
        )

    # Take first barcode
    decoded_value = barcodes[0].data.decode("utf-8")
    expected = self._get_expected_qr_sync(sku)
    
    return LabelQRStatus(
        qr_detected=True,
        qr_decoded=decoded_value,
        qr_expected=expected,
        qr_matched=(decoded_value == expected) if expected else None,
        label_anomaly_types=[] if (decoded_value == expected) else ["qr_mismatch"],
    )

def _get_expected_qr_sync(self, sku: str) -> Optional[str]:
    """
    Sync lookup: query MongoDB from sync context.
    Called from _check_label_qr which runs in the main thread.
    
    In production, this should use a cached dict in memory or Redis
    to avoid a DB call per frame (~10ms latency cost).
    """
    from database.session import get_db_sync  # implement this
    try:
        db = get_db_sync()
        product = db.products.find_one({"sku": sku}, {"label_info.qr_code": 1})
        if product and product.get("label_info", {}).get("qr_code"):
            return product["label_info"]["qr_code"]
    except Exception as exc:
        log.warning("qr_lookup_error", sku=sku, error=str(exc))
    return None

def _apply_verdict_logic(
    self,
    detections: list[Detection],
    uq: Optional[UQResult],
    label_qr: LabelQRStatus,  # ← new parameter
) -> tuple[Verdict, bool]:
    """
    Enhanced verdict logic: incorporate QR verification.
    
    Rules:
      1. If QR mismatch (wrong product) → FAIL (top priority)
      2. If no defects detected:
         - QR not found (label missing) → ESCALATE
         - else → PASS
      3. If defects detected: apply existing logic
    """
    # Rule 1: QR mismatch takes priority
    if label_qr.qr_detected and label_qr.qr_matched is False:
        log.warning(
            "qr_mismatch",
            decoded=label_qr.qr_decoded,
            expected=label_qr.qr_expected,
        )
        return Verdict.FAIL, True  # escalated=True for operator review

    # Rule 2: If no visual defects
    if not detections:
        if label_qr.qr_detected is False:
            return Verdict.ESCALATE, False  # Label missing/obscured
        return Verdict.PASS, False

    # Rule 3: Defects exist — apply existing logic
    mean_conf = (
        uq.mean_confidence if uq else max(d.confidence for d in detections)
    )
    is_uncertain = uq.is_uncertain if uq else False

    if mean_conf >= self.config.CONFIRMED_DEFECT_THRESHOLD and not is_uncertain:
        return Verdict.FAIL, False
    elif mean_conf >= self.config.ESCALATE_THRESHOLD:
        return Verdict.FAIL, True
    elif mean_conf >= self.config.HUMAN_REVIEW_THRESHOLD:
        return Verdict.ESCALATE, False
    else:
        return Verdict.REVIEW, False
```

### 2.3 Update `core/config.py`

Add QR verification settings:

```python
# core/config.py additions

class EdgeConfig(BaseSettings):
    ...existing fields...
    
    # QR verification
    QR_VERIFICATION_ENABLED: bool = True
    QR_MISMATCH_ESCALATES: bool = True  # if False, log only
    QR_MISSING_ESCALATES: bool = True
    QR_CACHE_TTL_SEC: int = 300  # cache product SKU→QR mapping for 5min
```

### 2.4 Dependencies

```
# requirements.txt additions
pyzbar==0.1.9
pillow==10.0.0  # for PIL Image conversion
```

### 2.5 Unit Tests

```python
# tests/unit/test_qr_verification.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from inference.pipeline import EdgeInferencePipeline
from core.schemas import LabelQRStatus, Verdict

@pytest.fixture
def pipeline():
    return EdgeInferencePipeline()

def test_qr_decode_success(pipeline):
    """QR found and matches expected."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with patch('inference.pipeline.pyzbar_decode') as mock_decode:
        with patch.object(pipeline, '_get_expected_qr_sync', return_value='SKU-123'):
            mock_barcode = MagicMock()
            mock_barcode.data = b'SKU-123'
            mock_decode.return_value = [mock_barcode]
            
            result = pipeline._check_label_qr(frame, sku='bottle_250ml')
            
            assert result.qr_detected is True
            assert result.qr_decoded == 'SKU-123'
            assert result.qr_matched is True
            assert result.label_anomaly_types == []

def test_qr_decode_mismatch(pipeline):
    """QR found but doesn't match expected."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with patch('inference.pipeline.pyzbar_decode') as mock_decode:
        with patch.object(pipeline, '_get_expected_qr_sync', return_value='SKU-123'):
            mock_barcode = MagicMock()
            mock_barcode.data = b'SKU-456'
            mock_decode.return_value = [mock_barcode]
            
            result = pipeline._check_label_qr(frame, sku='bottle_250ml')
            
            assert result.qr_detected is True
            assert result.qr_decoded == 'SKU-456'
            assert result.qr_matched is False
            assert 'qr_mismatch' in result.label_anomaly_types

def test_qr_not_found(pipeline):
    """QR not detected in frame."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with patch('inference.pipeline.pyzbar_decode', return_value=[]):
        result = pipeline._check_label_qr(frame, sku='bottle_250ml')
        
        assert result.qr_detected is False
        assert result.qr_decoded is None
        assert 'qr_not_found' in result.label_anomaly_types

def test_verdict_qr_mismatch_prioritized(pipeline):
    """QR mismatch causes FAIL even with no visual defects."""
    detections = []
    uq_result = None
    label_qr = LabelQRStatus(
        qr_detected=True,
        qr_decoded='SKU-999',
        qr_expected='SKU-123',
        qr_matched=False,
    )
    
    verdict, escalated = pipeline._apply_verdict_logic(detections, uq_result, label_qr)
    
    assert verdict == Verdict.FAIL
    assert escalated is True

def test_verdict_no_defects_no_qr(pipeline):
    """No visual defects, no QR detected → ESCALATE."""
    detections = []
    uq_result = None
    label_qr = LabelQRStatus(qr_detected=False)
    
    verdict, escalated = pipeline._apply_verdict_logic(detections, uq_result, label_qr)
    
    assert verdict == Verdict.ESCALATE
    assert escalated is False

def test_verdict_no_defects_qr_match(pipeline):
    """No visual defects, QR found and matched → PASS."""
    detections = []
    uq_result = None
    label_qr = LabelQRStatus(
        qr_detected=True,
        qr_decoded='SKU-123',
        qr_expected='SKU-123',
        qr_matched=True,
    )
    
    verdict, escalated = pipeline._apply_verdict_logic(detections, uq_result, label_qr)
    
    assert verdict == Verdict.PASS
    assert escalated is False
```

### 2.6 Integration Test

```python
# tests/integration/test_pipeline_with_qr.py

import pytest
import cv2
import numpy as np
from inference.pipeline import EdgeInferencePipeline

@pytest.fixture
def pipeline():
    p = EdgeInferencePipeline()
    p.load_models()
    return p

def test_pipeline_full_flow_with_qr_mock(pipeline, tmp_path):
    """End-to-end: frame with defect + matching QR → result."""
    # Create a dummy frame (in real test, load actual labeled test image)
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    result = pipeline.inspect(frame, sku='bottle_250ml', product_id='P001')
    
    # Assertions
    assert result.inspection_id is not None
    assert result.verdict in ['PASS', 'FAIL', 'ESCALATE', 'REVIEW']
    assert result.label_qr is not None
    assert result.latency_ms < 300  # real-time budget
```

---

## 3. Stage 2: Dataset Preparation & Optimization (Week 2—3)

### 3.1 Current Dataset Status

Your workspace includes:
- `DATASETS/mvtec_anomaly_detection/` — 15 object categories (bottle, cable, capsule, etc.)
- `DATASETS/can/` — canned product images
- `DATASETS/juice_bottles/` — beverage samples
- `DATASETS/d2s_amodal_images/` and `DATASETS/d2s_images/` — segmentation datasets

### 3.2 Optimization Strategy: Data Preprocessing Pipeline

```
Raw images
    │
    ├──► Class Balancing
    │    - Count defects per class
    │    - Oversample minority classes
    │    - Undersample majority classes
    │
    ├──► Augmentation (on-the-fly during training)
    │    - Rotation: ±15°
    │    - Brightness: ±15%
    │    - Contrast: ±15%
    │    - Gaussian blur: σ ∈ [0.5, 1.5]
    │    - JPEG compression: quality 70–95
    │
    ├──► Normalization
    │    - ImageNet stats or dataset-specific
    │    - If custom: compute mean/std on train set
    │
    ├──► Resolution Standardization
    │    - YOLOv11: 640×640 (default)
    │    - EfficientViT: 384×384 (from paper)
    │
    └──► Train/Val/Test Split
         - 70% train / 15% val / 15% test
         - Stratified by defect class + product category
```

### 3.3 Prepare Datasets Script

Create `scripts/prepare_datasets.py`:

```python
"""
scripts/prepare_datasets.py

Prepare datasets for YOLOv11 and EfficientViT training.
Combines MVTec, can, juice_bottle datasets into a unified format.
"""

import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
from tqdm import tqdm

def prepare_poduct_defect_dataset(
    src_dirs: list[str],
    output_dir: str = "data/prepared",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Combine multiple source directories into a unified YOLO format.
    
    Input: src_dirs = [
        'DATASETS/mvtec_anomaly_detection/bottle',
        'DATASETS/can/can',
        'DATASETS/juice_bottles/juice_bottle',
    ]
    
    Output structure (YOLO format):
    data/prepared/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirs
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images and their class labels
    all_images = []
    class_names = set()
    
    for src_dir in src_dirs:
        product_name = Path(src_dir).name
        good_dir = Path(src_dir) / 'good'
        anomaly_dirs = [d for d in Path(src_dir).iterdir() if d.is_dir() and d.name != 'good']
        
        # Good images (no defect, label=0)
        if good_dir.exists():
            for img_file in (good_dir / 'test').glob('*.png'):
                all_images.append({
                    'path': img_file,
                    'class': 'good',
                    'class_id': 0,
                    'product': product_name,
                })
        
        # Defective images
        for anomaly_type_dir in anomaly_dirs:
            anomaly_name = anomaly_type_dir.name
            for split_dir in anomaly_type_dir.iterdir():
                if split_dir.is_dir():
                    class_id = len(class_names)
                    class_names.add(anomaly_name)
                    
                    for img_file in split_dir.glob('*.png'):
                        all_images.append({
                            'path': img_file,
                            'class': anomaly_name,
                            'class_id': class_id,
                            'product': product_name,
                        })
    
    # Stratified split
    train_data, temp_data = train_test_split(
        all_images,
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=[d['class'] for d in all_images],
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=[d['class'] for d in temp_data],
    )
    
    # Copy and augment training images
    transform_train = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussBlur(blur_limit=(3, 3), sigma_limit=(0.5, 1.5), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
    ])
    
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        for item in tqdm(split_data, desc=f"Processing {split_name}"):
            src_img = cv2.imread(str(item['path']))
            if src_img is None:
                continue
            
            # Augment training set only
            if split_name == 'train':
                src_img = transform_train(image=src_img)['image']
            
            dst_img_path = output_path / 'images' / split_name / item['path'].name
            cv2.imwrite(str(dst_img_path), src_img)
            
            # Create YOLO label file (for detection: bounding box)
            # For classification-only: just store class_id in a text file
            label_file = output_path / 'labels' / split_name / item['path'].stem + '.txt'
            with open(label_file, 'w') as f:
                f.write(str(item['class_id']))
    
    # Write dataset.yaml
    dataset_yaml = f"""
path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {sorted(list(class_names))}
"""
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    print(f"✅ Dataset prepared at {output_path}")
    print(f"   Train: {len(train_data)} images")
    print(f"   Val: {len(val_data)} images")
    print(f"   Test: {len(test_data)} images")
    print(f"   Classes: {sorted(list(class_names))}")

if __name__ == '__main__':
    prepare_poduct_defect_dataset(
        src_dirs=[
            'DATASETS/mvtec_anomaly_detection/bottle',
            'DATASETS/can/can',
            'DATASETS/juice_bottles/juice_bottle',
        ]
    )
```

### 3.4 Run Data Prep

```bash
cd d:\Capstone Project code\Capstone-Project
python scripts/prepare_datasets.py
```

---

## 4. Stage 3: YOLOv11 Training Optimization (Week 3—4)

### 4.1 Training Script with Optimization

Create `training/train_yolov11_optimized.py`:

```python
"""
training/train_yolov11_optimized.py

YOLOv11 training with:
- Class weighting (handle imbalance)
- Mixed precision (faster, lower memory)
- Augmentation pipeline
- Early stopping
- Learning rate scheduling
- Validation on hard examples
"""

import torch
import yaml
from ultralytics import YOLO
from pathlib import Path

def train_yolov11_optimized(
    dataset_yaml: str = "data/prepared/dataset.yaml",
    model_size: str = "n",  # n=nano, s=small, m=medium (nano for edge)
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: int = 0,  # GPU device index
    output_dir: str = "runs/detect/train",
):
    """
    Optimized training pipeline for YOLOv11 detection.
    
    Args:
        model_size: 'n' (640MB), 's' (2GB), 'm' (5GB) — use 'n' for edge
        batch_size: 8–32 depending on GPU memory
        imgsz: 640 standard; can reduce to 416 for faster inference
        device: GPU index or 'cpu'
    """
    
    device_str = f"cuda:{device}" if device >= 0 else "cpu"
    
    # Load YOLOv11 pretrained
    model = YOLO(f"yolov11{model_size}.pt")
    
    # Read dataset config
    with open(dataset_yaml) as f:
        dataset_config = yaml.safe_load(f)
    
    # Compute class weights (inverse frequency)
    class_counts = {}  # populate from training data
    class_weights = None
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_str,
        patience=20,  # early stopping if no improvement for 20 epochs
        save=True,
        save_period=5,  # save checkpoint every 5 epochs
        
        # Optimization
        amp=True,  # mixed precision training (faster)
        workers=8,  # dataloader workers
        
        # Augmentation (YOLOv11 default is strong; tune if needed)
        hsv_h=0.015,  # HSV hue (±÷2)
        hsv_s=0.7,    # HSV saturation
        hsv_v=0.4,    # HSV value
        degrees=10.0, # rotation
        translate=0.1,  # translation
        scale=0.5,    # zoom 0.5x to 1.5x
        flipud=0.0,   # flip upside-down?
        fliplr=0.5,   # flip left-right 50%
        mosaic=1.0,   # mosaic augmentation
        
        # Learning rate schedule
        lr0=0.01,     # initial LR
        lrf=0.01,     # final LR (as fraction of initial)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Logging
        project=output_dir,
        name="yolov11_optimized",
        exist_ok=True,
    )
    
    print(f"✅ Training complete. Results: {results}")
    return model, results

def export_to_onnx(
    model_path: str,
    onnx_output_path: str = "models/yolov11n_best.onnx",
    imgsz: int = 640,
    opset: int = 12,
):
    """Export trained YOLOv11 to ONNX for edge deployment."""
    model = YOLO(model_path)
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,  # fixed input size for edge consistency
    )
    print(f"✅ Exported to ONNX: {export_path}")
    return export_path

def validate_model(model_path: str, dataset_yaml: str):
    """Validate trained model on test set."""
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)
    print(f"Metrics:\n{metrics}")
    return metrics

if __name__ == '__main__':
    # Step 1: Train
    model, results = train_yolov11_optimized(
        dataset_yaml="data/prepared/dataset.yaml",
        model_size="n",  # nano for edge
        epochs=100,
        batch_size=16,
        device=0,
    )
    
    # Step 2: Export to ONNX
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    export_to_onnx(str(best_model_path))
    
    # Step 3: Validate
    validate_model(str(best_model_path), "data/prepared/dataset.yaml")
```

### 4.2 Training Configuration Tuning

Create `configs/training_config.yaml`:

```yaml
# Hyperparameters for YOLOv11 training

# Model
model_size: "n"       # nano: 0.9M params, fast
backbone_depth: 0.67  # 0.67n, 1.0m, 2.0l

# Data
imgsz: 640
batch_size: 16
workers: 8

# Augmentation
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10
  translate: 0.1
  scale: 0.5
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0

# Optimizer
optimizer: "SGD"
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# Training
epochs: 100
patience: 20
save_period: 5

# Loss weights
box: 7.5
cls: 0.5
dfl: 1.5

# Class balancing
class_weights: null  # auto compute if null
```

### 4.3 Test Validation

```python
# tests/unit/test_yolov11_inference.py

import pytest
import numpy as np
import cv2
from inference.models.yolov11_detector import YOLOv11Detector

@pytest.fixture
def detector():
    return YOLOv11Detector()

def test_yolo_detect_performance(detector, benchmark):
    """Benchmark: YOLOv11 detection latency."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    result = benchmark(detector.detect, frame)
    
    assert isinstance(result, list)
    assert detector.config.YOLOV11_CONF_THRESHOLD > 0

def test_yolo_output_schema(detector):
    """Verify detection output schema."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    for det in detections:
        assert hasattr(det, 'class_id')
        assert hasattr(det, 'confidence')
        assert hasattr(det, 'bbox')
        assert 0 <= det.confidence <= 1
```

---

## 5. Stage 4: EfficientViT Training Optimization (Week 4—5)

### 5.1 EfficientViT Training Script

Create `training/train_efficientvit_optimized.py`:

```python
"""
training/train_efficientvit_optimized.py

EfficientViT training for defect classification with MC Dropout.
- Vision Transformer backbone (EfficientViT-M5)
- Dropout enabled for uncertainty quantification
- FocalLoss for class imbalance
- Mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import timm
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

class EfficientViTDefectClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        # EfficientViT-M5 from timm
        self.backbone = timm.create_model(
            'efficientvit_m5',
            pretrained=True,
            num_classes=0,  # headless
        )
        
        feature_dim = self.backbone.num_features
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def train_efficientvit_optimized(
    dataset_dir: str = "data/prepared",
    num_classes: int = 5,  # good + 4 defect types
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cuda:0",
):
    """
    Optimized EfficientViT training.
    
    Dataset structure (classification):
    data/prepared/
    ├── train/
    │   ├── good/
    │   ├── defect1/
    │   └── defect2/
    ├── val/
    └── test/
    """
    
    device_obj = torch.device(device)
    
    # Model
    model = EfficientViTDefectClassifier(num_classes=num_classes, dropout_rate=0.2)
    model = model.to(device_obj)
    
    # Loss: FocalLoss for imbalanced classes
    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        
        def forward(self, pred, target):
            ce = nn.functional.cross_entropy(pred, target, reduction='none')
            p_t = torch.exp(-ce)
            loss = (1 - p_t) ** self.gamma * ce
            if self.alpha is not None:
                loss = self.alpha[target] * loss
            return loss.mean()
    
    loss_fn = FocalLoss(gamma=2.0)
    
    # Optimizer: AdamW + weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    
    # Scheduler: cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    
    scaler = GradScaler()  # mixed precision
    
    # Training loop (simplified)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Load training data (would iterate over DataLoader in practice)
        for batch_idx in range(100):  # placeholder
            # Forward
            with autocast():
                logits = model(torch.randn(batch_size, 3, 384, 384).to(device_obj))
                loss = loss_fn(logits, torch.randint(0, num_classes, (batch_size,)).to(device_obj))
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / 100:.4f}")
    
    # Save
    torch.save(model.state_dict(), "models/efficientvit_m5_best.pth")
    print("✅ EfficientViT training complete")
    
    return model

if __name__ == '__main__':
    model = train_efficientvit_optimized(
        dataset_dir="data/prepared",
        num_classes=5,
        epochs=100,
        batch_size=32,
    )
```

### 5.2 Export EfficientViT to ONNX with Dropout

```python
def export_efficientvit_with_dropout_to_onnx(
    model_path: str,
    checkpoint_path: str,
    onnx_output_path: str = "models/efficientvit_m5_best.onnx",
    imgsz: int = 384,
):
    """
    Export EfficientViT to ONNX making sure dropout remains trainable
    (for MC Dropout during inference).
    """
    import onnx
    import onnxruntime as ort
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EfficientViTDefectClassifier(num_classes=5, dropout_rate=0.2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.train()  # ← keep in train mode to enable dropout
    
    # Dummy input
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=False,  # ← important: don't fold dropout
    )
    
    print(f"✅ EfficientViT exported to {onnx_output_path}")
```

---

## 6. Stage 5: Model Integration & Edge Testing (Week 5)

### 6.1 Load and Test Both Models

```python
# tests/integration/test_full_pipeline_inference.py

import pytest
import numpy as np
import time
from inference.pipeline import EdgeInferencePipeline
from inference.models.yolov11_detector import YOLOv11Detector
from inference.models.efficientvit_classifier import EfficientViTClassifier

@pytest.fixture
def pipeline():
    p = EdgeInferencePipeline()
    p.load_models()
    return p

def test_pipeline_latency(pipeline):
    """Full pipeline end-to-end latency."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    t0 = time.perf_counter()
    result = pipeline.inspect(frame, sku="bottle_250ml")
    t_elapsed = (time.perf_counter() - t0) * 1000
    
    assert result.latency_ms < 300  # industry standard
    assert t_elapsed < 350  # allow 50ms overhead
    print(f"✅ Pipeline latency: {result.latency_ms:.1f}ms ({t_elapsed:.1f}ms wall)")

def test_pipeline_with_qr_integration(pipeline):
    """QR + YOLOv11 + EfficientViT together."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    result = pipeline.inspect(frame, sku="bottle_250ml", product_id="P001")
    
    # Verify all components ran
    assert result.label_qr is not None
    assert result.detections is not None
    if result.detections:
        assert result.uq_result is not None
    assert result.verdict in ["PASS", "FAIL", "ESCALATE", "REVIEW"]
    print(f"✅ Full pipeline result: verdict={result.verdict}, latency={result.latency_ms}ms")

def test_model_onnx_inference_speed(pipeline):
    """Measure raw ONNX model inference speed."""
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # YOLOv11 speed
    t0 = time.perf_counter()
    detections = pipeline._run_detection(frame)
    t_yolo = (time.perf_counter() - t0) * 1000
    
    print(f"YOLOv11 inference: {t_yolo:.1f}ms")
    assert t_yolo < 100
    
    # EfficientViT speed (if detections exist)
    if detections:
        t0 = time.perf_counter()
        uq_result = pipeline._run_uq(frame, detections)
        t_vit = (time.perf_counter() - t0) * 1000
        print(f"EfficientViT+UQ inference: {t_vit:.1f}ms")
        assert t_vit < 100
```

---

## 7. Stage 6: Deployment & Validation (Week 5—6)

### 7.1 Create Model Registry

```python
# database/repositories/model_repository.py

from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from uuid import uuid4

class ModelRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.model_versions
    
    async def create(self, name: str, version_tag: str, detector_path: str, classifier_path: str, metrics: dict):
        """Register a new model version."""
        doc = {
            "_id": str(uuid4()),
            "name": name,
            "version_tag": version_tag,
            "detector_path": detector_path,
            "classifier_path": classifier_path,
            "metrics": metrics,  # mAP50, accuracy, etc.
            "is_active": False,
            "created_at": datetime.utcnow(),
            "__v": 0,
        }
        result = await self.collection.insert_one(doc)
        return doc
    
    async def activate(self, version_tag: str):
        """Set a model version as active (exclusive)."""
        # Deactivate all others
        await self.collection.update_many({}, {"$set": {"is_active": False}})
        # Activate target
        result = await self.collection.find_one_and_update(
            {"version_tag": version_tag, "__v": 0},  # optimistic concurrency
            {
                "$set": {"is_active": True},
                "$inc": {"__v": 1},
            },
            return_document=True,
        )
        if not result:
            raise Exception(f"Model {version_tag} not found or already modified")
        return result
```

### 7.2 Model Checklist

- [ ] YOLOv11 trained on combined dataset, exported to ONNX
- [ ] EfficientViT trained with dropout, exported to ONNX (training mode)
- [ ] Both models fit in edge container memory budget (~4GB)
- [ ] Inference latency: YOLOv11 <100ms, EfficientViT+UQ <100ms, total <300ms
- [ ] pyzbar integrated and tested with mock QR codes
- [ ] QR lookup from MongoDB tested (latency <10ms with caching)
- [ ] Full pipeline end-to-end tested with 100 sample frames
- [ ] Verdict logic with QR mismatch prioritization tested
- [ ] Uncertainty calibration: UQ std matches empirical defect rate
- [ ] Models registered in MongoDB model_versions collection

---

## Summary: Timeline

| Phase | Weeks | Deliverables |
|---|---|---|
| **1: pyzbar Integration** | W1 | pyzbar in pipeline, LabelQRStatus schema, unit tests |
| **2: Dataset Prep** | W2–3 | Combined dataset, stratified split, augmentation pipeline |
| **3: YOLOv11 Training** | W3–4 | Trained nano model, ONNX export, latency <100ms |
| **4: EfficientViT Training** | W4–5 | MC Dropout classifier, ONNX export, UQ calibration |
| **5: Integration & Testing** | W5 | Full pipeline e2e tests, latency validation |
| **6: Deployment** | W5–6 | Model registry, Cloud Run containerization |

