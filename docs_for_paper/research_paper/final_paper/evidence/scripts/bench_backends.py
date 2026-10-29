"""Per-stage latency of the deployed models across runtimes (run inside the backend container).

For each (model, runtime, input-size, batch) it reports Ultralytics' per-image
preprocess / inference / postprocess split and the wall time per call.
Then it times the full Station-2 cycle (two detectors + fill math + gate).
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, '/app')
OUT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CAP_DIR = Path('/app/YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights')
FILL_DIR = Path('/app/models/fill-level-detector')

cap_imgs = [cv2.imread(str(p)) for p in sorted(Path('/app/simulation/cap').iterdir())[:30]]
fill_imgs = [cv2.imread(str(p)) for p in sorted(Path('/app/simulation/fill').iterdir())]
fill_imgs = [i for i in fill_imgs if i is not None]


def pct(v, q):
    return float(np.percentile(np.array(v), q))


def bench(model, imgs, batch, device, n, imgsz=640, conf=0.35):
    for k in range(5):  # warm-up
        model.predict([imgs[(k + j) % len(imgs)] for j in range(batch)], imgsz=imgsz, conf=conf, device=device, verbose=False)
    wall, pre, inf, post = [], [], [], []
    for k in range(n):
        src = [imgs[(k * batch + j) % len(imgs)] for j in range(batch)]
        if device != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.predict(src, imgsz=imgsz, conf=conf, device=device, verbose=False)
        if device != 'cpu':
            torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(sum(r.speed['preprocess'] for r in res))
        inf.append(sum(r.speed['inference'] for r in res))
        post.append(sum(r.speed['postprocess'] for r in res))
    return {
        'batch': batch, 'n': n,
        'wall_p50': round(pct(wall, 50), 2), 'wall_p95': round(pct(wall, 95), 2), 'wall_mean': round(st.mean(wall), 2),
        'pre_p50': round(pct(pre, 50), 2), 'inf_p50': round(pct(inf, 50), 2), 'post_p50': round(pct(post, 50), 2),
        'inf_p95': round(pct(inf, 95), 2),
    }


results = {'env': {
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}}
try:
    import tensorrt
    results['env']['tensorrt'] = tensorrt.__version__
except Exception:
    pass
import ultralytics
results['env']['ultralytics'] = ultralytics.__version__

cap_640 = [cv2.resize(i, (640, 640), interpolation=cv2.INTER_AREA) for i in cap_imgs]
runs = []
cap_cfgs = [
    ('cap', 'pytorch_fp32_gpu', CAP_DIR / 'best.pt', '0'),
    ('cap', 'tensorrt_fp16', CAP_DIR / 'best_v1.engine', '0'),
    ('cap', 'onnx_fp32_cpu', CAP_DIR / 'best.onnx', 'cpu'),
]
for name, rt, path, dev in cap_cfgs:
    m = YOLO(str(path), task='segment')
    for label, imgs in (('native2048', cap_imgs), ('pre640', cap_640)):
        for b in (1, 3):
            n = N if dev != 'cpu' else max(10, N // 5)
            try:
                r = bench(m, imgs, b, dev, n)
            except Exception as e:  # e.g. engine batch limits
                r = {'error': str(e)[:200], 'batch': b}
            r.update(model=name, runtime=rt, input=label)
            print(json.dumps(r), flush=True)
            runs.append(r)
    del m
    torch.cuda.empty_cache()

fill_cfgs = [
    ('bottle_detector', 'pytorch_fp32_gpu', FILL_DIR / 'bottle_detector_best.pt', '0'),
    ('bottle_detector', 'tensorrt_fp16', FILL_DIR / 'bottle_detector_v1.engine', '0'),
    ('bottle_detector', 'onnx_fp32_cpu', FILL_DIR / 'bottle_detector_best.onnx', 'cpu'),
    ('water_surface', 'pytorch_fp32_gpu', FILL_DIR / 'best.pt', '0'),
    ('water_surface', 'tensorrt_fp16', FILL_DIR / 'water_surface_v1.engine', '0'),
    ('water_surface', 'onnx_fp32_cpu', FILL_DIR / 'water_surface_best.onnx', 'cpu'),
]
for name, rt, path, dev in fill_cfgs:
    m = YOLO(str(path), task='detect')
    n = N if dev != 'cpu' else max(10, N // 5)
    try:
        r = bench(m, fill_imgs, 1, dev, n, conf=0.15)
    except Exception as e:
        r = {'error': str(e)[:200], 'batch': 1}
    r.update(model=name, runtime=rt, input='native')
    print(json.dumps(r), flush=True)
    runs.append(r)
    del m
    torch.cuda.empty_cache()
results['runs'] = runs

# Full Station-2 cycle with the production classes (TensorRT engines on GPU)
from pipeline.fill_level.inference.engine import FillLevelInferenceEngine
from pipeline.fill_level.decision.gate import FillQualityGate
eng = FillLevelInferenceEngine()
gate = FillQualityGate()
lat, gate_lat, verdicts = [], [], {}
for k in range(N * 3):
    f = fill_imgs[k % len(fill_imgs)]
    t0 = time.perf_counter()
    r = eng.predict_single_frame(f)
    t1 = time.perf_counter()
    d = gate.evaluate(r, bottle_id=f'B{k}')
    t2 = time.perf_counter()
    lat.append((t1 - t0) * 1000)
    gate_lat.append((t2 - t1) * 1000)
    if k < len(fill_imgs):
        verdicts[d.verdict.value] = verdicts.get(d.verdict.value, 0) + 1
results['fill_cycle'] = {
    'bottle_model': eng.bottle_model_path.name, 'water_model': eng.water_model_path.name,
    'n': len(lat), 'images': len(fill_imgs), 'sizes': sorted({f'{i.shape[1]}x{i.shape[0]}' for i in fill_imgs})[:5],
    'infer_p50': round(pct(lat, 50), 2), 'infer_p95': round(pct(lat, 95), 2), 'infer_p99': round(pct(lat, 99), 2),
    'gate_p50': round(pct(gate_lat, 50), 4), 'gate_p99': round(pct(gate_lat, 99), 4),
    'verdicts_first_pass': verdicts,
}
print(json.dumps(results['fill_cycle']), flush=True)
OUT.write_text(json.dumps(results, indent=2))
