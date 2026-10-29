"""End-to-end Station-2 timing through FillPipelineService (inference + gate + audit + overlay)."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/app')
from pipeline.fill_level.api.service import FillPipelineService

svc = FillPipelineService.get_instance()
svc.ensure_engine_loaded()
imgs = [cv2.imread(str(p)) for p in sorted(Path('/app/simulation/fill').iterdir())]
imgs = [i for i in imgs if i is not None]
for k in range(10):
    svc.inspect_frame(imgs[k % len(imgs)], bottle_id=f'W{k}', render_base64=True)
lat = []
t_all = time.perf_counter()
for k in range(500):
    t0 = time.perf_counter()
    svc.inspect_frame(imgs[k % len(imgs)], bottle_id=f'E2E_{k}', render_base64=True)
    lat.append((time.perf_counter() - t0) * 1000)
dur = time.perf_counter() - t_all
p = lambda q: round(float(np.percentile(lat, q)), 2)
r = {'n': 500, 'p50': p(50), 'p90': p(90), 'p95': p(95), 'p99': p(99), 'max': round(max(lat), 2),
     'throughput_bpm': round(500 / dur * 60, 1),
     'models': [svc.engine.bottle_model_path.name, svc.engine.water_model_path.name]}
print(json.dumps(r))
Path('/app/audit/fill_e2e.json').write_text(json.dumps(r, indent=2))
