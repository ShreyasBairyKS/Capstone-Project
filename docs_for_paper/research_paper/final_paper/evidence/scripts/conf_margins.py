"""Confidence margins of the deployed FP16 engine on valid/test: top good_cap score on good images,
top defect score on defect images, and the highest wrong-polarity score."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
m = YOLO('/app/YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights/best_v1.engine', task='segment')
out = {}
for split in ('valid', 'test'):
    good_top, def_top, good_maxdef, def_maxgood = [], [], [], []
    for p in sorted(Path(f'/tmp/eval/cap/{split}/images').iterdir()):
        g = int((p.parent.parent / 'labels' / (p.stem + '.txt')).read_text().split()[0])
        r = m.predict(cv2.imread(str(p)), imgsz=640, conf=0.01, device=0, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
        gs = cf[cls == 1].max() if (cls == 1).any() else 0.0
        ds = cf[cls != 1].max() if (cls != 1).any() else 0.0
        if g == 1: good_top.append(gs); good_maxdef.append(ds)
        else: def_top.append(ds); def_maxgood.append(gs)
    q = lambda v: {k: round(float(np.percentile(v, k2)), 4) for k, k2 in (('min', 0), ('p5', 5), ('p50', 50), ('p95', 95), ('max', 100))}
    out[split] = {'good_top_goodcap': q(good_top), 'good_max_defect': q(good_maxdef),
                  'defect_top_defect': q(def_top), 'defect_max_goodcap': q(def_maxgood)}
print(json.dumps(out, indent=1)); Path('/app/audit/conf_margins.json').write_text(json.dumps(out, indent=2))
