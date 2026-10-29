"""Held-out evaluation of the deployed models (run inside the backend container).

Cap station : Ultralytics val (FP32 .pt and FP16 TensorRT engine) on valid + test,
              per-class metrics, top-1 image classification, and the production
              QualityGate replayed over a threshold grid (image-level FAR/FRR).
Fill station: detector val on held-out test splits and the fill-ratio error caused
              by meniscus localisation (predicted vs ground-truth water box).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, '/app')
from pipeline.decision.gate import QualityGate
from pipeline.decision.rules import InspectionRulesConfig
from pipeline.decision.schemas import InspectionVerdict
from pipeline.inference.schemas import CameraResult, Detection

OUT = Path(sys.argv[1])
E = Path('/tmp/eval')
CAP_W = Path('/app/YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights')
FILL_W = Path('/app/models/fill-level-detector')
NAMES = ['damaged_cap', 'good_cap', 'misplaced_cap', 'no_cap', 'open_cap', 'wet_cap']
rng = np.random.default_rng(0)
R = {}

(E / 'cap.yaml').write_text(f"path: {E/'cap'}\ntrain: valid/images\nval: valid/images\ntest: test/images\nnc: 6\nnames: {NAMES}\n")
(E / 'bcm.yaml').write_text(f"path: {E/'bcm'}\ntrain: test/images\nval: test/images\ntest: test/images\nnc: 2\nnames: ['bottle', 'cap']\n")
(E / 'wfm.yaml').write_text(f"path: {E/'wfm'}\ntrain: test/images\nval: test/images\ntest: test/images\nnc: 1\nnames: ['water_surface']\n")


def summarize(m, names, seg=True):
    out = {}
    parts = [('box', m.box)] + ([('mask', m.seg)] if seg else [])
    for tag, b in parts:
        out[tag] = {'P': round(float(b.mp), 4), 'R': round(float(b.mr), 4),
                    'mAP50': round(float(b.map50), 4), 'mAP50_95': round(float(b.map), 4)}
        per = {}
        for i, c in enumerate(b.ap_class_index):
            per[names[int(c)]] = {'P': round(float(b.p[i]), 4), 'R': round(float(b.r[i]), 4),
                                  'AP50': round(float(b.ap50[i]), 4), 'AP50_95': round(float(b.ap[i]), 4)}
        out[tag]['per_class'] = per
    out['speed_ms'] = {k: round(v, 2) for k, v in m.speed.items()}
    return out


# ---------------------------------------------------------------- cap: val metrics
R['cap_val'] = {}
for wname, wpath, batch in (('fp32_pt', CAP_W / 'best.pt', 16), ('fp16_trt', CAP_W / 'best_v1.engine', 1)):
    for split in ('val', 'test'):
        model = YOLO(str(wpath), task='segment')
        t0 = time.time()
        m = model.val(data=str(E / 'cap.yaml'), split=split, imgsz=640, batch=batch, device=0,
                      plots=False, verbose=False, workers=0, half=(wname == 'fp16_trt'))
        s = summarize(m, NAMES)
        s['seconds'] = round(time.time() - t0, 1)
        R['cap_val'][f'{wname}_{split}'] = s
        print(wname, split, json.dumps({k: s[k] for k in ('box', 'mask')}, default=str)[:400], flush=True)
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- cap: per-image predictions
def gt_of(img_path):
    lab = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
    line = lab.read_text().split('\n')[0].split()
    xy = np.array(list(map(float, line[1:])))
    return int(line[0]), float((xy[0::2].max() - xy[0::2].min()) * (xy[1::2].max() - xy[1::2].min()))


def collect(wpath, split):
    model = YOLO(str(wpath), task='segment')
    rows = []
    for p in sorted((E / 'cap' / split / 'images').iterdir()):
        img = cv2.imread(str(p))
        r = model.predict(img, imgsz=640, conf=0.01, device=0, verbose=False)[0]
        dets = [(int(c), float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        g, area = gt_of(p)
        rows.append({'gt': g, 'area': area, 'dets': dets})
    del model
    torch.cuda.empty_cache()
    return rows


def gate_eval(rows, pass_t=0.75, defect_t=0.45, floor_u=0.40, conf_floor=0.35):
    cfg = InspectionRulesConfig(pass_conf_threshold=pass_t, defect_conf_threshold=defect_t, uncertain_floor=floor_u)
    gate = QualityGate(cfg)
    out = []
    for row in rows:
        dets = [Detection(class_id=c, class_name=NAMES[c], confidence=s, box=[0, 0, 1, 1])
                for c, s in row['dets'] if s >= conf_floor]
        v = gate.evaluate_view(CameraResult(camera_id='cam', frame_width=640, frame_height=640,
                                            detections=dets, inference_latency_ms=0.0)).verdict
        out.append((row['gt'] != 1, v))
    return out


def rates(out):
    d = np.array([o[0] for o in out])
    v = np.array([o[1].value for o in out])
    rej = np.isin(v, ['REJECT', 'INSPECTION_FAILED'])
    pas = v == 'PASS'
    unc = v == 'UNCERTAIN'
    return {
        'n': int(len(v)), 'n_defect': int(d.sum()), 'n_good': int((~d).sum()),
        'FAR_defect_passed': float(pas[d].mean()),            # false accept: defective bottle certified PASS
        'FRR_good_rejected': float(rej[~d].mean()),           # false reject: good bottle diverted
        'review_rate': float(unc.mean()),
        'defect_caught': float(rej[d].mean()),
        'good_passed': float(pas[~d].mean()),
        'decisive_accuracy': float(((pas & ~d) | (rej & d)).sum() / max(1, (~unc).sum())),
        'auto_accuracy_all': float(((pas & ~d) | (rej & d)).mean()),
        'verdicts': {k: int((v == k).sum()) for k in ['PASS', 'REJECT', 'UNCERTAIN', 'INSPECTION_FAILED']},
    }


def boot_ci(out, key, n=2000):
    vals = []
    arr = np.array(out, dtype=object)
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))]
        vals.append(rates(list(s))[key])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def top1(rows, conf_floor=0.35):
    k = len(NAMES)
    cm = np.zeros((k, k + 1), dtype=int)  # last column: no detection
    for row in rows:
        ds = [d for d in row['dets'] if d[1] >= conf_floor]
        pred = max(ds, key=lambda d: d[1])[0] if ds else k
        cm[row['gt'], pred] += 1
    n = cm.sum()
    acc = np.trace(cm[:, :k]) / n
    per = {}
    f1s = []
    for i in range(k):
        tp = cm[i, i]
        p = tp / max(1, cm[:, i].sum())
        r = tp / max(1, cm[i].sum())
        f = 2 * p * r / max(1e-9, p + r)
        f1s.append(f)
        per[NAMES[i]] = {'P': round(float(p), 4), 'R': round(float(r), 4), 'F1': round(float(f), 4), 'support': int(cm[i].sum())}
    sq = cm[:, :k]
    po = acc
    pe = float((sq.sum(0) * cm.sum(1)).sum()) / n ** 2
    kappa = (po - pe) / (1 - pe)
    # multiclass MCC (Gorodkin), with "no detection" counted as a wrong prediction
    t = cm.sum(1).astype(float)
    pcol = np.append(sq.sum(0), cm[:, k].sum()).astype(float)
    c = float(np.trace(sq))
    mcc = (c * n - float((t * pcol[:k]).sum())) / np.sqrt((n ** 2 - float((pcol ** 2).sum())) * (n ** 2 - float((t ** 2).sum())))
    return {'accuracy': round(float(acc), 4), 'macro_F1': round(float(np.mean(f1s)), 4), 'kappa': round(float(kappa), 4),
            'MCC': round(float(mcc), 4), 'per_class': per, 'confusion': cm.tolist(),
            'confusion_cols': NAMES + ['none']}


R['cap_gate'] = {}
for wname, wpath in (('fp32_pt', CAP_W / 'best.pt'), ('fp16_trt', CAP_W / 'best_v1.engine')):
    for split in ('valid', 'test'):
        rows = collect(wpath, split)
        key = f'{wname}_{split}'
        default = gate_eval(rows)
        entry = {'default': rates(default), 'top1': top1(rows)}
        entry['default']['ci95'] = {k: boot_ci(default, k) for k in ('FAR_defect_passed', 'FRR_good_rejected', 'review_rate', 'auto_accuracy_all')}
        # size terciles (GT bbox area fraction) vs top-1 correctness
        areas = np.array([r['area'] for r in rows])
        q = np.percentile(areas, [33.3, 66.7])
        terc = {}
        for name, lo, hi in (('small', 0, q[0]), ('mid', q[0], q[1]), ('large', q[1], 9)):
            sel = [r for r in rows if lo <= r['area'] < hi]
            terc[name] = {'n': len(sel), 'area_range': [round(lo, 4), round(min(hi, areas.max()), 4)],
                          'top1_acc': top1(sel)['accuracy'],
                          'gate': {k: round(v, 4) for k, v in rates(gate_eval(sel)).items() if isinstance(v, float)}}
        entry['size_terciles'] = terc
        if wname == 'fp32_pt' or split == 'test':
            grid = []
            for pt in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
                for dt in (0.25, 0.35, 0.45, 0.55, 0.65):
                    r = rates(gate_eval(rows, pass_t=pt, defect_t=dt, floor_u=min(0.40, pt), conf_floor=min(0.35, dt)))
                    grid.append({'pass_t': pt, 'defect_t': dt, **{k: round(v, 4) for k, v in r.items() if isinstance(v, float)}})
            entry['grid'] = grid
        R['cap_gate'][key] = entry
        print(key, json.dumps(entry['default'])[:300], json.dumps({k: entry['top1'][k] for k in ('accuracy', 'macro_F1', 'kappa', 'MCC')}), flush=True)
        if wname == 'fp32_pt':
            R.setdefault('_rows', {})[key] = rows  # kept for the FP16 agreement check

# FP32 vs FP16 top-1 agreement on test
a = R['_rows']['fp32_pt_test']
R.pop('_rows')

# ---------------------------------------------------------------- fill: detector val
R['fill_val'] = {}
for tag, w, data, names in (
    ('bottle_fp32', FILL_W / 'bottle_detector_best.pt', 'bcm.yaml', ['bottle', 'cap']),
    ('bottle_fp16_trt', FILL_W / 'bottle_detector_v1.engine', 'bcm.yaml', ['bottle', 'cap']),
    ('water_fp32', FILL_W / 'best.pt', 'wfm.yaml', ['water_surface']),
    ('water_fp16_trt', FILL_W / 'water_surface_v1.engine', 'wfm.yaml', ['water_surface']),
):
    model = YOLO(str(w), task='detect')
    m = model.val(data=str(E / data), split='test', imgsz=640, batch=1, device=0, plots=False, verbose=False, workers=0,
                  conf=0.001, half=('trt' in tag))
    R['fill_val'][tag] = summarize(m, names, seg=False)
    print(tag, json.dumps(R['fill_val'][tag]['box'])[:300], flush=True)
    del model
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- fill: meniscus-induced ratio error
from pipeline.fill_level.inference.engine import FillLevelInferenceEngine
from pipeline.fill_level.decision.gate import FillQualityGate
eng = FillLevelInferenceEngine()
gate = FillQualityGate()
rows = []
for p in sorted((E / 'wfm' / 'test' / 'images').iterdir()):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    res = eng.predict_single_frame(img)
    lab = (E / 'wfm' / 'test' / 'labels' / (p.stem + '.txt')).read_text().split()
    gts = []
    for i in range(0, len(lab), 5):
        _, cx, cy, bw, bh = map(float, lab[i:i + 5])
        gts.append([int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)])
    row = {'img': p.name, 'size': [w, h], 'n_gt_water': len(gts), 'status': res.fill_verdict}
    if res.bottle is not None and gts:
        bx1, by1, bx2, by2 = res.bottle.box
        # associate ground-truth meniscus with the predicted bottle the same way the engine does
        inside = [g for g in gts if bx1 <= (g[0] + g[2]) // 2 <= bx2] or gts
        g = inside[0]
        r_gt = FillLevelInferenceEngine.compute_fill_ratio(res.bottle.box, g)
        row.update(bottle_h=by2 - by1, r_gtwater=r_gt,
                   v_gtwater='underfill' if r_gt < eng.underfill_threshold else 'overfill' if r_gt > eng.overfill_threshold else 'normal')
        if res.meniscus is not None:
            row.update(r_pred=res.fill_ratio, dR=round(abs(res.fill_ratio - r_gt), 4),
                       dy_px=abs(res.meniscus.water_y - (g[1] + g[3]) // 2), v_pred=res.fill_verdict,
                       men_conf=res.meniscus.confidence,
                       gate=gate.evaluate(res, bottle_id=p.stem).verdict.value)
    rows.append(row)
ok = [r for r in rows if 'dR' in r]
dR = np.array([r['dR'] for r in ok]) if ok else np.array([0.0])
R['fill_ratio_error'] = {
    'n_images': len(rows),
    'no_bottle': sum(r['status'] == 'no_bottle' for r in rows),
    'no_water': sum(r['status'] == 'no_water' for r in rows),
    'evaluated': len(ok),
    'MAE_R': round(float(dR.mean()), 4), 'median_dR': round(float(np.median(dR)), 4),
    'p90_dR': round(float(np.percentile(dR, 90)), 4), 'max_dR': round(float(dR.max()), 4),
    'within_0.02': round(float((dR <= 0.02).mean()), 4), 'within_0.05': round(float((dR <= 0.05).mean()), 4),
    'median_dy_px': float(np.median([r['dy_px'] for r in ok])) if ok else None,
    'median_bottle_h_px': float(np.median([r['bottle_h'] for r in ok])) if ok else None,
    'verdict_agreement': round(float(np.mean([r['v_pred'] == r['v_gtwater'] for r in ok])), 4) if ok else None,
    'gate_verdicts': {k: sum(r.get('gate') == k for r in ok) for k in ('PASS', 'REJECT', 'UNCERTAIN')},
    'gt_verdicts': {k: sum(r.get('v_gtwater') == k for r in ok) for k in ('underfill', 'normal', 'overfill')},
    'rows': rows,
}
print('fill_ratio_error', json.dumps({k: v for k, v in R['fill_ratio_error'].items() if k != 'rows'}), flush=True)
OUT.write_text(json.dumps(R, indent=2, default=str))
