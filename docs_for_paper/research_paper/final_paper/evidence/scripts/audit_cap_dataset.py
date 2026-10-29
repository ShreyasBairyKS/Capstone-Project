"""Audit of the bottle_cap_sdp.v7i six-class segmentation dataset.

Reports per-split image/instance counts, object-size distribution, Roboflow
source grouping (augmented copies share the stem before '.rf.'), and
cross-split near-duplicates by 64-bit dHash.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NAMES = ['damaged_cap', 'good_cap', 'misplaced_cap', 'no_cap', 'open_cap', 'wet_cap']
SPLITS = ['train', 'valid', 'test']


def dhash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def source_stem(name):
    return name.split('.rf.')[0]


report = {}
hashes = {}
stems = {}
for split in SPLITS:
    imgs = sorted((ROOT / split / 'images').iterdir())
    inst = Counter()
    img_has = Counter()
    per_img = []
    areas = []  # normalised bbox area of each instance
    sizes = Counter()
    for p in imgs:
        lab = ROOT / split / 'labels' / (p.stem + '.txt')
        cls_in_img = set()
        n = 0
        if lab.exists():
            for line in lab.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                xy = np.array(list(map(float, parts[1:])))
                xs, ys = xy[0::2], xy[1::2]
                areas.append(float((xs.max() - xs.min()) * (ys.max() - ys.min())))
                inst[NAMES[c]] += 1
                cls_in_img.add(NAMES[c])
                n += 1
        for c in cls_in_img:
            img_has[c] += 1
        per_img.append(n)
        if len(sizes) < 50 or split != 'train':
            im = cv2.imread(str(p))
            if im is not None:
                sizes[f'{im.shape[1]}x{im.shape[0]}'] += 1
    # COCO size buckets after resizing to the 640x640 network input
    a640 = np.array(areas) * 640 * 640
    st = [source_stem(p.name) for p in imgs]
    stems[split] = set(st)
    hashes[split] = [(p.name, dhash(p)) for p in imgs]
    report[split] = {
        'images': len(imgs),
        'instances': dict(inst),
        'images_containing_class': dict(img_has),
        'instances_total': int(sum(inst.values())),
        'objects_per_image_mean': float(np.mean(per_img)),
        'images_with_multiple_objects': int(sum(1 for n in per_img if n > 1)),
        'images_without_labels': int(sum(1 for n in per_img if n == 0)),
        'bbox_area_frac_median': float(np.median(areas)),
        'bbox_area_frac_p10': float(np.percentile(areas, 10)),
        'bbox_area_frac_p90': float(np.percentile(areas, 90)),
        'coco_small(<32^2@640)': int((a640 < 32 ** 2).sum()),
        'coco_medium': int(((a640 >= 32 ** 2) & (a640 < 96 ** 2)).sum()),
        'coco_large(>=96^2@640)': int((a640 >= 96 ** 2).sum()),
        'unique_source_stems': len(set(st)),
        'max_copies_per_stem': max(Counter(st).values()),
        'resolution_sample': dict(sizes.most_common(3)),
    }

# Cross-split leakage
report['stem_overlap'] = {
    'train&valid': len(stems['train'] & stems['valid']),
    'train&test': len(stems['train'] & stems['test']),
    'valid&test': len(stems['valid'] & stems['test']),
}
train_h = np.array([h for _, h in hashes['train'] if h is not None], dtype=np.uint64)


def popcount(x):
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.int64)
    while np.any(x):
        c += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return c


dup = {}
for split in ['valid', 'test']:
    dists = []
    for _, h in hashes[split]:
        if h is None:
            continue
        dists.append(int(popcount(train_h ^ np.uint64(h)).min()))
    d = np.array(dists)
    dup[split] = {
        'n': len(d),
        'exact_dhash(d=0)': int((d == 0).sum()),
        'near_dup(d<=4)': int((d <= 4).sum()),
        'near_dup(d<=8)': int((d <= 8).sum()),
        'median_min_dist': float(np.median(d)),
    }
# test vs valid
val_h = np.array([h for _, h in hashes['valid'] if h is not None], dtype=np.uint64)
d = np.array([int(popcount(val_h ^ np.uint64(h)).min()) for _, h in hashes['test'] if h is not None])
dup['test_vs_valid'] = {'exact(d=0)': int((d == 0).sum()), 'near_dup(d<=4)': int((d <= 4).sum()),
                        'near_dup(d<=8)': int((d <= 8).sum())}
report['dhash_min_hamming_to_train'] = dup
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
