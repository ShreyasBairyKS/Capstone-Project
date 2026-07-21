# Himalaya Label Defect Detection — Phase 1

## Quickstart (3 commands)

```bash
cd "himalaya_label_detection"
pip install -r requirements.txt
```

---

## Step 1 — Place your images

```
data/raw/good/    ← put the 80 good label images here
data/raw/bad/     ← put the 50 bad label images here
```

Any of these formats work: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

## Step 2 — Set up golden master + draw ROI boxes (run once)

```bash
python scripts/setup_golden_master.py
```

What happens:
1. You pick which good image becomes the golden master reference.
2. An interactive window opens — you **drag a box** around each of the 9 functional ROIs:
   - `ROI_LOGO`, `ROI_PRODUCT_NAME`, `ROI_INGREDIENT_BAR`, `ROI_GRAPHICS`
   - `ROI_CLAIMS`, `ROI_BARCODE`, `ROI_BATCH_CODE`, `ROI_CERT`, `ROI_TEXT_BLOCK`
3. Coordinates are saved to `config/roi_config.yaml`.

Controls: **drag to draw → SPACE/ENTER to confirm → 'c' to skip a ROI**

---

## Step 3 — Verify alignment quality (recommended)

```bash
python scripts/verify_alignment.py          # checks 5 random images
python scripts/verify_alignment.py --n 15   # checks 15 images
python scripts/verify_alignment.py --all    # checks all images
python scripts/verify_alignment.py --bad    # also includes bad images
```

Controls: **SPACE = next**, **'f' = flag this image**, **ESC = quit**

If you see poor alignment, check:
- Lighting — the label needs enough texture for ORB to find keypoints
- Increase `orb_max_features` in `config/pipeline_config.yaml`

---

## Step 4 — Run preprocessing

```bash
python scripts/run_preprocessing.py
```

This produces:

```
data/processed/
├── aligned/
│   ├── good/          ← 80 aligned good images
│   └── bad/           ← 50 aligned bad images
├── augmented/
│   └── good/          ← ~2050 augmented good images  (80 × 25 + originals)
└── rois/
    ├── ROI_LOGO/
    │   ├── good/      ← logo crops from good images
    │   └── bad/       ← logo crops from bad images
    ├── ROI_BARCODE/
    │   ├── good/
    │   └── bad/
    └── ...            ← one folder per ROI
```

A `preprocessing_report.txt` is written summarising counts, failures, and timing.

### Optional flags

```bash
# Test alignment without waiting for augmentation
python scripts/run_preprocessing.py --skip-augmentation

# Test without ROI extraction
python scripts/run_preprocessing.py --skip-rois
```

---

## Step 5 — Copy to training machine

```bash
# On Windows (robocopy preserves folder structure)
robocopy data\processed <TRAINING_MACHINE_PATH> /E

# Or zip it
Compress-Archive -Path data\processed -DestinationPath processed_data.zip
```

The training machine only needs `data/processed/` — nothing else from this repo.

---

## Configuration

Edit `config/pipeline_config.yaml` to change:

| Setting | Default | Effect |
|---|---|---|
| `image.width / height` | 1024 × 512 | Resize resolution for alignment |
| `alignment.orb_max_features` | 1500 | More features → better alignment, slower |
| `alignment.ransac_threshold` | 5.0 px | Lower → stricter matching |
| `augmentation.num_augmentations_per_image` | 25 | Controls total augmented dataset size |

---

## Project structure

```
himalaya_label_detection/
├── config/
│   ├── pipeline_config.yaml     # Sizes, thresholds, paths
│   └── roi_config.yaml          # ROI coordinates (set by setup_golden_master.py)
├── data/
│   ├── raw/good/                # ← YOUR 80 GOOD IMAGES GO HERE
│   ├── raw/bad/                 # ← YOUR 50 BAD IMAGES GO HERE
│   └── processed/               # Output — copy this to training machine
├── golden_master/               # Set by setup_golden_master.py
├── scripts/
│   ├── setup_golden_master.py   # Step 2 above
│   ├── verify_alignment.py      # Step 3 above
│   └── run_preprocessing.py     # Step 4 above
├── src/
│   ├── preprocessing/
│   │   ├── align.py             # LabelAligner (ORB + Homography)
│   │   ├── roi_extractor.py     # ROIExtractor (functional ROIs)
│   │   └── augment.py           # ImageAugmentor (albumentations)
│   └── utils/
│       └── image_utils.py       # load/save/resize helpers
└── requirements.txt
```
