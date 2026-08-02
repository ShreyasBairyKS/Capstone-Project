# StyleGAN2-ADA Training Guide — ROI_2 (Good Image Generator)

> **Goal:** Train StyleGAN2-ADA on the ~77 real "good" ROI_2 crops to generate thousands of synthetic realistic good label images. These synthetic images expand the training set for EfficientAD and serve as clean backgrounds for Stage 2 defect injection.

**ROI_2 Content:** `Jojoba Oil · Wheat Germ · Almond Oil` ingredient bar + description paragraph (blue background, white/light text).

---

## Hardware Requirements

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) ✅ |
| RAM | 128 GB ✅ |
| Disk | ~5 GB free (model + dataset + outputs) |
| OS | Windows (PowerShell) or Linux |

**Expected training time:** ~2–3 hours at 256×256 with transfer learning.

---

## File Structure

```
himalaya_label_detection/Image_generator/ROI_2/
├── augment_dataset.py       ← Step 1A: Pre-training augmentation (optional)
├── prepare_dataset.py       ← Step 1B: Convert PNGs → StyleGAN ZIP format
├── download_pretrained.py   ← Step 2:  Download FFHQ-256 transfer weights
├── train_stylegan.py        ← Step 3:  Launch training
├── generate_images.py       ← Step 4:  Sample synthetic images from model
├── TRAINING_GUIDE.md        ← This file
├── pretrained/              ← Downloaded base weights (auto-created)
├── dataset/                 ← Prepared dataset ZIP (auto-created)
│   └── roi2_good_256.zip
├── training_runs/           ← StyleGAN training output (auto-created)
│   └── <run_dir>/
│       ├── network-snapshot-000200.pkl
│       ├── network-snapshot-001000.pkl  ← Best snapshot (final)
│       └── training-options.json
└── generated/               ← Synthetic output images (auto-created)
    └── syn_roi2_00000.png
```

---

## Step 0: Clone StyleGAN2-ADA

Run once from the **project root** on the remote PC:

```powershell
cd "E:\P-25 Vision Food ai"
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

> [!IMPORTANT]
> The repo must be at `<project_root>/stylegan2-ada-pytorch/`. All scripts expect this path.

### Install StyleGAN2-ADA dependencies

```powershell
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
# torch and torchvision should already be installed from EfficientAD training
```

---

## Step 1A: Pre-Training Augmentation (Recommended)

Because you only have ~77 real images, augmenting before training improves StyleGAN diversity.

```powershell
# From project root
python himalaya_label_detection/Image_generator/ROI_2/augment_dataset.py
```

**What it does:** Creates 5 augmented variants per image (brightness/contrast jitter, slight blur, JPEG artifacts, hue shift) → expands to ~462 images.

**Output:** Files prefixed with `aug_` written alongside originals in `data/rois/ROI_2/good/`.

> [!NOTE]
> Horizontal flip is disabled (`ALLOW_HFLIP = False`) because flipped text is unreadable and would confuse the model.

---

## Step 1B: Prepare Dataset

Converts all PNG crops (originals + augmented) into the ZIP format required by StyleGAN2-ADA.

```powershell
python himalaya_label_detection/Image_generator/ROI_2/prepare_dataset.py
```

**What it does:**
- Reads all images from `data/rois/ROI_2/good/`
- Letterboxes each to **256×256** (aspect ratio preserved, black padding)
- Writes to `dataset/images/`
- Creates `dataset.json` (unconditional labels)
- Zips everything into `dataset/roi2_good_256.zip`

**Expected output:**
```
Found 462 images in .../data/rois/ROI_2/good
  [  1/462] roi2_good_001.png → roi2_0000.png
  ...
✅  Dataset ready: .../dataset/roi2_good_256.zip
   Images: 462  |  Resolution: 256x256
```

---

## Step 2: Download Pre-Trained Weights

Transfer learning from FFHQ-256 dramatically speeds up convergence for photorealistic images.

```powershell
python himalaya_label_detection/Image_generator/ROI_2/download_pretrained.py
```

**Downloads:** `ffhq-res256-mirror-paper256-noaug.pkl` (~350 MB) to `pretrained/transfer_ffhq256.pkl`.

> [!TIP]
> If the NVIDIA CDN is slow, you can also download manually from:
> https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/

---

## Step 3: Train StyleGAN2-ADA

```powershell
python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py
```

### Key Hyperparameters (pre-configured)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--aug ada` | ADA | Adaptive augmentation — **critical** for small datasets |
| `--augpipe bgcfnc` | Full pipeline | All aug types: blur, geom, color, filter, noise, cutout |
| `--mirror 0` | No flip | Prevents mirrored text in output |
| `--batch 8` | 8 | Safe for 24 GB VRAM at 256px |
| `--gamma 8` | 8 | Standard R1 reg. for 256px |
| `--kimg 1000` | 1000 | ~1M training images total; adjust if needed |
| `--snap 10` | 10 | Save snapshot every 10 ticks (~20k images seen) |
| `--resume` | FFHQ-256 | Transfer learning base |

### To resume a stopped training run:

```powershell
python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py --resume
```

### To extend training:

```powershell
python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py --resume --kimg 2000
```

---

## Monitoring Training Quality

Training output is in `training_runs/<run_dir>/`. Key things to watch:

### 1. FID Score (Fréchet Inception Distance)
Logged to console every 10 ticks. **Lower is better.**

| FID | Interpretation |
|-----|----------------|
| < 20 | Excellent — images are realistic |
| 20–50 | Good — some artifacts visible |
| 50–100 | Mediocre — review generated grids |
| > 100 | Poor — training may need more time or data |

### 2. Visual Grid Snapshots
StyleGAN saves `fakes000200.png`, `fakes001000.png` etc. in the run directory.
Open these periodically to visually inspect quality.

### 3. ADA Augmentation Strength (`aug_p`)
Printed during training. Should stay between **0.4–0.8**. If it hits 1.0, your dataset may be too small or too repetitive.

---

## Step 4: Generate Synthetic Images

Once training is complete (or at any snapshot):

```powershell
# Generate 500 synthetic good images
python himalaya_label_detection/Image_generator/ROI_2/generate_images.py --count 500

# Generate 1000 images with tighter truncation (more typical/clean results)
python himalaya_label_detection/Image_generator/ROI_2/generate_images.py --count 1000 --truncation 0.6

# Use a specific snapshot
python himalaya_label_detection/Image_generator/ROI_2/generate_images.py \
    --network training_runs/<run_dir>/network-snapshot-001000.pkl \
    --count 500
```

**Output:** PNG files in `generated/syn_roi2_XXXXX.png` at 256×256.

### Truncation Psi Guide

| `--truncation` | Effect |
|----------------|--------|
| `0.5` | Very typical, conservative — less variety |
| `0.7` | **Recommended** — good balance of realism and diversity |
| `1.0` | Full diversity — may include some unusual samples |

---

## Step 5: Quality Review & Integration

1. **Visually review** generated images in the `generated/` folder.
2. **Discard** any that show mode collapse (repeated identical images), obvious artifacts, or unreadable text.
3. **Copy approved images** to `data/rois/ROI_2/good/` to expand the EfficientAD training set.
4. **Retrain EfficientAD** with the expanded dataset:

```powershell
python himalaya_label_detection/scripts/train_classifiers.py \
    --rois data/gray_scale_rois \
    --out models/gray_scale_rois
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `--batch` to 4 in `train_stylegan.py` |
| `No module named 'dnnlib'` | StyleGAN repo not cloned or wrong path |
| `Training stuck at FID > 200` | Add more real images or increase `--kimg` |
| `All generated images look identical` | Mode collapse — reduce truncation or retrain from earlier snapshot |
| `Generated images have grid artifacts` | Normal early in training — wait for more kimg |
| `download_pretrained.py fails` | Download the `.pkl` manually from the NVlabs CDN |

---

## Next Stage

Once satisfied with the generated good images, proceed to **Stage 2: Defect Injection**.

See: [`../../../docs/Image_generator.md`](../../../docs/Image_generator.md)
