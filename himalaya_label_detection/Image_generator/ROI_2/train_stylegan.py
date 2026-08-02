"""
train_stylegan.py
─────────────────
Launches StyleGAN2-ADA training for ROI_2 good label images.

Prerequisites:
    1. StyleGAN2-ADA PyTorch repo cloned at:
       <project_root>/stylegan2-ada-pytorch/
       (clone command in TRAINING_GUIDE.md)

    2. Dataset ZIP prepared by prepare_dataset.py

    3. (Recommended) Pre-trained weights downloaded by download_pretrained.py

Hardware target: NVIDIA RTX A5000 (24 GB VRAM)
Expected training time: ~2–3 hours to convergence at 256×256

Usage (from project root on remote PC):
    python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py

    # Resume from latest snapshot:
    python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py --resume

    # Override kimg target (default 1000):
    python himalaya_label_detection/Image_generator/ROI_2/train_stylegan.py --kimg 1600
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[3]
THIS_DIR      = Path(__file__).parent
STYLEGAN_REPO = PROJECT_ROOT / "stylegan2-ada-pytorch"
DATASET_ZIP   = THIS_DIR / "dataset" / "roi2_good_256.zip"
OUTPUT_DIR    = THIS_DIR / "training_runs"
PRETRAINED    = THIS_DIR / "pretrained" / "transfer_ffhq256.pkl"

# Training hyperparameters (tuned for ROI_2 small dataset on A5000)
CFG = dict(
    gpus       = 1,
    batch      = 8,          # safe for 24 GB VRAM at 256×256; increase to 16 if stable
    gamma      = 8,          # R1 regularisation — default for 256px
    mirror     = 0,          # no horizontal flip (text would be mirrored)
    aug        = "ada",      # Adaptive Discriminator Augmentation — critical for small data
    augpipe    = "bgcfnc",   # full ADA augmentation pipeline
    target     = 0.6,        # ADA strength target (keep between 0.4–0.7)
    snap       = 10,         # save snapshot every 10 ticks (~2000 images)
    metrics    = "fid50k_full",   # use FID to monitor training quality
    kimg       = 1000,       # total training kimages — adjust with --kimg
)
# ─────────────────────────────────────────────────────────────────────────────


def build_cmd(resume_pkl: Optional[str], kimg_override: Optional[int]) -> List[str]:
    if not STYLEGAN_REPO.exists():
        print(f"❌  StyleGAN2-ADA repo not found at: {STYLEGAN_REPO}")
        print("    Run:  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git")
        sys.exit(1)

    if not DATASET_ZIP.exists():
        print(f"❌  Dataset ZIP not found at: {DATASET_ZIP}")
        print("    Run:  python prepare_dataset.py  first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = dict(CFG)
    if kimg_override is not None:
        cfg["kimg"] = kimg_override

    # Build --resume argument
    if resume_pkl:
        resume_arg = resume_pkl
    elif PRETRAINED.exists():
        print(f"  [Transfer] Using pre-trained weights: {PRETRAINED.name}")
        resume_arg = str(PRETRAINED)
    else:
        print("  [WARN] No pre-trained weights found. Training from scratch.")
        print("  Run download_pretrained.py for faster convergence.")
        resume_arg = "noresume"

    cmd = [
        sys.executable,
        str(STYLEGAN_REPO / "train.py"),
        f"--outdir={OUTPUT_DIR}",
        f"--data={DATASET_ZIP}",
        f"--gpus={cfg['gpus']}",
        f"--batch={cfg['batch']}",
        f"--gamma={cfg['gamma']}",
        f"--mirror={cfg['mirror']}",
        f"--aug={cfg['aug']}",
        f"--augpipe={cfg['augpipe']}",
        f"--target={cfg['target']}",
        f"--snap={cfg['snap']}",
        f"--metrics={cfg['metrics']}",
        f"--kimg={cfg['kimg']}",
        f"--resume={resume_arg}",
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from the latest snapshot in training_runs/")
    parser.add_argument("--resume-pkl", default=None,
                        help="Explicit path to .pkl snapshot to resume from")
    parser.add_argument("--kimg", type=int, default=None,
                        help="Total training kimages (default: 1000)")
    args = parser.parse_args()

    resume_pkl = args.resume_pkl
    if args.resume and resume_pkl is None:
        # Scope to training snapshots only — avoids picking up pretrained .pkl
        snaps = sorted(OUTPUT_DIR.rglob("network-snapshot-*.pkl"))
        if snaps:
            resume_pkl = str(snaps[-1])
            print(f"  [Resume] Found snapshot: {snaps[-1].name}")
        else:
            print("  [Resume] No snapshot found — starting from pretrained/scratch.")

    cmd = build_cmd(resume_pkl, args.kimg)
    print("\nLaunching StyleGAN2-ADA training:")
    print("  " + " \\\n    ".join(cmd))
    print()
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
