"""
generate_images.py
──────────────────
Samples from a trained StyleGAN2-ADA network to produce synthetic "good"
ROI_2 label crops.

Outputs images to:
    himalaya_label_detection/Image_generator/ROI_2/generated/

Usage:
    # Generate 500 images from the best snapshot:
    python himalaya_label_detection/Image_generator/ROI_2/generate_images.py --count 500

    # Specify a particular snapshot:
    python himalaya_label_detection/Image_generator/ROI_2/generate_images.py \
        --network training_runs/<run_dir>/network-snapshot-001000.pkl \
        --count 1000

    # Generate and resize to 256×256 for EfficientAD:
    python himalaya_label_detection/Image_generator/ROI_2/generate_images.py \
        --count 500 --output-size 256
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
THIS_DIR      = Path(__file__).parent
STYLEGAN_REPO = PROJECT_ROOT / "stylegan2-ada-pytorch"
RUNS_DIR      = THIS_DIR / "training_runs"
OUT_DIR       = THIS_DIR / "generated"


def find_latest_pkl() -> Path:
    snaps = sorted(RUNS_DIR.rglob("network-snapshot-*.pkl"))
    if not snaps:
        raise FileNotFoundError(
            f"No trained snapshot found in {RUNS_DIR}.\n"
            "Run train_stylegan.py first."
        )
    return snaps[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network",     default=None, help="Path to .pkl snapshot")
    parser.add_argument("--count",       type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output-size", type=int, default=256,
                        help="Final image size in px (256 matches EfficientAD training)")
    parser.add_argument("--truncation",  type=float, default=0.7,
                        help="Truncation psi (0.5–1.0). Lower = more typical images.")
    args = parser.parse_args()

    # Add StyleGAN2 repo to path
    if not STYLEGAN_REPO.exists():
        print(f"❌  StyleGAN2-ADA repo not found at: {STYLEGAN_REPO}")
        sys.exit(1)
    sys.path.insert(0, str(STYLEGAN_REPO))

    import torch
    import dnnlib
    import legacy

    network_pkl = args.network or str(find_latest_pkl())
    print(f"  Network : {network_pkl}")
    print(f"  Count   : {args.count}")
    print(f"  Truncation psi: {args.truncation}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device  : {device}\n")

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()

    import cv2
    rng = np.random.default_rng(args.seed)
    saved = 0

    print(f"Generating {args.count} images...")
    batch_size = 8
    for start in range(0, args.count, batch_size):
        end   = min(start + batch_size, args.count)
        n     = end - start
        seeds = rng.integers(0, 2**31, size=n)
        zs    = torch.from_numpy(
            np.stack([np.random.RandomState(int(s)).randn(G.z_dim) for s in seeds])
        ).to(device).float()
        # Unconditional model requires a zero class-conditioning tensor
        cs = torch.zeros([n, G.c_dim], device=device)

        with torch.no_grad():
            imgs = G(zs, cs, truncation_psi=args.truncation, noise_mode="const")
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        for i, img_rgb in enumerate(imgs):
            img_bgr = img_rgb[:, :, ::-1]   # RGB → BGR for OpenCV
            h, w = img_bgr.shape[:2]
            if h != args.output_size or w != args.output_size:
                img_bgr = cv2.resize(img_bgr, (args.output_size, args.output_size),
                                     interpolation=cv2.INTER_LANCZOS4)
            out_path = OUT_DIR / f"syn_roi2_{saved:05d}.png"
            cv2.imwrite(str(out_path), img_bgr)
            saved += 1
            if saved % 50 == 0:
                print(f"  {saved}/{args.count} images saved...")

    print(f"\n✅  Done. {saved} synthetic images saved to:\n   {OUT_DIR}")
    print("\nNext steps:")
    print("  1. Review generated images visually — discard any artifacts.")
    print("  2. Copy approved images to data/rois/ROI_2/good/ for EfficientAD training.")
    print("  3. Proceed to Stage 2 (defect injection) for generating bad images.")


if __name__ == "__main__":
    main()
