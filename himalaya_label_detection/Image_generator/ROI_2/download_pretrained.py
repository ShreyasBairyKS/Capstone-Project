"""
download_pretrained.py
──────────────────────
Downloads the recommended transfer-learning base model for StyleGAN2-ADA.

We use 'transfer-learning-source-nets' from NVlabs:
    ffhq256.pkl — 256×256, Flickr Faces HQ
    (best starting point for photorealistic texture / colour images)

Saves to:
    himalaya_label_detection/Image_generator/ROI_2/pretrained/transfer_ffhq256.pkl

Usage:
    python himalaya_label_detection/Image_generator/ROI_2/download_pretrained.py
"""

import urllib.request
from pathlib import Path

PRETRAINED_DIR = Path(__file__).parent / "pretrained"

# NVlabs official hosted model — 256×256 FFHQ
MODEL_URL  = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
MODEL_NAME = "transfer_ffhq256.pkl"


def main():
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRETRAINED_DIR / MODEL_NAME

    if out_path.exists():
        print(f"✅  Already downloaded: {out_path}")
        return

    print(f"Downloading FFHQ-256 pretrained weights (~350 MB)...")
    print(f"  URL: {MODEL_URL}")
    print(f"  Destination: {out_path}\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} / {total_size/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(MODEL_URL, str(out_path), reporthook=progress)
    print(f"\n\n✅  Saved: {out_path}")
    print("Next step → run: train_stylegan.py")


if __name__ == "__main__":
    main()
