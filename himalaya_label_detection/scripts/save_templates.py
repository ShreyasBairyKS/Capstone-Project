"""
save_templates.py
─────────────────
Run ONCE to extract the 3 reference template patches from the sample crops.

These patches are used by TemplateAnchorFinder (src/preprocessing/anchor.py)
to locate each ROI region in a new full image using cv2.matchTemplate.

Usage:
    python save_templates.py
    
Inputs (sample crops — one of each region type):
    1.bmp  = ROI_LOGO sample      (Himalaya logo + Winter Defense text)
    3.bmp  = ROI_INGREDIENT sample (Jojoba/Wheat Germ/Almond Oil text)
    9.bmp  = ROI_ADDRESS sample   (address, regulatory, net vol.)

Outputs:
    himalaya_label_detection/config/patch_logo.png
    himalaya_label_detection/config/patch_ingredient.png
    himalaya_label_detection/config/patch_address.png
"""

import cv2
from pathlib import Path

CONFIG_DIR = Path("himalaya_label_detection/config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ── ROI_LOGO patch ────────────────────────────────────────────────────────────
# Use center-top of logo crop where "Himalaya" text is most distinctive
logo = cv2.imread("1.bmp", cv2.IMREAD_GRAYSCALE)
assert logo is not None, "Cannot read 1.bmp — run from project root"
logo_patch = logo[100:400, 200:700]   # 300×500 px
cv2.imwrite(str(CONFIG_DIR / "patch_logo.png"), logo_patch)
print(f"patch_logo.png saved: {logo_patch.shape}")

# ── ROI_INGREDIENT patch ──────────────────────────────────────────────────────
# Use the ingredient text area (Jojoba Oil / Wheat Germ / Almond Oil lines)
ingr = cv2.imread("3.bmp", cv2.IMREAD_GRAYSCALE)
assert ingr is not None, "Cannot read 3.bmp — run from project root"
ingr_h, ingr_w = ingr.shape
ingr_patch = ingr[100:min(400, ingr_h-10), 100:min(600, ingr_w-10)]
cv2.imwrite(str(CONFIG_DIR / "patch_ingredient.png"), ingr_patch)
print(f"patch_ingredient.png saved: {ingr_patch.shape}")

# ── ROI_ADDRESS patch ─────────────────────────────────────────────────────────
# Use the address text block (most distinctive part)
addr = cv2.imread("9.bmp", cv2.IMREAD_GRAYSCALE)
assert addr is not None, "Cannot read 9.bmp — run from project root"
addr_h, addr_w = addr.shape
addr_patch = addr[200:min(550, addr_h-10), 50:min(650, addr_w-10)]
cv2.imwrite(str(CONFIG_DIR / "patch_address.png"), addr_patch)
print(f"patch_address.png saved: {addr_patch.shape}")

print()
print(f"All 3 patches saved to {CONFIG_DIR}/")
print("Next: run validate_anchors.py to verify they work on all images")
