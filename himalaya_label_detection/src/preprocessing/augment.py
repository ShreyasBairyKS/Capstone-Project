"""
augment.py
Augmentation pipeline for the 80 good label images.

Why augmentation is critical here:
- 80 images is too few for PatchCore / EfficientAD to cover the full
  range of normal printing variation.
- Augmentations simulate lighting drift, minor camera jitter, surface
  texture variation, and compression artefacts — all of which are present
  in real production but should NOT be flagged as defects.
- We deliberately avoid augmentations that could create defect-like
  artefacts (large blurs, heavy distortion, coarse noise) to prevent
  teaching the model that defects are "normal".
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.core.composition import Compose


def build_augmentation_pipeline() -> Compose:
    """
    Build the augmentation pipeline.

    All transforms are kept subtle to simulate real production variation
    without creating synthetic defects.

    Transform rationale:
    - RandomBrightnessContrast: LED ring brightness can drift ±10% over a shift.
    - GaussNoise: Camera sensor noise is always present at low levels.
    - GaussianBlur: Slight focus variation from label-to-camera distance change.
    - ElasticTransform: Simulates slight surface warp as tube rotates under camera.
    - ShiftScaleRotate: Minor mechanical jitter in conveyor positioning.
    - ColorJitter: Ink density variation between print runs (hue/saturation drift).
    - ImageCompression: Simulates JPEG artefacts from camera image buffers.
    - HorizontalFlip: Valid for symmetric sections of the label.
      (Disable if your label is asymmetric end-to-end.)
    - CoarseDropout: Simulates dust specks on lens; kept very small so it
      does NOT simulate defects — only 1–3 tiny 4×4 px patches at a time.
    """
    return A.Compose([
        # Lighting variation
        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.85
        ),

        # Sensor noise
        A.GaussNoise(
            var_limit=(2.0, 15.0),
            mean=0,
            p=0.5
        ),

        # Minor focus jitter
        A.GaussianBlur(
            blur_limit=(3, 3),   # only 3×3 kernel — keep edges sharp
            p=0.3
        ),

        # Surface warp (tube curvature)
        A.ElasticTransform(
            alpha=18,
            sigma=6,
            alpha_affine=0,      # no affine component — only elastic warp
            border_mode=cv2.BORDER_REPLICATE,
            p=0.3
        ),

        # Camera/conveyor jitter
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.02,
            rotate_limit=2,      # max ±2° rotation
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5
        ),

        # Ink density / color drift
        A.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.08,
            hue=0.015,
            p=0.4
        ),

        # Camera buffer JPEG artefacts
        A.ImageCompression(
            quality_lower=88,
            quality_upper=100,
            p=0.2
        ),

        # Tiny dust specks on lens (NOT defects — keep very small)
        A.CoarseDropout(
            max_holes=2,
            max_height=4,
            max_width=4,
            min_holes=1,
            fill_value=0,
            p=0.15
        ),
    ])


class ImageAugmentor:
    """
    Applies the augmentation pipeline to a set of good images,
    generating multiple augmented versions of each.

    Usage:
        augmentor = ImageAugmentor(num_per_image=25)
        augmentor.augment_directory("data/raw/good",
                                    "data/processed/augmented/good",
                                    copy_originals=True)
    """

    def __init__(self,
                 num_per_image: int = 25,
                 seed: Optional[int] = 42):
        """
        Args:
            num_per_image: Number of augmented versions to generate per input image.
            seed:          Random seed for reproducibility. Pass None for true random.
        """
        self.num_per_image = num_per_image
        self.pipeline = build_augmentation_pipeline()
        if seed is not None:
            np.random.seed(seed)

    def augment_image(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Generate `num_per_image` augmented copies of a single image.

        Args:
            image: Input BGR numpy array.

        Returns:
            List of augmented BGR numpy arrays.
        """
        results = []
        for _ in range(self.num_per_image):
            augmented = self.pipeline(image=image)["image"]
            results.append(augmented)
        return results

    def augment_directory(self,
                          input_dir: str | Path,
                          output_dir: str | Path,
                          copy_originals: bool = True,
                          extensions: Optional[set] = None) -> int:
        """
        Augment all images in `input_dir` and save to `output_dir`.

        Args:
            input_dir:       Folder containing good images.
            output_dir:      Destination folder.
            copy_originals:  If True, also copy the un-augmented originals.
            extensions:      File extensions to process.

        Returns:
            Total number of images saved.
        """
        from src.utils.image_utils import list_images, load_image, save_image
        from tqdm import tqdm

        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_dir, extensions)
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {input_dir}")

        total_saved = 0

        for img_path in tqdm(image_paths, desc="Augmenting images"):
            image = load_image(img_path)

            if copy_originals:
                out_path = output_dir / img_path.name
                save_image(image, out_path)
                total_saved += 1

            augmented_list = self.augment_image(image)
            stem = img_path.stem
            suffix = img_path.suffix

            for i, aug_img in enumerate(augmented_list):
                out_name = f"{stem}_aug{i:03d}{suffix}"
                save_image(aug_img, output_dir / out_name)
                total_saved += 1

        return total_saved
