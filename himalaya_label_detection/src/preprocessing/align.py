"""
align.py
Perspective correction using ORB keypoints + RANSAC homography.

Every incoming label image is warped into the coordinate system of the
golden master so that downstream ROI coordinates and anomaly models
see a geometrically consistent input.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from src.utils.image_utils import load_image, to_grayscale


@dataclass
class AlignmentResult:
    image: np.ndarray           # Warped image (same shape as golden master)
    homography: np.ndarray      # 3×3 homography matrix
    num_inliers: int            # Number of RANSAC inlier matches
    success: bool               # False if fewer than min_match_count inliers


class AlignmentError(Exception):
    """Raised when alignment cannot be performed reliably."""


class LabelAligner:
    """
    Aligns an incoming label image to a stored golden master image
    using ORB feature matching and RANSAC homography estimation.

    Why ORB + Homography:
    - ORB is fast (no GPU needed), patent-free, and robust for
      textured flat surfaces like printed labels.
    - Homography models all 8 degrees of freedom of a planar perspective
      transform: translation, rotation, scale, shear, and perspective tilt.
    - RANSAC rejects outlier matches caused by glare or partial occlusion.

    Usage:
        aligner = LabelAligner("golden_master/golden_master.jpg")
        result  = aligner.align(incoming_image)
        if result.success:
            process(result.image)
    """

    def __init__(self,
                 golden_master_path: str | Path,
                 target_size: Optional[Tuple[int, int]] = None,
                 orb_max_features: int = 1500,
                 match_keep_top: int = 300,
                 ransac_threshold: float = 5.0,
                 min_match_count: int = 10):
        """
        Args:
            golden_master_path: Path to the reference label image.
            target_size:        (width, height) to resize images before processing.
                                If None, images are used at their original size.
            orb_max_features:   Maximum ORB keypoints to detect per image.
            match_keep_top:     Keep only the top-N matches (by descriptor distance).
            ransac_threshold:   Maximum reprojection error (pixels) for RANSAC inliers.
            min_match_count:    Minimum inlier count for a valid alignment.
        """
        self.target_size = target_size
        self.ransac_threshold = ransac_threshold
        self.min_match_count = min_match_count
        self.match_keep_top = match_keep_top

        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=orb_max_features)

        # Brute-force matcher with Hamming distance (correct for ORB binary descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Load and cache golden master features
        golden_raw = load_image(golden_master_path, target_size=self.target_size)
        self.golden_bgr = golden_raw
        self.golden_gray = to_grayscale(golden_raw)
        self.golden_h, self.golden_w = self.golden_gray.shape[:2]

        self._kp_gold, self._desc_gold = self.orb.detectAndCompute(
            self.golden_gray, None
        )
        if self._desc_gold is None or len(self._kp_gold) < self.min_match_count:
            raise AlignmentError(
                f"Golden master has too few keypoints ({len(self._kp_gold or [])})."
                " Use an image with richer texture or increase orb_max_features."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, image: np.ndarray) -> AlignmentResult:
        """
        Align an incoming image to the golden master.

        Args:
            image: BGR (or grayscale) numpy array.

        Returns:
            AlignmentResult with the warped image, homography, inlier count,
            and a success flag.
        """
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        gray = to_grayscale(image)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None or len(kp) < self.min_match_count:
            return AlignmentResult(
                image=image, homography=np.eye(3),
                num_inliers=0, success=False
            )

        # Match and sort by descriptor distance
        matches = self.matcher.match(self._desc_gold, desc)
        matches = sorted(matches, key=lambda m: m.distance)
        matches = matches[:self.match_keep_top]

        if len(matches) < self.min_match_count:
            return AlignmentResult(
                image=image, homography=np.eye(3),
                num_inliers=len(matches), success=False
            )

        # Build point correspondences
        src_pts = np.float32(
            [self._kp_gold[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Estimate homography with RANSAC
        H, inlier_mask = cv2.findHomography(
            dst_pts, src_pts,
            cv2.RANSAC,
            self.ransac_threshold
        )

        if H is None:
            return AlignmentResult(
                image=image, homography=np.eye(3),
                num_inliers=0, success=False
            )

        num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        success = num_inliers >= self.min_match_count

        warped = cv2.warpPerspective(
            image, H, (self.golden_w, self.golden_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return AlignmentResult(
            image=warped,
            homography=H,
            num_inliers=num_inliers,
            success=success
        )

    def align_from_path(self, image_path: str | Path) -> AlignmentResult:
        """Convenience wrapper: load from disk, then align."""
        image = load_image(image_path)
        return self.align(image)

    @property
    def golden_master(self) -> np.ndarray:
        """Return the stored golden master image (BGR)."""
        return self.golden_bgr
