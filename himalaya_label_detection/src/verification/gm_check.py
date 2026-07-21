"""
gm_check.py
───────────
Golden Master similarity check using EfficientNet-B0 embeddings
and cosine similarity.

Why embeddings + cosine similarity rather than pixel SSIM:
  Pixel SSIM is too sensitive to acceptable printing variation — minor
  tonal differences, slight ink density shift between print runs.
  EfficientNet-B0 embeddings compress the image into a 1280-dim feature
  vector where such minor variations are absorbed, but structural
  differences (missing logo, wrong artwork, gross misalignment) remain
  large enough to drop the cosine similarity below the threshold.

The golden master embedding is computed once and saved to disk.
All inference calls compare against that stored vector.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GMCheckResult:
    similarity: float      # Cosine similarity 0.0–1.0
    passed:     bool
    threshold:  float


class GoldenMasterChecker:
    """
    Computes and compares EfficientNet-B0 embeddings for golden master verification.

    Two usage modes:

    Build mode (run once after setup_golden_master.py):
        checker = GoldenMasterChecker()
        checker.build_embedding(
            "golden_master/golden_master.jpg",
            "models/golden_master_embedding.pt",
        )

    Inference mode:
        checker = GoldenMasterChecker("models/golden_master_embedding.pt")
        result  = checker.score(aligned_bgr_image)
    """

    def __init__(self, embedding_path: Optional[str | Path] = None):
        self._model: Optional[torch.nn.Module] = None   # lazy-loaded
        self._gm_embedding: Optional[torch.Tensor] = None

        if embedding_path is not None:
            p = Path(embedding_path)
            if p.exists():
                self._gm_embedding = torch.load(
                    p, map_location="cpu", weights_only=True
                )

    # ── Private ──────────────────────────────────────────────────────────────

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            import torchvision.models as models
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            backbone.classifier = torch.nn.Identity()   # remove head → 1280-dim embedding
            backbone.eval()
            self._model = backbone
        return self._model

    def _preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        from PIL import Image
        rgb    = bgr_image[:, :, ::-1].copy()
        pil    = Image.fromarray(rgb)
        tensor = TF.to_tensor(TF.resize(pil, [224, 224]))
        tensor = TF.normalize(tensor,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0)

    def _embed(self, bgr_image: np.ndarray) -> torch.Tensor:
        model  = self._get_model()
        tensor = self._preprocess(bgr_image)
        with torch.no_grad():
            return model(tensor)    # (1, 1280)

    # ── Public API ───────────────────────────────────────────────────────────

    def build_embedding(
        self,
        golden_master_path: str | Path,
        save_path: str | Path,
    ) -> None:
        """Compute and save the golden master embedding. Run once."""
        import cv2
        image = cv2.imread(str(golden_master_path))
        if image is None:
            raise FileNotFoundError(f"Golden master not found: {golden_master_path}")
        emb = self._embed(image)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, save_path)
        self._gm_embedding = emb
        print(f"[GM] Embedding saved → {save_path}")

    def score(
        self,
        aligned_image: np.ndarray,
        threshold: float = 0.90,
    ) -> GMCheckResult:
        """
        Compare an aligned label image against the stored golden master embedding.

        Args:
            aligned_image: Aligned BGR label image.
            threshold:     Minimum cosine similarity to pass.

        Returns:
            GMCheckResult with similarity score and pass/fail.

        Raises:
            RuntimeError: If build_embedding() has not been called yet.
        """
        if self._gm_embedding is None:
            raise RuntimeError(
                "Golden master embedding not found.\n"
                "Run: python scripts/train_models.py --build-gm-embedding"
            )
        emb = self._embed(aligned_image)
        sim = F.cosine_similarity(self._gm_embedding, emb).item()
        return GMCheckResult(
            similarity=float(sim),
            passed=float(sim) >= threshold,
            threshold=threshold,
        )
