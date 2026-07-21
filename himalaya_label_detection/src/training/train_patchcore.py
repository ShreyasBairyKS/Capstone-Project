"""
train_patchcore.py
──────────────────
PatchCore training wrapper using the anomalib library.

Why PatchCore for this dataset (80 good / 50 bad, unannotated):
  PatchCore builds a coreset memory bank from patch features extracted
  from good images only. There is no gradient-based training — "fitting"
  is essentially a nearest-neighbour index build over one forward pass.
  This makes it robust with small datasets: 80 images augmented to ~2050
  is sufficient for the coreset to cover normal label appearance.

  At inference, anomaly score = distance to the nearest normal patch in
  the memory bank. Any patch far from all normal patches (tear, ink blob,
  wrinkle, smear) receives a high anomaly score automatically, without
  ever having seen a defective sample during training.

Training data expected layout (created by scripts/train_models.py):
    data/anomalib_ready/full_label/
        good/    ← augmented good images (~2050)
        bad/     ← aligned bad images (50, for threshold calibration only)

Model output:
    models/full_label/    ← anomalib Lightning checkpoint
"""

import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def prepare_anomalib_dataset(
    augmented_good_dir: Path,
    aligned_bad_dir: Path,
    anomalib_dir: Path,
    dataset_name: str = "full_label",
) -> Path:
    """
    Symlink (or copy) images into the anomalib Folder datamodule layout.

    anomalib Folder expects:
        <root>/
            good/    ← normal training images
            bad/     ← abnormal images (used for test / threshold eval only)

    Returns:
        Path to the dataset root directory.
    """
    dataset_root = anomalib_dir / dataset_name
    good_out = dataset_root / "good"
    bad_out  = dataset_root / "bad"

    good_out.mkdir(parents=True, exist_ok=True)
    bad_out.mkdir(parents=True, exist_ok=True)

    _link_or_copy(augmented_good_dir, good_out)
    _link_or_copy(aligned_bad_dir,   bad_out)

    n_good = len([p for p in good_out.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    n_bad  = len([p for p in bad_out.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXTENSIONS])

    print(f"[Dataset] {dataset_name}: {n_good} good  |  {n_bad} bad")
    return dataset_root


def _link_or_copy(src_dir: Path, dst_dir: Path) -> None:
    """Symlink each image from src into dst; falls back to copy on Windows."""
    for src in src_dir.iterdir():
        if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def train_patchcore(
    dataset_root: Path,
    dataset_name: str,
    output_dir: Path,
    image_size: tuple[int, int] = (256, 256),
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
) -> None:
    """
    Train a PatchCore model using anomalib.

    Args:
        dataset_root:            Root containing good/ and bad/ subdirs.
        dataset_name:            Name used in anomalib logs.
        output_dir:              Directory to save the trained checkpoint.
        image_size:              (height, width) to resize images to during training.
        coreset_sampling_ratio:  Fraction of patch features to keep in memory bank.
                                 0.1 = 10% of all patches (sufficient for ~2000 images).
                                 Lower = faster inference + lower RAM; slightly lower recall.
        num_neighbors:           k for k-NN lookup in the memory bank.
                                 9 is the PatchCore paper default.
    """
    try:
        from anomalib.data import Folder
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
    except ImportError as exc:
        raise ImportError(
            "anomalib is not installed on this machine.\n"
            "Install on the training machine with:\n"
            "  pip install anomalib==1.0.0"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = Folder(
        name=dataset_name,
        root=dataset_root.parent,     # anomalib expects root = parent of good/bad dirs
        normal_dir=str(dataset_root / "good"),
        abnormal_dir=str(dataset_root / "bad"),
        image_size=image_size,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        task="classification",
    )

    model = Patchcore(
        # wide_resnet50_2 outperforms EfficientNet on industrial texture benchmarks
        # (MVTec LOCO, VisA) — the extra compute is justified on a training machine.
        backbone="wide_resnet50_2",
        # layer2: stride-8 features (fine spatial detail for scratches / ink blobs)
        # layer3: stride-16 features (semantic context for logo / barcode presence)
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
    )

    engine = Engine(
        accelerator="auto",           # GPU if available, CPU otherwise
        devices=1,
        default_root_dir=str(output_dir),
        max_epochs=1,                 # PatchCore: 1 epoch = build memory bank once
    )

    print(f"\n[PatchCore] Training on: {dataset_root}")
    print(f"[PatchCore] Output:      {output_dir}")
    engine.fit(model=model, datamodule=datamodule)

    # Test pass computes anomaly scores on bad images and sets the adaptive threshold
    print(f"\n[PatchCore] Running test pass (threshold estimation)...")
    engine.test(model=model, datamodule=datamodule)

    print(f"\n[PatchCore] Done — checkpoint saved in: {output_dir}")
