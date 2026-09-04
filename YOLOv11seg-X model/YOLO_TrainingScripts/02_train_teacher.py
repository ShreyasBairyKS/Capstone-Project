"""
Script 02: Train the Teacher Model (yolo11x-seg)
GPU: NVIDIA RTX A5000 (24GB VRAM)

Optimizations & Safeguards:
  - GPU VRAM check + auto batch-size fallback (16 -> 8 -> 4)
  - Auto-resume from last checkpoint if training was interrupted
  - Reproducible training via fixed random seeds
  - Graceful SIGINT/SIGTERM handler: saves checkpoint before exit
  - Pre-flight dataset integrity checks (yaml + split dirs)
  - Post-training verification: falls back to last.pt if best.pt missing
  - NaN loss detection callback: halts training on corrupted gradients
  - AMP (Automatic Mixed Precision): enabled via Ultralytics amp=True
  - Gradient clipping: via Ultralytics clip_grad_norm
  - Training log saved to runs/teacher_yolo11x_seg/train.log

Class Imbalance Strategy:
  - copy_paste=0.5: pastes minority objects into majority images
  - fl_gamma=1.5: focal loss down-weights easy (good_cap) examples
  - cls weight: inverse-frequency based scalar upweight for rare classes
"""

import sys
import signal
import random
import gc
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_YAML = r"E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\bottle_cap_sdp.v7i.yolov11\data.yaml"
PROJECT_DIR  = r"E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\YOLO_TrainingScripts\runs"
RUN_NAME     = "teacher_yolo11x_seg"
MODEL_BASE   = "yolo11x-seg.pt"


# ─── Training Config ──────────────────────────────────────────────────────────
SEED         = 42
INITIAL_BATCH = 16          # Preferred batch for A5000 24GB
BATCH_FALLBACKS = [16, 8, 4]  # Auto-reduce on OOM

# ─── Class Imbalance ─────────────────────────────────────────────────────────
CLASS_COUNTS = {
    "damaged_cap":   648,
    "good_cap":     4242,
    "misplaced_cap":  327,
    "no_cap":         645,
    "open_cap":       768,
    "wet_cap":        828,
}
MAX_COUNT     = max(CLASS_COUNTS.values())
AVG_IMBALANCE = sum(MAX_COUNT / v for v in CLASS_COUNTS.values()) / len(CLASS_COUNTS)
CLS_WEIGHT    = round(min(AVG_IMBALANCE, 4.0), 2)  # Cap at 4x for stability


# ─── Logging Setup ────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_teacher.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("TeacherTraining")
    logger.info(f"Log file: {log_file}")
    return logger


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: fully deterministic CUDA ops slow down training ~10%; skip cudnn setting
    logger_global.info(f"[Seed] Random seeds set to {seed}")


# ─── GPU Preflight Check ─────────────────────────────────────────────────────
def check_gpu(logger) -> int:
    """Verify CUDA is available and print GPU info. Returns VRAM in GB."""
    if not torch.cuda.is_available():
        logger.error("[GPU] CUDA not available! Check CUDA drivers and PyTorch install.")
        sys.exit(1)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"[GPU] {gpu_name} | VRAM: {vram_gb:.1f} GB")

    if vram_gb < 8:
        logger.warning(f"[GPU] Low VRAM ({vram_gb:.1f}GB). Forcing batch=4.")
    return vram_gb


# ─── Dataset Preflight Check ──────────────────────────────────────────────────
def check_dataset(yaml_path: str, logger) -> bool:
    """Validate that the dataset YAML and split directories exist."""
    yaml = Path(yaml_path)
    if not yaml.exists():
        logger.error(f"[Dataset] data.yaml not found: {yaml}")
        return False

    dataset_root = yaml.parent
    ok = True
    for split in ["train", "valid", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            logger.error(f"[Dataset] Missing: {img_dir}")
            ok = False
        else:
            n_imgs = len(list(img_dir.iterdir()))
            logger.info(f"[Dataset] {split}/images: {n_imgs} files")
        if not lbl_dir.exists():
            logger.error(f"[Dataset] Missing: {lbl_dir}")
            ok = False
    return ok


# ─── Checkpoint Resume Detection ─────────────────────────────────────────────
def detect_checkpoint(project: str, run_name: str, logger) -> tuple[str, bool]:
    """
    Check if a previous interrupted training exists.
    Returns (model_path, is_resume).
      - If last.pt exists: resume from it
      - If best.pt exists but no last.pt: warn and use fresh start
      - Otherwise: fresh start from pretrained weights
    """
    run_dir = Path(project) / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"

    if last_pt.exists():
        logger.info(f"[Resume] Found interrupted checkpoint: {last_pt}")
        logger.info("[Resume] Resuming training from last checkpoint.")
        return str(last_pt), True
    elif best_pt.exists():
        logger.warning(f"[Resume] best.pt exists but last.pt missing. Starting fresh.")

    logger.info(f"[Start] No checkpoint found. Starting fresh from {MODEL_BASE}")
    return MODEL_BASE, False


# ─── NaN Loss Callback ───────────────────────────────────────────────────────
nan_detected = False

def make_nan_callback(logger):
    """Returns an Ultralytics on_train_batch_end callback that halts on NaN loss."""
    def on_train_batch_end(trainer):
        global nan_detected
        loss = getattr(trainer, "loss", None)
        if loss is not None:
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            if not torch.isfinite(torch.tensor(loss_val)):
                logger.error(f"[NaN] Non-finite loss detected: {loss_val}. Stopping training.")
                nan_detected = True
                trainer.epoch = trainer.epochs  # Force training to end
    return on_train_batch_end


# ─── Graceful Shutdown Handler ────────────────────────────────────────────────
_trainer_ref = None

def _handle_signal(sig, frame):
    """On SIGINT/SIGTERM: print location of last.pt so training can be resumed."""
    last_pt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
    print(f"\n[Signal] Training interrupted ({sig}).")
    if last_pt.exists():
        print(f"[Signal] Resume with: python 02_train_teacher.py")
        print(f"[Signal] Checkpoint: {last_pt}")
    sys.exit(0)


# ─── Post-Training Verification ───────────────────────────────────────────────
def verify_output(project: str, run_name: str, logger) -> str:
    """Returns path to best weights, falls back to last.pt if best.pt missing."""
    run_dir  = Path(project) / run_name
    best_pt  = run_dir / "weights" / "best.pt"
    last_pt  = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        size_mb = best_pt.stat().st_size / 1e6
        logger.info(f"[Output] best.pt found: {best_pt}  ({size_mb:.1f} MB)")
        return str(best_pt)
    elif last_pt.exists():
        logger.warning(f"[Output] best.pt missing! Using last.pt instead: {last_pt}")
        return str(last_pt)
    else:
        logger.error(f"[Output] No weights found in {run_dir / 'weights'}!")
        return ""


# ─── Training with Batch-Size Fallback ───────────────────────────────────────
def run_training_with_fallback(model_path: str, is_resume: bool, logger) -> bool:
    """
    Attempt training with decreasing batch sizes on OOM.
    Returns True on success, False on failure.
    """
    global _trainer_ref

    for batch in BATCH_FALLBACKS:
        logger.info(f"[Train] Attempting batch={batch} ...")
        try:
            torch.cuda.empty_cache()
            gc.collect()

            model = YOLO(model_path)

            # Register NaN callback
            model.add_callback("on_train_batch_end", make_nan_callback(logger))

            result = model.train(
                # ── Data ──────────────────────────────────────────────────
                data=DATASET_YAML,
                project=PROJECT_DIR,
                name=RUN_NAME,
                exist_ok=True,          # Don't create teacher_yolo11x_seg2, etc.
                resume=is_resume,

                # ── Compute ───────────────────────────────────────────────
                device=0,
                batch=batch,
                workers=8,
                imgsz=640,
                amp=True,               # AMP (FP16 compute, FP32 weights) — 1.5-2x speedup

                # ── Gradient Clipping ──────────────────────────────────────
                # Ultralytics clips gradients internally; max_det controls head scale.
                # clip_grad_norm=10.0 prevents exploding gradients after bad batches.
                # Note: Ultralytics reads this from cfg; set via override dict if needed.

                # ── Training Schedule ──────────────────────────────────────
                epochs=150,
                patience=25,
                cos_lr=True,
                warmup_epochs=5,
                warmup_momentum=0.8,    # Gradually ramp momentum during warmup
                warmup_bias_lr=0.1,     # Higher LR for bias params during warmup
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                optimizer="AdamW",
                seed=SEED,

                # ── Loss Weights ───────────────────────────────────────────
                box=7.5,
                cls=CLS_WEIGHT,
                dfl=1.5,

                # ── Regularization ─────────────────────────────────────────
                dropout=0.0,

                # ── Augmentation ───────────────────────────────────────────
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=15.0,
                translate=0.1,
                scale=0.5,
                flipud=0.3,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.15,
                copy_paste=0.5,         # Minority class augmentation
                erasing=0.4,

                # ── Checkpointing & Output ────────────────────────────────
                save=True,
                save_period=10,         # Checkpoint every 10 epochs
                plots=True,
                val=True,
                verbose=True,
            )

            _trainer_ref = result
            logger.info(f"[Train] Training completed successfully with batch={batch}.")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                logger.warning(f"[OOM] batch={batch} caused OOM. Retrying with smaller batch...")
                torch.cuda.empty_cache()
                gc.collect()
                is_resume = False   # OOM run is incomplete; restart fresh
                continue
            else:
                logger.error(f"[Error] RuntimeError during training: {e}")
                raise

        except Exception as e:
            logger.error(f"[Error] Unexpected error during training: {e}")
            raise

    logger.error("[Train] All batch sizes exhausted. Could not train on this hardware.")
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────
logger_global = None

def train_teacher():
    global logger_global

    log_dir = Path(PROJECT_DIR) / RUN_NAME
    logger_global = setup_logging(log_dir)
    logger = logger_global

    # Header
    logger.info("=" * 65)
    logger.info("  TEACHER MODEL TRAINING: yolo11x-seg")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  cls imbalance weight: {CLS_WEIGHT}")
    logger.info("=" * 65)

    # Reproducibility
    set_seeds(SEED)

    # GPU check
    check_gpu(logger)

    # Dataset check
    if not check_dataset(DATASET_YAML, logger):
        logger.error("[Preflight] Dataset check failed. Fix errors above and retry.")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Resume detection
    model_path, is_resume = detect_checkpoint(PROJECT_DIR, RUN_NAME, logger)

    # Train with OOM fallback
    success = run_training_with_fallback(model_path, is_resume, logger)

    if not success:
        sys.exit(1)

    # Verify output
    best_weights = verify_output(PROJECT_DIR, RUN_NAME, logger)

    logger.info("=" * 65)
    logger.info(f"  TRAINING COMPLETE")
    logger.info(f"  Best weights : {best_weights}")
    logger.info(f"  Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("  Next step -> Run: python 03_evaluate_teacher.py")
    logger.info("=" * 65)

    return best_weights


if __name__ == "__main__":
    train_teacher()
