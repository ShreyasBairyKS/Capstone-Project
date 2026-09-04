"""
Script 05: Train Nano Student (yolo11n-seg) with Knowledge Distillation

Loss = alpha * L_gt  +  beta * L_KD  +  gamma * L_feat
  L_gt   : Standard YOLO ground truth loss (box + cls + mask)
  L_KD   : KL-Divergence on class logits (temperature T=4)
  L_feat : MSE on PANet neck feature maps (channel-aligned via adapter convs)

Weights (alpha, beta, gamma) = (0.4, 0.5, 0.1)  |  Temperature T = 4

Optimizations & Safeguards:
  - GPU VRAM check + auto batch-size fallback (64 -> 32 -> 16 -> 8)
  - Pre-flight checks: teacher weights, soft_labels directory, dataset yaml
  - Auto-resume from last checkpoint if training interrupted
  - Reproducible training via fixed random seeds
  - Graceful SIGINT/SIGTERM handler: prints resume instructions
  - NaN loss detection callback: halts training on corrupted gradients
  - AMP (Automatic Mixed Precision): enabled via amp=True
  - Teacher model explicitly frozen before training starts
  - CUDA cache cleared after loading teacher (frees memory for student batch)
  - Post-training verification: falls back to last.pt if best.pt missing
  - Training log saved to runs/student_yolo11n_distilled/train_student.log
  - Soft label fallback: uses uniform distribution if .npy file missing
"""

import sys
import signal
import random
import gc
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_YAML    = r"D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\data.yaml"
TEACHER_WEIGHTS = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
TEACHER_FALLBACK= r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\last.pt"
SOFT_LABELS_DIR = r"D:\Yolo Dataset\YOLO_TrainingScripts\soft_labels"
PROJECT_DIR     = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs"
RUN_NAME        = "student_yolo11n_distilled"
MODEL_BASE      = "yolo11n-seg.pt"

# ─── Distillation Hyperparameters ─────────────────────────────────────────────
TEMPERATURE = 4.0   # T: higher = softer distributions
ALPHA       = 0.4   # GT loss weight
BETA        = 0.5   # KL Divergence (response-level KD) weight
GAMMA       = 0.1   # Feature MSE (feature-level KD) weight

# ─── Training Config ──────────────────────────────────────────────────────────
SEED            = 42
BATCH_FALLBACKS = [64, 32, 16, 8]

# ─── Class Imbalance Weights ──────────────────────────────────────────────────
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
CLS_WEIGHT    = round(min(AVG_IMBALANCE, 4.0), 2)


# ─── Logging Setup ────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_student.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("StudentDistillation")
    logger.info(f"Log file: {log_file}")
    return logger


logger_global = None


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger_global.info(f"[Seed] Random seeds set to {seed}")


# ─── GPU Check ───────────────────────────────────────────────────────────────
def check_gpu(logger) -> float:
    if not torch.cuda.is_available():
        logger.error("[GPU] CUDA not available!")
        sys.exit(1)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"[GPU] {gpu_name} | VRAM: {vram_gb:.1f} GB")
    return vram_gb


# ─── Preflight Checks ────────────────────────────────────────────────────────
def preflight_checks(logger) -> str:
    """
    Validates:
      1. DATASET_YAML exists
      2. Teacher weights exist (best.pt, falls back to last.pt)
      3. Soft labels directory exists and has .npy files

    Returns the teacher weights path to use.
    """
    errors = []

    # Dataset YAML
    if not Path(DATASET_YAML).exists():
        errors.append(f"data.yaml not found: {DATASET_YAML}")
    else:
        logger.info(f"[Preflight] data.yaml: OK")

    # Teacher weights
    teacher_path = TEACHER_WEIGHTS
    if not Path(teacher_path).exists():
        if Path(TEACHER_FALLBACK).exists():
            logger.warning(f"[Preflight] best.pt missing, using last.pt: {TEACHER_FALLBACK}")
            teacher_path = TEACHER_FALLBACK
        else:
            errors.append(
                f"Teacher weights not found.\n"
                f"  Expected: {TEACHER_WEIGHTS}\n"
                f"  Fallback: {TEACHER_FALLBACK}\n"
                f"  Run 02_train_teacher.py first."
            )
    else:
        size_mb = Path(teacher_path).stat().st_size / 1e6
        logger.info(f"[Preflight] Teacher weights: {Path(teacher_path).name} ({size_mb:.1f} MB) OK")

    # Soft labels
    soft_dir = Path(SOFT_LABELS_DIR)
    if not soft_dir.exists():
        logger.warning(
            f"[Preflight] soft_labels/ not found: {soft_dir}\n"
            f"  KD will use uniform distributions (no teacher soft labels).\n"
            f"  Run 04_generate_soft_labels.py for best results."
        )
    else:
        n_npy = len(list(soft_dir.glob("*.npy")))
        if n_npy == 0:
            logger.warning(f"[Preflight] soft_labels/ is empty. KL loss will use uniform fallback.")
        else:
            logger.info(f"[Preflight] soft_labels/: {n_npy} .npy files OK")

    if errors:
        for err in errors:
            logger.error(f"[Preflight] {err}")
        sys.exit(1)

    return teacher_path


# ─── Checkpoint Resume Detection ─────────────────────────────────────────────
def detect_checkpoint(project: str, run_name: str, logger) -> tuple[str, bool]:
    run_dir = Path(project) / run_name
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        logger.info(f"[Resume] Found interrupted checkpoint: {last_pt}")
        return str(last_pt), True
    logger.info(f"[Start] No checkpoint found. Starting from {MODEL_BASE}")
    return MODEL_BASE, False


# ─── Signal Handler ──────────────────────────────────────────────────────────
def _handle_signal(sig, frame):
    last_pt = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
    print(f"\n[Signal] Training interrupted ({sig}).")
    if last_pt.exists():
        print(f"[Signal] Resume training by re-running: python 05_train_student_distill.py")
        print(f"[Signal] Checkpoint saved at: {last_pt}")
    sys.exit(0)


# ─── NaN Loss Callback ───────────────────────────────────────────────────────
def make_nan_callback(logger):
    def on_train_batch_end(trainer):
        loss = getattr(trainer, "loss", None)
        if loss is not None:
            val = loss.item() if hasattr(loss, "item") else float(loss)
            if not torch.isfinite(torch.tensor(val)):
                logger.error(f"[NaN] Non-finite loss: {val}. Stopping training.")
                trainer.epoch = trainer.epochs   # Force end
    return on_train_batch_end


# ─── Post-Training Verification ───────────────────────────────────────────────
def verify_output(logger) -> str:
    run_dir = Path(PROJECT_DIR) / RUN_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if best_pt.exists():
        logger.info(f"[Output] best.pt: {best_pt} ({best_pt.stat().st_size/1e6:.1f} MB)")
        return str(best_pt)
    elif last_pt.exists():
        logger.warning(f"[Output] best.pt missing! Falling back to last.pt: {last_pt}")
        return str(last_pt)
    logger.error("[Output] No weights found after training!")
    return ""


# ─── Feature Adapter ─────────────────────────────────────────────────────────
class FeatureAdapter(nn.Module):
    """1x1 Conv to align student channel dims to teacher channel dims."""
    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")

    def forward(self, x):
        return self.conv(x)


# ─── KL Divergence Loss ───────────────────────────────────────────────────────
def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_soft_probs: np.ndarray,
    temperature: float,
) -> torch.Tensor:
    """
    Temperature-scaled KL Divergence: L_KD = T^2 * KL(p_teacher || p_student)

    Args:
        student_logits    : (N, C) student class logits (raw, unscaled)
        teacher_soft_probs: (N, C) teacher softened probabilities (from .npy files)
        temperature       : float T used when generating soft labels

    Returns:
        Scalar KD loss scaled by T^2
    """
    if student_logits.shape[0] == 0:
        return torch.tensor(0.0, device=student_logits.device)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    t_probs = torch.tensor(teacher_soft_probs, dtype=torch.float32, device=student_logits.device)

    # Guard against shape mismatch
    if t_probs.shape != student_log_probs.shape:
        return torch.tensor(0.0, device=student_logits.device)

    kl = F.kl_div(student_log_probs, t_probs, reduction="batchmean", log_target=False)
    return (temperature ** 2) * kl


# ─── Feature MSE Loss ─────────────────────────────────────────────────────────
def feature_distillation_loss(
    student_feats: list,
    teacher_feats: list,
    adapters: nn.ModuleList,
) -> torch.Tensor:
    """
    MSE between channel-adapted student features and frozen teacher features.

    L_feat = mean over layers( ||Adapter(F_student) - F_teacher||^2_F )
    """
    total = torch.tensor(0.0, device=student_feats[0].device if student_feats else "cpu")
    count = 0

    for s_feat, t_feat, adapter in zip(student_feats, teacher_feats, adapters):
        try:
            adapted = adapter(s_feat)
            if adapted.shape[-2:] != t_feat.shape[-2:]:
                adapted = F.interpolate(adapted, size=t_feat.shape[-2:],
                                        mode="bilinear", align_corners=False)
            total = total + F.mse_loss(adapted, t_feat.detach())
            count += 1
        except Exception:
            continue   # Skip any level that errors (shape mismatch, etc.)

    return total / max(count, 1)


# ─── Custom Distillation Trainer ─────────────────────────────────────────────
class DistillationTrainer(SegmentationTrainer):
    """
    Ultralytics SegmentationTrainer extended with KD loss.
    L = alpha * L_gt  +  beta * L_KD  +  gamma * L_feat
    """

    def __init__(self, teacher_model, soft_labels_dir, adapters, cfg=None,
                 overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.teacher         = teacher_model
        self.soft_labels_dir = Path(soft_labels_dir)
        self.adapters        = nn.ModuleList(adapters)
        # Freeze teacher
        self.teacher.model.eval()
        for p in self.teacher.model.parameters():
            p.requires_grad = False

    def _move_adapters(self):
        """Move adapters to training device after trainer initialises device."""
        self.adapters = self.adapters.to(self.device)

    def _load_soft_labels(self, img_paths: list, num_classes: int) -> np.ndarray:
        """
        Load teacher soft probabilities for each image in the batch.
        Falls back to uniform distribution if .npy file not found.
        """
        batch_soft = []
        uniform = np.ones(num_classes, dtype=np.float32) / num_classes

        for p in img_paths:
            npy_path = self.soft_labels_dir / f"{Path(p).stem}.npy"
            try:
                if npy_path.exists():
                    data = np.load(str(npy_path))
                    if len(data) > 0:
                        row = data[:, :num_classes].mean(axis=0)
                        # Ensure valid probability distribution
                        row = np.clip(row, 1e-7, 1.0)
                        row = row / row.sum()
                        batch_soft.append(row)
                    else:
                        batch_soft.append(uniform)
                else:
                    batch_soft.append(uniform)
            except Exception:
                batch_soft.append(uniform)

        return np.array(batch_soft, dtype=np.float32)

    def criterion(self, preds, batch):
        """
        Override criterion to inject KD losses into the backward pass.
        """
        # ── Ground Truth Loss (box + cls + mask from Ultralytics) ─────────
        L_gt, loss_items = super().criterion(preds, batch)

        # ── Move adapters to device (first call only) ─────────────────────
        if next(self.adapters.parameters()).device != self.device:
            self._move_adapters()

        # ── Teacher Forward (no grad) ─────────────────────────────────────
        try:
            with torch.no_grad():
                imgs = batch["img"].to(self.device)
                # Move teacher to same device
                if next(self.teacher.model.parameters()).device != self.device:
                    self.teacher.model = self.teacher.model.to(self.device)
                teacher_out = self.teacher.model(imgs)
        except Exception as e:
            teacher_out = None

        # ── Response-Level KD: KL Divergence on class logits ─────────────
        L_KD = torch.tensor(0.0, device=self.device)
        try:
            # preds[0]: (B, 4 + nc + 32, num_anchors) at largest scale
            student_cls = preds[0][:, 4: 4 + self.model.nc, :]
            student_cls = student_cls.mean(dim=-1)  # (B, nc) avg over anchors

            img_paths  = batch.get("im_file", [f"img_{i}" for i in range(len(student_cls))])
            soft_probs = self._load_soft_labels(img_paths, self.model.nc)

            L_KD = kl_distillation_loss(student_cls, soft_probs, TEMPERATURE)
        except Exception:
            L_KD = torch.tensor(0.0, device=self.device)

        # ── Feature-Level KD: MSE on PANet neck outputs ───────────────────
        L_feat = torch.tensor(0.0, device=self.device)
        try:
            if teacher_out is not None and isinstance(preds, (list, tuple)):
                # preds[1] = list of feature maps from the student head
                # teacher_out[1] = corresponding teacher feature maps
                s_neck = preds[1] if isinstance(preds[1], list) else list(preds[1])
                t_neck = teacher_out[1] if isinstance(teacher_out[1], list) else list(teacher_out[1])

                if len(s_neck) > 0 and len(t_neck) > 0:
                    L_feat = feature_distillation_loss(
                        s_neck[:len(self.adapters)],
                        t_neck[:len(self.adapters)],
                        self.adapters,
                    )
        except Exception:
            L_feat = torch.tensor(0.0, device=self.device)

        # ── Combined Loss ─────────────────────────────────────────────────
        total_loss = ALPHA * L_gt + BETA * L_KD + GAMMA * L_feat
        return total_loss, loss_items


# ─── Training with Batch-Size Fallback ───────────────────────────────────────
def run_training_with_fallback(
    model_path: str,
    is_resume: bool,
    teacher_path: str,
    logger,
) -> bool:
    """Attempt student training; fall back to smaller batch on OOM."""

    for batch in BATCH_FALLBACKS:
        logger.info(f"[Train] Attempting batch={batch} ...")
        try:
            torch.cuda.empty_cache()
            gc.collect()

            # ── Load frozen teacher ───────────────────────────────────────
            logger.info(f"[Teacher] Loading {Path(teacher_path).name} ...")
            teacher_model = YOLO(teacher_path)
            teacher_model.model.eval()
            teacher_model.model = teacher_model.model.to("cuda:0")
            for p in teacher_model.model.parameters():
                p.requires_grad = False
            logger.info("[Teacher] Loaded and frozen.")

            # Clear reserved VRAM before loading student
            torch.cuda.empty_cache()

            # ── Feature adapters ──────────────────────────────────────────
            adapters = [
                FeatureAdapter(256, 512),
                FeatureAdapter(512, 1024),
            ]

            # ── Wire DistillationTrainer via closure ───────────────────────
            _teacher  = teacher_model
            _adapters = adapters
            _soft_dir = SOFT_LABELS_DIR

            class BoundDistillationTrainer(DistillationTrainer):
                def __init__(self, cfg=None, overrides=None, _callbacks=None):
                    super().__init__(
                        teacher_model=_teacher,
                        soft_labels_dir=_soft_dir,
                        adapters=_adapters,
                        cfg=cfg,
                        overrides=overrides,
                        _callbacks=_callbacks,
                    )

            # ── Student model ─────────────────────────────────────────────
            student_model = YOLO(model_path)
            student_model.add_callback("on_train_batch_end", make_nan_callback(logger))

            student_model.train(
                # ── Data ──────────────────────────────────────────────────
                data=DATASET_YAML,
                project=PROJECT_DIR,
                name=RUN_NAME,
                exist_ok=True,
                resume=is_resume,
                trainer=BoundDistillationTrainer,  # KD trainer active

                # ── Compute ───────────────────────────────────────────────
                device=0,
                batch=batch,
                workers=8,
                imgsz=640,
                amp=True,               # AMP: ~1.5x throughput on A5000

                # ── Training Schedule ──────────────────────────────────────
                epochs=180,
                patience=30,
                cos_lr=True,
                warmup_epochs=5,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                lr0=0.002,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                optimizer="AdamW",
                seed=SEED,

                # ── Loss Weights ───────────────────────────────────────────
                box=7.5,
                cls=CLS_WEIGHT,
                dfl=1.5,
                fl_gamma=1.5,

                # ── Regularization ─────────────────────────────────────────
                label_smoothing=0.0,    # Off: soft labels provide smoothing
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
                copy_paste=0.5,
                erasing=0.4,

                # ── Output ────────────────────────────────────────────────
                save=True,
                save_period=10,
                plots=True,
                val=True,
                verbose=True,
            )

            logger.info(f"[Train] Completed successfully with batch={batch}.")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"[OOM] batch={batch} OOM. Trying smaller batch...")
                torch.cuda.empty_cache()
                gc.collect()
                is_resume = False
                continue
            logger.error(f"[Error] RuntimeError: {e}")
            raise

        except Exception as e:
            logger.error(f"[Error] Unexpected: {e}")
            raise

    logger.error("[Train] All batch sizes failed. Check GPU memory.")
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────
def train_student():
    global logger_global

    log_dir = Path(PROJECT_DIR) / RUN_NAME
    logger_global = setup_logging(log_dir)
    logger = logger_global

    logger.info("=" * 65)
    logger.info("  STUDENT KNOWLEDGE DISTILLATION TRAINING: yolo11n-seg")
    logger.info(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Loss        : {ALPHA}*L_gt + {BETA}*L_KD + {GAMMA}*L_feat")
    logger.info(f"  Temperature : {TEMPERATURE}")
    logger.info(f"  cls weight  : {CLS_WEIGHT}")
    logger.info("=" * 65)

    # Reproducibility
    set_seeds(SEED)

    # GPU check
    check_gpu(logger)

    # Preflight: dataset + teacher weights + soft labels
    teacher_path = preflight_checks(logger)

    # Signal handlers
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Resume detection
    model_path, is_resume = detect_checkpoint(PROJECT_DIR, RUN_NAME, logger)

    # Train with OOM fallback
    success = run_training_with_fallback(model_path, is_resume, teacher_path, logger)

    if not success:
        sys.exit(1)

    # Verify output
    best_weights = verify_output(logger)

    logger.info("=" * 65)
    logger.info("  STUDENT TRAINING COMPLETE")
    logger.info(f"  Best weights : {best_weights}")
    logger.info(f"  Finished     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("  Next step -> Run: python 06_evaluate_and_compare.py")
    logger.info("=" * 65)

    return best_weights


if __name__ == "__main__":
    train_student()
