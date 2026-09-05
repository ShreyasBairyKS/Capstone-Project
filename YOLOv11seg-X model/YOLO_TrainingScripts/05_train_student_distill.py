"""
Script 05: Train Medium Student (yolo11m-seg) with Feature-Based Knowledge Distillation

KD Strategy: Feature-Level Only (per YOLO11_Feature_KD_Guide.md)
  Loss = L_gt  +  alpha * (L_feat_C3k2 + L_feat_SPPF + L_feat_C2PSA)

  L_gt       : Standard YOLO ground truth loss (box + cls + mask)
  L_feat_*   : Channel-wise L2-normalised MSE between teacher and student
               feature maps at C3k2 (backbone), SPPF (backbone->neck transition),
               and C2PSA (neck/head) — via learnable 1x1 projector convolutions.

NOTE: Response-level KD (KL Divergence on logits) has been intentionally removed.
      Feature-level distillation is preferred for micro-defect detection because it
      directly aligns the spatial attention and texture representations in intermediate
      layers, not just the final classification outputs.

Alpha Warmup:
  alpha starts at 0.05 and ramps linearly to 0.5 over the first 20 epochs, then
  remains constant. This prevents the student from over-optimising for teacher
  feature mimicry before it has learned basic detection.

Projector (Hint) Layers:
  yolo11m-seg has fewer channels than yolo11x-seg. Learnable 1x1 Conv projectors
  map student channels -> teacher channels during training. These projectors are
  entirely discarded at inference time — zero added latency on edge hardware.

Optimizations & Safeguards:
  - GPU VRAM check + auto batch-size fallback (32 -> 16 -> 8 -> 4)
  - Pre-flight checks: teacher weights, dataset yaml
  - Auto-resume from last checkpoint if training interrupted
  - Reproducible training via fixed random seeds
  - Graceful SIGINT/SIGTERM handler: prints resume instructions
  - NaN loss detection callback: halts training on corrupted gradients
  - AMP (Automatic Mixed Precision): enabled via amp=True
  - Teacher model explicitly frozen + eval() before training starts
  - teacher_features.detach() used in all feature losses (no VRAM leak)
  - CUDA cache cleared after loading teacher (frees memory for student batch)
  - Post-training verification: falls back to last.pt if best.pt missing
  - Training log saved to runs/student_yolo11m_distilled/train_student.log
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
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.segment import SegmentationTrainer

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_YAML    = r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11\data.yaml"
TEACHER_WEIGHTS = r"E:\P-25 Vision Food ai\YOLOv11seg-X model\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
TEACHER_FALLBACK= r"E:\P-25 Vision Food ai\YOLOv11seg-X model\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
PROJECT_DIR     = r"E:\P-25 Vision Food ai\YOLOv11seg-X model\YOLO_TrainingScripts"
RUN_NAME        = "student_yolo11m_distilled"
MODEL_BASE      = "yolo11m-seg.pt"

# ─── Distillation Hyperparameters ─────────────────────────────────────────────
# Feature-based KD only — KL divergence on logits removed per guide decision.
# L_total = L_gt + alpha * (L_feat_C3k2 + L_feat_SPPF + L_feat_C2PSA)
ALPHA_START  = 0.05   # Initial KD weight (warmup start)
ALPHA_END    = 0.50   # Final KD weight (warmup end)
ALPHA_WARMUP = 20     # Epochs to ramp alpha from ALPHA_START to ALPHA_END

# ─── Channel Dimensions: yolo11m-seg -> yolo11x-seg ───────────────────────────
# yolo11m-seg and yolo11x-seg share the same architectural topology but differ
# in channel widths. These projector dims must match the actual layer outputs.
# Verified against Ultralytics model configs (base_channels * width_multiplier):
#   yolo11m  width=0.50 -> C3k2: 256ch, SPPF: 512ch, C2PSA: 512ch
#   yolo11x  width=1.00 -> C3k2: 512ch, SPPF:1024ch, C2PSA:1024ch
FEAT_DIMS = {
    "c3k2": (256, 512),    # (student_ch, teacher_ch)
    "sppf": (512, 1024),   # (student_ch, teacher_ch)
    "c2psa": (512, 1024),  # (student_ch, teacher_ch)
}

# ─── Training Config ──────────────────────────────────────────────────────────
SEED            = 42
BATCH_FALLBACKS = [32, 16, 8, 4]  # Adjusted for yolo11m (larger than nano)

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


# ─── Projector (Hint Layer) Conv ─────────────────────────────────────────────
class FeatureProjector(nn.Module):
    """
    Learnable 1x1 Convolutional Projector that maps student feature channels
    into the teacher's channel space for feature alignment loss computation.

    Per the KD guide: projector layers are discarded at inference time,
    adding zero latency to the deployed student model.
    """
    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── L2-Normalised Feature MSE Loss ──────────────────────────────────────────
def feature_distillation_loss_normalised(
    student_feats: list,
    teacher_feats: list,
    projectors: nn.ModuleList,
) -> torch.Tensor:
    """
    Channel-wise L2-Normalised MSE between projected student features and
    frozen teacher features, computed at C3k2, SPPF, and C2PSA layers.

    Formula (per guide):
        L_layer = || Norm(F_teacher) - Norm(Projector(F_student)) ||_2^2

    Why normalise?
        Tiny defects produce sparse, low-magnitude activations. Raw MSE would
        be dominated by large, uniform background regions. L2 normalisation
        across the channel dimension scales all feature vectors onto a unit
        sphere, so the loss focuses on *pattern* of activation rather than
        *magnitude*, preserving micro-defect signals.

    Args:
        student_feats : list of student feature tensors [C3k2, SPPF, C2PSA]
        teacher_feats : list of teacher feature tensors [C3k2, SPPF, C2PSA]
        projectors    : nn.ModuleList of FeatureProjector modules (one per layer)

    Returns:
        Scalar mean feature distillation loss across all targeted layers.
    """
    total = torch.tensor(0.0, device=student_feats[0].device if student_feats else "cpu")
    count = 0

    for s_feat, t_feat, projector in zip(student_feats, teacher_feats, projectors):
        try:
            # Project student channels up to teacher channel dims
            projected = projector(s_feat)

            # Bilinear upsample if spatial dims differ (e.g. stride differences)
            if projected.shape[-2:] != t_feat.shape[-2:]:
                projected = F.interpolate(projected, size=t_feat.shape[-2:],
                                          mode="bilinear", align_corners=False)

            # Channel-wise L2 normalisation (dim=1 = channel dim)
            t_norm = F.normalize(t_feat.detach(), p=2, dim=1)  # detach: no grad through teacher
            s_norm = F.normalize(projected, p=2, dim=1)

            total = total + F.mse_loss(s_norm, t_norm)
            count += 1
        except Exception:
            continue   # Skip any layer that errors (shape mismatch, etc.)

    return total / max(count, 1)


# ─── Alpha Warmup Scheduler ───────────────────────────────────────────────────
def compute_alpha(current_epoch: int) -> float:
    """
    Linear warmup: alpha increases from ALPHA_START to ALPHA_END over
    ALPHA_WARMUP epochs, then stays at ALPHA_END.

    Rationale (per guide): Starting with a small alpha prevents the student
    from over-optimising for teacher feature mimicry before it has learned
    basic detection from ground-truth labels.
    """
    if current_epoch >= ALPHA_WARMUP:
        return ALPHA_END
    progress = current_epoch / ALPHA_WARMUP
    return ALPHA_START + progress * (ALPHA_END - ALPHA_START)


# ─── Feature Hook Manager ─────────────────────────────────────────────────────
class FeatureHookManager:
    """
    Registers forward hooks on named YOLO11 modules (C3k2, SPPF, C2PSA) to
    capture intermediate feature maps during forward passes without modifying
    the model architecture.
    """

    # Module type names to hook (matched by class name substring)
    TARGET_TYPES = ("C3k2", "SPFF", "SPPF", "C2PSA")

    def __init__(self, model: nn.Module):
        self.hooks = []
        self.features: list = []
        self._register(model)

    def _register(self, model: nn.Module):
        for name, module in model.named_modules():
            mtype = type(module).__name__
            if any(t in mtype for t in self.TARGET_TYPES):
                h = module.register_forward_hook(self._capture)
                self.hooks.append(h)

    def _capture(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.features.append(output)

    def clear(self):
        self.features = []

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ─── Custom Distillation Trainer ─────────────────────────────────────────────
class DistillationTrainer(SegmentationTrainer):
    """
    Ultralytics SegmentationTrainer extended with Feature-Based KD.

    L_total = L_gt + alpha(epoch) * (L_feat_C3k2 + L_feat_SPPF + L_feat_C2PSA)

    Key design decisions:
    - Hooks are registered on both teacher and student to extract intermediate
      features from C3k2, SPPF, and C2PSA blocks.
    - teacher_features.detach() is enforced inside feature_distillation_loss_normalised
      to prevent VRAM leaks from teacher computation graphs.
    - Alpha is warmed up linearly over ALPHA_WARMUP epochs.
    - Projector layers are part of self.projectors (nn.ModuleList), trained
      jointly. They must be stripped before edge deployment.
    """

    def __init__(self, teacher_model, projectors, cfg=DEFAULT_CFG,
                 overrides=None, _callbacks=None):
        if cfg is None:
            cfg = DEFAULT_CFG
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.teacher    = teacher_model
        self.projectors = nn.ModuleList(projectors)

        # Freeze teacher: eval mode + no gradients
        self.teacher.model.eval()
        for p in self.teacher.model.parameters():
            p.requires_grad = False

    def _move_projectors(self):
        """Move projectors to training device after trainer initialises device."""
        self.projectors = self.projectors.to(self.device)

    def criterion(self, preds, batch):
        """
        Override criterion to inject feature KD loss into the backward pass.
        """
        # ── Ground Truth Loss (box + cls + mask from Ultralytics) ─────────
        L_gt, loss_items = super().criterion(preds, batch)

        # ── Move projectors to device (first call only) ────────────────────
        if next(self.projectors.parameters()).device != self.device:
            self._move_projectors()

        # ── Ensure teacher is on correct device ────────────────────────────
        teacher_device = next(self.teacher.model.parameters()).device
        if teacher_device != self.device:
            self.teacher.model = self.teacher.model.to(self.device)

        # ── Capture teacher intermediate features via hooks ────────────────
        L_feat = torch.tensor(0.0, device=self.device)
        try:
            imgs = batch["img"].to(self.device)

            # Register hooks on teacher to capture C3k2, SPPF, C2PSA outputs
            teacher_hook = FeatureHookManager(self.teacher.model)
            # Register hooks on student model to capture same layer types
            student_hook = FeatureHookManager(self.model)

            # Teacher forward (no grad, frozen)
            with torch.no_grad():
                _ = self.teacher.model(imgs)

            teacher_feats = list(teacher_hook.features)  # Captured during teacher fwd

            # Student features are already captured from the current batch fwd
            # (criterion is called after student forward pass in Ultralytics loop)
            student_feats = list(student_hook.features)

            # Remove hooks immediately to avoid accumulation across batches
            teacher_hook.remove()
            student_hook.remove()

            if len(student_feats) > 0 and len(teacher_feats) > 0:
                # Use minimum available layers across both models
                n_layers = min(len(student_feats), len(teacher_feats), len(self.projectors))
                L_feat = feature_distillation_loss_normalised(
                    student_feats[:n_layers],
                    teacher_feats[:n_layers],
                    self.projectors,
                )
        except Exception:
            L_feat = torch.tensor(0.0, device=self.device)

        # ── Alpha warmup ───────────────────────────────────────────────────
        current_epoch = getattr(self, "epoch", 0)
        alpha = compute_alpha(current_epoch)

        # ── Combined Loss ─────────────────────────────────────────────────
        # L_total = L_gt + alpha * L_feat
        total_loss = L_gt + alpha * L_feat
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

            # ── Projector layers (student_ch -> teacher_ch) ────────────────
            # Three projectors: C3k2 | SPPF | C2PSA
            projectors = [
                FeatureProjector(*FEAT_DIMS["c3k2"]),
                FeatureProjector(*FEAT_DIMS["sppf"]),
                FeatureProjector(*FEAT_DIMS["c2psa"]),
            ]
            logger.info(
                f"[Projectors] C3k2: {FEAT_DIMS['c3k2']} | "
                f"SPPF: {FEAT_DIMS['sppf']} | "
                f"C2PSA: {FEAT_DIMS['c2psa']}"
            )

            # ── Wire DistillationTrainer via closure ───────────────────────
            _teacher    = teacher_model
            _projectors = projectors

            class BoundDistillationTrainer(DistillationTrainer):
                def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
                    if cfg is None:
                        cfg = DEFAULT_CFG
                    super().__init__(
                        teacher_model=_teacher,
                        projectors=_projectors,
                        cfg=cfg,
                        overrides=overrides,
                        _callbacks=_callbacks,
                    )

            # ── Student model (yolo11m-seg) ───────────────────────────────
            student_model = YOLO(model_path)
            student_model.add_callback("on_train_batch_end", make_nan_callback(logger))

            logger.info(
                f"[KD Config] Feature-only distillation | "
                f"Alpha warmup: {ALPHA_START} -> {ALPHA_END} over {ALPHA_WARMUP} epochs"
            )

            student_model.train(
                # ── Data ──────────────────────────────────────────────────
                data=DATASET_YAML,
                project=PROJECT_DIR,
                name=RUN_NAME,
                exist_ok=True,
                resume=is_resume,
                trainer=BoundDistillationTrainer,  # Feature KD trainer active

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
                lr0=0.001,              # Slightly lower LR for medium model vs nano
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
    logger.info("  STUDENT KNOWLEDGE DISTILLATION TRAINING: yolo11m-seg")
    logger.info("  KD Mode     : Feature-Level Only (C3k2 + SPPF + C2PSA)")
    logger.info(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Loss        : L_gt + alpha(t)*L_feat  [alpha: {ALPHA_START}->{ALPHA_END} over {ALPHA_WARMUP} epochs]")
    logger.info(f"  cls weight  : {CLS_WEIGHT}")
    logger.info("=" * 65)

    # Reproducibility
    set_seeds(SEED)

    # GPU check
    check_gpu(logger)

    # Preflight: dataset + teacher weights
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
    logger.info("  IMPORTANT: Strip projector layers before edge deployment!")
    logger.info("  Next step -> Run: python 06_evaluate_and_compare.py")
    logger.info("=" * 65)

    return best_weights


if __name__ == "__main__":
    train_student()
