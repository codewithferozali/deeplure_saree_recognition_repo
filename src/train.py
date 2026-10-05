"""
Training Engine for DeepLure Saree Recognition.
Features:
  - Discriminative Layer-Wise Learning Rates (Backbone vs. Metric Head)
  - Automatic Mixed Precision (AMP)
  - Cosine Annealing LR Schedule
  - P-K Balanced Metric Learning Sampler
  - Early Stopping based on Retrieval Recall@1
  - Validation-Driven Checkpoint Calibration
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import Config
from src.dataset import SareeDataset, SareeItem, build_gallery_query_split
from src.augmentations import get_train_transform, get_eval_transform, MultiViewColorInvariantTransform
from src.sampler import BalancedPKBatchSampler
from src.losses import SupConLoss, BatchHardTripletLoss, CombinedMetricLoss
from src.model import SareeDesignModel
from src.metrics import compute_identification_metrics
from src.verification import SareeVerifier
from src.utils import save_checkpoint, plot_training_curves

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(model: SareeDesignModel, cfg: Config, steps_per_epoch: int):
    """
    Sets up AdamW with discriminative learning rates:
    Backbone receives lower LR to preserve generic representations while
    the projection head receives full LR to rapidly map motifs onto the metric sphere.
    """
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    if hasattr(model, "pool") and isinstance(model.pool, nn.Module):
        head_params.extend(list(model.pool.parameters()))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.LEARNING_RATE * cfg.BACKBONE_LR_RATIO},
        {"params": head_params, "lr": cfg.LEARNING_RATE},
    ], weight_decay=cfg.WEIGHT_DECAY)

    total_steps = cfg.EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.LEARNING_RATE * 1e-2,
    )

    return optimizer, scheduler


def extract_embeddings_for_items(
    model: nn.Module,
    items: List[SareeItem],
    image_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Extracts L2-normalized embeddings for a list of SareeItems."""
    model.eval()
    transform = get_eval_transform(image_size=image_size)
    dataset = SareeDataset(items, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embs_list = []
    designs = []
    colorways = []

    with torch.no_grad():
        for images, labels, metas in loader:
            images = images.to(device)
            embs = model(images)
            embs_list.append(embs.cpu())
            designs.extend(metas["design_id"])
            colorways.extend(metas["colorway_id"])

    if len(embs_list) == 0:
        return torch.empty(0, model.embedding_dim), [], []

    return torch.cat(embs_list, dim=0), designs, colorways


def evaluate_retrieval_split(
    model: nn.Module,
    items: List[SareeItem],
    cfg: Config,
) -> Dict[str, float]:
    """Evaluates Recall@K, MRR, and mAP on a disjoint gallery/query split."""
    gallery_items, query_items = build_gallery_query_split(items, seed=cfg.SEED)

    gallery_embs, gallery_designs, _ = extract_embeddings_for_items(
        model, gallery_items, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, device=cfg.DEVICE
    )
    query_embs, query_designs, _ = extract_embeddings_for_items(
        model, query_items, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, device=cfg.DEVICE
    )

    if len(gallery_embs) == 0 or len(query_embs) == 0:
        return {"Recall@1": 0.0, "Recall@5": 0.0, "MRR": 0.0, "mAP": 0.0}

    metrics = compute_identification_metrics(
        query_embeddings=query_embs.numpy(),
        query_labels=query_designs,
        gallery_embeddings=gallery_embs.numpy(),
        gallery_labels=gallery_designs,
        top_k_list=cfg.TOP_K_LIST,
    )
    return metrics


def train_model(
    cfg: Config,
    train_items: List[SareeItem],
    val_items: List[SareeItem],
    model: Optional[SareeDesignModel] = None,
    transform_mode: str = "strong_color_invariant",
    use_multiview: bool = False,
    run_name: str = "final_model",
) -> Tuple[SareeDesignModel, Dict[str, Any]]:
    """
    Executes end-to-end training loop for the Saree Design Recognition model.
    """
    logger.info(f"Starting training run: {run_name} (Transform: {transform_mode}, MultiView: {use_multiview})")

    # 1. Instantiate Model if not provided
    if model is None:
        model = SareeDesignModel(
            backbone=cfg.BACKBONE,
            embedding_dim=cfg.EMBEDDING_DIM,
            pretrained=cfg.PRETRAINED,
            dropout=cfg.DROPOUT,
            pooling_type=cfg.POOLING_TYPE,
            gem_p=cfg.GEM_P,
        )
    model.to(cfg.DEVICE)

    # 2. Setup Data Pipeline
    if use_multiview:
        train_transform = MultiViewColorInvariantTransform(image_size=cfg.IMAGE_SIZE, num_views=2)
    else:
        train_transform = get_train_transform(image_size=cfg.IMAGE_SIZE, mode=transform_mode)

    train_dataset = SareeDataset(train_items, transform=train_transform, is_multiview=use_multiview)
    
    # Target class labels for PK sampling
    train_labels = [train_dataset.design_to_idx[item.design_id] for item in train_items]
    sampler = BalancedPKBatchSampler(
        labels=train_labels,
        p_classes=cfg.P_CLASSES,
        k_samples=cfg.K_SAMPLES,
        seed=cfg.SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.DEVICE.startswith("cuda")),
    )

    # 3. Setup Loss Function
    if cfg.LOSS_TYPE == "supcon":
        criterion = SupConLoss(temperature=cfg.TEMPERATURE)
    elif cfg.LOSS_TYPE == "triplet":
        criterion = BatchHardTripletLoss(margin=cfg.TRIPLET_MARGIN)
    else:
        criterion = CombinedMetricLoss(
            temperature=cfg.TEMPERATURE,
            triplet_margin=cfg.TRIPLET_MARGIN,
            supcon_weight=cfg.SUPCON_WEIGHT,
            triplet_weight=cfg.TRIPLET_WEIGHT,
        )

    # 4. Setup Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.USE_AMP)

    # 5. Training Loop
    history = {
        "train_loss": [],
        "val_recall_at_1": [],
        "val_recall_at_5": [],
        "val_mrr": [],
    }

    best_val_recall = -1.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = cfg.CHECKPOINT_DIR / f"{run_name}_latest.pth"
    best_checkpoint_path = cfg.CHECKPOINT_DIR / f"{run_name}_best.pth"

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_data, labels, _ in train_loader:
            labels = labels.to(cfg.DEVICE)
            optimizer.zero_grad()

            with autocast(enabled=cfg.USE_AMP):
                if use_multiview and isinstance(batch_data, list):
                    # batch_data is list of view tensors [B, 3, H, W]
                    view_embs = [model(v.to(cfg.DEVICE)) for v in batch_data]
                    # Stack into [B, n_views, D]
                    features = torch.stack(view_embs, dim=1)
                else:
                    images = batch_data.to(cfg.DEVICE)
                    features = model(images)

                loss = criterion(features, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(1, num_batches)
        history["train_loss"].append(avg_train_loss)

        # Validation every epoch
        val_metrics = evaluate_retrieval_split(model, val_items, cfg)
        val_r1 = val_metrics.get("Recall@1", 0.0)
        val_r5 = val_metrics.get("Recall@5", 0.0)
        val_mrr = val_metrics.get("MRR", 0.0)

        history["val_recall_at_1"].append(val_r1)
        history["val_recall_at_5"].append(val_r5)
        history["val_mrr"].append(val_mrr)

        logger.info(
            f"Epoch {epoch:02d}/{cfg.EPOCHS:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Recall@1: {val_r1:.4f} | "
            f"Val Recall@5: {val_r5:.4f} | "
            f"Val MRR: {val_mrr:.4f}"
        )

        is_best = val_r1 > best_val_recall
        if is_best:
            best_val_recall = val_r1
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall_at_1": val_r1,
                "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
            },
            filepath=checkpoint_path,
            is_best=is_best,
            best_filepath=best_checkpoint_path,
        )

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch} (Best Val Recall@1: {best_val_recall:.4f})")
            break

    # Load best checkpoint before returning
    if best_checkpoint_path.exists():
        _ = torch.load(str(best_checkpoint_path), map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(_["model_state_dict"])
        logger.info(f"Restored best weights from epoch {best_epoch} with Val Recall@1 = {best_val_recall:.4f}")

    # Plot training curve
    curve_path = cfg.FIGURES_DIR / f"{run_name}_loss_curve.png"
    plot_training_curves(history, curve_path)

    # Calibrate verification threshold on validation set
    val_dataset = SareeDataset(val_items, transform=get_eval_transform(cfg.IMAGE_SIZE))
    best_thresh, verify_metrics = SareeVerifier.calibrate_threshold_on_dataset(
        model, val_dataset, device=cfg.DEVICE, num_pairs=1000, seed=cfg.SEED
    )
    logger.info(f"Calibrated Verification Threshold: {best_thresh:.4f} (Val F1: {verify_metrics.get('f1', 0.0):.4f})")

    results_summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_recall_at_1": best_val_recall,
        "calibrated_threshold": best_thresh,
        "validation_verification_metrics": verify_metrics,
        "history": history,
    }

    return model, results_summary
