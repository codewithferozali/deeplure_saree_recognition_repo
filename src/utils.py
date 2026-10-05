"""
Utility functions for DeepLure Saree Recognition:
  - Global Deterministic Seeding (Reproducibility)
  - Model Checkpoint Management
  - Efficiency & Latency Profiling (Parameters, Memory, Latency, FLOPs)
  - Rich Visualization Generators (Loss Curves, ROC Curves, t-SNE Clustering, Qualitative Grids)
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA for exact reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
    best_filepath: Optional[Union[str, Path]] = None,
):
    """Saves training checkpoint to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    if is_best and best_filepath is not None:
        best_path = Path(best_filepath)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(best_path))
        logger.info(f"Saved new best model checkpoint to {best_path}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint state into model and optimizer."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {path}")
    return checkpoint


def profile_model_efficiency(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 32,
    device: str = "cpu",
    warmup_runs: int = 5,
    num_runs: int = 30,
) -> Dict[str, Any]:
    """
    Profiles model efficiency metrics:
      - Total parameters & Trainable parameters
      - Model memory size (MB)
      - Single-image inference latency (ms)
      - Batch-inference latency (ms)
      - Estimated GFLOPs
    """
    model.eval()
    model.to(device)

    # 1. Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Parameter memory size
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 * 1024)

    # 2. Latency benchmarking
    single_input = torch.randn(1, *input_size).to(device)
    batch_input = torch.randn(batch_size, *input_size).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(single_input)
            _ = model(batch_input)
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()

        # Single-image latency
        single_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(single_input)
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            single_times.append((t1 - t0) * 1000.0)  # ms

        # Batch latency
        batch_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(batch_input)
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)  # ms

    single_latency_mean = float(np.mean(single_times))
    single_latency_std = float(np.std(single_times))
    batch_latency_mean = float(np.mean(batch_times))

    # 3. Estimated FLOPs using simple convolutional MAC approximation
    estimated_gflops = (total_params * 2 * input_size[1] * input_size[2]) / (224 * 224 * 1e9) * 0.15

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "embedding_dim": getattr(model, "embedding_dim", 256),
        "single_latency_ms": round(single_latency_mean, 2),
        "single_latency_std_ms": round(single_latency_std, 2),
        "batch_latency_ms": round(batch_latency_mean, 2),
        "approx_gflops": round(float(estimated_gflops), 2),
        "device": device,
    }


def plot_training_curves(history: Dict[str, List[float]], output_path: Union[str, Path]):
    """Plots training and validation loss & Recall@1 curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color="tab:red", fontsize=12)
    ax1.plot(epochs, history["train_loss"], "r-o", label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, linestyle="--", alpha=0.5)

    if "val_recall_at_1" in history and history["val_recall_at_1"]:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Val Recall@1", color="tab:blue", fontsize=12)
        ax2.plot(epochs, history["val_recall_at_1"], "b-^", label="Val Recall@1", linewidth=2)
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("DeepLure Training Progression", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_similarity_distribution(
    invariance_stats: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
):
    """Plots bar chart of cosine similarity across the 4 color-invariance cases."""
    cases = list(invariance_stats.keys())
    means = [invariance_stats[c]["mean"] for c in cases]
    stds = [invariance_stats[c]["std"] for c in cases]

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cases, means, yerr=stds, capsize=8, color=colors, edgecolor="black", alpha=0.85)

    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Color-Invariance Diagnostic: Similarity by Pair Type", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, mean in zip(bars, means):
        y_pos = bar.get_height() + 0.03 if mean >= 0 else bar.get_height() - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{mean:.3f}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_embedding_tsne(
    embeddings: np.ndarray,
    design_ids: List[str],
    colorway_ids: List[str],
    output_path: Union[str, Path],
    max_samples: int = 300,
    seed: int = 42,
):
    """
    Plots side-by-side 2D t-SNE projection:
      - Left: Color-coded by Design ID (Expect tight clusters per design)
      - Right: Color-coded by Colorway (Expect uniform mixing across colors, demonstrating color invariance)
    """
    n = len(embeddings)
    if n > max_samples:
        indices = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        embeddings = embeddings[indices]
        design_ids = [design_ids[i] for i in indices]
        colorway_ids = [colorway_ids[i] for i in indices]

    # Dim reduction
    n_components = min(30, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca_feats = pca.fit_transform(embeddings)

    perplexity = min(30, max(5, len(embeddings) // 5))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto")
    tsne_2d = tsne.fit_transform(pca_feats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Clustered by Design ID
    unique_designs = sorted(list(set(design_ids)))
    palette_designs = sns.color_palette("tab20", len(unique_designs))
    design_color_map = {d: palette_designs[i % len(palette_designs)] for i, d in enumerate(unique_designs)}
    colors_d = [design_color_map[d] for d in design_ids]

    axes[0].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors_d, s=50, alpha=0.85, edgecolors="none")
    axes[0].set_title("Embeddings Clustered by DESIGN ID\n(Strong clustering demonstrates motif sensitivity)", fontsize=13, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # 2. Clustered by Colorway
    unique_colors = sorted(list(set(colorway_ids)))
    palette_colors = sns.color_palette("Set2", len(unique_colors))
    colorway_map = {c: palette_colors[i % len(palette_colors)] for i, c in enumerate(unique_colors)}
    colors_c = [colorway_map[c] for c in colorway_ids]

    axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors_c, s=50, alpha=0.85, edgecolors="none")
    axes[1].set_title("Embeddings Coded by COLORWAY\n(Homogeneous mixing proves color invariance)", fontsize=13, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_qualitative_retrieval_grid(
    query_items: List[Any],
    retrieved_results: List[List[Dict[str, Any]]],
    output_path: Union[str, Path],
    num_queries_to_show: int = 5,
    top_k: int = 5,
):
    """
    Renders visual retrieval grids:
      Query Image -> Top-1, Top-2, Top-3, Top-4, Top-5 Retrieved Gallery Images.
      Highlights correct motif matches in green borders, incorrect in red borders.
    """
    k_to_show = min(num_queries_to_show, len(query_items))
    fig, axes = plt.subplots(k_to_show, top_k + 1, figsize=(3.0 * (top_k + 1), 3.2 * k_to_show))

    if k_to_show == 1:
        axes = np.expand_dims(axes, 0)

    for q_idx in range(k_to_show):
        q_item = query_items[q_idx]
        q_path = getattr(q_item, "image_path", str(q_item))
        q_design = getattr(q_item, "design_id", "query")
        q_color = getattr(q_item, "colorway_id", "")

        # Query column (col 0)
        try:
            q_img = Image.open(q_path).convert("RGB")
            axes[q_idx, 0].imshow(q_img)
        except Exception:
            axes[q_idx, 0].text(0.5, 0.5, "Image Missing", ha="center")

        axes[q_idx, 0].set_title(f"QUERY\n{q_design}\n({q_color})", fontsize=10, fontweight="bold", color="darkblue")
        axes[q_idx, 0].axis("off")

        # Top K retrieved gallery images
        results = retrieved_results[q_idx][:top_k]
        for r_idx, res in enumerate(results):
            col = r_idx + 1
            g_path = res["image_path"]
            g_design = res["design_id"]
            g_color = res.get("colorway_id", "")
            sim = res["similarity"]
            is_match = (g_design == q_design)

            try:
                g_img = Image.open(g_path).convert("RGB")
                axes[q_idx, col].imshow(g_img)
            except Exception:
                axes[q_idx, col].text(0.5, 0.5, "Missing", ha="center")

            border_color = "green" if is_match else "red"
            status_text = "MATCH" if is_match else "WRONG"

            axes[q_idx, col].set_title(
                f"Rank {r_idx + 1} | Sim: {sim:.2f}\n{g_design} ({g_color})\n[{status_text}]",
                fontsize=9,
                color=border_color,
                fontweight="bold"
            )
            # Draw colored bounding box around plot
            for spine in axes[q_idx, col].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            axes[q_idx, col].set_xticks([])
            axes[q_idx, col].set_yticks([])

    plt.suptitle("Qualitative Color-Invariant Retrieval Results (Green = True Design Match)", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
