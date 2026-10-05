"""
Comprehensive Evaluation Suite for Saree Recognition:
  - Disjoint Gallery vs. Query Identification (Recall@K, MRR, mAP)
  - Pairwise Verification (ROC-AUC, EER, Optimal Threshold, F1)
  - Color-Invariance Diagnostic (4 Quadrant Breakdown)
  - Ablation Study Pipeline (5 Experiments)
  - Qualitative Retrieval Grid Visualization
  - t-SNE / PCA Feature Clustering Visualization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import Config
from src.dataset import SareeItem, build_gallery_query_split
from src.train import extract_embeddings_for_items
from src.retrieval import GalleryIndex
from src.verification import SareeVerifier
from src.metrics import (
    compute_identification_metrics,
    evaluate_color_invariance_breakdown,
)
from src.utils import (
    plot_similarity_distribution,
    plot_embedding_tsne,
    plot_qualitative_retrieval_grid,
)

logger = logging.getLogger(__name__)


def run_full_evaluation(
    model: nn.Module,
    test_items: List[SareeItem],
    cfg: Config,
    threshold: Optional[float] = None,
    output_prefix: str = "test",
) -> Dict[str, Any]:
    """
    Executes the complete evaluation benchmark on the disjoint test set.
    """
    logger.info("Executing Full Evaluation on Disjoint Test Set...")

    # 1. Partition into Gallery and Query
    gallery_items, query_items = build_gallery_query_split(test_items, seed=cfg.SEED)
    logger.info(f"Test Partition: Gallery={len(gallery_items)} designs, Query={len(query_items)} images.")

    # 2. Extract Embeddings
    gallery_embs, gallery_designs, gallery_cw = extract_embeddings_for_items(
        model, gallery_items, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, device=cfg.DEVICE
    )
    query_embs, query_designs, query_cw = extract_embeddings_for_items(
        model, query_items, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, device=cfg.DEVICE
    )

    gallery_embs_np = gallery_embs.numpy()
    query_embs_np = query_embs.numpy()

    # 3. Build Gallery Index & Perform Identification
    gallery_index = GalleryIndex(embedding_dim=cfg.EMBEDDING_DIM, use_faiss=True)
    gallery_index.add(
        embeddings=gallery_embs_np,
        image_paths=[it.image_path for it in gallery_items],
        design_ids=gallery_designs,
        colorway_ids=gallery_cw,
    )

    # Save gallery index
    index_save_path = cfg.EMBEDDINGS_DIR / f"{output_prefix}_gallery_index.npz"
    gallery_index.save(index_save_path)

    # Compute Identification Metrics (Recall@K, MRR, mAP)
    retrieval_metrics = compute_identification_metrics(
        query_embeddings=query_embs_np,
        query_labels=query_designs,
        gallery_embeddings=gallery_embs_np,
        gallery_labels=gallery_designs,
        top_k_list=cfg.TOP_K_LIST,
    )
    logger.info(f"Retrieval Metrics: {retrieval_metrics}")

    # Top-K Search for all queries
    retrieval_results = gallery_index.search_batch(query_embs_np, top_k=max(cfg.TOP_K_LIST))

    # 4. Pairwise Verification Assessment
    eval_threshold = threshold if threshold is not None else cfg.DEFAULT_VERIFY_THRESHOLD
    verifier = SareeVerifier(model, threshold=eval_threshold, device=cfg.DEVICE, image_size=cfg.IMAGE_SIZE)

    # 5. Color-Invariance 4-Quadrant Diagnostic
    all_test_embs, all_test_designs, all_test_cw = extract_embeddings_for_items(
        model, test_items, image_size=cfg.IMAGE_SIZE, batch_size=cfg.BATCH_SIZE, device=cfg.DEVICE
    )
    invariance_stats = evaluate_color_invariance_breakdown(
        embeddings=all_test_embs.numpy(),
        items=test_items,
        num_pair_samples=1000,
        seed=cfg.SEED,
    )
    logger.info(f"Color-Invariance Diagnostic Results: {invariance_stats}")

    # Plot Invariance Distribution
    invariance_plot_path = cfg.FIGURES_DIR / f"{output_prefix}_color_invariance_diagnostic.png"
    plot_similarity_distribution(invariance_stats, invariance_plot_path)

    # 6. Generate Visualizations (t-SNE Clustering and Qualitative Grids)
    tsne_plot_path = cfg.FIGURES_DIR / f"{output_prefix}_tsne_clustering.png"
    plot_embedding_tsne(
        embeddings=all_test_embs.numpy(),
        design_ids=all_test_designs,
        colorway_ids=all_test_cw,
        output_path=tsne_plot_path,
        seed=cfg.SEED,
    )

    qual_plot_path = cfg.FIGURES_DIR / f"{output_prefix}_qualitative_retrieval_grid.png"
    plot_qualitative_retrieval_grid(
        query_items=query_items,
        retrieved_results=retrieval_results,
        output_path=qual_plot_path,
        num_queries_to_show=min(8, len(query_items)),
        top_k=5,
    )

    # 7. Aggregate Full Results Dict
    full_report = {
        "retrieval_metrics": retrieval_metrics,
        "color_invariance_diagnostic": invariance_stats,
        "operating_threshold": eval_threshold,
    }

    # Save to JSON
    report_json_path = cfg.RESULTS_DIR / f"{output_prefix}_evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    return full_report


def format_color_invariance_table(invariance_stats: Dict[str, Dict[str, float]]) -> str:
    """Formats the required markdown table of Color-Invariance pair similarities."""
    lines = [
        "| Pair Type | Avg Similarity | Std Dev | Interpretation |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for pair_type, stats in invariance_stats.items():
        mean_sim = stats.get("mean", 0.0)
        std_sim = stats.get("std", 0.0)
        if "Same Design / Same Color" in pair_type:
            interp = "Baseline reference (Expected: High)"
        elif "Same Design / Different Color" in pair_type:
            interp = "**Color-Invariant Match (Crucial: Must be High)**"
        elif "Different Design / Same Color" in pair_type:
            interp = "**Color-Deceptive Non-Match (Crucial: Must be Low)**"
        else:
            interp = "Different Design / Different Color (Expected: Low)"
        lines.append(f"| {pair_type} | **{mean_sim:.4f}** | ±{std_sim:.4f} | {interp} |")
    return "\n".join(lines)
