"""
Evaluation Metrics for Retrieval and Pairwise Verification.
Computes:
  - Identification: Top-1, Top-5, Top-10 Accuracy, Recall@K, MRR, mAP
  - Verification: ROC-AUC, EER (Equal Error Rate), Precision, Recall, F1, Optimal Threshold
  - Color Invariance Diagnostic: Pairwise similarity matrix categorized by design and colorway
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score


def compute_identification_metrics(
    query_embeddings: np.ndarray,
    query_labels: List[str],
    gallery_embeddings: np.ndarray,
    gallery_labels: List[str],
    top_k_list: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Computes standard image retrieval ranking metrics:
      - Recall@K / Top-K Accuracy
      - Mean Reciprocal Rank (MRR)
      - Mean Average Precision (mAP)
      
    Args:
        query_embeddings: Array of shape [N_q, D] (L2-normalized).
        query_labels: Ground truth design IDs for each query [N_q].
        gallery_embeddings: Array of shape [N_g, D] (L2-normalized).
        gallery_labels: Design IDs for each gallery item [N_g].
        top_k_list: Tuple of K values to evaluate.
        
    Returns:
        Dictionary of computed retrieval metrics.
    """
    # Cosine similarity matrix: [N_q, N_g]
    sim_matrix = np.dot(query_embeddings, gallery_embeddings.T)

    num_queries = len(query_labels)
    gallery_labels_arr = np.array(gallery_labels)

    recalls = {k: 0 for k in top_k_list}
    mrr_total = 0.0
    ap_total = 0.0

    for i in range(num_queries):
        target_design = query_labels[i]
        sims = sim_matrix[i]

        # Rank gallery items in descending order of similarity
        ranked_indices = np.argsort(-sims)
        ranked_designs = gallery_labels_arr[ranked_indices]

        # Check matches
        matches = (ranked_designs == target_design)

        # Recall@K
        for k in top_k_list:
            if np.any(matches[:k]):
                recalls[k] += 1

        # MRR (first matching rank)
        match_positions = np.where(matches)[0]
        if len(match_positions) > 0:
            first_rank = match_positions[0] + 1  # 1-indexed
            mrr_total += 1.0 / first_rank

        # Average Precision (AP) for this query
        if len(match_positions) > 0:
            cumsum_matches = np.cumsum(matches)
            ranks = np.arange(1, len(matches) + 1)
            precisions = cumsum_matches / ranks
            ap = np.sum(precisions * matches) / np.sum(matches)
            ap_total += ap

    results = {
        "num_queries": float(num_queries),
        "gallery_size": float(len(gallery_labels)),
        "MRR": float(mrr_total / num_queries) if num_queries > 0 else 0.0,
        "mAP": float(ap_total / num_queries) if num_queries > 0 else 0.0,
    }

    for k in top_k_list:
        results[f"Recall@{k}"] = float(recalls[k] / num_queries) if num_queries > 0 else 0.0
        results[f"Top_{k}_Accuracy"] = results[f"Recall@{k}"]

    return results


def find_optimal_verification_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Finds the optimal operating threshold on validation pairs.
    Avoids arbitrary 0.5 thresholding.
    
    Args:
        similarities: 1D array of cosine similarity scores in [-1.0, 1.0].
        labels: 1D array of binary ground truth (1 for SAME design, 0 for DIFFERENT design).
        metric: 'f1' to maximize F1-score, or 'youden' to maximize TPR - FPR.
        
    Returns:
        (optimal_threshold, performance_dict_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = float(auc(fpr, tpr))

    # Calculate Equal Error Rate (EER)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Search optimal threshold based on candidate points
    candidate_thresholds = np.linspace(-0.2, 0.99, 200)
    best_thresh = eer_threshold
    best_score = -1.0
    best_metrics = {}

    for thresh in candidate_thresholds:
        preds = (similarities >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

        score = f1 if metric == "f1" else (acc)
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_metrics = {
                "threshold": float(thresh),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "eer": float(eer),
            }

    return best_thresh, best_metrics


def evaluate_color_invariance_breakdown(
    embeddings: np.ndarray,
    items: list,
    num_pair_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Conducts the dedicated 4-quadrant Color Invariance Diagnostic:
      Case A: Same Design / Same Color
      Case B: Same Design / Different Color (CRITICAL: MUST HAVE HIGH SIMILARITY)
      Case C: Different Design / Same Color (CRITICAL: MUST HAVE LOW SIMILARITY)
      Case D: Different Design / Different Color
      
    Returns dictionary with mean, std, median similarity for each quadrant.
    """
    rng = np.random.default_rng(seed)
    n = len(items)
    if n < 4:
        return {}

    # Pre-index by design and colorway
    design_ids = np.array([it.design_id for it in items])
    colorways = np.array([it.colorway_id for it in items])

    # Compute full pairwise similarity matrix
    sim_mat = np.dot(embeddings, embeddings.T)

    case_a_sims = []  # Same design, same color
    case_b_sims = []  # Same design, diff color
    case_c_sims = []  # Diff design, same color
    case_d_sims = []  # Diff design, diff color

    for _ in range(num_pair_samples):
        i, j = rng.choice(n, size=2, replace=False)
        same_design = (design_ids[i] == design_ids[j])
        same_color = (colorways[i] == colorways[j]) and (colorways[i] != "unknown")
        sim = float(sim_mat[i, j])

        if same_design and same_color:
            case_a_sims.append(sim)
        elif same_design and not same_color:
            case_b_sims.append(sim)
        elif not same_design and same_color:
            case_c_sims.append(sim)
        else:
            case_d_sims.append(sim)

    def stats(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "count": int(len(arr)),
        }

    return {
        "Same Design / Same Color": stats(case_a_sims),
        "Same Design / Different Color": stats(case_b_sims),
        "Different Design / Same Color": stats(case_c_sims),
        "Different Design / Different Color": stats(case_d_sims),
    }
