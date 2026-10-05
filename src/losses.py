"""
Metric Learning Loss Functions for Color-Invariant Motif Retrieval.
Implements:
  1. Supervised Contrastive Loss (SupCon)
  2. Batch-Hard Triplet Margin Loss with Online Mining
  3. ArcFace (Additive Angular Margin Loss)
  4. Combined Metric Learning Objective
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Enforces that all samples with the same design_id (across all colorways and views)
    are tightly clustered together on the hypersphere, while actively repelling samples
    with different design_ids, even if they share the same color palette.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Hidden vector of shape [B, n_views, D] or [B, D]. Must be L2-normalized.
            labels: Ground truth design labels of shape [B].
        Returns:
            Scalar contrastive loss.
        """
        device = features.device

        # If single-view [B, D], reshape to [B, 1, D]
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Reshape to [B * n_views, D]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_features
            anchor_count = n_views
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        # Compute cosine similarity matrix between all contrast features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_features.T),
            self.temperature
        )

        # For numerical stability, subtract row-wise max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Build mask for positive pairs [B * n_views, B * n_views]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels does not match batch size.")
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(anchor_count, n_views)

        # Mask-out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss with Online Hard Negative Mining (Hermans et al., 2017).
    For each anchor:
      - Finds the hardest positive (same design, lowest cosine similarity / largest distance).
      - Finds the hardest negative (different design, highest cosine similarity / smallest distance).
    Enforces: d(a, n) - d(a, p) >= margin
    """
    def __init__(self, margin: float = 0.3, distance_metric: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings of shape [B, D].
            labels: Design labels of shape [B].
        """
        device = embeddings.device
        labels = labels.view(-1, 1)

        # Distance matrix
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            similarity = torch.matmul(embeddings, embeddings.T)
            dist_mat = 1.0 - similarity
        else:
            # Euclidean distance between normalized vectors: ||u - v||_2
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Masks for positives and negatives
        is_pos = torch.eq(labels, labels.T)
        is_neg = ~is_pos

        # Exclude self-comparisons from positives
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=device)
        is_pos_no_diag = is_pos & ~eye

        # For each anchor, find hardest positive (maximum distance)
        # Set negative positions to -infinity
        pos_dists = dist_mat.clone()
        pos_dists[~is_pos_no_diag] = -float("inf")
        hardest_pos, _ = torch.max(pos_dists, dim=1)

        # For each anchor, find hardest negative (minimum distance)
        # Set positive positions to +infinity
        neg_dists = dist_mat.clone()
        neg_dists[~is_neg] = float("inf")
        hardest_neg, _ = torch.min(neg_dists, dim=1)

        # Triplet loss with margin
        # Only compute on valid anchors that have at least one valid positive and negative
        valid_anchors = (hardest_pos > -float("inf")) & (hardest_neg < float("inf"))
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        triplet_loss = F.relu(hardest_pos[valid_anchors] - hardest_neg[valid_anchors] + self.margin)
        return triplet_loss.mean()


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace, Deng et al., CVPR 2019).
    Useful as an auxiliary loss when training on fixed known designs.
    """
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0, margin: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and embeddings
        norm_w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, norm_w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return F.cross_entropy(output, labels)


class CombinedMetricLoss(nn.Module):
    """
    Blended Supervised Contrastive Loss + Batch-Hard Triplet Margin Loss.
    Combines dense global hypersphere clustering (SupCon) with aggressive
    local boundary margin enforcement (Batch-Hard Triplet).
    """
    def __init__(
        self,
        temperature: float = 0.07,
        triplet_margin: float = 0.3,
        supcon_weight: float = 1.0,
        triplet_weight: float = 0.5,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.triplet = BatchHardTripletLoss(margin=triplet_margin)
        self.supcon_weight = supcon_weight
        self.triplet_weight = triplet_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # If multi-view, embeddings shape is [B, V, D], compute supcon directly, and flatten for triplet
        if len(embeddings.shape) == 3:
            loss_supcon = self.supcon(embeddings, labels)
            # Use anchor view (view 0) for hard triplet mining
            anchor_emb = embeddings[:, 0, :]
            loss_triplet = self.triplet(anchor_emb, labels)
        else:
            loss_supcon = self.supcon(embeddings, labels)
            loss_triplet = self.triplet(embeddings, labels)

        return self.supcon_weight * loss_supcon + self.triplet_weight * loss_triplet
