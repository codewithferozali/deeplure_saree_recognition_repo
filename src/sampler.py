"""
Intelligent Metric Learning Batch Sampler.
Implements the P-K Class-Aware Balanced Sampler:
Guarantees that each mini-batch contains P distinct designs, with K samples per design.
This ensures abundant informative positive pairs (same design, different colorways)
and negative pairs (different designs, potentially sharing color palettes) for SupCon and Triplet mining.
"""

import random
from typing import Iterator, List, Dict
import torch
from torch.utils.data import Sampler


class BalancedPKBatchSampler(Sampler[List[int]]):
    """
    Batch Sampler that yields mini-batches of size P * K:
      - P: Number of unique design classes per batch.
      - K: Number of instances per design class in each batch.
    
    Args:
        labels: List or tensor of integer class indices corresponding to each dataset index.
        p_classes: Number of distinct classes per mini-batch.
        k_samples: Number of samples per class per mini-batch.
        batches_per_epoch: Optional limit on number of batches per epoch. If None, derived from dataset size.
        seed: Random seed for reproducibility.
    """
    def __init__(
        self,
        labels: List[int],
        p_classes: int = 8,
        k_samples: int = 4,
        batches_per_epoch: int = None,
        seed: int = 42,
    ):
        super().__init__()
        self.labels = labels
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.batch_size = p_classes * k_samples
        self.rng = random.Random(seed)

        # Index dataset items by their class label
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            self.class_to_indices.setdefault(lbl, []).append(idx)

        self.unique_classes = sorted(list(self.class_to_indices.keys()))

        if len(self.unique_classes) < self.p_classes:
            # Fall back to using all available unique classes if fewer than p_classes
            self.p_classes = max(2, len(self.unique_classes))
            self.batch_size = self.p_classes * self.k_samples

        if batches_per_epoch is not None:
            self.batches_per_epoch = batches_per_epoch
        else:
            # Number of batches to cover approximately one pass over the dataset
            self.batches_per_epoch = max(1, len(labels) // self.batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.batches_per_epoch):
            # Sample P distinct classes
            selected_classes = self.rng.sample(self.unique_classes, self.p_classes)
            batch = []

            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                if len(indices) >= self.k_samples:
                    # Sample K instances without replacement
                    selected_indices = self.rng.sample(indices, self.k_samples)
                else:
                    # Sample with replacement if class has fewer than K instances
                    selected_indices = [self.rng.choice(indices) for _ in range(self.k_samples)]
                batch.extend(selected_indices)

            yield batch
