"""
Pairwise Design Verification Engine.
Given two saree images, determines whether they feature the same structural motif,
regardless of their colorway, using a validation-calibrated decision threshold.
"""

from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.preprocessing import safe_load_image
from src.augmentations import get_eval_transform
from src.metrics import find_optimal_verification_threshold


class SareeVerifier:
    """
    Inference-time verification engine.
    Computes cosine similarity between two saree images and classifies them as
    'SAME DESIGN' or 'DIFFERENT DESIGN' using a calibrated decision threshold.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        threshold: float = 0.70,
        device: str = "cpu",
        image_size: int = 224,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.transform = get_eval_transform(image_size=image_size)

    @torch.no_grad()
    def embed_image(self, image_or_path: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extracts L2-normalized embedding for a single image."""
        if isinstance(image_or_path, (str, Path)):
            img = safe_load_image(image_or_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_or_path}")
        else:
            img = image_or_path

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb.cpu().numpy()[0]

    def verify(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verifies whether image1 and image2 share the same textile design.
        
        Returns:
            {
                "similarity": 0.87,
                "threshold": 0.72,
                "prediction": "SAME DESIGN"  # or "DIFFERENT DESIGN"
            }
        """
        emb1 = self.embed_image(image1)
        emb2 = self.embed_image(image2)

        # Cosine similarity of unit vectors is their dot product
        similarity = float(np.dot(emb1, emb2))
        active_threshold = custom_threshold if custom_threshold is not None else self.threshold

        is_match = similarity >= active_threshold
        prediction_str = "SAME DESIGN" if is_match else "DIFFERENT DESIGN"

        return {
            "similarity": round(similarity, 4),
            "threshold": round(active_threshold, 4),
            "prediction": prediction_str,
            "is_same_design": bool(is_match),
        }

    @staticmethod
    def calibrate_threshold_on_dataset(
        model: torch.nn.Module,
        dataset,
        device: str = "cpu",
        num_pairs: int = 2000,
        seed: int = 42,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Extracts embeddings for a validation dataset, creates balanced positive
        and negative pairs, and discovers the optimal F1 threshold.
        """
        model.eval()
        rng = np.random.default_rng(seed)

        # Extract all embeddings
        all_embeddings = []
        all_labels = []

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(device)
                embs = model(images)
                all_embeddings.append(embs.cpu().numpy())
                all_labels.extend(labels.numpy())

        embs_arr = np.vstack(all_embeddings)
        labels_arr = np.array(all_labels)
        n = len(labels_arr)

        if n < 2:
            return 0.70, {"threshold": 0.70, "accuracy": 1.0, "roc_auc": 1.0}

        # Generate balanced pairs
        pair_sims = []
        pair_targets = []

        # Find positive and negative pairs
        unique_labels = np.unique(labels_arr)
        label_to_indices = {lbl: np.where(labels_arr == lbl)[0] for lbl in unique_labels}

        # Half positives, half negatives
        target_pos = num_pairs // 2
        pos_count = 0
        for _ in range(target_pos * 3):
            if pos_count >= target_pos:
                break
            lbl = rng.choice(unique_labels)
            idx_list = label_to_indices[lbl]
            if len(idx_list) >= 2:
                i, j = rng.choice(idx_list, size=2, replace=False)
                sim = np.dot(embs_arr[i], embs_arr[j])
                pair_sims.append(float(sim))
                pair_targets.append(1)
                pos_count += 1

        target_neg = pos_count
        for _ in range(target_neg):
            lbl1, lbl2 = rng.choice(unique_labels, size=2, replace=False)
            i = rng.choice(label_to_indices[lbl1])
            j = rng.choice(label_to_indices[lbl2])
            sim = np.dot(embs_arr[i], embs_arr[j])
            pair_sims.append(float(sim))
            pair_targets.append(0)

        best_thresh, metrics = find_optimal_verification_threshold(
            np.array(pair_sims),
            np.array(pair_targets),
            metric="f1",
        )

        return best_thresh, metrics
