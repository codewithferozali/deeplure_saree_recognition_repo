"""
Gallery Index and Retrieval Engine for Saree Design Identification.
Supports:
  - Exact Cosine Similarity Retrieval (PyTorch / NumPy dot product)
  - Optional FAISS IndexFlatIP for large-scale production indexing
  - Top-K retrieval with full metadata ranking
  - Serialization and Deserialization of Gallery Embeddings
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class GalleryIndex:
    """
    Searchable Gallery Reference Database of Saree Designs.
    Stores L2-normalized embeddings, design IDs, colorways, and image paths.
    """
    def __init__(self, embedding_dim: int = 256, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        self.embeddings: Optional[np.ndarray] = None
        self.image_paths: List[str] = []
        self.design_ids: List[str] = []
        self.colorway_ids: List[str] = []
        
        self.faiss_index = None
        if self.use_faiss:
            # Inner Product index on normalized vectors equals Cosine Similarity
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def add(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        image_paths: List[str],
        design_ids: List[str],
        colorway_ids: Optional[List[str]] = None,
    ):
        """
        Adds references to the gallery index.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)

        # L2-normalize just in case
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.image_paths.extend(image_paths)
        self.design_ids.extend(design_ids)
        if colorway_ids is not None:
            self.colorway_ids.extend(colorway_ids)
        else:
            self.colorway_ids.extend(["unknown"] * len(image_paths))

        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(embeddings)

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Searches the gallery for the Top-K most similar design matches.
        
        Returns:
            List of dicts: [
              {"rank": 1, "similarity": 0.892, "design_id": "design_01", "image_path": "...", "colorway_id": "..."},
              ...
            ]
        """
        results = self.search_batch(query_embedding, top_k=top_k)
        return results[0] if len(results) > 0 else []

    def search_batch(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Searches the gallery for a batch of query embeddings.
        """
        if self.embeddings is None or len(self.image_paths) == 0:
            raise ValueError("GalleryIndex is empty. Add gallery references before searching.")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        query_embeddings = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / np.clip(norms, 1e-12, None)

        k = min(top_k, len(self.image_paths))

        if self.use_faiss and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(query_embeddings, k)
        else:
            # Fallback to NumPy matrix multiplication: [N_q, N_g]
            sims = np.dot(query_embeddings, self.embeddings.T)
            indices = np.argsort(-sims, axis=1)[:, :k]
            distances = np.take_along_axis(sims, indices, axis=1)

        batch_results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for rank_idx, match_idx in enumerate(indices[i]):
                query_results.append({
                    "rank": rank_idx + 1,
                    "similarity": float(distances[i][rank_idx]),
                    "design_id": self.design_ids[match_idx],
                    "colorway_id": self.colorway_ids[match_idx],
                    "image_path": self.image_paths[match_idx],
                })
            batch_results.append(query_results)

        return batch_results

    def save(self, filepath: Union[str, Path]):
        """Serializes gallery embeddings and metadata to an NPZ/JSON archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            str(path),
            embeddings=self.embeddings,
            image_paths=np.array(self.image_paths),
            design_ids=np.array(self.design_ids),
            colorway_ids=np.array(self.colorway_ids),
            embedding_dim=self.embedding_dim,
        )
        logger.info(f"Saved gallery index with {len(self.image_paths)} items to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path], use_faiss: bool = True) -> "GalleryIndex":
        """Loads gallery index from saved file."""
        data = np.load(str(filepath), allow_pickle=True)
        emb_dim = int(data["embedding_dim"])
        index = cls(embedding_dim=emb_dim, use_faiss=use_faiss)
        index.add(
            embeddings=data["embeddings"],
            image_paths=list(data["image_paths"]),
            design_ids=list(data["design_ids"]),
            colorway_ids=list(data["colorway_ids"]),
        )
        logger.info(f"Loaded gallery index with {len(index.image_paths)} items from {filepath}")
        return index
