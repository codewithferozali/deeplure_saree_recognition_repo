"""
Production Inference Script for DeepLure Saree Design Recognition.
Provides CLI and programmatic interfaces for:
  1. Top-K Gallery Retrieval (Identification)
  2. Pairwise Verification (SAME / DIFFERENT DESIGN)

Usage Examples:
  # 1. Retrieve matching designs for a query image:
  python inference.py retrieve --query path/to/query.jpg --gallery path/to/gallery_or_index.npz --top_k 5

  # 2. Verify whether two saree images have the same design:
  python inference.py verify --image1 path/to/saree_red.jpg --image2 path/to/saree_blue.jpg
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from config import Config, config as default_config
from src.model import SareeDesignModel
from src.retrieval import GalleryIndex
from src.verification import SareeVerifier
from src.dataset import scan_directory_dataset
from src.train import extract_embeddings_for_items
from src.utils import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model(
    checkpoint_path: Optional[str] = None,
    backbone: str = "convnext_tiny",
    embedding_dim: int = 256,
    device: str = "cpu",
) -> SareeDesignModel:
    """Builds and loads model weights from checkpoint."""
    model = SareeDesignModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        pretrained=False,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        logger.warning("No checkpoint found; running with initialized model weights.")
    
    model.to(device)
    model.eval()
    return model


def run_retrieval(args) -> Dict[str, Any]:
    """Handles Top-K retrieval command."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.checkpoint, backbone=args.backbone, embedding_dim=args.embedding_dim, device=device)

    gallery_path = Path(args.gallery)
    if gallery_path.suffix == ".npz":
        # Load precomputed gallery index
        gallery_index = GalleryIndex.load(gallery_path, use_faiss=True)
    elif gallery_path.is_dir():
        # Build gallery index on the fly from directory
        logger.info(f"Scanning gallery directory: {gallery_path}")
        gallery_items = scan_directory_dataset(gallery_path)
        if not gallery_items:
            raise ValueError(f"No valid images found in gallery directory {gallery_path}")

        gallery_embs, gallery_designs, gallery_cw = extract_embeddings_for_items(
            model, gallery_items, image_size=args.image_size, device=device
        )
        gallery_index = GalleryIndex(embedding_dim=args.embedding_dim, use_faiss=True)
        gallery_index.add(
            embeddings=gallery_embs.numpy(),
            image_paths=[it.image_path for it in gallery_items],
            design_ids=gallery_designs,
            colorway_ids=gallery_cw,
        )
    else:
        raise FileNotFoundError(f"Gallery path {gallery_path} is neither an NPZ file nor a directory.")

    # Embed query image
    verifier = SareeVerifier(model, device=device, image_size=args.image_size)
    query_emb = verifier.embed_image(args.query)

    # Search
    matches = gallery_index.search(query_emb, top_k=args.top_k)

    print("\n" + "=" * 65)
    print(f"DEEPLURE IDENTIFICATION RESULTS for: {Path(args.query).name}")
    print("=" * 65)
    print(f"{'Rank':<6}{'Sim Score':<12}{'Design ID':<18}{'Colorway':<15}{'Image File'}")
    print("-" * 65)
    for m in matches:
        print(f"{m['rank']:<6}{m['similarity']:<12.4f}{m['design_id']:<18}{m.get('colorway_id', ''):<15}{Path(m['image_path']).name}")
    print("=" * 65 + "\n")

    return {"query": str(args.query), "matches": matches}


def run_verification(args) -> Dict[str, Any]:
    """Handles Pairwise verification command."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.checkpoint, backbone=args.backbone, embedding_dim=args.embedding_dim, device=device)

    threshold = args.threshold if args.threshold is not None else default_config.DEFAULT_VERIFY_THRESHOLD
    verifier = SareeVerifier(model, threshold=threshold, device=device, image_size=args.image_size)

    result = verifier.verify(args.image1, args.image2)

    print("\n" + "=" * 55)
    print("DEEPLURE PAIRWISE DESIGN VERIFICATION")
    print("=" * 55)
    print(f"Image 1:    {Path(args.image1).name}")
    print(f"Image 2:    {Path(args.image2).name}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Threshold:  {result['threshold']:.4f}")
    print(f"Prediction: {result['prediction']}")
    print("=" * 55 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="DeepLure Color-Invariant Saree Design Recognition CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Retrieve command
    parser_ret = subparsers.add_parser("retrieve", help="Retrieve Top-K matching designs from gallery")
    parser_ret.add_argument("--query", type=str, required=True, help="Path to query saree image")
    parser_ret.add_argument("--gallery", type=str, required=True, help="Path to gallery folder or index .npz")
    parser_ret.add_argument("--top_k", type=int, default=5, help="Number of nearest matches to return")
    parser_ret.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser_ret.add_argument("--backbone", type=str, default="convnext_tiny", help="Model backbone name")
    parser_ret.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser_ret.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser_ret.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")

    # Verify command
    parser_ver = subparsers.add_parser("verify", help="Verify if two saree images have the same motif")
    parser_ver.add_argument("--image1", type=str, required=True, help="Path to first saree image")
    parser_ver.add_argument("--image2", type=str, required=True, help="Path to second saree image")
    parser_ver.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: validation-calibrated)")
    parser_ver.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser_ver.add_argument("--backbone", type=str, default="convnext_tiny", help="Model backbone name")
    parser_ver.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension")
    parser_ver.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser_ver.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")

    args = parser.parse_args()

    if args.command == "retrieve":
        run_retrieval(args)
    elif args.command == "verify":
        run_verification(args)


if __name__ == "__main__":
    main()
