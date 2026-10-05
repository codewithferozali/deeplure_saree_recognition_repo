"""
Global Configuration for DeepLure Color-Invariant Saree Design Recognition.
Centralizes all hyperparameters, dataset paths, training settings, and runtime flags.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # ---------------- Reproducibility & Device ----------------
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 2 if os.name != "nt" else 0  # 0 on Windows to avoid IPC overhead

    # ---------------- Image & Preprocessing ----------------
    IMAGE_SIZE: int = 224
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)

    # ---------------- Model Architecture ----------------
    BACKBONE: str = "convnext_tiny"          # Final strong model
    BASELINE_BACKBONE: str = "resnet50"      # Baseline model for ablation comparison
    EMBEDDING_DIM: int = 256
    DROPOUT: float = 0.2
    PRETRAINED: bool = True
    POOLING_TYPE: str = "gem"                # 'gem' (Generalized Mean) or 'avg' (Adaptive Avg)
    GEM_P: float = 3.0

    # ---------------- Training & Optimization ----------------
    EPOCHS: int = 20
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    BACKBONE_LR_RATIO: float = 0.1           # Lower learning rate for pretrained backbone
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 5
    WARMUP_EPOCHS: int = 2
    LR_SCHEDULER: str = "cosine"             # 'cosine' or 'step'
    USE_AMP: bool = torch.cuda.is_available()  # Automatic Mixed Precision

    # ---------------- Metric Learning & Sampling ----------------
    P_CLASSES: int = 8                       # Number of distinct designs per batch
    K_SAMPLES: int = 4                       # Number of colorway/instance samples per design
    TEMPERATURE: float = 0.07                # SupCon temperature parameter
    TRIPLET_MARGIN: float = 0.3              # Hard triplet loss margin
    LOSS_TYPE: str = "supcon_triplet"        # 'supcon', 'triplet', 'supcon_triplet'
    SUPCON_WEIGHT: float = 1.0
    TRIPLET_WEIGHT: float = 0.5

    # ---------------- Augmentation Flags ----------------
    COLOR_JITTER_BRIGHTNESS: float = 0.4
    COLOR_JITTER_CONTRAST: float = 0.4
    COLOR_JITTER_SATURATION: float = 0.6
    COLOR_JITTER_HUE: float = 0.3
    GRAYSCALE_PROB: float = 0.35
    CHANNEL_PERMUTE_PROB: float = 0.20
    SOLARIZE_PROB: float = 0.15

    # ---------------- Filepaths & Directory Tree ----------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_ROOT: Path = field(default_factory=lambda: Path(os.environ.get("DATA_ROOT", "data")))
    DEEPLURE_DIR: str = "deeplure"
    INDIAN_SAREE_DIR: str = "indian_saree_patterns"
    SYNTHETIC_DIR: str = "synthetic_saree_data"

    CHECKPOINT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "checkpoints")
    OUTPUT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    FIGURES_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "figures")
    EMBEDDINGS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "embeddings")
    RESULTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs" / "results")

    # ---------------- Evaluation & Verification ----------------
    TOP_K_LIST: tuple = (1, 5, 10)
    DEFAULT_VERIFY_THRESHOLD: float = 0.70   # Overridden dynamically by validation calibration
    TEST_SPLIT_RATIO: float = 0.20
    VAL_SPLIT_RATIO: float = 0.15

    def __post_init__(self):
        for directory in [self.CHECKPOINT_DIR, self.OUTPUT_DIR, self.FIGURES_DIR, self.EMBEDDINGS_DIR, self.RESULTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# Default singleton instance
config = Config()
