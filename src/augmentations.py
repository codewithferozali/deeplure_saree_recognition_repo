"""
Color-Invariant Augmentation Strategies for Textile Motif Recognition.
Ensures network features encode spatial motifs, structural weave, and geometric textures
while explicitly penalizing and decoupling reliance on RGB color palettes.
"""

import random
from typing import Tuple, List, Optional
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageEnhance
from src.preprocessing import resize_and_pad


class RandomChannelPermutation:
    """
    Randomly permutes RGB channels with probability p.
    For example: (R, G, B) -> (B, R, G) or (G, B, R).
    Disrupts the specific RGB chromatic signature while retaining 100% of spatial gradients and contours.
    """
    def __init__(self, p: float = 0.25):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)
        return img


class RandomEqualization:
    """
    Randomly applies histogram equalization to normalize lighting and contrast variations.
    """
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.equalize(img)
        return img


class RandomSolarize:
    """
    Inverts all pixel values above a threshold, creating harsh color shifts while preserving motif edges.
    """
    def __init__(self, threshold: int = 128, p: float = 0.15):
        self.threshold = threshold
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img, threshold=self.threshold)
        return img


def get_train_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    mode: str = "strong_color_invariant",
) -> T.Compose:
    """
    Builds single-view training augmentations for specified ablation/training modes.
    
    Modes:
      - 'baseline': Minimal spatial augmentations, standard RGB.
      - 'color_jitter': Baseline spatial + basic ColorJitter.
      - 'strong_color_invariant': Full color disruption (Jitter, Grayscale, Channel Shuffle, Solarize).
      - 'grayscale_only': Converts all training images to 3-channel grayscale.
    """
    spatial_transforms = [
        T.Lambda(lambda img: resize_and_pad(img, target_size=image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
    ]

    color_transforms: List = []
    if mode == "baseline":
        color_transforms = []
    elif mode == "color_jitter":
        color_transforms = [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ]
    elif mode == "grayscale_only":
        color_transforms = [
            T.Grayscale(num_output_channels=3)
        ]
    elif mode == "strong_color_invariant":
        color_transforms = [
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.6, hue=0.3),
            T.RandomGrayscale(p=0.35),
            RandomChannelPermutation(p=0.25),
            RandomEqualization(p=0.20),
            RandomSolarize(threshold=140, p=0.15),
        ]
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    tensor_transforms = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    return T.Compose(spatial_transforms + color_transforms + tensor_transforms)


def get_eval_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Deterministic evaluation transformation:
    Aspect-ratio preserving padding -> Tensor conversion -> ImageNet Normalization.
    """
    return T.Compose([
        T.Lambda(lambda img: resize_and_pad(img, target_size=image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class MultiViewColorInvariantTransform:
    """
    Generates multi-view representations for Contrastive Learning:
      - View 1 (Anchor): Standard spatial augmentation with mild appearance variation.
      - View 2 (Color-Perturbed Positive): Aggressive chromatic shift (HSV jitter, channel shuffle).
      - View 3 (Luminance/Grayscale Positive): Pure structural/grayscale representation.
    """
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_views: int = 2,
    ):
        self.num_views = num_views
        self.base_spatial = T.Compose([
            T.Lambda(lambda img: resize_and_pad(img, target_size=image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ])
        
        self.view_a_color = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

        self.view_b_color = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.7, hue=0.4),
            T.RandomGrayscale(p=0.4),
            RandomChannelPermutation(p=0.35),
            RandomSolarize(threshold=130, p=0.2),
        ])

        self.view_c_color = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ColorJitter(contrast=0.3, brightness=0.2)
        ])

        self.to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        views = []
        # View A: Standard RGB anchor
        v1 = self.base_spatial(img)
        views.append(self.to_tensor_norm(self.view_a_color(v1)))

        # View B: Heavy chromatic perturbation
        v2 = self.base_spatial(img)
        views.append(self.to_tensor_norm(self.view_b_color(v2)))

        if self.num_views >= 3:
            # View C: Luminance / Grayscale structural view
            v3 = self.base_spatial(img)
            views.append(self.to_tensor_norm(self.view_c_color(v3)))

        return views
