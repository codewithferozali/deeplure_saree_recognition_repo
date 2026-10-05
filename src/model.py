"""
Neural Network Architectures for Color-Invariant Saree Design Recognition.
Implements:
  - Generalized Mean Pooling (GeM)
  - SareeDesignModel with configurable backbones (ConvNeXt-Tiny, ResNet50)
  - Multi-layer non-linear Projection / Embedding Head
  - L2 Hypersphere Normalization (||z||_2 = 1.0)
"""

import logging
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM):
    f(x) = (1/|x| * sum_{x} x^p)^(1/p)
    When p=1 it equals Average Pooling; as p -> infinity it approaches Max Pooling.
    Focuses on prominent structural motifs and salient geometric regions.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        x = x.clamp(min=self.eps)
        p = self.p.clamp(min=1.0)
        x_pow = x.pow(p)
        pooled = F.adaptive_avg_pool2d(x_pow, (1, 1)).pow(1.0 / p)
        return pooled.flatten(1)


class SareeDesignModel(nn.Module):
    """
    Color-Invariant Saree Design Retrieval Architecture.
    
    Structure:
      Backbone (ConvNeXt-Tiny / ResNet-50)
         ↓
      Feature Map [B, C, H, W]
         ↓
      Pooling (GeM / AdaptiveAvgPool2d) -> [B, C]
         ↓
      Linear(C, 512)
         ↓
      BatchNorm1d(512)
         ↓
      ReLU / GELU
         ↓
      Dropout(p=0.2)
         ↓
      Linear(512, EMBEDDING_DIM=256)
         ↓
      L2 Normalization: z / ||z||_2
    """
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
        pooling_type: str = "gem",
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type

        # 1. Instantiate Feature Backbone
        self.backbone, in_features = self._build_backbone(self.backbone_name, pretrained)

        # 2. Pooling Layer
        if pooling_type == "gem":
            self.pool = GeMPooling(p=gem_p)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Projection / Metric Embedding Head
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, embedding_dim),
        )

        self._init_weights(self.head)

    def _build_backbone(self, backbone_name: str, pretrained: bool):
        """Constructs backbone and extracts raw convolutional feature map extractor."""
        if backbone_name == "convnext_tiny":
            try:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                model = models.convnext_tiny(weights=weights)
            except Exception as e:
                logger.warning(f"Could not load official weights for convnext_tiny ({e}). Initializing random.")
                model = models.convnext_tiny(weights=None)
            
            # ConvNeXt features module outputs [B, 768, 7, 7] for 224x224 input
            in_features = model.classifier[2].in_features  # 768
            features = model.features
            return features, in_features

        elif backbone_name == "resnet50":
            try:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                model = models.resnet50(weights=weights)
            except Exception as e:
                logger.warning(f"Could not load official weights for resnet50 ({e}). Initializing random.")
                model = models.resnet50(weights=None)

            in_features = model.fc.in_features  # 2048
            # Strip avgpool and fc, keeping convolutional trunk
            features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            return features, in_features

        elif backbone_name in ["efficientnet_b0", "efficientnet_b2"]:
            try:
                if backbone_name == "efficientnet_b0":
                    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                    model = models.efficientnet_b0(weights=weights)
                else:
                    weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
                    model = models.efficientnet_b2(weights=weights)
            except Exception:
                model = models.efficientnet_b0(weights=None)

            in_features = model.classifier[1].in_features
            features = model.features
            return features, in_features

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose convnext_tiny, resnet50, or efficientnet_b0.")

    def _init_weights(self, module):
        """Kaiming normal initialization for projection head layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts pre-pooling feature map from backbone."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          Input: [B, 3, H, W]
          Output: L2-normalized embedding vector [B, EMBEDDING_DIM]
        """
        feat_map = self.extract_features(x)
        
        if self.pooling_type == "gem":
            pooled = self.pool(feat_map)
        else:
            pooled = self.pool(feat_map).flatten(1)

        raw_emb = self.head(pooled)
        # Unit hyper-sphere L2 normalization
        norm_emb = F.normalize(raw_emb, p=2, dim=1)
        return norm_emb
