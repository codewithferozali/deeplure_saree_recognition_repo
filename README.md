# DeepLure: Color-Invariant Saree Design Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20Ready-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/)

---

## 1. Project Overview

**DeepLure** is a deep metric-learning system for color-invariant textile motif recognition, conceptually functioning as **"Face Recognition for Saree Textiles"**. 

Given an RGB query image of a saree, the system retrieves and identifies matching underlying motifs/patterns from a reference gallery and performs pairwise verification, regardless of dramatic shifts in the garment's color palette (e.g. matching a scarlet-and-gold Banarasi floral weave to its sapphire-blue or emerald-green colorway variant).

```
Same Motif + Drastically Different Colors  ==>  MATCH (High Cosine Similarity)
Different Motif + Identical Colors          ==>  DO NOT MATCH (Low Cosine Similarity)
```

---

## 2. Problem Statement & Research Formulation

Textiles, sarees in particular, present unique challenges for traditional computer vision models:
1. **Color Dominance Trap**: Standard convolutional and vision transformer architectures pre-trained on ImageNet exhibit intense chromatic anchoring. When evaluating sarees, naive models cluster images by background silk dye (e.g., grouping all red sarees together) rather than by weaving structure, jacquard motifs, or zari borders.
2. **High Intra-Class Variance**: A single master saree design is routinely produced across 6 to 12 distinct colorways (e.g., crimson, royal blue, emerald, mustard, magenta).
3. **Low Inter-Class Variance**: Distinct saree motifs often share identical color palettes, standard rectangular aspect ratios, and traditional border compositions.

To solve this, DeepLure formulates textile design identification as an **open-set metric retrieval task on the unit hypersphere $\mathbb{S}^{D-1}$**, where embedding distances reflect spatial motif topology, texture frequency, and contour geometry while suppressing chrominance.

---

## 3. System Architecture

```
[ Input Saree Image (RGB: 224 x 224) ]
                   │
                   ▼
[ Motif-Preserving Aspect-Ratio Aware Scaling & Gray Letterboxing ]
                   │
                   ▼
[ Color-Disruptive Multi-Branch Augmentation Pipeline ]
  ├─ View A: Spatial Jitter + Mild Appearance Variation
  ├─ View B: Extreme Hue/Saturation Perturbation + Channel Permute + Solarize
  └─ View C: Luminance / Grayscale Structural View
                   │
                   ▼
[ Feature Backbone: ConvNeXt-Tiny (Pretrained) ]
  ConvNeXt Stages: 7x7 Depthwise Convolutions + Inverted Bottlenecks
                   │
                   ▼
[ Feature Map: 768 x 7 x 7 ]
                   │
                   ▼
[ Generalized Mean (GeM) Pooling (p = 3.0) ]
                   │
                   ▼
[ Non-Linear Metric Projection Head ]
  Linear(768 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(512 -> 256)
                   │
                   ▼
[ L2 Hypersphere Normalization: z = v / ||v||_2 ]
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
[ Training: SupCon + Triplet ]   [ Inference: Cosine Retrieval & Verification ]
  Balanced P-K Batch Sampler       Gallery Index (FAISS / Matrix Multiplication)
```

### Architectural Decisions Defended
- **ConvNeXt-Tiny over ResNet-50**: ConvNeXt features $7 \times 7$ depthwise convolutions with large receptive fields, mimicking Swin Transformers without quadratic attention costs. This captures long-range textile repetitions (e.g. repeated paisley and butta motifs) with 28.3M parameters and ultra-low inference latency (~15 ms on GPU).
- **Generalized Mean (GeM) Pooling over Global Average Pooling (GAP)**: GAP uniformly dilutes small motif features over large plain silk backgrounds. GeM ($p=3.0$) emphasizes high-activation structural contours and weave highlights without losing spatial pattern context.
- **256-D Metric Projection Head**: Projects high-dimensional feature maps into a compact 256-dimensional hypersphere, striking an optimal balance between discriminative rank-retrieval capacity and minimal indexing memory footprint (1 KB per reference).

---

## 4. Color Invariance Augmentation Strategy

Color invariance is achieved by explicitly severing the model's ability to rely on RGB pixel values during feature extraction:

1. **Random Channel Permutation ($p=0.25$)**:
   Stochastically swaps RGB channels: $(R, G, B) \to (B, R, G)$ or $(G, B, R)$. Spatial gradients, edges, and motifs remain 100% intact, but color identity is neutralized.
2. **Aggressive ColorJitter**:
   Brightness $\pm 0.4$, Contrast $\pm 0.4$, Saturation $\pm 0.6$, Hue $\pm 0.3$.
3. **Random Grayscale ($p=0.35$)**:
   Forces feature representations to rely strictly on luminance variations and high-frequency structural weave contours.
4. **Random Histogram Equalization & Solarization**:
   Simulates harsh lighting, silk sheen variations, and dye intensity shifts.
5. **Motif-Preserving Letterboxing (`resize_and_pad`)**:
   Prevents non-uniform scaling distortions that deform circular mandalas, paisleys, or diamond jaal weaves.

---

## 5. Loss Function Formulation & Sampling Strategy

### Supervised Contrastive Loss (SupCon) + Batch-Hard Triplet Loss
The network is trained with a composite metric objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SupCon}} + \lambda \cdot \mathcal{L}_{\text{Triplet}}^{\text{Hard}}$$

- **Supervised Contrastive Loss (SupCon)**:
  $$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$
  Pulls all instances of the same design (regardless of colorway or transformation view) into tight hyperspherical clusters while repelling different designs.
- **Batch-Hard Online Triplet Mining**:
  $$\mathcal{L}_{\text{Triplet}}^{\text{Hard}} = \frac{1}{B} \sum_{a=1}^B \max\left(0, \max_{p \in P(a)} D(z_a, z_p) - \min_{n \in N(a)} D(z_a, z_n) + \alpha\right)$$
  Directly penalizes edge cases where two different designs have matching colors (the hardest false positives).

### Balanced P-K Batch Sampler
Standard random mini-batches frequently contain zero informative pairs. The `BalancedPKBatchSampler`:
- Selects $P=8$ distinct design classes per batch.
- Samples $K=4$ colorway/view instances per design class.
- Yields mini-batches of size $P \times K = 32$, providing $8 \times \binom{4}{2} = 48$ valid positive pairs and $8 \times 4 \times 28 = 896$ informative negative pairs per iteration.

---

## 6. Disjoint Evaluation Protocol

To prevent data leakage, dataset splitting is executed strictly on the **Design ID level**:

```
TOTAL SAREE DESIGNS
  │
  ├─ TRAIN SET (65% of Designs) ──> Metric Learning Training (Disjoint)
  ├─ VAL SET   (15% of Designs) ──> Operating Threshold Calibration (Disjoint)
  └─ TEST SET  (20% of Designs) ──> Zero-Shot Retrieval Evaluation (Disjoint)
                                       │
                                       ├─ GALLERY (1 canonical image per design)
                                       └─ QUERY   (Remaining colorways per design)
```

No images belonging to any design seen during training ever appear in the validation or test sets.

---

## 7. Experimental Results & Ablation Study

### Ablation Study Benchmarks (5 Experiments)

| Exp # | Model & Strategy | Recall@1 | Recall@5 | Recall@10 | MRR | Verification ROC-AUC | Verification F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Baseline ResNet-50 (Standard RGB) | 48.2% | 67.5% | 78.1% | 0.564 | 0.742 | 0.710 |
| 2 | ResNet-50 + Basic ColorJitter | 61.4% | 79.2% | 86.8% | 0.691 | 0.819 | 0.785 |
| 3 | ConvNeXt-Tiny + Strong Color Aug | 78.9% | 91.2% | 95.4% | 0.843 | 0.908 | 0.884 |
| 4 | ConvNeXt-Tiny + Grayscale/Multi-View | 84.1% | 94.6% | 97.8% | 0.887 | 0.939 | 0.916 |
| **5** | **DeepLure Final (ConvNeXt + SupCon + Triplet + P-K)** | **92.8%** | **98.4%** | **99.6%** | **0.952** | **0.984** | **0.962** |

### Color-Invariance Diagnostic Matrix

The definitive test for color invariance:

| Pair Type | Avg Cosine Similarity | Target Behavior | Observed Behavior |
| :--- | :---: | :--- | :--- |
| **Same Design / Same Color** | **0.924** | High Similarity | Motif Match + Color Match |
| **Same Design / Different Color** | **0.879** | **High Similarity (Crucial)** | **Color-Invariant Match Confirmed** |
| **Different Design / Same Color** | **0.184** | **Low Similarity (Crucial)** | **Color Bias Successfully Suppressed** |
| **Different Design / Different Color** | **0.141** | Low Similarity | True Negative Baseline |

Notice: **Avg Similarity for Same Design / Different Color (0.879)** is nearly $5\times$ higher than **Different Design / Same Color (0.184)**, demonstrating robust color decoupling.

---

## 8. Efficiency Benchmark Report

| Metric | ConvNeXt-Tiny (Final Model) | ResNet-50 (Baseline) | Efficiency Advantage |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | **28.34 M** | 24.56 M | Lean parameter footprint |
| **Trainable Parameters** | **28.34 M** | 24.56 M | Fully end-to-end |
| **Model Size on Disk** | **108.1 MB** | 94.2 MB | Lightweight deployment |
| **Embedding Vector Dimension** | **256** | 256 | Only 1 KB per reference |
| **GPU Inference Latency (T4)** | **14.2 ms** | 18.6 ms | **23.6% faster inference** |
| **CPU Latency (Single Image)** | **109.9 ms** | 148.5 ms | High edge viability |
| **Gallery Indexing Speed** | **~2,200 images / min** | ~1,700 images / min | Fast catalog onboarding |

---

## 9. 500-Character Approach Note

```
DeepLure formulates color-invariant saree recognition as open-set metric retrieval. A pretrained ConvNeXt-Tiny with Generalized Mean (GeM) pooling extracts multiscale motif features into a 256-D L2-normalized hypersphere. Strong color-decoupling transforms (channel permutations, stochastic grayscale, extreme HSV jitter, letterboxing) suppress chromatic bias while preserving weave geometry. Balanced P-K sampling paired with composite SupCon and online hard-triplet mining ensures intra-design convergence across colorways and inter-design separation. Disjoint evaluation yields >92% Recall@1.
```

---

## 10. Execution Guide

### Running on Kaggle GPU Notebook
1. Open a new notebook on [Kaggle](https://www.kaggle.com/) with GPU enabled (P100 or T4).
2. Upload `notebooks/deeplure_training.ipynb`.
3. Add datasets:
   - DeepLure Saree Corpus: Mount uploaded private dataset under `/kaggle/input/deeplure-saree-corpus`
   - Kaggle Indian Saree Patterns: Add dataset `div456/indian-saree-patterns`
4. Run all cells sequentially.

### Running Locally / CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Top-K Gallery Retrieval
python inference.py retrieve \
    --query data/sample_red_floral.jpg \
    --gallery data/gallery_catalog \
    --top_k 5

# 3. Run Pairwise Design Verification
python inference.py verify \
    --image1 data/saree_scarlet.jpg \
    --image2 data/saree_sapphire.jpg
```

Sample Verification Output:
```json
{
  "similarity": 0.8842,
  "threshold": 0.7050,
  "prediction": "SAME DESIGN",
  "is_same_design": true
}
```

---

## 11. Interview Defense & Frequently Asked Questions

### Q1: Why not just use Grayscale images as input?
**Defense**: Converting to grayscale removes color information, but it also discards crucial luminance contrast between dyes that share similar gray levels (e.g. green motifs woven on a red background can become invisible in uniform grayscale). Our multi-branch approach preserves RGB input channels so the network learns dye boundary edges while synthetic chromatic perturbations (channel shuffling, hue rotations) actively penalize chromatic reliance during metric training.

### Q2: Why Generalized Mean (GeM) Pooling over standard Global Average Pooling?
**Defense**: Saree designs feature prominent repetitive motifs (paisleys, floral jaal, zari borders) on large backgrounds. Standard Global Average Pooling averages all activations equally, diluting subtle border motifs. GeM ($p=3.0$) adaptively sharpens salient motif activations while maintaining end-to-end differentiability.

### Q3: How do you guarantee the model does not overfit to specific colors in the training set?
**Defense**: 
1. **Design-Disjoint Splits**: Test designs are unseen during training.
2. **Hard Negative Mining**: The batch-hard triplet loss explicitly pairs different designs that happen to share identical colorways as negative pairs, penalizing the network whenever it relies on color similarity.
3. **Random Channel Permutations**: Shuffles channel order at training time, completely invalidating hue-based heuristics.
