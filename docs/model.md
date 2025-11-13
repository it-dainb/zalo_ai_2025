# YOLOv8n-RefDet Model Architecture Analysis

## Overview

YOLOv8n-RefDet is a hybrid reference-based object detection system designed for UAV search-and-rescue applications. The model combines real-time detection capabilities with few-shot learning through reference image support.

**Key Specifications:**
- **Total Parameters:** ~10.4M (20.8% of 50M budget)
- **Target Performance:** 25-30 FPS on Jetson Xavier NX
- **Input Resolution:** 640×640 for query images, 256×256 for support images
- **Detection Modes:** Standard (base classes), Prototype (novel classes), Dual (both)

---

## Architecture Pipeline

```
Query Image (640×640) ──→ YOLOv8n Backbone ──→ [P3, P4, P5]
                                                   ↓
Support Images (256×256) ──→ DINOv3 Encoder ──→ Prototypes
                                                   ↓
                                               CHEAF Fusion
                                                   ↓
                                               Dual Head
                                                   ↓
                                              Detections
```

---

## Module 1: DINOv3 Support Encoder

**File:** `src/models/dino_encoder.py`

### Purpose
Extracts rich semantic features from reference/support images to create class prototypes for reference-based detection.

### Architecture Details

#### Core Model
- **Base:** DINOv3 ViT-Small/16 (Vision Transformer)
- **Pretrained:** LVD-1689M dataset (1.689 billion images)
- **Input:** (B, 3, 256, 256) RGB images
- **Feature Dimension:** 384 (CLS token)
- **Parameters:** ~21.6M

#### Key Features
1. **Rotary Position Embeddings (RoPE):** Instead of learned positional embeddings, uses rotary embeddings for better generalization
2. **Patch Size:** 16×16 patches (256×256 image → 16×16 grid = 256 patches + 1 CLS token)
3. **Frozen Layers:** First 6 transformer blocks frozen by default to preserve pretrained knowledge

#### Multi-Scale Projection
Projects 384-dim CLS token to match YOLOv8n backbone channels:

| Scale | Output Dimension | Purpose | Architecture |
|-------|-----------------|---------|--------------|
| P3    | 64              | Small objects | Linear(384→64) + LayerNorm + GELU |
| P4    | 128             | Medium objects | Linear(384→128) + LayerNorm + GELU |
| P5    | 256             | Large objects | Linear(384→256) + LayerNorm + GELU |

#### Input Preprocessing
**Critical Requirement:** Images MUST be normalized with ImageNet statistics:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Use `encoder.get_transforms()` to obtain proper preprocessing pipeline.

#### Output Format
```python
{
    'prototype': (B, 384),  # Raw CLS token (normalized)
    'p3': (B, 64),          # Small object scale
    'p4': (B, 128),         # Medium object scale
    'p5': (B, 256),         # Large object scale
    'global_feat': (B, 384) # Optional: raw CLS for triplet loss
}
```

#### L2 Normalization
All outputs are L2-normalized (norm ≈ 1.0) for cosine similarity matching in the prototype head.

#### Key Methods

**`forward(support_images, normalize=True, return_global_feat=False)`**
- Extracts CLS token features
- Projects to multi-scale dimensions
- L2-normalizes for cosine similarity

**`compute_average_prototype(support_images)`**
- Averages features from multiple support images (K-shot learning)
- Useful for few-shot scenarios with multiple reference examples

**Design Philosophy:** "Lego-like" design with matching dimensions (64, 128, 256) to avoid hidden projections in downstream modules.

---

## Module 2: YOLOv8n Backbone Extractor

**File:** `src/models/yolov8_backbone.py`

### Purpose
Extracts multi-scale features from query images using YOLOv8n's proven detection backbone.

### Architecture Details

#### Core Model
- **Base:** YOLOv8n CSPDarknet + PANet
- **Weights:** Pre-trained on COCO or custom UAV dataset
- **Input:** (B, 3, 640, 640) RGB images
- **Parameters:** ~3.2M (backbone only)

#### Feature Extraction Strategy
Uses **forward hooks** to capture intermediate feature maps from PANet neck outputs (NOT backbone stages):

| Scale | Layer Index | Output Shape | Stride | Channels |
|-------|-------------|--------------|--------|----------|
| P3    | 15          | (B, 64, 80, 80)   | 8      | 64  |
| P4    | 18          | (B, 128, 40, 40)  | 16     | 128 |
| P5    | 21          | (B, 256, 20, 20)  | 32     | 256 |

**Why PANet outputs?**
- Provides multi-scale information with [64, 128, 256] channels
- Better feature fusion compared to backbone stages [256, 512, 512]
- Matches DINOv3 projection dimensions perfectly (Lego design)

#### YOLOv8n Architecture
```
Backbone (0-9): CSPDarknet feature extraction
├─ Conv + C2f blocks at multiple scales
├─ SPPF (Spatial Pyramid Pooling Fast)
└─ Outputs: [256, 512, 512] channels

Neck (10-21): PANet feature fusion
├─ Top-down pathway (upsampling)
├─ Bottom-up pathway (downsampling)
└─ Outputs: [64, 128, 256] channels ← **Extracted here**

Head (22): Detection head (not used)
```

#### Gradient Flow
All parameters have `requires_grad=True` by default, enabling fine-tuning. Can be frozen via `freeze_backbone=True`.

#### Key Methods

**`forward(x, return_global_feat=False)`**
- Runs forward pass through YOLOv8
- Hooks capture P3, P4, P5 features
- Optionally extracts global feature via adaptive pooling on P5: (B, 256, H, W) → (B, 256)

**`get_feature_dims()`**
- Returns expected dimensions for each scale
- Adjusts for different input sizes

---

## Module 3: CHEAF Fusion Module

**File:** `src/models/cheaf_fusion.py`

### Purpose
Fuses query features with support prototypes across multiple scales using spatial and channel correlations.

### Architecture Overview
Inspired by AirDet's Support-Guided Cross-Scale fusion, with 40% parameter reduction through depthwise separable convolutions.

### Components

#### 1. Depthwise Separable Convolution
**Reduction:** ~8× fewer parameters than standard convolution

```
Input (C_in, H, W)
  ↓
Depthwise Conv (groups=C_in) → BN → SiLU
  ↓
Pointwise Conv (1×1) → BN → SiLU
  ↓
Output (C_out, H, W)
```

**Parameters:** `C_in × K² + C_in × C_out` vs standard `C_in × C_out × K²`

#### 2. Channel Attention (Squeeze-and-Excitation)
```
Input (B, C, H, W)
  ↓
Global Avg Pool → (B, C)
  ↓
Linear(C → C/16) → ReLU
  ↓
Linear(C/16 → C) → Sigmoid
  ↓
Multiply with input → (B, C, H, W)
```

**Purpose:** Recalibrates channel-wise feature importance

#### 3. Spatial Relation Branch
Captures cross-image spatial correlation between query and support features.

```
Query Feat (B, C, H, W) ⊗ Support Proto (B, C, 1, 1)
  ↓ [element-wise multiplication]
Correlation Map (B, C, H, W)
  ↓
DepthwiseSeparableConv → BN → SiLU
  ↓
DepthwiseSeparableConv → BN → SiLU
  ↓
Output (B, out_channels, H, W)
```

**Key Insight:** Element-wise multiplication between query features and broadcast support prototypes creates a spatial correlation map highlighting regions similar to the reference.

#### 4. Channel Relation Branch
Recalibrates query features based on semantic importance.

```
Query Feat (B, C, H, W)
  ↓
Channel Attention
  ↓
Conv 1×1 → BN → SiLU
  ↓
Output (B, out_channels, H, W)
```

#### 5. Scale Fusion Module
Combines spatial and channel branches with residual connection.

```
Query Feat ─┬─→ Spatial Branch ──┐
            │                     ├─→ Concat → Conv1×1 → BN → SiLU ─┐
            └─→ Channel Branch ───┘                                 ├─→ ADD → Output
                                                                     │
            Residual (Conv1×1 or Identity) ──────────────────────────┘
```

### Multi-Scale Configuration

| Scale | Query Ch | Support Ch | Output Ch | Design Note |
|-------|----------|------------|-----------|-------------|
| P3    | 64       | 64         | 256       | Matches perfectly |
| P4    | 128      | 128        | 512       | Matches perfectly |
| P5    | 256      | 256        | 512       | Matches perfectly |

**Lego Design:** Query and support channels match at each scale, eliminating need for hidden projections.

### Bypass Mode
When `support_features=None`, fusion is bypassed and only projection is applied via the residual path:
```python
output = fusion_module.residual(query_feat)
```

This allows seamless switching between standard detection (no support) and reference-based detection.

### Parameters
- **Total:** ~1.2M (40% reduction vs original AirDet)
- **Per-scale:** ~0.4M each

---

## Module 4: Dual Detection Head

**File:** `src/models/dual_head.py`

### Purpose
Combines standard YOLOv8 detection (base classes) with prototype-based matching (novel classes) for unified detection.

### Architecture Components

#### 1. Standard Detection Head
Classic YOLOv8 decoupled head for base class detection.

```
Features [P3, P4, P5] at each scale:
  ├─→ Classification Branch (cv3)
  │   ├─ Conv(C, c3, 3) → BN → SiLU
  │   ├─ Conv(c3, c3, 3) → BN → SiLU
  │   └─ Conv(c3, nc, 1) → Class Logits
  │
  └─→ Box Regression Branch (cv2)
      ├─ Conv(C, c2, 3) → BN → SiLU
      ├─ Conv(c2, c2, 3) → BN → SiLU
      └─ Conv(c2, 4×reg_max, 1) → Box Distribution
```

**Output:**
- `standard_boxes`: List of 3 tensors (P3, P4, P5) with box distributions
- `standard_cls`: List of 3 tensors with class logits (B, nc, H, W)

#### 2. Distribution Focal Loss (DFL) Module
YOLOv8's box regression approach using soft binning.

```
Box Distribution (B, 4×reg_max, anchors)
  ↓ [reshape to (B, 4, reg_max, anchors)]
Softmax over reg_max dimension
  ↓
Conv1×1 with fixed weights [0, 1, 2, ..., reg_max-1]
  ↓
Box Coordinates (B, 4, anchors)
```

**Advantage:** Soft binning provides better localization than direct regression.

#### 3. Prototype Detection Head
Novel class detection via cosine similarity matching.

```
Features (B, C, H, W) at each scale
  ↓
Feature Projection → (B, proto_dim, H, W)
  ├─ Conv(C, proto_dim, 1)
  └─ Conv(proto_dim, proto_dim, 1)
  ↓
Cosine Similarity with Prototypes (K, proto_dim)
  ├─ L2 Normalize features
  ├─ L2 Normalize prototypes
  ├─ MatMul: (K, C) @ (B, C, H×W) → (B, K, H×W)
  └─ Reshape → (B, K, H, W)
  ↓
Temperature Scaling (learnable T ≈ 10.0)
  ↓
Similarity Scores (B, K, H, W)
```

**Scale-Specific Prototype Dimensions:**

| Scale | Feature Ch | Proto Dim | Purpose |
|-------|-----------|-----------|---------|
| P3    | 256       | 64        | Small objects |
| P4    | 512       | 128       | Medium objects |
| P5    | 512       | 256       | Large objects |

**Temperature Scaling:** Learnable parameter that controls the sharpness of similarity scores. Higher T → sharper distributions.

**Box Regression:** Shares the same cv2 branch as standard head for box predictions.

#### 4. DualDetectionHead Integration

```
Input: Features (P3, P4, P5) + Prototypes
  │
  ├─→ Standard Head (if nc_base > 0 and mode in ['standard', 'dual'])
  │   └─→ {standard_boxes, standard_cls}
  │
  └─→ Prototype Head (if prototypes provided and mode in ['prototype', 'dual'])
      └─→ {prototype_boxes, prototype_sim}
```

**Modes:**
- `standard`: Only standard head (base classes)
- `prototype`: Only prototype head (novel classes)
- `dual`: Both heads (base + novel classes)

**NMS Strategy:** Post-processing merges detections from both heads using class-agnostic NMS with configurable thresholds:
- `conf_thres`: 0.25 (default)
- `iou_thres`: 0.45 (default)

### Parameters
- **Total:** ~0.5M
- **Standard Head:** ~0.25M
- **Prototype Head:** ~0.25M

---

## Module 5: YOLOv8nRefDet (End-to-End Model)

**File:** `src/models/yolov8n_refdet.py`

### Purpose
Integrates all modules into a complete end-to-end reference-based detection system.

### Full Pipeline

```python
# Stage 1: Support Feature Extraction (offline or cached)
support_images (K, 3, 256, 256) → DINOv3 Encoder
  ↓
support_features = {
    'p3': (1, 64),
    'p4': (1, 128),
    'p5': (1, 256),
    'prototype': (1, 384)
}

# Stage 2: Query Feature Extraction
query_image (B, 3, 640, 640) → YOLOv8n Backbone
  ↓
query_features = {
    'p3': (B, 64, 80, 80),
    'p4': (B, 128, 40, 40),
    'p5': (B, 256, 20, 20)
}

# Stage 3: Cross-Scale Fusion
query_features + support_features → CHEAF Fusion
  ↓
fused_features = {
    'p3': (B, 256, 80, 80),
    'p4': (B, 512, 40, 40),
    'p5': (B, 512, 20, 20)
}

# Stage 4: Detection
fused_features + prototypes → Dual Head
  ↓
detections = {
    'standard_boxes': [...],
    'standard_cls': [...],
    'prototype_boxes': [...],
    'prototype_sim': [...]
}
```

### Key Features

#### 1. Support Feature Caching
**Purpose:** Optimize inference by pre-computing support features once.

```python
# Cache support features (run once)
model.set_reference_images(support_images, average_prototypes=True)

# Inference with cached features (run many times)
outputs = model(query_image, mode='dual', use_cache=True)
```

**Benefits:**
- 10-15% faster inference (DINOv3 computation amortized)
- Seamless batch processing with single reference
- Supports K-shot averaging

#### 2. Flexible Inference Modes

| Mode | Base Classes | Novel Classes | Use Case |
|------|--------------|---------------|----------|
| `standard` | ✓ | ✗ | General detection (COCO classes) |
| `prototype` | ✗ | ✓ | Pure few-shot detection |
| `dual` | ✓ | ✓ | Combined detection |

#### 3. K-Shot Support
Averages features from multiple reference images for more robust prototypes:

```python
support_list = [img1, img2, img3, img4, img5]  # 5-shot
model.set_reference_images(support_list, average_prototypes=True)
```

**Implementation:**
1. Extract features from each support image
2. Average across all images: `prototype = mean([p1, p2, ..., pk])`
3. Re-normalize after averaging

#### 4. Triplet Loss Support
For metric learning during training, can extract global features:

```python
# Extract features for triplet loss
query_feat = model.extract_features(query_imgs, image_type='query')    # (B, 256)
support_feat = model.extract_features(support_imgs, image_type='support')  # (B, 384)

# Or through forward pass
outputs = model(query_img, support_img, return_features=True)
triplet_loss = compute_triplet(
    anchor=outputs['support_global_feat'],
    positive=outputs['query_global_feat_pos'],
    negative=outputs['query_global_feat_neg']
)
```

**Feature Extraction:**
- Query (YOLOv8): Global avg pool on P5 → (B, 256)
- Support (DINOv3): Raw CLS token → (B, 384)

### Initialization Parameters

```python
YOLOv8nRefDet(
    yolo_weights='yolov8n.pt',           # YOLOv8 checkpoint
    nc_base=80,                          # Number of base classes
    dinov2_model='vit_small_patch16_dinov3.lvd1689m',  # DINOv3 model
    freeze_yolo=False,                   # Freeze YOLOv8 backbone
    freeze_dinov2=True,                  # Freeze DINOv3 encoder
    freeze_dinov2_layers=6,              # Freeze first 6 transformer blocks
    conf_thres=0.25,                     # Confidence threshold
    iou_thres=0.45,                      # IoU threshold for NMS
)
```

### Parameter Budget Allocation

| Component | Parameters | Percentage | Notes |
|-----------|-----------|------------|-------|
| DINOv3 Support Encoder | 21.6M | 43.2% | 6/12 blocks frozen |
| YOLOv8n Backbone | 3.2M | 6.4% | Can be frozen |
| CHEAF Fusion Module | 1.2M | 2.4% | Always trainable |
| Dual Detection Head | 0.5M | 1.0% | Always trainable |
| **Total** | **26.5M** | **53.0%** | **23.5M remaining** |

**Note:** With `freeze_dinov2=True`, only ~5M parameters are trainable.

### Memory Footprint
- **Model Size:** ~105 MB (FP32) or ~53 MB (FP16)
- **GPU Memory (single image):** ~2-3 GB (includes gradients)
- **Inference (FP16):** ~1.5 GB

---

## Data Flow Example

### Training Forward Pass

```python
# Batch configuration
batch = {
    'query_images': (4, 3, 640, 640),      # 4 query images
    'support_images': (4, 3, 256, 256),    # 4 support images (1 per query)
    'targets': [...],                       # Ground truth boxes
}

# Forward pass
outputs = model(
    query_image=batch['query_images'],
    support_images=batch['support_images'],
    mode='dual',
    use_cache=False,
    return_features=True  # For triplet loss
)

# Compute losses
det_loss = detection_loss(outputs, targets)        # YOLOv8 detection loss
proto_loss = prototype_loss(outputs, targets)      # Cosine similarity loss
triplet_loss = metric_learning_loss(
    outputs['support_global_feat'],
    outputs['query_global_feat']
)
total_loss = det_loss + proto_loss + 0.1 * triplet_loss
```

### Inference Forward Pass

```python
# Cache support features once
model.set_reference_images(support_imgs)

# Inference on multiple queries
for query_img in query_loader:
    outputs = model(
        query_image=query_img,
        mode='dual',
        use_cache=True,  # Fast!
    )
    
    # Post-process detections
    detections = postprocess(outputs, conf_thres=0.5, iou_thres=0.45)
```

---

## Design Principles

### 1. Lego-Like Modularity
All components have matching dimensions at interfaces:
- YOLOv8 output: [64, 128, 256] ↔ DINOv3 projection: [64, 128, 256]
- No hidden dimension mismatches requiring projection layers

### 2. Gradient Flow Preservation
All modules preserve gradients for end-to-end training:
- YOLOv8: Can be frozen or fine-tuned
- DINOv3: Partial freezing (first 6 blocks)
- CHEAF + Head: Always trainable

### 3. Efficiency Optimizations
- Depthwise separable convolutions (8× parameter reduction)
- Support feature caching (10-15% speed-up)
- FP16 inference support (2× memory reduction)

### 4. Multi-Scale Consistency
All processing operates at 3 scales (P3, P4, P5) consistently:
- Feature extraction: 3 scales
- Fusion: 3 scales
- Detection: 3 scales

---

## Training Pipeline (3-Stage)

### Stage 1: Base Training
**Objective:** Learn base class detection on COCO/UAV dataset

```
Mode: standard
Freeze: None
Epochs: 100
Loss: YOLOv8 detection loss
```

### Stage 2: Meta-Learning
**Objective:** Learn prototype matching on few-shot episodes

```
Mode: prototype
Freeze: YOLOv8 backbone
Epochs: 50
Loss: Prototype matching loss + Triplet loss
```

### Stage 3: Fine-Tuning
**Objective:** Joint optimization of both heads

```
Mode: dual
Freeze: None
Epochs: 30
Loss: Detection + Prototype + Triplet (weighted)
```

---

## Inference Optimization

### 1. Support Feature Caching
```python
# Pre-compute once
model.set_reference_images(support_imgs)

# Inference with cached features (10-15% faster)
outputs = model(query_img, mode='dual', use_cache=True)
```

### 2. FP16 Inference
```python
model = model.half()  # Convert to FP16
query_img = query_img.half()
outputs = model(query_img, mode='dual')
```

**Speed-up:** ~2× faster, 2× less memory

### 3. Batch Processing
```python
# Process multiple queries with single support
batch_queries = torch.stack([img1, img2, img3, img4])
outputs = model(batch_queries, mode='dual', use_cache=True)
```

### 4. ONNX Export (TODO)
```python
torch.onnx.export(model, ...)
# TensorRT optimization for Jetson Xavier NX
```

---

## Key Files Reference

| File | Lines | Description |
|------|-------|-------------|
| `dino_encoder.py` | 396 | DINOv3 support encoder with multi-scale projection |
| `yolov8_backbone.py` | 304 | YOLOv8n backbone feature extractor |
| `cheaf_fusion.py` | 429 | Support-guided cross-scale fusion module |
| `dual_head.py` | 466 | Dual detection head (standard + prototype) |
| `yolov8n_refdet.py` | 491 | End-to-end integrated model |
| `__init__.py` | 33 | Module exports |

**Total:** ~2,119 lines of model code

---

## Testing

Each module includes standalone test functions:

```bash
# Test individual modules
python src/models/dino_encoder.py
python src/models/yolov8_backbone.py
python src/models/cheaf_fusion.py
python src/models/dual_head.py
python src/models/yolov8n_refdet.py
```

**Test Coverage:**
- Single image processing
- Batch processing
- Multi-scale feature extraction
- Support feature caching
- K-shot averaging
- Parameter counting
- GPU memory usage

---

## Future Enhancements

1. **NMS Post-Processing:** Implement proper NMS fusion for dual head outputs
2. **ONNX Export:** Enable deployment to TensorRT/ONNX Runtime
3. **Dynamic Input Size:** Support arbitrary query image sizes
4. **Attention Visualization:** Add CAM/Grad-CAM for prototype matching
5. **Quantization:** INT8 quantization for Jetson deployment
6. **Multi-Reference Support:** Handle multiple reference classes simultaneously

---

## References

- **YOLOv8:** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **DINOv3:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **AirDet:** Support-Guided Cross-Scale Fusion (paper reference needed)
- **Few-Shot Detection:** Meta-learning approaches for object detection

---

## Contact

For questions or issues with the model architecture, please refer to:
- Documentation: `README.md`
- Test scripts: `src/tests/`
- Training scripts: `train.py`, `evaluate.py`

**Author:** Zalo AI Challenge 2025  
**Version:** 0.1.0  
**Last Updated:** 2025-11-12
