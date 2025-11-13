# YOLOv8n-RefDet: Complete Training Guide

**A Step-by-Step Tutorial for Training Few-Shot UAV Object Detectors**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Before You Begin](#before-you-begin)
3. [Data Preparation](#data-preparation)
4. [Training Workflow](#training-workflow)
5. [Stage-by-Stage Training](#stage-by-stage-training)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Monitoring Training](#monitoring-training)
8. [Advanced Techniques](#advanced-techniques)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [Production Checklist](#production-checklist)

---

## Introduction

This guide will walk you through **training YOLOv8n-RefDet from scratch** to deployment-ready model. Whether you're training on a single GPU or a workstation, this guide covers everything you need to know.

### What You'll Learn

- How to prepare your dataset for few-shot learning
- Complete 3-stage training pipeline
- How to tune hyperparameters for your specific use case
- Advanced techniques (triplet training, mixed batches)
- How to debug common training issues
- Best practices for production deployment

### Time Commitment

- **Data Preparation**: 1-2 hours
- **Stage 2 Training**: 4-6 hours (100 epochs on RTX 3090)
- **Stage 3 Fine-tuning**: 30-60 minutes
- **Total**: ~6-9 hours for complete training

---

## Before You Begin

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 2060 (6GB VRAM)
- RAM: 16GB
- Storage: 50GB SSD

**Recommended:**
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
- RAM: 32GB+
- Storage: 100GB NVMe SSD

**Production Target:**
- NVIDIA Jetson Xavier NX (deployment)

### Software Setup

```bash
# 1. Create conda environment
conda create -n zalo python=3.10
conda activate zalo

# 2. Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Download Pre-trained Weights

```bash
# Create models directory
mkdir -p ./models/base

# Download YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P ./models/base/

# DINOv2 weights (auto-downloaded via timm on first run)
```

### Verify Installation

```bash
# Run test suite
cd src/tests && python run_all_tests.py

# Expected output:
# ========================================
# Running all RefDet tests...
# ========================================
# 
# test_data_loading.py ........ 12 passed ✅
# test_model_components.py ..... 10 passed, 5 skipped ✅
# test_training_components.py .. 15 passed ✅
# test_training_full.py ........ 5 passed ✅
#
# Total: 42 passed, 5 skipped ✅
```

---

## Data Preparation

### Dataset Structure

Your dataset should follow this exact structure:

```
datasets/
├── train/
│   ├── samples/
│   │   ├── Backpack_0/
│   │   │   ├── drone_video.mp4          # Source video
│   │   │   └── object_images/           # Reference images
│   │   │       ├── img_1.jpg            # 1st reference
│   │   │       ├── img_2.jpg            # 2nd reference
│   │   │       └── img_3.jpg            # 3rd reference
│   │   ├── Backpack_1/
│   │   ├── Motorbike_0/
│   │   └── ...
│   └── annotations/
│       └── annotations.json             # All annotations
└── test/
    └── (same structure)
```

### Annotations Format

**`annotations.json` format:**

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          {
            "frame": 3483,
            "x1": 321,
            "y1": 0,
            "x2": 381,
            "y2": 12
          },
          {
            "frame": 3484,
            "x1": 320,
            "y1": 1,
            "x2": 380,
            "y2": 13
          }
        ]
      }
    ]
  }
]
```

**Key Points:**
- `video_id` must match the folder name in `samples/`
- Coordinates are absolute pixel values (not normalized)
- Frame numbers are 0-indexed
- Multiple bboxes per frame are supported

### Validate Your Dataset

```bash
# Quick validation script
python -c "
from src.datasets.refdet_dataset import RefDetDataset

# Load dataset
ds = RefDetDataset(
    data_root='./datasets/train/samples',
    annotations_file='./datasets/train/annotations/annotations.json',
    mode='train'
)

print(f'Dataset loaded successfully!')
print(f'Total samples: {len(ds)}')
print(f'Classes: {ds.classes}')

# Test loading one sample
sample = ds[0]
print(f'\\nSample 0:')
print(f'  Video ID: {sample[\"video_id\"]}')
print(f'  Query frame shape: {sample[\"query_frame\"].shape}')
print(f'  Number of support images: {len(sample[\"support_images\"])}')
print(f'  Bboxes: {sample[\"bboxes\"]}')
"
```

**Expected Output:**
```
Dataset loaded successfully!
Total samples: 16595
Classes: ['Backpack_0', 'Jacket_0', 'Jacket_1', ...]

Sample 0:
  Video ID: Backpack_0
  Query frame shape: (1080, 1920, 3)
  Number of support images: 3
  Bboxes: [[321.  0. 381. 12.]]
```

---

## Training Workflow

### Overview of Training Stages

```
┌──────────────────────────────────────────────────────────┐
│                  Training Pipeline                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Stage 1: Base Pre-training (Optional)                   │
│  ├─ Dataset: COCO/VisDrone                               │
│  ├─ Loss: bbox + cls + dfl                               │
│  ├─ Duration: 50 epochs (~2 hours)                       │
│  └─ Purpose: Learn general object features               │
│                                                           │
│            ⬇ [checkpoint_stage1.pt]                      │
│                                                           │
│  Stage 2: Few-Shot Meta-Learning (Main)                  │
│  ├─ Dataset: Competition data (episodic)                 │
│  ├─ Loss: bbox + cls + dfl + supcon + cpe                │
│  ├─ Duration: 100 epochs (~4-6 hours)                    │
│  └─ Purpose: Learn to adapt to novel objects             │
│                                                           │
│            ⬇ [best_model.pt]                             │
│                                                           │
│  Stage 3: Fine-Tuning (Optional)                         │
│  ├─ Dataset: Competition data (episodic)                 │
│  ├─ Loss: reduced contrastive + triplet                  │
│  ├─ Duration: 30 epochs (~30-60 min)                     │
│  └─ Purpose: Prevent catastrophic forgetting             │
│                                                           │
│            ⬇ [final_model.pt]                            │
│                                                           │
│  Ready for Deployment! ✓                                 │
└──────────────────────────────────────────────────────────┘
```

### Quick Start (Skip to Stage 2)

**For most users, you can skip Stage 1** and start directly with Stage 2 using pretrained YOLOv8n weights:

```bash
python train.py \
    --stage 2 \
    --epochs 100 \
    --n_way 2 \
    --n_query 4 \
    --mixed_precision
```

This leverages YOLOv8n's pretrained weights on COCO for general object features.

---

## Stage-by-Stage Training

### Stage 1: Base Pre-training (Optional)

**When to use:**
- You have access to a large base dataset (COCO, VisDrone)
- You want to learn domain-specific features (e.g., aerial view objects)
- You have time for extended training

**When to skip:**
- Using pretrained YOLOv8n weights (recommended)
- Limited training time
- Similar domain to COCO dataset

**Training Command:**

```bash
python train.py \
    --stage 1 \
    --data_root ./datasets/coco/train \
    --annotations ./datasets/coco/annotations.json \
    --epochs 50 \
    --n_way 5 \
    --n_query 8 \
    --batch_size 8 \
    --lr 1e-3 \
    --mixed_precision \
    --checkpoint_dir ./checkpoints/stage1
```

**Key Parameters:**
- `n_way=5`: More classes for diverse training
- `n_query=8`: More examples per class
- `lr=1e-3`: Higher learning rate for faster convergence
- No contrastive losses (only detection losses)

**Training Time:** ~2 hours (50 epochs on RTX 3090)

**Expected Loss Curve:**
```
Epoch 1:  Loss 2.145
Epoch 10: Loss 0.845
Epoch 25: Loss 0.512
Epoch 50: Loss 0.356
```

---

### Stage 2: Few-Shot Meta-Learning (MAIN STAGE)

**This is the core training stage** where the model learns to detect novel objects from few examples.

#### Basic Training

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --stage 2 \
    --epochs 100 \
    --n_way 2 \
    --n_query 4 \
    --mixed_precision \
    --checkpoint_dir ./checkpoints/stage2
```

#### Advanced Training with All Features

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --test_data_root ./datasets/test/samples \
    --test_annotations ./datasets/test/annotations/annotations.json \
    --stage 2 \
    --epochs 100 \
    --n_way 2 \
    --n_query 4 \
    --n_episodes 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --batch_size 4 \
    --gradient_accumulation 2 \
    --mixed_precision \
    --num_workers 8 \
    --checkpoint_dir ./checkpoints/stage2_full \
    --save_interval 5 \
    --bbox_weight 7.5 \
    --cls_weight 0.5 \
    --dfl_weight 1.5 \
    --supcon_weight 1.0 \
    --cpe_weight 0.5
```

#### Parameter Breakdown

**Data Parameters:**
- `--data_root`: Path to training videos and reference images
- `--annotations`: Path to annotations.json
- `--test_data_root`: Optional validation set
- `--test_annotations`: Optional validation annotations

**Training Parameters:**
- `--stage 2`: Few-shot meta-learning stage
- `--epochs 100`: Number of training epochs
- `--n_way 2`: Number of classes per episode (2-4 recommended)
- `--n_query 4`: Query frames per class (4-8 recommended)
- `--n_episodes 100`: Episodes per epoch (100-200)

**Optimizer Parameters:**
- `--lr 1e-4`: Base learning rate (critical!)
- `--weight_decay 0.05`: L2 regularization
- `--gradient_accumulation 2`: Effective batch size = batch_size * grad_acc

**Loss Weights:**
- `--bbox_weight 7.5`: WIoU v3 (bounding box regression)
- `--cls_weight 0.5`: BCE (classification)
- `--dfl_weight 1.5`: Distribution Focal Loss
- `--supcon_weight 1.0`: Supervised Contrastive Learning
- `--cpe_weight 0.5`: Contrastive Proposal Encoding

**Training Settings:**
- `--mixed_precision`: Enable AMP (2x speedup, lower memory)
- `--num_workers 8`: Data loading workers (adjust for your CPU)
- `--checkpoint_dir`: Where to save checkpoints
- `--save_interval 5`: Save every N epochs

#### Training Time

| Hardware | Batch Config | Time per Epoch | Total Time (100 epochs) |
|----------|--------------|----------------|-------------------------|
| RTX 3090 | n_way=2, n_query=4 | ~3 min | ~5 hours |
| RTX 4090 | n_way=2, n_query=4 | ~2 min | ~3.5 hours |
| RTX 2060 | n_way=2, n_query=2 | ~8 min | ~13 hours |

#### Expected Training Progress

```
Epoch 1 Summary (4.61s):
  Train Loss: 0.8245
  Val Loss: 0.7532
  ✓ New best model!
Saved checkpoint: ./checkpoints/stage2/checkpoint_epoch_1.pt

Epoch 10 Summary (4.58s):
  Train Loss: 0.5621
  Val Loss: 0.5234
  ✓ New best model!

Epoch 50 Summary (4.55s):
  Train Loss: 0.3124
  Val Loss: 0.3456
  ✓ New best model!

Epoch 100 Summary (4.52s):
  Train Loss: 0.2245
  Val Loss: 0.2789
  ✓ New best model!
```

**Good Training Signs:**
- ✅ Train loss steadily decreasing
- ✅ Val loss following train loss (not diverging)
- ✅ Loss stabilizing around epoch 70-80
- ✅ New best model saved every few epochs initially

**Warning Signs:**
- ⚠️ Val loss much higher than train loss (overfitting)
- ⚠️ Loss oscillating wildly (learning rate too high)
- ⚠️ Loss plateaus early (<0.5 after 50 epochs)
- ⚠️ NaN losses (gradient explosion)

---

### Stage 3: Fine-Tuning with Triplet Loss

**Purpose:** Prevent catastrophic forgetting when adapting to specific objects.

**When to use:**
- You want to specialize on specific object instances
- You need to maintain base class performance
- You're deploying for a specific mission

#### Basic Fine-Tuning

```bash
python train.py \
    --stage 3 \
    --epochs 30 \
    --n_way 2 \
    --n_query 4 \
    --supcon_weight 0.5 \
    --cpe_weight 0.3 \
    --triplet_weight 0.2 \
    --resume ./checkpoints/stage2/best_model.pt \
    --checkpoint_dir ./checkpoints/stage3
```

#### Advanced: Triplet Training

Enable triplet loss training for better feature separation:

```bash
python train.py \
    --stage 3 \
    --epochs 30 \
    --n_way 2 \
    --n_query 4 \
    --use_triplet \
    --triplet_ratio 0.3 \
    --negative_strategy mixed \
    --triplet_batch_size 8 \
    --supcon_weight 0.5 \
    --cpe_weight 0.3 \
    --triplet_weight 0.2 \
    --resume ./checkpoints/stage2/best_model.pt \
    --checkpoint_dir ./checkpoints/stage3_triplet
```

**Triplet Parameters:**
- `--use_triplet`: Enable triplet loss training
- `--triplet_ratio 0.3`: 30% of batches are triplet batches
- `--negative_strategy mixed`: Use both background and cross-class negatives
- `--triplet_batch_size 8`: Batch size for triplet training

**Training Time:** ~30-60 minutes (30 epochs)

---

## Hyperparameter Tuning

### Learning Rate Selection

**Learning rate is the most important hyperparameter.** Here's how to choose:

#### Method 1: Start with Defaults

```bash
# Use default layerwise learning rates
python train.py --stage 2 --lr 1e-4
```

**Layerwise LRs (automatic):**
- DINOv2: `1e-5` (lr × 0.1)
- YOLOv8 Backbone: `1e-4` (base lr)
- CHEAF Fusion: `2e-4` (lr × 2.0)
- Detection Head: `2e-4` (lr × 2.0)

#### Method 2: Grid Search

Test different learning rates:

```bash
# Too low (slow convergence)
python train.py --stage 2 --lr 5e-5 --epochs 10

# Default (recommended)
python train.py --stage 2 --lr 1e-4 --epochs 10

# Higher (faster but may overshoot)
python train.py --stage 2 --lr 2e-4 --epochs 10

# Too high (divergence)
python train.py --stage 2 --lr 5e-4 --epochs 10
```

Compare losses after 10 epochs and pick the best.

#### Method 3: Learning Rate Finder

```python
# lr_finder.py
import torch
from train import create_dataloaders, create_model, create_loss_fn
import matplotlib.pyplot as plt

lrs = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
losses = []

for lr in lrs:
    # Train for 5 epochs
    # Record average loss
    pass

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.savefig('lr_finder.png')
```

### Loss Weight Tuning

**Default weights work well for most cases**, but you can tune them:

#### Scenario 1: Poor Localization (Boxes not tight)

**Problem:** Model predicts correct class but boxes are off.

**Solution:** Increase bbox weight

```bash
python train.py \
    --stage 2 \
    --bbox_weight 10.0 \
    --cls_weight 0.5 \
    --dfl_weight 1.5
```

#### Scenario 2: Poor Classification (Wrong classes)

**Problem:** Model detects objects but assigns wrong classes.

**Solution:** Increase classification weight

```bash
python train.py \
    --stage 2 \
    --bbox_weight 7.5 \
    --cls_weight 1.0 \
    --dfl_weight 1.5
```

#### Scenario 3: Poor Few-Shot Generalization

**Problem:** Model doesn't adapt well to novel objects.

**Solution:** Increase contrastive loss weights

```bash
python train.py \
    --stage 2 \
    --supcon_weight 1.5 \
    --cpe_weight 0.8
```

### Episodic Sampling Configuration

#### Small Dataset (few classes)

```bash
python train.py \
    --stage 2 \
    --n_way 2 \
    --n_query 4 \
    --n_episodes 50
```

#### Medium Dataset (5-10 classes)

```bash
python train.py \
    --stage 2 \
    --n_way 3 \
    --n_query 6 \
    --n_episodes 100
```

#### Large Dataset (10+ classes)

```bash
python train.py \
    --stage 2 \
    --n_way 4 \
    --n_query 8 \
    --n_episodes 200
```

**Rules of Thumb:**
- `n_way`: Number of classes per episode
  - Too low (1-2): Faster training but less diversity
  - Too high (5+): Slower training, may hurt convergence
  - **Recommended: 2-3**

- `n_query`: Frames per class
  - Too low (1-2): Unstable gradients
  - Too high (10+): Overfitting to episode
  - **Recommended: 4-8**

- `n_episodes`: Episodes per epoch
  - More episodes = more training diversity
  - **Recommended: 100-200**

---

## Monitoring Training

### Real-Time Monitoring

**Training outputs comprehensive logs:**

```
============================================================
Epoch 50/100
============================================================

Episode [1/100]: 100%|███████████| 100/100 [03:45<00:00, 2.25s/it]
  Loss: 0.3124 | bbox: 0.1834 | cls: 0.0245 | dfl: 0.0512 | supcon: 0.0345 | cpe: 0.0188

Epoch 50 Summary (3.75s):
  Train Loss: 0.3124
  Val Loss: 0.3456
  ✓ New best model!
Saved checkpoint: ./checkpoints/stage2/checkpoint_epoch_50.pt
Saved best model: ./checkpoints/stage2/best_model.pt
```

### Key Metrics to Watch

**1. Total Loss**
- Should decrease steadily
- Stage 2: Target <0.3 by epoch 100
- Stage 3: Target <0.25 by epoch 30

**2. Component Losses**
- **bbox loss**: Should be largest component (~60% of total)
- **cls loss**: Should stabilize early (~10% of total)
- **dfl loss**: Should decrease steadily (~20% of total)
- **supcon + cpe**: Should converge to small values (~10% total)

**3. Validation Loss**
- Should track train loss (gap <0.05)
- Large gap (>0.1) indicates overfitting

### Visualizing Training Progress

**Extract loss curves from logs:**

```python
# parse_logs.py
import re
import matplotlib.pyplot as plt

losses = {'train': [], 'val': []}

with open('training.log', 'r') as f:
    for line in f:
        if 'Train Loss:' in line:
            loss = float(re.search(r'Train Loss: ([\d.]+)', line).group(1))
            losses['train'].append(loss)
        if 'Val Loss:' in line:
            loss = float(re.search(r'Val Loss: ([\d.]+)', line).group(1))
            losses['val'].append(loss)

plt.plot(losses['train'], label='Train')
plt.plot(losses['val'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
```

### Checkpointing Strategy

**Automatic checkpointing:**

```
checkpoints/
├── checkpoint_epoch_1.pt     # First epoch
├── checkpoint_epoch_5.pt     # Every 5 epochs (configurable)
├── checkpoint_epoch_10.pt
├── ...
├── checkpoint_epoch_100.pt   # Final epoch
└── best_model.pt             # Best validation loss
```

**Resume from checkpoint:**

```bash
# Resume training from specific epoch
python train.py \
    --stage 2 \
    --resume ./checkpoints/checkpoint_epoch_50.pt \
    --epochs 100  # Will train from epoch 51 to 100
```

---

## Advanced Techniques

### Technique 1: Mixed Batch Training

**Combine episodic and triplet batches for better feature learning:**

```bash
python train.py \
    --stage 2 \
    --epochs 100 \
    --n_way 2 \
    --n_query 4 \
    --use_triplet \
    --triplet_ratio 0.3 \
    --negative_strategy mixed \
    --triplet_batch_size 8
```

**How it works:**
- 70% of batches: Normal episodic training
- 30% of batches: Triplet training (anchor, positive, negative)
- Better feature separation between classes

### Technique 2: Gradient Accumulation

**Simulate larger batch sizes on limited GPU memory:**

```bash
python train.py \
    --stage 2 \
    --n_way 2 \
    --n_query 4 \
    --batch_size 4 \
    --gradient_accumulation 4  # Effective batch = 4 × 4 = 16
```

**Benefits:**
- Train with larger effective batch sizes
- More stable gradients
- Better convergence

**Trade-off:**
- Slower training (4× more forward passes)

### Technique 3: Dynamic Loss Weighting

**Automatically adjust loss weights during training:**

```python
# Add to trainer.py
def adjust_loss_weights(self, epoch, total_epochs):
    """Reduce contrastive loss weights in later epochs."""
    progress = epoch / total_epochs
    
    if progress > 0.7:  # After 70% of training
        self.loss_fn.weights['supcon'] *= 0.5
        self.loss_fn.weights['cpe'] *= 0.5
```

### Technique 4: Curriculum Learning

**Start with easier episodes, gradually increase difficulty:**

```bash
# Epochs 1-30: Easy (2-way, 4-query)
python train.py --stage 2 --epochs 30 --n_way 2 --n_query 4

# Epochs 31-60: Medium (3-way, 6-query)
python train.py --stage 2 --epochs 60 --n_way 3 --n_query 6 --resume checkpoints/checkpoint_epoch_30.pt

# Epochs 61-100: Hard (4-way, 8-query)
python train.py --stage 2 --epochs 100 --n_way 4 --n_query 8 --resume checkpoints/checkpoint_epoch_60.pt
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.34 GiB
```

**Solutions:**

**Solution A: Reduce episodic batch size**
```bash
python train.py --stage 2 --n_way 2 --n_query 2  # Reduce from 4 to 2
```

**Solution B: Enable gradient accumulation**
```bash
python train.py --stage 2 --gradient_accumulation 2
```

**Solution C: Reduce number of workers**
```bash
python train.py --stage 2 --num_workers 2  # Reduce from 4
```

**Solution D: Use mixed precision (should be enabled by default)**
```bash
python train.py --stage 2 --mixed_precision
```

**Solution E: Clear cache before training**
```python
import torch
torch.cuda.empty_cache()
```

### Issue 2: Loss Not Decreasing

**Symptoms:**
```
Epoch 10: Loss 0.8234
Epoch 20: Loss 0.8156
Epoch 30: Loss 0.8089
...
```

**Diagnosis:**

**Step 1: Check learning rate**
```bash
# Try lower LR
python train.py --stage 2 --lr 5e-5

# Try higher LR
python train.py --stage 2 --lr 2e-4
```

**Step 2: Verify data loading**
```python
# test_data.py
from src.datasets.refdet_dataset import RefDetDataset

ds = RefDetDataset(
    data_root='./datasets/train/samples',
    annotations_file='./datasets/train/annotations/annotations.json',
    mode='train'
)

# Check sample
sample = ds[0]
print(f"Query shape: {sample['query_frame'].shape}")
print(f"Bboxes: {sample['bboxes']}")
print(f"Support images: {len(sample['support_images'])} images")

# Verify bboxes are valid (not all zeros)
assert len(sample['bboxes']) > 0, "No bboxes found!"
assert sample['query_frame'].max() > 0, "Query frame is all zeros!"
```

**Step 3: Check augmentation**
```bash
# Disable augmentation temporarily
# Edit src/augmentations/augmentation_config.py
# Set all augmentation probabilities to 0.0
```

**Step 4: Verify model initialization**
```python
# test_model.py
from src.models.yolov8n_refdet import YOLOv8nRefDet

model = YOLOv8nRefDet(yolo_weights='./models/base/yolov8n.pt')
print(f"Model loaded successfully!")
print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

### Issue 3: NaN Losses

**Symptoms:**
```
Epoch 5: Loss nan
```

**Causes & Solutions:**

**Cause A: Gradient explosion**
```bash
# Solution: Lower learning rate
python train.py --stage 2 --lr 5e-5
```

**Cause B: Invalid bboxes**
```python
# Check for invalid bboxes in dataset
# Add validation in refdet_dataset.py:
def validate_bbox(self, bbox):
    x1, y1, x2, y2 = bbox
    assert x1 < x2, f"Invalid bbox: x1 ({x1}) >= x2 ({x2})"
    assert y1 < y2, f"Invalid bbox: y1 ({y1}) >= y2 ({y2})"
    assert x1 >= 0 and x2 <= self.img_size, "Bbox out of bounds"
    assert y1 >= 0 and y2 <= self.img_size, "Bbox out of bounds"
```

**Cause C: Division by zero in loss**
```python
# Add epsilon to prevent division by zero
# In loss functions, replace:
# loss = x / y
# with:
# loss = x / (y + 1e-7)
```

### Issue 4: Slow Training

**Symptoms:**
- Epoch taking >10 minutes on RTX 3090
- GPU utilization <50%

**Solutions:**

**Solution A: Enable mixed precision**
```bash
python train.py --stage 2 --mixed_precision
```

**Solution B: Increase num_workers**
```bash
python train.py --stage 2 --num_workers 8  # Increase from 4
```

**Solution C: Enable frame caching**
```python
# In refdet_dataset.py
ds = RefDetDataset(
    data_root='./datasets/train/samples',
    annotations_file='./datasets/train/annotations/annotations.json',
    cache_frames=True,  # Enable caching
    cache_size=1000,    # Cache 1000 frames
)
```

**Solution D: Use SSD for data storage**
```bash
# Move dataset to faster storage
mv ./datasets /mnt/nvme/datasets
python train.py --stage 2 --data_root /mnt/nvme/datasets/train/samples
```

### Issue 5: Poor Generalization (High Val Loss)

**Symptoms:**
```
Epoch 50:
  Train Loss: 0.2245
  Val Loss: 0.4532  # Much higher than train
```

**Solutions:**

**Solution A: Increase regularization**
```bash
python train.py --stage 2 --weight_decay 0.1  # Increase from 0.05
```

**Solution B: More episodic diversity**
```bash
python train.py --stage 2 --n_way 3 --n_episodes 200
```

**Solution C: Stronger augmentation**
```python
# Edit src/augmentations/augmentation_config.py
# Increase augmentation probabilities
```

**Solution D: Reduce model complexity**
```bash
# Train fewer epochs to prevent overfitting
python train.py --stage 2 --epochs 50  # Instead of 100
```

---

## Best Practices

### 1. Start Simple, Add Complexity

**Phase 1: Baseline**
```bash
python train.py --stage 2 --epochs 10 --n_way 2 --n_query 4
```
- Verify data loading works
- Check loss is decreasing
- Ensure no errors

**Phase 2: Full Training**
```bash
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --mixed_precision
```
- Train to convergence
- Monitor validation loss

**Phase 3: Advanced**
```bash
python train.py --stage 2 --epochs 100 --n_way 3 --n_query 6 --use_triplet --triplet_ratio 0.3
```
- Add triplet training
- Increase episodic complexity

### 2. Always Use Validation Set

```bash
python train.py \
    --stage 2 \
    --data_root ./datasets/train/samples \
    --test_data_root ./datasets/test/samples \
    --test_annotations ./datasets/test/annotations/annotations.json
```

**Benefits:**
- Detect overfitting early
- Select best model based on generalization
- Monitor real-world performance

### 3. Save Checkpoints Frequently

```bash
python train.py --stage 2 --save_interval 5  # Save every 5 epochs
```

**Why:**
- Training may crash (power outage, OOM, etc.)
- You can resume from last checkpoint
- Compare models at different epochs

### 4. Monitor GPU Utilization

```bash
# In another terminal
watch -n 1 nvidia-smi

# Expected:
# GPU Utilization: 80-95%
# Memory Usage: 8-12 GB (RTX 3090)
# Temperature: 60-80°C
```

**Low GPU utilization (<50%):**
- Data loading is bottleneck → Increase `num_workers`
- CPU preprocessing too slow → Use AlbumentationsX (already enabled)
- Enable `pin_memory=True` in DataLoader (already enabled)

### 5. Use Git for Version Control

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit"

# Before major changes
git checkout -b experiment/higher_lr
python train.py --stage 2 --lr 2e-4

# If successful
git commit -am "Improved convergence with lr=2e-4"
git checkout main
git merge experiment/higher_lr

# If failed
git checkout main
git branch -D experiment/higher_lr
```

### 6. Document Your Experiments

Create an experiments log:

```markdown
# experiments.md

## Experiment 1: Baseline
- **Date**: 2025-11-11
- **Config**: stage=2, epochs=100, n_way=2, n_query=4
- **Result**: Val loss 0.3456, mAP@0.5 52.3%
- **Notes**: Good baseline, but slow convergence

## Experiment 2: Higher LR
- **Date**: 2025-11-12
- **Config**: Same as Exp 1, lr=2e-4 (default: 1e-4)
- **Result**: Val loss 0.3234, mAP@0.5 54.1%
- **Notes**: Faster convergence! Use this LR going forward

## Experiment 3: Triplet Training
- **Date**: 2025-11-13
- **Config**: Exp 2 + use_triplet, triplet_ratio=0.3
- **Result**: Val loss 0.3012, mAP@0.5 56.7%
- **Notes**: Best so far! Triplet loss helps feature separation
```

---

## Production Checklist

### Before Deployment

**✅ Model Training**
- [ ] Stage 2 trained to convergence (loss <0.3)
- [ ] Validation loss within 0.05 of train loss
- [ ] Checkpoints saved and backed up
- [ ] Best model identified

**✅ Model Evaluation**
- [ ] Evaluate on test set: `python evaluate.py --checkpoint best_model.pt`
- [ ] mAP@0.5 > 50% (3-shot)
- [ ] Precision > 60%
- [ ] Recall > 55%

**✅ Inference Testing**
- [ ] Test inference on sample videos
- [ ] Verify detections are correct
- [ ] Check inference speed (target: real-time on Jetson Xavier NX)

**✅ Model Optimization**
- [ ] Export to ONNX (coming soon)
- [ ] Quantization for edge deployment (INT8)
- [ ] Test on Jetson Xavier NX hardware

**✅ Code Quality**
- [ ] All tests passing: `pytest src/tests/ -v`
- [ ] No critical warnings
- [ ] Documentation updated

### Deployment Commands

**Export Model:**
```python
# export.py (coming soon)
import torch
from src.models.yolov8n_refdet import YOLOv8nRefDet

model = YOLOv8nRefDet()
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# Export to ONNX
torch.onnx.export(
    model,
    (query_input, support_input),
    'model.onnx',
    opset_version=14,
)
```

**Deploy to Jetson Xavier NX:**
```bash
# Transfer model
scp best_model.pt jetson@192.168.1.100:/home/jetson/models/

# SSH to Jetson
ssh jetson@192.168.1.100

# Install dependencies
pip install torch torchvision

# Run inference
python inference.py --checkpoint /home/jetson/models/best_model.pt
```

---

## Summary

**Key Takeaways:**

1. **Start with Stage 2** using pretrained YOLOv8n weights
2. **Use default hyperparameters first**, then tune if needed
3. **Always use a validation set** to monitor generalization
4. **Enable mixed precision** for faster training
5. **Save checkpoints frequently** to avoid losing progress
6. **Monitor training closely** and adjust if needed
7. **Document experiments** to track what works

**Typical Training Timeline:**

| Day | Activity | Duration |
|-----|----------|----------|
| **Day 1** | Data preparation & validation | 2 hours |
| **Day 1** | Stage 2 training (100 epochs) | 5 hours |
| **Day 2** | Evaluation & analysis | 1 hour |
| **Day 2** | Stage 3 fine-tuning (30 epochs) | 1 hour |
| **Day 2** | Final evaluation & optimization | 2 hours |
| **Total** | | **11 hours** |

**Next Steps:**

1. ✅ Complete this training guide
2. ⏭️ Follow the training workflow
3. ⏭️ Evaluate your model
4. ⏭️ Deploy to production

**Need Help?**
- Check `docs/TRAINING_PIPELINE_GUIDE.md` for architecture details
- Review test files in `src/tests/` for code examples
- Open an issue on GitHub for bugs or questions

---

**Good luck with your training! 🚀**
