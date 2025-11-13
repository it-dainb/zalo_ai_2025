# YOLOv8n-RefDet Training Pipeline Documentation

## Overview

This training pipeline implements a complete 3-stage training approach for few-shot reference-based object detection using YOLOv8n-RefDet.

## Architecture Components

### 1. Model (`src/models/yolov8n_refdet.py`)
- **DINOv2 Support Encoder**: Extracts features from reference images (518×518)
- **YOLOv8n Backbone**: Extracts features from query images (640×640)
- **CHEAF Fusion Module**: Combines query and support features across scales
- **Dual Detection Head**: Predicts bounding boxes and classes
- **Total Parameters**: ~10.4M (well under 50M budget)

### 2. Loss Functions (`src/losses/combined_loss.py`)
Stage-specific loss weighting:
- **Stage 1** (Base pre-training): bbox + cls + dfl
- **Stage 2** (Few-shot meta): + supcon + cpe
- **Stage 3** (Fine-tuning): reduced contrastive + triplet

Loss components:
- **WIoU v3** (7.5): Bounding box regression
- **BCE** (0.5): Classification
- **DFL** (1.5): Distribution Focal Loss
- **SupCon** (1.0→0.5): Supervised contrastive learning
- **CPE** (0.5→0.3): Contrastive proposal encoding
- **Triplet** (0.2): Prevents catastrophic forgetting (Stage 3)

### 3. Augmentations (`src/augmentations/`)
**Query Path** (drone frames):
- Ultralytics: Mosaic (prob=1.0), MixUp (prob=0.15)
- AlbumentationsX: Color, geometric, blur, erasing (10-23x faster)
- Target: 640×640

**Support Path** (reference images):
- Weak mode: Conservative augmentation
- Strong mode: Aggressive for contrastive learning
- Target: 518×518 for DINOv2

## Dataset Structure

```
datasets/
├── train/
│   ├── samples/
│   │   ├── Backpack_0/
│   │   │   ├── drone_video.mp4          # Query frames source
│   │   │   └── object_images/           # Support images
│   │   │       ├── img_1.jpg
│   │   │       ├── img_2.jpg
│   │   │       └── img_3.jpg
│   │   └── ...
│   └── annotations/
│       └── annotations.json             # Frame-level bbox annotations
└── test/
    └── (same structure)
```

### Annotations Format
```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          {
            "frame": 3483,
            "x1": 321, "y1": 0,
            "x2": 381, "y2": 12
          },
          ...
        ]
      }
    ]
  }
]
```

## Training Pipeline

### Key Components

1. **RefDetDataset** (`datasets/refdet_dataset.py`)
   - Parses annotations.json
   - Extracts video frames on-the-fly with caching
   - Loads support images from object_images/
   - Returns: query_frame, bboxes, class_id, support_images

2. **EpisodicBatchSampler** (`datasets/refdet_dataset.py`)
   - N-way K-shot Q-query sampling
   - Each episode samples N classes
   - K support images per class (3 in dataset)
   - Q query frames per class

3. **RefDetCollator** (`datasets/collate.py`)
   - Applies augmentations to query and support paths
   - Prepares batches in model-ready format
   - Handles variable number of objects per image

4. **RefDetTrainer** (`training/trainer.py`)
   - Main training loop
   - Mixed precision training (AMP)
   - Gradient accumulation
   - Checkpointing and logging

### Training Stages

#### Stage 1: Base Pre-training (Optional)
```bash
python train.py \
  --stage 1 \
  --epochs 50 \
  --n_way 2 \
  --n_query 4 \
  --bbox_weight 7.5 \
  --cls_weight 0.5 \
  --dfl_weight 1.5
```

**Purpose**: Learn general detection features on base classes (e.g., COCO)
**Loss**: Detection losses only (bbox + cls + dfl)
**Note**: Can skip if using pretrained YOLOv8n

#### Stage 2: Few-Shot Meta-Learning (Main Training)
```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4 \
  --n_episodes 100 \
  --supcon_weight 1.0 \
  --cpe_weight 0.5 \
  --mixed_precision
```

**Purpose**: Learn to detect novel objects from few examples
**Loss**: Full loss stack (detection + contrastive)
**Key**: Episodic training with N-way K-shot Q-query format

#### Stage 3: Fine-Tuning
```bash
python train.py \
  --stage 3 \
  --epochs 30 \
  --n_way 2 \
  --n_query 4 \
  --supcon_weight 0.5 \
  --cpe_weight 0.3 \
  --triplet_weight 0.2 \
  --resume checkpoints/best_model.pt
```

**Purpose**: Fine-tune on specific objects, prevent forgetting
**Loss**: Reduced contrastive + triplet loss

## Usage Examples

### Basic Training
```bash
# Stage 2 meta-learning (recommended starting point)
python train.py \
  --data_root ./datasets/train/samples \
  --annotations ./datasets/train/annotations/annotations.json \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4
```

### Advanced Training with Custom Settings
```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --n_way 3 \
  --n_query 8 \
  --n_episodes 200 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --batch_size 8 \
  --gradient_accumulation 2 \
  --mixed_precision \
  --num_workers 8 \
  --checkpoint_dir ./checkpoints_stage2 \
  --save_interval 5
```

### Resume Training
```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --resume ./checkpoints/checkpoint_epoch_50.pt
```

### Evaluation
```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --data_root ./datasets/test/samples \
  --annotations ./datasets/test/annotations/annotations.json \
  --n_way 2 \
  --n_query 4 \
  --n_episodes 50
```

## Hyperparameter Guide

### Learning Rates (Layerwise)
- DINOv2: `lr * 0.1` (1e-5) - frozen/slow learning
- YOLOv8 Backbone: `lr` (1e-4) - base rate
- CHEAF Fusion: `lr * 2.0` (2e-4) - faster learning
- Detection Head: `lr * 2.0` (2e-4) - faster learning

### Loss Weights (Recommended)
**Stage 2** (Few-shot meta):
- bbox: 7.5
- cls: 0.5
- dfl: 1.5
- supcon: 1.0
- cpe: 0.5

**Stage 3** (Fine-tuning):
- bbox: 7.5
- cls: 0.5
- dfl: 1.5
- supcon: 0.5 (reduced)
- cpe: 0.3 (reduced)
- triplet: 0.2 (new)

### Episodic Sampling
- **N-way**: 2-4 classes per episode
- **K-shot**: 3 (fixed by dataset structure)
- **Q-query**: 4-8 frames per class
- **Episodes/epoch**: 100-200

### Training Duration
- **Stage 2**: 50-100 epochs (~2-4 hours on RTX 3090)
- **Stage 3**: 20-30 epochs (~30-60 minutes)

## Expected Performance

### Few-Shot Detection Metrics
- **1-shot mAP@0.5**: 35-45% (after Stage 2)
- **3-shot mAP@0.5**: 50-60% (after Stage 2)
- **5-shot mAP@0.5**: 60-70% (after Stage 3)

### Training Speed
- **Mixed precision**: ~2x speedup
- **Batch size**: Limited by N*Q (episodic)
- **Gradient accumulation**: Use if GPU memory limited

## Troubleshooting

### Out of Memory
- Reduce `n_way` or `n_query`
- Enable `gradient_accumulation`
- Reduce `num_workers`
- Disable `mixed_precision` (paradoxically can help)

### Loss Not Decreasing
- Check learning rate (try 1e-5 to 1e-3)
- Verify data augmentation isn't too aggressive
- Ensure matched predictions exist (check target matching)
- Try Stage 1 pre-training first

### Poor Generalization
- Increase episodic diversity (higher `n_way`)
- More episodes per epoch
- Stronger augmentation on support path
- Add more novel classes to training set

### Video Frame Extraction Slow
- Enable frame caching in dataset
- Increase cache size
- Use SSD for data storage
- Pre-extract frames to disk

## File Structure

```
zalo_ai_2025/
├── datasets/
│   ├── refdet_dataset.py        # Dataset and sampler
│   └── collate.py               # Batch collation and augmentation
├── src/
│   ├── models/                  # Model components
│   ├── losses/                  # Loss functions
│   └── augmentations/           # Data augmentation
├── training/
│   ├── trainer.py               # Training loop
│   └── loss_utils.py            # Loss preparation utilities
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
└── requirements.txt             # Dependencies
```

## Dependencies

Key packages:
- torch >= 2.0
- torchvision
- ultralytics
- albumentations
- opencv-python
- timm (for DINOv2)
- scipy (for Hungarian matching)
- tqdm

Install:
```bash
pip install -r requirements.txt
```

## Citation & References

Based on:
- YOLOv8: Ultralytics (2023)
- DINOv2: Meta AI (2023)
- Few-Shot Object Detection: Meta-YOLO, FSOD, RefDet

## Tips for Best Results

1. **Start with Stage 2**: Skip Stage 1 if using pretrained YOLOv8
2. **Use validation set**: Monitor generalization to novel classes
3. **Experiment with N-way**: Balance between diversity and stability
4. **Cache support features**: Use model.set_reference_images() for inference
5. **Monitor loss components**: Ensure all losses contribute meaningfully
6. **Save frequently**: Keep multiple checkpoints for comparison
7. **Mixed precision**: Almost always beneficial for speed and memory

## Future Improvements

- [ ] Implement proper Task-Aligned Assigner for target matching
- [ ] Add support for video sequence training (temporal consistency)
- [ ] Implement online hard example mining
- [ ] Add TensorBoard logging
- [ ] Pre-extract and cache all video frames
- [ ] Support for distributed training (DDP)
- [ ] Implement model quantization for deployment
- [ ] Add ONNX export for production inference
