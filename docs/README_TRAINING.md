# YOLOv8n-RefDet Training Pipeline

Complete training pipeline for few-shot reference-based object detection with YOLOv8n-RefDet.

## 📋 Overview

This pipeline implements a **3-stage training approach** for detecting novel objects using reference images:

1. **Stage 1**: Base pre-training (optional)
2. **Stage 2**: Few-shot meta-learning (main training)
3. **Stage 3**: Fine-tuning with triplet loss

### Key Features

- ✅ **Episodic Training**: N-way K-shot Q-query sampling for few-shot learning
- ✅ **Video Frame Extraction**: Automatic extraction from drone videos with caching
- ✅ **Hybrid Augmentation**: Ultralytics (Mosaic/MixUp) + AlbumentationsX (10-23x faster)
- ✅ **Multi-Scale Loss**: WIoU + BCE + DFL + SupCon + CPE + Triplet
- ✅ **Mixed Precision**: Automatic mixed precision training (AMP)
- ✅ **Gradient Accumulation**: Support for large effective batch sizes
- ✅ **Layerwise Learning Rates**: Different LRs for different components

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
# YOLOv8n weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. Verify Setup

```bash
python test_pipeline.py
```

This will test:
- Dataset loading
- Episodic sampler
- Collate function
- Model forward pass
- Loss computation

### 4. Start Training

```bash
# Stage 2: Few-shot meta-learning (recommended start)
python train.py \
  --data_root ./datasets/train/samples \
  --annotations ./datasets/train/annotations/annotations.json \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4 \
  --mixed_precision
```

### 5. Evaluate

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --data_root ./datasets/test/samples \
  --annotations ./datasets/test/annotations/annotations.json
```

## 📁 Project Structure

```
zalo_ai_2025/
├── datasets/
│   ├── refdet_dataset.py        # Dataset + episodic sampler
│   └── collate.py               # Batch collation with augmentation
├── src/
│   ├── models/                  # YOLOv8n-RefDet components
│   │   ├── yolov8n_refdet.py   # Main model
│   │   ├── dinov2_encoder.py   # Support encoder
│   │   ├── yolov8_backbone.py  # Query encoder
│   │   ├── cheaf_fusion.py       # Feature fusion
│   │   └── dual_head.py        # Detection head
│   ├── losses/                  # Loss functions
│   │   ├── combined_loss.py    # Main loss
│   │   ├── wiou_loss.py        # Bbox loss
│   │   ├── bce_loss.py         # Classification loss
│   │   ├── dfl_loss.py         # Distribution focal loss
│   │   ├── supervised_contrastive_loss.py
│   │   ├── cpe_loss.py         # Contrastive proposal encoding
│   │   └── triplet_loss.py     # Stage 3 loss
│   └── augmentations/           # Data augmentation
│       ├── augmentation_config.py
│       ├── query_augmentation.py
│       ├── support_augmentation.py
│       └── temporal_augmentation.py
├── training/
│   ├── trainer.py               # Main training loop
│   └── loss_utils.py            # Loss preparation utilities
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── test_pipeline.py             # Pipeline verification
├── TRAINING_PIPELINE_GUIDE.md   # Detailed documentation
└── requirements.txt
```

## 📊 Dataset Format

### Directory Structure
```
datasets/
├── train/
│   ├── samples/
│   │   ├── Backpack_0/
│   │   │   ├── drone_video.mp4          # Query frames
│   │   │   └── object_images/           # Support images
│   │   │       ├── img_1.jpg
│   │   │       ├── img_2.jpg
│   │   │       └── img_3.jpg
│   │   └── ...
│   └── annotations/
│       └── annotations.json
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
          {"frame": 3483, "x1": 321, "y1": 0, "x2": 381, "y2": 12},
          ...
        ]
      }
    ]
  }
]
```

## 🎯 Training Stages

### Stage 2: Few-Shot Meta-Learning (Main)

**Purpose**: Learn to detect novel objects from few examples

```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4 \
  --n_episodes 100 \
  --supcon_weight 1.0 \
  --cpe_weight 0.5 \
  --lr 1e-4 \
  --mixed_precision
```

**Loss Components**:
- WIoU (7.5): Bounding box regression
- BCE (0.5): Classification
- DFL (1.5): Distribution focal loss
- SupCon (1.0): Supervised contrastive learning
- CPE (0.5): Contrastive proposal encoding

**Training Time**: ~2-4 hours on RTX 3090

### Stage 3: Fine-Tuning (Optional)

**Purpose**: Fine-tune on specific objects, prevent forgetting

```bash
python train.py \
  --stage 3 \
  --epochs 30 \
  --supcon_weight 0.5 \
  --cpe_weight 0.3 \
  --triplet_weight 0.2 \
  --resume ./checkpoints/best_model.pt
```

**Additional Loss**:
- Triplet (0.2): Prevents catastrophic forgetting

**Training Time**: ~30-60 minutes

## 🔧 Key Hyperparameters

### Episodic Sampling
- **n_way**: 2-4 (number of classes per episode)
- **n_query**: 4-8 (query frames per class)
- **n_episodes**: 100-200 (episodes per epoch)

### Learning Rates (Layerwise)
- DINOv2: `1e-5` (frozen or slow)
- YOLOv8 Backbone: `1e-4` (base)
- CHEAF Fusion: `2e-4` (faster)
- Detection Head: `2e-4` (faster)

### Loss Weights
**Stage 2**:
- bbox: 7.5, cls: 0.5, dfl: 1.5
- supcon: 1.0, cpe: 0.5

**Stage 3**:
- bbox: 7.5, cls: 0.5, dfl: 1.5
- supcon: 0.5, cpe: 0.3, triplet: 0.2

## 📈 Expected Performance

- **1-shot mAP@0.5**: 35-45% (Stage 2)
- **3-shot mAP@0.5**: 50-60% (Stage 2)
- **5-shot mAP@0.5**: 60-70% (Stage 3)

## 💡 Tips & Best Practices

1. **Start with Stage 2**: Skip Stage 1 if using pretrained YOLOv8n
2. **Use validation set**: Monitor generalization to novel classes
3. **Enable mixed precision**: ~2x speedup, half memory usage
4. **Cache support features**: Use `model.set_reference_images()` for inference
5. **Monitor loss components**: Ensure all losses contribute
6. **Save frequently**: Keep multiple checkpoints
7. **Adjust n_way**: Balance diversity (higher) vs stability (lower)

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --n_way 2 --n_query 2 --gradient_accumulation 2
```

### Loss Not Decreasing
- Check learning rate (try 1e-5 to 1e-3)
- Verify data augmentation isn't too aggressive
- Try Stage 1 pre-training first

### Video Loading Slow
- Enable frame caching in dataset
- Use SSD for data storage
- Pre-extract frames to disk

## 📚 Documentation

- **[TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md)**: Comprehensive training guide
- **[augmentation-guide.md](augmentation-guide.md)**: Augmentation strategies
- **[loss-functions-guide.md](loss-functions-guide.md)**: Loss function details
- **[models_selections.md](models_selections.md)**: Model architecture choices
- **[implementation-guide.md](implementation-guide.md)**: Implementation details

## 🔗 Components Used

### Model Architecture
- **YOLOv8n**: Ultralytics detection backbone
- **DINOv2**: Meta AI vision transformer for support encoding
- **CHEAF Fusion**: Cross-scale feature combination
- **Dual Head**: Base + novel class detection

### Key Technologies
- PyTorch 2.0+ (AMP, gradient checkpointing)
- Ultralytics (Mosaic, MixUp augmentation)
- AlbumentationsX (10-23x faster augmentation)
- OpenCV (video frame extraction)
- timm (DINOv2 pretrained models)

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@misc{yolov8n-refdet-pipeline,
  title={YOLOv8n-RefDet Training Pipeline},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/zalo_ai_2025}}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Proper Task-Aligned Assigner
- [ ] Temporal consistency loss
- [ ] TensorBoard logging
- [ ] Distributed training (DDP)
- [ ] Model quantization
- [ ] ONNX export

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- Meta AI for DINOv2
- Few-shot object detection research community

---

**Need Help?** Check [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md) for detailed documentation.
