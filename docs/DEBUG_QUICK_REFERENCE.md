# Debug Logging Quick Reference

## Enable Debug Mode

```bash
python train.py --stage 2 --epochs 10 --debug
```

## What Gets Logged

### 📊 Batch-Level (every 50 batches + first batch)
- **Batch Contents**: Shapes, dtypes, ranges, NaN/Inf checks
- **Model Outputs**: Prediction shapes and ranges
- **Loss Values**: All components (bbox, cls, dfl, supcon)
- **Gradients**: Norms per layer, total norm
- **Clipping**: Before/after gradient norms
- **Parameters**: Learning rate updates

### 🔍 Always Logged
- **Model Summary**: Parameter counts per module
- **NaN/Inf Detection**: Immediate alerts with full context
- **Error Recovery**: Batch skipping with detailed error info

## Output Locations

- **Console**: `[HH:MM:SS] LEVEL - message`
- **File**: `./checkpoints/training_debug.log`

## Common Patterns

### ✅ Healthy Training
```
Loss Components:
  Total Loss: 2.3456 → 1.2345 → 0.8901 (decreasing)
  bbox_loss: 1.234 (largest, stable)
  cls_loss: 0.456 (moderate)
  
Gradient Norms:
  Total Grad Norm: 3.45e+01 (reasonable)
  After clip: 1.00e+00 (clipped to threshold)
```

### ⚠️ Warning Signs
```
# Loss exploding
Total Loss: 2.3 → 5.6 → 15.2 → NaN

# Gradients vanishing
Total Grad Norm: 1.23e-08 (too small!)

# Gradients exploding
Total Grad Norm: 1.23e+10 (too large!)

# Data issues
query_images: Has NaN: True
```

## Quick Fixes

| Problem | Solution |
|---------|----------|
| NaN loss | Reduce LR 10x, increase grad clip to 0.5 |
| Loss not decreasing | Check LR, check frozen layers |
| Gradients vanishing | Increase LR, check loss weights |
| Gradients exploding | Reduce LR, increase grad clip |
| Data NaN | Check augmentations, check data files |

## Debug Commands

```bash
# Minimal debug (5 epochs, fast)
python train.py --stage 2 --epochs 5 --n_episodes 50 --debug

# Debug gradients (strong clipping)
python train.py --stage 2 --epochs 10 --gradient_clip_norm 0.5 --debug

# Debug triplet loss
python train.py --stage 2 --epochs 10 --use_triplet --debug

# Debug data (disable cache)
python train.py --stage 2 --epochs 5 --disable_cache --debug
```

## Performance Impact

- **Overhead**: ~5-10% slower training
- **Log Size**: ~1 MB per epoch
- **When to use**: Development, debugging, initial validation
- **When to disable**: Production runs (>50 epochs)

## Key Metrics to Watch

### Loss Components (expected ranges)
- `total_loss`: 2-5 → 0.5-2.0 (should decrease)
- `bbox_loss`: 0.5-2.0 (largest component)
- `cls_loss`: 0.1-0.5 (classification)
- `dfl_loss`: 0.2-1.0 (distribution focal)
- `supcon_loss`: 0.1-1.0 (contrastive)

### Gradient Norms (healthy ranges)
- **Before clipping**: 1-100
- **After clipping**: Equals `gradient_clip_norm` parameter
- **Per-layer**: 0.001-10 (varies by depth)

### Data Ranges
- **Images**: [0.0, 1.0] or [-2.0, 2.0] (normalized)
- **Bboxes**: [0, 640] (pixel coordinates)
- **Scores**: [0.0, 1.0] (probabilities)
- **Classes**: [0, N-1] (integer class IDs)
