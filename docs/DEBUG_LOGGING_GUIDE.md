# Debug Logging Guide for YOLOv8n-RefDet Trainer

This guide explains how to use the detailed debug logging features added to the training pipeline.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Debug Features](#debug-features)
3. [Understanding the Logs](#understanding-the-logs)
4. [Common Debugging Scenarios](#common-debugging-scenarios)
5. [Performance Considerations](#performance-considerations)

---

## Quick Start

### Enable Debug Mode

Add the `--debug` flag to your training command:

```bash
# Basic debug training
python train.py --stage 2 --epochs 10 --debug

# Debug with specific batch size
python train.py --stage 2 --epochs 10 --n_way 2 --n_query 4 --batch_size 2 --debug

# Debug with triplet loss
python train.py --stage 2 --epochs 10 --use_triplet --triplet_ratio 0.3 --debug
```

### Debug Output Locations

When debug mode is enabled:
- **Console**: Real-time debug information
- **Log File**: `./checkpoints/training_debug.log` (persistent record)

---

## Debug Features

### 1. Model Summary

At the start of training, prints detailed model architecture:

```
==================================================
MODEL SUMMARY
==================================================
Total Parameters: 31,520,000
Trainable Parameters: 20,340,000 (64.53%)
Frozen Parameters: 11,180,000 (35.47%)

Module Breakdown:
  backbone:
    Total: 3,160,000
    Trainable: 3,160,000
    Frozen: 0
  support_encoder:
    Total: 21,770,000
    Trainable: 0
    Frozen: 21,770,000
  ...
```

### 2. Batch-Level Logging

Logs detailed information for:
- **First batch** of each epoch (batch 0)
- **Every 50th batch** during training

#### Batch Contents
```
======================================================================
BATCH 0 - Type: detection
======================================================================

Batch Contents:
  support_images:
    Shape: torch.Size([2, 3, 3, 256, 256])
    Dtype: torch.float32
    Device: cuda:0
    Range: [0.0000, 1.0000]
    Mean: 0.4523
    Std: 0.2156
    Has NaN: False
    Has Inf: False
  
  query_images:
    Shape: torch.Size([8, 3, 640, 640])
    ...
```

### 3. Forward Pass Logging

#### Detection Forward Pass
```
Detection Forward Pass:
  Support images: N=2, K=3, C=3, H=256, W=256
  Query images: torch.Size([8, 3, 640, 640])

Model Outputs:
  pred_bboxes: torch.Size([8, 8400, 4]), range=[0.0125, 638.4523]
  pred_scores: torch.Size([8, 8400]), range=[0.0001, 0.9856]
  pred_class_ids: torch.Size([8, 8400]), range=[0.0000, 1.0000]
  ...
```

#### Triplet Forward Pass
```
Triplet Forward Pass:
  Anchor images: torch.Size([8, 3, 256, 256])
  Positive images: torch.Size([8, 3, 640, 640])
  Negative images: torch.Size([8, 3, 640, 640])

Triplet Features:
  Anchor: torch.Size([8, 384]), norm=12.3456
  Positive: torch.Size([8, 256]), norm=8.9012
  Negative: torch.Size([8, 256]), norm=7.6543
```

### 4. Loss Component Logging

```
Loss Components:
  Total Loss: 2.345678
  bbox_loss: 1.234567
  cls_loss: 0.567890
  dfl_loss: 0.345678
  supcon_loss: 0.197543
```

### 5. Gradient Statistics

```
Gradient Norms:
  Total Grad Norm: 3.4567e+01
  backbone.conv1.weight: 5.6789e+00
  head.cls_conv.weight: 4.3210e+00
  head.reg_conv.weight: 3.8901e+00
  fusion.cross_attn.weight: 2.9876e+00
  ...
```

### 6. Gradient Clipping

```
Gradient Clipping:
  Norm before clip: 5.6789e+01
  Norm after clip: 1.0000e+00
  Clip threshold: 1.0
```

### 7. Parameter Updates

```
Parameter Updates (Step 123):
  Learning Rate: 1.000000e-04
```

### 8. NaN/Inf Detection

Automatic detection and detailed reporting:

```
⚠️ NaN/Inf gradient in backbone.layer3.conv.weight
   Grad stats: min=-inf, max=1.2345e+10, mean=nan

❌ NaN/Inf gradients detected at batch 45. Skipping optimizer step.
```

---

## Understanding the Logs

### Normal Training Patterns

#### Healthy Loss Values
- **Total Loss**: Should decrease over time (typically 2-5 → 0.5-2.0)
- **bbox_loss**: Usually largest component (0.5-2.0)
- **cls_loss**: Should be relatively low (0.1-0.5)
- **dfl_loss**: Moderate values (0.2-1.0)
- **supcon_loss**: Variable, but stable (0.1-1.0)

#### Healthy Gradient Norms
- **Total Grad Norm**: 1-100 (before clipping)
- **After Clipping**: Should match `gradient_clip_norm` parameter
- **Individual Layer Norms**: 0.001-10 (varies by layer depth)

### Warning Signs

#### Loss Issues
```
❌ ERROR: NaN or Inf detected in loss at batch 45
```
**Possible causes:**
- Learning rate too high
- Data augmentation creating invalid inputs
- Numerical instability in loss computation

#### Gradient Issues
```
⚠️ NaN/Inf gradient in backbone.layer3.conv.weight
```
**Possible causes:**
- Exploding gradients (reduce learning rate or increase clipping)
- Invalid loss computation
- Mixed precision issues

#### Data Issues
```
⚠️ query_images: Has NaN: True
```
**Possible causes:**
- Corrupted data files
- Data augmentation bug
- Normalization issues

---

## Common Debugging Scenarios

### Scenario 1: Training Loss Not Decreasing

**Debug Steps:**

1. Check learning rate:
   ```
   Parameter Updates (Step X):
     Learning Rate: 1.000000e-04  # Is this too small?
   ```

2. Check gradient norms:
   ```
   Gradient Norms:
     Total Grad Norm: 1.2345e-05  # Too small = learning rate issue
   ```

3. Check loss components:
   ```
   Loss Components:
     bbox_loss: 0.000001  # Too small = wrong weight or scale
     cls_loss: 5.678900   # Dominant loss = imbalanced
   ```

**Solutions:**
- Increase learning rate if gradients are tiny
- Adjust loss weights if one component dominates
- Check if model weights are frozen accidentally

### Scenario 2: NaN/Inf During Training

**Debug Steps:**

1. Enable debug mode to see exact batch:
   ```bash
   python train.py --stage 2 --epochs 10 --debug
   ```

2. Check batch contents for NaN:
   ```
   Batch Contents:
     query_images: Has NaN: True  # Found the issue!
   ```

3. Check gradients before optimizer step:
   ```
   Gradient Norms:
     backbone.conv1.weight: inf  # Exploding gradients
   ```

**Solutions:**
- Reduce learning rate by 10x
- Increase gradient clipping (try 0.5 or 0.1)
- Check data loading and augmentation
- Try disabling mixed precision: `--no-mixed_precision`

### Scenario 3: Gradients Vanishing

**Debug Steps:**

1. Check gradient norms:
   ```
   Gradient Norms:
     Total Grad Norm: 1.2345e-10  # Too small!
   ```

2. Check parameter updates:
   ```
   Parameter Updates:
     Learning Rate: 1.0e-06  # Very small
   ```

**Solutions:**
- Increase learning rate
- Check if too many layers are frozen
- Verify loss scaling is appropriate

### Scenario 4: Slow Training

**Debug Steps:**

1. Check batch shapes:
   ```
   Batch Contents:
     query_images: Shape: torch.Size([2, 3, 640, 640])  # Small batch
   ```

2. Check gradient accumulation:
   ```
   Gradient Accumulation: 1  # No accumulation
   ```

**Solutions:**
- Increase batch size if memory allows
- Use gradient accumulation: `--gradient_accumulation 4`
- Reduce `num_aug` parameter
- Enable caching if disabled

---

## Performance Considerations

### Debug Mode Overhead

Debug logging adds **5-10% overhead** to training time:
- Batch info logging: ~2-3%
- Gradient statistics: ~3-5%
- File I/O: ~1-2%

### When to Use Debug Mode

✅ **Use debug mode when:**
- Initial training setup and validation
- Investigating loss/gradient issues
- Debugging data loading problems
- Testing new augmentations or loss functions

❌ **Disable debug mode when:**
- Running production training (>100 epochs)
- Training is stable and working correctly
- GPU memory is limited

### Log File Management

Debug logs can grow large:
- **~1 MB per epoch** with full debug logging
- **~100 MB for 100 epochs**

Clean up old logs periodically:
```bash
# Keep only last 5 debug logs
cd checkpoints
ls -t training_debug.log* | tail -n +6 | xargs rm -f
```

---

## Additional Utilities

### Print Model Summary Programmatically

```python
from src.training.trainer import RefDetTrainer

# After creating trainer
trainer.print_model_summary()
```

### Log Parameter Statistics

```python
# Only works in debug mode
trainer.log_parameter_statistics()
```

### Get Model Summary Dictionary

```python
summary = trainer.get_model_summary()
print(f"Total params: {summary['total_parameters']:,}")
print(f"Trainable: {summary['trainable_parameters']:,}")
```

---

## Troubleshooting

### Debug logs not appearing

**Check:**
1. `--debug` flag is added
2. Logs are in `./checkpoints/training_debug.log`
3. Console handler is working (you should see timestamps)

### Log file too large

**Solutions:**
1. Reduce logging frequency (modify `log_interval`)
2. Log only first few epochs
3. Use log rotation

### Too much output

**Solutions:**
1. Increase logging interval in code:
   ```python
   # In trainer.py, change:
   if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
   # to:
   if self.debug_mode and (batch_idx == 0 or batch_idx % 200 == 0):
   ```

---

## Example Commands

```bash
# Debug with minimal settings (fast iteration)
python train.py --stage 2 --epochs 5 --n_episodes 50 --debug

# Debug specific loss components
python train.py --stage 2 --epochs 10 --bbox_weight 10.0 --cls_weight 1.0 --debug

# Debug with gradient clipping tuning
python train.py --stage 2 --epochs 10 --gradient_clip_norm 0.5 --debug

# Debug triplet loss
python train.py --stage 2 --epochs 10 --use_triplet --triplet_ratio 0.3 --debug

# Debug with cache disabled (isolate data issues)
python train.py --stage 2 --epochs 5 --disable_cache --debug
```

---

## Summary

The debug logging system provides comprehensive visibility into:
- ✅ Data loading and batch composition
- ✅ Model forward pass outputs
- ✅ Loss computation and components
- ✅ Gradient flow and statistics
- ✅ Parameter updates and learning rates
- ✅ NaN/Inf detection and recovery

Use debug mode liberally during development, and disable it for production training runs.

For questions or issues, check:
1. Console output for immediate feedback
2. `training_debug.log` for detailed history
3. Model summary for architecture verification
