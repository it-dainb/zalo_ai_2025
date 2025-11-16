# BBox Loss Fix - Quick Reference

## Problem
**BBox loss oscillating 2.40-2.55 with no downward trend, while other losses decrease normally**

## Root Cause
**bbox_weight=7.5 is too high** (15x higher than cls_weight=0.5)
- Amplifies small WIoU fluctuations
- Creates gradient imbalance
- Makes optimization harder

## Quick Fix

### Option 1: Restart Training with Lower Weight (Recommended)
```bash
python train.py --stage 2 --epochs 10 \
    --bbox_weight 2.0 \
    --n_way 2 --n_query 4 \
    --lr 1e-4
```

### Option 2: Continue Current Training (if far into epochs)
Edit `train.py` line where default is set:
```python
# Change from:
parser.add_argument('--bbox_weight', type=float, default=7.5,

# To:
parser.add_argument('--bbox_weight', type=float, default=2.0,
```

Then restart from checkpoint:
```bash
python train.py --stage 2 --resume checkpoints/stage2/best_model.pt \
    --bbox_weight 2.0
```

## Expected Results After Fix

### Immediate (Steps 0-100)
- BBox loss: 2.5 → 2.2 (clear downward trend)
- Less volatility (smoother curve)

### Short-term (Epoch 1)
- BBox loss: ~1.8-2.0
- mAP@0.5: > 0.05 (some detections working)
- Total loss: continues decreasing

### Long-term (Epoch 5)
- BBox loss: ~1.5
- mAP@0.5: > 0.15
- Stable training progression

## Why This Works

### Current State
```
bbox_weight=7.5 → Raw WIoU 0.32 becomes weighted 2.40
                   Raw WIoU 0.34 becomes weighted 2.55
                   Change: 0.15 (dominates total loss)
```

### After Fix
```
bbox_weight=2.0 → Raw WIoU 0.32 becomes weighted 0.64
                  Raw WIoU 0.34 becomes weighted 0.68
                  Change: 0.04 (balanced with other losses)
```

## Alternative Weight Configurations

### Conservative (if still unstable)
```bash
--bbox_weight 1.5 --cls_weight 0.5 --supcon_weight 1.0
```

### Balanced (recommended)
```bash
--bbox_weight 2.0 --cls_weight 0.5 --supcon_weight 1.0
```

### Aggressive (if confident in features)
```bash
--bbox_weight 3.0 --cls_weight 0.5 --supcon_weight 0.5
```

## Monitoring

After restart, check at step 100:
```
✅ Good: bbox_loss decreasing, oscillations reduced
⚠️  Concern: bbox_loss still flat → check anchor assignment
❌ Bad: bbox_loss increasing → reduce weight further to 1.5
```

## Additional Diagnostics (if still not working)

Add to training loop to check for deeper issues:
```python
# In trainer.py, after loss computation
if batch_idx % 10 == 0:
    print(f"Assigned anchors: {len(pred_bboxes)}")
    print(f"Mean IoU: {iou.mean():.3f}")
```

Expected values:
- Assigned anchors: 10-50 per batch
- Mean IoU: 0.1-0.3 early training, 0.4+ after epoch 3

## Full Analysis
See `BBOX_LOSS_ANALYSIS.md` for detailed root cause analysis.
