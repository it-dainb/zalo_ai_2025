# Gradient Explosion Fix - Hyperparameter Recommendations

## Problem Summary

**Observed Issues:**
- Total gradient norm: **4.97M** before clipping
- Gradient clipped to: **8.0** with threshold 1.0
- Loss instability: Jumps from 6.49 → 26.19
- DFL loss abnormally high: **11.86** (should be 0.5-2.0)

**Root Cause:**
The extreme gradient clipping (4.97M → 8.0) means the optimizer is essentially taking random tiny steps rather than following the true gradient direction. This prevents any meaningful learning.

## Recommended Fixes

### 1. Increase Gradient Clip Threshold (CRITICAL)

**Current:** `--gradient_clip_norm 1.0`
**Recommended:** `--gradient_clip_norm 10.0`

**Rationale:**
- Current clipping is too aggressive (reducing gradients by 621,125x!)
- Threshold of 10.0 allows gradients up to 10x the typical norm
- Still prevents explosion while preserving gradient direction
- Can gradually reduce to 5.0 after first few epochs

**Progressive Strategy:**
```bash
# Epochs 1-10: Higher threshold for warmup
--gradient_clip_norm 10.0

# Epochs 11-50: Moderate threshold
--gradient_clip_norm 5.0

# Epochs 51+: Final threshold
--gradient_clip_norm 3.0
```

### 2. Reduce DFL Loss Weight

**Current:** `--dfl_weight 1.5`
**Recommended:** `--dfl_weight 0.75`

**Rationale:**
- DFL loss at 11.86 is **5-10x** higher than typical values
- This suggests either:
  - Incorrect DFL target preparation
  - Anchors not properly aligned with targets
  - Weight too high relative to other losses
- Reducing weight by 50% will help stabilize total loss

**Alternative:** Try `--dfl_weight 0.5` if instability persists

### 3. Adjust Learning Rate (Optional)

**Current:** `--lr 1e-4` (default)
**Recommended:** Keep at `1e-4` initially

**Rationale:**
- Learning rate itself is reasonable
- The problem is the gradient clipping, not the LR
- After fixing gradient clipping, monitor if LR needs adjustment

**If loss remains unstable after fixing clipping:**
- Try `--lr 5e-5` (reduce by 50%)
- Or implement learning rate warmup (see below)

### 4. Add Learning Rate Warmup (Recommended)

The trainer already has warmup support. Use it:

```bash
--warmup_epochs 3
```

**Rationale:**
- Gradients are often larger at the start of training
- Warmup allows the model to stabilize before full learning rate
- Reduces risk of early divergence

## Recommended Training Command

### Stage 2 Training (Current Issue)

```bash
python train.py \
    --stage 2 \
    --epochs 100 \
    --batch_size 32 \
    --n_way 4 \
    --n_query 4 \
    --lr 1e-4 \
    --gradient_clip_norm 10.0 \
    --dfl_weight 0.75 \
    --warmup_epochs 3 \
    --data_root ./datasets/train \
    --val_data_root ./datasets/val \
    --checkpoint <path_to_stage1_checkpoint>
```

### If Still Unstable

Try even more conservative settings:

```bash
python train.py \
    --stage 2 \
    --epochs 100 \
    --batch_size 32 \
    --n_way 4 \
    --n_query 4 \
    --lr 5e-5 \
    --gradient_clip_norm 10.0 \
    --dfl_weight 0.5 \
    --bbox_weight 5.0 \
    --warmup_epochs 5 \
    --data_root ./datasets/train \
    --val_data_root ./datasets/val \
    --checkpoint <path_to_stage1_checkpoint>
```

## Loss Weight Analysis

From your logs, the loss components are:

```
bbox_loss: 0.70 ✓ (reasonable)
cls_loss: 0.98 ✓ (reasonable)
dfl_loss: 11.86 ❌ (TOO HIGH!)
supcon_loss: 1.62 ✓ (reasonable)
cpe_loss: 0.00 → NOW FIXED ✓
triplet_loss: 1.67 ✓ (reasonable)
```

**Current weights (from code):**
- bbox: 7.5 (WIoU)
- cls: 0.5 (BCE)
- dfl: 1.5 (DFL)
- supcon: 1.0
- cpe: 0.5
- triplet: 0.2

**Recommended weights:**
```bash
--bbox_weight 7.5      # Keep (WIoU is working well)
--cls_weight 0.5       # Keep (classification stable)
--dfl_weight 0.75      # REDUCE from 1.5 (DFL too high)
--supcon_weight 1.0    # Keep (contrastive working)
--cpe_weight 0.5       # Keep (now working)
--triplet_weight 0.2   # Keep (triplet stable)
```

## Monitoring Guidelines

### Healthy Training Indicators

**Gradient Norms:**
- Pre-clip: 10 - 1000 (typical range)
- Post-clip: Should be close to pre-clip (not drastically different)
- **Warning:** If pre-clip > 10,000, gradients are exploding

**Loss Values:**
- bbox_loss: 0.5 - 2.0
- cls_loss: 0.5 - 1.5
- dfl_loss: 0.5 - 2.0 (**currently 11.86 - PROBLEM!**)
- supcon_loss: 0.5 - 2.0
- cpe_loss: 0.0 - 1.0
- triplet_loss: 0.5 - 2.0

**Loss Stability:**
- Total loss should decrease gradually (not jump around)
- Sudden jumps > 2x indicate instability
- Example: 6.49 → 26.19 is a **4x jump - UNSTABLE!**

### Red Flags

1. **Gradient norm > 100,000**: Immediate explosion risk
2. **Pre-clip vs post-clip ratio > 100**: Clipping too aggressive
3. **Loss increases by >2x**: Training diverging
4. **DFL loss > 5.0**: Something wrong with bbox regression

## DFL Loss Investigation

The DFL loss being 11.86 suggests a deeper issue. Let's check:

### Potential Causes

1. **Incorrect DFL target normalization** (lines 163-168 in loss_utils.py)
   ```python
   # Current: Normalizing to [0, reg_max]
   normalized_box[:, [0, 2]] = (normalized_box[:, [0, 2]] / img_size * reg_max).clamp(0, reg_max)
   ```
   - This might be wrong for anchor-based assignment
   - DFL expects **distances from anchor**, not absolute coordinates

2. **Anchor assignment mismatch**
   - Anchors might not be properly aligned with targets
   - This causes large regression errors

3. **Scale mismatch**
   - Different scales (strides 4, 8, 16, 32) need different normalization

### Recommended Investigation

Add debugging to `src/losses/dfl_loss.py`:

```python
# In DFLoss.forward(), add:
print(f"[DFL DEBUG] pred shape: {pred.shape}, target shape: {target.shape}")
print(f"[DFL DEBUG] pred range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"[DFL DEBUG] target range: [{target.min()}, {target.max()}]")
```

## Quick Test Script

Create `test_gradient_fix.sh`:

```bash
#!/bin/bash

# Test with recommended hyperparameters
python train.py \
    --stage 2 \
    --epochs 5 \
    --batch_size 16 \
    --n_way 2 \
    --n_query 2 \
    --lr 1e-4 \
    --gradient_clip_norm 10.0 \
    --dfl_weight 0.75 \
    --warmup_epochs 1 \
    --data_root ./datasets/train \
    --val_data_root ./datasets/val \
    --save_dir ./test_gradient_fix \
    2>&1 | tee test_gradient_fix.log

# Check for stability
echo "Checking training stability..."
grep "Total Grad Norm:" test_gradient_fix.log | tail -20
grep "Total Loss:" test_gradient_fix.log | tail -20
```

## Expected Results

After applying these fixes, you should see:

1. **Gradient norms:** 10 - 1000 (not millions!)
2. **DFL loss:** 0.5 - 2.0 (not 11.86!)
3. **Total loss:** Gradually decreasing (not jumping 4x)
4. **Training speed:** Faster convergence

## Next Steps

1. **Immediate:** Apply gradient clip threshold increase to 10.0
2. **Immediate:** Reduce DFL weight to 0.75
3. **Test:** Run 5 epochs with new settings
4. **Monitor:** Check gradient norms and loss stability
5. **Investigate:** If DFL still high, debug DFL target preparation

## Summary

| Parameter | Current | Recommended | Reason |
|-----------|---------|-------------|--------|
| gradient_clip_norm | 1.0 | **10.0** | Too aggressive clipping |
| dfl_weight | 1.5 | **0.75** | DFL loss too high |
| lr | 1e-4 | 1e-4 | OK as-is |
| warmup_epochs | 0 | **3** | Stabilize early training |

**Priority:** Fix gradient clipping FIRST - this is blocking all learning!
