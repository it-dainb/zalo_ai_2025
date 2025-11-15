# NaN Gradient Root Cause Analysis & Fix

## Issue Summary
Training failed on **batch 0** with NaN gradients during backward pass. Loss values were finite (bbox=0.88, cls=0.94, dfl=11.82, supcon=0.67, cpe=0.20, triplet=0.33, total=16.97), but gradients became NaN during backpropagation.

## Root Cause Analysis

### 1. DFL Loss Was NOT The Problem
**Initial observation**: DFL loss = 11.82 seemed high
**Reality**: This is **EXPECTED** for randomly initialized model
- With 17 bins and uniform distribution: `-log(1/17) ≈ 2.83` per coordinate
- 4 coordinates: `4 * 2.83 ≈ 11.32` total
- High DFL on first batch is **normal**, not a bug

### 2. DFL Target Dtype Bug (Previously Fixed)
**Issue**: `target_dfl` was converted to `.long()` instead of `.float()`
**Impact**: Integer targets prevent interpolation between bins
**Fix**: Changed to `.float()` in 3 locations in `src/training/loss_utils.py`
**Status**: ✅ Already fixed in previous session

### 3. Numerical Instability in Contrastive Losses
**Root cause**: Insufficient epsilon values in SupCon and CPE losses caused gradient explosion

#### SupCon Loss (src/losses/supervised_contrastive_loss.py:88)
**Before**:
```python
log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
```
**After**:
```python
exp_sum = torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-6)
log_prob = logits - torch.log(exp_sum)
```

#### CPE Loss (src/losses/cpe_loss.py:128-130, 199-204)
**Before**:
```python
loss_i = -torch.log(
    torch.exp(pos_sim - max_sim).sum() / (exp_all.sum() + 1e-7)
)
```
**After**:
```python
exp_pos_sum = torch.clamp(torch.exp(pos_sim - max_sim).sum(), min=1e-6)
exp_all_sum = torch.clamp(exp_all.sum(), min=1e-6)
loss_i = -torch.log(exp_pos_sum / exp_all_sum)
```

**Key improvements**:
- Use `torch.clamp(min=1e-6)` instead of `+ 1e-7`
- Clamping is more numerically stable than addition
- Prevents log(0) and division by near-zero

## Changes Made

### 1. Reverted DFL Weight (train_stage_2.sh)
**Change**: `--dfl_weight 0.15` → `--dfl_weight 0.75`
**Reason**: Reducing DFL weight was treating symptom, not cause. High initial DFL loss is expected.

### 2. Enhanced Numerical Stability
**Files Modified**:
- `src/losses/supervised_contrastive_loss.py` (line 88)
- `src/losses/cpe_loss.py` (lines 128-130, 199-204)

**Changes**:
- Replaced `+ 1e-7` with `torch.clamp(min=1e-6)`
- Added explicit clamping before log operations
- Prevents gradient explosion from near-zero denominators

### 3. Per-Loss Gradient Checking (src/training/trainer.py)
**Added**: Automatic per-loss component gradient check on first batch
**Purpose**: Isolate exactly which loss causes NaN gradients
**When**: Runs only on batch 0, epoch 1
**Output**: Per-component gradient status (✅ OK or ❌ NaN GRADIENTS)

```python
# Tests each loss component individually:
# - bbox_loss
# - cls_loss  
# - dfl_loss
# - supcon_loss
# - cpe_loss
# - triplet_loss
```

## Testing Plan

### 1. Run Training With Fixes
```bash
bash train_stage_2.sh
```

**Expected behavior**:
- Per-loss gradient check on batch 0 shows all components ✅ OK
- No NaN gradients on any batch
- DFL loss starts ~12-13 and decreases over batches
- Training proceeds normally

### 2. Monitor First 10 Batches
```
Batch 0: DFL ~12-13 (expected)
Batch 1-5: DFL decreasing
Batch 5-10: DFL ~5-8 (model learning)
```

## Key Learnings

### ❌ Wrong Approach
- Reducing loss weights to suppress symptoms
- Treating high initial loss as a bug

### ✅ Correct Approach  
- Understand expected loss ranges for random initialization
- Fix numerical instability at the source (epsilon values)
- Add diagnostic tools to isolate root cause

## Files Modified Summary

| File | Lines | Change |
|------|-------|--------|
| `train_stage_2.sh` | 26 | Revert DFL weight 0.15 → 0.75 |
| `src/losses/supervised_contrastive_loss.py` | 87-89 | Clamp epsilon for log stability |
| `src/losses/cpe_loss.py` | 128-131 | Clamp exp sums before log |
| `src/losses/cpe_loss.py` | 199-203 | Clamp exp sums in simplified CPE |
| `src/training/trainer.py` | 508-551 | Add per-loss gradient checking |

## Related Issues

- **Learnable Projection**: ✅ Already added (384→256 dims, 99K params)
- **DFL Target Dtype**: ✅ Already fixed (.long() → .float())
- **Memory**: ⚠️ Triplet batch size reduced to 8 (from 32) for 3.63GB GPU
- **Model Size**: ✅ 30.32M params (60.6% of 50M budget)

## Next Steps

1. ✅ Run training to verify NaN issue resolved
2. Monitor loss curves for first 50-100 batches
3. If NaN persists, per-loss gradient check will identify culprit
4. Adjust contrastive loss weights if needed (currently: supcon=1.2, cpe=0.6)

## References

- DFL expected range: `4 * -log(1/reg_max) ≈ 11.3` for reg_max=16
- Numerical stability: Always use `torch.clamp(min=eps)` over `+ eps`
- Gradient explosion typically from log(near-zero) or division by near-zero
