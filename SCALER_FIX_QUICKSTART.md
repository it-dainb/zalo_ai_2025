# Gradient Scaler Fix - Quick Reference

## Problem
```
RuntimeError: unscale_() has already been called on this optimizer since the last update().
```

## Root Cause
`scaler.unscale_()` was called every batch, but `scaler.update()` only called every N batches (gradient accumulation).

## Solution
Move `unscale_()` inside the gradient accumulation block:

```python
# ❌ BEFORE (WRONG)
if self.mixed_precision:
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)  # Called every batch

if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
    self.scaler.step(self.optimizer)
    self.scaler.update()  # Called every N batches

# ✅ AFTER (CORRECT)
if self.mixed_precision:
    self.scaler.scale(loss).backward()

if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
    if self.mixed_precision:
        self.scaler.unscale_(self.optimizer)  # Now called every N batches
    
    # Check gradients, clip, step optimizer
    self.scaler.step(self.optimizer)
    self.scaler.update()
```

## File Changed
- `src/training/trainer.py` (lines 560-641)

## Status
✅ **FIXED** - Training runs without errors

## Test Command
```bash
python train.py --stage 2 --epochs 1 --batch_size 2 --n_way 2 --n_query 2
```

## Expected Result
- No scaler errors
- Stable loss values (~20.6-20.9)
- No NaN/Inf gradients
- Training progresses smoothly

## Full Documentation
See `GRADIENT_SCALER_FIX.md` for detailed analysis.
