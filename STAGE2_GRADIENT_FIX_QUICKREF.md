# Stage 2 Gradient Explosion Fix - Quick Reference

## TL;DR - What Changed

```bash
# train_stage_2.sh
--dfl_weight 0.15      # Was: 0.5 (reduced 70%)
--lr 2e-5              # Was: 3e-5 (reduced 33%)
--gradient_clip_norm 5.0  # Was: 10.0 (reduced 50%)
```

```python
# src/losses/dfl_loss.py
loss_mean = torch.clamp(loss_mean, max=15.0)  # Added final clamping
```

## Problem
- DFL loss: **5.9-7.3** (too high)
- Weighted contribution: **3.0-3.7** (dominating)
- Result: Gradient explosion → NaN → skipped batches

## Solution
- DFL weight: 0.5 → **0.15** (weighted contribution now: 0.9-1.1)
- Added loss clamping at **15.0** (prevents explosion before backward)
- Tighter gradient clip: 10.0 → **5.0** (safety net)
- Lower LR: 3e-5 → **2e-5** (more stable)

## Verification
```bash
python test_gradient_fix.py
# Expected output:
# Loss value: 12.93 (< 15.0) ✅
# Has NaN/Inf: False ✅
# Clipped norm: 0.089 (< 5.0) ✅
```

## Resume Training
```bash
bash train_stage_2.sh
```

## Expected Behavior
```
✅ dfl_loss: 0.9-1.5 (not 5.9-7.3)
✅ total_loss: 4.6-6.2 (stable)
✅ No NaN/Inf warnings
✅ All 500 episodes train (no skipped batches)
```

## Troubleshooting
If issues persist:
1. Reduce DFL weight further: `--dfl_weight 0.1`
2. Lower learning rate: `--lr 1e-5`
3. Disable mixed precision: `--no-mixed_precision`

## Files Modified
- `train_stage_2.sh` (3 parameters)
- `src/losses/dfl_loss.py` (loss initialization + clamping)

## Related Docs
- `GRADIENT_EXPLOSION_FIX_STAGE2.md` (full analysis)
- `test_gradient_fix.py` (validation test)
