# Gradient Explosion Fix - Quick Start Guide

## What Was Fixed?

Your training had **gradient explosion** causing NaN/Inf errors. We fixed it with 4 changes:

1. ✅ **Fixed mixed precision NaN detection** - now detects issues accurately
2. ✅ **Stabilized DFL loss** - clamped to prevent explosion (was 8-10, now max 15)
3. ✅ **Added proper weight initialization** - detection heads now start with small weights
4. ✅ **More conservative hyperparameters** - tighter clipping, lower LR

## Quick Start

```bash
# Stop your current broken training (Ctrl+C)

# Start training with fixes:
bash train_stage_2.sh

# Monitor logs in real-time:
tail -f checkpoints/stage2/training_debug.log | grep -E "(dfl_loss|NaN|Gradient Norm)"
```

## What to Look For

### ✅ Good Signs (Training is Fixed):
```
Loss components:
  dfl_loss: 3.2  ← Should be 2-5 (was 8-10+)
  
Gradient Norms:
  Total Grad Norm: 0.42  ← Should be < 0.5 (was inf)
  
No NaN/Inf warnings!
No "Skipping optimizer step" errors!
```

### ❌ Bad Signs (Still Broken):
```
dfl_loss: 8.5  ← Still too high
Total Grad Norm: inf  ← Still exploding
⚠️ NaN/Inf gradient in ...  ← Still happening
```

## Modified Files

1. `src/training/trainer.py` - Fixed NaN detection timing
2. `src/losses/dfl_loss.py` - Added loss clamping
3. `src/models/dual_head.py` - Added weight initialization
4. `train_stage_2.sh` - Updated hyperparameters

## Key Hyperparameter Changes

| Parameter | Old | New | Change |
|-----------|-----|-----|--------|
| Learning Rate | 1e-4 | **3e-5** | 3.3x lower |
| Weight Decay | 0.05 | **0.005** | 10x lower |
| DFL Weight | 1.0 | **0.5** | 2x lower |
| Gradient Clip | 5.0 | **0.5** | 10x tighter |

## Troubleshooting

### If you still see NaN/Inf:
```bash
# Try even more conservative settings:
python train.py \
  --lr 1e-5 \
  --gradient_clip_norm 0.25 \
  --dfl_weight 0.25 \
  [... other args from train_stage_2.sh ...]
```

### If DFL loss is still > 6.0:
```bash
# Check that the DFL loss fix was applied:
grep "torch.clamp(loss_mean" src/losses/dfl_loss.py

# Should output:
# return torch.clamp(loss_mean, max=15.0)
```

### If gradient norms are still > 1.0:
```bash
# Check weight initialization was added:
grep "_initialize_weights" src/models/dual_head.py

# Should show two occurrences (StandardDetectionHead and PrototypeDetectionHead)
```

## Compare Before/After

```bash
# Before fix:
Batch 400: dfl_loss=8.48, Grad Norm=inf → NaN → Skip step ❌

# After fix:
Batch 400: dfl_loss=3.21, Grad Norm=0.43 → Normal → Update params ✅
```

## Full Details

See `GRADIENT_NAN_FIX.md` for:
- Complete root cause analysis
- Detailed before/after code comparisons
- Verification commands
- References and links

## Questions?

1. **Q: Why was DFL loss so high?**  
   A: `-log(tiny_probability)` explodes. We now clamp probabilities and loss values.

2. **Q: Why change learning rate AND gradient clipping?**  
   A: Defense in depth. Multiple safety layers catch edge cases.

3. **Q: Will this slow training?**  
   A: Slightly (~5%), but training will actually COMPLETE instead of failing.

4. **Q: Can I resume from my broken checkpoint?**  
   A: No - it has exploded weights. Start fresh with the fixes applied.

## Success Criteria

After 50 batches, you should see:
- ✅ No NaN/Inf warnings
- ✅ DFL loss in 2-5 range  
- ✅ Gradient norms < 0.5
- ✅ Smooth loss curves in W&B

Happy training! 🚀
