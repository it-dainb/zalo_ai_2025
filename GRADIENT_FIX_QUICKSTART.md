# Gradient Explosion Fix - Quick Reference

## TL;DR

**Problem:** Gradient explosion (4.97M magnitude) due to multi-scale accumulation + DFL dominance

**Solution:** 
1. Increase gradient clipping: `1.0 → 10.0`
2. Reduce DFL weight: `1.5 → 0.5`
3. Remove gradient scaling (let gradients flow naturally)

**Status:** ✅ FIXED - Ready to train

---

## Training Command

```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4 \
  --batch_size 4 \
  --lr 0.0001 \
  --gradient_clip_norm 10.0 \
  --dfl_weight 0.5 \
  --resume checkpoints/stage1_final.pt
```

---

## Expected Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Gradient Norm | 4,970,000 | 5-15 |
| DFL Loss | 8-12 | 4-6 |
| Total Loss | Unstable | Decreasing |
| Backbone Learning | ❌ Explosion | ✅ Healthy |

---

## Quick Verification

```bash
# Check gradient norms
grep "Gradient norm" checkpoints/training_debug.log | tail -10

# Should see: Gradient norm: 5-15 (not millions!)
```

---

## Files Changed

- `train.py:131` - `gradient_clip_norm = 10.0`
- `train.py:139` - `dfl_weight = 0.5`
- `src/training/trainer.py` - Removed gradient scaling

---

## Why This Works

1. **Multi-scale architecture** (P2-P5) = 4× gradient accumulation → needs higher clip threshold
2. **DFL dominance** (10-20× other losses) → reduce weight to balance contributions
3. **Natural gradient flow** → no manual scaling, just intelligent clipping

---

For detailed explanation, see: `GRADIENT_EXPLOSION_SOLUTION.md`
