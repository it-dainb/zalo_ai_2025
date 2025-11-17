# Training Speed Optimization - Quick Reference

## Immediate Actions (Copy & Paste)

### 1. Profile Current Performance (DO THIS FIRST)
```bash
python profile_training_bottlenecks.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --profile_episodes 10 \
    --stage 2
```

**Expected Output**: Timing breakdown + cache hit rates + recommendations

---

### 2. Quick Win #1: Enable torch.compile() (PyTorch 2.0+)
**Expected Speedup**: 20-30% | **Risk**: Low

Add to `train.py` after line 248 (after model initialization):

```python
# Enable torch.compile() for speedup (PyTorch 2.0+)
if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
    print("\n⚡ Enabling torch.compile() for 20-30% speedup...")
    model = torch.compile(model, mode='reduce-overhead')
    print("✓ Model compiled successfully\n")
```

---

### 3. Quick Win #2: Increase Cache Sizes
**Expected Speedup**: 10-20% | **Risk**: None (just uses more RAM)

```bash
python train.py --stage 2 --epochs 100 \
    --frame_cache_size 1000 \
    --support_cache_size_mb 400
```

---

### 4. Quick Win #3: Increase DataLoader Workers
**Expected Speedup**: 10-15% if data loading is bottleneck | **Risk**: Low

```bash
python train.py --stage 2 --epochs 100 \
    --num_workers 4  # Up from default 1
```

**Monitor**: Watch GPU utilization with `nvidia-smi`. If <80%, increase workers.

---

### 5. Medium Risk: Smaller DINOv3 Variant
**Expected Speedup**: 4x faster DINOv3 | **Risk**: Medium (may affect accuracy)

Edit `src/models/dino_encoder.py` line 28:

```python
# Option A: Larger patches (faster, same model size)
model_name = "vit_small_patch32_dinov3.lvd1689m"  # 4x fewer tokens

# Option B: Smaller model (much faster, lower quality)
model_name = "vit_tiny_patch16_dinov3.lvd1689m"  # ~5M params vs 21M
```

**Test on validation set first!**

---

## Expected Total Speedup

| Optimizations | Speedup | Time per Epoch | Risk |
|--------------|---------|----------------|------|
| **Current** | 1.0x | 60 min | - |
| Quick Wins (1+2+3) | 1.5-1.7x | 35-40 min | Low |
| + Smaller DINOv3 | 3-5x | 12-20 min | Medium |

---

## Verify Caching is Working

After epoch 1, you should see:
- Support cache hit rate: **>80%** ✓
- Frame cache hit rate: **>50%** ✓
- DINOv3 support encoding: **Only runs once per episode** ✓

Check with profiling script or training logs.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU util <50% | Data loading slow | Increase `--num_workers` |
| OOM error | Batch too large | Reduce `--batch_size`, increase `--gradient_accumulation` |
| Cache hit <50% | Cache too small | Increase cache sizes |
| Loss not decreasing | Model too aggressive | Roll back DINOv3 changes |

---

## Monitoring During Training

```bash
# Terminal 1: Training
python train.py --stage 2 --epochs 100

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Look for:
# - GPU Utilization: Should be 80-100%
# - GPU Memory: Should be stable (~8-12GB)
# - Temperature: <85°C
```

---

## Current Configuration Verified

✅ **Support feature caching**: Enabled (`trainer.py:1147-1158`)  
✅ **Mixed precision (AMP)**: Enabled (`trainer.py:8, 307, 508, 910`)  
✅ **Data caching**: Enabled (500 frames + 200MB support images)  
✅ **Gradient accumulation**: 4 steps (effective batch = 16)  

---

## Full Details

See `docs/SPEED_OPTIMIZATION_GUIDE.md` for complete analysis and all optimization options.

---

**Last Updated**: 2025-11-17  
**Status**: Ready to apply
