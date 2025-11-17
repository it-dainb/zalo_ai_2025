# Quick Start: Speed Optimization
**Goal**: Reduce training time from ~20 minutes/epoch to <10 minutes/epoch

## Current Status (from profiling)

```
DINOv3 Support Encoding: 53.7ms (79% of time) ← BOTTLENECK
YOLOv8 Backbone:         15.9ms (23%)
Fusion + Detection:      12.9ms (19%)
───────────────────────────────────────────
Total iteration:         68.0ms
Epoch time:              18.8 minutes
```

## Step 1: Apply Quick Wins (5 minutes, low risk)

### 1a. Enable torch.compile()
```bash
python train.py --stage 2 --epochs 100 --compile
```

**Expected**: 20-30% speedup → ~14-15 min/epoch

### 1b. Increase DataLoader workers
```bash
python train.py --stage 2 --epochs 100 --compile --num_workers 4
```

**Expected**: Additional 10-15% → ~12-13 min/epoch

### 1c. Increase cache sizes
```bash
python train.py --stage 2 --epochs 100 \
    --compile \
    --num_workers 4 \
    --frame_cache_size 2000 \
    --support_cache_size_mb 400
```

**Expected**: Additional 5-10% → ~11-12 min/epoch

**Verify**:
```bash
python profile_simple.py --episodes 10
```

## Step 2: DINOv3 Optimization (if still too slow)

### Option A: INT8 Quantization (RECOMMENDED, lowest risk)

**File**: `src/models/dino_encoder.py`

Add after line 130 (after model loading):

```python
# Apply INT8 quantization to frozen DINOv3 for 2-2.7x speedup
if self.freeze_backbone:
    self.dino = torch.quantization.quantize_dynamic(
        self.dino,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    print("✓ Applied INT8 quantization to DINOv3")
```

**Expected**: DINOv3: 53.7ms → ~20-25ms  
**Total epoch time**: ~8-10 minutes

**Risk**: <1% accuracy loss (typically negligible)

### Option B: Smaller DINOv3 (if quantization not enough)

**File**: `train.py` command line:

```bash
python train.py --stage 2 --epochs 100 \
    --compile \
    --num_workers 4 \
    --dinov3_model vit_tiny_patch16_dinov3.lvd1689m
```

**OR edit** `src/models/dino_encoder.py:59`:
```python
# Change from:
model_name = "vit_small_patch16_dinov3.lvd1689m"  # 21.87M params

# To:
model_name = "vit_tiny_patch16_dinov3.lvd1689m"  # ~5M params
```

**Expected**: DINOv3: 53.7ms → ~13ms (4x faster)  
**Total epoch time**: ~6-8 minutes

**Risk**: Medium - may affect small object detection

### Option C: Larger Patches (FASTEST, highest risk)

**File**: `src/models/dino_encoder.py:59`:
```python
model_name = "vit_small_patch32_dinov3.lvd1689m"  # patch32 instead of patch16
```

**Expected**: DINOv3: 53.7ms → ~13ms (4x faster)  
**Total epoch time**: ~6-8 minutes

**Risk**: High - spatial resolution reduced, may miss small objects

## Step 3: Verify & Iterate

After each optimization:

### 3a. Profile again
```bash
python profile_simple.py --episodes 10
```

### 3b. Check training
```bash
# Start training and watch first few batches
python train.py --stage 2 --epochs 1 --compile --num_workers 4
```

### 3c. Validate accuracy (after full training)
```bash
python evaluate.py --checkpoint checkpoints/best.pt \
    --test_data_root ./datasets/test/samples
```

## Recommended Path

**For fastest safe optimization** (target: 8-10 min/epoch):

1. Quick wins (5 min to apply)
   ```bash
   python train.py --stage 2 --epochs 100 \
       --compile \
       --num_workers 4 \
       --frame_cache_size 2000
   ```

2. Add INT8 quantization (10 min to apply + test)
   - Edit `src/models/dino_encoder.py` as shown above
   - Retrain from Stage 2

3. If still slow, use vit_tiny (requires retraining)
   ```bash
   python train.py --stage 2 --epochs 100 \
       --compile \
       --num_workers 4 \
       --dinov3_model vit_tiny_patch16_dinov3.lvd1689m
   ```

## Troubleshooting

### torch.compile() errors
```bash
# If compile fails, disable it:
python train.py --stage 2 --epochs 100 --num_workers 4
```

### OOM with more workers
```bash
# Reduce workers:
python train.py --stage 2 --compile --num_workers 2
```

### INT8 quantization not working
```bash
# Check PyTorch version (need 1.8+):
python -c "import torch; print(torch.__version__)"

# If unsupported, skip quantization and use vit_tiny instead
```

## Expected Results Summary

| Optimization | Effort | Risk | Epoch Time | Speedup |
|--------------|--------|------|------------|---------|
| Baseline | - | - | 18.8 min | 1.0x |
| + torch.compile | 1 min | Low | 14-15 min | 1.3x |
| + num_workers=4 | 1 min | Low | 12-13 min | 1.5x |
| + cache increase | 1 min | Low | 11-12 min | 1.6x |
| **+ INT8 quant** | **10 min** | **Low** | **8-10 min** | **2.0x** |
| + vit_tiny | 30 min | Med | 6-8 min | 2.7x |
| + patch32 | 30 min | High | 6-8 min | 2.7x |

**Best option**: Quick Wins + INT8 Quantization = 8-10 min/epoch with low risk
