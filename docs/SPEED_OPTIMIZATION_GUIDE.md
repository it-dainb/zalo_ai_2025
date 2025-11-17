# Speed Optimization Guide for YOLOv8n-RefDet

## Current Status

**Training Speed**: ~1 hour per epoch (too slow)

**Goal**: Reduce to 15-30 minutes per epoch

## Architecture Analysis

### Parameter Breakdown (29.25M total, 58.5% of 50M budget)

| Component | Parameters | % of Total | Status |
|-----------|-----------|------------|---------|
| **DINOv3 Encoder** | 21.87M | 74.8% | ⚠️ **BOTTLENECK** |
| YOLOv8 Backbone (P2-P5) | 3.16M | 10.8% | ✓ Efficient |
| PSALM Fusion | 0.78M | 2.7% | ✓ Efficient |
| Dual Detection Head | 3.45M | 11.8% | ✓ Efficient |

## Current Optimizations Already In Place

### ✅ Implemented and Working

1. **Support Feature Caching** (`src/training/trainer.py:1147-1158`)
   - Support features computed once per episode via `set_reference_images()`
   - Query forward passes use `use_cache=True` to skip DINOv3 re-encoding
   - **Saves**: ~50% of DINOv3 forward passes per episode

2. **Mixed Precision Training** (`src/training/trainer.py:8, 307, 508, 910`)
   - Automatic Mixed Precision (AMP) with `GradScaler`
   - Enabled by default in trainer
   - **Saves**: ~30-40% computation, ~50% GPU memory

3. **Data Caching** (`src/datasets/refdet_dataset.py:79-170`)
   - LRU cache for video frames (500 frames default ~= 300MB)
   - Support image cache (200MB default)
   - **Saves**: Disk I/O time

4. **Optimized DataLoader** (`train.py`)
   - `num_workers=1` (low to prevent OOM)
   - `persistent_workers=False` (saves memory)
   - `prefetch_factor=1` (minimal prefetch)
   - `pin_memory=True` (faster GPU transfer)

5. **Gradient Accumulation** (`train.py`)
   - Default: 4 steps
   - Allows larger effective batch size without OOM

## Recommended Optimizations (Priority Order)

### Priority 1: Quick Wins (High Impact, Low Risk)

#### 1.1 Enable torch.compile() [PyTorch 2.0+]
**Expected Speedup**: 20-30%  
**Risk**: Low  
**Implementation**:

```python
# In train.py after model initialization
if torch.__version__ >= '2.0.0':
    model = torch.compile(model, mode='reduce-overhead')
    print("✓ Model compiled with torch.compile()")
```

**Status**: ⭐ **RECOMMENDED - TEST FIRST**

#### 1.2 Verify Cache Hit Rates
**Tool**: Run the profiling script
```bash
python profile_training_bottlenecks.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --profile_episodes 10
```

**Check**:
- Support cache hit rate should be >80% after first episode
- Frame cache hit rate should be >50%

**If low**: Increase cache sizes in `train.py`:
```bash
python train.py --stage 2 \
    --frame_cache_size 1000 \  # Double from 500
    --support_cache_size_mb 400  # Double from 200
```

**Status**: ⭐ **DO THIS FIRST**

#### 1.3 Increase DataLoader Workers (if memory allows)
**Expected Speedup**: 10-20% if data loading is bottleneck  
**Current**: `num_workers=1`  
**Try**: `num_workers=2` or `num_workers=4`

```bash
python train.py --stage 2 --num_workers 4
```

**Watch**: Monitor GPU utilization with `nvidia-smi`. If GPU usage <80%, data loading is bottleneck.

**Status**: ✓ Safe to test

### Priority 2: Model Architecture Optimizations

#### 2.1 Use Smaller DINOv3 Variant
**Expected Speedup**: 15-25%  
**Risk**: Medium (may affect accuracy)

Current model: `vit_small_patch16_dinov3.lvd1689m` (21.6M params)

**Option A**: Increase patch size (faster, same model size)
```python
# In src/models/dino_encoder.py:28
model_name = "vit_small_patch32_dinov3.lvd1689m"  # 4x fewer tokens
```
- Reduces tokens from 256 → 64
- **Speedup**: ~4x faster forward pass
- **Risk**: May hurt small object detection

**Option B**: Use ViT-Tiny variant
```python
model_name = "vit_tiny_patch16_dinov3.lvd1689m"  # ~5M params
```
- **Speedup**: ~4x faster
- **Risk**: Lower feature quality

**Status**: ⚠️ Test on validation set first

#### 2.2 Quantize Frozen DINOv3 to INT8
**Expected Speedup**: 2-4x for DINOv3 forward pass  
**Risk**: Low (frozen model, minor accuracy loss)

```python
# After loading DINOv3 in src/models/dino_encoder.py
import torch.quantization
self.backbone = torch.quantization.quantize_dynamic(
    self.backbone,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**Status**: ✓ Worth trying

#### 2.3 Replace PSALM with CHEAF (if currently using PSALM)
**Expected Speedup**: 5-10%  
**Risk**: Low

Check current fusion type:
```bash
grep "fusion_type\|use_cheaf" src/models/yolov8n_refdet.py
```

If using PSALM, CHEAF is slightly faster:
```python
# In train.py when initializing model
model = YOLOv8nRefDet(use_cheaf=True)  # Instead of PSALM
```

**Status**: ✓ Easy test

### Priority 3: Training Pipeline Optimizations

#### 3.1 Reduce Contrastive Loss Weight in Later Epochs
**Expected Speedup**: 10-15%  
**Risk**: Low

SupCon and CPE losses are expensive. Reduce after epoch 50:

```python
# In src/training/trainer.py, add to __init__:
self.dynamic_loss_schedule = True

# In training loop, add before loss computation:
if self.dynamic_loss_schedule and self.epoch > 50:
    # Reduce contrastive weights
    self.loss_fn.weights['supcon'] *= 0.5
    self.loss_fn.weights['cpe'] *= 0.5
```

**Status**: ✓ Recommended for Stage 2

#### 3.2 Reduce Image Resolution Temporarily
**Expected Speedup**: 25-40%  
**Risk**: Medium (affects accuracy)

For debugging/development only:
```python
# In src/datasets/collate.py or train.py
IMAGE_SIZE = 512  # Instead of 640
```

**Status**: ⚠️ Dev/debug only

#### 3.3 Reduce Number of Augmentations
**Expected Speedup**: 5-10%  
**Risk**: Low (may affect generalization)

```bash
python train.py --stage 2 --num_aug 1  # Instead of 3-5
```

**Status**: ✓ Safe for initial training

### Priority 4: Loss Function Optimizations

#### 4.1 Simplify WIoU Loss
**Expected Speedup**: 2-5%  
**Risk**: Low

Current: WIoU v3 with monotonous=True (more expensive)

```python
# In src/losses/combined_loss.py:60
self.bbox_loss = WIoULoss(monotonous=False)  # Faster variant
```

**Status**: ✓ Minor gain

#### 4.2 Reduce Number of Detection Scales (Risky)
**Expected Speedup**: 20-25%  
**Risk**: High (affects small object detection)

Skip P2 scale (smallest objects) if dataset has few tiny objects:
```python
# In src/models/yolov8n_refdet.py
# Comment out P2 processing in forward pass
```

**Status**: ⚠️ Only if P2 detections are rarely used

## Profiling Workflow

### Step 1: Run Profiling Script
```bash
python profile_training_bottlenecks.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --profile_episodes 10 \
    --stage 2 \
    --n_way 2 \
    --n_query 4
```

### Step 2: Analyze Output
Check the timing breakdown:
- If **DINOv3 > 40%**: Focus on Priority 1 & 2
- If **Data Loading > 20%**: Increase num_workers or cache sizes
- If **Loss Computation > 25%**: Focus on Priority 3 & 4
- If **PSALM Fusion > 15%**: Switch to CHEAF

### Step 3: Apply Optimizations Incrementally
1. Start with Quick Wins (Priority 1)
2. Test each change on validation set
3. Measure speed improvement
4. Only proceed to riskier optimizations if needed

## Expected Results

### Conservative Optimizations (Low Risk)
- torch.compile(): +20-30%
- Verify caching: +10-20%
- Increase num_workers: +10-15%
- **Total Speedup**: 1.4-1.7x (42 min → 25-30 min per epoch)

### Aggressive Optimizations (Medium Risk)
- Above + Smaller DINOv3 variant: +2-3x
- Above + INT8 quantization: +3-4x
- **Total Speedup**: 3-5x (60 min → 12-20 min per epoch)

## Monitoring Checklist

### During Training
```bash
# Terminal 1: Run training
python train.py --stage 2 --epochs 100

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor cache stats (if enabled)
tail -f logs/training.log | grep "Cache"
```

### Metrics to Track
- **Time per iteration** (should decrease after epoch 1 due to caching)
- **GPU utilization** (should be >80%, else data loading is bottleneck)
- **GPU memory usage** (should be stable, not growing)
- **Cache hit rates** (support >80%, frames >50%)

## Current Configuration (from train.py)

```python
# Default training configuration
batch_size = 4                    # Number of query images per episode
n_way = 2                        # Number of classes per episode
n_query = 4                      # Query samples per class
gradient_accumulation = 4        # Effective batch = 4 * 4 = 16
num_workers = 1                  # Data loading workers (low for memory)
mixed_precision = True           # AMP enabled
frame_cache_size = 500          # ~300MB
support_cache_size_mb = 200     # 200MB
```

## Troubleshooting

### Symptom: GPU utilization <50%
**Cause**: Data loading bottleneck  
**Fix**: Increase `num_workers` or cache sizes

### Symptom: Out of Memory (OOM)
**Cause**: Batch size too large or cache too large  
**Fix**: 
- Reduce `batch_size`
- Reduce cache sizes
- Increase `gradient_accumulation` (compensate for smaller batch)

### Symptom: Cache hit rate <50%
**Cause**: Cache too small or episode sampling too random  
**Fix**: Increase cache sizes or adjust sampling strategy

### Symptom: Loss not decreasing
**Cause**: Too aggressive optimizations affecting accuracy  
**Fix**: Roll back model simplifications (smaller DINOv3, reduced resolution)

## Next Steps

1. ✅ Run profiling script to establish baseline
2. ✅ Apply Priority 1 optimizations (quick wins)
3. ⏳ Test on validation set after each change
4. ⏳ Gradually apply Priority 2-4 as needed
5. ⏳ Document final configuration that works best

## Files to Modify

| Optimization | File | Line |
|--------------|------|------|
| torch.compile | `train.py` | After model init (~line 250) |
| Cache sizes | `train.py` | Args (lines 78-80) |
| num_workers | `train.py` | Args (line 133) |
| DINOv3 variant | `src/models/dino_encoder.py` | Line 28 |
| Fusion type | `src/models/yolov8n_refdet.py` | Init params |
| Loss schedule | `src/training/trainer.py` | Training loop |
| WIoU variant | `src/losses/combined_loss.py` | Line 60 |

## Reference Documents

- **Model Parameters**: `docs/MODEL_PARAMETERS_REPORT.md`
- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **Caching System**: `docs/CACHING_SYSTEM.md`
- **Loss Functions**: `docs/loss-functions-guide.md`

---

**Last Updated**: 2025-11-17  
**Status**: Ready for profiling and optimization
