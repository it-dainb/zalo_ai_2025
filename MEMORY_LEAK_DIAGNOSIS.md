# Memory Leak Diagnosis & Fixes

## Problem
Training RAM grows from 1.12GB → 62GB → OOM kill at line 38 of train_stage_2.sh

##Configuration
```bash
--n_way 4
--n_query 8
--num_aug 5
--auto_episodes
--triplet_batch_size 64
--num_workers 8
```

## Root Causes Identified

### 1. **Validation Loop - Unbounded List Growth** ⚠️ CRITICAL
**Location:** `src/training/trainer.py` lines 595-693

**Problem:**
```python
# These lists grow unbounded during validation:
all_st_ious = []  
all_pred_bboxes = []  # Accumulates ALL predictions
all_pred_scores = []
all_pred_classes = []
all_gt_bboxes = []
all_gt_classes = []

# For EACH batch, append numpy arrays
for batch in val_loader:
    for i in range(len(batch['query_images'])):  # e.g., 32 samples
        all_pred_bboxes.append(sample_pred_bboxes)  # (N, 4) numpy
        all_pred_scores.append(sample_pred_scores)  # (N,) numpy
        # ...accumulates to thousands of arrays
```

**Impact:** With 20 val episodes × 32 samples = 640 arrays × multiple epochs = MASSIVE growth

**Fix:** Clear lists after validation completes

---

### 2. **WandB Logging - Keeping References** ⚠️ HIGH
**Location:** `src/training/trainer.py` lines 464-475, 548-552, 754-756

**Problem:**
```python
# WandB keeps references to logged data
wandb.log(step_log, step=self.global_step)
wandb.log(wandb_log, step=self.global_step)
```

**Impact:** Tens of thousands of logged metrics accumulate in memory

**Fix:** Already has cleanup at line 761-762, but need to ensure it runs

---

### 3. **Triplet Dataset - Large Batch Size** ⚠️ HIGH  
**Configuration:** `--triplet_batch_size 64` with `--use_batch_hard_triplet`

**Problem:**
- Triplet batch size 64 is 8× larger than typical detection batch (8)
- Each triplet has 3 images (anchor, positive, negative) = 192 images per batch
- With mixed training (30% triplet), this causes memory spikes

**Fix:** Reduce triplet_batch_size to 8-16

---

### 4. **Dataset Caching - Video Frame Cache** ⚠️ MEDIUM
**Location:** `src/datasets/refdet_dataset.py` lines 82-170

**Current:** `--frame_cache_size 500` (default) = ~300MB per video
**Problem:** Multiple video extractors × 500 frames each

**Impact:** Moderate but additive

**Fix:** Already has LRU eviction, but check for video extractor cleanup

---

### 5. **Gradient History - Already Fixed** ✅
**Location:** Lines 421-423, 444-446
Already bounded to `grad_norm_window` size - OK

---

## Fixes to Apply

### Fix 1: Clear Validation Lists After Use
**File:** `src/training/trainer.py` line 763

```python
# Memory cleanup to prevent memory leak
import gc

# IMPORTANT: Delete large validation lists
del all_st_ious, all_pred_bboxes, all_pred_scores, all_pred_classes
del all_gt_bboxes, all_gt_classes
try:
    del all_pred_bboxes_flat, all_pred_scores_flat, all_pred_classes_flat
    del all_gt_bboxes_flat, all_gt_classes_flat
except:
    pass

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

---

### Fix 2: Reduce Triplet Batch Size
**File:** `train_stage_2.sh` line 15

```bash
# Before:
--triplet_batch_size 64 \

# After:
--triplet_batch_size 8 \
```

**Rationale:** 64 × 3 images = 192 images/batch is excessive

---

### Fix 3: Reduce Number of Workers
**File:** `train_stage_2.sh` line 21

```bash
# Before:
--num_workers 8 \

# After:
--num_workers 4 \
```

**Rationale:** Each worker loads data into RAM. 8 workers × large batches = high RAM

---

### Fix 4: Clear Model Cache Periodically
**File:** `src/training/trainer.py` - Add to end of `train_epoch` (after line 554)

```python
# Clear model's internal caches every epoch
if hasattr(self.model, 'clear_cache'):
    self.model.clear_cache()

# Force garbage collection
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

### Fix 5: Limit Validation Episodes
**File:** `train.py` line 281

```bash
# Current:
n_episodes=20,  # Fewer episodes for validation

# Change to:
n_episodes=10,  # Even fewer for memory safety
```

---

## Memory Monitoring Commands

### Before Training
```bash
# Check baseline
nvidia-smi
free -h
```

### During Training (separate terminal)
```bash
# Monitor every 2 seconds
watch -n 2 'nvidia-smi; free -h'

# Or use the diagnostic tool:
python diagnose_memory_leak.py --monitor_interval 10
```

### Log Analysis
```bash
# Check for memory growth in logs
grep -i "memory\|oom\|killed" train.log
```

---

## Expected Memory Usage After Fixes

| Component | Before | After |
|-----------|--------|-------|
| Baseline | 1.12 GB | 1.12 GB |
| Model + Optimizer | ~8 GB | ~8 GB |
| Training Batch | ~2 GB | ~2 GB |
| Triplet Batch | ~8 GB | ~1 GB ✓ |
| Validation | ~40 GB | ~2 GB ✓ |
| **Total Peak** | **~62 GB** | **~14 GB** ✓ |

---

## Testing Steps

1. Apply all fixes
2. Run with memory monitoring:
   ```bash
   bash train_stage_2.sh 2>&1 | tee train_memory_test.log
   ```
3. Watch for:
   - RAM staying < 20GB
   - No "Killed" message
   - Validation completing without spikes
4. If still OOM, reduce:
   - `--n_way 4` → `2`
   - `--n_query 8` → `4`
   - `--num_aug 5` → `3`

---

## Quick Reference: All Changes Needed

1. ✅ `src/training/trainer.py:763` - Add list deletion
2. ✅ `src/training/trainer.py:554` - Add epoch cleanup
3. ✅ `train_stage_2.sh:15` - triplet_batch_size: 64 → 8
4. ✅ `train_stage_2.sh:21` - num_workers: 8 → 4
5. ✅ `train.py:281` - val episodes: 20 → 10
