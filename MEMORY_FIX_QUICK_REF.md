# Memory Leak Fix - Quick Reference

## Problem
Training killed at line 38 of `train_stage_2.sh` with RAM growth: 1.12GB → 62GB → OOM

## Root Cause
**Validation loop accumulated unbounded lists of numpy arrays without cleanup**

## Fixes Applied ✅

### 1. Validation List Cleanup (`src/training/trainer.py:758-783`)
- Added explicit deletion of 6 large accumulator lists after validation
- Prevents ~500MB-2GB from accumulating per validation

### 2. Epoch Memory Cleanup (`src/training/trainer.py:554-569`)
- Added garbage collection after each epoch
- Clears model caches and CUDA memory

### 3. Reduced Triplet Batch Size (`train_stage_2.sh:15`)
- Changed from 64 → 8
- Saves ~7GB per triplet batch

### 4. Reduced Data Workers (`train_stage_2.sh:21`)
- Changed from 8 → 4 workers
- Reduces prefetch memory overhead by 50%

### 5. Reduced Validation Episodes (`train.py:281`)
- Changed from 20 → 10 episodes
- Halves validation memory usage

## Expected Result
- **Before:** Peak ~62GB → OOM crash
- **After:** Peak ~14GB → stable training ✓

## Test Command
```bash
# Monitor in separate terminal
watch -n 2 'nvidia-smi; echo "---"; free -h'

# Run training
bash train_stage_2.sh 2>&1 | tee train_memory_test.log
```

## If Still OOM
Reduce in order:
1. `--n_way 4` → `2`
2. `--n_query 8` → `4`
3. `--num_aug 5` → `3`
4. Disable `--use_triplet` temporarily

## Files Modified
1. `src/training/trainer.py` (2 locations)
2. `train_stage_2.sh` (2 lines)
3. `train.py` (1 line)

## Documentation
- `MEMORY_LEAK_DIAGNOSIS.md` - Detailed analysis
- `MEMORY_LEAK_FIXES_SUMMARY.md` - Complete fix documentation
- `diagnose_memory_leak.py` - Memory profiling tool
