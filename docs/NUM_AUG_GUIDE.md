# Data Augmentation Multiplier (`num_aug`) Feature

## Overview

The `--num_aug` parameter allows you to increase your effective dataset size by generating multiple augmented versions of each training sample. This is particularly useful for small datasets or when you want more diverse training examples.

## How It Works

When `num_aug > 1`, the dataset creates multiple "virtual" samples from each real frame:
- Each frame is logically replicated `num_aug` times
- Each replica gets a different augmentation seed (via `aug_idx`)
- The augmentation pipeline applies random transformations, so each replica looks different

**Example:**
- Original dataset: 1000 frames
- With `--num_aug 4`: 4000 effective samples (1000 frames × 4 augmentations)

## Usage

### Basic Usage

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --stage 2 \
    --epochs 100 \
    --num_aug 4  # Generate 4 augmented versions per frame
```

### Recommended Values

| Dataset Size | Recommended `num_aug` | Reasoning |
|--------------|----------------------|-----------|
| < 500 frames | 4-5 | Small dataset needs more diversity |
| 500-2000 frames | 2-3 | Moderate augmentation helps |
| > 2000 frames | 1-2 | Large dataset has enough diversity |

### Advanced Configuration

For best results, combine `num_aug` with appropriate augmentation strategies per stage:

#### Stage 1 (Base Training)
```bash
--stage 1 --num_aug 3
```
- Strong augmentation is already applied (mosaic, mixup, etc.)
- Moderate `num_aug` adds diversity without overfitting

#### Stage 2 (Meta-Learning)
```bash
--stage 2 --num_aug 4
```
- Few-shot learning benefits from diverse examples
- Higher `num_aug` helps learn robust prototypes

#### Stage 3 (Fine-Tuning)
```bash
--stage 3 --num_aug 2
```
- Conservative augmentation for final convergence
- Lower `num_aug` prevents interference with fine-tuning

## Implementation Details

### Dataset Modification

The `RefDetDataset` was enhanced to support `num_aug`:

```python
class RefDetDataset(Dataset):
    def __init__(self, ..., num_aug: int = 1):
        self.num_aug = max(1, num_aug)
    
    def __len__(self):
        base_count = sum(len(data['frame_indices']) for data in self.class_data.values())
        return base_count * self.num_aug  # Multiply by augmentation factor
    
    def __getitem__(self, idx):
        # Map augmented index to (base_idx, aug_idx)
        base_idx = idx % base_count
        aug_idx = idx // base_count  # Which augmentation version (0 to num_aug-1)
        
        # Load the actual frame at base_idx
        # aug_idx is stored in the sample dict for potential use
```

### Augmentation Pipeline

The augmentation pipeline automatically generates different versions:
- **Query augmentation**: Mosaic, MixUp, color jitter, geometric transforms
- **Support augmentation**: Conservative transforms to preserve prototype quality
- Each `aug_idx` ensures different random seeds in the augmentation pipeline

### Memory & Performance

**Memory Usage:**
- `num_aug` does NOT increase memory usage
- Original images are loaded once and augmented on-the-fly
- Only augmented tensors are kept in memory temporarily

**Training Speed:**
- Minimal impact: augmentation happens during data loading
- With `num_workers > 0`, augmentation is parallelized
- Recommended: `--num_workers 8` for smooth training

## Examples

### Example 1: Small Dataset with High Augmentation

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --stage 2 \
    --epochs 150 \
    --n_way 4 \
    --n_query 8 \
    --n_episodes 200 \
    --num_aug 5 \
    --use_triplet \
    --triplet_ratio 0.3 \
    --negative_strategy mixed \
    --mixed_precision \
    --num_workers 8
```

**Result:** With 500 base frames, you get 2500 effective samples.

### Example 2: Medium Dataset with Moderate Augmentation

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --stage 2 \
    --epochs 100 \
    --n_way 4 \
    --n_query 8 \
    --num_aug 3 \
    --mixed_precision
```

**Result:** With 1000 base frames, you get 3000 effective samples.

### Example 3: Large Dataset with Minimal Augmentation

```bash
python train.py \
    --data_root ./datasets/train/samples \
    --annotations ./datasets/train/annotations/annotations.json \
    --stage 2 \
    --epochs 80 \
    --num_aug 1  # Default, no multiplier
```

**Result:** Standard training without augmentation multiplier.

## Best Practices

### 1. Start Conservative
- Begin with `--num_aug 2` and increase if needed
- Monitor validation performance for overfitting signs

### 2. Combine with Other Techniques
```bash
--num_aug 4 \
--use_triplet \
--triplet_ratio 0.3 \
--negative_strategy mixed
```

### 3. Adjust Training Duration
- More augmented samples → may need fewer epochs
- Example: With `--num_aug 4`, reduce epochs by 20-30%

### 4. Monitor Training Metrics
- Check if training loss decreases smoothly
- Watch for validation mAP improvements
- Look for signs of overfitting (train >> val performance)

## Validation Behavior

**Important:** Validation/test datasets always use `num_aug=1`
- Ensures consistent evaluation
- No augmentation multiplier during validation
- Fair comparison across different training configurations

## Troubleshooting

### Issue: Training is too slow
**Solution:**
- Increase `--num_workers` (try 8 or 12)
- Reduce `--num_aug` (try 2 or 3)
- Enable `--mixed_precision`

### Issue: Out of memory errors
**Solution:**
- Reduce `--batch_size` or `--n_query`
- Reduce `--num_workers` (frees worker memory)
- Note: `num_aug` itself doesn't increase memory

### Issue: Overfitting (train >> val performance)
**Solution:**
- Increase `--num_aug` for more diversity
- Add `--use_triplet` for contrastive learning
- Increase `--weight_decay`

### Issue: Underfitting (both train and val performance low)
**Solution:**
- Decrease `--num_aug` (too much augmentation can hurt)
- Train longer (`--epochs`)
- Increase model capacity (but YOLOv8n is fixed)

## Technical Notes

### Thread Safety
- Each worker process has independent random state
- `aug_idx` ensures reproducibility when needed
- Safe for multi-worker data loading

### Reproducibility
To reproduce results with `num_aug`:
```bash
# Set random seeds in your training script
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Then run with num_aug
python train.py --num_aug 4 --seed 42
```

### Episodic Sampling Compatibility
`num_aug` works seamlessly with episodic sampling:
- Episodes are sampled from the augmented dataset
- Each episode can contain different augmented versions of the same frames
- Increases episodic diversity

## Summary

The `--num_aug` parameter is a powerful tool to increase dataset diversity:
- ✅ Easy to use: just add `--num_aug N`
- ✅ No memory overhead: on-the-fly augmentation
- ✅ Works with all training stages
- ✅ Compatible with episodic sampling and triplet learning
- ✅ Recommended for small-to-medium datasets

**Quick Start:**
```bash
python train.py --num_aug 4 [other args...]
```
