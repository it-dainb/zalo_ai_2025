# Complete Debug Logging Guide for YOLOv8n-RefDet

## Overview

The YOLOv8n-RefDet training system now includes a **comprehensive debug logging system** that captures EVERYTHING during training when the `--debug` flag is enabled. This allows you to trace every detail of the training process for debugging and analysis.

---

## Quick Start

### Training with Debug Logging

```bash
# WITH --debug flag: Captures EVERYTHING at DEBUG level
# - Logs to file: checkpoints/training_stage2_2way_4query_bs2_20251116_232000.log
# - Also prints to console for real-time monitoring
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --debug

# WITHOUT --debug flag: Normal operation (INFO level only)
# - Logs to file: checkpoints/training_stage2_2way_4query_bs2_20251116_232000.log
# - Clean console output (no spam)
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4
```

### Evaluation with Debug Logging

```bash
# WITH --debug flag
python evaluate.py --checkpoint checkpoints/best_model.pt --test_data_root ./datasets/test/samples --debug

# WITHOUT --debug flag
python evaluate.py --checkpoint checkpoints/best_model.pt --test_data_root ./datasets/test/samples
```

---

## What Gets Logged at DEBUG Level

When `--debug` flag is enabled, the system captures:

### 1. **Batch Information** (every 50 batches + batch 0)
```
DEBUG    | ======================================================================
DEBUG    | BATCH 0 - Type: detection
DEBUG    | ======================================================================
DEBUG    | Batch Contents:
DEBUG    |   query_images:
DEBUG    |     Shape: torch.Size([2, 3, 640, 640])
DEBUG    |     Dtype: torch.float32
DEBUG    |     Device: cuda:0
DEBUG    |     Range: [-2.1179, 2.6400]
DEBUG    |     Mean: 0.0934
DEBUG    |     Std: 1.0123
DEBUG    |     Has NaN: False
DEBUG    |     Has Inf: False
DEBUG    |   support_images:
DEBUG    |     Shape: torch.Size([2, 1, 3, 512, 512])
DEBUG    |     ...
```

### 2. **Forward Pass Details**
```
DEBUG    | Detection Forward Pass:
DEBUG    |   Support images: N=2, K=1, C=3, H=512, W=512
DEBUG    |   Query images: torch.Size([2, 3, 640, 640])
DEBUG    | 
DEBUG    | Model Outputs:
DEBUG    |   proto_boxes_0: torch.Size([2, 4, 80, 80]), range=[-0.0123, 9.9876]
DEBUG    |   proto_sim_0: torch.Size([2, 2, 80, 80]), range=[0.0234, 0.9876]
DEBUG    |   proto_boxes_1: torch.Size([2, 4, 40, 40]), range=[-0.0156, 9.9789]
DEBUG    |   ...
```

### 3. **Loss Components** (every 50 batches + batch 0)
```
DEBUG    | Loss Components:
DEBUG    |   Total Loss: 12.345678
DEBUG    |   wiou_loss: 8.234567
DEBUG    |   bce_loss: 2.345678
DEBUG    |   dfl_loss: 1.765433
DEBUG    |   supcon_loss: 0.000000
```

### 4. **Gradient Information** (every 50 batches + batch 0)
```
DEBUG    | Gradient Norms:
DEBUG    |   Total Grad Norm: 3.4567e+01
DEBUG    |   backbone.conv1.weight: 5.6789e+00
DEBUG    |   backbone.layer1.0.conv1.weight: 4.2345e+00
DEBUG    |   detection_head.cls_head.weight: 3.8901e+00
DEBUG    |   detection_head.reg_head.weight: 2.7654e+00
DEBUG    |   support_encoder.blocks.0.attn.qkv.weight: 1.9876e+00
DEBUG    | 
DEBUG    | Gradient Clipping:
DEBUG    |   Norm before clip: 3.4567e+01
DEBUG    |   Norm after clip: 1.0000e+01
DEBUG    |   Clip threshold: 10.0
```

### 5. **Triplet Loss Details** (when using triplet training)
```
DEBUG    | Triplet Forward Pass:
DEBUG    |   Anchor images: torch.Size([8, 3, 256, 256])
DEBUG    |   Positive images: torch.Size([8, 3, 640, 640])
DEBUG    |   Negative images: torch.Size([8, 3, 640, 640])
DEBUG    | 
DEBUG    | Triplet Features:
DEBUG    |   Anchor: torch.Size([8, 384]), norm=12.3456
DEBUG    |   Positive: torch.Size([8, 256]), norm=10.2345
DEBUG    |   Negative: torch.Size([8, 256]), norm=9.8765
```

### 6. **NaN/Inf Detection**
```
WARNING  | ⚠️ NaN/Inf gradient in backbone.layer2.0.conv1.weight
DEBUG    |    Grad stats: min=-1.2345e+10, max=3.4567e+10, mean=nan
ERROR    | ❌ NaN/Inf gradients detected at batch 42. Skipping optimizer step.
```

### 7. **Anchor Assignment Details** (first batch of epoch 1)
The system performs detailed per-loss gradient checking on the first detection batch to isolate potential NaN sources:
```
🔍 Performing per-loss gradient check on first DETECTION batch (batch_idx=0)...
Available loss components: ['wiou_loss', 'bce_loss', 'dfl_loss', 'supcon_loss']
  Testing wiou_loss (value=8.234567)...
  Testing bce_loss (value=2.345678)...
  Testing dfl_loss (value=1.765433)...
  Testing supcon_loss (value=0.000000)...

Per-Loss Gradient Test Results:
  wiou_loss: ✅ OK
  bce_loss: ✅ OK
  dfl_loss: ✅ OK
  supcon_loss: SKIPPED (zero)
```

---

## Log File Structure

### Filename Format
```
{script_name}_stage{stage}_{experiment_name}_{timestamp}.log
```

**Examples:**
- `training_stage2_2way_4query_bs2_20251116_232000.log`
- `training_stage3_triplet_finetune_20251116_234500.log`
- `evaluation_best_model_2way_4query_50episodes_20251117_010000.log`

### Log File Location
- **Training logs**: `{checkpoint_dir}/training_*.log`
- **Evaluation logs**: `{checkpoint_dir}/evaluation_*.log` (or custom `--log_dir`)

---

## Log Levels Explained

| Level | When Active | What It Captures |
|-------|-------------|------------------|
| **DEBUG** | Only with `--debug` flag | Everything: batch contents, tensor shapes/stats, gradients, intermediate computations |
| **INFO** | Always | High-level progress: epoch start/end, loss values, validation metrics |
| **WARNING** | Always | Potential issues: NaN gradients, memory warnings, convergence concerns |
| **ERROR** | Always | Critical failures: NaN losses, invalid tensors, crashes |

### Log Level Behavior

**Without `--debug` flag** (INFO level):
- File contains: INFO, WARNING, ERROR messages
- Console: Clean progress output
- Example: "Epoch 1/100: Loss=5.234, ST-IoU=0.678"

**With `--debug` flag** (DEBUG level):
- File contains: DEBUG, INFO, WARNING, ERROR messages (everything!)
- Console: Full debug output + progress
- Example: "DEBUG | Batch 0: query_images shape=[2,3,640,640], mean=0.093..."

---

## Advanced Usage

### 1. Grep for Specific Information

```bash
# Find all NaN/Inf warnings
grep -i "nan\|inf" checkpoints/training_stage2_2way_4query_bs2_20251116_232000.log

# Find all gradient norm values
grep "Total Grad Norm" checkpoints/training_stage2_*.log

# Find loss components for batch 0
grep -A 10 "BATCH 0" checkpoints/training_stage2_*.log | grep "Loss Components"

# Track gradient explosion
grep "Norm before clip" checkpoints/training_stage2_*.log | awk '{print $NF}'
```

### 2. Monitor Log in Real-Time

```bash
# Follow training log as it's being written
tail -f checkpoints/training_stage2_2way_4query_bs2_20251116_232000.log

# Follow and filter for errors only
tail -f checkpoints/training_stage2_*.log | grep -i "error\|warning"
```

### 3. Compare Runs

```bash
# Compare loss progression across runs
for log in checkpoints/training_stage2_*.log; do
    echo "=== $log ==="
    grep "Total Loss:" "$log" | head -5
done
```

---

## What's Being Logged Where

| Component | Location | Log Level | Frequency |
|-----------|----------|-----------|-----------|
| Batch content details | `trainer.py:_log_batch_info()` | DEBUG | Every 50 batches + batch 0 |
| Forward pass outputs | `trainer.py:_forward_detection_step()` | DEBUG | Every 50 batches + batch 0 |
| Loss components | `trainer.py:train_epoch()` | DEBUG | Every 50 batches + batch 0 |
| Gradient norms | `trainer.py:train_epoch()` | DEBUG | Every 50 batches + batch 0 |
| Gradient clipping | `trainer.py:train_epoch()` | DEBUG | Every 50 batches + batch 0 |
| NaN/Inf detection | `trainer.py:train_epoch()` | WARNING/ERROR | Every occurrence |
| Per-loss gradient check | `trainer.py:train_epoch()` | INFO | First detection batch of epoch 1 |
| Epoch metrics | `trainer.py:train()` | INFO | Every epoch |
| Validation results | `trainer.py:validate()` | INFO | Every validation |
| Checkpoint saving | `trainer.py:save_checkpoint()` | INFO | Every save |

---

## Logging Architecture

```python
# In train.py:
logger = setup_logger(
    name='training',
    log_dir=args.checkpoint_dir,
    stage=args.stage,
    debug=args.debug,  # Controls DEBUG level
    experiment_name='2way_4query_bs2'
)

# Logger passed to trainer:
trainer = RefDetTrainer(..., logger=logger, debug_mode=args.debug)

# Inside trainer methods:
if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
    self.logger.debug(f"Detailed information...")

# Always log critical info:
self.logger.info(f"Epoch {epoch}: Loss={loss:.4f}")
self.logger.warning(f"⚠️ Warning message")
self.logger.error(f"❌ Critical error")
```

---

## Troubleshooting with Debug Logs

### Problem: Training loss explodes
**Solution:**
```bash
# Enable debug logging and check gradients
python train.py --stage 2 --epochs 1 --debug

# In the log file, look for:
grep "Grad Norm" checkpoints/training_*.log
grep "Gradient Clipping" checkpoints/training_*.log
```

### Problem: NaN loss appears
**Solution:**
```bash
# Debug logs show exact batch and loss component causing NaN
grep -B 20 -A 5 "NaN or Inf detected" checkpoints/training_*.log

# Check the per-loss gradient test results (first batch of training)
grep -A 10 "Per-Loss Gradient Test Results" checkpoints/training_*.log
```

### Problem: Slow convergence
**Solution:**
```bash
# Check loss component balance
grep "Loss Components" checkpoints/training_*.log | head -20

# Check if one component dominates
awk '/Total Loss:/{print $NF}' checkpoints/training_*.log > total_losses.txt
awk '/wiou_loss:/{print $NF}' checkpoints/training_*.log > wiou_losses.txt
awk '/bce_loss:/{print $NF}' checkpoints/training_*.log > bce_losses.txt
```

### Problem: Memory issues
**Solution:**
```bash
# Check batch sizes and tensor shapes
grep "Shape:" checkpoints/training_*.log | sort | uniq -c

# Look for shape mismatches
grep "Shape:" checkpoints/training_*.log | grep -E "BATCH 0|BATCH 50|BATCH 100"
```

---

## Performance Considerations

### Debug Mode Overhead
- **CPU overhead**: ~5-10% due to additional logging calls
- **Disk I/O**: Log files can grow to 100MB-1GB for long training runs
- **Memory overhead**: Negligible (loggers are efficient)

### When to Use Debug Mode
✅ **Use `--debug` when:**
- Starting a new experiment (first 1-2 epochs)
- Investigating training instability
- Debugging NaN/Inf issues
- Tuning hyperparameters
- Verifying data loading is correct

❌ **Don't use `--debug` when:**
- Running final long training (100+ epochs)
- You're confident everything works
- Disk space is limited
- You need maximum performance

### Recommendation
```bash
# Start with debug for first 2 epochs
python train.py --stage 2 --epochs 2 --debug

# If everything looks good, continue without debug
python train.py --stage 2 --epochs 100 --resume checkpoints/checkpoint_epoch_2.pt
```

---

## Examples

### Example 1: Debug First Training Run
```bash
python train.py \
    --stage 2 \
    --epochs 5 \
    --n_way 2 \
    --n_query 4 \
    --batch_size 2 \
    --data_root ./datasets/train/samples \
    --debug

# Check log file
ls -lh checkpoints/training_stage2_2way_4query_bs2_*.log
tail -100 checkpoints/training_stage2_2way_4query_bs2_*.log
```

### Example 2: Production Training (No Debug)
```bash
python train.py \
    --stage 2 \
    --epochs 100 \
    --n_way 2 \
    --n_query 4 \
    --batch_size 4 \
    --data_root ./datasets/train/samples

# Monitor training progress
tail -f checkpoints/training_stage2_2way_4query_bs4_*.log | grep "INFO"
```

### Example 3: Debug Specific Batch Range
```bash
# Run with debug, then analyze specific batches
python train.py --stage 2 --epochs 1 --debug

# Extract batch 0 details
sed -n '/BATCH 0/,/BATCH 1/p' checkpoints/training_*.log > batch_0_analysis.txt

# Extract loss progression for first 100 batches
grep "Total Loss:" checkpoints/training_*.log | head -100 > loss_progression.txt
```

---

## Integration with Other Tools

### 1. WandB Integration
Debug logs complement WandB by providing:
- **WandB**: High-level metrics visualization, experiment comparison
- **Debug logs**: Low-level debugging, tensor inspection, gradient analysis

```python
# Both enabled simultaneously
python train.py --stage 2 --epochs 100 --use_wandb --debug
```

### 2. TensorBoard (Future)
Debug logs provide textual trace while TensorBoard provides visual analysis.

### 3. Custom Analysis Scripts
```python
import re

def parse_loss_components(log_file):
    """Extract loss components from debug log."""
    losses = []
    with open(log_file) as f:
        for line in f:
            if "Total Loss:" in line:
                match = re.search(r"Total Loss: ([\d.]+)", line)
                if match:
                    losses.append(float(match.group(1)))
    return losses

losses = parse_loss_components("checkpoints/training_stage2_*.log")
print(f"Loss progression: {losses[:10]}")
```

---

## Summary

The enhanced debug logging system provides:
- ✅ **Complete visibility** into training when `--debug` is enabled
- ✅ **Zero console spam** when debug is disabled (clean user experience)
- ✅ **Persistent logs** always saved to file (both modes)
- ✅ **Rich context** with meaningful, timestamped filenames
- ✅ **Detailed traces** of batches, gradients, losses, and model outputs
- ✅ **NaN/Inf tracking** with automatic isolation of problematic components
- ✅ **Minimal overhead** when debug is disabled (~0% performance impact)

**Key takeaway**: Use `--debug` flag liberally during development and troubleshooting. It captures everything you need to diagnose training issues without any code changes.
