# Diagnostic Data Quick Reference

## 🔍 What is diagnostic_data?

A dictionary containing anchor assignment information for debugging and monitoring training progress.

**Contents**:
```python
{
    'anchor_points': Tensor,       # (M, 2) - Center points of matched anchors
    'strides': Tensor,             # (M,) - Stride values (8, 16, 32, 64)
    'proto_boxes_list': List[Tensor],  # Raw predictions before assignment
    'proto_sim_list': List[Tensor],    # Raw similarities before assignment
}
```

## 🔄 Flow Overview

```
prepare_loss_inputs → loss_inputs (with diagnostic_data)
                              ↓
                    Extract diagnostic_data
                              ↓
                    loss_fn (without diagnostic_data)
                              ↓
                    Store for logging (training only)
                              ↓
                    TrainingDiagnostics.log_batch_diagnostics()
```

## 📝 How to Use

### In Training Code
```python
# Prepare loss inputs (includes diagnostic_data)
loss_inputs = prepare_loss_inputs(model_outputs, batch, stage=2)

# Extract diagnostic_data BEFORE passing to loss function
diagnostic_data = loss_inputs.pop('diagnostic_data', None)

# Compute loss (clean kwargs)
losses = loss_fn(**loss_inputs)

# Store for logging (optional, training only)
if diagnostics.enable and diagnostic_data is not None:
    self._current_diagnostic_data = {
        'loss_inputs': loss_inputs,
        'diagnostic_data': diagnostic_data,
        'losses_dict': losses_dict,
        'batch': batch,
    }
```

### In Test Code
```python
# Prepare loss inputs
loss_inputs = prepare_loss_inputs(model_outputs, batch, stage=2)

# Remove diagnostic_data before passing to loss function
diagnostic_data = loss_inputs.pop('diagnostic_data', None)

# Compute loss
losses = loss_fn(**loss_inputs)
```

## ⚠️ Common Pitfalls

### ❌ DON'T: Pass diagnostic_data to loss function
```python
loss_inputs = prepare_loss_inputs(...)
losses = loss_fn(**loss_inputs)  # ERROR! diagnostic_data not expected
```

### ✅ DO: Extract it first
```python
loss_inputs = prepare_loss_inputs(...)
diagnostic_data = loss_inputs.pop('diagnostic_data', None)
losses = loss_fn(**loss_inputs)  # OK!
```

## 🔧 Loss Function Signature

```python
def forward(
    self,
    # Detection outputs
    pred_bboxes,
    pred_cls_logits,
    # Targets
    target_bboxes,
    target_cls,
    # Contrastive learning (optional)
    query_features=None,
    support_prototypes=None,
    feature_labels=None,
    proposal_features=None,
    proposal_labels=None,
    # Triplet loss (optional)
    triplet_anchors=None,
    triplet_positives=None,
    triplet_negatives=None,
    triplet_embeddings=None,
    triplet_labels=None,
    # ❌ NO diagnostic_data parameter!
):
```

## 📊 When is diagnostic_data Available?

### ✅ Available When:
- Model outputs contain `prototype_boxes` and `prototype_sim`
- Anchor-based target assignment is used
- Training with raw detection head outputs

### ❌ Not Available When:
- Using decoded predictions (backward compatibility mode)
- Model outputs don't contain raw head outputs
- `diagnostic_data` will be `None` in these cases

## 🛠️ Diagnostic Logging

### Enable/Disable
```python
# Enable diagnostics
trainer = RefDetTrainer(
    model=model,
    enable_diagnostics=True,    # Enable diagnostic logging
    log_frequency=10,            # Log every 10 batches
    detailed_frequency=50,       # Detailed logs every 50 batches
)

# Disable diagnostics (default)
trainer = RefDetTrainer(
    model=model,
    enable_diagnostics=False,   # No diagnostic overhead
)
```

### What Gets Logged
1. **Anchor Assignment Quality**
   - Number of matched anchors
   - Distribution across scales (P2, P3, P4, P5)
   
2. **Coordinate Format Verification**
   - Check for invalid coordinates
   - Verify xyxy format
   
3. **BBox Statistics**
   - IoU between predictions and targets
   - BBox sizes and aspect ratios
   
4. **Loss Components**
   - Individual loss values
   - NaN/Inf detection

## 📍 Key Files

| File | Purpose |
|------|---------|
| `src/training/loss_utils.py:369` | Creates diagnostic_data |
| `src/training/trainer.py:926` | Extracts in validation |
| `src/training/trainer.py:1185` | Extracts in training |
| `src/training/trainer.py:525` | Logs diagnostics |
| `src/training/diagnostics.py:73` | Diagnostic logging logic |
| `src/losses/combined_loss.py:111` | Loss function (no diagnostic_data) |

## 🧪 Testing

### Verify Fix Works
```bash
# Test that loss function doesn't receive diagnostic_data
python test_supcon_fix.py

# Train with diagnostics enabled
python train.py --stage 2 --enable_diagnostics

# Check logs for diagnostic output
tail -f logs/train_stage2_*.log | grep "Anchor Assignment"
```

## 💡 Key Takeaways

1. **diagnostic_data is NOT a loss input** - it's metadata for logging
2. **Always extract before passing to loss_fn** - use `.pop('diagnostic_data', None)`
3. **Validation doesn't need diagnostics** - extract but don't store
4. **Training stores for logging** - only when diagnostics are enabled
5. **None is OK** - code handles missing diagnostic_data gracefully

## 📚 Related Documentation

- [DIAGNOSTIC_FLOW_ANALYSIS.md](DIAGNOSTIC_FLOW_ANALYSIS.md) - Complete flow diagram
- [DIAGNOSTIC_FLOW_FIX_SUMMARY.md](DIAGNOSTIC_FLOW_FIX_SUMMARY.md) - Fix details
- [DEBUG_LOGGING_GUIDE.md](DEBUG_LOGGING_GUIDE.md) - General debugging guide
