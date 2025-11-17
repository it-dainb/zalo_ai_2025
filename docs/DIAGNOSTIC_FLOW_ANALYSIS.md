# Diagnostic Flow Analysis

## Overview
This document provides a comprehensive analysis of the diagnostic data flow through the training pipeline, including inputs, outputs, and usage patterns.

## Complete Flow Diagram

```
prepare_loss_inputs (loss_utils.py)
         |
         | Creates diagnostic_data dict (line 369-374)
         v
    loss_inputs dict
    ├── pred_bboxes
    ├── pred_cls_logits
    ├── target_bboxes
    ├── target_cls
    ├── [contrastive/triplet inputs...]
    └── diagnostic_data  ← Contains anchor assignment info
         |
         v
    Trainer._forward_step() / validate() (trainer.py)
         |
         | 1. Extract diagnostic_data (line 927, 1186)
         | 2. Pop from loss_inputs before passing to loss_fn
         v
    diagnostic_data → loss_fn(**loss_inputs)  ← No diagnostic_data passed
         |                    |
         | (stored)           v
         |              ReferenceBasedDetectionLoss.forward()
         |              (combined_loss.py)
         |              - Does NOT accept diagnostic_data
         |              - Only accepts detection/contrastive params
         v
    _current_diagnostic_data (trainer.py line 1201)
    ├── loss_inputs
    ├── diagnostic_data
    ├── losses_dict
    └── batch
         |
         v
    TrainingDiagnostics.log_batch_diagnostics() (line 525)
         |
         | Extracts and logs:
         | - Anchor assignment quality
         | - BBox statistics
         | - Loss components
         v
    Cleared after logging (line 540)
```

## Key Components

### 1. diagnostic_data Creation (loss_utils.py:369-374)

**Location**: `src/training/loss_utils.py:336-374`

**When Created**: Only when raw detection head outputs are available:
```python
if 'prototype_boxes' in model_outputs and 'prototype_sim' in model_outputs:
```

**Contents**:
```python
diagnostic_data = {
    'anchor_points': matched_anchor_points,      # (M, 2) anchor centers in pixels
    'strides': matched_assigned_strides,         # (M,) stride values
    'proto_boxes_list': proto_boxes,             # List[Tensor] raw proto boxes
    'proto_sim_list': proto_sim,                 # List[Tensor] raw proto similarities
}
```

**Purpose**: Store anchor assignment information for debugging and monitoring

### 2. diagnostic_data Extraction (trainer.py)

**Two Extraction Points**:

#### A. Validation Loop (line 926-927)
```python
# Remove diagnostic_data before passing to loss function
diagnostic_data = loss_inputs.pop('diagnostic_data', None)
losses = self.loss_fn(**loss_inputs)
```
- **Used in**: `validate()` method
- **Purpose**: Prevent TypeError when passing to loss function
- **Note**: Extracted but NOT stored (no diagnostics in validation)

#### B. Training Loop (line 1185-1206)
```python
# Extract diagnostic_data before passing to loss function
diagnostic_data = loss_inputs.pop('diagnostic_data', None)

# Compute loss
losses = self.loss_fn(**loss_inputs)

# Store diagnostic data for later use
if self.diagnostics.enable and diagnostic_data is not None:
    self._current_diagnostic_data = {
        'loss_inputs': loss_inputs,
        'diagnostic_data': diagnostic_data,
        'losses_dict': losses_dict,
        'batch': batch,
    }
```
- **Used in**: `_forward_step()` method
- **Purpose**: Store for diagnostic logging after backward pass
- **Stored as**: `self._current_diagnostic_data`

### 3. diagnostic_data Usage (trainer.py:519-540)

**Location**: Training loop after loss backward

**Condition**: `if self.diagnostics.enable and hasattr(self, '_current_diagnostic_data')`

**Unpacking**:
```python
diag_data = self._current_diagnostic_data
loss_inputs = diag_data['loss_inputs']
diagnostic_data = diag_data.get('diagnostic_data')

if diagnostic_data is not None:
    self.diagnostics.log_batch_diagnostics(
        step=self.global_step,
        pred_bboxes=loss_inputs['pred_bboxes'],
        target_bboxes=loss_inputs['target_bboxes'],
        pred_cls_logits=loss_inputs['pred_cls_logits'],
        target_cls=loss_inputs['target_cls'],
        anchor_points=diagnostic_data['anchor_points'],
        strides=diagnostic_data['strides'],
        losses_dict=diag_data['losses_dict'],
        proto_boxes_list=diagnostic_data['proto_boxes_list'],
        proto_sim_list=diagnostic_data['proto_sim_list'],
        gt_bboxes_list=diag_data['batch']['target_bboxes'],
        gt_classes_list=diag_data['batch']['target_classes'],
    )
```

**Cleanup**: `delattr(self, '_current_diagnostic_data')` after logging

### 4. TrainingDiagnostics.log_batch_diagnostics() (diagnostics.py:73-104)

**Parameters**:
```python
def log_batch_diagnostics(
    step: int,
    pred_bboxes: torch.Tensor,           # (M, 4) matched predictions
    target_bboxes: torch.Tensor,         # (M, 4) matched targets
    pred_cls_logits: Optional[Tensor],   # (M, K) classification logits
    target_cls: Optional[Tensor],        # (M, K) one-hot targets
    anchor_points: Optional[Tensor],     # (M, 2) anchor centers
    strides: Optional[Tensor],           # (M,) stride values
    losses_dict: Optional[Dict],         # Loss components
    proto_boxes_list: Optional[List],    # Raw proto boxes (before assignment)
    proto_sim_list: Optional[List],      # Raw proto similarities
    gt_bboxes_list: Optional[List],      # List of GT boxes per image
    gt_classes_list: Optional[List],     # List of GT classes per image
)
```

**Actions**:
1. **Anchor Assignment Quality** (every `log_frequency` batches)
   - Count matched anchors
   - Analyze assignment distribution across scales

2. **Coordinate Format Verification** 
   - Verify xyxy format
   - Check for invalid coordinates

3. **BBox Statistics**
   - IoU between predictions and targets
   - BBox sizes and aspect ratios
   - Track statistics history

4. **Loss Component Breakdown**
   - Log each loss component value
   - Monitor for NaN/Inf

## Data Flow Summary

### Input Sources
1. **model_outputs** from `YOLOv8nRefDet.forward()`
   - `prototype_boxes`: List of raw bbox predictions per scale
   - `prototype_sim`: List of raw similarity scores per scale
   - `query_features`: For contrastive learning
   - `support_prototypes`: For contrastive learning

2. **batch** from `RefDetCollator`
   - `query_images`: (B, 3, H, W)
   - `target_bboxes`: List of GT boxes per image
   - `target_classes`: List of GT classes per image
   - `num_classes`: Number of classes in episode

### Output Flow
1. **loss_inputs** (without diagnostic_data) → `ReferenceBasedDetectionLoss.forward()`
2. **diagnostic_data** → Stored temporarily → `TrainingDiagnostics.log_batch_diagnostics()`
3. **losses** → Backward pass → Metrics logging

## Critical Fix Applied

### Issue
`ReferenceBasedDetectionLoss.forward()` received unexpected keyword argument `diagnostic_data`

### Root Cause
- `prepare_loss_inputs()` adds `diagnostic_data` to loss_inputs dict (line 430)
- `ReferenceBasedDetectionLoss.forward()` doesn't accept this parameter
- Loss function signature only accepts detection/contrastive parameters

### Solution
Extract `diagnostic_data` BEFORE passing to loss function:
```python
# trainer.py:926-928 (validation)
diagnostic_data = loss_inputs.pop('diagnostic_data', None)
losses = self.loss_fn(**loss_inputs)

# trainer.py:1185-1189 (training)
diagnostic_data = loss_inputs.pop('diagnostic_data', None)
losses = self.loss_fn(**loss_inputs)
```

### Verification
✅ Both call sites (training and validation) now extract diagnostic_data  
✅ diagnostic_data is only used for logging, not loss computation  
✅ Loss function receives only the parameters it expects  
✅ No functionality is lost - diagnostics still work when enabled

## Usage Patterns

### When diagnostic_data is None
- Model outputs don't contain raw detection head outputs
- Fallback to decoded predictions (backward compatibility)
- No anchor assignment diagnostics available

### When diagnostic_data is Present
- Model outputs contain `prototype_boxes` and `prototype_sim`
- Anchor-based target assignment performed
- Full diagnostic information available
- Logged every `log_frequency` batches (default: 10)

### When Diagnostics are Disabled
- `diagnostic_data` is still created and passed through
- Simply not stored or logged (performance optimization)
- Can be enabled/disabled without code changes

## Related Files

1. **src/training/loss_utils.py**
   - `prepare_loss_inputs()`: Creates diagnostic_data
   - `assign_targets_to_anchors()`: Performs anchor assignment

2. **src/training/trainer.py**
   - `_forward_step()`: Extracts and stores diagnostic_data (training)
   - `validate()`: Extracts diagnostic_data (validation, no storage)
   - Training loop: Uses diagnostic_data for logging

3. **src/losses/combined_loss.py**
   - `ReferenceBasedDetectionLoss.forward()`: Does NOT accept diagnostic_data
   - Only accepts detection and contrastive learning parameters

4. **src/training/diagnostics.py**
   - `TrainingDiagnostics.log_batch_diagnostics()`: Consumes diagnostic_data
   - Performs comprehensive monitoring and logging

## Future Considerations

### Potential Enhancements
1. Add gradient flow diagnostics using diagnostic_data
2. Save diagnostic snapshots for offline analysis
3. Add visualization hooks for anchor assignments
4. Export diagnostic_data to TensorBoard/WandB

### Optimization Opportunities
1. Only create diagnostic_data when diagnostics are enabled
2. Use sampling instead of full logging at every interval
3. Add configurable verbosity levels
4. Compress diagnostic_data for large batches

## Testing
To verify diagnostic flow:
```bash
# Enable diagnostics
python train.py --stage 2 --enable_diagnostics --debug_mode

# Disable diagnostics (default)
python train.py --stage 2

# Run diagnostic tests
pytest src/tests/test_training_components.py::test_diagnostic_logging -v
```
