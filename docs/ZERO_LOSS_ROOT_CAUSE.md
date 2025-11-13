# Zero Loss Root Cause & Solution

## Problem Statement

Training fails with `AssertionError: No inf checks were recorded for this optimizer` on **every batch**. All loss components are 0.0.

## Root Cause

The training pipeline has a **critical architectural mismatch** between model outputs and loss computation:

```
┌─────────────────────────────────────────────────────────────────────┐
│ MODEL FORWARD PASS                                                   │
│   YOLOv8n Backbone + PSALM Fusion + Dual Detection Head            │
│   ↓                                                                  │
│   Returns: {'prototype_boxes': List[(B,68,H,W)],                   │
│             'prototype_sim': List[(B,K,H,W)]}                       │
│   (Raw multi-scale detection head outputs - NOT decoded)            │
└─────────────────────────────────────────────────────────────────────┘
                             ↓
                    ❌ MISSING COMPONENT ❌
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ LOSS COMPUTATION                                                     │
│   prepare_loss_inputs() expects:                                    │
│     - 'pred_bboxes': (N, 4) decoded boxes in xyxy format           │
│     - 'pred_scores': (N,) confidence scores                         │
│     - 'pred_cls_logits': (N, K) class logits                       │
│     - 'pred_dfl_dist': (N, 68) DFL distributions                   │
│   ↓                                                                  │
│   Gets empty tensors (0, 4), (0,), etc.                            │
│   ↓                                                                  │
│   match_predictions_to_targets() finds 0 matches                    │
│   ↓                                                                  │
│   ALL LOSSES = 0.0 → No gradients → GradScaler assertion error     │
└─────────────────────────────────────────────────────────────────────┘
```

### What's Missing

**Missing Step**: Decode raw detection head outputs into unified prediction format

OR

**Alternative**: Use anchor-based target assignment (YOLOv8-style) instead of prediction matching

## Evidence

From debug logs:
```
Loss components: {'bbox_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0, ...}
0/546 parameters have non-zero gradients
```

From trainer.py:687-751 debug output:
```python
# Model outputs contain raw detection head tensors:
model_outputs = {
    'prototype_boxes': [Tensor(B, 68, H, W), ...],  # 4 scales
    'prototype_sim': [Tensor(B, K, H, W), ...],      # 4 scales
}

# But prepare_loss_inputs expects:
expected = {
    'pred_bboxes': Tensor(N, 4),  # ← Does NOT exist in model_outputs
    'pred_scores': Tensor(N),      # ← Does NOT exist
    ...
}
```

##  Solution Options

### Option A: Add Prediction Decoding (Simple, Fast)

Add a `decode_predictions()` function to `src/training/loss_utils.py`:

```python
def decode_predictions(
    proto_boxes: List[torch.Tensor],  # [(B, 68, H, W), ...]  
    proto_sim: List[torch.Tensor],     # [(B, K, H, W), ...]
    conf_thres: float = 0.001,          # Very low for training
) -> Dict[str, torch.Tensor]:
    """
    Decode raw detection head outputs to prediction format.
    
    During training, use very low confidence threshold to keep
    most predictions so model can learn from errors.
    """
    # For each scale:
    #   1. Flatten spatial dimensions: (B, C, H, W) → (B*H*W, C)
    #   2. Get top-K by similarity score (e.g., K=100 per scale)
    #   3. Decode DFL distribution to xyxy boxes
    #   4. Return concatenated predictions from all scales
    ...
```

**Pros**:
- Minimal changes to existing code
- Matches current loss function expectations
- Can control number of predictions via top-K

**Cons**:
- Still filters predictions (may lose some gradient flow)
- Adds computational overhead during training

### Option B: Anchor-Based Assignment (Correct, YOLOv8-style)

Redesign `prepare_loss_inputs()` to work with raw anchor predictions:

```python
def assign_targets_to_anchors(
    proto_boxes: List[torch.Tensor],  # Raw anchor predictions
    proto_sim: List[torch.Tensor],
    target_bboxes: List[torch.Tensor],  # Ground truth boxes
    target_classes: List[torch.Tensor],
) -> Dict:
    """
    Assign ground truth targets to anchor points using:
    1. Center-based assignment (GT center falls in anchor cell)
    2. IoU-based filtering (optional)
    3. Task-aligned assignment (optional, for advanced)
    
    Returns ALL anchors with assigned/unassigned flags.
    This ensures gradient flow even for background anchors.
    """
    # For each GT box:
    #   1. Find which anchor points it overlaps (by center or IoU)
    #   2. Assign that GT as positive target for those anchors
    #   3. Mark other anchors as negative/background
    # Return: (anchor_predictions, anchor_targets, positive_mask)
    ...
```

**Pros**:
- Correct YOLOv8 approach
- Maximum gradient flow (uses ALL anchors)
- No artificial confidence filtering

**Cons**:
- Requires rewriting `prepare_loss_inputs()` and potentially loss functions
- More complex implementation

### Option C: Hybrid Approach (Recommended for Now)

1. **Short-term**: Add simple decoding with very low threshold (Option A)
2. **Long-term**: Migrate to anchor-based assignment (Option B)

## Recommended Action

### Immediate Fix (Unblock Training)

Add this to `src/training/loss_utils.py`:

```python
def decode_detection_outputs_simple(
    model_outputs: Dict[str, torch.Tensor],
    num_classes: int,
    top_k: int = 300,  # Keep top-300 predictions per image
) -> Dict[str, torch.Tensor]:
    """Simple decoding to unblock training."""
    
    proto_boxes = model_outputs['prototype_boxes']  # List[(B, 68, H, W)]
    proto_sim = model_outputs['prototype_sim']      # List[(B, K, H, W)]
    
    # For each scale, flatten and take top-K by max similarity
    all_boxes, all_scores, all_logits, all_dfl = [], [], [], []
    
    for boxes, sim in zip(proto_boxes, proto_sim):
        B, C_box, H, W = boxes.shape
        _, K, _, _ = sim.shape
        
        # Flatten: (B, C, H, W) → (B, H*W, C)
        boxes_flat = boxes.permute(0, 2, 3, 1).reshape(B, -1, C_box)  # (B, H*W, 68)
        sim_flat = sim.permute(0, 2, 3, 1).reshape(B, -1, K)          # (B, H*W, K)
        
        # Get scores: max similarity per anchor
        scores, _ = sim_flat.max(dim=2)  # (B, H*W)
        
        # Take top-K per image
        topk_scores, topk_idx = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        
        # Gather corresponding predictions
        for b in range(B):
            idx = topk_idx[b]
            all_boxes.append(boxes_flat[b, idx])  # (K, 68)
            all_scores.append(topk_scores[b])      # (K,)
            all_logits.append(sim_flat[b, idx])    # (K, num_classes)
            all_dfl.append(boxes_flat[b, idx])      # (K, 68) - same as boxes
    
    # TODO: Decode boxes from DFL to xyxy (for now, use dummy coordinates)
    device = proto_boxes[0].device
    total_preds = sum(len(s) for s in all_scores)
    
    return {
        'pred_bboxes': torch.zeros((total_preds, 4), device=device),  # TODO: decode
        'pred_scores': torch.cat(all_scores),
        'pred_cls_logits': torch.cat(all_logits),
        'pred_dfl_dist': torch.cat(all_dfl),
    }
```

Then modify `prepare_loss_inputs()` line 127-130:

```python
# Check if outputs need decoding
if 'pred_bboxes' not in model_outputs:
    # Decode raw detection head outputs
    decoded = decode_detection_outputs_simple(
        model_outputs, 
        num_classes=batch['num_classes'],
        top_k=300
    )
    pred_bboxes = decoded['pred_bboxes']
    pred_scores = decoded['pred_scores']
    pred_cls_logits = decoded['pred_cls_logits']
    pred_dfl_dist = decoded['pred_dfl_dist']
else:
    # Already decoded (backward compat)
    pred_bboxes = model_outputs['pred_bboxes']
    pred_scores = model_outputs['pred_scores']
    pred_cls_logits = model_outputs['pred_cls_logits']
    pred_dfl_dist = model_outputs['pred_dfl_dist']
```

## Testing Plan

1. Add decode function
2. Run 1 epoch to verify:
   - Non-zero losses
   - Gradients flow to all parameters
   - No GradScaler errors
3. If working, proceed with full training
4. Plan migration to anchor-based assignment for better performance

## Files to Modify

- `src/training/loss_utils.py`: Add `decode_detection_outputs_simple()` and update `prepare_loss_inputs()`
- `src/training/trainer.py`: Remove debug logging after verification

## References

- YOLOv8 Task-Aligned Assigner: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py
- DFL decoding: `src/losses/dfl_loss.py`
- Detection head: `src/models/dual_head.py`
