# ST-IoU Quick Reference Guide

## What is ST-IoU?

**Spatio-Temporal IoU (ST-IoU)** measures detection accuracy across both space (bounding box location) and time (frame coverage) in video sequences.

### Formula
```
ST-IoU = Σ(IoU(B_f, B'_f) for f in intersection) / |union|
```

Where:
- `IoU(B_f, B'_f)` = Spatial IoU between GT and predicted box at frame f
- `intersection` = Frames where both GT and prediction exist
- `union` = All frames where GT or prediction exists

### Example
```python
Ground Truth:  [Frame 0] [Frame 1] [Frame 2] --------
Prediction:    -------- [Frame 1] [Frame 2] [Frame 3]

intersection = {1, 2}  (2 frames)
union = {0, 1, 2, 3}   (4 frames)

ST-IoU = (IoU@frame1 + IoU@frame2) / 4
       = (0.85 + 0.82) / 4
       = 0.4175
```

---

## Quick Start

### 1. Import Metrics
```python
from src.metrics.st_iou import compute_st_iou, compute_spatial_iou
from src.metrics.detection_metrics import compute_map, compute_precision_recall
```

### 2. Compute ST-IoU
```python
# Single video
gt_detections = {
    0: [10, 10, 50, 50],  # Frame 0: [x1, y1, x2, y2]
    1: [15, 15, 55, 55],  # Frame 1
    2: [20, 20, 60, 60],  # Frame 2
}

pred_detections = {
    1: [12, 12, 52, 52],
    2: [18, 18, 58, 58],
    3: [25, 25, 65, 65],
}

st_iou = compute_st_iou(gt_detections, pred_detections)
print(f"ST-IoU: {st_iou:.4f}")
```

### 3. Compute Detection Metrics
```python
# Precision, Recall, F1
metrics = compute_precision_recall(
    pred_boxes,    # (N, 4) numpy array
    pred_scores,   # (N,) numpy array
    pred_classes,  # (N,) numpy array
    gt_boxes,      # (M, 4) numpy array
    gt_classes,    # (M,) numpy array
    iou_threshold=0.5
)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")

# mAP@0.5 and mAP@0.75
map_50, ap_per_class = compute_map(pred_boxes, pred_scores, pred_classes,
                                   gt_boxes, gt_classes, iou_threshold=0.5)
print(f"mAP@0.5: {map_50:.4f}")
```

---

## Training with ST-IoU

### Start Training
```bash
# Stage 2: Meta-learning with ST-IoU tracking
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb

# With triplet loss (recommended)
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
    --use_triplet --triplet_ratio 0.3 --use_wandb
```

### Console Output
```
Epoch 15/100 | Train Loss: 2.34
Validation:
  Total Loss: 2.12
  ST-IoU: 0.6753          ← Primary metric
  mAP@0.5: 0.7234
  mAP@0.75: 0.5821
  Precision: 0.8521
  Recall: 0.7865
  F1: 0.8179
  ✓ New best model! (ST-IoU: 0.6753)
```

### WandB Metrics
Navigate to your WandB dashboard to see:
- `val/st_iou` - Current ST-IoU
- `val/best_st_iou` - Best ST-IoU so far
- `val/map_50` - mAP at IoU 0.5
- `val/map_75` - mAP at IoU 0.75
- `val/precision` - Detection precision
- `val/recall` - Detection recall
- `val/f1` - F1 score

---

## Checkpoints

### Best Model Selection
Models are saved based on **ST-IoU** (primary) or **loss** (secondary):

```python
# Saved when ST-IoU improves
if current_st_iou > best_st_iou:
    save_checkpoint('best_model.pt', is_best_st_iou=True)
    
# Or when ST-IoU tied but loss improves
elif current_st_iou == best_st_iou and current_loss < best_loss:
    save_checkpoint('best_model.pt', is_best=True)
```

### Checkpoint Contents
```python
checkpoint = {
    'epoch': 50,
    'global_step': 5000,
    'best_val_loss': 2.12,
    'best_st_iou': 0.6753,      ← Tracked since last session
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
}
```

### Load Checkpoint
```python
import torch

ckpt = torch.load('./checkpoints/best_model.pt')
print(f"Best ST-IoU: {ckpt['best_st_iou']:.4f}")
print(f"Best Loss: {ckpt['best_val_loss']:.4f}")
print(f"Epoch: {ckpt['epoch']}")
```

---

## API Reference

### `compute_spatial_iou(box1, box2)`
Compute 2D IoU between bounding boxes.

**Args:**
- `box1`: `[x1, y1, x2, y2]` (numpy array or torch tensor)
- `box2`: `[x1, y1, x2, y2]` (numpy array or torch tensor)

**Returns:**
- `iou`: Float between 0 and 1

**Example:**
```python
from src.metrics.st_iou import compute_spatial_iou
import numpy as np

box1 = np.array([10, 10, 50, 50])
box2 = np.array([12, 12, 52, 52])
iou = compute_spatial_iou(box1, box2)
print(f"IoU: {iou:.4f}")  # 0.8223
```

---

### `compute_st_iou(gt_detections, pred_detections, video_length=None)`
Compute ST-IoU for a single video.

**Args:**
- `gt_detections`: `Dict[int, np.ndarray]` - Frame ID → bbox `[x1, y1, x2, y2]`
- `pred_detections`: `Dict[int, np.ndarray]` - Frame ID → bbox `[x1, y1, x2, y2]`
- `video_length`: Optional int (for normalization, not used in current implementation)

**Returns:**
- `st_iou`: Float between 0 and 1

**Example:**
```python
from src.metrics.st_iou import compute_st_iou

gt = {0: [10, 10, 50, 50], 1: [15, 15, 55, 55]}
pred = {0: [12, 12, 52, 52], 1: [17, 17, 57, 57]}
st_iou = compute_st_iou(gt, pred)
print(f"ST-IoU: {st_iou:.4f}")
```

---

### `compute_st_iou_batch(gt_batch, pred_batch, video_lengths=None)`
Compute ST-IoU for multiple videos.

**Args:**
- `gt_batch`: `List[Dict[int, np.ndarray]]` - List of GT detection dicts
- `pred_batch`: `List[Dict[int, np.ndarray]]` - List of pred detection dicts
- `video_lengths`: Optional `List[int]`

**Returns:**
- `mean_st_iou`: Float - Mean ST-IoU across videos
- `st_iou_per_video`: `List[float]` - Individual ST-IoU scores

**Example:**
```python
from src.metrics.st_iou import compute_st_iou_batch

gt_batch = [
    {0: [10, 10, 50, 50], 1: [15, 15, 55, 55]},
    {0: [20, 20, 60, 60], 1: [25, 25, 65, 65]},
]
pred_batch = [
    {0: [12, 12, 52, 52], 1: [17, 17, 57, 57]},
    {0: [22, 22, 62, 62], 1: [27, 27, 67, 67]},
]

mean_st_iou, per_video = compute_st_iou_batch(gt_batch, pred_batch)
print(f"Mean ST-IoU: {mean_st_iou:.4f}")
print(f"Per-video: {[f'{x:.4f}' for x in per_video]}")
```

---

### `compute_precision_recall(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5)`
Compute precision, recall, and F1 score.

**Args:**
- `pred_boxes`: `(N, 4)` numpy array of predicted boxes
- `pred_scores`: `(N,)` numpy array of confidence scores
- `pred_classes`: `(N,)` numpy array of predicted class IDs
- `gt_boxes`: `(M, 4)` numpy array of ground truth boxes
- `gt_classes`: `(M,)` numpy array of ground truth class IDs
- `iou_threshold`: Float (default 0.5)

**Returns:**
- Dict with keys: `'precision'`, `'recall'`, `'f1'`

**Example:**
```python
from src.metrics.detection_metrics import compute_precision_recall
import numpy as np

pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
pred_scores = np.array([0.9, 0.8])
pred_classes = np.array([0, 1])

gt_boxes = np.array([[12, 12, 52, 52], [62, 62, 102, 102]])
gt_classes = np.array([0, 1])

metrics = compute_precision_recall(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes, iou_threshold=0.5
)
print(metrics)  # {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
```

---

### `compute_map(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5)`
Compute mean Average Precision (mAP).

**Args:**
- Same as `compute_precision_recall`

**Returns:**
- `map_score`: Float - Mean AP across all classes
- `ap_per_class`: `Dict[int, float]` - AP for each class

**Example:**
```python
from src.metrics.detection_metrics import compute_map

map_50, ap_per_class = compute_map(
    pred_boxes, pred_scores, pred_classes,
    gt_boxes, gt_classes, iou_threshold=0.5
)
print(f"mAP@0.5: {map_50:.4f}")
print(f"AP per class: {ap_per_class}")
```

---

## Troubleshooting

### ST-IoU is always 0.0
**Cause:** No predictions above confidence threshold  
**Fix:** Lower threshold in validation
```python
# In trainer.py, modify validation score threshold
sample_pred_scores >= 0.1  # Instead of 0.25
```

### ST-IoU not improving
**Cause:** Model not learning to localize correctly  
**Fix:**
1. Check if bbox loss is decreasing
2. Increase DFL loss weight: `dfl_weight=1.5`
3. Verify ground truth annotations

### mAP high but ST-IoU low
**Cause:** Temporal consistency issues  
**Explanation:** mAP ignores time dimension, ST-IoU penalizes missing frames
**Fix:** Focus on improving detection consistency across frames

### WandB not showing ST-IoU
**Cause:** WandB not enabled  
**Fix:**
```bash
pip install wandb
python train.py --use_wandb
```

---

## Testing

### Run Unit Tests
```bash
# All ST-IoU tests
pytest src/tests/test_st_iou.py -v

# Single test
pytest src/tests/test_st_iou.py::test_st_iou_single_video -v

# With output
pytest src/tests/test_st_iou.py -v -s
```

### Verify Integration
```bash
# Run verification script
python verify_st_iou_integration.py
```

**Expected Output:**
```
✅ PASS - Metrics Imports
✅ PASS - ST-IoU Computation
✅ PASS - Detection Metrics
✅ PASS - Trainer Integration
✅ PASS - Checkpoint Compatibility
✅ PASS - Unit Tests
```

---

## Files Modified

### New Files
- `src/metrics/st_iou.py` (310 lines)
- `src/metrics/detection_metrics.py` (257 lines)
- `src/metrics/__init__.py` (module exports)
- `src/tests/test_st_iou.py` (191 lines)
- `verify_st_iou_integration.py` (verification script)
- `ST_IOU_INTEGRATION_SUMMARY.md` (this summary)

### Modified Files
- `src/training/trainer.py` (~100 lines changed)
  - Added `best_st_iou` tracking (line 97)
  - Enhanced `validate()` with metrics (lines 242-419)
  - Updated training loop (lines 657-777)
  - Modified checkpoint save/load (lines 588-651)

---

## Next Steps

1. ✅ **Verification Complete** - All tests passing
2. ⏳ **Integration Test** - Run 2-epoch training to verify end-to-end
3. ⏳ **WandB Check** - Verify metrics appear in dashboard
4. ⏳ **Full Training** - Train full model and monitor ST-IoU

### Integration Test Command
```bash
python train.py --stage 2 --epochs 2 --n_way 2 --n_query 4 --use_wandb
```

---

## Support

**Documentation:**
- Full summary: `ST_IOU_INTEGRATION_SUMMARY.md`
- Training guide: `TRAINING_GUIDE.md`
- Agent guidelines: `AGENTS.md`

**Test Files:**
- Unit tests: `src/tests/test_st_iou.py`
- Verification: `verify_st_iou_integration.py`

**Code Locations:**
- ST-IoU: `src/metrics/st_iou.py:56-111`
- Validation: `src/training/trainer.py:242-419`
- Best model selection: `src/training/trainer.py:697-727`

---

*Last Updated: 2025-11-13*
