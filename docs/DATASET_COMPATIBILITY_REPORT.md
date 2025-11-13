# Dataset Compatibility Report: split_dataset.py + Trainer + ST-IoU

**Date:** 2025-11-13  
**Status:** ✅ FULLY COMPATIBLE  
**Session:** Resumed from ST-IoU integration

---

## Executive Summary

The `split_dataset.py` script has been successfully enhanced with ST-IoU evaluation support and verified for full compatibility with:

1. **RefDetDataset** - Data loading pipeline
2. **ST-IoU Metrics** - Spatio-temporal evaluation
3. **RefDetTrainer** - Training pipeline with validation

All verification tests pass ✅

---

## What Was Updated

### 1. Enhanced `split_dataset.py`

**New Functions Added:**

```python
extract_st_iou_metadata(annotations)          # Lines 108-157
validate_st_iou_compatibility(annotations)    # Lines 160-248
convert_annotations_to_st_iou_format(annotations)  # Lines 251-304
save_st_iou_metadata(metadata, st_iou_gt, ...)    # Lines 307-382
```

**New Command-Line Flags:**

```bash
--validate_st_iou       # Validate ST-IoU compatibility before splitting
--save_st_iou_metadata  # Generate ST-IoU evaluation files
```

**Example Usage:**

```bash
# Split with ST-IoU validation and metadata generation
python split_dataset.py \
  --input_dir ./raw \
  --output_dir ./datasets \
  --validate_st_iou \
  --save_st_iou_metadata

# Dry run with validation only
python split_dataset.py \
  --input_dir ./raw \
  --output_dir ./datasets \
  --dry_run \
  --validate_st_iou
```

---

## Output Files Generated

When `--save_st_iou_metadata` is used, three additional files are created per split:

### Test Split Example

```
datasets/test/annotations/
├── annotations.json              # Original annotations (for training/loading)
├── test_st_iou_metadata.json     # Video metadata for ST-IoU evaluation
├── test_st_iou_gt.npz           # Ground truth bboxes (numpy format, fast loading)
└── test_st_iou_summary.json     # Evaluation summary statistics
```

### File Formats

#### 1. `test_st_iou_metadata.json`

```json
{
  "Backpack_1": {
    "num_frames": 1454,
    "frame_range": [2524, 4316],
    "num_bboxes": 1456,
    "bbox_format": "x1_y1_x2_y2",
    "frames": [2524, 2525, ..., 4316]
  },
  "Person1_0": {
    ...
  }
}
```

#### 2. `test_st_iou_gt.npz` (NumPy Compressed)

```python
# Load with: np.load('test_st_iou_gt.npz')
{
  'Backpack_1_frame_ids': np.array([2524, 2525, ...], dtype=np.int32),
  'Backpack_1_bboxes': np.array([[x1,y1,x2,y2], ...], dtype=np.float32),
  'Person1_0_frame_ids': np.array([...], dtype=np.int32),
  'Person1_0_bboxes': np.array([[...], ...], dtype=np.float32),
}
```

#### 3. `test_st_iou_summary.json`

```json
{
  "split": "test",
  "num_videos": 2,
  "total_frames": 3511,
  "total_bboxes": 3520,
  "avg_frames_per_video": 1755.5,
  "video_ids": ["Backpack_1", "Person1_0"]
}
```

---

## Compatibility Verification Results

### ✅ Test 1: Annotation Format Compatibility

**Status:** PASSED  
**Details:**
- Loaded 2 test videos, 12 train videos
- All annotations have required fields: `video_id`, `annotations`, `bboxes`
- All bboxes have required fields: `frame`, `x1`, `y1`, `x2`, `y2`
- Format matches RefDetDataset expectations exactly

### ✅ Test 2: ST-IoU Metadata Files

**Status:** PASSED  
**Details:**
- Metadata JSON has correct structure
- Ground truth NPZ has frame_ids and bboxes arrays
- Data types: frame_ids (int32), bboxes (float32, shape Nx4)
- Summary statistics are accurate

**Test Dataset Statistics:**
- Videos: 2 (Backpack_1, Person1_0)
- Total frames: 3,511
- Total bboxes: 3,520
- Avg frames/video: 1,755.5

### ✅ Test 3: Trainer ST-IoU Integration

**Status:** PASSED  
**Details:**
- ST-IoU metrics module imports successfully
- Detection metrics module imports successfully
- Trainer has `best_st_iou` tracking (line 97)
- `Trainer.validate()` computes ST-IoU metrics (lines 242-419)
- Best model selection prioritizes ST-IoU over loss (lines 697-727)

### ✅ Test 4: Dataset Loading (RefDetDataset)

**Status:** PASSED  
**Details:**
- Dataset created successfully
- Classes: 2 test, 12 train
- Total samples: 3,511 annotated frames
- Sample loaded successfully:
  - video_id: Backpack_1
  - frame_idx: 2524
  - bboxes: (1, 4)
  - query_frame: (576, 1024, 3)
  - support_images: 3 images

---

## Annotation Structure (Confirmed Compatible)

### Input/Output Format (annotations.json)

```json
[
  {
    "video_id": "Backpack_1",
    "annotations": [
      {
        "bboxes": [
          {
            "frame": 2524,
            "x1": 321,
            "y1": 0,
            "x2": 381,
            "y2": 12
          },
          ...
        ]
      }
    ]
  }
]
```

**Key Compatibility Points:**

1. ✅ `video_id` field matches RefDetDataset expectations
2. ✅ `annotations` list structure matches dataset parser
3. ✅ `bboxes` have `frame` field for ST-IoU temporal tracking
4. ✅ Bbox format `[x1, y1, x2, y2]` matches ST-IoU requirements
5. ✅ Frame IDs are integers for temporal ordering

---

## How The Trainer Uses ST-IoU Metadata

### During Training Validation

The trainer's `validate()` method (src/training/trainer.py:242-419):

```python
def validate(self, val_loader, compute_detection_metrics=True):
    # 1. Compute loss on validation set
    # 2. If compute_detection_metrics=True:
    #    a. Run model inference
    #    b. Compute spatial IoU (ST-IoU for single frames)
    #    c. Compute mAP, precision, recall
    #    d. Track best ST-IoU score
    # 3. Save checkpoint if ST-IoU improves
```

### Model Selection Strategy

```python
# From trainer.py:697-727
if st_iou > self.best_st_iou:
    # Save best model based on ST-IoU (not loss)
    self.best_st_iou = st_iou
    self.save_checkpoint(is_best=True)
```

### WandB Logging

```python
# ST-IoU metrics are automatically logged to WandB:
wandb.log({
    'val/st_iou': st_iou,
    'val/map_50': map_50,
    'val/precision': precision,
    'val/recall': recall,
    ...
})
```

---

## How To Use ST-IoU Evaluation Files

### Loading Ground Truth

```python
import numpy as np
import json

# Load metadata
with open('datasets/test/annotations/test_st_iou_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load ground truth bboxes
gt_data = np.load('datasets/test/annotations/test_st_iou_gt.npz')

# Access specific video
video_id = 'Backpack_1'
frame_ids = gt_data[f'{video_id}_frame_ids']  # (N,) int32
bboxes = gt_data[f'{video_id}_bboxes']        # (N, 4) float32

# Each bbox corresponds to a frame
for frame_id, bbox in zip(frame_ids, bboxes):
    x1, y1, x2, y2 = bbox
    print(f"Frame {frame_id}: [{x1}, {y1}, {x2}, {y2}]")
```

### Computing ST-IoU on Predictions

```python
from src.metrics.st_iou import compute_st_iou
import numpy as np

# Load ground truth for a video
gt_data = np.load('datasets/test/annotations/test_st_iou_gt.npz')
video_id = 'Backpack_1'
gt_frame_ids = gt_data[f'{video_id}_frame_ids']
gt_bboxes = gt_data[f'{video_id}_bboxes']

# Create ground truth dict: {frame_id: bbox}
gt_dict = {int(fid): bbox for fid, bbox in zip(gt_frame_ids, gt_bboxes)}

# Your model predictions (same format)
pred_dict = {
    2524: np.array([320, 1, 380, 13]),
    2525: np.array([301, 1, 386, 22]),
    # ...
}

# Compute ST-IoU
st_iou_score = compute_st_iou(
    gt_bboxes=gt_dict,
    pred_bboxes=pred_dict,
    tau=0.5  # Temporal smoothness threshold
)

print(f"ST-IoU: {st_iou_score:.4f}")
```

---

## Training Pipeline Integration

### Current Dataset Flow

```
Raw Dataset
    ↓
split_dataset.py (with --save_st_iou_metadata)
    ↓
├── train/
│   ├── annotations/annotations.json  ← RefDetDataset loads this
│   ├── annotations/train_st_iou_*.{json,npz}  ← For evaluation
│   └── samples/{video_id}/
└── test/
    ├── annotations/annotations.json  ← RefDetDataset loads this
    ├── annotations/test_st_iou_*.{json,npz}  ← For evaluation
    └── samples/{video_id}/
```

### Training Command

```bash
# Stage 2: Meta-learning with ST-IoU tracking
python train.py \
  --stage 2 \
  --train_data_root ./datasets/train/samples \
  --train_annotations ./datasets/train/annotations/annotations.json \
  --val_data_root ./datasets/test/samples \
  --val_annotations ./datasets/test/annotations/annotations.json \
  --epochs 100 \
  --use_wandb  # ST-IoU metrics logged automatically
```

### What Happens During Training

1. **Data Loading:** RefDetDataset loads `annotations.json` ✅
2. **Training:** Model trains on episodic batches ✅
3. **Validation:** 
   - Computes loss ✅
   - Computes ST-IoU metrics (if `compute_detection_metrics=True`) ✅
   - Logs to WandB ✅
4. **Checkpointing:** Saves best model based on ST-IoU ✅

---

## Evaluation Pipeline Integration

### Standalone Evaluation

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_data_root ./datasets/test/samples \
  --test_annotations ./datasets/test/annotations/annotations.json \
  --output_dir ./results
```

**Note:** The evaluate.py script can be enhanced to use the precomputed ST-IoU ground truth files for faster evaluation:

```python
# Instead of parsing annotations.json every time:
gt_data = np.load('datasets/test/annotations/test_st_iou_gt.npz')
# Use gt_data directly for ST-IoU computation
```

---

## Verification Tools

### 1. Dataset Compatibility Check

```bash
python verify_dataset_compatibility.py
```

**What it checks:**
- Annotation format compatibility with RefDetDataset
- ST-IoU metadata files exist and are valid
- Trainer ST-IoU integration is working
- Dataset can be loaded successfully

**Output:** Comprehensive report with ✅/❌ for each check

### 2. ST-IoU Integration Verification

```bash
python verify_st_iou_integration.py
```

**What it checks:**
- ST-IoU metrics module functionality
- Detection metrics module functionality
- Trainer integration
- End-to-end metric computation

---

## Bug Fixes Applied

### Issue: Variable Name Conflict

**Problem:** Parameter `save_st_iou_metadata` had same name as function `save_st_iou_metadata()`, causing:
```
TypeError: 'bool' object is not callable
```

**Solution:** Renamed parameter to `generate_st_iou_files` in `split_dataset()` function

**Files Modified:**
- `split_dataset.py:391` - Function signature
- `split_dataset.py:437` - Docstring
- `split_dataset.py:510` - Function body
- `split_dataset.py:655` - Function call from main()

---

## Key Takeaways

### ✅ What Works

1. **Dataset Splitting:** Creates train/test splits with 9:1 ratio by video class
2. **ST-IoU Metadata:** Generates evaluation-ready ground truth files
3. **Data Loading:** RefDetDataset loads annotations correctly
4. **Training:** Trainer computes ST-IoU during validation
5. **Model Selection:** Best model saved based on ST-IoU performance
6. **Logging:** All metrics logged to WandB automatically

### ⚠️ Future Enhancements (Optional)

1. **Multi-frame ST-IoU:** Current implementation treats each frame independently. For true temporal ST-IoU, enhance validation to track predictions across consecutive frames.

2. **Faster Evaluation:** Modify `evaluate.py` to load precomputed ground truth from `.npz` files instead of parsing JSON every time.

3. **Video-level Metrics:** Add video-level ST-IoU aggregation in addition to frame-level metrics.

4. **Cross-validation:** Add k-fold cross-validation support using ST-IoU metadata.

---

## Testing Performed

### Test Dataset
- **Source:** `./raw/annotations/annotations.json`
- **Videos:** 14 classes total (12 train, 2 test)
- **Test Classes:** Backpack_1, Person1_0
- **Test Frames:** 3,511 annotated frames
- **Test Bboxes:** 3,520 bounding boxes

### Verification Results
```
✅ Annotation format: PASSED
✅ ST-IoU metadata: PASSED  
✅ Trainer integration: PASSED
✅ Dataset loading: PASSED
```

### Files Generated Successfully
```
datasets/test/annotations/
├── test_st_iou_metadata.json  (42 KB)
├── test_st_iou_gt.npz        (29 KB compressed)
└── test_st_iou_summary.json   (178 B)
```

---

## Conclusion

The `split_dataset.py` script has been successfully enhanced with ST-IoU evaluation support. All components of the training pipeline (dataset loading, trainer, metrics) are fully compatible with the generated dataset format.

**Ready for:**
- ✅ Training with ST-IoU tracking
- ✅ Validation with detection metrics
- ✅ Evaluation with spatio-temporal IoU
- ✅ Model selection based on ST-IoU performance

**No breaking changes** - Existing code continues to work without ST-IoU metadata files. ST-IoU features are opt-in via command-line flags.
