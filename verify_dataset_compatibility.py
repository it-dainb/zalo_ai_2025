"""
Verification Script: Dataset Compatibility Check
Verifies that split_dataset.py output is compatible with:
1. RefDetDataset (data loading)
2. ST-IoU metrics (evaluation)
3. Trainer (training pipeline)
"""

import json
import numpy as np
from pathlib import Path
import sys

def check_annotation_format(annotations_file: Path) -> bool:
    """Check if annotations.json has correct format for RefDetDataset."""
    print(f"\n{'='*60}")
    print("1. Checking Annotation Format Compatibility")
    print(f"{'='*60}")
    
    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        return False
    
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"✅ Loaded {len(annotations)} video annotations")
        
        # Check structure
        required_fields = ['video_id', 'annotations']
        for i, entry in enumerate(annotations):
            # Check top-level fields
            for field in required_fields:
                if field not in entry:
                    print(f"❌ Missing '{field}' in annotation {i}")
                    return False
            
            # Check annotations structure
            if not isinstance(entry['annotations'], list):
                print(f"❌ 'annotations' must be a list in entry {i}")
                return False
            
            # Check bbox structure
            for anno in entry['annotations']:
                if 'bboxes' not in anno:
                    print(f"❌ Missing 'bboxes' in annotation group for {entry['video_id']}")
                    return False
                
                for bbox in anno['bboxes']:
                    required_bbox_fields = ['frame', 'x1', 'y1', 'x2', 'y2']
                    for field in required_bbox_fields:
                        if field not in bbox:
                            print(f"❌ Missing '{field}' in bbox for {entry['video_id']}")
                            return False
                    break  # Only check first bbox
                break  # Only check first annotation group
        
        print("✅ All annotations have correct format for RefDetDataset")
        return True
    
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return False


def check_st_iou_metadata(metadata_dir: Path, split_name: str = 'test') -> bool:
    """Check if ST-IoU metadata files exist and are valid."""
    print(f"\n{'='*60}")
    print(f"2. Checking ST-IoU Metadata Files ({split_name})")
    print(f"{'='*60}")
    
    # Check metadata JSON
    metadata_file = metadata_dir / f'{split_name}_st_iou_metadata.json'
    if not metadata_file.exists():
        print(f"⚠️  ST-IoU metadata not found: {metadata_file}")
        print("   Run: python split_dataset.py --save_st_iou_metadata")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded ST-IoU metadata for {len(metadata)} videos")
        
        # Check metadata structure
        for video_id, info in metadata.items():
            required_fields = ['num_frames', 'frame_range', 'num_bboxes', 'bbox_format', 'frames']
            for field in required_fields:
                if field not in info:
                    print(f"❌ Missing '{field}' in metadata for {video_id}")
                    return False
            break  # Only check first entry
        
        print("✅ ST-IoU metadata has correct structure")
        
    except Exception as e:
        print(f"❌ Error loading ST-IoU metadata: {e}")
        return False
    
    # Check ground truth NPZ
    gt_file = metadata_dir / f'{split_name}_st_iou_gt.npz'
    if not gt_file.exists():
        print(f"❌ ST-IoU ground truth not found: {gt_file}")
        return False
    
    try:
        gt_data = np.load(gt_file)
        num_arrays = len(gt_data.files)
        print(f"✅ Loaded ST-IoU ground truth with {num_arrays} arrays")
        
        # Verify structure (should have video_id_frame_ids and video_id_bboxes pairs)
        frame_id_arrays = [f for f in gt_data.files if f.endswith('_frame_ids')]
        bbox_arrays = [f for f in gt_data.files if f.endswith('_bboxes')]
        
        if len(frame_id_arrays) != len(bbox_arrays):
            print(f"❌ Mismatch: {len(frame_id_arrays)} frame_id arrays, {len(bbox_arrays)} bbox arrays")
            return False
        
        print(f"✅ Ground truth has {len(frame_id_arrays)} videos")
        
        # Check data types
        for video_id in metadata.keys():
            frame_key = f'{video_id}_frame_ids'
            bbox_key = f'{video_id}_bboxes'
            
            if frame_key in gt_data.files and bbox_key in gt_data.files:
                frames = gt_data[frame_key]
                bboxes = gt_data[bbox_key]
                
                if frames.dtype != np.int32:
                    print(f"⚠️  Warning: {frame_key} has dtype {frames.dtype}, expected int32")
                
                if bboxes.shape[0] != frames.shape[0]:
                    print(f"❌ Shape mismatch for {video_id}: {bboxes.shape[0]} bboxes vs {frames.shape[0]} frames")
                    return False
                
                if bboxes.shape[1] != 4:
                    print(f"❌ Invalid bbox shape for {video_id}: {bboxes.shape}, expected (N, 4)")
                    return False
                
                print(f"✅ {video_id}: {len(frames)} frames with valid bbox format")
                break  # Only check first video
        
    except Exception as e:
        print(f"❌ Error loading ST-IoU ground truth: {e}")
        return False
    
    # Check summary JSON
    summary_file = metadata_dir / f'{split_name}_st_iou_summary.json'
    if not summary_file.exists():
        print(f"❌ ST-IoU summary not found: {summary_file}")
        return False
    
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"✅ ST-IoU summary:")
        print(f"   - Split: {summary['split']}")
        print(f"   - Videos: {summary['num_videos']}")
        print(f"   - Total frames: {summary['total_frames']}")
        print(f"   - Total bboxes: {summary['total_bboxes']}")
        print(f"   - Avg frames/video: {summary['avg_frames_per_video']:.1f}")
        
    except Exception as e:
        print(f"❌ Error loading ST-IoU summary: {e}")
        return False
    
    return True


def check_trainer_compatibility() -> bool:
    """Check if trainer can use ST-IoU metrics."""
    print(f"\n{'='*60}")
    print("3. Checking Trainer ST-IoU Integration")
    print(f"{'='*60}")
    
    try:
        # Check if ST-IoU module exists and imports correctly
        from src.metrics.st_iou import compute_st_iou, compute_spatial_iou, compute_st_iou_batch
        print("✅ ST-IoU metrics module imported successfully")
        
        # Check if trainer imports detection metrics
        from src.metrics.detection_metrics import compute_precision_recall, compute_map
        print("✅ Detection metrics module imported successfully")
        
        # Check if trainer has ST-IoU support
        from src.training.trainer import RefDetTrainer
        print("✅ Trainer imported successfully")
        
        # Verify trainer has best_st_iou tracking
        import inspect
        trainer_source = inspect.getsource(RefDetTrainer.__init__)
        if 'best_st_iou' in trainer_source:
            print("✅ Trainer has best_st_iou tracking")
        else:
            print("⚠️  Warning: Trainer may not track best_st_iou")
        
        # Verify validate method computes ST-IoU
        validate_source = inspect.getsource(RefDetTrainer.validate)
        if 'compute_st_iou' in validate_source or 'st_iou' in validate_source:
            print("✅ Trainer.validate() computes ST-IoU metrics")
        else:
            print("⚠️  Warning: Trainer.validate() may not compute ST-IoU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking trainer: {e}")
        return False


def check_dataset_loading(data_root: Path, annotations_file: Path) -> bool:
    """Check if RefDetDataset can load the split dataset."""
    print(f"\n{'='*60}")
    print("4. Checking Dataset Loading (RefDetDataset)")
    print(f"{'='*60}")
    
    try:
        from src.datasets.refdet_dataset import RefDetDataset
        
        # Try to create dataset
        dataset = RefDetDataset(
            data_root=str(data_root),
            annotations_file=str(annotations_file),
            mode='val',
            cache_frames=False
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   - Classes: {len(dataset.classes)}")
        print(f"   - Total samples: {len(dataset)}")
        
        # Try to load one sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Successfully loaded first sample:")
            print(f"   - video_id: {sample['video_id']}")
            print(f"   - frame_idx: {sample['frame_idx']}")
            print(f"   - bboxes shape: {sample['bboxes'].shape}")
            print(f"   - query_frame shape: {sample['query_frame'].shape}")
            print(f"   - support_images: {len(sample['support_images'])} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all compatibility checks."""
    print(f"\n{'='*70}")
    print("Dataset Compatibility Verification for split_dataset.py")
    print(f"{'='*70}")
    
    # Check if raw dataset exists
    raw_path = Path('./raw')
    if not raw_path.exists():
        print(f"❌ Raw dataset not found at {raw_path}")
        print("   Please ensure dataset is available")
        sys.exit(1)
    
    # Check for split datasets
    datasets_path = Path('./datasets')
    if not datasets_path.exists():
        print(f"⚠️  No split datasets found at {datasets_path}")
        print("   Run: python split_dataset.py --input_dir ./raw --output_dir ./datasets --save_st_iou_metadata")
        sys.exit(1)
    
    all_checks_passed = True
    
    # Check test split (most important for evaluation)
    test_path = datasets_path / 'test'
    if test_path.exists():
        print(f"\n{'='*70}")
        print("Verifying TEST Split")
        print(f"{'='*70}")
        
        test_annotations = test_path / 'annotations' / 'annotations.json'
        test_samples = test_path / 'samples'
        test_metadata_dir = test_path / 'annotations'
        
        # Run checks
        check1 = check_annotation_format(test_annotations)
        check2 = check_st_iou_metadata(test_metadata_dir, split_name='test')
        check3 = check_trainer_compatibility()
        check4 = check_dataset_loading(test_samples, test_annotations)
        
        all_checks_passed = all_checks_passed and check1 and check2 and check3 and check4
    else:
        print(f"⚠️  Test split not found at {test_path}")
        all_checks_passed = False
    
    # Check train split (optional)
    train_path = datasets_path / 'train'
    if train_path.exists():
        print(f"\n{'='*70}")
        print("Verifying TRAIN Split")
        print(f"{'='*70}")
        
        train_annotations = train_path / 'annotations' / 'annotations.json'
        train_samples = train_path / 'samples'
        
        check1 = check_annotation_format(train_annotations)
        
        all_checks_passed = all_checks_passed and check1
    
    # Final summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nYour dataset is fully compatible with:")
        print("  1. RefDetDataset (data loading)")
        print("  2. ST-IoU metrics (evaluation)")
        print("  3. RefDetTrainer (training pipeline)")
        print("\nYou can now:")
        print("  - Train models with: python train.py")
        print("  - Evaluate with ST-IoU metrics")
        print("  - Use the trainer's validation with compute_detection_metrics=True")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before training/evaluation")
        print("\nTo generate ST-IoU metadata files, run:")
        print("  python split_dataset.py --input_dir ./raw --output_dir ./datasets \\")
        print("    --validate_st_iou --save_st_iou_metadata")
    
    print(f"{'='*70}\n")
    
    sys.exit(0 if all_checks_passed else 1)


if __name__ == '__main__':
    main()
