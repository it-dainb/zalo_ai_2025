"""
Dataset Splitting Script for Reference-Based Detection Dataset
Splits dataset into train/test sets with 9:1 ratio.

Usage:
    python split_dataset.py --input_dir <input_path> --output_dir <output_path> [--train_ratio 0.9] [--seed 42]
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np


def load_annotations(annotations_file: Path) -> List[Dict]:
    """Load annotations from JSON file."""
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def split_annotations(
    annotations: List[Dict], 
    train_ratio: float = 0.9, 
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split annotations into train and test sets.
    
    Args:
        annotations: List of annotation dictionaries
        train_ratio: Ratio of data to use for training (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_annotations, test_annotations)
    """
    random.seed(seed)
    
    # Get unique video IDs (classes)
    video_ids = [item['video_id'] for item in annotations]
    print(f"Found {len(video_ids)} video classes")
    
    # Shuffle video IDs
    shuffled_ids = video_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split point
    split_idx = int(len(shuffled_ids) * train_ratio)
    
    train_ids = set(shuffled_ids[:split_idx])
    test_ids = set(shuffled_ids[split_idx:])
    
    print(f"Train classes: {len(train_ids)}")
    print(f"Test classes: {len(test_ids)}")
    
    # Split annotations
    train_annotations = [item for item in annotations if item['video_id'] in train_ids]
    test_annotations = [item for item in annotations if item['video_id'] in test_ids]
    
    return train_annotations, test_annotations


def copy_samples(
    video_ids: List[str],
    input_samples_dir: Path,
    output_samples_dir: Path
) -> None:
    """
    Copy sample directories for given video IDs.
    
    Args:
        video_ids: List of video IDs to copy
        input_samples_dir: Source samples directory
        output_samples_dir: Destination samples directory
    """
    output_samples_dir.mkdir(parents=True, exist_ok=True)
    
    for video_id in video_ids:
        src_dir = input_samples_dir / video_id
        dst_dir = output_samples_dir / video_id
        
        if not src_dir.exists():
            print(f"Warning: Source directory not found: {src_dir}")
            continue
            
        if dst_dir.exists():
            print(f"Skipping existing directory: {dst_dir}")
            continue
        
        print(f"Copying {video_id}...")
        shutil.copytree(src_dir, dst_dir)


def save_annotations(annotations: List[Dict], output_file: Path) -> None:
    """Save annotations to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(annotations)} annotations to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)


def extract_st_iou_metadata(annotations: List[Dict]) -> Dict:
    """
    Extract metadata for ST-IoU evaluation from annotations.
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        metadata: Dict with ST-IoU evaluation info
            - video_id -> {
                'num_frames': int,
                'frame_range': (min_frame, max_frame),
                'num_bboxes': int,
                'bbox_format': str,
              }
    """
    metadata = {}
    
    for video_item in annotations:
        video_id = video_item['video_id']
        video_annotations = video_item.get('annotations', [])
        
        if not video_annotations:
            print(f"Warning: No annotations found for {video_id}")
            continue
        
        # Extract frame-level bounding boxes
        all_frames = []
        total_bboxes = 0
        
        for ann_group in video_annotations:
            bboxes = ann_group.get('bboxes', [])
            for bbox in bboxes:
                frame_id = bbox.get('frame', None)
                if frame_id is not None:
                    all_frames.append(frame_id)
                    total_bboxes += 1
        
        if all_frames:
            metadata[video_id] = {
                'num_frames': len(set(all_frames)),
                'frame_range': (min(all_frames), max(all_frames)),
                'num_bboxes': total_bboxes,
                'bbox_format': 'x1_y1_x2_y2',
                'frames': sorted(set(all_frames)),
            }
        else:
            print(f"Warning: No valid frames found for {video_id}")
    
    return metadata


def validate_st_iou_compatibility(annotations: List[Dict], verbose: bool = True) -> bool:
    """
    Validate that annotations are compatible with ST-IoU evaluation.
    
    Checks:
    1. Each video has frame-level annotations
    2. Bounding boxes are in correct format [x1, y1, x2, y2]
    3. Frame IDs are valid integers
    4. At least one bbox per video
    
    Args:
        annotations: List of annotation dictionaries
        verbose: If True, print validation details
        
    Returns:
        is_valid: True if all validations pass
    """
    is_valid = True
    issues = []
    
    for video_item in annotations:
        video_id = video_item['video_id']
        video_annotations = video_item.get('annotations', [])
        
        if not video_annotations:
            issues.append(f"{video_id}: No annotations found")
            is_valid = False
            continue
        
        # Check frame-level bboxes
        total_bboxes = 0
        frame_ids = []
        
        for ann_group in video_annotations:
            bboxes = ann_group.get('bboxes', [])
            
            for bbox in bboxes:
                # Check frame ID
                frame_id = bbox.get('frame', None)
                if frame_id is None:
                    issues.append(f"{video_id}: Missing 'frame' field in bbox")
                    is_valid = False
                    continue
                
                if not isinstance(frame_id, int):
                    issues.append(f"{video_id}: Frame ID must be integer, got {type(frame_id)}")
                    is_valid = False
                    continue
                
                frame_ids.append(frame_id)
                
                # Check bbox format [x1, y1, x2, y2]
                required_fields = ['x1', 'y1', 'x2', 'y2']
                for field in required_fields:
                    if field not in bbox:
                        issues.append(f"{video_id} frame {frame_id}: Missing '{field}' in bbox")
                        is_valid = False
                        continue
                
                # Validate bbox coordinates
                try:
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    if x2 <= x1 or y2 <= y1:
                        issues.append(f"{video_id} frame {frame_id}: Invalid bbox dimensions [{x1}, {y1}, {x2}, {y2}]")
                        is_valid = False
                except (ValueError, TypeError) as e:
                    issues.append(f"{video_id} frame {frame_id}: Invalid bbox coordinates - {e}")
                    is_valid = False
                    continue
                
                total_bboxes += 1
        
        if total_bboxes == 0:
            issues.append(f"{video_id}: No valid bounding boxes found")
            is_valid = False
    
    if verbose:
        if is_valid:
            print("✅ ST-IoU Validation PASSED")
            print(f"   All {len(annotations)} videos have valid frame-level annotations")
        else:
            print("❌ ST-IoU Validation FAILED")
            print(f"   Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"   - {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more issues")
    
    return is_valid


def convert_annotations_to_st_iou_format(annotations: List[Dict]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Convert annotations to ST-IoU evaluation format.
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        st_iou_format: Dict mapping video_id -> {frame_id -> bbox [x1, y1, x2, y2]}
        
    Example:
        {
            'Backpack_0': {
                0: np.array([100, 100, 200, 200]),
                1: np.array([105, 105, 205, 205]),
                ...
            },
            'Car_1': {
                0: np.array([50, 50, 150, 150]),
                ...
            }
        }
    """
    st_iou_format = {}
    
    for video_item in annotations:
        video_id = video_item['video_id']
        video_annotations = video_item.get('annotations', [])
        
        frame_bboxes = {}
        
        for ann_group in video_annotations:
            bboxes = ann_group.get('bboxes', [])
            
            for bbox in bboxes:
                frame_id = bbox.get('frame')
                if frame_id is None:
                    continue
                
                # Extract bbox in [x1, y1, x2, y2] format
                x1 = bbox.get('x1')
                y1 = bbox.get('y1')
                x2 = bbox.get('x2')
                y2 = bbox.get('y2')
                
                if None in [x1, y1, x2, y2]:
                    continue
                
                # Store as numpy array
                frame_bboxes[frame_id] = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        st_iou_format[video_id] = frame_bboxes
    
    return st_iou_format


def save_st_iou_metadata(
    metadata: Dict,
    st_iou_gt: Dict[str, Dict[int, np.ndarray]],
    output_dir: Path,
    split_name: str = 'test'
) -> None:
    """
    Save ST-IoU evaluation metadata and ground truth in evaluation-ready format.
    
    Args:
        metadata: Video metadata from extract_st_iou_metadata()
        st_iou_gt: Ground truth in ST-IoU format
        output_dir: Output directory
        split_name: Split name ('train' or 'test')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata as JSON
    metadata_file = output_dir / f'{split_name}_st_iou_metadata.json'
    with open(metadata_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metadata = {}
        for video_id, info in metadata.items():
            json_metadata[video_id] = {
                'num_frames': int(info['num_frames']),
                'frame_range': [int(info['frame_range'][0]), int(info['frame_range'][1])],
                'num_bboxes': int(info['num_bboxes']),
                'bbox_format': info['bbox_format'],
                'frames': [int(f) for f in info['frames']],
            }
        json.dump(json_metadata, f, indent=2)
    
    print(f"✅ Saved ST-IoU metadata to {metadata_file}")
    
    # Save ground truth in numpy format for fast loading
    gt_file = output_dir / f'{split_name}_st_iou_gt.npz'
    
    # Convert to numpy-serializable format
    np_gt = {}
    for video_id, frame_bboxes in st_iou_gt.items():
        # Store as frame_ids and bboxes arrays
        if len(frame_bboxes) > 0:
            frame_ids = sorted(frame_bboxes.keys())
            bboxes = np.array([frame_bboxes[fid] for fid in frame_ids])
            
            np_gt[f'{video_id}_frame_ids'] = np.array(frame_ids, dtype=np.int32)
            np_gt[f'{video_id}_bboxes'] = bboxes
    
    np.savez_compressed(gt_file, **np_gt)
    print(f"✅ Saved ST-IoU ground truth to {gt_file}")
    
    # Create evaluation summary
    summary = {
        'split': split_name,
        'num_videos': len(metadata),
        'total_frames': sum(info['num_frames'] for info in metadata.values()),
        'total_bboxes': sum(info['num_bboxes'] for info in metadata.values()),
        'avg_frames_per_video': np.mean([info['num_frames'] for info in metadata.values()]),
        'video_ids': list(metadata.keys()),
    }
    
    summary_file = output_dir / f'{split_name}_st_iou_summary.json'
    with open(summary_file, 'w') as f:
        # Convert numpy types
        summary_json = {k: (v.item() if isinstance(v, np.ndarray) else v) 
                       for k, v in summary.items()}
        json.dump(summary_json, f, indent=2)
    
    print(f"✅ Saved ST-IoU summary to {summary_file}")
    print(f"\n{'='*60}")
    print(f"ST-IoU Evaluation Files Created:")
    print(f"{'='*60}")
    print(f"  Metadata:     {metadata_file.name}")
    print(f"  Ground Truth: {gt_file.name}")
    print(f"  Summary:      {summary_file.name}")
    print(f"{'='*60}\n")


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
    validate_st_iou: bool = False,
    generate_st_iou_files: bool = False
) -> None:
    """
    Split dataset into train and test sets with optional ST-IoU support.
    
    Expected input structure:
        input_dir/
            annotations/
                annotations.json
            samples/
                VideoID_0/
                    video.mp4
                    object_images/
                        *.jpg
                VideoID_1/
                    ...
    
    Output structure:
        output_dir/
            train/
                annotations/
                    annotations.json
                    [train_st_iou_metadata.json]  # if save_st_iou_metadata=True
                    [train_st_iou_gt.npz]         # if save_st_iou_metadata=True
                    [train_st_iou_summary.json]   # if save_st_iou_metadata=True
                samples/
                    VideoID_0/
                    VideoID_1/
                    ...
            test/
                annotations/
                    annotations.json
                    [test_st_iou_metadata.json]   # if save_st_iou_metadata=True
                    [test_st_iou_gt.npz]          # if save_st_iou_metadata=True
                    [test_st_iou_summary.json]    # if save_st_iou_metadata=True
                samples/
                    VideoID_X/
                    VideoID_Y/
                    ...
    
    Args:
        input_dir: Path to input dataset directory
        output_dir: Path to output directory
        train_ratio: Ratio of data for training (default: 0.9 for 9:1 split)
        seed: Random seed for reproducibility
        validate_st_iou: If True, validate ST-IoU compatibility before splitting
        generate_st_iou_files: If True, save ST-IoU evaluation files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Validate input structure
    annotations_file = input_path / 'annotations' / 'annotations.json'
    samples_dir = input_path / 'samples'
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    # Load and split annotations
    annotations = load_annotations(annotations_file)
    
    # Validate ST-IoU compatibility if requested
    if validate_st_iou:
        print(f"\n{'='*60}")
        print("Validating ST-IoU Compatibility...")
        print(f"{'='*60}")
        is_valid = validate_st_iou_compatibility(annotations, verbose=True)
        if not is_valid:
            print("\n⚠️  Warning: Dataset has ST-IoU compatibility issues")
            print("   Continuing with split, but ST-IoU evaluation may fail")
        print(f"{'='*60}\n")
    
    train_annotations, test_annotations = split_annotations(
        annotations, train_ratio=train_ratio, seed=seed
    )
    
    # Extract video IDs
    train_video_ids = [item['video_id'] for item in train_annotations]
    test_video_ids = [item['video_id'] for item in test_annotations]
    
    print(f"\n{'='*60}")
    print(f"Split Summary:")
    print(f"{'='*60}")
    print(f"Total classes: {len(annotations)}")
    print(f"Train classes: {len(train_video_ids)} ({len(train_video_ids)/len(annotations)*100:.1f}%)")
    print(f"Test classes: {len(test_video_ids)} ({len(test_video_ids)/len(annotations)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Create output directories
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    
    # Save train annotations
    train_annotations_file = train_dir / 'annotations' / 'annotations.json'
    save_annotations(train_annotations, train_annotations_file)
    
    # Save test annotations
    test_annotations_file = test_dir / 'annotations' / 'annotations.json'
    save_annotations(test_annotations, test_annotations_file)
    
    # Copy train samples
    print(f"\nCopying train samples...")
    copy_samples(
        train_video_ids,
        samples_dir,
        train_dir / 'samples'
    )
    
    # Copy test samples
    print(f"\nCopying test samples...")
    copy_samples(
        test_video_ids,
        samples_dir,
        test_dir / 'samples'
    )
    
    # Generate ST-IoU metadata if requested
    if generate_st_iou_files:
        print(f"\n{'='*60}")
        print("Generating ST-IoU Evaluation Files...")
        print(f"{'='*60}\n")
        
        # Train set
        print("Processing train set...")
        train_metadata = extract_st_iou_metadata(train_annotations)
        train_st_iou_gt = convert_annotations_to_st_iou_format(train_annotations)
        save_st_iou_metadata(
            train_metadata,
            train_st_iou_gt,
            train_dir / 'annotations',
            split_name='train'
        )
        
        # Test set
        print("Processing test set...")
        test_metadata = extract_st_iou_metadata(test_annotations)
        test_st_iou_gt = convert_annotations_to_st_iou_format(test_annotations)
        save_st_iou_metadata(
            test_metadata,
            test_st_iou_gt,
            test_dir / 'annotations',
            split_name='test'
        )
    
    print(f"\n{'='*60}")
    print(f"Dataset split complete!")
    print(f"{'='*60}")
    print(f"Train data saved to: {train_dir}")
    print(f"Test data saved to: {test_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Split dataset with default 9:1 ratio
    python split_dataset.py --input_dir ./raw --output_dir ./datasets
    
    # Split with custom ratio (8:2)
    python split_dataset.py --input_dir ./raw --output_dir ./datasets --train_ratio 0.8
    
    # Split with ST-IoU validation and metadata generation
    python split_dataset.py --input_dir ./raw --output_dir ./datasets --validate_st_iou --save_st_iou_metadata
    
    # Dry run with ST-IoU validation
    python split_dataset.py --input_dir ./raw --output_dir ./datasets --dry_run --validate_st_iou
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to input dataset directory containing annotations/ and samples/'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to output directory where train/ and test/ will be created'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Ratio of data to use for training (default: 0.9 for 9:1 split)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print split information without copying files'
    )
    
    parser.add_argument(
        '--validate_st_iou',
        action='store_true',
        help='Validate that annotations are compatible with ST-IoU evaluation'
    )
    
    parser.add_argument(
        '--save_st_iou_metadata',
        action='store_true',
        help='Generate and save ST-IoU evaluation files (metadata, ground truth, summary)'
    )
    
    args = parser.parse_args()
    
    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        parser.error("train_ratio must be between 0 and 1")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied\n")
        input_path = Path(args.input_dir)
        annotations_file = input_path / 'annotations' / 'annotations.json'
        
        if not annotations_file.exists():
            print(f"Error: Annotations file not found: {annotations_file}")
            return
        
        annotations = load_annotations(annotations_file)
        
        # Validate ST-IoU if requested
        if args.validate_st_iou:
            print(f"\n{'='*60}")
            print("Validating ST-IoU Compatibility...")
            print(f"{'='*60}")
            is_valid = validate_st_iou_compatibility(annotations, verbose=True)
            if not is_valid:
                print("\n⚠️  Warning: Dataset has ST-IoU compatibility issues")
            print(f"{'='*60}\n")
        
        train_annotations, test_annotations = split_annotations(
            annotations, train_ratio=args.train_ratio, seed=args.seed
        )
        
        print(f"\nSplit Summary (Dry Run):")
        print(f"Total classes: {len(annotations)}")
        print(f"Train classes: {len(train_annotations)} ({len(train_annotations)/len(annotations)*100:.1f}%)")
        print(f"Test classes: {len(test_annotations)} ({len(test_annotations)/len(annotations)*100:.1f}%)")
        print(f"\nTrain video IDs: {[item['video_id'] for item in train_annotations]}")
        print(f"\nTest video IDs: {[item['video_id'] for item in test_annotations]}")
    else:
        split_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            validate_st_iou=args.validate_st_iou,
            generate_st_iou_files=args.save_st_iou_metadata
        )


if __name__ == '__main__':
    main()
