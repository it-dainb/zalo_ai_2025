"""
Analyze bbox sizes and anchor-to-edge distances in training data.

This script helps determine if reg_max=16 is sufficient or if we need
multi-scale anchor assignment.
"""

import json
import numpy as np
from pathlib import Path
import sys


def analyze_dataset_statistics(data_root: str, annotations_file: str, num_samples: int = 100):
    """Analyze bbox statistics from the dataset."""
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} video samples")
    print(f"Analyzing bboxes from all samples...\n")
    
    # Statistics collectors
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    
    # Parse all bboxes from annotations
    total_bboxes = 0
    for entry in annotations[:num_samples]:  # Limit to num_samples videos
        for anno in entry['annotations']:
            for bbox in anno['bboxes']:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_areas.append(area)
                total_bboxes += 1
    
    print(f"Total bboxes analyzed: {total_bboxes}")
    
    # Convert to numpy for analysis
    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    bbox_areas = np.array(bbox_areas)
    
    print("=" * 70)
    print("BBOX SIZE STATISTICS (in pixels)")
    print("=" * 70)
    
    print("\nWidth:")
    print(f"  Mean: {bbox_widths.mean():.2f}")
    print(f"  Median: {np.median(bbox_widths):.2f}")
    print(f"  Min: {bbox_widths.min():.2f}")
    print(f"  Max: {bbox_widths.max():.2f}")
    print(f"  P25: {np.percentile(bbox_widths, 25):.2f}")
    print(f"  P75: {np.percentile(bbox_widths, 75):.2f}")
    print(f"  P95: {np.percentile(bbox_widths, 95):.2f}")
    
    print("\nHeight:")
    print(f"  Mean: {bbox_heights.mean():.2f}")
    print(f"  Median: {np.median(bbox_heights):.2f}")
    print(f"  Min: {bbox_heights.min():.2f}")
    print(f"  Max: {bbox_heights.max():.2f}")
    print(f"  P25: {np.percentile(bbox_heights, 25):.2f}")
    print(f"  P75: {np.percentile(bbox_heights, 75):.2f}")
    print(f"  P95: {np.percentile(bbox_heights, 95):.2f}")
    
    print("\nArea:")
    print(f"  Mean: {bbox_areas.mean():.2f}")
    print(f"  Median: {np.median(bbox_areas):.2f}")
    print(f"  Min: {bbox_areas.min():.2f}")
    print(f"  Max: {bbox_areas.max():.2f}")
    
    # Analyze maximum distance from bbox center to edge
    max_center_to_edge = []
    for w, h in zip(bbox_widths, bbox_heights):
        max_dist = max(w/2, h/2)
        max_center_to_edge.append(max_dist)
    
    max_center_to_edge = np.array(max_center_to_edge)
    
    print("\n" + "=" * 70)
    print("MAX DISTANCE FROM BBOX CENTER TO EDGE (pixels)")
    print("=" * 70)
    print(f"  Mean: {max_center_to_edge.mean():.2f}")
    print(f"  Median: {np.median(max_center_to_edge):.2f}")
    print(f"  Min: {max_center_to_edge.min():.2f}")
    print(f"  Max: {max_center_to_edge.max():.2f}")
    print(f"  P95: {np.percentile(max_center_to_edge, 95):.2f}")
    
    # Analyze representability with different strides and reg_max=16
    print("\n" + "=" * 70)
    print("DFL REPRESENTABILITY ANALYSIS (reg_max=16)")
    print("=" * 70)
    
    strides = [4, 8, 16, 32]
    reg_max = 16
    
    for stride in strides:
        max_representable = reg_max * stride
        can_represent = max_center_to_edge <= max_representable
        pct = 100 * can_represent.sum() / len(can_represent)
        
        print(f"\nStride {stride:2d} (max distance: {max_representable:3d} pixels):")
        print(f"  Can represent: {pct:.1f}% of bboxes")
        print(f"  Cannot represent: {100-pct:.1f}% of bboxes")
        
        if pct < 100:
            over_limit = max_center_to_edge[~can_represent]
            print(f"  Excess distance (mean): {(over_limit - max_representable).mean():.2f} pixels")
            print(f"  Excess distance (max): {(over_limit - max_representable).max():.2f} pixels")
    
    # Multi-scale matching feasibility
    print("\n" + "=" * 70)
    print("MULTI-SCALE MATCHING STRATEGY")
    print("=" * 70)
    
    # For each bbox, determine which stride(s) can represent it
    for stride in strides:
        max_representable = reg_max * stride
        can_represent = max_center_to_edge <= max_representable
        count = can_represent.sum()
        print(f"Stride {stride:2d}: Can represent {count:4d} / {len(max_center_to_edge)} bboxes")
    
    # Determine optimal stride for each bbox
    optimal_strides = []
    for dist in max_center_to_edge:
        # Find smallest stride that can represent this distance
        optimal = None
        for stride in strides:
            if dist <= reg_max * stride:
                optimal = stride
                break
        if optimal is None:
            optimal = strides[-1]  # Use largest stride if none can represent
        optimal_strides.append(optimal)
    
    optimal_strides = np.array(optimal_strides)
    print("\nOptimal Stride Distribution:")
    for stride in strides:
        count = (optimal_strides == stride).sum()
        pct = 100 * count / len(optimal_strides)
        print(f"  Stride {stride:2d}: {count:4d} bboxes ({pct:.1f}%)")
    
    unrepresentable = np.sum(max_center_to_edge > reg_max * strides[-1])
    if unrepresentable > 0:
        pct = 100 * unrepresentable / len(max_center_to_edge)
        print(f"\n⚠️  WARNING: {unrepresentable} bboxes ({pct:.1f}%) cannot be represented even with stride=32!")
        print(f"  Consider increasing reg_max from 16 to 32 or 64")
    else:
        print(f"\n✅ All bboxes can be represented with multi-scale matching")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze bbox and anchor statistics")
    parser.add_argument("--data_root", type=str, default="./datasets/train/samples",
                       help="Path to training data")
    parser.add_argument("--annotations", type=str, default="./datasets/train/annotations.json",
                       help="Path to annotations file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of video samples to analyze")
    
    args = parser.parse_args()
    
    analyze_dataset_statistics(args.data_root, args.annotations, args.num_samples)
