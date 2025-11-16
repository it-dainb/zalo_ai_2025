"""
Debug anchor assignment to understand why distances are all clamped to 10.0
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.loss_utils import bbox2dist, dist2bbox


def debug_anchor_assignment():
    """Debug anchor assignment with sample data."""
    
    device = 'cpu'
    reg_max = 16
    
    # Simulate YOLOv8 anchor grid for 640x640 image
    # P2: stride=4, 160x160 grid
    # P3: stride=8, 80x80 grid
    # P4: stride=16, 40x40 grid
    # P5: stride=32, 20x20 grid
    
    print("=" * 80)
    print("ANCHOR GRID ANALYSIS")
    print("=" * 80)
    
    strides = [4, 8, 16, 32]
    img_size = 640
    
    for stride in strides:
        grid_size = img_size // stride
        print(f"\nStride {stride:2d}: Grid {grid_size}x{grid_size}")
        print(f"  Max representable distance: {reg_max * stride} pixels")
        print(f"  Anchor spacing: {stride} pixels")
    
    # Test case: Small bbox in the center (typical UAV object)
    print("\n" + "=" * 80)
    print("TEST CASE 1: Small bbox in center (typical UAV object)")
    print("=" * 80)
    
    # Bbox: [300, 300, 340, 340] (40x40 pixels)
    gt_bbox = torch.tensor([[300.0, 300.0, 340.0, 340.0]], device=device)
    
    # Find which anchors would be assigned using center-based assignment
    bbox_cx = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2  # 320
    bbox_cy = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2  # 320
    
    print(f"\nGT Bbox: {gt_bbox[0].tolist()}")
    print(f"Bbox center: ({bbox_cx[0]:.1f}, {bbox_cy[0]:.1f})")
    print(f"Bbox size: 40x40 pixels")
    print(f"Max distance from center to edge: 20 pixels")
    
    for stride in strides:
        # Find anchor at bbox center
        anchor_i = int(bbox_cx[0] / stride)
        anchor_j = int(bbox_cy[0] / stride)
        
        # Anchor position in pixels
        anchor_x = anchor_i * stride + stride / 2
        anchor_y = anchor_j * stride + stride / 2
        
        anchor = torch.tensor([[anchor_x, anchor_y]], device=device)
        
        print(f"\n  Stride {stride:2d}:")
        print(f"    Anchor grid position: ({anchor_i}, {anchor_j})")
        print(f"    Anchor pixel position: ({anchor_x:.1f}, {anchor_y:.1f})")
        
        # Calculate distances using bbox2dist
        distances = bbox2dist(anchor, gt_bbox, reg_max=reg_max)
        
        print(f"    Distances (lt, rt, rb, lb): {distances[0].tolist()}")
        
        # Check if clamped
        if torch.any(distances >= reg_max - 0.01):
            print(f"    ⚠️  CLAMPED! Some distances hit reg_max={reg_max}")
        
        # Decode back to bbox
        decoded_bbox = dist2bbox(distances, anchor, xywh=False, dim=1)
        
        # Calculate error
        error = torch.abs(decoded_bbox - gt_bbox).max().item()
        print(f"    Decoded bbox: {decoded_bbox[0].tolist()}")
        print(f"    Reconstruction error: {error:.2f} pixels")
    
    # Test case 2: Bbox NOT at anchor center
    print("\n" + "=" * 80)
    print("TEST CASE 2: Bbox offset from anchor grid")
    print("=" * 80)
    
    # Bbox: [295, 295, 335, 335] (40x40 pixels, but center at 315,315)
    gt_bbox = torch.tensor([[295.0, 295.0, 335.0, 335.0]], device=device)
    bbox_cx = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2  # 315
    bbox_cy = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2  # 315
    
    print(f"\nGT Bbox: {gt_bbox[0].tolist()}")
    print(f"Bbox center: ({bbox_cx[0]:.1f}, {bbox_cy[0]:.1f})")
    print(f"Bbox size: 40x40 pixels")
    
    # For stride=4, nearest anchor is at 314, 314 or 318, 318
    stride = 4
    anchor_i = int(bbox_cx[0] / stride)  # 78
    anchor_j = int(bbox_cy[0] / stride)  # 78
    
    anchor_x = anchor_i * stride + stride / 2  # 78*4 + 2 = 314
    anchor_y = anchor_j * stride + stride / 2  # 314
    
    anchor = torch.tensor([[anchor_x, anchor_y]], device=device)
    
    print(f"\n  Stride {stride}:")
    print(f"    Anchor position: ({anchor_x:.1f}, {anchor_y:.1f})")
    print(f"    Offset from bbox center: ({bbox_cx[0] - anchor_x:.1f}, {bbox_cy[0] - anchor_y:.1f}) pixels")
    
    # Calculate distances
    distances = bbox2dist(anchor, gt_bbox, reg_max=reg_max)
    print(f"    Distances: {distances[0].tolist()}")
    
    # Decode back
    decoded_bbox = dist2bbox(distances, anchor, xywh=False, dim=1)
    error = torch.abs(decoded_bbox - gt_bbox).max().item()
    print(f"    Decoded bbox: {decoded_bbox[0].tolist()}")
    print(f"    Reconstruction error: {error:.2f} pixels")
    
    # Test case 3: What causes clamping?
    print("\n" + "=" * 80)
    print("TEST CASE 3: When does clamping occur?")
    print("=" * 80)
    
    stride = 4
    reg_max = 16
    max_dist_pixels = reg_max * stride  # 64 pixels
    
    print(f"\nStride: {stride}, reg_max: {reg_max}")
    print(f"Max representable distance: {max_dist_pixels} pixels")
    
    # Anchor at (320, 320) in PIXELS
    anchor_pixels = torch.tensor([[320.0, 320.0]], device=device)
    
    # Test bboxes of increasing size
    sizes = [20, 40, 60, 80, 100, 120, 140]
    
    print("\n  WITHOUT stride normalization (BUG):")
    for size in sizes:
        half = size / 2
        gt_bbox_pixels = torch.tensor([[320-half, 320-half, 320+half, 320+half]], device=device)
        
        distances = bbox2dist(anchor_pixels, gt_bbox_pixels, reg_max=reg_max)
        
        clamped = "⚠️ CLAMPED" if torch.any(distances >= reg_max - 0.01) else "✅ OK"
        print(f"    Size {size:3d}x{size:3d}: distances={distances[0, 0].item():.2f} {clamped}")
    
    print("\n  WITH stride normalization (CORRECT):")
    for size in sizes:
        half = size / 2
        gt_bbox_pixels = torch.tensor([[320-half, 320-half, 320+half, 320+half]], device=device)
        
        # Normalize by stride
        anchor_grid = anchor_pixels / stride
        gt_bbox_grid = gt_bbox_pixels / stride
        
        distances = bbox2dist(anchor_grid, gt_bbox_grid, reg_max=reg_max)
        
        clamped = "⚠️ CLAMPED" if torch.any(distances >= reg_max - 0.01) else "✅ OK"
        print(f"    Size {size:3d}x{size:3d}: distances={distances[0, 0].item():.2f} {clamped}")


if __name__ == "__main__":
    debug_anchor_assignment()
