"""
Test anchor-based target assignment to verify non-zero losses.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.loss_utils import assign_targets_to_anchors, prepare_loss_inputs


def test_anchor_assignment():
    """Test that anchor assignment produces non-zero results."""
    print("\n" + "="*60)
    print("Testing Anchor-Based Target Assignment")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create dummy model outputs (multi-scale)
    batch_size = 2
    num_classes = 3
    reg_max = 16
    strides = [4, 8, 16, 32]
    
    # Simulate prototype detection head outputs
    proto_boxes_list = []
    proto_sim_list = []
    
    for stride in strides:
        H = W = 640 // stride
        # Box predictions: (B, 4*(reg_max+1), H, W)
        proto_boxes_list.append(torch.randn(batch_size, 4 * (reg_max + 1), H, W, device=device))
        # Similarity scores: (B, K, H, W)
        proto_sim_list.append(torch.randn(batch_size, num_classes, H, W, device=device))
    
    print("Created model outputs:")
    for i, (boxes, sim) in enumerate(zip(proto_boxes_list, proto_sim_list)):
        print(f"  Scale {i} (stride={strides[i]}): boxes {boxes.shape}, sim {sim.shape}")
    print()
    
    # Create dummy ground truth targets
    target_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32, device=device),  # Image 1: 2 boxes
        torch.tensor([[150, 150, 250, 250]], dtype=torch.float32, device=device),  # Image 2: 1 box
    ]
    target_classes = [
        torch.tensor([0, 1], dtype=torch.long, device=device),  # Image 1: classes 0, 1
        torch.tensor([2], dtype=torch.long, device=device),  # Image 2: class 2
    ]
    
    print("Created ground truth targets:")
    for i, (boxes, classes) in enumerate(zip(target_bboxes, target_classes)):
        print(f"  Image {i}: {len(boxes)} boxes, classes {classes.tolist()}")
    print()
    
    # Test anchor assignment
    print("Running anchor assignment...")
    (matched_pred_bboxes, matched_pred_cls_logits, matched_pred_dfl_dist,
     matched_anchor_points, matched_assigned_strides, matched_target_bboxes, target_cls_onehot, target_dfl) = assign_targets_to_anchors(
        proto_boxes_list=proto_boxes_list,
        proto_sim_list=proto_sim_list,
        target_bboxes=target_bboxes,
        target_classes=target_classes,
        img_size=640,
        reg_max=reg_max,
        strides=strides,
    )
    
    print("\nAnchor assignment results:")
    print(f"  Matched predictions: {matched_pred_bboxes.shape[0]}")
    print(f"  Matched pred bboxes: {matched_pred_bboxes.shape}")
    print(f"  Matched pred cls logits: {matched_pred_cls_logits.shape}")
    print(f"  Matched pred DFL dist: {matched_pred_dfl_dist.shape}")
    print(f"  Matched anchor points: {matched_anchor_points.shape}")
    print(f"  Matched assigned strides: {matched_assigned_strides.shape}")
    print(f"  Target bboxes: {matched_target_bboxes.shape}")
    print(f"  Target cls onehot: {target_cls_onehot.shape}")
    print(f"  Target DFL: {target_dfl.shape}")
    print()
    
    # Check for non-zero assignments
    if matched_pred_bboxes.shape[0] > 0:
        print("✓ Successfully assigned targets to anchors!")
        print(f"  - Total assignments: {matched_pred_bboxes.shape[0]}")
        print(f"  - Pred bbox stats: min={matched_pred_bboxes.min():.3f}, max={matched_pred_bboxes.max():.3f}, mean={matched_pred_bboxes.mean():.3f}")
        print(f"  - Pred cls stats: min={matched_pred_cls_logits.min():.3f}, max={matched_pred_cls_logits.max():.3f}, mean={matched_pred_cls_logits.mean():.3f}")
        print(f"  - Target bbox stats: min={matched_target_bboxes.min():.3f}, max={matched_target_bboxes.max():.3f}, mean={matched_target_bboxes.mean():.3f}")
    else:
        print("✗ No assignments made - this would cause zero loss!")
        return False
    
    # Test prepare_loss_inputs
    print("\n" + "-"*60)
    print("Testing prepare_loss_inputs with anchor-based assignment")
    print("-"*60 + "\n")
    
    model_outputs = {
        'prototype_boxes': proto_boxes_list,
        'prototype_sim': proto_sim_list,
    }
    
    batch = {
        'query_images': torch.randn(batch_size, 3, 640, 640, device=device),
        'target_bboxes': target_bboxes,
        'target_classes': target_classes,
        'num_classes': num_classes,
    }
    
    loss_inputs = prepare_loss_inputs(model_outputs, batch, stage=2, reg_max=reg_max)
    
    print("Loss inputs prepared:")
    for key, value in loss_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    print()
    
    # Check that all required keys are present
    required_keys = ['pred_bboxes', 'pred_cls_logits', 'pred_dfl_dist', 
                     'target_bboxes', 'target_cls', 'target_dfl']
    missing_keys = [k for k in required_keys if k not in loss_inputs]
    
    if missing_keys:
        print(f"✗ Missing required keys: {missing_keys}")
        return False
    
    # Check for non-zero tensors
    if loss_inputs['pred_bboxes'].shape[0] == 0:
        print("✗ pred_bboxes is empty - this would cause zero loss!")
        return False
    
    print("✓ All loss inputs prepared successfully!")
    print(f"  - Number of matched predictions: {loss_inputs['pred_bboxes'].shape[0]}")
    
    return True


if __name__ == '__main__':
    success = test_anchor_assignment()
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("  Anchor-based assignment is working correctly!")
        print("  Ready to test with full training loop.")
    else:
        print("✗ TESTS FAILED")
        print("  Please check the anchor assignment logic.")
    print("="*60 + "\n")
    
    sys.exit(0 if success else 1)
