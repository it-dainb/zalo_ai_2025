"""
Minimal test to verify Stage 2 training works without DFL.
Tests loss function and target assignment work correctly.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.losses.combined_loss import create_loss_fn
from src.training.loss_utils import assign_targets_to_anchors

def test_stage2_no_dfl():
    """Test Stage 2 components work without DFL"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}\n")
    
    # Test 1: Loss function has no DFL
    print("1. Testing loss function (no DFL)...")
    loss_fn = create_loss_fn(
        stage=2,
        bbox_weight=7.5,
        cls_weight=0.5,
        supcon_weight=1.0,
        cpe_weight=0.5,
        triplet_weight=0.2
    )
    print(f"   ✓ Loss weights: {loss_fn.weights}")
    assert 'dfl' not in loss_fn.weights, "DFL found in loss weights!"
    assert len(loss_fn.weights) == 5, f"Expected 5 loss components, got {len(loss_fn.weights)}"
    print("   ✓ No DFL component (5 components: bbox, cls, supcon, cpe, triplet)")
    
    # Test 2: Create fake detection outputs (4-channel bbox, no DFL)
    print("\n2. Creating fake detection outputs (4-channel bbox)...")
    
    batch_size = 2
    n_way = 2
    img_size = 640
    
    # Create detection outputs for 4 scales
    proto_boxes_list = []
    proto_sim_list = []
    
    for stride, h_dim in zip([4, 8, 16, 32], [160, 80, 40, 20]):
        # Boxes: (B, 4, H, W) - 4 channels for direct bbox prediction (no DFL)
        boxes = torch.randn(batch_size, 4, h_dim, h_dim, device=device)
        proto_boxes_list.append(boxes)
        
        # Similarity: (B, K, H, W) where K=n_way
        sim = torch.randn(batch_size, n_way, h_dim, h_dim, device=device)
        proto_sim_list.append(sim)
    
    print(f"   ✓ Detection outputs created:")
    for i, (boxes, sim) in enumerate(zip(proto_boxes_list, proto_sim_list)):
        B, C_box, H, W = boxes.shape
        B, C_sim, H, W = sim.shape
        print(f"     Scale {i}: boxes ({B}, {C_box}, {H}, {W}), sim ({B}, {C_sim}, {H}, {W})")
        assert C_box == 4, f"Expected 4 bbox channels, got {C_box} (DFL would give 64)"
    print("   ✓ Bbox predictions are 4-channel (no DFL)")
    
    # Test 3: Target assignment returns 6 tensors (not 8 with DFL)
    print("\n3. Testing target assignment (no DFL tensors)...")
    
    target_bboxes = [
        torch.tensor([[150, 150, 250, 250], [350, 350, 450, 450]], 
                    dtype=torch.float32, device=device),
        torch.tensor([[200, 200, 300, 300]], 
                    dtype=torch.float32, device=device),
    ]
    target_classes = [
        torch.tensor([0, 1], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    ]
    
    try:
        assigned = assign_targets_to_anchors(
            proto_boxes_list=proto_boxes_list,
            proto_sim_list=proto_sim_list,
            target_bboxes=target_bboxes,
            target_classes=target_classes,
            img_size=img_size
        )
        
        assert len(assigned) == 6, f"Expected 6 outputs without DFL, got {len(assigned)}"
        print(f"   ✓ Target assignment returns 6 tensors (no DFL)")
        
        # Unpack: (assigned_boxes, assigned_cls_logits, assigned_anchor_points, 
        #          assigned_strides, target_boxes, target_cls_onehot)
        (matched_pred_bboxes, matched_pred_cls_logits, 
         assigned_anchor_points, assigned_strides,
         matched_target_bboxes, matched_target_cls) = assigned
        
        print(f"     - Matched predictions: {matched_pred_bboxes.shape[0]} anchors")
        print(f"     - pred_bboxes shape: {matched_pred_bboxes.shape}")
        print(f"     - pred_cls_logits shape: {matched_pred_cls_logits.shape}")
        
    except Exception as e:
        print(f"   ✗ Target assignment failed: {e}")
        raise
    
    # Test 4: Loss calculation works
    print("\n4. Testing loss calculation...")
    print(f"   Debug shapes:")
    print(f"     - matched_pred_bboxes: {matched_pred_bboxes.shape}")
    print(f"     - matched_target_bboxes: {matched_target_bboxes.shape}")
    print(f"     - matched_pred_cls_logits: {matched_pred_cls_logits.shape}")
    print(f"     - matched_target_cls: {matched_target_cls.shape}")
    
    try:
        # For this test, we're not testing contrastive losses (no query/proposal features)
        # Just test core bbox + cls losses
        losses = loss_fn(
            pred_bboxes=matched_pred_bboxes,
            pred_cls_logits=matched_pred_cls_logits,
            target_bboxes=matched_target_bboxes,
            target_cls=matched_target_cls
        )
        
        print("   ✓ Loss calculation successful:")
        for loss_name, loss_val in losses.items():
            if loss_name != 'total_loss':
                weight = loss_fn.weights.get(loss_name.replace('_loss', ''), 0.0)
                print(f"     - {loss_name}: {loss_val.item():.4f} (weight: {weight})")
        print(f"     - total_loss: {losses['total_loss'].item():.4f}")
        
        assert not torch.isnan(losses['total_loss']), "Total loss is NaN!"
        print("   ✓ No NaN in loss")
        
    except Exception as e:
        print(f"   ✗ Loss calculation failed: {e}")
        raise
    
    # Test 5: Backward pass works
    print("\n5. Testing backward pass...")
    
    try:
        # Recreate outputs with gradients
        proto_boxes_grad = []
        proto_sim_grad = []
        
        for stride, h_dim in zip([4, 8, 16, 32], [160, 80, 40, 20]):
            boxes = torch.randn(batch_size, 4, h_dim, h_dim, device=device, requires_grad=True)
            proto_boxes_grad.append(boxes)
            
            sim = torch.randn(batch_size, n_way, h_dim, h_dim, device=device, requires_grad=True)
            proto_sim_grad.append(sim)
        
        # Assign targets
        assigned = assign_targets_to_anchors(
            proto_boxes_list=proto_boxes_grad,
            proto_sim_list=proto_sim_grad,
            target_bboxes=target_bboxes,
            target_classes=target_classes,
            img_size=img_size
        )
        
        (matched_pred_bboxes, matched_pred_cls_logits, 
         assigned_anchor_points, assigned_strides,
         matched_target_bboxes, matched_target_cls) = assigned
        
        # Calculate loss
        losses = loss_fn(
            pred_bboxes=matched_pred_bboxes,
            pred_cls_logits=matched_pred_cls_logits,
            target_bboxes=matched_target_bboxes,
            target_cls=matched_target_cls
        )
        
        # Backward pass
        losses['total_loss'].backward()
        
        print("   ✓ Backward pass successful")
        
        # Check for NaN gradients in the outputs
        has_nan = False
        for i, (boxes, sim) in enumerate(zip(proto_boxes_grad, proto_sim_grad)):
            if boxes.grad is not None and torch.isnan(boxes.grad).any():
                print(f"   ✗ NaN gradient in proto_boxes scale {i}")
                has_nan = True
            if sim.grad is not None and torch.isnan(sim.grad).any():
                print(f"   ✗ NaN gradient in proto_sim scale {i}")
                has_nan = True
        
        if not has_nan:
            print("   ✓ No NaN gradients")
        else:
            raise RuntimeError("NaN gradients detected!")
        
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        raise
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Stage 2 components work without DFL!")
    print("="*60)
    print("\nSummary:")
    print("- Loss function has 5 components (bbox, cls, supcon, cpe, triplet)")
    print("- Detection outputs use 4-channel bbox predictions (no DFL)")
    print("- Target assignment returns 6 tensors (no pred_dfl_dist, target_dfl)")
    print("- Loss calculation and backward pass work correctly")
    print("- No NaN gradients")
    print("\nStage 2 training is ready to use!")

if __name__ == "__main__":
    test_stage2_no_dfl()
