"""Test DFL loss fix with actual training step"""
import torch
import sys
sys.path.insert(0, '.')

from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.loss_utils import assign_targets_to_anchors

def test_training_step():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reg_max = 16
    batch_size = 2
    n_way = 4
    
    print("=" * 70)
    print("TEST: Full Training Step with DFL Loss Fix")
    print("=" * 70)
    
    # Create fake detection outputs
    proto_boxes_list = []
    proto_sim_list = []
    for stride, h_dim in zip([4, 8, 16, 32], [160, 80, 40, 20]):
        boxes = torch.randn(batch_size, 4*reg_max, h_dim, h_dim, device=device, requires_grad=True)
        sim = torch.randn(batch_size, n_way, h_dim, h_dim, device=device, requires_grad=True)
        proto_boxes_list.append(boxes)
        proto_sim_list.append(sim)
    
    # Create targets
    target_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], device=device, dtype=torch.float32),
        torch.tensor([[150, 150, 250, 250]], device=device, dtype=torch.float32),
    ]
    
    target_classes = [
        torch.tensor([0, 1], device=device, dtype=torch.long),
        torch.tensor([2], device=device, dtype=torch.long),
    ]
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num classes (n_way): {n_way}")
    print(f"Target bboxes per image: {[len(t) for t in target_bboxes]}")
    
    # Assign targets to anchors
    print("\n" + "-" * 70)
    print("Step 1: Assign targets to anchors")
    print("-" * 70)
    
    result = assign_targets_to_anchors(
        proto_boxes_list=proto_boxes_list,
        proto_sim_list=proto_sim_list,
        target_bboxes=target_bboxes,
        target_classes=target_classes,
        img_size=640,
        reg_max=reg_max,
    )
    
    matched_pred_bboxes, matched_pred_cls, matched_pred_dfl, matched_anchors, matched_strides, target_boxes, target_cls_onehot, target_dfl = result
    
    print(f"Assigned {matched_pred_bboxes.shape[0]} anchors")
    print(f"  Predictions: {matched_pred_bboxes.shape}")
    print(f"  Target DFL: {target_dfl.shape}")
    print(f"  Target DFL range: [{target_dfl.min():.2f}, {target_dfl.max():.2f}]")
    print(f"  Target DFL mean: {target_dfl.mean():.2f}")
    
    # Compute loss
    print("\n" + "-" * 70)
    print("Step 2: Compute losses")
    print("-" * 70)
    
    loss_fn = ReferenceBasedDetectionLoss(
        stage=1,
        reg_max=reg_max,
        bbox_weight=1.0,
        cls_weight=1.0,
        dfl_weight=0.5,
        supcon_weight=0.0,
        cpe_weight=0.1,
    ).to(device)
    
    # Note: pred_bboxes expects decoded bboxes [N, 4] but we have DFL dist
    # For this test, we just need to verify DFL loss works
    # So we'll create fake decoded bboxes
    from src.training.loss_utils import dist2bbox
    from src.losses.dfl_loss import DFLoss
    
    # Decode DFL predictions to bboxes
    dfl_decoder = DFLoss(reg_max=reg_max).to(device)
    decoded_dists = dfl_decoder.decode(matched_pred_dfl)  # [N, 4]
    pred_bboxes_decoded = dist2bbox(decoded_dists, matched_anchors, xywh=False, dim=1)  # [N, 4]
    
    losses = loss_fn(
        pred_bboxes=pred_bboxes_decoded,
        pred_cls_logits=matched_pred_cls,
        pred_dfl_dist=matched_pred_dfl,
        target_bboxes=target_boxes,
        target_cls=target_cls_onehot,
        target_dfl=target_dfl,
    )
    
    print(f"Losses computed:")
    for name, value in losses.items():
        has_nan = torch.isnan(value).any().item()
        has_inf = torch.isinf(value).any().item()
        status = "✅" if not (has_nan or has_inf) else "❌"
        print(f"  {status} {name:12s}: {value.item():10.6f} (nan={has_nan}, inf={has_inf})")
    
    total_loss = losses['total_loss']
    
    # Test backward pass
    print("\n" + "-" * 70)
    print("Step 3: Backward pass")
    print("-" * 70)
    
    try:
        total_loss.backward()
        
        # Check gradients
        grad_stats = []
        for i, (boxes, sim) in enumerate(zip(proto_boxes_list, proto_sim_list)):
            if boxes.grad is not None:
                has_nan = torch.isnan(boxes.grad).any().item()
                has_inf = torch.isinf(boxes.grad).any().item()
                grad_norm = boxes.grad.norm().item()
                grad_stats.append(("boxes", i, has_nan, has_inf, grad_norm))
            
            if sim.grad is not None:
                has_nan = torch.isnan(sim.grad).any().item()
                has_inf = torch.isinf(sim.grad).any().item()
                grad_norm = sim.grad.norm().item()
                grad_stats.append(("sim", i, has_nan, has_inf, grad_norm))
        
        print(f"Gradient statistics:")
        all_good = True
        for name, scale_idx, has_nan, has_inf, grad_norm in grad_stats:
            status = "✅" if not (has_nan or has_inf) else "❌"
            print(f"  {status} {name}_scale{scale_idx}: norm={grad_norm:.6f}, nan={has_nan}, inf={has_inf}")
            if has_nan or has_inf:
                all_good = False
        
        if all_good:
            print(f"\n{'='*70}")
            print("✅ SUCCESS: All losses and gradients are valid!")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n{'='*70}")
            print("❌ FAILURE: Some gradients contain NaN or Inf")
            print(f"{'='*70}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error during backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_step()
    sys.exit(0 if success else 1)
