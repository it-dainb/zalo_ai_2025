"""
Test compatibility between loss_utils.py and all loss functions
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training.loss_utils import prepare_loss_inputs
from src.losses.combined_loss import ReferenceBasedDetectionLoss


def test_loss_compatibility():
    """Test that loss_utils outputs are compatible with combined_loss inputs"""
    print("\n" + "="*70)
    print("Testing Loss Compatibility: loss_utils → combined_loss")
    print("="*70 + "\n")
    
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
        proto_boxes_list.append(torch.randn(batch_size, 4 * (reg_max + 1), H, W, device=device))
        proto_sim_list.append(torch.randn(batch_size, num_classes, H, W, device=device))
    
    # Create dummy ground truth targets
    target_bboxes = [
        torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32, device=device),
        torch.tensor([[150, 150, 250, 250]], dtype=torch.float32, device=device),
    ]
    target_classes = [
        torch.tensor([0, 1], dtype=torch.long, device=device),
        torch.tensor([2], dtype=torch.long, device=device),
    ]
    
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
    
    print("Step 1: Prepare loss inputs using loss_utils")
    print("-" * 70)
    loss_inputs = prepare_loss_inputs(model_outputs, batch, stage=2, reg_max=reg_max)
    
    print("Loss inputs prepared:")
    for key, value in loss_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: shape={str(value.shape):20s} dtype={value.dtype}")
        else:
            print(f"  {key:20s}: {type(value)}")
    print()
    
    # Check required keys
    required_keys = ['pred_bboxes', 'pred_cls_logits', 'pred_dfl_dist', 
                     'target_bboxes', 'target_cls', 'target_dfl']
    
    print("Step 2: Validate required keys")
    print("-" * 70)
    missing_keys = [k for k in required_keys if k not in loss_inputs]
    if missing_keys:
        print(f"✗ Missing keys: {missing_keys}")
        return False
    print(f"✓ All required keys present: {required_keys}\n")
    
    # Check tensor shapes and dtypes
    print("Step 3: Validate tensor shapes and dtypes")
    print("-" * 70)
    
    checks = []
    
    # Check pred_bboxes
    pred_bboxes = loss_inputs['pred_bboxes']
    target_bboxes_tensor = loss_inputs['target_bboxes']
    checks.append(('pred_bboxes shape[-1] == 4', pred_bboxes.shape[-1] == 4))
    checks.append(('pred_bboxes matches target_bboxes count', 
                   pred_bboxes.shape[0] == target_bboxes_tensor.shape[0]))
    
    # Check pred_cls_logits
    pred_cls = loss_inputs['pred_cls_logits']
    target_cls = loss_inputs['target_cls']
    checks.append(('pred_cls_logits.shape == target_cls.shape',
                   pred_cls.shape == target_cls.shape))
    checks.append(('pred_cls_logits has num_classes dimension',
                   pred_cls.shape[-1] == num_classes))
    
    # Check pred_dfl_dist
    pred_dfl = loss_inputs['pred_dfl_dist']
    target_dfl = loss_inputs['target_dfl']
    expected_dfl_dim = 4 * (reg_max + 1)
    checks.append(('pred_dfl_dist has correct dimension',
                   pred_dfl.shape[-1] == expected_dfl_dim))
    checks.append(('target_dfl shape[-1] == 4',
                   target_dfl.shape[-1] == 4))
    
    # Print results
    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False
    print()
    
    if not all_passed:
        print("✗ Shape validation failed")
        return False
    
    # Test with actual loss function
    print("Step 4: Test with ReferenceBasedDetectionLoss")
    print("-" * 70)
    
    # Stage 1: Detection only
    loss_fn_stage1 = ReferenceBasedDetectionLoss(stage=1, reg_max=reg_max).to(device)
    
    try:
        losses_stage1 = loss_fn_stage1(
            pred_bboxes=loss_inputs['pred_bboxes'],
            pred_cls_logits=loss_inputs['pred_cls_logits'],
            pred_dfl_dist=loss_inputs['pred_dfl_dist'],
            target_bboxes=loss_inputs['target_bboxes'],
            target_cls=loss_inputs['target_cls'],
            target_dfl=loss_inputs['target_dfl'],
        )
        
        print("Stage 1 losses:")
        for key, value in losses_stage1.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:20s}: {value.item():.6f}")
        print()
        
        # Check for non-zero losses
        core_losses = ['bbox_loss', 'cls_loss', 'dfl_loss']
        non_zero = [k for k in core_losses if losses_stage1[k].item() > 0]
        if len(non_zero) == len(core_losses):
            print(f"✓ All core losses are non-zero: {non_zero}")
        else:
            zero_losses = [k for k in core_losses if losses_stage1[k].item() == 0]
            print(f"✗ Some losses are zero: {zero_losses}")
            return False
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ ALL COMPATIBILITY TESTS PASSED")
    print("="*70)
    print("\nSummary:")
    print(f"  - loss_utils correctly prepares inputs")
    print(f"  - Tensor shapes match loss function expectations")
    print(f"  - WIoU expects: pred_bboxes (M, 4) xyxy ✓")
    print(f"  - BCE expects: pred_cls_logits (M, K) logits ✓")
    print(f"  - DFL expects: pred_dfl_dist (M, 4×(reg_max+1)) ✓")
    print(f"  - All losses compute successfully ✓")
    print(f"  - All losses are non-zero ✓")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    success = test_loss_compatibility()
    sys.exit(0 if success else 1)
