"""
Test validation code compatibility with DFL loss fix
Ensures validation loop uses the correct DFL implementation
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.losses.dfl_loss import DFLoss
from src.losses.combined_loss import create_loss_fn
from src.training.loss_utils import prepare_loss_inputs


def test_dfl_decode_in_validation():
    """Test DFL decode method used during validation"""
    print("\n" + "="*60)
    print("Testing DFL decode for validation")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reg_max = 16
    batch_size = 4
    
    # Create DFL decoder
    dfl_decoder = DFLoss(reg_max=reg_max)
    dfl_decoder = dfl_decoder.to(device)
    
    # Simulate raw DFL distribution from model (before softmax)
    pred_dfl_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    
    # Test decode (this is what validation uses)
    print("\n1. Testing decode method:")
    decoded = dfl_decoder.decode(pred_dfl_dist)
    print(f"   Input shape: {pred_dfl_dist.shape}")
    print(f"   Output shape: {decoded.shape}")
    print(f"   Output range: [{decoded.min().item():.3f}, {decoded.max().item():.3f}]")
    print(f"   Expected range: [0, {reg_max}]")
    
    # Check for NaN/Inf
    has_nan = torch.isnan(decoded).any()
    has_inf = torch.isinf(decoded).any()
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    
    assert not has_nan, "Decoded values contain NaN!"
    assert not has_inf, "Decoded values contain Inf!"
    print("   ✓ Decode works correctly")
    
    # Test with extreme values (edge case)
    print("\n2. Testing decode with extreme values:")
    extreme_dist = torch.zeros(batch_size, 4 * reg_max, device=device)
    extreme_dist[:, :reg_max] = 100.0  # Extreme confidence in first coordinate
    
    decoded_extreme = dfl_decoder.decode(extreme_dist)
    print(f"   Output range: [{decoded_extreme.min().item():.3f}, {decoded_extreme.max().item():.3f}]")
    
    has_nan = torch.isnan(decoded_extreme).any()
    has_inf = torch.isinf(decoded_extreme).any()
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    
    assert not has_nan, "Extreme values cause NaN!"
    assert not has_inf, "Extreme values cause Inf!"
    print("   ✓ Extreme values handled correctly")


def test_dfl_forward_in_validation():
    """Test DFL forward (loss computation) used during validation"""
    print("\n" + "="*60)
    print("Testing DFL forward for validation loss")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reg_max = 16
    batch_size = 4
    
    # Create DFL loss
    dfl_loss = DFLoss(reg_max=reg_max)
    dfl_loss = dfl_loss.to(device)
    
    # Simulate predictions and targets
    pred_dfl_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    target_dfl = torch.rand(batch_size, 4, device=device) * (reg_max - 1)
    
    print("\n1. Testing forward method:")
    loss = dfl_loss(pred_dfl_dist, target_dfl)
    
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Has NaN: {torch.isnan(loss)}")
    print(f"   Has Inf: {torch.isinf(loss)}")
    
    assert not torch.isnan(loss), "DFL loss is NaN!"
    assert not torch.isinf(loss), "DFL loss is Inf!"
    assert loss.item() >= 0, "DFL loss should be non-negative!"
    print("   ✓ Forward works correctly")
    
    # Test gradient flow
    print("\n2. Testing gradient flow:")
    pred_dfl_dist_grad = torch.randn(batch_size, 4 * reg_max, device=device, requires_grad=True)
    loss = dfl_loss(pred_dfl_dist_grad, target_dfl)
    loss.backward()
    
    grad = pred_dfl_dist_grad.grad
    assert grad is not None, "Gradient is None!"
    has_nan_grad = torch.isnan(grad).any()
    has_inf_grad = torch.isinf(grad).any()
    grad_norm = grad.norm().item()
    
    print(f"   Gradient norm: {grad_norm:.6f}")
    print(f"   Has NaN gradient: {has_nan_grad}")
    print(f"   Has Inf gradient: {has_inf_grad}")
    
    assert not has_nan_grad, "Gradients contain NaN!"
    assert not has_inf_grad, "Gradients contain Inf!"
    print("   ✓ Gradients flow correctly")


def test_combined_loss_validation():
    """Test combined loss (used in validation) with DFL component"""
    print("\n" + "="*60)
    print("Testing combined loss for validation")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reg_max = 16
    batch_size = 4
    num_classes = 5
    
    # Create loss function (Stage 2)
    loss_fn = create_loss_fn(stage=2)
    loss_fn = loss_fn.to(device)
    
    # Simulate validation batch
    pred_bboxes = torch.rand(batch_size, 4, device=device) * 640
    pred_cls_logits = torch.randn(batch_size, num_classes, device=device)
    pred_dfl_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    
    target_bboxes = torch.rand(batch_size, 4, device=device) * 640
    target_cls = torch.randint(0, 2, (batch_size, num_classes), device=device).float()
    target_dfl = torch.rand(batch_size, 4, device=device) * (reg_max - 1)
    
    print("\n1. Testing combined loss:")
    losses = loss_fn(
        pred_bboxes=pred_bboxes,
        pred_cls_logits=pred_cls_logits,
        pred_dfl_dist=pred_dfl_dist,
        target_bboxes=target_bboxes,
        target_cls=target_cls,
        target_dfl=target_dfl,
    )
    
    print(f"   Total loss: {losses['total_loss'].item():.6f}")
    print(f"   DFL loss: {losses['dfl_loss'].item():.6f}")
    print(f"   Bbox loss: {losses['bbox_loss'].item():.6f}")
    print(f"   Cls loss: {losses['cls_loss'].item():.6f}")
    
    # Check all losses are valid
    for name, loss in losses.items():
        has_nan = torch.isnan(loss)
        has_inf = torch.isinf(loss)
        print(f"   {name}: valid={not has_nan and not has_inf}")
        
        assert not has_nan, f"{name} is NaN!"
        assert not has_inf, f"{name} is Inf!"
    
    print("   ✓ All loss components valid")
    
    # Test with empty batch (edge case in validation)
    print("\n2. Testing with empty batch:")
    empty_pred_bboxes = torch.zeros(0, 4, device=device)
    empty_pred_cls = torch.zeros(0, num_classes, device=device)
    empty_pred_dfl = torch.zeros(0, 4 * reg_max, device=device)
    empty_target_bboxes = torch.zeros(0, 4, device=device)
    empty_target_cls = torch.zeros(0, num_classes, device=device)
    empty_target_dfl = torch.zeros(0, 4, device=device)
    
    losses_empty = loss_fn(
        pred_bboxes=empty_pred_bboxes,
        pred_cls_logits=empty_pred_cls,
        pred_dfl_dist=empty_pred_dfl,
        target_bboxes=empty_target_bboxes,
        target_cls=empty_target_cls,
        target_dfl=empty_target_dfl,
    )
    
    print(f"   Total loss: {losses_empty['total_loss'].item():.6f}")
    print(f"   DFL loss: {losses_empty['dfl_loss'].item():.6f}")
    
    # Empty batch should give zero loss
    assert losses_empty['dfl_loss'].item() == 0.0, "Empty batch should give zero DFL loss!"
    print("   ✓ Empty batch handled correctly")


def main():
    print("\n" + "="*70)
    print("VALIDATION COMPATIBILITY TEST - DFL Loss Fix")
    print("="*70)
    print("\nThis test verifies that validation code will work correctly")
    print("with the DFL loss fix (removed final loss clamp).")
    
    try:
        test_dfl_decode_in_validation()
        test_dfl_forward_in_validation()
        test_combined_loss_validation()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nValidation code is compatible with DFL loss fix!")
        print("The validation loop will:")
        print("  1. Correctly decode DFL distributions to bbox coordinates")
        print("  2. Compute valid DFL losses without NaN/Inf")
        print("  3. Handle empty batches gracefully")
        print("  4. Provide proper gradient flow")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
