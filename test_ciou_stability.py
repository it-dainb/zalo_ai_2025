"""
Test CIoU Loss numerical stability with edge cases.
This validates the fixes to prevent NaN gradients during training.
"""

import torch
from src.losses.ciou_loss import CIoULoss


def test_ciou_stability():
    """Test CIoU loss with various edge cases that can cause NaN"""
    
    print("Testing CIoU Loss Numerical Stability")
    print("=" * 60)
    
    loss_fn = CIoULoss(eps=1e-7)
    
    # Test cases that historically caused NaN gradients
    test_cases = [
        {
            "name": "Normal boxes",
            "pred": torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32),
            "target": torch.tensor([[12.0, 12.0, 48.0, 48.0]], dtype=torch.float32),
        },
        {
            "name": "Very small boxes (near-zero dimensions)",
            "pred": torch.tensor([[10.0, 10.0, 10.01, 10.01]], dtype=torch.float32),
            "target": torch.tensor([[10.0, 10.0, 10.02, 10.02]], dtype=torch.float32),
        },
        {
            "name": "Extreme aspect ratio (wide)",
            "pred": torch.tensor([[0.0, 0.0, 100.0, 1.0]], dtype=torch.float32),
            "target": torch.tensor([[0.0, 0.0, 90.0, 1.0]], dtype=torch.float32),
        },
        {
            "name": "Extreme aspect ratio (tall)",
            "pred": torch.tensor([[0.0, 0.0, 1.0, 100.0]], dtype=torch.float32),
            "target": torch.tensor([[0.0, 0.0, 1.0, 90.0]], dtype=torch.float32),
        },
        {
            "name": "Large coordinates",
            "pred": torch.tensor([[1000.0, 1000.0, 1200.0, 1200.0]], dtype=torch.float32),
            "target": torch.tensor([[1010.0, 1010.0, 1190.0, 1190.0]], dtype=torch.float32),
        },
        {
            "name": "Zero overlap",
            "pred": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            "target": torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float32),
        },
        {
            "name": "Mixed precision simulation (FP16 range)",
            "pred": torch.tensor([[10.5, 10.5, 50.25, 50.75]], dtype=torch.float16).float(),
            "target": torch.tensor([[12.25, 12.75, 48.5, 48.25]], dtype=torch.float16).float(),
        },
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['name']}")
        print("-" * 60)
        
        pred = case["pred"].requires_grad_(True)
        target = case["target"]
        
        try:
            # Forward pass
            loss = loss_fn(pred, target)
            
            # Check loss value
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  ❌ FAILED: Forward pass produced NaN/Inf")
                print(f"     Loss: {loss.item()}")
                all_passed = False
                continue
            
            print(f"  Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            if pred.grad is None:
                print(f"  ❌ FAILED: No gradients computed")
                all_passed = False
                continue
            
            if torch.isnan(pred.grad).any() or torch.isinf(pred.grad).any():
                print(f"  ❌ FAILED: Backward pass produced NaN/Inf gradients")
                print(f"     Grad: {pred.grad}")
                all_passed = False
                continue
            
            grad_norm = pred.grad.norm().item()
            print(f"  Gradient norm: {grad_norm:.6f}")
            print(f"  ✅ PASSED")
            
        except Exception as e:
            print(f"  ❌ FAILED: Exception occurred")
            print(f"     {type(e).__name__}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests PASSED - CIoU loss is numerically stable")
    else:
        print("❌ Some tests FAILED - CIoU loss has numerical issues")
    print("=" * 60)
    
    return all_passed


def test_mixed_precision_stability():
    """Test CIoU loss specifically with mixed precision (FP16)"""
    
    print("\n\nTesting Mixed Precision (FP16) Stability")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping mixed precision test")
        return True
    
    device = torch.device("cuda")
    loss_fn = CIoULoss(eps=1e-7).to(device)
    
    # Simulate realistic training batch
    batch_size = 32
    pred = torch.rand(batch_size, 4, device=device) * 640  # Random boxes in [0, 640]
    pred[:, 2:] += pred[:, :2] + 10  # Ensure x2 > x1, y2 > y1
    
    target = pred + torch.randn_like(pred) * 20  # Add noise
    target[:, 2:] = torch.maximum(target[:, 2:], target[:, :2] + 5)  # Ensure valid boxes
    
    pred = pred.requires_grad_(True)
    
    try:
        # Test with autocast (FP16)
        from torch.amp import autocast
        
        with autocast(device_type='cuda', dtype=torch.float16):
            loss = loss_fn(pred, target)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"  ❌ FAILED: Forward pass produced NaN/Inf in FP16")
            return False
        
        print(f"  Loss (FP16): {loss.item():.6f}")
        
        # Backward
        loss.backward()
        
        if pred.grad is None or torch.isnan(pred.grad).any() or torch.isinf(pred.grad).any():
            print(f"  ❌ FAILED: Backward pass produced NaN/Inf gradients in FP16")
            return False
        
        grad_norm = pred.grad.norm().item()
        print(f"  Gradient norm (FP16): {grad_norm:.6f}")
        print(f"  ✅ PASSED - Mixed precision is stable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: Exception in mixed precision test")
        print(f"     {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    # Run stability tests
    test1_passed = test_ciou_stability()
    test2_passed = test_mixed_precision_stability()
    
    print("\n\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Basic stability: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Mixed precision: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("\n✅ CIoU loss is ready for training!")
        exit(0)
    else:
        print("\n❌ CIoU loss needs further fixes")
        exit(1)
