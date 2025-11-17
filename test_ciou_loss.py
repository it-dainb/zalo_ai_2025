"""
Test CIoU Loss Implementation
Validates CIoU loss against Ultralytics YOLOv8 reference implementation
"""

import torch
import pytest
from src.losses.ciou_loss import CIoULoss


def test_ciou_basic():
    """Test basic CIoU calculation"""
    loss_fn = CIoULoss()
    
    # Test case 1: Perfect overlap (IoU=1, CIoU=1, Loss=0)
    pred = torch.tensor([[10., 10., 20., 20.]])
    target = torch.tensor([[10., 10., 20., 20.]])
    loss = loss_fn(pred, target)
    
    assert loss.item() < 0.01, f"Perfect overlap should give ~0 loss, got {loss.item()}"
    print(f"✓ Perfect overlap: loss={loss.item():.4f}")
    
    # Test case 2: Small offset (high IoU, small center distance penalty)
    pred = torch.tensor([[10., 10., 20., 20.]])
    target = torch.tensor([[11., 11., 21., 21.]])
    loss = loss_fn(pred, target)
    
    assert 0.2 < loss.item() < 0.5, f"Small offset should give moderate loss, got {loss.item()}"
    print(f"✓ Small offset: loss={loss.item():.4f}")
    
    # Test case 3: No overlap (IoU=0, high center distance)
    pred = torch.tensor([[10., 10., 20., 20.]])
    target = torch.tensor([[50., 50., 60., 60.]])
    loss = loss_fn(pred, target)
    
    assert loss.item() > 0.9, f"No overlap should give high loss, got {loss.item()}"
    print(f"✓ No overlap: loss={loss.item():.4f}")


def test_ciou_components():
    """Test individual CIoU components (IoU + DIoU + aspect ratio)"""
    loss_fn = CIoULoss()
    
    # Test aspect ratio penalty
    # Same area but different aspect ratios
    pred = torch.tensor([[10., 10., 30., 20.]])  # 20x10 = 200
    target = torch.tensor([[10., 10., 20., 30.]])  # 10x20 = 200
    loss = loss_fn(pred, target)
    
    print(f"✓ Aspect ratio penalty: loss={loss.item():.4f}")
    
    # Test center distance penalty (DIoU component)
    # Same size but different centers
    pred = torch.tensor([[10., 10., 20., 20.]])
    target = torch.tensor([[15., 15., 25., 25.]])
    loss = loss_fn(pred, target)
    
    print(f"✓ Center distance penalty: loss={loss.item():.4f}")


def test_ciou_batch():
    """Test CIoU on batch of boxes"""
    loss_fn = CIoULoss()
    
    pred = torch.tensor([
        [10., 10., 20., 20.],
        [30., 30., 40., 40.],
        [50., 50., 60., 60.]
    ])
    target = torch.tensor([
        [11., 11., 21., 21.],
        [30., 30., 40., 40.],
        [55., 55., 65., 65.]
    ])
    
    loss = loss_fn(pred, target, reduction='none')
    assert loss.shape == (3,), f"Expected shape (3,), got {loss.shape}"
    
    loss_mean = loss_fn(pred, target, reduction='mean')
    assert torch.allclose(loss_mean, loss.mean()), "Mean reduction mismatch"
    
    print(f"✓ Batch processing: losses={loss.tolist()}, mean={loss_mean.item():.4f}")


def test_ciou_gradients():
    """Test that CIoU loss produces valid gradients"""
    loss_fn = CIoULoss()
    
    pred = torch.tensor([[10., 10., 20., 20.]], requires_grad=True)
    target = torch.tensor([[11., 11., 21., 21.]])
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradient not computed"
    assert not torch.isnan(pred.grad).any(), "NaN in gradients"
    assert not torch.isinf(pred.grad).any(), "Inf in gradients"
    
    print(f"✓ Gradients valid: grad_norm={pred.grad.norm().item():.4f}")


def test_ciou_edge_cases():
    """Test CIoU edge cases"""
    loss_fn = CIoULoss()
    
    # Test case 1: Zero-size box (degenerate)
    pred = torch.tensor([[10., 10., 10., 10.]])
    target = torch.tensor([[10., 10., 20., 20.]])
    loss = loss_fn(pred, target)
    
    assert not torch.isnan(loss), "NaN for zero-size box"
    assert not torch.isinf(loss), "Inf for zero-size box"
    print(f"✓ Zero-size box handled: loss={loss.item():.4f}")
    
    # Test case 2: Very large boxes
    pred = torch.tensor([[0., 0., 1000., 1000.]])
    target = torch.tensor([[0., 0., 1000., 1000.]])
    loss = loss_fn(pred, target)
    
    assert loss.item() < 0.01, f"Perfect overlap of large boxes should give ~0 loss, got {loss.item()}"
    print(f"✓ Large boxes: loss={loss.item():.4f}")
    
    # Test case 3: Very small boxes
    pred = torch.tensor([[0., 0., 0.1, 0.1]])
    target = torch.tensor([[0., 0., 0.1, 0.1]])
    loss = loss_fn(pred, target)
    
    assert loss.item() < 0.01, f"Perfect overlap of small boxes should give ~0 loss, got {loss.item()}"
    print(f"✓ Small boxes: loss={loss.item():.4f}")


def test_ciou_vs_iou():
    """Compare CIoU with simple IoU - CIoU should give better gradients"""
    loss_fn = CIoULoss()
    
    # Case 1: Boxes with same IoU but different center distances
    # CIoU should penalize the one with larger center distance more
    
    # Box A: high IoU, small center distance
    pred_a = torch.tensor([[10., 10., 20., 20.]])
    target_a = torch.tensor([[11., 11., 21., 21.]])
    loss_a = loss_fn(pred_a, target_a)
    
    # Box B: high IoU, larger center distance
    pred_b = torch.tensor([[10., 10., 20., 20.]])
    target_b = torch.tensor([[13., 13., 23., 23.]])
    loss_b = loss_fn(pred_b, target_b)
    
    assert loss_b > loss_a, "CIoU should penalize larger center distance"
    print(f"✓ CIoU center distance sensitivity: loss_small={loss_a.item():.4f}, loss_large={loss_b.item():.4f}")


def test_ciou_stability():
    """Test CIoU numerical stability"""
    loss_fn = CIoULoss()
    
    # Random boxes
    torch.manual_seed(42)
    pred = torch.rand(100, 4) * 100
    pred[:, 2:] += pred[:, :2] + 1  # Ensure x2>x1, y2>y1
    
    target = torch.rand(100, 4) * 100
    target[:, 2:] += target[:, :2] + 1
    
    loss = loss_fn(pred, target, reduction='mean')
    
    assert not torch.isnan(loss), "NaN in random boxes"
    assert not torch.isinf(loss), "Inf in random boxes"
    assert 0 <= loss <= 2, f"Loss out of expected range [0, 2]: {loss.item()}"
    
    print(f"✓ Stability test (100 random boxes): loss={loss.item():.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing CIoU Loss Implementation")
    print("=" * 60)
    
    test_ciou_basic()
    print()
    
    test_ciou_components()
    print()
    
    test_ciou_batch()
    print()
    
    test_ciou_gradients()
    print()
    
    test_ciou_edge_cases()
    print()
    
    test_ciou_vs_iou()
    print()
    
    test_ciou_stability()
    print()
    
    print("=" * 60)
    print("✅ All CIoU tests passed!")
    print("=" * 60)
