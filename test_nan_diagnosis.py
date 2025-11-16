"""
Diagnostic script to identify source of NaN losses
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.losses.wiou_loss import WIoULoss
from src.losses.dfl_loss import DFLoss
from src.losses.cpe_loss import SimplifiedCPELoss


def test_wiou_with_nan_inputs():
    """Test WIoU loss with various edge cases"""
    print("\n" + "="*70)
    print("Testing WIoU Loss for NaN Propagation")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wiou_loss = WIoULoss().to(device)
    
    # Test 1: Normal boxes
    pred = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], device=device, dtype=torch.float32)
    target = torch.tensor([[15, 15, 45, 45], [105, 105, 145, 145]], device=device, dtype=torch.float32)
    
    loss = wiou_loss(pred, target)
    print(f"Test 1 - Normal boxes: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 2: Zero-area boxes (degenerate)
    pred = torch.tensor([[10, 10, 10, 10]], device=device, dtype=torch.float32)
    target = torch.tensor([[15, 15, 20, 20]], device=device, dtype=torch.float32)
    
    loss = wiou_loss(pred, target)
    print(f"Test 2 - Zero-area pred: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 3: Extreme coordinates
    pred = torch.tensor([[1000, 1000, 5000, 5000]], device=device, dtype=torch.float32)
    target = torch.tensor([[2000, 2000, 4000, 4000]], device=device, dtype=torch.float32)
    
    loss = wiou_loss(pred, target)
    print(f"Test 3 - Extreme coords: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 4: Inverted boxes (x1 > x2)
    pred = torch.tensor([[50, 50, 10, 10]], device=device, dtype=torch.float32)  # Inverted
    target = torch.tensor([[15, 15, 45, 45]], device=device, dtype=torch.float32)
    
    try:
        loss = wiou_loss(pred, target)
        print(f"Test 4 - Inverted boxes: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    except Exception as e:
        print(f"Test 4 - Inverted boxes: ERROR - {e}")
    
    # Test 5: Out of image bounds (negative coords)
    pred = torch.tensor([[-10, -10, 50, 50]], device=device, dtype=torch.float32)
    target = torch.tensor([[15, 15, 45, 45]], device=device, dtype=torch.float32)
    
    loss = wiou_loss(pred, target)
    print(f"Test 5 - Negative coords: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    print("\n✓ WIoU Loss Tests Complete\n")


def test_dfl_with_nan_inputs():
    """Test DFL loss with various edge cases"""
    print("="*70)
    print("Testing DFL Loss for NaN Propagation")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dfl_loss = DFLoss(reg_max=16).to(device)
    
    batch_size = 4
    reg_max = 16
    
    # Test 1: Normal distributions
    pred_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    target_dist = torch.rand(batch_size, 4, device=device) * reg_max
    
    loss = dfl_loss(pred_dist, target_dist)
    print(f"Test 1 - Normal dists: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 2: Extreme pred values
    pred_dist = torch.ones(batch_size, 4 * reg_max, device=device) * 100.0  # Extreme
    target_dist = torch.rand(batch_size, 4, device=device) * reg_max
    
    loss = dfl_loss(pred_dist, target_dist)
    print(f"Test 2 - Extreme pred: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 3: Target out of range
    pred_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    target_dist = torch.ones(batch_size, 4, device=device) * 20.0  # > reg_max
    
    loss = dfl_loss(pred_dist, target_dist)
    print(f"Test 3 - Target > reg_max: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 4: Negative targets
    pred_dist = torch.randn(batch_size, 4 * reg_max, device=device)
    target_dist = torch.ones(batch_size, 4, device=device) * -5.0  # Negative
    
    loss = dfl_loss(pred_dist, target_dist)
    print(f"Test 4 - Negative target: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    print("\n✓ DFL Loss Tests Complete\n")


def test_cpe_with_nan_inputs():
    """Test CPE loss with various edge cases"""
    print("="*70)
    print("Testing CPE Loss for NaN Propagation")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpe_loss = SimplifiedCPELoss(temperature=0.1).to(device)
    
    # Test 1: Normal features
    features = torch.randn(10, 128, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], device=device, dtype=torch.long)
    
    loss = cpe_loss(features, labels)
    print(f"Test 1 - Normal features: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 2: Single class (no negatives)
    features = torch.randn(10, 128, device=device)
    labels = torch.zeros(10, device=device, dtype=torch.long)
    
    loss = cpe_loss(features, labels)
    print(f"Test 2 - Single class: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 3: All background (-1 labels)
    features = torch.randn(10, 128, device=device)
    labels = torch.ones(10, device=device, dtype=torch.long) * -1
    
    loss = cpe_loss(features, labels)
    print(f"Test 3 - All background: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    # Test 4: Extreme feature values
    features = torch.ones(10, 128, device=device) * 1000.0
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], device=device, dtype=torch.long)
    
    loss = cpe_loss(features, labels)
    print(f"Test 4 - Extreme features: loss={loss.item():.6f}, has_nan={torch.isnan(loss).any().item()}")
    
    print("\n✓ CPE Loss Tests Complete\n")


def test_dist2bbox_conversion():
    """Test dist2bbox conversion for potential NaN sources"""
    print("="*70)
    print("Testing dist2bbox Conversion")
    print("="*70 + "\n")
    
    from src.training.loss_utils import dist2bbox, bbox2dist
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Normal conversion
    anchor_points = torch.tensor([[320, 320], [160, 160]], device=device, dtype=torch.float32)
    distances = torch.tensor([[50, 50, 50, 50], [30, 30, 30, 30]], device=device, dtype=torch.float32)
    
    bboxes = dist2bbox(distances, anchor_points, xywh=False)
    print(f"Test 1 - Normal conversion:")
    print(f"  Anchors: {anchor_points}")
    print(f"  Distances: {distances}")
    print(f"  Bboxes: {bboxes}")
    print(f"  Has NaN: {torch.isnan(bboxes).any().item()}")
    print(f"  Has Inf: {torch.isinf(bboxes).any().item()}")
    
    # Test 2: Large distances
    distances = torch.tensor([[1000, 1000, 1000, 1000]], device=device, dtype=torch.float32)
    anchor_points = torch.tensor([[320, 320]], device=device, dtype=torch.float32)
    
    bboxes = dist2bbox(distances, anchor_points, xywh=False)
    print(f"\nTest 2 - Large distances:")
    print(f"  Bboxes: {bboxes}")
    print(f"  Has NaN: {torch.isnan(bboxes).any().item()}")
    print(f"  Has negative coords: {(bboxes < 0).any().item()}")
    
    # Test 3: Roundtrip conversion
    original_bboxes = torch.tensor([[100, 100, 200, 200]], device=device, dtype=torch.float32)
    anchor_points = torch.tensor([[150, 150]], device=device, dtype=torch.float32)
    
    # Convert bbox to dist
    distances = bbox2dist(anchor_points, original_bboxes, reg_max=16)
    # Convert back to bbox
    reconstructed = dist2bbox(distances, anchor_points, xywh=False)
    
    print(f"\nTest 3 - Roundtrip:")
    print(f"  Original: {original_bboxes}")
    print(f"  Distances: {distances}")
    print(f"  Reconstructed: {reconstructed}")
    print(f"  Error: {(original_bboxes - reconstructed).abs().max().item():.6f}")
    
    print("\n✓ dist2bbox Tests Complete\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("NaN DIAGNOSIS - Testing All Loss Components")
    print("="*70)
    
    test_wiou_with_nan_inputs()
    test_dfl_with_nan_inputs()
    test_cpe_with_nan_inputs()
    test_dist2bbox_conversion()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE")
    print("="*70 + "\n")
