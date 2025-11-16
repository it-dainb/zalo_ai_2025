"""
Test to verify ReLU + epsilon bbox decoding fix produces valid bboxes.

This test simulates the detection head -> decoding pipeline to verify:
1. ReLU forces ltrb offsets to be positive
2. Epsilon ensures x2 > x1 and y2 > y1 even when offsets are zero
3. Decoded bboxes have valid format (x2 > x1, y2 > y1)
"""

import torch
import torch.nn.functional as F

def simulate_detection_head_with_relu(batch_size=2, H=4, W=4):
    """Simulate detection head output with ReLU fix."""
    # Simulate raw conv output (could be negative)
    raw_output = torch.randn(batch_size, 4, H, W) - 0.5  # Bias toward negative
    
    # Apply ReLU + clamp (as in dual_head.py lines 113-114 and 330-331)
    box_pred = F.relu(raw_output)
    box_pred = torch.clamp(box_pred, min=0.0, max=10.0)
    
    return box_pred

def decode_ltrb_with_epsilon(box_preds, anchor_x, anchor_y, stride=8):
    """
    Decode ltrb offsets to xyxy format with epsilon fix.
    
    Matches logic in src/training/loss_utils.py lines 132-137.
    CRITICAL: epsilon added AFTER stride multiplication for numerical stability.
    Using 1e-4 to account for float32 precision limits (spacing at 320.0 is ~3e-5).
    """
    eps = 1e-4
    
    decoded = torch.stack([
        anchor_x - box_preds[:, 0] * stride,  # x1 (left)
        anchor_y - box_preds[:, 1] * stride,  # y1 (top)
        anchor_x + box_preds[:, 2] * stride + eps,  # x2 (right) + eps AFTER stride
        anchor_y + box_preds[:, 3] * stride + eps,  # y2 (bottom) + eps AFTER stride
    ], dim=1)
    
    return decoded

def test_relu_epsilon_fix():
    """Test that ReLU + epsilon produces 100% valid bbox format."""
    print("="*60)
    print("TEST: ReLU + Epsilon BBox Fix Verification")
    print("="*60)
    
    # Simulate detection head outputs
    batch_size, H, W = 2, 4, 4
    stride = 8
    
    # Get box predictions with ReLU fix
    box_preds = simulate_detection_head_with_relu(batch_size, H, W)  # (B, 4, H, W)
    
    print(f"\n1. Detection Head Outputs (after ReLU + clamp):")
    print(f"   Shape: {box_preds.shape}")
    print(f"   Min: {box_preds.min().item():.6f}, Max: {box_preds.max().item():.6f}")
    print(f"   All positive: {(box_preds >= 0).all().item()}")
    
    # Create anchor grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    anchor_x = (grid_x + 0.5) * stride  # (H, W)
    anchor_y = (grid_y + 0.5) * stride  # (H, W)
    
    # Flatten for decoding
    anchor_x_flat = anchor_x.flatten()  # (H*W,)
    anchor_y_flat = anchor_y.flatten()  # (H*W,)
    
    # Process each batch
    all_decoded = []
    for b in range(batch_size):
        box_pred_flat = box_preds[b].reshape(4, -1).t()  # (H*W, 4)
        
        # Decode with epsilon fix
        decoded = decode_ltrb_with_epsilon(
            box_pred_flat,
            anchor_x_flat,
            anchor_y_flat,
            stride=stride
        )
        all_decoded.append(decoded)
    
    all_decoded = torch.cat(all_decoded, dim=0)  # (B*H*W, 4)
    
    print(f"\n2. Decoded BBoxes (xyxy format):")
    print(f"   Shape: {all_decoded.shape}")
    print(f"   x range: [{all_decoded[:, 0].min().item():.1f}, {all_decoded[:, 2].max().item():.1f}]")
    print(f"   y range: [{all_decoded[:, 1].min().item():.1f}, {all_decoded[:, 3].max().item():.1f}]")
    
    # Check validity
    valid_x = (all_decoded[:, 2] > all_decoded[:, 0]).float().mean().item()
    valid_y = (all_decoded[:, 3] > all_decoded[:, 1]).float().mean().item()
    
    print(f"\n3. BBox Format Validation:")
    print(f"   Valid x2 > x1: {valid_x*100:.1f}%")
    print(f"   Valid y2 > y1: {valid_y*100:.1f}%")
    
    # Compute aspect ratios
    widths = all_decoded[:, 2] - all_decoded[:, 0]
    heights = all_decoded[:, 3] - all_decoded[:, 1]
    aspects = widths / (heights + 1e-9)
    
    print(f"\n4. Aspect Ratios (width/height):")
    print(f"   Mean: {aspects.mean().item():.2f}")
    print(f"   Min: {aspects.min().item():.2f}, Max: {aspects.max().item():.2f}")
    print(f"   Reasonable (<100): {(aspects < 100).float().mean().item()*100:.1f}%")
    
    # Test edge case: all zeros (early training)
    print(f"\n5. Edge Case: Zero Predictions (early training)")
    zero_preds = torch.zeros(16, 4)  # (N, 4) all zeros
    anchor_x_test = torch.full((16,), 320.0)  # Center of image
    anchor_y_test = torch.full((16,), 320.0)
    
    decoded_zeros = decode_ltrb_with_epsilon(zero_preds, anchor_x_test, anchor_y_test, stride=8)
    
    valid_x_zeros = (decoded_zeros[:, 2] > decoded_zeros[:, 0]).float().mean().item()
    valid_y_zeros = (decoded_zeros[:, 3] > decoded_zeros[:, 1]).float().mean().item()
    
    print(f"   Valid x2 > x1: {valid_x_zeros*100:.1f}%")
    print(f"   Valid y2 > y1: {valid_y_zeros*100:.1f}%")
    print(f"   x1={decoded_zeros[0, 0]:.6f}, x2={decoded_zeros[0, 2]:.6f}, diff={decoded_zeros[0, 2]-decoded_zeros[0, 0]:.6f}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if valid_x == 1.0 and valid_y == 1.0 and valid_x_zeros == 1.0 and valid_y_zeros == 1.0:
        print("✅ TEST PASSED: ReLU + Epsilon fix produces 100% valid bboxes")
    else:
        print("❌ TEST FAILED: Some bboxes still have invalid format")
        print(f"   Expected: 100% valid, Got: {min(valid_x, valid_y)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    torch.manual_seed(42)
    test_relu_epsilon_fix()
