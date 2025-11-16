"""
Detailed DFL Loss diagnostic - understand why loss is always 10.0
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.losses.dfl_loss import DFLoss


def diagnose_dfl_loss():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_max = 16
    dfl_loss = DFLoss(reg_max=reg_max).to(device)
    
    print("=" * 80)
    print("DFL LOSS DETAILED DIAGNOSIS")
    print("=" * 80)
    
    # Test 1: Simple case - target=5.0
    print("\n" + "-" * 80)
    print("Test 1: Target=5.0 (mid-range)")
    print("-" * 80)
    
    # Create prediction that "guesses" target perfectly
    pred_dist = torch.zeros((1, 4, reg_max), device=device)
    
    # For each coordinate, put high probability at target bin
    target_bin = 5
    pred_dist[:, :, target_bin] = 10.0  # High logit at target
    pred_dist = pred_dist.reshape(1, 4 * reg_max)
    
    target = torch.tensor([[5.0, 5.0, 5.0, 5.0]], device=device)
    
    loss = dfl_loss(pred_dist, target)
    decoded = dfl_loss.decode(pred_dist)
    
    print(f"Target: {target[0].tolist()}")
    print(f"Decoded prediction: {decoded[0].tolist()}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Expected: Loss should be close to 0")
    
    # Test 2: Random init
    print("\n" + "-" * 80)
    print("Test 2: Random initialization")
    print("-" * 80)
    
    pred_dist = torch.randn((1, 4 * reg_max), device=device) * 0.1
    target = torch.tensor([[5.0, 5.0, 5.0, 5.0]], device=device)
    
    loss = dfl_loss(pred_dist, target)
    decoded = dfl_loss.decode(pred_dist)
    
    print(f"Target: {target[0].tolist()}")
    print(f"Decoded prediction: {decoded[0].tolist()}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Expected: Loss ~2-4 (reasonable for random init)")
    
    # Test 3: Check clamping behavior
    print("\n" + "-" * 80)
    print("Test 3: Check where clamping occurs")
    print("-" * 80)
    
    targets = [1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 15.99]
    
    for t in targets:
        pred_dist = torch.randn((1, 4 * reg_max), device=device) * 0.1
        target = torch.tensor([[t, t, t, t]], device=device)
        
        loss = dfl_loss(pred_dist, target)
        print(f"  Target={t:5.2f}: Loss={loss.item():.6f}, Clamped={loss.item() >= 9.99}")
    
    # Test 4: Manual step-through of loss calculation
    print("\n" + "-" * 80)
    print("Test 4: Manual calculation")
    print("-" * 80)
    
    batch_size = 1
    pred_dist = torch.randn((batch_size, 4 * reg_max), device=device) * 0.1
    target = torch.tensor([[5.0, 5.0, 5.0, 5.0]], device=device)
    
    # Reshape
    pred_reshaped = pred_dist.reshape(batch_size, 4, reg_max)
    print(f"Pred shape after reshape: {pred_reshaped.shape}")
    
    # Clamp
    pred_clamped = torch.clamp(pred_reshaped, min=-10.0, max=10.0)
    print(f"Pred clamped range: [{pred_clamped.min().item():.2f}, {pred_clamped.max().item():.2f}]")
    
    # Target bins
    target_clamped = torch.clamp(target, min=0, max=reg_max - 1e-6)
    target_left = target_clamped.long()
    target_right = target_left + 1
    
    print(f"Target: {target[0].tolist()}")
    print(f"Target bins (left): {target_left[0].tolist()}")
    print(f"Target bins (right): {target_right[0].tolist()}")
    
    # Weights
    weight_left = target_right.float() - target_clamped
    weight_right = target_clamped - target_left.float()
    
    print(f"Weight left: {weight_left[0].tolist()}")
    print(f"Weight right: {weight_right[0].tolist()}")
    
    # Loss for coordinate 0
    import torch.nn.functional as F
    dist = pred_clamped[:, 0, :]  # [1, reg_max]
    dist_probs = F.softmax(dist, dim=-1)
    
    print(f"\nCoordinate 0 distribution (probs):")
    print(f"  Min prob: {dist_probs.min().item():.6f}")
    print(f"  Max prob: {dist_probs.max().item():.6f}")
    print(f"  Sum: {dist_probs.sum().item():.6f}")
    
    target_l = int(target_left[0, 0].item())  # 5
    target_r = int(target_right[0, 0].item())  # 6
    
    prob_left = dist_probs[0, target_l]
    prob_right = dist_probs[0, target_r]
    
    print(f"  Prob at bin {target_l}: {prob_left.item():.6f}")
    print(f"  Prob at bin {target_r}: {prob_right.item():.6f}")
    
    # Clamp probs
    prob_left_clamped = torch.clamp(prob_left, min=1e-6, max=1.0)
    prob_right_clamped = torch.clamp(prob_right, min=1e-6, max=1.0)
    
    loss_left = -torch.log(prob_left_clamped) * weight_left[0, 0]
    loss_right = -torch.log(prob_right_clamped) * weight_right[0, 0]
    
    print(f"\n  -log(prob_left) = {-torch.log(prob_left_clamped).item():.6f}")
    print(f"  -log(prob_right) = {-torch.log(prob_right_clamped).item():.6f}")
    print(f"  loss_left (weighted) = {loss_left.item():.6f}")
    print(f"  loss_right (weighted) = {loss_right.item():.6f}")
    
    # Clamp individual losses
    loss_left_clamped = torch.clamp(loss_left, max=20.0)
    loss_right_clamped = torch.clamp(loss_right, max=20.0)
    
    print(f"  loss_left (after clamp) = {loss_left_clamped.item():.6f}")
    print(f"  loss_right (after clamp) = {loss_right_clamped.item():.6f}")
    print(f"  total for coord 0 = {(loss_left_clamped + loss_right_clamped).item():.6f}")
    
    # Full loss
    loss_full = dfl_loss(pred_dist, target)
    print(f"\nFull DFL loss (4 coords): {loss_full.item():.6f}")
    print(f"Expected (~4x coord 0): ~{4 * (loss_left_clamped + loss_right_clamped).item():.6f}")


if __name__ == "__main__":
    diagnose_dfl_loss()
