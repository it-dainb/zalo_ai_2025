"""
Diagnose DFL NaN gradient issue
"""
import torch
import torch.nn.functional as F
from src.losses.dfl_loss import DFLoss

# Simulate the problematic scenario
reg_max = 16
batch_size = 303  # From your log

# Create DFL loss
dfl_loss_fn = DFLoss(reg_max=reg_max)

# Create random predictions (simulating model outputs)
# These could be extreme values causing NaN
pred_dist = torch.randn(batch_size, 4 * reg_max) * 5  # Scale by 5 to simulate potentially large values
pred_dist.requires_grad = True

# Create random targets
target_dfl = torch.rand(batch_size, 4) * (reg_max - 1)

print("=" * 80)
print("DFL LOSS DIAGNOSIS")
print("=" * 80)
print(f"\nInput Statistics:")
print(f"  pred_dist shape: {pred_dist.shape}")
print(f"  pred_dist range: [{pred_dist.min().item():.4f}, {pred_dist.max().item():.4f}]")
print(f"  pred_dist mean: {pred_dist.mean().item():.4f}")
print(f"  pred_dist std: {pred_dist.std().item():.4f}")
print(f"  pred_dist has NaN: {torch.isnan(pred_dist).any()}")
print(f"  pred_dist has Inf: {torch.isinf(pred_dist).any()}")

print(f"\n  target_dfl shape: {target_dfl.shape}")
print(f"  target_dfl range: [{target_dfl.min().item():.4f}, {target_dfl.max().item():.4f}]")
print(f"  target_dfl mean: {target_dfl.mean().item():.4f}")
print(f"  target_dfl has NaN: {torch.isnan(target_dfl).any()}")

# Compute loss
print("\n" + "=" * 80)
print("FORWARD PASS")
print("=" * 80)
try:
    loss = dfl_loss_fn(pred_dist, target_dfl)
    print(f"✓ Loss computed successfully: {loss.item():.6f}")
    print(f"  Loss has NaN: {torch.isnan(loss).any()}")
    print(f"  Loss has Inf: {torch.isinf(loss).any()}")
except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    exit(1)

# Backward pass
print("\n" + "=" * 80)
print("BACKWARD PASS")
print("=" * 80)
try:
    loss.backward()
    print(f"✓ Backward pass completed")
    
    if pred_dist.grad is not None:
        grad_has_nan = torch.isnan(pred_dist.grad).any()
        grad_has_inf = torch.isinf(pred_dist.grad).any()
        
        print(f"  Gradient shape: {pred_dist.grad.shape}")
        print(f"  Gradient range: [{pred_dist.grad.min().item():.4f}, {pred_dist.grad.max().item():.4f}]")
        print(f"  Gradient mean: {pred_dist.grad.mean().item():.4f}")
        print(f"  Gradient std: {pred_dist.grad.std().item():.4f}")
        print(f"  Gradient has NaN: {grad_has_nan}")
        print(f"  Gradient has Inf: {grad_has_inf}")
        
        if grad_has_nan:
            print("\n❌ GRADIENT CONTAINS NaN!")
            # Find where NaN gradients occur
            nan_mask = torch.isnan(pred_dist.grad)
            nan_indices = nan_mask.nonzero()
            print(f"  Number of NaN gradients: {nan_mask.sum().item()}")
            print(f"  First 10 NaN indices: {nan_indices[:10].tolist()}")
        else:
            print("\n✅ NO NaN GRADIENTS DETECTED")
    else:
        print("  ✗ No gradient computed")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Now test with EXTREME values to trigger NaN
print("\n" + "=" * 80)
print("TESTING WITH EXTREME VALUES")
print("=" * 80)

# Test 1: Very large positive values
print("\nTest 1: Very large positive values (50.0)")
pred_extreme = torch.full((batch_size, 4 * reg_max), 50.0, requires_grad=True)
try:
    loss_extreme = dfl_loss_fn(pred_extreme, target_dfl)
    print(f"  Loss: {loss_extreme.item():.6f}")
    loss_extreme.backward()
    if pred_extreme.grad is not None:
        print(f"  Gradient has NaN: {torch.isnan(pred_extreme.grad).any()}")
        print(f"  Gradient has Inf: {torch.isinf(pred_extreme.grad).any()}")
except Exception as e:
    print(f"  Failed: {e}")

# Test 2: Very large negative values
print("\nTest 2: Very large negative values (-50.0)")
pred_extreme2 = torch.full((batch_size, 4 * reg_max), -50.0, requires_grad=True)
try:
    loss_extreme2 = dfl_loss_fn(pred_extreme2, target_dfl)
    print(f"  Loss: {loss_extreme2.item():.6f}")
    loss_extreme2.backward()
    if pred_extreme2.grad is not None:
        print(f"  Gradient has NaN: {torch.isnan(pred_extreme2.grad).any()}")
        print(f"  Gradient has Inf: {torch.isinf(pred_extreme2.grad).any()}")
except Exception as e:
    print(f"  Failed: {e}")

# Test 3: Mixed extreme values
print("\nTest 3: Mixed extreme values [-100, 100]")
pred_extreme3 = torch.randn(batch_size, 4 * reg_max) * 100
pred_extreme3.requires_grad = True
try:
    loss_extreme3 = dfl_loss_fn(pred_extreme3, target_dfl)
    print(f"  Loss: {loss_extreme3.item():.6f}")
    loss_extreme3.backward()
    if pred_extreme3.grad is not None:
        print(f"  Gradient has NaN: {torch.isnan(pred_extreme3.grad).any()}")
        print(f"  Gradient has Inf: {torch.isinf(pred_extreme3.grad).any()}")
        if torch.isnan(pred_extreme3.grad).any():
            print(f"  ❌ NaN gradients detected with extreme values!")
except Exception as e:
    print(f"  Failed: {e}")

# Test 4: Test the actual scenario from training
print("\n" + "=" * 80)
print("SIMULATING ACTUAL TRAINING SCENARIO")
print("=" * 80)

# From your log: dfl_loss value is 11.101562
# This suggests the loss is very high but not NaN yet
# The NaN happens during backward pass

# Create a scenario that produces similar loss
pred_training = torch.randn(303, 64) * 3  # Random init, scaled
pred_training.requires_grad = True
target_training = torch.rand(303, 4) * 15  # Random targets

print(f"\nPrediction stats:")
print(f"  Shape: {pred_training.shape}")
print(f"  Range: [{pred_training.min():.4f}, {pred_training.max():.4f}]")

print(f"\nTarget stats:")
print(f"  Shape: {target_training.shape}")
print(f"  Range: [{target_training.min():.4f}, {target_training.max():.4f}]")

try:
    loss_training = dfl_loss_fn(pred_training, target_training)
    print(f"\nLoss: {loss_training.item():.6f}")
    
    loss_training.backward()
    
    if pred_training.grad is not None:
        grad_has_nan = torch.isnan(pred_training.grad).any()
        grad_has_inf = torch.isinf(pred_training.grad).any()
        
        print(f"Gradient has NaN: {grad_has_nan}")
        print(f"Gradient has Inf: {grad_has_inf}")
        
        if not grad_has_nan:
            print("✅ No NaN gradients in simulated training scenario")
        else:
            print("❌ NaN gradients in simulated training scenario!")
            
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
