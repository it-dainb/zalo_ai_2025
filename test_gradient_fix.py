"""
Quick test to verify DFL loss gradient stability
"""
import torch
from src.losses.dfl_loss import DFLoss

def test_dfl_gradient_stability():
    """Test DFL loss with high loss values (simulating current issue)"""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create DFL loss
    dfl = DFLoss(reg_max=16).to(device)
    
    # Simulate worst-case: random predictions (very wrong)
    batch_size = 339  # From error log
    pred_dist = torch.randn(batch_size, 64, device=device, requires_grad=True)
    
    # Random targets (in valid range)
    target = torch.rand(batch_size, 4, device=device) * 15.0
    
    # Forward pass
    loss = dfl(pred_dist, target)
    
    print(f"Loss value: {loss.item():.4f}")
    assert loss.item() <= 15.0, f"Loss should be clamped at 15.0, got {loss.item()}"
    
    # Backward pass with mixed precision
    scaler = torch.cuda.amp.GradScaler()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Check gradients (with None check)
    assert pred_dist.grad is not None, "Gradient is None!"
    has_nan = torch.isnan(pred_dist.grad).any() or torch.isinf(pred_dist.grad).any()
    grad_max = pred_dist.grad.abs().max().item()
    grad_mean = pred_dist.grad.abs().mean().item()
    
    print(f"Gradient stats:")
    print(f"  Has NaN/Inf: {has_nan}")
    print(f"  Max: {grad_max:.4e}")
    print(f"  Mean: {grad_mean:.4e}")
    
    # Unscale gradients
    scaler.unscale_(torch.optim.SGD([pred_dist], lr=0.01))
    
    # Clip gradients
    clipped_norm = torch.nn.utils.clip_grad_norm_([pred_dist], max_norm=5.0)
    print(f"  Clipped norm: {clipped_norm:.4e}")
    
    # After clipping, should be stable
    assert pred_dist.grad is not None, "Gradient is None after clipping!"
    has_nan_after = torch.isnan(pred_dist.grad).any() or torch.isinf(pred_dist.grad).any()
    print(f"  Has NaN/Inf after clip: {has_nan_after}")
    
    if has_nan_after:
        print("❌ FAIL: Gradients still have NaN/Inf after clipping!")
        return False
    
    print("✅ PASS: Gradients are stable!")
    return True

if __name__ == '__main__':
    print("Testing DFL loss gradient stability...")
    print("=" * 60)
    success = test_dfl_gradient_stability()
    print("=" * 60)
    if success:
        print("\n✅ All tests passed! DFL loss should be stable now.")
    else:
        print("\n❌ Tests failed! Further investigation needed.")
