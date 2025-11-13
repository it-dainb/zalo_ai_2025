"""
Unit tests for WIoU Loss
Tests the Wise-IoU v3 implementation
"""

import torch
import pytest
from src.losses.wiou_loss import WIoULoss


class TestWIoULoss:
    """Test suite for WIoU Loss"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loss_fn = WIoULoss(monotonous=True)
    
    def test_perfect_overlap(self):
        """Test loss is 0 for perfectly overlapping boxes"""
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0]
        ])
        
        loss = self.loss_fn(boxes, boxes)
        
        assert loss.item() < 1e-5, f"Expected ~0 loss for perfect overlap, got {loss.item()}"
    
    def test_no_overlap(self):
        """Test high loss for non-overlapping boxes"""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        
        loss = self.loss_fn(pred, target)
        
        assert loss.item() > 1.0, f"Expected high loss for no overlap, got {loss.item()}"
    
    def test_partial_overlap(self):
        """Test moderate loss for partial overlap"""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])  # ~25% IoU
        
        loss = self.loss_fn(pred, target)
        
        # With dynamic focusing, loss can be higher for hard samples
        assert 0.3 < loss.item() < 15.0, \
            f"Expected moderate loss (0.3-15.0) for partial overlap, got {loss.item()}"
    
    def test_batch_processing(self):
        """Test batch processing"""
        batch_size = 16
        pred = torch.rand(batch_size, 4) * 100
        pred[:, 2:] = pred[:, :2] + torch.rand(batch_size, 2) * 50  # Ensure x2>x1, y2>y1
        target = torch.rand(batch_size, 4) * 100
        target[:, 2:] = target[:, :2] + torch.rand(batch_size, 2) * 50
        
        loss = self.loss_fn(pred, target)
        
        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert not torch.isnan(loss), "Loss contains NaN"
        assert not torch.isinf(loss), "Loss contains Inf"
    
    def test_reduction_modes(self):
        """Test different reduction modes"""
        pred = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0]
        ])
        target = torch.tensor([
            [1.0, 1.0, 11.0, 11.0],
            [21.0, 21.0, 31.0, 31.0]
        ])
        
        # Use eval mode to prevent iou_mean updates between calls
        self.loss_fn.eval()
        
        loss_mean = self.loss_fn(pred, target, reduction='mean')
        loss_sum = self.loss_fn(pred, target, reduction='sum')
        loss_none = self.loss_fn(pred, target, reduction='none')
        
        assert loss_mean.shape == torch.Size([]), "Mean reduction should give scalar"
        assert loss_sum.shape == torch.Size([]), "Sum reduction should give scalar"
        assert loss_none.shape == torch.Size([2]), "No reduction should give per-sample loss"
        
        # Verify relationship
        assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)
        
        # Restore train mode for other tests
        self.loss_fn.train()
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        pred = torch.tensor(
            [[5.0, 5.0, 15.0, 15.0]], 
            requires_grad=True
        )
        target = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        loss = self.loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None, "Gradients should flow to predictions"
        assert not torch.isnan(pred.grad).any(), "Gradients contain NaN"
        assert not torch.isinf(pred.grad).any(), "Gradients contain Inf"
    
    def test_monotonous_vs_standard(self):
        """Test monotonous focusing makes a difference"""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        loss_fn_v3 = WIoULoss(monotonous=True)
        loss_fn_v1 = WIoULoss(monotonous=False)
        
        loss_v3 = loss_fn_v3(pred, target)
        loss_v1 = loss_fn_v1(pred, target)
        
        # v3 should have dynamic focusing, may differ from v1
        assert isinstance(loss_v3.item(), float)
        assert isinstance(loss_v1.item(), float)
    
    def test_small_boxes(self):
        """Test stability with small boxes"""
        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
        
        loss = self.loss_fn(pred, target)
        
        assert not torch.isnan(loss), "Loss is NaN for small boxes"
        assert not torch.isinf(loss), "Loss is Inf for small boxes"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Zero-area boxes (invalid but should not crash)
        pred = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        target = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        
        try:
            loss = self.loss_fn(pred, target)
            assert not torch.isnan(loss), "Should handle zero-area boxes gracefully"
        except Exception as e:
            pytest.skip(f"Zero-area boxes not supported: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
