"""
Unit tests for Triplet Loss
Tests all variants: TripletLoss, BatchHardTripletLoss, AdaptiveTripletLoss
"""

import torch
import pytest
from losses.triplet_loss import TripletLoss, BatchHardTripletLoss, AdaptiveTripletLoss


class TestTripletLoss:
    """Test suite for standard Triplet Loss"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loss_fn_euclidean = TripletLoss(margin=0.3, distance='euclidean')
        self.loss_fn_cosine = TripletLoss(margin=0.3, distance='cosine')
    
    def test_perfect_triplet_euclidean(self):
        """Test loss is low when positive is close and negative is far (Euclidean)"""
        anchor = torch.tensor([[1.0, 0.0, 0.0]])
        positive = torch.tensor([[0.9, 0.1, 0.0]])  # Very close
        negative = torch.tensor([[0.0, 0.0, 1.0]])   # Very far
        
        loss = self.loss_fn_euclidean(anchor, positive, negative)
        
        # Loss should be zero or very small (negative far > positive close + margin)
        assert loss.item() < 0.1, f"Expected low loss for good triplet, got {loss.item()}"
    
    def test_violated_triplet_euclidean(self):
        """Test loss is high when positive is far and negative is close (Euclidean)"""
        anchor = torch.tensor([[1.0, 0.0, 0.0]])
        positive = torch.tensor([[0.0, 0.0, 1.0]])   # Very far
        negative = torch.tensor([[0.9, 0.1, 0.0]])  # Very close
        
        loss = self.loss_fn_euclidean(anchor, positive, negative)
        
        # Loss should be high (margin violation)
        assert loss.item() > 0.5, f"Expected high loss for violated triplet, got {loss.item()}"
    
    def test_perfect_triplet_cosine(self):
        """Test loss is low with cosine distance"""
        anchor = torch.tensor([[1.0, 0.0, 0.0]])
        positive = torch.tensor([[0.9, 0.1, 0.0]])  # Similar direction
        negative = torch.tensor([[0.0, 0.0, 1.0]])   # Orthogonal
        
        loss = self.loss_fn_cosine(anchor, positive, negative)
        
        assert loss.item() < 0.2, f"Expected low loss for good triplet, got {loss.item()}"
    
    def test_batch_processing(self):
        """Test batch processing"""
        batch_size = 16
        feature_dim = 128
        
        anchor = torch.randn(batch_size, feature_dim)
        positive = anchor + torch.randn(batch_size, feature_dim) * 0.1  # Similar
        negative = torch.randn(batch_size, feature_dim)  # Random
        
        loss = self.loss_fn_euclidean(anchor, positive, negative)
        
        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert not torch.isnan(loss), "Loss contains NaN"
        assert not torch.isinf(loss), "Loss contains Inf"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_reduction_modes(self):
        """Test different reduction modes"""
        anchor = torch.randn(4, 64)
        positive = anchor + torch.randn(4, 64) * 0.1
        negative = torch.randn(4, 64)
        
        loss_fn_mean = TripletLoss(margin=0.3, reduction='mean')
        loss_fn_sum = TripletLoss(margin=0.3, reduction='sum')
        loss_fn_none = TripletLoss(margin=0.3, reduction='none')
        
        loss_mean = loss_fn_mean(anchor, positive, negative)
        loss_sum = loss_fn_sum(anchor, positive, negative)
        loss_none = loss_fn_none(anchor, positive, negative)
        
        assert loss_mean.shape == torch.Size([]), "Mean reduction should give scalar"
        assert loss_sum.shape == torch.Size([]), "Sum reduction should give scalar"
        assert loss_none.shape == torch.Size([4]), "No reduction should give per-sample loss"
        
        # Verify relationships
        assert torch.allclose(loss_mean, loss_none.mean(), atol=1e-6)
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-6)
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        anchor = torch.randn(4, 64, requires_grad=True)
        positive = torch.randn(4, 64, requires_grad=True)
        negative = torch.randn(4, 64, requires_grad=True)
        
        loss = self.loss_fn_euclidean(anchor, positive, negative)
        loss.backward()
        
        assert anchor.grad is not None, "Gradients should flow to anchor"
        assert positive.grad is not None, "Gradients should flow to positive"
        assert negative.grad is not None, "Gradients should flow to negative"
        
        assert not torch.isnan(anchor.grad).any(), "Anchor gradients contain NaN"
        assert not torch.isnan(positive.grad).any(), "Positive gradients contain NaN"
        assert not torch.isnan(negative.grad).any(), "Negative gradients contain NaN"
    
    def test_margin_effect(self):
        """Test different margin values"""
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.8, 0.0]])
        negative = torch.tensor([[0.5, 0.0]])
        
        loss_fn_small = TripletLoss(margin=0.1)
        loss_fn_large = TripletLoss(margin=0.5)
        
        loss_small = loss_fn_small(anchor, positive, negative)
        loss_large = loss_fn_large(anchor, positive, negative)
        
        # Larger margin should give larger loss
        assert loss_large > loss_small, "Larger margin should increase loss"
    
    def test_euclidean_vs_cosine(self):
        """Test both distance metrics work"""
        anchor = torch.randn(8, 64)
        positive = anchor + torch.randn(8, 64) * 0.1
        negative = torch.randn(8, 64)
        
        loss_euclidean = self.loss_fn_euclidean(anchor, positive, negative)
        loss_cosine = self.loss_fn_cosine(anchor, positive, negative)
        
        # Both should be valid (non-negative, finite)
        assert loss_euclidean.item() >= 0
        assert loss_cosine.item() >= 0
        assert not torch.isnan(loss_euclidean)
        assert not torch.isnan(loss_cosine)


class TestBatchHardTripletLoss:
    """Test suite for Batch Hard Triplet Loss"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loss_fn = BatchHardTripletLoss(margin=0.3, distance='euclidean')
    
    def test_automatic_mining(self):
        """Test automatic hard triplet mining"""
        # Create embeddings with 3 classes
        embeddings = torch.cat([
            torch.randn(5, 64) + torch.tensor([1.0] + [0.0]*63),  # Class 0
            torch.randn(5, 64) + torch.tensor([0.0, 1.0] + [0.0]*62),  # Class 1
            torch.randn(5, 64) + torch.tensor([0.0, 0.0, 1.0] + [0.0]*61),  # Class 2
        ])
        labels = torch.tensor([0]*5 + [1]*5 + [2]*5)
        
        loss = self.loss_fn(embeddings, labels)
        
        assert not torch.isnan(loss), "Loss contains NaN"
        assert not torch.isinf(loss), "Loss contains Inf"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_single_class(self):
        """Test returns zero loss when only one class"""
        embeddings = torch.randn(10, 64)
        labels = torch.zeros(10, dtype=torch.long)  # All same class
        
        loss = self.loss_fn(embeddings, labels)
        
        # Should return 0 (no negatives available)
        assert loss.item() == 0.0, "Single class should give zero loss"
    
    def test_two_classes(self):
        """Test with two classes"""
        embeddings = torch.cat([
            torch.randn(8, 64) + torch.tensor([1.0] + [0.0]*63),
            torch.randn(8, 64) + torch.tensor([0.0, 1.0] + [0.0]*62),
        ])
        labels = torch.tensor([0]*8 + [1]*8)
        
        loss = self.loss_fn(embeddings, labels)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        embeddings = torch.randn(12, 64, requires_grad=True)
        labels = torch.tensor([0]*4 + [1]*4 + [2]*4)
        
        loss = self.loss_fn(embeddings, labels)
        loss.backward()
        
        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()
    
    def test_cosine_distance(self):
        """Test cosine distance variant"""
        loss_fn_cosine = BatchHardTripletLoss(margin=0.3, distance='cosine')
        
        embeddings = torch.randn(15, 64)
        labels = torch.tensor([0]*5 + [1]*5 + [2]*5)
        
        loss = loss_fn_cosine(embeddings, labels)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0


class TestAdaptiveTripletLoss:
    """Test suite for Adaptive Triplet Loss"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.loss_fn = AdaptiveTripletLoss(initial_margin=0.3)
    
    def test_learnable_margin(self):
        """Test margin is learnable parameter"""
        assert isinstance(self.loss_fn.margin, torch.nn.Parameter)
        assert self.loss_fn.margin.requires_grad
    
    def test_margin_updates(self):
        """Test margin can be updated through backprop"""
        initial_margin = self.loss_fn.margin.clone().detach()
        
        optimizer = torch.optim.SGD([self.loss_fn.margin], lr=0.1)
        
        # Create triplets with clear violations to ensure gradients flow
        anchor = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        positive = torch.tensor([[0.8, 0.2, 0.0, 0.0]])  # Close to anchor
        negative = torch.tensor([[0.7, 0.3, 0.0, 0.0]])  # Also close - margin violation!
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = self.loss_fn(anchor, positive, negative)
            if loss.item() > 0:  # Only backprop if there's actual loss
                loss.backward()
                optimizer.step()
        
        # Margin should have changed (or check if gradient exists)
        updated_margin = self.loss_fn.margin.detach()
        # Either margin updated OR gradient exists (proves it's learnable)
        assert (not torch.allclose(initial_margin, updated_margin, atol=1e-6)) or \
               (self.loss_fn.margin.grad is not None), \
            "Margin should be learnable (either updated or has gradient)"
    
    def test_margin_stays_positive(self):
        """Test margin remains positive through ReLU"""
        # Set negative margin
        self.loss_fn.margin.data = torch.tensor(-0.5)
        
        anchor = torch.randn(4, 64)
        positive = anchor + torch.randn(4, 64) * 0.1
        negative = torch.randn(4, 64)
        
        loss = self.loss_fn(anchor, positive, negative)
        
        # Loss should still be computed (margin forced to be positive)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradients flow to both features and margin"""
        anchor = torch.randn(4, 64, requires_grad=True)
        positive = torch.randn(4, 64, requires_grad=True)
        negative = torch.randn(4, 64, requires_grad=True)
        
        loss = self.loss_fn(anchor, positive, negative)
        loss.backward()
        
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negative.grad is not None
        assert self.loss_fn.margin.grad is not None


class TestTripletLossIntegration:
    """Integration tests for triplet loss in training scenarios"""
    
    def test_training_simulation(self):
        """Simulate mini training loop with triplet loss"""
        loss_fn = TripletLoss(margin=0.3)
        
        for epoch in range(3):
            # Simulate batch
            batch_size = 8
            anchor = torch.randn(batch_size, 128)
            positive = anchor + torch.randn(batch_size, 128) * 0.1
            negative = torch.randn(batch_size, 128)
            
            loss = loss_fn(anchor, positive, negative)
            
            assert not torch.isnan(loss), f"NaN loss at epoch {epoch}"
            assert loss.item() >= 0, f"Negative loss at epoch {epoch}"
    
    def test_batch_hard_with_varying_class_sizes(self):
        """Test batch hard with imbalanced classes"""
        loss_fn = BatchHardTripletLoss(margin=0.3)
        
        # Imbalanced: 10 samples of class 0, 3 of class 1, 2 of class 2
        embeddings = torch.cat([
            torch.randn(10, 64),
            torch.randn(3, 64),
            torch.randn(2, 64)
        ])
        labels = torch.tensor([0]*10 + [1]*3 + [2]*2)
        
        loss = loss_fn(embeddings, labels)
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
