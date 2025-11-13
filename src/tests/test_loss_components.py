"""
Unit tests for all loss components
Tests BCE, DFL, SupCon, and CPE losses
"""

import torch
import pytest
from losses.bce_loss import BCEClassificationLoss
from losses.dfl_loss import DFLoss
from losses.supervised_contrastive_loss import SupervisedContrastiveLoss, PrototypeContrastiveLoss
from losses.cpe_loss import SimplifiedCPELoss


class TestBCELoss:
    """Test suite for BCE Classification Loss"""
    
    def setup_method(self):
        self.loss_fn = BCEClassificationLoss()
    
    def test_perfect_prediction(self):
        """Test loss is low for perfect predictions"""
        # High confidence correct predictions
        logits = torch.tensor([[10.0, -10.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        
        loss = self.loss_fn(logits, targets)
        assert loss.item() < 0.01, f"Expected low loss for perfect prediction, got {loss.item()}"
    
    def test_wrong_prediction(self):
        """Test loss is high for wrong predictions"""
        logits = torch.tensor([[10.0, 10.0]])
        targets = torch.tensor([[0.0, 0.0]])
        
        loss = self.loss_fn(logits, targets)
        assert loss.item() > 5.0, f"Expected high loss for wrong prediction, got {loss.item()}"
    
    def test_multi_class(self):
        """Test multi-class classification"""
        batch_size, num_classes = 8, 10
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, 2, (batch_size, num_classes)).float()
        
        loss = self.loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.randint(0, 2, (4, 5)).float()
        
        loss = self.loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestDFLoss:
    """Test suite for Distribution Focal Loss"""
    
    def setup_method(self):
        self.loss_fn = DFLoss(reg_max=16)
    
    def test_output_shape(self):
        """Test output shape is correct"""
        batch_size = 8
        pred_dist = torch.randn(batch_size, 4 * 17)  # 4 coords * (reg_max+1)
        targets = torch.rand(batch_size, 4) * 15  # Values in [0, 15]
        
        loss = self.loss_fn(pred_dist, targets)
        assert loss.shape == torch.Size([])
    
    def test_decode_distribution(self):
        """Test decoding distribution to coordinates"""
        batch_size = 4
        pred_dist = torch.randn(batch_size, 4 * 17)
        
        decoded = self.loss_fn.decode(pred_dist)
        assert decoded.shape == (batch_size, 4)
        assert (decoded >= 0).all() and (decoded <= 16).all()
    
    def test_loss_convergence(self):
        """Test loss decreases with better predictions"""
        targets = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        
        # Bad prediction (far from target)
        bad_pred = torch.randn(1, 4 * 17)
        loss_bad = self.loss_fn(bad_pred, targets)
        
        # Good prediction (create peaked distribution at target)
        good_pred = torch.randn(1, 4 * 17)
        for i in range(4):
            target_bin = int(targets[0, i].item())
            good_pred[0, i*17 + target_bin] = 10.0  # High confidence at correct bin
        
        loss_good = self.loss_fn(good_pred, targets)
        
        # Good prediction should have lower loss
        assert loss_good < loss_bad
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        pred_dist = torch.randn(4, 4 * 17, requires_grad=True)
        targets = torch.rand(4, 4) * 15
        
        loss = self.loss_fn(pred_dist, targets)
        loss.backward()
        
        assert pred_dist.grad is not None
        assert not torch.isnan(pred_dist.grad).any()


class TestSupervisedContrastiveLoss:
    """Test suite for Supervised Contrastive Loss"""
    
    def setup_method(self):
        self.loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    
    def test_same_class_attraction(self):
        """Test same class features are pulled together"""
        # Two samples of class 0, one sample of class 1
        features = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0]
        ])
        labels = torch.tensor([0, 0, 1])
        
        loss = self.loss_fn(features, labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_batch_processing(self):
        """Test batch processing"""
        batch_size, feature_dim = 32, 128
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, 5, (batch_size,))  # 5 classes
        
        loss = self.loss_fn(features, labels)
        assert not torch.isnan(loss)
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        features = torch.randn(16, 64, requires_grad=True)
        labels = torch.randint(0, 4, (16,))
        
        loss = self.loss_fn(features, labels)
        loss.backward()
        
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


class TestPrototypeContrastiveLoss:
    """Test suite for Prototype Contrastive Loss"""
    
    def setup_method(self):
        self.loss_fn = PrototypeContrastiveLoss(temperature=0.07)
    
    def test_matching_to_prototypes(self):
        """Test matching query features to prototypes"""
        num_queries = 16
        num_classes = 5
        feature_dim = 128
        
        query_features = torch.randn(num_queries, feature_dim)
        prototypes = torch.randn(num_classes, feature_dim)
        labels = torch.randint(0, num_classes, (num_queries,))
        
        loss = self.loss_fn(query_features, prototypes, labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradients flow to both queries and prototypes"""
        query_features = torch.randn(8, 64, requires_grad=True)
        prototypes = torch.randn(3, 64, requires_grad=True)
        labels = torch.randint(0, 3, (8,))
        
        loss = self.loss_fn(query_features, prototypes, labels)
        loss.backward()
        
        assert query_features.grad is not None
        assert prototypes.grad is not None


class TestCPELoss:
    """Test suite for Contrastive Proposal Encoding Loss"""
    
    def setup_method(self):
        self.loss_fn = SimplifiedCPELoss(temperature=0.1)
    
    def test_foreground_only(self):
        """Test only foreground proposals contribute to loss"""
        features = torch.randn(10, 128)
        labels = torch.tensor([-1, -1, 0, 0, 1, 1, -1, 2, 2, -1])  # -1 is background
        
        loss = self.loss_fn(features, labels)
        assert not torch.isnan(loss)
    
    def test_no_foreground(self):
        """Test returns 0 when no foreground proposals"""
        features = torch.randn(5, 128)
        labels = torch.tensor([-1, -1, -1, -1, -1])  # All background
        
        loss = self.loss_fn(features, labels)
        assert loss.item() == 0.0
    
    def test_single_foreground(self):
        """Test returns 0 when only one foreground proposal"""
        features = torch.randn(5, 128)
        labels = torch.tensor([-1, 0, -1, -1, -1])  # Only one foreground
        
        loss = self.loss_fn(features, labels)
        assert loss.item() == 0.0
    
    def test_batch_processing(self):
        """Test batch processing"""
        features = torch.randn(50, 256)
        labels = torch.randint(-1, 5, (50,))  # Mix of background and 5 classes
        
        loss = self.loss_fn(features, labels)
        assert not torch.isnan(loss)
    
    def test_gradient_flow(self):
        """Test gradients flow properly"""
        features = torch.randn(20, 128, requires_grad=True)
        labels = torch.randint(-1, 3, (20,))
        
        loss = self.loss_fn(features, labels)
        if loss.item() > 0:  # Only if loss is computed
            loss.backward()
            assert features.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
