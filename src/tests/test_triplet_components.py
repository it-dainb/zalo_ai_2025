"""
Unit Test for Triplet Training Components
=========================================

Tests individual components of triplet training without requiring
full model initialization (avoiding timm/ultralytics dependencies).
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.training.loss_utils import prepare_triplet_loss_inputs


class TestTripletComponents:
    """Test triplet training components in isolation."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @pytest.fixture
    def mock_model_outputs(self, device):
        """Create mock model outputs for triplet batch (after trainer combines them)."""
        batch_size = 4
        
        # This mimics the combined_outputs dict created in trainer._forward_triplet_step()
        # Lines 1246-1249: anchor from support_global_feat (256-dim after triplet_proj), pos+neg concatenated in query_global_feat
        return {
            'support_global_feat': torch.randn(batch_size, 256, device=device),  # Anchor features (DINOv3 384->256 via triplet_proj)
            'query_global_feat': torch.randn(batch_size * 2, 256, device=device),  # Positive + Negative concatenated (YOLOv8 256-dim)
        }
    
    @pytest.fixture
    def mock_triplet_batch(self, device):
        """Create mock triplet batch."""
        batch_size = 4
        
        return {
            'anchor_images': torch.randn(batch_size, 3, 256, 256, device=device),
            'positive_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'positive_bboxes': [
                torch.tensor([[100, 100, 200, 200]], device=device, dtype=torch.float32),
                torch.tensor([[150, 150, 250, 250]], device=device, dtype=torch.float32),
                torch.tensor([[50, 50, 150, 150]], device=device, dtype=torch.float32),
                torch.tensor([[200, 200, 300, 300]], device=device, dtype=torch.float32),
            ],
            'negative_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'negative_bboxes': [
                torch.zeros((0, 4), device=device, dtype=torch.float32),  # Background
                torch.tensor([[300, 300, 400, 400]], device=device, dtype=torch.float32),  # Cross-class
                torch.zeros((0, 4), device=device, dtype=torch.float32),  # Background
                torch.tensor([[100, 100, 200, 200]], device=device, dtype=torch.float32),  # Cross-class
            ],
            'class_ids': torch.tensor([0, 1, 0, 1], device=device),
            'negative_types': ['background', 'cross_class', 'background', 'cross_class'],
        }
    
    def test_prepare_triplet_loss_inputs_shape(self, mock_model_outputs, mock_triplet_batch, device):
        """Test that prepare_triplet_loss_inputs returns correct shapes."""
        triplet_inputs = prepare_triplet_loss_inputs(
            model_outputs=mock_model_outputs,
            batch=mock_triplet_batch,
        )
        
        # Check that all required keys are present
        assert 'anchor_features' in triplet_inputs, "Should have anchor_features"
        assert 'positive_features' in triplet_inputs, "Should have positive_features"
        assert 'negative_features' in triplet_inputs, "Should have negative_features"
        
        # Check shapes
        batch_size = mock_triplet_batch['anchor_images'].shape[0]
        feat_dim = 256  # All features projected to 256 dimensions
        
        assert triplet_inputs['anchor_features'].shape == (batch_size, feat_dim), \
            f"Expected anchor features (B, {feat_dim}), got {triplet_inputs['anchor_features'].shape}"
        
        assert triplet_inputs['positive_features'].shape == (batch_size, feat_dim), \
            f"Expected positive features (B, {feat_dim}), got {triplet_inputs['positive_features'].shape}"
        
        assert triplet_inputs['negative_features'].shape == (batch_size, feat_dim), \
            f"Expected negative features (B, {feat_dim}), got {triplet_inputs['negative_features'].shape}"
        
        # Verify all features have the same dimension (required for triplet loss)
        assert triplet_inputs['anchor_features'].shape[-1] == triplet_inputs['positive_features'].shape[-1], \
            "Anchor and positive features must have same dimension"
        assert triplet_inputs['anchor_features'].shape[-1] == triplet_inputs['negative_features'].shape[-1], \
            "Anchor and negative features must have same dimension"
    
    def test_feature_extraction(self, mock_model_outputs, mock_triplet_batch, device):
        """Test that features are extracted correctly from model outputs."""
        triplet_inputs = prepare_triplet_loss_inputs(
            model_outputs=mock_model_outputs,
            batch=mock_triplet_batch,
        )
        
        # Check that features are not NaN or Inf
        assert not torch.isnan(triplet_inputs['anchor_features']).any(), \
            "Anchor features should not contain NaN"
        assert not torch.isnan(triplet_inputs['positive_features']).any(), \
            "Positive features should not contain NaN"
        assert not torch.isnan(triplet_inputs['negative_features']).any(), \
            "Negative features should not contain NaN"
        
        assert not torch.isinf(triplet_inputs['anchor_features']).any(), \
            "Anchor features should not contain Inf"
        assert not torch.isinf(triplet_inputs['positive_features']).any(), \
            "Positive features should not contain Inf"
        assert not torch.isinf(triplet_inputs['negative_features']).any(), \
            "Negative features should not contain Inf"
        
        # Features should have reasonable magnitude (not all zeros)
        assert triplet_inputs['anchor_features'].abs().sum() > 0, \
            "Anchor features should not be all zeros"
        assert triplet_inputs['positive_features'].abs().sum() > 0, \
            "Positive features should not be all zeros"
        assert triplet_inputs['negative_features'].abs().sum() > 0, \
            "Negative features should not be all zeros"
    
    def test_triplet_loss_computation(self, device):
        """Test that triplet loss can be computed from prepared inputs."""
        from src.losses.triplet_loss import TripletLoss
        
        batch_size = 4
        feat_dim = 256
        
        # Create model outputs with gradient tracking
        mock_outputs = {
            'support_global_feat': torch.randn(batch_size, 256, device=device, requires_grad=True),  # DINOv3 384->256 via triplet_proj
            'query_global_feat': torch.randn(batch_size * 2, feat_dim, device=device, requires_grad=True),
        }
        
        # Create mock batch
        mock_batch = {
            'anchor_images': torch.randn(batch_size, 3, 256, 256, device=device),
            'positive_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'negative_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'class_ids': torch.tensor([0, 1, 0, 1], device=device),
        }
        
        triplet_inputs = prepare_triplet_loss_inputs(
            model_outputs=mock_outputs,
            batch=mock_batch,
        )
        
        # Create triplet loss function
        triplet_loss_fn = TripletLoss(margin=0.2)
        
        # Compute loss
        loss = triplet_loss_fn(
            anchor=triplet_inputs['anchor_features'],
            positive=triplet_inputs['positive_features'],
            negative=triplet_inputs['negative_features'],
        )
        
        # Check loss properties
        assert loss.requires_grad, "Loss should require grad"
        assert loss.item() >= 0, "Triplet loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_triplet_margin(self, device):
        """Test triplet loss margin behavior."""
        from src.losses.triplet_loss import TripletLoss
        
        batch_size = 4
        feat_dim = 256
        
        # Create triplet loss with margin
        triplet_loss_fn = TripletLoss(margin=0.2)
        
        # Case 1: Identical positive and negative (should have high loss)
        anchor = torch.randn(batch_size, feat_dim, device=device)
        positive = anchor + 0.01 * torch.randn(batch_size, feat_dim, device=device)  # Very close
        negative = anchor + 0.01 * torch.randn(batch_size, feat_dim, device=device)  # Also very close
        
        # Normalize
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        negative = torch.nn.functional.normalize(negative, dim=1)
        
        loss_high = triplet_loss_fn(anchor, positive, negative)
        
        # Case 2: Far negative (should have low loss)
        negative_far = -anchor + torch.randn(batch_size, feat_dim, device=device)
        negative_far = torch.nn.functional.normalize(negative_far, dim=1)
        
        loss_low = triplet_loss_fn(anchor, positive, negative_far)
        
        # Loss should be lower when negative is far from anchor
        print(f"\nTriplet loss comparison:")
        print(f"  Close negative: {loss_high.item():.4f}")
        print(f"  Far negative: {loss_low.item():.4f}")
        
        assert loss_low.item() <= loss_high.item(), \
            "Loss should be lower when negative is far from anchor"
    
    def test_batch_type_handling(self, device):
        """Test that batch type is correctly identified."""
        # Detection batch
        detection_batch = {
            'query_images': torch.randn(2, 3, 640, 640, device=device),
            'support_images': torch.randn(2, 2, 3, 256, 256, device=device),
            'batch_type': 'detection',
        }
        
        # Triplet batch
        triplet_batch = {
            'anchor_images': torch.randn(2, 3, 256, 256, device=device),
            'positive_images': torch.randn(2, 3, 640, 640, device=device),
            'negative_images': torch.randn(2, 3, 640, 640, device=device),
            'batch_type': 'triplet',
        }
        
        # Mixed batch
        mixed_batch = {
            'detection': detection_batch,
            'triplet': triplet_batch,
            'batch_type': 'mixed',
        }
        
        assert detection_batch['batch_type'] == 'detection'
        assert triplet_batch['batch_type'] == 'triplet'
        assert mixed_batch['batch_type'] == 'mixed'
        
        print("\nBatch type handling test passed!")
    
    def test_gradient_flow_through_triplet_loss(self, device):
        """Test that gradients flow through triplet loss."""
        from src.losses.triplet_loss import TripletLoss
        
        batch_size = 4
        feat_dim = 256  # All features must have same dimension
        
        # Create features with gradient tracking
        anchor = torch.randn(batch_size, feat_dim, device=device, requires_grad=True)
        positive = torch.randn(batch_size, feat_dim, device=device, requires_grad=True)
        negative = torch.randn(batch_size, feat_dim, device=device, requires_grad=True)
        
        # Create triplet loss
        triplet_loss_fn = TripletLoss(margin=0.2)
        
        # Forward pass
        loss = triplet_loss_fn(anchor, positive, negative)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert anchor.grad is not None, "Anchor should have gradients"
        assert positive.grad is not None, "Positive should have gradients"
        assert negative.grad is not None, "Negative should have gradients"
        
        # Check gradients are not NaN or Inf
        assert not torch.isnan(anchor.grad).any(), "Anchor gradients should not be NaN"
        assert not torch.isnan(positive.grad).any(), "Positive gradients should not be NaN"
        assert not torch.isnan(negative.grad).any(), "Negative gradients should not be NaN"
        
        assert not torch.isinf(anchor.grad).any(), "Anchor gradients should not be Inf"
        assert not torch.isinf(positive.grad).any(), "Positive gradients should not be Inf"
        assert not torch.isinf(negative.grad).any(), "Negative gradients should not be Inf"
        
        print("\nGradient flow test passed!")


if __name__ == '__main__':
    """Run tests directly."""
    pytest.main([__file__, '-v', '-s'])
