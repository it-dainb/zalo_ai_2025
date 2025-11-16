"""
End-to-End tests for Combined Loss
Tests the full loss integration and training stages
"""

import torch
import pytest
from src.losses.combined_loss import ReferenceBasedDetectionLoss, create_loss_fn


class TestCombinedLoss:
    """Test suite for Combined Reference-Based Detection Loss"""
    
    def test_stage1_weights(self):
        """Test Stage 1 (base pre-training) uses correct weights"""
        loss_fn = create_loss_fn(stage=1)
        
        assert loss_fn.weights['bbox'] == 7.5
        assert loss_fn.weights['cls'] == 0.5
        assert loss_fn.weights['dfl'] == 1.5
        assert loss_fn.weights['supcon'] == 0.0  # No contrastive in stage 1
        assert loss_fn.weights['cpe'] == 0.0
    
    def test_stage2_weights(self):
        """Test Stage 2 (few-shot meta) enables contrastive losses"""
        loss_fn = create_loss_fn(stage=2)
        
        assert loss_fn.weights['bbox'] == 7.5
        assert loss_fn.weights['cls'] == 0.5
        assert loss_fn.weights['dfl'] == 1.5
        assert loss_fn.weights['supcon'] == 1.0  # Full contrastive
        assert loss_fn.weights['cpe'] == 0.5
    
    def test_stage3_weights(self):
        """Test Stage 3 (fine-tuning) reduces contrastive weights and adds triplet"""
        loss_fn = create_loss_fn(stage=3)
        
        assert loss_fn.weights['bbox'] == 7.5
        assert loss_fn.weights['cls'] == 0.5
        assert loss_fn.weights['dfl'] == 1.5
        assert loss_fn.weights['supcon'] == 0.5  # Reduced to 0.5
        assert loss_fn.weights['cpe'] == 0.3  # Reduced to 0.3
        assert loss_fn.weights['triplet'] == 0.2  # Added in stage 3
    
    def test_stage1_forward_basic(self):
        """Test Stage 1 forward pass (detection only)"""
        loss_fn = create_loss_fn(stage=1)
        
        batch_size = 8
        num_classes = 10
        reg_max = 16
        
        # Create dummy predictions
        pred_bboxes = torch.rand(batch_size, 4) * 100
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.randn(batch_size, 4 * reg_max)
        
        # Create dummy targets
        target_bboxes = torch.rand(batch_size, 4) * 100
        target_bboxes[:, 2:] = target_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        target_cls = torch.randint(0, 2, (batch_size, num_classes)).float()
        target_dfl = torch.rand(batch_size, 4) * 15
        
        # Forward pass
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl
        )
        
        # Check outputs
        assert 'total_loss' in losses
        assert 'bbox_loss' in losses
        assert 'cls_loss' in losses
        assert 'dfl_loss' in losses
        assert 'supcon_loss' in losses
        assert 'cpe_loss' in losses
        
        # Check loss values are valid
        assert not torch.isnan(losses['total_loss'])
        assert losses['total_loss'].item() > 0
        
        # Check contrastive losses are zero in stage 1
        assert losses['supcon_loss'].item() == 0.0
        assert losses['cpe_loss'].item() == 0.0
    
    def test_stage2_forward_with_contrastive(self):
        """Test Stage 2 forward pass (with contrastive learning)"""
        loss_fn = create_loss_fn(stage=2)
        
        batch_size = 8
        num_classes = 10
        num_prototypes = 5
        feature_dim = 256
        reg_max = 16
        
        # Detection predictions/targets
        pred_bboxes = torch.rand(batch_size, 4) * 100
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.randn(batch_size, 4 * reg_max)
        target_bboxes = torch.rand(batch_size, 4) * 100
        target_bboxes[:, 2:] = target_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        target_cls = torch.randint(0, 2, (batch_size, num_classes)).float()
        target_dfl = torch.rand(batch_size, 4) * 15
        
        # Contrastive learning inputs
        query_features = torch.randn(batch_size, feature_dim)
        support_prototypes = torch.randn(num_prototypes, feature_dim)
        feature_labels = torch.randint(0, num_prototypes, (batch_size,))
        proposal_features = torch.randn(batch_size * 4, feature_dim)
        proposal_labels = torch.randint(-1, num_prototypes, (batch_size * 4,))
        
        # Forward pass
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels,
            proposal_features=proposal_features,
            proposal_labels=proposal_labels
        )
        
        # Check all losses are computed
        assert losses['supcon_loss'].item() > 0, "SupCon loss should be positive"
        assert not torch.isnan(losses['total_loss'])
    
    def test_set_stage_updates_weights(self):
        """Test dynamically changing training stage"""
        loss_fn = create_loss_fn(stage=1)
        
        # Initially stage 1
        assert loss_fn.weights['supcon'] == 0.0
        
        # Change to stage 2
        loss_fn.set_stage(2)
        assert loss_fn.weights['supcon'] == 1.0
        
        # Change to stage 3
        loss_fn.set_stage(3)
        assert loss_fn.weights['supcon'] == 0.5
    
    def test_gradient_flow_all_components(self):
        """Test gradients flow through all loss components"""
        loss_fn = create_loss_fn(stage=2)
        
        batch_size = 4
        num_classes = 5
        num_prototypes = 3
        feature_dim = 128
        reg_max = 16
        
        # Create model parameters to simulate network output
        bbox_params = torch.nn.Parameter(torch.rand(batch_size, 4) * 100)
        cls_params = torch.nn.Parameter(torch.randn(batch_size, num_classes))
        dfl_params = torch.nn.Parameter(torch.randn(batch_size, 4 * reg_max))
        query_params = torch.nn.Parameter(torch.randn(batch_size, feature_dim))
        proto_params = torch.nn.Parameter(torch.randn(num_prototypes, feature_dim))
        
        # Create targets
        target_bboxes = torch.rand(batch_size, 4) * 100
        target_cls = torch.randint(0, 2, (batch_size, num_classes)).float()
        target_dfl = torch.rand(batch_size, 4) * 15
        feature_labels = torch.randint(0, num_prototypes, (batch_size,))
        
        # Forward and backward
        losses = loss_fn(
            pred_bboxes=bbox_params,
            pred_cls_logits=cls_params,
            pred_dfl_dist=dfl_params,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_params,
            support_prototypes=proto_params,
            feature_labels=feature_labels
        )
        
        losses['total_loss'].backward()
        
        # Check gradients exist and are valid
        assert bbox_params.grad is not None
        assert cls_params.grad is not None
        assert dfl_params.grad is not None
        assert query_params.grad is not None
        assert proto_params.grad is not None
        
        assert not torch.isnan(bbox_params.grad).any()
        assert not torch.isnan(cls_params.grad).any()
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches"""
        loss_fn = create_loss_fn(stage=1)
        
        # Empty predictions/targets
        pred_bboxes = torch.empty(0, 4)
        pred_cls_logits = torch.empty(0, 10)
        pred_dfl_dist = torch.empty(0, 4 * 17)
        target_bboxes = torch.empty(0, 4)
        target_cls = torch.empty(0, 10)
        target_dfl = torch.empty(0, 4)
        
        # Should not crash
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl
        )
        
        # Losses should be zero
        assert losses['bbox_loss'].item() == 0.0
        assert losses['cls_loss'].item() == 0.0
        assert losses['dfl_loss'].item() == 0.0
    
    def test_loss_weighting_contribution(self):
        """Test individual loss components contribute proportionally"""
        loss_fn = create_loss_fn(stage=1)
        
        batch_size = 8
        num_classes = 10
        reg_max = 16
        
        pred_bboxes = torch.rand(batch_size, 4) * 100
        pred_bboxes[:, 2:] = pred_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.randn(batch_size, 4 * reg_max)
        target_bboxes = torch.rand(batch_size, 4) * 100
        target_bboxes[:, 2:] = target_bboxes[:, :2] + torch.rand(batch_size, 2) * 50
        target_cls = torch.randint(0, 2, (batch_size, num_classes)).float()
        target_dfl = torch.rand(batch_size, 4) * 15
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl
        )
        
        # Manually compute expected total
        expected_total = (
            7.5 * losses['bbox_loss'] +
            0.5 * losses['cls_loss'] +
            1.5 * losses['dfl_loss']
        )
        
        assert torch.allclose(
            losses['total_loss'], 
            expected_total, 
            atol=1e-5
        ), "Total loss should equal weighted sum of components"
    
    def test_custom_weights(self):
        """Test custom loss weights"""
        loss_fn = ReferenceBasedDetectionLoss(
            stage=2,
            bbox_weight=10.0,
            cls_weight=1.0,
            dfl_weight=2.0,
            supcon_weight=0.5,
            cpe_weight=0.25
        )
        
        assert loss_fn.weights['bbox'] == 10.0
        assert loss_fn.weights['cls'] == 1.0
        assert loss_fn.weights['dfl'] == 2.0
        assert loss_fn.weights['supcon'] == 0.5
        assert loss_fn.weights['cpe'] == 0.25


class TestLossIntegration:
    """Integration tests simulating real training scenarios"""
    
    def test_training_loop_simulation(self):
        """Simulate a mini training loop"""
        loss_fn = create_loss_fn(stage=1)
        optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)], lr=0.001)
        
        for epoch in range(3):
            # Simulate batch
            batch_size = 4
            pred_bboxes = torch.rand(batch_size, 4, requires_grad=True) * 100
            pred_cls_logits = torch.randn(batch_size, 10, requires_grad=True)
            pred_dfl_dist = torch.randn(batch_size, 4 * 17, requires_grad=True)
            target_bboxes = torch.rand(batch_size, 4) * 100
            target_cls = torch.randint(0, 2, (batch_size, 10)).float()
            target_dfl = torch.rand(batch_size, 4) * 15
            
            # Forward
            losses = loss_fn(
                pred_bboxes=pred_bboxes,
                pred_cls_logits=pred_cls_logits,
                pred_dfl_dist=pred_dfl_dist,
                target_bboxes=target_bboxes,
                target_cls=target_cls,
                target_dfl=target_dfl
            )
            
            # Should not crash
            assert not torch.isnan(losses['total_loss'])
    
    def test_progressive_training_stages(self):
        """Test transitioning through training stages"""
        loss_fn = create_loss_fn(stage=1)
        
        batch_size = 4
        
        # Stage 1: Base training
        for _ in range(2):
            pred_bboxes = torch.rand(batch_size, 4) * 100
            pred_cls_logits = torch.randn(batch_size, 10)
            pred_dfl_dist = torch.randn(batch_size, 4 * 17)
            target_bboxes = torch.rand(batch_size, 4) * 100
            target_cls = torch.randint(0, 2, (batch_size, 10)).float()
            target_dfl = torch.rand(batch_size, 4) * 15
            
            losses = loss_fn(
                pred_bboxes=pred_bboxes,
                pred_cls_logits=pred_cls_logits,
                pred_dfl_dist=pred_dfl_dist,
                target_bboxes=target_bboxes,
                target_cls=target_cls,
                target_dfl=target_dfl
            )
            assert losses['supcon_loss'].item() == 0.0
        
        # Transition to Stage 2
        loss_fn.set_stage(2)
        
        # Stage 2: Few-shot meta-learning
        query_features = torch.randn(batch_size, 256)
        support_prototypes = torch.randn(5, 256)
        feature_labels = torch.randint(0, 5, (batch_size,))
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels
        )
        assert losses['supcon_loss'].item() > 0, "Stage 2 should compute supcon loss"


class TestTripletLossIntegration:
    """Test triplet loss integration in Stage 3"""
    
    def test_stage3_with_batch_hard_triplet(self):
        """Test Stage 3 with batch-hard triplet loss"""
        loss_fn = create_loss_fn(stage=3, use_batch_hard_triplet=True)
        
        batch_size = 12
        num_classes = 10
        reg_max = 16
        feature_dim = 256
        
        # Detection predictions
        pred_bboxes = torch.rand(batch_size, 4)
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.softmax(torch.randn(batch_size, 4 * reg_max), dim=-1)  # 4 coords * (reg_max) bins
        
        # Targets
        target_bboxes = torch.rand(batch_size, 4)
        target_cls_indices = torch.randint(0, num_classes, (batch_size,))
        target_cls = torch.zeros(batch_size, num_classes).scatter_(1, target_cls_indices.unsqueeze(1), 1.0)  # One-hot
        target_dfl = torch.rand(batch_size, 4) * reg_max  # Coordinates in [0, reg_max]
        
        # Contrastive features
        query_features = torch.randn(batch_size, feature_dim)
        support_prototypes = torch.randn(num_classes, feature_dim)
        feature_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Triplet: batch-hard variant (only needs embeddings + labels)
        triplet_embeddings = torch.randn(batch_size, feature_dim)
        triplet_labels = torch.randint(0, 5, (batch_size,))  # 5 classes
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels,
            triplet_embeddings=triplet_embeddings,
            triplet_labels=triplet_labels
        )
        
        # Verify all components computed
        assert 'bbox_loss' in losses
        assert 'cls_loss' in losses
        assert 'dfl_loss' in losses
        assert 'supcon_loss' in losses
        assert 'cpe_loss' in losses
        assert 'triplet_loss' in losses
        assert 'total_loss' in losses
        
        # Verify triplet loss is non-negative (should mine hard triplets)
        assert losses['triplet_loss'].item() >= 0, "Triplet loss should be non-negative"
        
        # Verify weights are applied
        assert loss_fn.weights['triplet'] == 0.2
    
    def test_stage3_with_regular_triplet(self):
        """Test Stage 3 with regular triplet loss (explicit anchor/pos/neg)"""
        loss_fn = create_loss_fn(stage=3, use_batch_hard_triplet=False)
        
        batch_size = 8
        num_classes = 10
        reg_max = 16
        feature_dim = 256
        
        # Detection predictions
        pred_bboxes = torch.rand(batch_size, 4)
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.softmax(torch.randn(batch_size, 4 * reg_max), dim=-1)  # 4 coords * (reg_max) bins
        
        # Targets
        target_bboxes = torch.rand(batch_size, 4)
        target_cls_indices = torch.randint(0, num_classes, (batch_size,))
        target_cls = torch.zeros(batch_size, num_classes).scatter_(1, target_cls_indices.unsqueeze(1), 1.0)  # One-hot
        target_dfl = torch.rand(batch_size, 4) * reg_max  # Coordinates in [0, reg_max]
        
        # Contrastive features
        query_features = torch.randn(batch_size, feature_dim)
        support_prototypes = torch.randn(num_classes, feature_dim)
        feature_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Triplet: regular variant (needs anchor/positive/negative)
        triplet_anchors = torch.randn(batch_size, feature_dim)
        triplet_positives = triplet_anchors + torch.randn(batch_size, feature_dim) * 0.1
        triplet_negatives = torch.randn(batch_size, feature_dim)
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels,
            triplet_anchors=triplet_anchors,
            triplet_positives=triplet_positives,
            triplet_negatives=triplet_negatives
        )
        
        assert 'triplet_loss' in losses
        assert 'total_loss' in losses
        assert losses['triplet_loss'].item() >= 0
    
    def test_stage3_gradient_flow_with_triplet(self):
        """Test gradients flow through all components including triplet"""
        loss_fn = create_loss_fn(stage=3, use_batch_hard_triplet=True)
        
        batch_size = 8
        num_classes = 5
        reg_max = 16
        feature_dim = 128
        
        # Detection predictions (with gradients)
        pred_bboxes = torch.rand(batch_size, 4, requires_grad=True)
        pred_cls_logits = torch.randn(batch_size, num_classes, requires_grad=True)
        pred_dfl_dist = torch.softmax(torch.randn(batch_size, 4 * reg_max, requires_grad=True), dim=-1)
        
        # Targets
        target_bboxes = torch.rand(batch_size, 4)
        target_cls_indices = torch.randint(0, num_classes, (batch_size,))
        target_cls = torch.zeros(batch_size, num_classes).scatter_(1, target_cls_indices.unsqueeze(1), 1.0)  # One-hot
        target_dfl = torch.rand(batch_size, 4) * reg_max  # Coordinates in [0, reg_max]
        
        # Contrastive features (with gradients)
        query_features = torch.randn(batch_size, feature_dim, requires_grad=True)
        support_prototypes = torch.randn(num_classes, feature_dim, requires_grad=True)
        feature_labels = torch.randint(0, num_classes, (batch_size,))
        
        # Triplet embeddings (with gradients)
        triplet_embeddings = torch.randn(batch_size, feature_dim, requires_grad=True)
        triplet_labels = torch.randint(0, 3, (batch_size,))  # 3 classes
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels,
            triplet_embeddings=triplet_embeddings,
            triplet_labels=triplet_labels
        )
        
        # Backpropagate
        losses['total_loss'].backward()
        
        # Check gradients exist for all learnable parameters
        assert pred_bboxes.grad is not None, "Gradients should flow to bboxes"
        assert pred_cls_logits.grad is not None, "Gradients should flow to cls"
        assert query_features.grad is not None, "Gradients should flow to query features"
        assert support_prototypes.grad is not None, "Gradients should flow to prototypes"
        assert triplet_embeddings.grad is not None, "Gradients should flow to triplet embeddings"
        
        # Check no NaN gradients
        assert not torch.isnan(pred_bboxes.grad).any()
        assert not torch.isnan(pred_cls_logits.grad).any()
        assert not torch.isnan(query_features.grad).any()
        assert not torch.isnan(triplet_embeddings.grad).any()
    
    def test_stage3_without_triplet_inputs(self):
        """Test Stage 3 gracefully handles missing triplet inputs (returns 0 loss)"""
        loss_fn = create_loss_fn(stage=3)
        
        batch_size = 8
        num_classes = 10
        reg_max = 16
        feature_dim = 256
        
        # Only detection + contrastive (no triplet inputs)
        pred_bboxes = torch.rand(batch_size, 4)
        pred_cls_logits = torch.randn(batch_size, num_classes)
        pred_dfl_dist = torch.softmax(torch.randn(batch_size, 4 * reg_max), dim=-1)  # 4 coords * (reg_max) bins
        
        target_bboxes = torch.rand(batch_size, 4)
        target_cls_indices = torch.randint(0, num_classes, (batch_size,))
        target_cls = torch.zeros(batch_size, num_classes).scatter_(1, target_cls_indices.unsqueeze(1), 1.0)  # One-hot
        target_dfl = torch.rand(batch_size, 4) * reg_max  # Coordinates in [0, reg_max]
        
        query_features = torch.randn(batch_size, feature_dim)
        support_prototypes = torch.randn(num_classes, feature_dim)
        feature_labels = torch.randint(0, num_classes, (batch_size,))
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            pred_dfl_dist=pred_dfl_dist,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
            target_dfl=target_dfl,
            query_features=query_features,
            support_prototypes=support_prototypes,
            feature_labels=feature_labels
            # No triplet inputs provided
        )
        
        # Triplet loss should be 0 when no inputs provided
        assert 'triplet_loss' in losses
        assert losses['triplet_loss'].item() == 0.0, "Triplet loss should be 0 without inputs"
    
    def test_dynamic_stage_switching_with_triplet(self):
        """Test switching between stages adjusts triplet weight"""
        loss_fn = ReferenceBasedDetectionLoss()
        
        # Stage 1: No triplet
        loss_fn.set_stage(1)
        assert loss_fn.weights['triplet'] == 0.0
        
        # Stage 2: No triplet
        loss_fn.set_stage(2)
        assert loss_fn.weights['triplet'] == 0.0
        
        # Stage 3: Triplet enabled
        loss_fn.set_stage(3)
        assert loss_fn.weights['triplet'] == 0.2
        
        # Custom weight
        loss_fn.set_stage(3, triplet_weight=0.5)
        assert loss_fn.weights['triplet'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
