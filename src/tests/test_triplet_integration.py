"""
Integration Test for Triplet Training Pipeline
==============================================

Tests the complete triplet training integration:
1. Model returns features when return_features=True
2. Features have correct shapes (anchor: 384-dim, query: 256-dim)
3. Trainer can process triplet batches without errors
4. Gradient flows from triplet loss to model parameters
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.trainer import RefDetTrainer
from src.training.loss_utils import prepare_triplet_loss_inputs
from src.augmentations import get_stage_config


class TestTripletIntegration:
    """Test triplet training integration end-to-end."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @pytest.fixture
    def model(self, device):
        """Create model for testing."""
        model = YOLOv8nRefDet(
            yolo_weights='yolov8n.pt',
            nc_base=0,
            dinov3_model='vit_small_patch16_dinov3.lvd1689m',
            freeze_yolo=False,
            freeze_dinov3=True,
        )
        return model.to(device)
    
    @pytest.fixture
    def triplet_batch(self, device):
        """Create mock triplet batch."""
        batch_size = 2
        
        return {
            'anchor_images': torch.randn(batch_size, 3, 256, 256, device=device),
            'positive_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'positive_bboxes': [
                torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]], device=device, dtype=torch.float32),
                torch.tensor([[50, 50, 150, 150]], device=device, dtype=torch.float32),
            ],
            'negative_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'negative_bboxes': [
                torch.zeros((0, 4), device=device, dtype=torch.float32),  # Background
                torch.tensor([[300, 300, 400, 400]], device=device, dtype=torch.float32),  # Cross-class
            ],
            'class_ids': torch.tensor([0, 1], device=device),
            'negative_types': ['background', 'cross_class'],
        }
    
    def test_model_returns_features(self, model, triplet_batch, device):
        """Test that model returns global features when return_features=True."""
        model.eval()
        
        with torch.no_grad():
            # Forward pass with return_features=True
            output = model(
                query_image=triplet_batch['positive_images'],
                support_images=triplet_batch['anchor_images'],  # Add K dimension
                return_features=True,
            )
        
        # Check that features are returned
        assert 'query_global_feat' in output, "Model should return query_global_feat"
        assert 'support_global_feat' in output, "Model should return support_global_feat"
        
        # Check feature shapes
        batch_size = triplet_batch['positive_images'].shape[0]
        assert output['query_global_feat'].shape == (batch_size, 256), \
            f"Expected query features (B, 256), got {output['query_global_feat'].shape}"
        assert output['support_global_feat'].shape == (batch_size, 384), \
            f"Expected support features (B, 384), got {output['support_global_feat'].shape}"
    
    def test_feature_extraction_shapes(self, model, triplet_batch, device):
        """Test feature dimensions for triplet loss."""
        model.eval()
        
        with torch.no_grad():
            # Extract anchor features (from support images) using direct extraction
            anchor_features = model.extract_features(
                images=triplet_batch['anchor_images'],
                image_type='support',
            )
            
            # Extract positive features (from positive images)
            positive_features = model.extract_features(
                images=triplet_batch['positive_images'],
                image_type='query',
            )
            
            # Extract negative features (from negative images)
            negative_features = model.extract_features(
                images=triplet_batch['negative_images'],
                image_type='query',
            )
        
        # Check dimensions
        batch_size = triplet_batch['positive_images'].shape[0]
        
        # Anchor: DINOv3 features (384-dim)
        assert anchor_features.shape == (batch_size, 384), \
            f"Anchor features should be (B, 384) (DINOv3), got {anchor_features.shape}"
        
        # Positive: YOLOv8 features (256-dim)
        assert positive_features.shape == (batch_size, 256), \
            f"Positive features should be (B, 256) (YOLOv8), got {positive_features.shape}"
        
        # Negative: YOLOv8 features (256-dim)
        assert negative_features.shape == (batch_size, 256), \
            f"Negative features should be (B, 256) (YOLOv8), got {negative_features.shape}"
    
    def test_prepare_triplet_loss_inputs(self, model, triplet_batch, device):
        """Test that loss input preparation works correctly."""
        model.eval()
        
        with torch.no_grad():
            # Extract features directly using the new extract_features method
            anchor_features = model.extract_features(
                images=triplet_batch['anchor_images'],
                image_type='support',
            )
            
            positive_features = model.extract_features(
                images=triplet_batch['positive_images'],
                image_type='query',
            )
            
            negative_features = model.extract_features(
                images=triplet_batch['negative_images'],
                image_type='query',
            )
            
            # Create mock outputs dict matching what prepare_triplet_loss_inputs expects
            model_outputs = {
                'support_global_feat': anchor_features,
                'query_global_feat': torch.cat([positive_features, negative_features], dim=0),
            }
        
        # Prepare triplet loss inputs
        triplet_inputs = prepare_triplet_loss_inputs(
            model_outputs=model_outputs,
            batch=triplet_batch,
        )
        
        # Check that inputs are properly prepared
        assert 'anchor_features' in triplet_inputs
        assert 'positive_features' in triplet_inputs
        assert 'negative_features' in triplet_inputs
        
        # Check shapes
        # After prepare_triplet_loss_inputs, all features are projected to same dimension (256)
        # This is handled by the function to ensure compatibility for triplet loss
        batch_size = triplet_batch['positive_images'].shape[0]
        assert triplet_inputs['anchor_features'].shape == (batch_size, 256), \
            f"Anchor features should be projected to (B, 256), got {triplet_inputs['anchor_features'].shape}"
        assert triplet_inputs['positive_features'].shape == (batch_size, 256), \
            f"Positive features should be (B, 256), got {triplet_inputs['positive_features'].shape}"
        assert triplet_inputs['negative_features'].shape == (batch_size, 256), \
            f"Negative features should be (B, 256), got {triplet_inputs['negative_features'].shape}"
    
    def test_triplet_batch_forward(self, model, triplet_batch, device):
        """Test that trainer can process triplet batch without errors."""
        model.train()
        
        # Create loss function
        loss_fn = ReferenceBasedDetectionLoss(
            stage=2,
            triplet_weight=0.2,
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create trainer
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,  # Disable for testing
        )
        
        # Add batch_type marker
        triplet_batch['batch_type'] = 'triplet'
        
        # Forward step
        loss, losses_dict = trainer._forward_triplet_step(triplet_batch)
        
        # Check that loss is computed
        assert loss is not None, "Loss should be computed"
        assert loss.requires_grad, "Loss should require grad"
        assert loss.item() >= 0, "Loss should be non-negative"
        
        # Check loss components
        assert 'triplet_loss' in losses_dict, "Should have triplet_loss"
        assert losses_dict['triplet_loss'] >= 0, "Triplet loss should be non-negative"
    
    def test_gradient_flow(self, model, triplet_batch, device):
        """Test that gradients flow from triplet loss to model parameters."""
        model.train()
        
        # Create loss function
        loss_fn = ReferenceBasedDetectionLoss(
            stage=2,
            triplet_weight=0.2,
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create trainer
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Add batch_type marker
        triplet_batch['batch_type'] = 'triplet'
        
        # Forward step
        loss, losses_dict = trainer._forward_triplet_step(triplet_batch)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_grads = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        assert has_grads, "At least some parameters should have gradients"
    
    def test_mixed_detection_triplet_batch(self, model, triplet_batch, device):
        """Test that trainer can handle mixed detection + triplet batches."""
        model.train()
        
        # Create detection batch
        batch_size = 2
        n_way = 2
        k_shot = 2
        
        detection_batch = {
            'query_images': torch.randn(batch_size, 3, 640, 640, device=device),
            'support_images': torch.randn(n_way, k_shot, 3, 256, 256, device=device),
            'target_bboxes': [
                torch.tensor([[100, 100, 200, 200]], device=device, dtype=torch.float32),
                torch.tensor([[50, 50, 150, 150]], device=device, dtype=torch.float32),
            ],
            'target_classes': [
                torch.tensor([0], device=device, dtype=torch.long),
                torch.tensor([1], device=device, dtype=torch.long),
            ],
            'class_ids': torch.tensor([0, 1], device=device),
            'num_classes': n_way,
        }
        
        # Create loss function
        loss_fn = ReferenceBasedDetectionLoss(
            stage=2,
            triplet_weight=0.2,
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create trainer
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixed_precision=False,
        )
        
        # Test detection batch
        detection_batch['batch_type'] = 'detection'
        det_loss, det_losses = trainer._forward_detection_step(detection_batch)
        assert det_loss.item() >= 0, "Detection loss should be non-negative"
        
        # Test triplet batch
        triplet_batch['batch_type'] = 'triplet'
        trip_loss, trip_losses = trainer._forward_triplet_step(triplet_batch)
        assert trip_loss.item() >= 0, "Triplet loss should be non-negative"
        
        print("\nIntegration test passed!")
        print(f"  Detection loss: {det_loss.item():.4f}")
        print(f"  Triplet loss: {trip_loss.item():.4f}")


if __name__ == '__main__':
    """Run tests directly."""
    import sys
    
    # Run pytest
    pytest.main([__file__, '-v', '-s'])
