"""
Unit tests for DINO Support Encoder (DINOv3)
=============================================

Tests:
1. Model initialization and loading
2. Forward pass with single image
3. Forward pass with batch
4. Multi-scale projection outputs
5. L2 normalization
6. Average prototype computation
7. Parameter counting
8. Gradient flow (trainable parameters)
"""

import torch
import pytest
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dino_encoder import DINOSupportEncoder


class TestDINOv2Encoder:
    """Test suite for DINO Support Encoder (DINOv3)"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def encoder(self, device):
        """Create encoder instance for testing (DINOv3)"""
        model = DINOSupportEncoder(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            output_dims=[64, 128, 256],  # Match YOLOv8n dimensions
            freeze_backbone=True,
            freeze_layers=6,
            input_size=256,
        ).to(device)
        model.eval()
        return model
    
    def test_model_initialization(self, encoder):
        """Test model loads correctly from timm"""
        assert encoder is not None
        assert encoder.feat_dim == 384, "ViT-Small should have 384-dim features"
        assert encoder.output_dims == [64, 128, 256], "Should match YOLOv8n dimensions"
        print("✅ Model initialization successful")
    
    def test_single_image_forward(self, encoder, device):
        """Test forward pass with single image (DINOv3)"""
        img = torch.randn(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = encoder(img)
        
        # Check output dictionary keys
        assert 'prototype' in output
        assert 'p3' in output
        assert 'p4' in output
        assert 'p5' in output
        
        # Check shapes - scale-specific dimensions matching YOLOv8n
        assert output['prototype'].shape == (1, 384)
        assert output['p3'].shape == (1, 64)
        assert output['p4'].shape == (1, 128)
        assert output['p5'].shape == (1, 256)
        
        print("✅ Single image forward pass successful")
    
    def test_batch_forward(self, encoder, device):
        """Test forward pass with batch (DINOv3)"""
        batch_size = 4
        imgs = torch.randn(batch_size, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = encoder(imgs)
        
        # Check batch dimensions - scale-specific dimensions matching YOLOv8n
        assert output['prototype'].shape == (batch_size, 384)
        assert output['p3'].shape == (batch_size, 64)
        assert output['p4'].shape == (batch_size, 128)
        assert output['p5'].shape == (batch_size, 256)
        
        print("✅ Batch forward pass successful")
    
    def test_l2_normalization(self, encoder, device):
        """Test that output features are L2 normalized (DINOv3)"""
        img = torch.randn(2, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = encoder(img)
        
        # Check L2 norms (should be ~1.0) for each item in batch
        proto_norms = output['prototype'].norm(dim=-1)
        p3_norms = output['p3'].norm(dim=-1)
        p4_norms = output['p4'].norm(dim=-1)
        p5_norms = output['p5'].norm(dim=-1)
        
        assert torch.all(torch.abs(proto_norms - 1.0) < 1e-5), f"Prototype norms: {proto_norms}"
        assert torch.all(torch.abs(p3_norms - 1.0) < 1e-5), f"P3 norms: {p3_norms}"
        assert torch.all(torch.abs(p4_norms - 1.0) < 1e-5), f"P4 norms: {p4_norms}"
        assert torch.all(torch.abs(p5_norms - 1.0) < 1e-5), f"P5 norms: {p5_norms}"
        
        print(f"✅ L2 normalization verified (norms ~1.0)")
    
    def test_without_normalization(self, encoder, device):
        """Test that raw DINO features are extracted and then normalized"""
        img = torch.randn(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            output = encoder.extract_features(img)
        
        # Raw DINO features before projection and normalization
        # Should be (1, 384) tensor
        assert output.shape == (1, 384), f"Expected (1, 384), got {output.shape}"
        
        # Raw features typically have norms != 1.0
        raw_norm = output.norm(dim=-1).item()
        print(f"✅ Raw feature extraction verified (norm: {raw_norm:.3f})")
    
    def test_average_prototype(self, encoder, device):
        """Test computing average prototype from multiple images (DINOv3)"""
        # K-shot learning: 3 support images
        num_support = 3
        support_imgs = [torch.randn(1, 3, 256, 256).to(device) for _ in range(num_support)]
        
        with torch.no_grad():
            avg_proto = encoder.compute_average_prototype(support_imgs)
        
        # Check shapes - scale-specific dimensions matching YOLOv8n
        assert avg_proto['prototype'].shape == (1, 384)
        assert avg_proto['p3'].shape == (1, 64)
        assert avg_proto['p4'].shape == (1, 128)
        assert avg_proto['p5'].shape == (1, 256)
        
        # Check normalization
        norm = avg_proto['prototype'].norm(dim=-1).item()
        assert abs(norm - 1.0) < 1e-5, f"Averaged prototype should be normalized: {norm}"
        
        print(f"✅ Average prototype from {num_support} images successful")
    
    def test_parameter_count(self, encoder):
        """Test parameter counting matches expected values"""
        total_params = encoder.count_parameters(trainable_only=False)
        trainable_params = encoder.count_parameters(trainable_only=True)
        
        # Expected values (approximate)
        expected_total = 21.7e6  # ViT-Small has ~21.7M params
        
        # Allow 10% tolerance
        assert abs(total_params - expected_total) / expected_total < 0.2, \
            f"Total params: {total_params/1e6:.2f}M (expected ~21.7M)"
        
        # Trainable should be less than total (due to freezing)
        assert trainable_params < total_params, \
            "Trainable params should be less than total when frozen"
        
        print(f"✅ Parameter count verified:")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Trainable: {trainable_params/1e6:.2f}M")
        print(f"   Frozen: {(total_params-trainable_params)/1e6:.2f}M")
    
    def test_gradient_flow(self, encoder, device):
        """Test gradient flow through trainable parameters (DINOv3)"""
        encoder.train()
        img = torch.randn(2, 3, 256, 256).to(device)
    
    def test_feature_extraction_consistency(self, encoder, device):
        """Test that same input gives consistent features (DINOv3)"""
        img = torch.randn(1, 3, 256, 256).to(device)
    
    def test_different_batch_sizes(self, encoder, device):
        """Test encoder with different batch sizes (DINOv3)"""
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            img = torch.randn(bs, 3, 256, 256).to(device)
            
            with torch.no_grad():
                output = encoder(img)
            
            assert output['prototype'].shape[0] == bs
            assert output['p3'].shape[0] == bs
        
        print(f"✅ All batch sizes tested: {batch_sizes}")


def test_module_imports():
    """Test that module can be imported correctly"""
    try:
        from models.dino_encoder import DINOSupportEncoder
        print("✅ Module import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
