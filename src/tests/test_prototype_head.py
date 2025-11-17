"""
Unit tests for Prototype Detection Head
========================================

Tests:
1. Prototype head initialization and forward
2. Batch processing
3. Cosine similarity computation
4. Parameter counting
5. Temperature scaling
6. Multi-scale prototype matching
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.prototype_head import PrototypeDetectionHead, Conv


class TestPrototypeHead:
    """Test suite for Prototype Detection Head"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def prototype_head(self, device):
        """Create prototype head instance for testing"""
        model = PrototypeDetectionHead(
            ch=[256, 512, 512],
            proto_dims=[64, 128, 256],  # Scale-specific prototype dimensions
            temperature=10.0,
        ).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def mock_features(self, device):
        """Create mock feature maps"""
        return {
            'p3': torch.randn(1, 256, 80, 80).to(device),
            'p4': torch.randn(1, 512, 40, 40).to(device),
            'p5': torch.randn(1, 512, 20, 20).to(device),
        }
    
    @pytest.fixture
    def mock_prototypes(self, device):
        """Create mock prototype vectors - scale-specific dimensions"""
        return {
            'p3': torch.randn(3, 64).to(device),   # P3: 64 dims
            'p4': torch.randn(3, 128).to(device),  # P4: 128 dims
            'p5': torch.randn(3, 256).to(device),  # P5: 256 dims
        }
    
    def test_prototype_head_init(self, device):
        """Test prototype detection head initialization"""
        proto_head = PrototypeDetectionHead(
            ch=[256, 512, 512],
            proto_dims=[64, 128, 256],  # Scale-specific prototype dimensions
            temperature=10.0
        ).to(device)
        
        assert proto_head.nl == 3
        assert proto_head.proto_dims == [64, 128, 256]
        assert len(proto_head.feature_proj) == 3
        
        print("✅ Prototype head initialization successful")
    
    def test_prototype_head_forward(self, device, mock_features, mock_prototypes):
        """Test prototype head forward pass"""
        proto_head = PrototypeDetectionHead(
            ch=[256, 512, 512],
            proto_dims=[64, 128, 256]  # Scale-specific prototype dimensions
        ).to(device)
        proto_head.eval()
        
        feat_list = [mock_features['p3'], mock_features['p4'], mock_features['p5']]
        
        with torch.no_grad():
            box_preds, sim_scores = proto_head(feat_list, mock_prototypes)
        
        # Check outputs
        assert len(box_preds) == 3
        assert len(sim_scores) == 3
        
        # Check similarity scores have correct number of classes
        assert sim_scores[0].shape[1] == 3  # 3 prototype classes
        
        print("✅ Prototype head forward pass successful")
    
    def test_cosine_similarity(self, device):
        """Test cosine similarity computation"""
        proto_head = PrototypeDetectionHead(ch=[256, 512, 512]).to(device)
        
        features = torch.randn(2, 256, 80, 80).to(device)
        prototypes = torch.randn(5, 256).to(device)
        
        with torch.no_grad():
            similarity = proto_head.compute_similarity(features, prototypes)
        
        # Check shape
        assert similarity.shape == (2, 5, 80, 80)
        
        # Check similarity values are in reasonable range (after temperature scaling)
        # Cosine similarity is in [-1, 1], after temp scaling can be larger
        assert similarity.min() >= -100 and similarity.max() <= 100
        
        print("✅ Cosine similarity computation verified")
    
    def test_batch_processing(self, prototype_head, device, mock_prototypes):
        """Test with different batch sizes"""
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            feat_list = [
                torch.randn(bs, 256, 80, 80).to(device),
                torch.randn(bs, 512, 40, 40).to(device),
                torch.randn(bs, 512, 20, 20).to(device),
            ]
            
            with torch.no_grad():
                box_preds, sim_scores = prototype_head(feat_list, mock_prototypes)
            
            # Check batch dimension
            assert box_preds[0].shape[0] == bs
            assert sim_scores[0].shape[0] == bs
        
        print(f"✅ Batch processing tested: {batch_sizes}")
    
    def test_parameter_count(self, prototype_head):
        """Test parameter counting"""
        total_params = sum(p.numel() for p in prototype_head.parameters())
        
        # Expected range for prototype head only (much smaller without standard head)
        expected_range = (0.1e6, 3.0e6)
        
        assert expected_range[0] <= total_params <= expected_range[1], \
            f"Params: {total_params/1e6:.2f}M (expected 0.1-3.0M)"
        
        print(f"✅ Parameter count: {total_params/1e6:.2f}M")
    
    def test_temperature_scaling(self, device):
        """Test temperature parameter is learnable"""
        proto_head = PrototypeDetectionHead(ch=[256, 512, 512]).to(device)
        
        # Check temperature is a parameter
        assert isinstance(proto_head.temperature, nn.Parameter)
        assert proto_head.temperature.requires_grad
        
        # Test gradient flow through temperature
        proto_head.train()
        features = torch.randn(1, 256, 80, 80, requires_grad=True).to(device)
        prototypes = torch.randn(2, 256).to(device)
        
        similarity = proto_head.compute_similarity(features, prototypes)
        loss = similarity.sum()
        loss.backward()
        
        # Temperature should have gradient
        assert proto_head.temperature.grad is not None
        
        print("✅ Temperature scaling verified")
    
    def test_different_num_prototypes(self, prototype_head, device):
        """Test with different numbers of prototype classes"""
        feat_list = [
            torch.randn(1, 256, 80, 80).to(device),
            torch.randn(1, 512, 40, 40).to(device),
            torch.randn(1, 512, 20, 20).to(device),
        ]
        
        for num_protos in [1, 3, 5, 10]:
            prototypes = {
                'p3': torch.randn(num_protos, 64).to(device),   # P3: 64 dims
                'p4': torch.randn(num_protos, 128).to(device),  # P4: 128 dims
                'p5': torch.randn(num_protos, 256).to(device),  # P5: 256 dims
            }
            
            with torch.no_grad():
                box_preds, sim_scores = prototype_head(feat_list, prototypes)
            
            # Check similarity has correct number of classes
            assert sim_scores[0].shape[1] == num_protos
        
        print("✅ Different numbers of prototypes handled correctly")
    
    def test_conv_module(self, device):
        """Test Conv building block"""
        conv = Conv(256, 128, kernel_size=3, padding=1).to(device)
        x = torch.randn(1, 256, 80, 80).to(device)
        
        with torch.no_grad():
            out = conv(x)
        
        assert out.shape == (1, 128, 80, 80)
        
        print("✅ Conv module verified")
    

    def test_gradient_flow(self, prototype_head, device, mock_prototypes):
        """Test gradients flow through prototype head"""
        prototype_head.train()
        
        # Create features with gradients
        feat_list = [
            torch.randn(1, 256, 80, 80, requires_grad=True).to(device),
            torch.randn(1, 512, 40, 40, requires_grad=True).to(device),
            torch.randn(1, 512, 20, 20, requires_grad=True).to(device),
        ]
        
        box_preds, sim_scores = prototype_head(feat_list, mock_prototypes)
        
        # Compute dummy loss
        loss = torch.zeros(1, device=device)
        for b in box_preds:
            loss = loss + b.sum()
        for s in sim_scores:
            loss = loss + s.sum()
        
        loss.backward()
        
        # Check some parameters have gradients
        has_grad = any(p.grad is not None for p in prototype_head.parameters())
        
        assert has_grad, "Prototype head should have gradients"
        
        prototype_head.eval()
        print("✅ Gradient flow verified through prototype head")


def test_module_imports():
    """Test that all components can be imported"""
    try:
        from models.prototype_head import PrototypeDetectionHead, Conv
        print("✅ Module imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
