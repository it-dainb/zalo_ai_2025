"""
Unit tests for CHEAF Fusion Module
===================================

Tests:
1. Module initialization
2. Single image fusion
3. Batch processing
4. Feature dimension consistency
5. Individual components (Linear Attention, Short-Long Conv, Pyramid)
6. Parameter counting
7. Gradient flow
8. Different scale combinations
9. Bypass mode
"""

import torch
import pytest
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cheaf_fusion import (
    CHEAFFusionModule,
    EfficientLinearAttention,
    ShortLongConvModule,
    ChannelSpatialGate,
    CrossScalePyramidRefinement
)


class TestCHEAFFusion:
    """Test suite for CHEAF Fusion Module"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def cheaf_module(self, device):
        """Create CHEAF module instance for testing - matching dimensions for Lego architecture"""
        model = CHEAFFusionModule(
            query_channels=[64, 128, 256],     # Matching YOLOv8n backbone
            support_channels=[64, 128, 256],   # Matching DINOv3 encoder
            out_channels=[256, 512, 512],
            num_heads=4,
            use_pyramid_refinement=True,
            use_short_long_conv=True,
        ).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def mock_features(self, device):
        """Create mock query and support features - matching dimensions for Lego architecture"""
        query_feats = {
            'p3': torch.randn(1, 64, 80, 80).to(device),
            'p4': torch.randn(1, 128, 40, 40).to(device),
            'p5': torch.randn(1, 256, 20, 20).to(device),
        }
        
        support_feats = {
            'p3': torch.randn(1, 64).to(device),
            'p4': torch.randn(1, 128).to(device),
            'p5': torch.randn(1, 256).to(device),
        }
        
        return query_feats, support_feats
    
    def test_module_initialization(self, cheaf_module):
        """Test CHEAF module initializes correctly"""
        assert cheaf_module is not None
        assert len(cheaf_module.efficient_attentions) == 3
        assert all(scale in cheaf_module.efficient_attentions for scale in ['p3', 'p4', 'p5'])
        print("✅ CHEAF module initialization successful")
    
    def test_single_image_fusion(self, cheaf_module, mock_features, device):
        """Test fusion with single image"""
        query_feats, support_feats = mock_features
        
        with torch.no_grad():
            fused = cheaf_module(query_feats, support_feats)
        
        # Check all scales present
        assert 'p3' in fused
        assert 'p4' in fused
        assert 'p5' in fused
        
        # Check output shapes match expected
        assert fused['p3'].shape == (1, 256, 80, 80)
        assert fused['p4'].shape == (1, 512, 40, 40)
        assert fused['p5'].shape == (1, 512, 20, 20)
        
        print("✅ Single image fusion successful")
    
    def test_batch_fusion(self, cheaf_module, device):
        """Test fusion with batch of images"""
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            query_batch = {
                'p3': torch.randn(bs, 64, 80, 80).to(device),
                'p4': torch.randn(bs, 128, 40, 40).to(device),
                'p5': torch.randn(bs, 256, 20, 20).to(device),
            }
            
            support_batch = {
                'p3': torch.randn(bs, 64).to(device),
                'p4': torch.randn(bs, 128).to(device),
                'p5': torch.randn(bs, 256).to(device),
            }
            
            with torch.no_grad():
                fused = cheaf_module(query_batch, support_batch)
            
            assert fused['p3'].shape == (bs, 256, 80, 80)
            assert fused['p4'].shape == (bs, 512, 40, 40)
            assert fused['p5'].shape == (bs, 512, 20, 20)
        
        print(f"✅ Batch fusion successful for batch sizes: {batch_sizes}")
    
    def test_bypass_mode(self, cheaf_module, mock_features, device):
        """Test bypass mode (no support features)"""
        query_feats, _ = mock_features
        
        with torch.no_grad():
            fused = cheaf_module(query_feats, None)
        
        # Check all scales present
        assert 'p3' in fused
        assert 'p4' in fused
        assert 'p5' in fused
        
        # Check output shapes match expected
        assert fused['p3'].shape == (1, 256, 80, 80)
        assert fused['p4'].shape == (1, 512, 40, 40)
        assert fused['p5'].shape == (1, 512, 20, 20)
        
        print("✅ Bypass mode successful")
    
    def test_feature_dimensions(self, cheaf_module, device):
        """Test various feature dimensions"""
        test_configs = [
            # (batch, H, W)
            (1, 80, 80),
            (2, 80, 80),
            (4, 40, 40),
        ]
        
        for bs, h, w in test_configs:
            query_feats = {
                'p3': torch.randn(bs, 64, h, w).to(device),
                'p4': torch.randn(bs, 128, h//2, w//2).to(device),
                'p5': torch.randn(bs, 256, h//4, w//4).to(device),
            }
            
            support_feats = {
                'p3': torch.randn(bs, 64).to(device),
                'p4': torch.randn(bs, 128).to(device),
                'p5': torch.randn(bs, 256).to(device),
            }
            
            with torch.no_grad():
                fused = cheaf_module(query_feats, support_feats)
            
            assert fused['p3'].shape[0] == bs
            assert fused['p3'].shape[2:] == (h, w)
        
        print(f"✅ Feature dimension tests passed")
    
    def test_parameter_count(self, cheaf_module):
        """Test parameter counting"""
        total_params = cheaf_module.count_parameters()
        trainable_params = cheaf_module.count_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
        
        # CHEAF should be more complex than SCS (~1.2M) but still efficient
        assert total_params < 10e6, f"CHEAF too large: {total_params/1e6:.2f}M params"
        
        param_stats = cheaf_module.get_parameter_count()
        assert 'total' in param_stats
        assert param_stats['total'] == total_params
        
        print(f"✅ Parameter count: {total_params/1e6:.2f}M params")
    
    def test_gradient_flow(self, cheaf_module, mock_features, device):
        """Test gradient flow through module"""
        cheaf_module.train()
        query_feats, support_feats = mock_features
        
        # Make features require grad
        for k in query_feats:
            query_feats[k].requires_grad_(True)
        
        fused = cheaf_module(query_feats, support_feats)
        
        # Compute dummy loss
        loss = sum(f.sum() for f in fused.values())
        loss.backward()
        
        # Check gradients exist
        for k in query_feats:
            assert query_feats[k].grad is not None
            assert not torch.isnan(query_feats[k].grad).any()
        
        cheaf_module.eval()
        print("✅ Gradient flow successful")
    
    def test_efficient_linear_attention(self, device):
        """Test EfficientLinearAttention component"""
        in_channels = 128
        # New API: in_channels, key_channels, value_channels, head_count
        attn = EfficientLinearAttention(
            in_channels=in_channels,
            key_channels=in_channels // 2,  # 64
            value_channels=in_channels // 2,  # 64
            head_count=4
        ).to(device)
        attn.eval()
        
        query_feat = torch.randn(2, in_channels, 40, 40).to(device)
        support_proto = torch.randn(2, in_channels).to(device)
        
        with torch.no_grad():
            output = attn(query_feat, support_proto)
        
        assert output.shape == query_feat.shape
        print("✅ EfficientLinearAttention test passed")
    
    def test_short_long_conv(self, device):
        """Test ShortLongConvModule component"""
        channels = 128
        conv = ShortLongConvModule(channels=channels).to(device)
        conv.eval()
        
        x = torch.randn(2, channels, 40, 40).to(device)
        
        with torch.no_grad():
            output = conv(x)
        
        assert output.shape == x.shape
        print("✅ ShortLongConvModule test passed")
    
    def test_channel_spatial_gate(self, device):
        """Test ChannelSpatialGate component"""
        channels = 128
        gate = ChannelSpatialGate(channels=channels).to(device)
        gate.eval()
        
        x = torch.randn(2, channels, 40, 40).to(device)
        
        with torch.no_grad():
            output = gate(x)
        
        assert output.shape == x.shape
        print("✅ ChannelSpatialGate test passed")
    
    def test_pyramid_refinement(self, device):
        """Test CrossScalePyramidRefinement component"""
        channels = [64, 128, 256]
        pyramid = CrossScalePyramidRefinement(channels=channels).to(device)
        pyramid.eval()
        
        features = [
            torch.randn(2, 64, 80, 80).to(device),
            torch.randn(2, 128, 40, 40).to(device),
            torch.randn(2, 256, 20, 20).to(device),
        ]
        
        with torch.no_grad():
            refined = pyramid(features)
        
        assert len(refined) == 3
        for i, feat in enumerate(refined):
            assert feat.shape == features[i].shape
        
        print("✅ CrossScalePyramidRefinement test passed")
    
    def test_missing_support_scale(self, cheaf_module, device):
        """Test handling of missing support features for some scales"""
        query_feats = {
            'p3': torch.randn(1, 64, 80, 80).to(device),
            'p4': torch.randn(1, 128, 40, 40).to(device),
            'p5': torch.randn(1, 256, 20, 20).to(device),
        }
        
        # Only provide support for p4 and p5
        support_feats = {
            'p4': torch.randn(1, 128).to(device),
            'p5': torch.randn(1, 256).to(device),
        }
        
        with torch.no_grad():
            fused = cheaf_module(query_feats, support_feats)
        
        # Should still produce outputs for all scales
        assert 'p3' in fused
        assert 'p4' in fused
        assert 'p5' in fused
        
        print("✅ Missing support scale test passed")
    
    def test_output_consistency(self, cheaf_module, mock_features, device):
        """Test that repeated forward passes give same results"""
        query_feats, support_feats = mock_features
        
        with torch.no_grad():
            out1 = cheaf_module(query_feats, support_feats)
            out2 = cheaf_module(query_feats, support_feats)
        
        for scale in ['p3', 'p4', 'p5']:
            assert torch.allclose(out1[scale], out2[scale], rtol=1e-5, atol=1e-7)
        
        print("✅ Output consistency test passed")


def test_cheaf_standalone():
    """Standalone test for CHEAF module"""
    print("\n" + "="*60)
    print("CHEAF Fusion Module Standalone Test")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    cheaf = CHEAFFusionModule(
        query_channels=[64, 128, 256],    # YOLOv8n backbone output
        support_channels=[64, 128, 256],  # DINOv3 encoder output (must match!)
        out_channels=[256, 512, 512],
        num_heads=4,
        use_pyramid_refinement=True,
        use_short_long_conv=True,
    ).to(device)
    
    cheaf.eval()
    
    query_feats = {
        'p3': torch.randn(1, 64, 80, 80).to(device),
        'p4': torch.randn(1, 128, 40, 40).to(device),
        'p5': torch.randn(1, 256, 20, 20).to(device),
    }
    
    support_feats = {
        'p3': torch.randn(1, 64).to(device),
        'p4': torch.randn(1, 128).to(device),
        'p5': torch.randn(1, 256).to(device),
    }
    
    with torch.no_grad():
        fused = cheaf(query_feats, support_feats)
    
    print("\nOutput shapes:")
    for scale, feat in fused.items():
        print(f"  {scale}: {feat.shape}")
    
    param_stats = cheaf.get_parameter_count()
    print(f"\nParameter breakdown:")
    for component, count in param_stats.items():
        print(f"  {component}: {count/1e6:.2f}M")
    
    print("\n" + "="*60)
    print("✅ Standalone test passed!")
    print("="*60)


if __name__ == "__main__":
    test_cheaf_standalone()
