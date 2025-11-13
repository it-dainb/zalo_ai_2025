"""
Test Model Components
=====================

Tests individual model components in isolation:
1. DINOv2 encoder
2. YOLOv8 backbone
3. CHEAF fusion module
4. Dual detection head
5. Component integration
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dino_encoder import DINOSupportEncoder
from models.yolov8_backbone import YOLOv8BackboneExtractor
from models.cheaf_fusion import CHEAFFusionModule
from models.dual_head import DualDetectionHead


class TestDINOv2Encoder:
    """Test DINOv2 support encoder"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def encoder(self, device):
        """Create DINOv2 encoder"""
        # For now, skip DINOv2 tests as they require special model loading
        # The model is tested in the full YOLOv8nRefDet integration tests
        pytest.skip("DINOv2 encoder requires special model loading - tested in integration")
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly"""
        assert encoder is not None
        assert encoder.backbone is not None
        print(f"✅ DINOv2 encoder initialized")
    
    def test_encoder_forward(self, encoder, device):
        """Test encoder forward pass"""
        # Create dummy support images (256x256 for DINOv2 reg4)
        support_images = torch.randn(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            features = encoder(support_images)
        
        # Check output structure
        assert isinstance(features, dict)
        assert 'scale_0' in features
        assert 'scale_1' in features
        assert 'scale_2' in features
        
        print(f"✅ Encoder forward pass successful")
        for scale, feat in features.items():
            print(f"   {scale}: {feat.shape}")
    
    def test_encoder_batch_processing(self, encoder, device):
        """Test encoder with batch of images"""
        # Create batch of support images (256x256)
        batch_size = 4
        support_images = torch.randn(batch_size, 3, 256, 256).to(device)
        
        with torch.no_grad():
            features = encoder(support_images)
        
        # Check batch dimension
        for scale, feat in features.items():
            assert feat.shape[0] == batch_size
        
        print(f"✅ Encoder batch processing successful (batch_size={batch_size})")
    
    def test_encoder_prototype_averaging(self, encoder, device):
        """Test prototype averaging functionality"""
        # Create list of support images (256x256)
        support_list = [
            torch.randn(1, 3, 256, 256).to(device) for _ in range(3)
        ]
        
        with torch.no_grad():
            avg_features = encoder.compute_average_prototype(support_list)
        
        assert isinstance(avg_features, dict)
        print(f"✅ Prototype averaging successful")


class TestYOLOv8Backbone:
    """Test YOLOv8 backbone extractor"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def backbone(self, device):
        """Create YOLOv8 backbone"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        backbone = YOLOv8BackboneExtractor(
            weights_path=weights_path,
            extract_scales=['p3', 'p4', 'p5'],
            freeze_backbone=False,
        ).to(device)
        backbone.eval()
        return backbone
    
    def test_backbone_initialization(self, backbone):
        """Test backbone initializes correctly"""
        assert backbone is not None
        print(f"✅ YOLOv8 backbone initialized")
    
    def test_backbone_forward(self, backbone, device):
        """Test backbone forward pass"""
        # Create dummy query image
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features = backbone(query_image)
        
        # Check output structure
        assert isinstance(features, dict)
        assert 'p3' in features
        assert 'p4' in features
        assert 'p5' in features
        
        print(f"✅ Backbone forward pass successful")
        for scale, feat in features.items():
            print(f"   {scale}: {feat.shape}")
    
    def test_backbone_feature_dimensions(self, backbone, device):
        """Test backbone produces correct feature dimensions"""
        query_image = torch.randn(2, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features = backbone(query_image)
        
        # Check expected dimensions for YOLOv8n
        assert features['p3'].shape[1] == 64  # channels
        assert features['p4'].shape[1] == 128
        assert features['p5'].shape[1] == 256
        
        # Check spatial dimensions
        assert features['p3'].shape[2] == 80  # 640/8
        assert features['p4'].shape[2] == 40  # 640/16
        assert features['p5'].shape[2] == 20  # 640/32
        
        print(f"✅ Feature dimensions correct")


class TestCHEAFFusion:
    """Test CHEAF fusion module"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def fusion_module(self, device):
        """Create CHEAF fusion module - matching dimensions for Lego architecture"""
        fusion = CHEAFFusionModule(
            query_channels=[32, 64, 128, 256],  # 4 scales: P2, P3, P4, P5
            support_channels=[32, 64, 128, 256],  # Matching DINOv3 encoder output
            out_channels=[128, 256, 512, 512],  # 4-scale output
            num_heads=4,
            use_pyramid_refinement=True,
            use_short_long_conv=True,
        ).to(device)
        fusion.eval()
        return fusion
    
    def test_fusion_initialization(self, fusion_module):
        """Test fusion module initializes correctly"""
        assert fusion_module is not None
        print(f"✅ CHEAF fusion module initialized")
    
    def test_fusion_forward(self, fusion_module, device):
        """Test fusion forward pass"""
        # Create dummy features - matching Lego architecture dimensions (4 scales: P2-P5)
        query_feats = {
            'p2': torch.randn(1, 32, 160, 160).to(device),
            'p3': torch.randn(1, 64, 80, 80).to(device),
            'p4': torch.randn(1, 128, 40, 40).to(device),
            'p5': torch.randn(1, 256, 20, 20).to(device),
        }
        
        support_feats = {
            'p2': torch.randn(1, 32).to(device),  # Prototype vectors
            'p3': torch.randn(1, 64).to(device),
            'p4': torch.randn(1, 128).to(device),
            'p5': torch.randn(1, 256).to(device),
        }
        
        with torch.no_grad():
            fused_feats = fusion_module(query_feats, support_feats)
        
        # Check output structure
        assert isinstance(fused_feats, dict)
        assert 'p2' in fused_feats
        assert 'p3' in fused_feats
        assert 'p4' in fused_feats
        assert 'p5' in fused_feats
        
        print(f"✅ Fusion forward pass successful")
        for scale, feat in fused_feats.items():
            print(f"   {scale}: {feat.shape}")
    
    def test_fusion_output_channels(self, fusion_module, device):
        """Test fusion produces correct output channels"""
        query_feats = {
            'p2': torch.randn(2, 32, 160, 160).to(device),
            'p3': torch.randn(2, 64, 80, 80).to(device),
            'p4': torch.randn(2, 128, 40, 40).to(device),
            'p5': torch.randn(2, 256, 20, 20).to(device),
        }
        
        support_feats = {
            'p2': torch.randn(2, 32).to(device),  # Prototype vectors
            'p3': torch.randn(2, 64).to(device),
            'p4': torch.randn(2, 128).to(device),
            'p5': torch.randn(2, 256).to(device),
        }
        
        with torch.no_grad():
            fused_feats = fusion_module(query_feats, support_feats)
        
        # Check expected output channels (4 scales: P2-P5)
        assert fused_feats['p2'].shape[1] == 128
        assert fused_feats['p3'].shape[1] == 256
        assert fused_feats['p4'].shape[1] == 512
        assert fused_feats['p5'].shape[1] == 512
        
        print(f"✅ Fusion output channels correct")


class TestDualDetectionHead:
    """Test dual detection head"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def detection_head(self, device):
        """Create dual detection head"""
        head = DualDetectionHead(
            nc_base=80,
            ch=[128, 256, 512, 512],  # 4 scales: P2, P3, P4, P5
            proto_dims=[32, 64, 128, 256],  # Scale-specific prototype dimensions
            temperature=10.0,
            conf_thres=0.25,
            iou_thres=0.45,
        ).to(device)
        head.eval()
        return head
    
    def test_head_initialization(self, detection_head):
        """Test head initializes correctly"""
        assert detection_head is not None
        print(f"✅ Dual detection head initialized")
    
    def test_head_standard_mode(self, detection_head, device):
        """Test head in standard mode"""
        # Create dummy fused features (4 scales: P2-P5)
        fused_feats = {
            'p2': torch.randn(1, 128, 160, 160).to(device),
            'p3': torch.randn(1, 256, 80, 80).to(device),
            'p4': torch.randn(1, 512, 40, 40).to(device),
            'p5': torch.randn(1, 512, 20, 20).to(device),
        }
        
        with torch.no_grad():
            outputs = detection_head(fused_feats, mode='standard')
        
        # Check output structure
        assert 'standard_boxes' in outputs
        assert 'standard_cls' in outputs
        
        print(f"✅ Head standard mode successful")
        # Outputs might be lists or tensors
        if isinstance(outputs['standard_boxes'], list):
            print(f"   Boxes: list with {len(outputs['standard_boxes'])} scales")
        else:
            print(f"   Boxes: {outputs['standard_boxes'].shape}")
    
    def test_head_prototype_mode(self, detection_head, device):
        """Test head in prototype mode"""
        fused_feats = {
            'p2': torch.randn(1, 128, 160, 160).to(device),
            'p3': torch.randn(1, 256, 80, 80).to(device),
            'p4': torch.randn(1, 512, 40, 40).to(device),
            'p5': torch.randn(1, 512, 20, 20).to(device),
        }
        
        # Create dummy prototype features (dict format to match API) - 4 scales
        prototype_feats = {
            'p2': torch.randn(1, 32).to(device),
            'p3': torch.randn(1, 64).to(device),
            'p4': torch.randn(1, 128).to(device),
            'p5': torch.randn(1, 256).to(device),
        }
        
        with torch.no_grad():
            outputs = detection_head(
                fused_feats,
                mode='prototype',
                prototypes=prototype_feats
            )
        
        # Check output structure
        assert 'prototype_boxes' in outputs
        assert 'prototype_sim' in outputs
        
        print(f"✅ Head prototype mode successful")
    
    def test_head_dual_mode(self, detection_head, device):
        """Test head in dual mode"""
        fused_feats = {
            'p2': torch.randn(1, 128, 160, 160).to(device),
            'p3': torch.randn(1, 256, 80, 80).to(device),
            'p4': torch.randn(1, 512, 40, 40).to(device),
            'p5': torch.randn(1, 512, 20, 20).to(device),
        }
        
        prototype_feats = {
            'p2': torch.randn(1, 32).to(device),
            'p3': torch.randn(1, 64).to(device),
            'p4': torch.randn(1, 128).to(device),
            'p5': torch.randn(1, 256).to(device),
        }
        
        with torch.no_grad():
            outputs = detection_head(
                fused_feats,
                mode='dual',
                prototypes=prototype_feats
            )
        
        # Check both outputs are present
        assert 'standard_boxes' in outputs
        assert 'standard_cls' in outputs
        assert 'prototype_boxes' in outputs
        assert 'prototype_sim' in outputs
        
        print(f"✅ Head dual mode successful")


class TestComponentIntegration:
    """Test integration of all components"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_components_compatibility(self, device):
        """Test that all components work together"""
        # Skip DINOv2-based integration test (tested in full model tests)
        pytest.skip("Full component integration tested in YOLOv8nRefDet model tests")
        
        # This test is kept as documentation of how components connect
        model_path = Path("vit_small_patch16_dinov3.lvd1689m")
        if not model_path.exists():
            pytest.skip("DINOv2 model not available")
        
        # Initialize all components
        encoder = DINOSupportEncoder(
            model_name=str(model_path),
            output_dims=[128, 256, 512],
            freeze_backbone=True,
        ).to(device)
        
        backbone = YOLOv8BackboneExtractor(
            weights_path="yolov8n.pt",
            extract_scales=['p3', 'p4', 'p5'],
            freeze_backbone=False,
        ).to(device)
        
        fusion = CHEAFFusionModule(
            query_channels=[32, 64, 128, 256],  # 4 scales: P2, P3, P4, P5
            support_channels=[32, 64, 128, 256],  # Matching DINOv3 encoder output
            out_channels=[128, 256, 512, 512],  # 4-scale output
            num_heads=4,
            use_pyramid_refinement=True,
            use_short_long_conv=True,
        ).to(device)
        
        head = DualDetectionHead(
            nc_base=80,
            ch=[128, 256, 512, 512],  # 4 scales: P2, P3, P4, P5
            proto_dims=[32, 64, 128, 256],  # Scale-specific prototype dimensions
        ).to(device)
        
        # Create dummy inputs (256x256 for DINOv2)
        support_images = torch.randn(1, 3, 256, 256).to(device)
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        # Forward pass through all components
        with torch.no_grad():
            support_feats = encoder(support_images)
            query_feats = backbone(query_image)
            fused_feats = fusion(query_feats, support_feats)
            outputs = head(fused_feats, mode='dual', prototypes=support_feats)
        
        # Verify outputs
        assert outputs is not None
        assert 'standard_boxes' in outputs
        assert 'prototype_boxes' in outputs
        
        print(f"✅ All components integrated successfully")
        print(f"   Support features: {[f.shape for f in support_feats.values()]}")
        print(f"   Query features: {[f.shape for f in query_feats.values()]}")
        print(f"   Fused features: {[f.shape for f in fused_feats.values()]}")
        print(f"   Output keys: {list(outputs.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
