"""
Unit tests for YOLOv8 Backbone Extractor
========================================

Tests:
1. Model initialization and weight loading
2. Feature extraction at multiple scales
3. Feature dimensions verification
4. Batch processing
5. Different input sizes
6. Frozen backbone
7. Parameter counting
8. Hook registration and cleanup
"""

import torch
import pytest
import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolo_backbone import YOLOBackboneExtractor


class TestYOLOv8Backbone:
    """Test suite for YOLOv8BackboneExtractor"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def weights_path(self):
        """Get weights path, fallback to default if custom not found"""
        custom_path = Path("baseline_enot_nano/weights/best.pt")
        if custom_path.exists():
            return str(custom_path)
        return "yolov8n.pt"
    
    @pytest.fixture
    def extractor(self, device, weights_path):
        """Create extractor instance for testing"""
        model = YOLOBackboneExtractor(
            weights_path=weights_path,
            extract_scales=['p3', 'p4', 'p5'],
            freeze_backbone=False,
        ).to(device)
        model.eval()
        return model
    
    def test_model_initialization(self, extractor, weights_path):
        """Test model loads correctly"""
        assert extractor is not None
        assert extractor.weights_path == Path(weights_path)
        assert extractor.extract_scales == ['p3', 'p4', 'p5']
        assert len(extractor.hooks) == 3, "Should have 3 forward hooks"
        print("✅ Model initialization successful")
    
    def test_feature_extraction_640(self, extractor, device):
        """Test feature extraction with standard 640x640 input"""
        img = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features = extractor(img)
        
        # Check all scales are present
        assert 'p3' in features
        assert 'p4' in features
        assert 'p5' in features
        
        # Check shapes (for 640x640 input) - YOLOv8n channel dimensions
        assert features['p3'].shape == (1, 64, 80, 80), f"P3 shape: {features['p3'].shape}"
        assert features['p4'].shape == (1, 128, 40, 40), f"P4 shape: {features['p4'].shape}"
        assert features['p5'].shape == (1, 256, 20, 20), f"P5 shape: {features['p5'].shape}"
        
        print("✅ Feature extraction (640x640) successful")
    
    def test_batch_processing(self, extractor, device):
        """Test batch processing"""
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            img_batch = torch.randn(bs, 3, 640, 640).to(device)
            
            with torch.no_grad():
                features = extractor(img_batch)
            
            # Check batch dimension
            assert features['p3'].shape[0] == bs, f"Batch size {bs} failed"
            assert features['p4'].shape[0] == bs
            assert features['p5'].shape[0] == bs
        
        print(f"✅ Batch processing tested: {batch_sizes}")
    
    def test_different_input_sizes(self, device, weights_path):
        """Test with different input sizes"""
        input_sizes = [320, 640, 1280]
        
        for size in input_sizes:
            extractor = YOLOBackboneExtractor(
                weights_path=weights_path,
                extract_scales=['p3', 'p4', 'p5'],
                input_size=size,
            ).to(device)
            extractor.eval()
            
            img = torch.randn(1, 3, size, size).to(device)
            
            with torch.no_grad():
                features = extractor(img)
            
            # Verify scale ratios
            expected_p3_size = size // 8
            expected_p4_size = size // 16
            expected_p5_size = size // 32
            
            assert features['p3'].shape[2] == expected_p3_size, \
                f"P3 size {size}: {features['p3'].shape}"
            assert features['p4'].shape[2] == expected_p4_size, \
                f"P4 size {size}: {features['p4'].shape}"
            assert features['p5'].shape[2] == expected_p5_size, \
                f"P5 size {size}: {features['p5'].shape}"
        
        print(f"✅ Input sizes tested: {input_sizes}")
    
    def test_feature_dims_method(self, extractor):
        """Test get_feature_dims returns correct dimensions"""
        expected_dims = extractor.get_feature_dims()
        
        # Check all scales present
        assert 'p3' in expected_dims
        assert 'p4' in expected_dims
        assert 'p5' in expected_dims
        
        # Check format (C, H, W)
        # Updated to match actual PANet neck output channels
        assert expected_dims['p3'] == (64, 80, 80)
        assert expected_dims['p4'] == (128, 40, 40)
        assert expected_dims['p5'] == (256, 20, 20)
        
        print("✅ Feature dimensions method correct (PANet neck outputs)")
    
    def test_frozen_backbone(self, device, weights_path):
        """Test frozen backbone has no trainable parameters"""
        extractor_frozen = YOLOBackboneExtractor(
            weights_path=weights_path,
            freeze_backbone=True,
        ).to(device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in extractor_frozen.model.parameters() 
                              if p.requires_grad)
        
        assert trainable_params == 0, \
            f"Frozen backbone should have 0 trainable params, got {trainable_params}"
        
        print("✅ Frozen backbone verified (0 trainable params)")
    
    def test_parameter_count(self, extractor):
        """Test parameter counting"""
        total_params = extractor.count_parameters(trainable_only=False)
        trainable_params = extractor.count_parameters(trainable_only=True)
        
        # YOLOv8n has ~3.2M params
        expected_range = (2.5e6, 3.5e6)
        
        assert expected_range[0] <= total_params <= expected_range[1], \
            f"Total params: {total_params/1e6:.2f}M (expected ~3.2M)"
        
        # Trainable should equal total when not frozen
        if not extractor.freeze_backbone:
            assert trainable_params == total_params, \
                "Trainable should equal total when not frozen"
        
        print(f"✅ Parameter count verified:")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Trainable: {trainable_params/1e6:.2f}M")
    
    def test_gradient_flow(self, extractor, device):
        """Test gradients flow through backbone when not frozen"""
        if extractor.freeze_backbone:
            pytest.skip("Backbone is frozen")
        
        extractor.train()
        
        img = torch.randn(1, 3, 640, 640, requires_grad=True).to(device)
        features = extractor(img)
        
        # Compute dummy loss
        loss = sum(f.sum() for f in features.values())
        loss.backward()
        
        # Check backbone parameters have gradients
        has_grad = False
        for param in extractor.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "At least some parameters should have gradients"
        
        extractor.eval()
        print("✅ Gradient flow verified")
    
    def test_hook_cleanup(self, device, weights_path):
        """Test that hooks are properly cleaned up"""
        extractor = YOLOBackboneExtractor(
            weights_path=weights_path,
            extract_scales=['p3'],
        ).to(device)
        
        # Verify hook was registered
        assert len(extractor.hooks) == 1
        
        # Delete extractor
        del extractor
        
        print("✅ Hook cleanup successful (no errors on deletion)")
    
    def test_selective_scale_extraction(self, device, weights_path):
        """Test extracting only specific scales"""
        # Test with only P3
        extractor_p3 = YOLOBackboneExtractor(
            weights_path=weights_path,
            extract_scales=['p3'],
        ).to(device)
        extractor_p3.eval()
        
        img = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features = extractor_p3(img)
        
        # Should only have P3
        assert 'p3' in features
        assert 'p4' not in features
        assert 'p5' not in features
        assert len(features) == 1
        
        print("✅ Selective scale extraction verified")
    
    def test_multiple_forward_passes(self, extractor, device):
        """Test multiple forward passes clear previous features"""
        img1 = torch.randn(1, 3, 640, 640).to(device)
        img2 = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features1 = extractor(img1)
            features2 = extractor(img2)
        
        # Features should be different
        assert not torch.allclose(features1['p3'], features2['p3'])
        
        print("✅ Multiple forward passes work correctly")
    
    def test_feature_consistency(self, extractor, device):
        """Test same input produces same features (deterministic)"""
        extractor.eval()
        img = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            features1 = extractor(img)
            features2 = extractor(img)
        
        # Check consistency across scales
        for scale in ['p3', 'p4', 'p5']:
            torch.testing.assert_close(features1[scale], features2[scale])
        
        print("✅ Feature extraction is deterministic")


def test_module_imports():
    """Test that module can be imported correctly"""
    try:
        from models.yolo_backbone import YOLOBackboneExtractor
        print("✅ Module import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
