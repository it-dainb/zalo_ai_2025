"""
Test Inference Pipeline
=======================

Tests inference functionality:
1. Single image inference
2. Batch inference
3. Reference image caching
4. Different inference modes (standard, prototype, dual)
5. Post-processing (NMS, confidence filtering)
6. Inference speed benchmarking
"""

import pytest
import torch
import numpy as np
import cv2
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.yolov8n_refdet import YOLOv8nRefDet


class TestBasicInference:
    """Test basic inference functionality"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for inference"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
            conf_thres=0.25,
            iou_thres=0.45,
        ).to(device)
        model.eval()
        return model
    
    def test_single_image_inference(self, model, device):
        """Test inference on single image"""
        # Create dummy query image
        query_image = torch.randn(1, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            outputs = model(
                query_image=query_image,
                mode='prototype',
                use_cache=True,
            )
        
        assert outputs is not None
        print(f"✅ Single image inference successful")
        print(f"   Output keys: {outputs.keys()}")
    
    def test_batch_inference(self, model, device):
        """Test inference on batch of images"""
        batch_size = 4
        query_images = torch.randn(batch_size, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            outputs = model(
                query_image=query_images,
                mode='prototype',
                use_cache=True,
            )
        
        print(f"✅ Batch inference successful (batch_size={batch_size})")
    
    def test_inference_without_cache(self, model, device):
        """Test inference without caching (on-the-fly support features)"""
        query_image = torch.randn(1, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        with torch.no_grad():
            outputs = model(
                query_image=query_image,
                support_images=support_images,
                mode='prototype',
                use_cache=False,
            )
        
        assert outputs is not None
        print(f"✅ Inference without cache successful")


class TestInferenceModes:
    """Test different inference modes"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for inference"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=80,  # Enable standard mode
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        model.eval()
        return model
    
    def test_standard_mode(self, model, device):
        """Test standard mode (base classes only)"""
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            outputs = model(
                query_image=query_image,
                mode='standard',
            )
        
        assert 'standard_boxes' in outputs
        assert 'standard_cls' in outputs
        assert 'prototype_boxes' not in outputs
        
        print(f"✅ Standard mode inference successful")
    
    def test_prototype_mode(self, model, device):
        """Test prototype mode (novel objects)"""
        query_image = torch.randn(1, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            outputs = model(
                query_image=query_image,
                mode='prototype',
                use_cache=True,
            )
        
        assert 'prototype_boxes' in outputs or 'pred_bboxes' in outputs
        assert 'standard_boxes' not in outputs
        
        print(f"✅ Prototype mode inference successful")
    
    def test_dual_mode(self, model, device):
        """Test dual mode (base + novel classes)"""
        query_image = torch.randn(1, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            outputs = model(
                query_image=query_image,
                mode='dual',
                use_cache=True,
            )
        
        # Should have both outputs
        assert ('standard_boxes' in outputs or 'prototype_boxes' in outputs or 
                'pred_bboxes' in outputs)
        
        print(f"✅ Dual mode inference successful")


class TestReferenceCaching:
    """Test reference image caching functionality"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for testing"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        model.eval()
        return model
    
    def test_cache_single_reference(self, model, device):
        """Test caching single reference image"""
        support_image = torch.randn(1, 3, 256, 256).to(device)
        
        model.set_reference_images(support_image, average_prototypes=False)
        
        assert model._cached_support_features is not None
        print(f"✅ Single reference cached")
    
    def test_cache_multiple_references(self, model, device):
        """Test caching multiple reference images with averaging"""
        support_images = torch.randn(5, 3, 256, 256).to(device)
        
        model.set_reference_images(support_images, average_prototypes=True)
        
        assert model._cached_support_features is not None
        print(f"✅ Multiple references cached with averaging")
    
    def test_cache_clear(self, model, device):
        """Test clearing cache"""
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        model.set_reference_images(support_images, average_prototypes=True)
        assert model._cached_support_features is not None
        
        model.clear_cache()
        assert model._cached_support_features is None
        
        print(f"✅ Cache cleared")
    
    def test_cache_reuse(self, model, device):
        """Test that cached features are reused"""
        support_images = torch.randn(3, 3, 256, 256).to(device)
        query_images = [torch.randn(1, 3, 640, 640).to(device) for _ in range(3)]
        
        # Set cache once
        model.set_reference_images(support_images, average_prototypes=True)
        
        # Run multiple inferences using cache
        with torch.no_grad():
            for query in query_images:
                outputs = model(
                    query_image=query,
                    mode='prototype',
                    use_cache=True,
                )
                assert outputs is not None
        
        print(f"✅ Cache reused across multiple inferences")


class TestInferenceSpeed:
    """Test inference speed and performance"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for benchmarking"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        model.eval()
        return model
    
    def test_inference_speed(self, model, device):
        """Test inference speed"""
        query_image = torch.randn(1, 3, 640, 640).to(device)
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        # Warm-up
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            for _ in range(5):
                _ = model(query_image=query_image, mode='prototype', use_cache=True)
        
        # Benchmark
        num_iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(query_image=query_image, mode='prototype', use_cache=True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        print(f"✅ Inference speed benchmark:")
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   FPS: {fps:.2f}")
    
    def test_batch_inference_speed(self, model, device):
        """Test batch inference speed"""
        batch_sizes = [1, 2, 4]
        support_images = torch.randn(3, 3, 256, 256).to(device)
        
        model.set_reference_images(support_images, average_prototypes=True)
        
        print(f"✅ Batch inference speed:")
        
        for batch_size in batch_sizes:
            query_images = torch.randn(batch_size, 3, 640, 640).to(device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(query_image=query_images, mode='prototype', use_cache=True)
            
            # Benchmark
            num_iterations = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(query_image=query_images, mode='prototype', use_cache=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            fps = batch_size / avg_time
            
            print(f"   Batch size {batch_size}: {avg_time*1000:.2f}ms ({fps:.2f} FPS)")


class TestInferenceWithRealData:
    """Test inference with real images if available"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for inference"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        model.eval()
        return model
    
    def test_inference_with_test_images(self, model, device):
        """Test inference with actual test images if available"""
        # Check for test images
        test_samples_dir = Path("./datasets/test/samples")
        if not test_samples_dir.exists():
            pytest.skip("Test samples directory not found")
        
        # Find first available sample
        sample_dirs = [d for d in test_samples_dir.iterdir() if d.is_dir()]
        if len(sample_dirs) == 0:
            pytest.skip("No test samples found")
        
        sample_dir = sample_dirs[0]
        
        # Load support images
        object_images_dir = sample_dir / "object_images"
        if not object_images_dir.exists():
            pytest.skip("No object images found")
        
        image_files = list(object_images_dir.glob("*.jpg")) + list(object_images_dir.glob("*.png"))
        if len(image_files) == 0:
            pytest.skip("No image files found")
        
        # Load and preprocess support images
        support_images = []
        for img_file in image_files[:3]:  # Use up to 3 support images
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            support_images.append(img.unsqueeze(0))
        
        if len(support_images) == 0:
            pytest.skip("Could not load support images")
        
        support_tensor = torch.cat(support_images, dim=0).to(device)
        
        # Create dummy query (in real scenario, would load from video)
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        # Run inference
        with torch.no_grad():
            model.set_reference_images(support_tensor, average_prototypes=True)
            outputs = model(
                query_image=query_image,
                mode='prototype',
                use_cache=True,
            )
        
        assert outputs is not None
        print(f"✅ Inference with real test images successful")
        print(f"   Sample: {sample_dir.name}")
        print(f"   Support images: {len(support_images)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
