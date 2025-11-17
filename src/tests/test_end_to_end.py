"""
Unit tests for End-to-End YOLOv8n-RefDet Model
==============================================

Tests:
1. Model initialization and component integration
2. Standard mode (base classes only)
3. Prototype mode (reference-based)
4. Dual mode (combined)
5. Support feature caching
6. Batch processing
7. K-shot learning (multiple references)
8. Parameter counting and budget
9. Memory profiling
10. Inference pipeline
"""

import torch
import pytest
import sys
from pathlib import Path
import time

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolo_refdet import YOLOv8nRefDet


class TestEndToEnd:
    """Test suite for YOLOv8n-RefDet end-to-end model"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def weights_path(self):
        """Get weights path with fallback"""
        custom_path = Path("baseline_enot_nano/weights/best.pt")
        if custom_path.exists():
            return str(custom_path)
        return "yolov8n.pt"
    
    @pytest.fixture
    def model(self, device, weights_path):
        """Create model instance for testing"""
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            dinov3_model="vit_small_patch14_reg4_dinov2",
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def query_image(self, device):
        """Create mock query image"""
        return torch.randn(1, 3, 640, 640).to(device)
    
    @pytest.fixture
    def support_images(self, device):
        """Create mock support images (DINOv3 uses 256x256)"""
        return torch.randn(3, 3, 256, 256).to(device)
    
    def test_model_initialization(self, model):
        """Test model initializes with all components"""
        assert model.support_encoder is not None
        assert model.backbone is not None
        assert model.scs_fusion is not None
        assert model.detection_head is not None
        
        print("✅ Model initialization successful")
    
    def test_prototype_mode_with_support(self, model, query_image, support_images):
        """Test prototype mode with on-the-fly support computation"""
        with torch.no_grad():
            outputs = model(query_image, support_images)
        
        # Should have prototype outputs
        assert 'prototype_boxes' in outputs
        assert 'prototype_sim' in outputs
        
        print("✅ Prototype mode (on-the-fly) successful")
    
    def test_support_caching(self, model, query_image, support_images):
        """Test support feature caching mechanism"""
        # Cache support features
        model.set_reference_images(support_images)
        
        # Check cache exists
        assert model._cached_support_features is not None
        
        # Use cache in forward pass
        with torch.no_grad():
            outputs = model(query_image, use_cache=True)
        
        assert 'prototype_boxes' in outputs
        
        # Clear cache
        model.clear_cache()
        assert model._cached_support_features is None
        
        print("✅ Support feature caching successful")
    
    def test_cache_efficiency(self, model, query_image, support_images, device):
        """Test that caching improves inference speed"""
        # Warm-up
        with torch.no_grad():
            _ = model(query_image, support_images)
        
        # Time without cache
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(query_image, support_images, use_cache=False)
        time_no_cache = time.time() - start
        
        # Cache support features
        model.set_reference_images(support_images)
        
        # Time with cache
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(query_image, use_cache=True)
        time_with_cache = time.time() - start
        
        print(f"  Without cache: {time_no_cache*100:.2f}ms (avg)")
        print(f"  With cache: {time_with_cache*100:.2f}ms (avg)")
        print(f"  Speedup: {time_no_cache/time_with_cache:.2f}x")
        
        # Cache should be faster
        assert time_with_cache < time_no_cache
        
        print("✅ Cache efficiency verified")
    
    def test_batch_processing(self, model, device, support_images):
        """Test with different batch sizes"""
        batch_sizes = [1, 2, 4]
        
        model.set_reference_images(support_images)
        
        for bs in batch_sizes:
            query_batch = torch.randn(bs, 3, 640, 640).to(device)
            
            with torch.no_grad():
                outputs = model(query_batch, use_cache=True)
            
            # Check batch dimension
            assert outputs['prototype_sim'][0].shape[0] == bs
        
        print(f"✅ Batch processing tested: {batch_sizes}")
    
    def test_kshot_learning(self, model, query_image, device):
        """Test K-shot learning with multiple reference images"""
        k_values = [1, 3, 5]
        
        for k in k_values:
            # Create K reference images (DINOv3 uses 256x256)
            support_list = [torch.randn(1, 3, 256, 256).to(device) for _ in range(k)]
            
            # Cache with averaging
            model.set_reference_images(support_list, average_prototypes=True)
            
            # Inference
            with torch.no_grad():
                outputs = model(query_image, use_cache=True)
            
            assert 'prototype_boxes' in outputs
        
        print(f"✅ K-shot learning tested: {k_values}")
    
    def test_parameter_budget(self, model):
        """Test model stays within parameter budget"""
        total_params = model.count_parameters()
        
        # Updated architecture: DINOv2(21.94M) + YOLOv8n(3.16M) + PSALM(0.78M) + PrototypeHead(0.75M) = ~26.6M
        # Much lighter after removing DualDetectionHead and using PSALM fusion
        expected_range = (25e6, 30e6)
        assert expected_range[0] <= total_params <= expected_range[1], \
            f"Params: {total_params/1e6:.2f}M (expected 25-30M)"
        
        # Should be well under 50M limit
        assert total_params < 50e6, \
            f"Exceeds 50M limit: {total_params/1e6:.2f}M"
        
        budget_used = total_params / 50e6 * 100
        
        print(f"✅ Parameter budget:")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Budget used: {budget_used:.1f}%")
        print(f"   Remaining: {(50e6-total_params)/1e6:.2f}M")
    
    def test_trainable_vs_frozen(self, model):
        """Test frozen parameters configuration"""
        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)
        frozen_params = total_params - trainable_params
        
        # With freeze_dinov2=True, should have frozen params
        assert frozen_params > 0, "Should have frozen parameters"
        
        print(f"✅ Parameter breakdown:")
        print(f"   Total: {total_params/1e6:.2f}M")
        print(f"   Trainable: {trainable_params/1e6:.2f}M")
        print(f"   Frozen: {frozen_params/1e6:.2f}M")
    
    def test_inference_method(self, model, query_image, support_images):
        """Test high-level inference method"""
        model.set_reference_images(support_images)
        
        with torch.no_grad():
            outputs = model.inference(
                query_image,
                conf_thres=0.5,
                iou_thres=0.45
            )
        
        # Should have outputs
        assert len(outputs) > 0
        
        print("✅ Inference method successful")
    
    def test_error_handling_no_support(self, model, query_image):
        """Test error handling when lacks support images"""
        model.clear_cache()
        
        with pytest.raises(ValueError):
            # Should raise error: no support images and no cache
            model(query_image, use_cache=True)
        
        print("✅ Error handling verified")
    
    def test_gradient_flow_end_to_end(self, model, query_image, support_images, device):
        """Test gradients flow through complete pipeline"""
        model.train()
        
        # Enable gradients
        query_image = query_image.clone().requires_grad_(True)
        support_images = support_images.clone().requires_grad_(True)
        
        # Forward pass
        outputs = model(query_image, support_images)
        
        # Compute dummy loss
        loss = sum(s.sum() for s in outputs['prototype_sim'])
        
        loss.backward()
        
        # Check that trainable parameters have gradients
        has_grad = any(
            p.grad is not None 
            for p in model.parameters() 
            if p.requires_grad
        )
        
        assert has_grad, "Trainable parameters should have gradients"
        
        model.eval()
        print("✅ End-to-end gradient flow verified")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, model, query_image, support_images, device):
        """Test GPU memory usage is reasonable"""
        if device.type != 'cuda':
            pytest.skip("GPU not available")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference
        model.set_reference_images(support_images)
        
        with torch.no_grad():
            _ = model(query_image, use_cache=True)
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✅ GPU memory usage:")
        print(f"   Current: {memory_allocated:.2f} GB")
        print(f"   Peak: {max_memory:.2f} GB")
        
        # Should use less than 8GB (reasonable for Jetson Xavier NX)
        assert max_memory < 8.0, f"Memory usage too high: {max_memory:.2f} GB"
    
    def test_inference_timing(self, model, query_image, support_images, device):
        """Test inference timing benchmarks"""
        model.set_reference_images(support_images)
        model.eval()
        
        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                _ = model(query_image, use_cache=True)
        
        # Benchmark
        num_runs = 20
        times = []
        
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                _ = model(query_image, use_cache=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.time() - start) * 1000)  # ms
        
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time
        
        print(f"✅ Inference timing:")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Target: >25 FPS (40ms)")
        
        # On GPU, should be reasonably fast
        if device.type == 'cuda':
            assert avg_time < 100, f"Inference too slow: {avg_time:.2f}ms"
    
    def test_component_outputs(self, model, query_image, support_images):
        """Test that all components produce expected outputs"""
        # Extract intermediate outputs
        with torch.no_grad():
            # Support encoder - average multiple support images
            if support_images.shape[0] > 1:
                support_list = [support_images[i:i+1] for i in range(support_images.shape[0])]
                support_feats = model.support_encoder.compute_average_prototype(support_list)
            else:
                support_feats = model.support_encoder(support_images)
            assert 'prototype' in support_feats
            assert 'p3' in support_feats
            
            # Backbone
            query_feats = model.backbone(query_image)
            assert 'p3' in query_feats
            assert 'p4' in query_feats
            assert 'p5' in query_feats
            
            # PSALM Fusion (now includes P2 scale)
            fused = model.scs_fusion(query_feats, support_feats)
            assert len(fused) == 4  # P2, P3, P4, P5
            
            # Detection head - use scale-specific prototypes (Lego architecture)
            # Convert dict to list for detection head
            fused_list = [fused['p2'], fused['p3'], fused['p4'], fused['p5']]
            prototypes = {
                'p2': support_feats['p2'],  # 32-dim
                'p3': support_feats['p3'],  # 64-dim
                'p4': support_feats['p4'],  # 128-dim
                'p5': support_feats['p5'],  # 256-dim
            }
            box_preds, sim_scores = model.detection_head(fused_list, prototypes)
            assert len(box_preds) == 4  # P2, P3, P4, P5
            assert len(sim_scores) == 4  # P2, P3, P4, P5
        
        print("✅ All component outputs verified")


def test_module_imports():
    """Test that end-to-end module can be imported"""
    try:
        from models.yolo_refdet import YOLOv8nRefDet
        print("✅ Module import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
