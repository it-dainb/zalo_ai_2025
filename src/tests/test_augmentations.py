"""
Comprehensive tests for augmentation modules.
Tests query, support, temporal augmentations and configuration.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentations.query_augmentation import (
    QueryAugmentation,
    MosaicAugmentation,
    MixUpAugmentation,
    get_query_augmentation
)
from augmentations.support_augmentation import (
    SupportAugmentation,
    FeatureSpaceAugmentation,
    ContrastiveAugmentation,
    get_support_augmentation
)
from augmentations.temporal_augmentation import (
    TemporalConsistentAugmentation,
    VideoFrameSampler,
    get_temporal_augmentation
)
from augmentations.augmentation_config import (
    AugmentationConfig,
    get_stage_config,
    get_yolov8_augmentation_params
)


class TestMosaicAugmentation:
    """Test mosaic augmentation for query images."""
    
    def test_mosaic_init(self):
        """Test mosaic augmentation initialization."""
        mosaic = MosaicAugmentation(img_size=640, prob=1.0)
        assert mosaic.img_size == 640
        assert mosaic.prob == 1.0
    
    def test_mosaic_augmentation(self):
        """Test mosaic augmentation with 4 images."""
        mosaic = MosaicAugmentation(img_size=640, prob=1.0)
        
        # Create 4 dummy images
        images = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        bboxes_list = [
            np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32),
            np.array([[150, 150, 250, 250]], dtype=np.float32),
            np.array([[50, 50, 150, 150], [500, 500, 600, 600]], dtype=np.float32),
            np.array([[200, 200, 300, 300]], dtype=np.float32)
        ]
        labels_list = [
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([1], dtype=np.int64)
        ]
        
        mosaic_img, mosaic_bboxes, mosaic_labels = mosaic(images, bboxes_list, labels_list)
        
        # Check output shape
        assert mosaic_img.shape == (640, 640, 3)
        assert mosaic_bboxes.shape[1] == 4  # [N, 4]
        assert len(mosaic_labels) == len(mosaic_bboxes)
        
        # Check bboxes are within image bounds
        assert np.all(mosaic_bboxes[:, [0, 2]] >= 0)
        assert np.all(mosaic_bboxes[:, [0, 2]] <= 640)
        assert np.all(mosaic_bboxes[:, [1, 3]] >= 0)
        assert np.all(mosaic_bboxes[:, [1, 3]] <= 640)
        
        print(f"✓ Mosaic created {len(mosaic_bboxes)} boxes from 4 images")
    
    def test_mosaic_probability(self):
        """Test mosaic probability control."""
        mosaic = MosaicAugmentation(img_size=640, prob=0.0)
        
        images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]
        bboxes_list = [np.array([[100, 100, 200, 200]], dtype=np.float32) for _ in range(4)]
        labels_list = [np.array([0], dtype=np.int64) for _ in range(4)]
        
        result_img, result_bboxes, result_labels = mosaic(images, bboxes_list, labels_list)
        
        # With prob=0.0, should return first image unchanged
        assert np.array_equal(result_img, images[0])
        print("✓ Mosaic probability control works")


class TestMixUpAugmentation:
    """Test mixup augmentation."""
    
    def test_mixup_init(self):
        """Test mixup initialization."""
        mixup = MixUpAugmentation(alpha=0.1, prob=0.15)
        assert mixup.alpha == 0.1
        assert mixup.prob == 0.15
    
    def test_mixup_augmentation(self):
        """Test mixup blending."""
        mixup = MixUpAugmentation(alpha=32.0, prob=1.0)
        
        img1 = np.ones((640, 640, 3), dtype=np.uint8) * 100
        img2 = np.ones((640, 640, 3), dtype=np.uint8) * 200
        bboxes1 = np.array([[100, 100, 200, 200]], dtype=np.float32)
        bboxes2 = np.array([[300, 300, 400, 400]], dtype=np.float32)
        labels1 = np.array([0], dtype=np.int64)
        labels2 = np.array([1], dtype=np.int64)
        
        mixed_img, mixed_bboxes, mixed_labels = mixup(
            img1, bboxes1, labels1, img2, bboxes2, labels2
        )
        
        # Check outputs
        assert mixed_img.shape == (640, 640, 3)
        assert len(mixed_bboxes) == 2  # Combined from both images
        assert len(mixed_labels) == 2
        # Image should be a blend (not purely 100 or 200)
        assert mixed_img.mean() > 100 and mixed_img.mean() < 200
        
        print(f"✓ MixUp blended images successfully")


class TestQueryAugmentation:
    """Test complete query augmentation pipeline."""
    
    def test_query_aug_stages(self):
        """Test stage-specific query augmentation."""
        for stage in ["stage1", "stage2", "stage3"]:
            aug = get_query_augmentation(stage=stage, img_size=640)
            assert aug.stage == stage
            assert aug.img_size == 640
            print(f"✓ Created {stage} query augmentation")
    
    def test_query_augmentation_forward(self):
        """Test query augmentation forward pass."""
        aug = QueryAugmentation(img_size=640, stage="stage1")
        
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        
        result = aug(image, bboxes, labels, apply_mosaic=False)
        
        # Check outputs
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 640, 640)
        assert isinstance(result['bboxes'], torch.Tensor)
        assert isinstance(result['labels'], torch.Tensor)
        
        print("✓ Query augmentation forward pass successful")
    
    def test_query_aug_with_no_boxes(self):
        """Test query augmentation with empty bboxes."""
        aug = QueryAugmentation(img_size=640, stage="stage1")
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        bboxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
        
        result = aug(image, bboxes, labels)
        
        assert result['image'].shape == (3, 640, 640)
        assert len(result['bboxes']) == 0
        assert len(result['labels']) == 0
        
        print("✓ Query augmentation handles empty bboxes")


class TestSupportAugmentation:
    """Test support augmentation for reference images."""
    
    def test_support_aug_modes(self):
        """Test weak and strong support augmentation."""
        weak_aug = get_support_augmentation(mode="weak", img_size=224)
        strong_aug = get_support_augmentation(mode="strong", img_size=224)
        
        assert weak_aug.mode == "weak"
        assert strong_aug.mode == "strong"
        print("✓ Created weak and strong support augmentations")
    
    def test_support_augmentation_forward(self):
        """Test support augmentation forward pass."""
        aug = SupportAugmentation(img_size=224, mode="weak")
        
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = aug(image)
        
        # Check output
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        # Check normalization (should be roughly in [-2, 2] range after ImageNet normalization)
        assert result.min() > -3 and result.max() < 3
        
        print("✓ Support augmentation forward pass successful")
    
    def test_support_aug_batch(self):
        """Test batch augmentation for support images."""
        aug = SupportAugmentation(img_size=224, mode="weak")
        
        images = [np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8) for _ in range(3)]
        result = aug.augment_batch(images)
        
        assert result.shape == (3, 3, 224, 224)
        print("✓ Support batch augmentation successful")


class TestFeatureSpaceAugmentation:
    """Test feature-space augmentation."""
    
    def test_feature_aug_init(self):
        """Test feature augmentation initialization."""
        aug = FeatureSpaceAugmentation(feature_dim=384, noise_std=0.1, dropout_rate=0.1)
        assert aug.feature_dim == 384
        assert aug.noise_std == 0.1
    
    def test_feature_augmentation(self):
        """Test feature augmentation forward."""
        aug = FeatureSpaceAugmentation(feature_dim=384)
        
        features = torch.randn(8, 384)  # Batch of 8 features
        
        # Training mode
        aug_features = aug(features, training=True)
        assert aug_features.shape == (8, 384)
        # Check L2 normalization
        norms = torch.norm(aug_features, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # Inference mode (should return unchanged, normalized)
        aug.eval()
        aug_features = aug(features, training=False)
        assert torch.allclose(features, aug_features)
        
        print("✓ Feature space augmentation works")
    
    def test_prototype_augmentation(self):
        """Test prototype augmentation."""
        aug = FeatureSpaceAugmentation(feature_dim=384)
        
        prototypes = torch.randn(5, 384)  # 5 class prototypes
        aug_prototypes = aug.augment_prototypes(prototypes, num_augmentations=3)
        
        assert aug_prototypes.shape == (15, 384)  # 5 * 3
        print("✓ Prototype augmentation creates multiple versions")


class TestContrastiveAugmentation:
    """Test contrastive augmentation for support images."""
    
    def test_contrastive_aug(self):
        """Test dual-view contrastive augmentation."""
        aug = ContrastiveAugmentation(img_size=224)
        
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        views = aug(image)
        
        assert 'view1' in views and 'view2' in views
        assert views['view1'].shape == (3, 224, 224)
        assert views['view2'].shape == (3, 224, 224)
        # Views should be different (augmented)
        assert not torch.equal(views['view1'], views['view2'])
        
        print("✓ Contrastive augmentation creates two views")
    
    def test_contrastive_batch(self):
        """Test batch contrastive augmentation."""
        aug = ContrastiveAugmentation(img_size=224)
        
        images = [np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8) for _ in range(3)]
        views = aug.augment_batch(images)
        
        assert views['view1'].shape == (3, 3, 224, 224)
        assert views['view2'].shape == (3, 3, 224, 224)
        print("✓ Batch contrastive augmentation successful")


class TestTemporalConsistentAugmentation:
    """Test temporal consistency for video frames."""
    
    def test_temporal_aug_init(self):
        """Test temporal augmentation initialization."""
        aug = get_temporal_augmentation(stage="stage1", img_size=640, consistency_window=8)
        assert aug.img_size == 640
        assert aug.consistency_window == 8
    
    def test_temporal_consistency(self):
        """Test that augmentation parameters stay consistent across frames."""
        aug = TemporalConsistentAugmentation(img_size=640, stage="stage1", consistency_window=5)
        
        # Reset to start fresh
        aug.reset()
        
        # Process frames and check that augmentation actually occurs
        results = []
        for i in range(10):
            image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
            
            result = aug(image, bboxes, labels)
            assert result['image'].shape == (3, 640, 640)
            results.append(result)
        
        # Test that cache is cleared after consistency window
        aug.reset()
        aug.frame_counter = 0
        
        # Get params twice within window
        params1 = aug._sample_augmentation_params()
        aug.cached_params = params1
        aug.frame_counter = 2
        
        # Should return same params (from cache)
        returned_params = aug._get_params()
        assert aug.frame_counter == 3  # Should have incremented
        
        # After window expires, should generate new params
        aug.frame_counter = 10  # Beyond consistency_window
        new_params = aug._get_params()
        assert aug.frame_counter == 1  # Should have reset
        
        print("✓ Temporal consistency mechanism works correctly")
    
    def test_temporal_reset(self):
        """Test resetting temporal augmentation."""
        aug = TemporalConsistentAugmentation(img_size=640, stage="stage1")
        
        # Get params
        params1 = aug._get_params()
        aug.frame_counter = 100
        
        # Reset
        aug.reset()
        assert aug.cached_params is None
        assert aug.frame_counter == 0
        
        print("✓ Temporal augmentation reset works")


class TestVideoFrameSampler:
    """Test video frame sampling."""
    
    def test_sampler_init(self):
        """Test sampler initialization."""
        sampler = VideoFrameSampler(frame_stride=2, sequence_length=8, overlap=4)
        assert sampler.frame_stride == 2
        assert sampler.sequence_length == 8
        assert sampler.overlap == 4
    
    def test_sample_frames(self):
        """Test frame sequence sampling."""
        sampler = VideoFrameSampler(frame_stride=1, sequence_length=8, overlap=4)
        
        # Create dummy video (30 frames)
        video_frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(30)]
        annotations = [{'bboxes': np.array([[100, 100, 200, 200]]), 'labels': np.array([0])} for _ in range(30)]
        
        sequences = sampler.sample_frames(video_frames, annotations)
        
        # Check sequences
        assert len(sequences) > 0
        for seq_frames, seq_annots in sequences:
            assert len(seq_frames) == 8
            assert len(seq_annots) == 8
        
        print(f"✓ Sampled {len(sequences)} sequences from 30 frames")
    
    def test_sample_random_sequence(self):
        """Test random sequence sampling."""
        sampler = VideoFrameSampler(sequence_length=8)
        
        video_frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(20)]
        annotations = [{'bboxes': np.array([[100, 100, 200, 200]])} for _ in range(20)]
        
        seq_frames, seq_annots = sampler.sample_random_sequence(video_frames, annotations)
        
        assert len(seq_frames) == 8
        assert len(seq_annots) == 8
        print("✓ Random sequence sampling successful")


class TestAugmentationConfig:
    """Test augmentation configuration."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = AugmentationConfig()
        assert config.query_img_size == 640
        assert config.support_img_size == 256
        print("✓ Default configuration created")
    
    def test_stage_configs(self):
        """Test stage-specific configurations."""
        for stage in ["stage1", "stage2", "stage3"]:
            config = get_stage_config(stage)
            assert isinstance(config, AugmentationConfig)
            print(f"✓ {stage} configuration created")
        
        # Verify progressive reduction
        stage1 = get_stage_config("stage1")
        stage2 = get_stage_config("stage2")
        stage3 = get_stage_config("stage3")
        
        assert stage1.query_mosaic_prob >= stage2.query_mosaic_prob >= stage3.query_mosaic_prob
        assert stage1.blur_prob >= stage2.blur_prob >= stage3.blur_prob
        print("✓ Progressive augmentation reduction verified")
    
    def test_yolov8_params(self):
        """Test YOLOv8 parameter conversion."""
        params = get_yolov8_augmentation_params("stage1")
        
        assert 'mosaic' in params
        assert 'mixup' in params
        assert 'hsv_h' in params
        assert 'degrees' in params
        
        # Check value ranges
        assert 0 <= params['mosaic'] <= 1
        assert 0 <= params['mixup'] <= 1
        
        print("✓ YOLOv8 parameters generated correctly")
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = get_stage_config("stage1")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'query_img_size' in config_dict
        assert 'support_img_size' in config_dict
        print("✓ Configuration serialization works")


def test_integration():
    """Integration test: full augmentation pipeline."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Full Augmentation Pipeline")
    print("="*80)
    
    # Stage 1 config
    config = get_stage_config("stage1")
    
    # Query augmentation
    query_aug = get_query_augmentation(stage="stage1")
    query_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    query_bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    query_labels = np.array([0], dtype=np.int64)
    query_result = query_aug(query_img, query_bboxes, query_labels, apply_mosaic=False)
    
    # Support augmentation
    support_aug = get_support_augmentation(mode="weak")
    support_imgs = [np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8) for _ in range(3)]
    support_result = support_aug.augment_batch(support_imgs)
    
    # Feature space augmentation
    feature_aug = FeatureSpaceAugmentation(feature_dim=384)
    features = torch.randn(3, 384)
    aug_features = feature_aug(features, training=True)
    
    # Temporal augmentation
    temporal_aug = get_temporal_augmentation(stage="stage1")
    temporal_result = temporal_aug(query_img, query_bboxes, query_labels)
    
    print(f"✓ Query augmentation output: {query_result['image'].shape}")
    print(f"✓ Support augmentation output: {support_result.shape}")
    print(f"✓ Feature augmentation output: {aug_features.shape}")
    print(f"✓ Temporal augmentation output: {temporal_result['image'].shape}")
    print("\n✓✓✓ All augmentation components integrated successfully!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING AUGMENTATION TESTS")
    print("="*80 + "\n")
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run integration test
    test_integration()
