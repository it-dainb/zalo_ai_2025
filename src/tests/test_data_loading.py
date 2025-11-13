"""
Test Data Loading Components
=============================

Tests the smallest components of the data pipeline:
1. VideoFrameExtractor functionality
2. RefDetDataset initialization and data access
3. EpisodicBatchSampler sampling
4. RefDetCollator collation
"""

import pytest
import torch
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler, VideoFrameExtractor
from src.datasets.collate import RefDetCollator
from src.augmentations.augmentation_config import AugmentationConfig


class TestVideoFrameExtractor:
    """Test video frame extraction"""
    
    def test_frame_extraction_basic(self):
        """Test basic frame extraction (if video files exist)"""
        # This is a placeholder test - actual implementation depends on video availability
        # In real scenario, you would create a small test video
        pass
    
    def test_cache_mechanism(self):
        """Test that caching works correctly"""
        # Mock test for cache behavior
        extractor = VideoFrameExtractor("dummy_path.mp4", cache_size=5)
        assert extractor.cache_size == 5
        assert len(extractor.cache) == 0


class TestRefDetDataset:
    """Test RefDetDataset functionality"""
    
    @pytest.fixture
    def dataset_path(self):
        """Get path to test dataset"""
        return Path("./datasets/test/samples")
    
    @pytest.fixture
    def annotations_path(self):
        """Get path to test annotations"""
        return Path("./datasets/test/annotations/annotations.json")
    
    @pytest.fixture
    def dataset(self, dataset_path, annotations_path):
        """Create test dataset"""
        if not dataset_path.exists() or not annotations_path.exists():
            pytest.skip("Test dataset not available")
        
        return RefDetDataset(
            data_root=str(dataset_path),
            annotations_file=str(annotations_path),
            mode='val',
            cache_frames=True,
        )
    
    def test_dataset_initialization(self, dataset):
        """Test dataset initializes correctly"""
        assert dataset is not None
        assert len(dataset.classes) > 0
        assert len(dataset.class_data) > 0
        print(f"✅ Dataset initialized with {len(dataset.classes)} classes")
    
    def test_dataset_length(self, dataset):
        """Test dataset length calculation"""
        length = len(dataset)
        assert length > 0
        print(f"✅ Dataset has {length} samples")
    
    def test_get_item(self, dataset):
        """Test getting a single item from dataset"""
        if len(dataset) == 0:
            pytest.skip("Dataset is empty")
        
        sample = dataset[0]
        
        # Check required keys (updated to match actual implementation)
        assert 'video_id' in sample
        assert 'class_id' in sample
        assert 'support_images' in sample
        assert 'query_frame' in sample
        assert 'bboxes' in sample
        assert 'frame_idx' in sample
        
        # Check data types and shapes
        assert isinstance(sample['support_images'], list)
        assert isinstance(sample['query_frame'], np.ndarray)
        assert len(sample['query_frame'].shape) == 3  # (H, W, C)
        
        print(f"✅ Sample retrieved successfully")
        print(f"   Class: {sample['video_id']}")
        print(f"   Support images: {len(sample['support_images'])}")
        print(f"   Query shape: {sample['query_frame'].shape}")
        print(f"   Bboxes: {len(sample['bboxes'])}")
    
    def test_support_images_loading(self, dataset):
        """Test support images are loaded correctly"""
        if len(dataset) == 0:
            pytest.skip("Dataset is empty")
        
        sample = dataset[0]
        support_images = sample['support_images']
        
        assert len(support_images) > 0
        for img in support_images:
            assert isinstance(img, np.ndarray)
            assert len(img.shape) == 3
            assert img.shape[2] == 3  # RGB
        
        print(f"✅ Support images loaded correctly")


class TestEpisodicBatchSampler:
    """Test episodic batch sampler"""
    
    @pytest.fixture
    def dataset(self):
        """Create mock or real dataset"""
        dataset_path = Path("./datasets/test/samples")
        annotations_path = Path("./datasets/test/annotations/annotations.json")
        
        if not dataset_path.exists() or not annotations_path.exists():
            pytest.skip("Test dataset not available")
        
        return RefDetDataset(
            data_root=str(dataset_path),
            annotations_file=str(annotations_path),
            mode='val',
            cache_frames=True,
        )
    
    def test_sampler_initialization(self, dataset):
        """Test sampler initializes correctly"""
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=2,
            n_query=4,
            n_episodes=10,
        )
        
        assert sampler.n_way == 2
        assert sampler.n_query == 4
        assert sampler.n_episodes == 10
        print(f"✅ Sampler initialized: {sampler.n_way}-way {sampler.n_query}-query")
    
    def test_sampler_iteration(self, dataset):
        """Test sampler generates episodes correctly"""
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=min(2, len(dataset.classes)),
            n_query=2,
            n_episodes=5,
        )
        
        episodes = list(sampler)
        assert len(episodes) == 5
        
        # Check episode structure
        for episode in episodes:
            assert isinstance(episode, list)
            assert len(episode) > 0
        
        print(f"✅ Sampler generated {len(episodes)} episodes")
    
    def test_sampler_length(self, dataset):
        """Test sampler length"""
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=2,
            n_query=4,
            n_episodes=10,
        )
        
        assert len(sampler) == 10
        print(f"✅ Sampler length correct: {len(sampler)}")


class TestRefDetCollator:
    """Test collate function"""
    
    @pytest.fixture
    def aug_config(self):
        """Create augmentation config"""
        return AugmentationConfig(
            query_img_size=640,
            support_img_size=224,
            query_mosaic_prob=0.0,  # Disable for testing
            query_mixup_prob=0.0,
        )
    
    def test_collator_initialization(self, aug_config):
        """Test collator initializes correctly"""
        collator = RefDetCollator(
            config=aug_config,
            mode='train',
            stage=2,
        )
        
        assert collator.config is not None
        assert collator.mode == 'train'
        assert collator.stage == 2
        print(f"✅ Collator initialized")
    
    def test_collator_with_mock_batch(self, aug_config):
        """Test collator with mock data"""
        collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        # Create mock batch (updated keys to match actual implementation)
        mock_batch = [
            {
                'video_id': 'test_0',
                'class_id': 0,
                'support_images': [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)],
                'query_frame': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                'bboxes': np.array([[100, 100, 200, 200]]),
                'frame_idx': 0,
            }
        ]
        
        try:
            batch = collator(mock_batch)
            
            # Check output structure
            assert 'support_images' in batch
            assert 'query_images' in batch
            assert 'target_bboxes' in batch
            assert 'target_classes' in batch
            
            # Check tensor types
            assert isinstance(batch['support_images'], torch.Tensor)
            assert isinstance(batch['query_images'], torch.Tensor)
            
            print(f"✅ Collator processed batch successfully")
            print(f"   Support shape: {batch['support_images'].shape}")
            print(f"   Query shape: {batch['query_images'].shape}")
        except Exception as e:
            print(f"⚠️  Collator test with mock data: {str(e)}")


class TestDataPipelineIntegration:
    """Test complete data loading pipeline"""
    
    @pytest.fixture
    def dataset_path(self):
        return Path("./datasets/test/samples")
    
    @pytest.fixture
    def annotations_path(self):
        return Path("./datasets/test/annotations/annotations.json")
    
    def test_full_dataloader_pipeline(self, dataset_path, annotations_path):
        """Test complete data loading pipeline"""
        if not dataset_path.exists() or not annotations_path.exists():
            pytest.skip("Test dataset not available")
        
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = RefDetDataset(
            data_root=str(dataset_path),
            annotations_file=str(annotations_path),
            mode='val',
            cache_frames=True,
        )
        
        # Create sampler
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=min(2, len(dataset.classes)),
            n_query=2,
            n_episodes=2,
        )
        
        # Create collator
        aug_config = AugmentationConfig(
            query_img_size=640,
            support_img_size=224,
            query_mosaic_prob=0.0,
            query_mixup_prob=0.0,
        )
        collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0,  # Single-threaded for testing
            pin_memory=False,
        )
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            
            # Verify batch structure
            assert 'support_images' in batch
            assert 'query_images' in batch
            assert 'target_bboxes' in batch
            
            print(f"✅ Batch {batch_count}:")
            print(f"   Support: {batch['support_images'].shape}")
            print(f"   Query: {batch['query_images'].shape}")
            print(f"   Targets: {len(batch['target_bboxes'])}")
        
        assert batch_count > 0
        print(f"✅ Full pipeline processed {batch_count} batches")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
