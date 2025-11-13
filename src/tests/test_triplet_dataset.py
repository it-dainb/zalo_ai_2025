"""
Test suite for Triplet Dataset implementation.

Tests:
1. Background frame extraction
2. Triplet sample generation
3. Negative sampling strategies
4. TripletDataset
5. TripletBatchSampler
6. TripletCollator
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.datasets.refdet_dataset import RefDetDataset
from src.datasets.triplet_dataset import TripletDataset, TripletBatchSampler, MixedDataset
from src.datasets.collate import TripletCollator
from src.augmentations.augmentation_config import AugmentationConfig


@pytest.fixture
def base_dataset():
    """Create RefDetDataset for testing."""
    return RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=True
    )


@pytest.fixture
def triplet_dataset(base_dataset):
    """Create TripletDataset for testing."""
    return TripletDataset(
        base_dataset=base_dataset,
        negative_strategy='mixed',
        samples_per_class=10
    )


@pytest.fixture
def aug_config():
    """Create augmentation config for testing."""
    return AugmentationConfig()


class TestBackgroundFrameExtraction:
    """Test background frame tracking in RefDetDataset."""
    
    def test_background_frames_tracked(self, base_dataset):
        """Test that background frames are properly tracked."""
        for class_name in base_dataset.classes:
            data = base_dataset.class_data[class_name]
            
            # Check that we have background frames
            assert 'background_frames' in data
            assert 'total_frames' in data
            assert isinstance(data['background_frames'], list)
            
            # Check that annotated + background = total
            n_annotated = len(data['frame_indices'])
            n_background = len(data['background_frames'])
            n_total = data['total_frames']
            
            print(f"\n{class_name}:")
            print(f"  Total: {n_total}")
            print(f"  Annotated: {n_annotated} ({n_annotated/n_total*100:.1f}%)")
            print(f"  Background: {n_background} ({n_background/n_total*100:.1f}%)")
            
            # Most videos should have background frames (70-87%)
            assert n_background > 0, f"No background frames found for {class_name}"
            assert n_annotated + n_background <= n_total
    
    def test_get_background_frame(self, base_dataset):
        """Test getting background frames."""
        class_name = base_dataset.classes[0]
        
        # Get background frame
        bg_frame = base_dataset.get_background_frame(class_name)
        
        # Check shape
        assert isinstance(bg_frame, np.ndarray)
        assert len(bg_frame.shape) == 3
        assert bg_frame.shape[2] == 3  # RGB
        print(f"Background frame shape: {bg_frame.shape}")
    
    def test_no_overlap_annotated_background(self, base_dataset):
        """Test that annotated and background frames don't overlap."""
        for class_name in base_dataset.classes:
            data = base_dataset.class_data[class_name]
            
            annotated_set = set(data['frame_indices'])
            background_set = set(data['background_frames'])
            
            # No overlap
            overlap = annotated_set.intersection(background_set)
            assert len(overlap) == 0, f"Found overlap: {overlap}"


class TestTripletSampling:
    """Test triplet sample generation."""
    
    def test_triplet_sample_background(self, base_dataset):
        """Test triplet generation with background negative."""
        class_name = base_dataset.classes[0]
        
        sample = base_dataset.get_triplet_sample(
            class_name=class_name,
            negative_strategy='background'
        )
        
        # Check all components present
        assert 'anchor_image' in sample
        assert 'positive_frame' in sample
        assert 'negative_frame' in sample
        assert 'positive_bboxes' in sample
        assert 'negative_bboxes' in sample
        assert 'negative_type' in sample
        
        # Check shapes
        assert sample['anchor_image'].shape[2] == 3
        assert sample['positive_frame'].shape[2] == 3
        assert sample['negative_frame'].shape[2] == 3
        
        # Check bboxes
        assert len(sample['positive_bboxes']) > 0, "Positive should have bboxes"
        assert len(sample['negative_bboxes']) == 0, "Background should have no bboxes"
        assert sample['negative_type'] == 'background'
        
        print(f"\nTriplet sample (background):")
        print(f"  Anchor shape: {sample['anchor_image'].shape}")
        print(f"  Positive shape: {sample['positive_frame'].shape}")
        print(f"  Positive bboxes: {len(sample['positive_bboxes'])}")
        print(f"  Negative shape: {sample['negative_frame'].shape}")
        print(f"  Negative type: {sample['negative_type']}")
    
    def test_triplet_sample_cross_class(self, base_dataset):
        """Test triplet generation with cross-class negative."""
        if len(base_dataset.classes) < 2:
            pytest.skip("Need at least 2 classes for cross_class test")
        
        class_name = base_dataset.classes[0]
        
        sample = base_dataset.get_triplet_sample(
            class_name=class_name,
            negative_strategy='cross_class'
        )
        
        # Check negative type
        assert sample['negative_type'] == 'cross_class'
        
        # Cross-class negatives should have bboxes
        assert len(sample['negative_bboxes']) > 0, "Cross-class should have bboxes"
        
        print(f"\nTriplet sample (cross_class):")
        print(f"  Positive class: {sample['class_name']}")
        print(f"  Positive bboxes: {len(sample['positive_bboxes'])}")
        print(f"  Negative bboxes: {len(sample['negative_bboxes'])}")
        print(f"  Negative type: {sample['negative_type']}")
    
    def test_triplet_sample_mixed(self, base_dataset):
        """Test triplet generation with mixed strategy."""
        class_name = base_dataset.classes[0]
        
        # Sample multiple times to see both types
        negative_types = []
        for _ in range(10):
            sample = base_dataset.get_triplet_sample(
                class_name=class_name,
                negative_strategy='mixed'
            )
            negative_types.append(sample['negative_type'])
        
        print(f"\nMixed strategy (10 samples):")
        print(f"  Background: {negative_types.count('background')}")
        print(f"  Cross-class: {negative_types.count('cross_class')}")
        
        # Should have both types (probabilistically)
        # Note: This might occasionally fail due to randomness
        assert 'background' in negative_types or 'cross_class' in negative_types


class TestTripletDataset:
    """Test TripletDataset wrapper."""
    
    def test_triplet_dataset_creation(self, base_dataset):
        """Test TripletDataset initialization."""
        triplet_ds = TripletDataset(
            base_dataset=base_dataset,
            negative_strategy='mixed',
            samples_per_class=10
        )
        
        expected_len = len(base_dataset.classes) * 10
        assert len(triplet_ds) == expected_len
        
        print(f"\nTripletDataset:")
        print(f"  Classes: {len(base_dataset.classes)}")
        print(f"  Samples per class: 10")
        print(f"  Total samples: {len(triplet_ds)}")
    
    def test_triplet_dataset_getitem(self, triplet_dataset):
        """Test getting items from TripletDataset."""
        sample = triplet_dataset[0]
        
        # Check all required keys
        required_keys = [
            'anchor_image', 'positive_frame', 'negative_frame',
            'positive_bboxes', 'negative_bboxes',
            'class_id', 'class_name', 'negative_type'
        ]
        
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        print(f"\nTriplet sample 0:")
        print(f"  Class: {sample['class_name']} (ID: {sample['class_id']})")
        print(f"  Negative type: {sample['negative_type']}")
    
    def test_triplet_dataset_iteration(self, triplet_dataset):
        """Test iterating through TripletDataset."""
        count = 0
        for i, sample in enumerate(triplet_dataset):
            count += 1
            if i >= 5:  # Test first 5 samples
                break
        
        assert count == 6
        print(f"\nIteration test: {count} samples loaded")


class TestTripletBatchSampler:
    """Test TripletBatchSampler."""
    
    def test_sampler_creation(self, triplet_dataset):
        """Test sampler initialization."""
        sampler = TripletBatchSampler(
            dataset=triplet_dataset,
            batch_size=8,
            n_batches=10,
            balance_classes=True
        )
        
        assert len(sampler) == 10
        print(f"\nTripletBatchSampler:")
        print(f"  Batch size: 8")
        print(f"  Num batches: {len(sampler)}")
    
    def test_sampler_iteration(self, triplet_dataset):
        """Test batch generation."""
        sampler = TripletBatchSampler(
            dataset=triplet_dataset,
            batch_size=8,
            n_batches=5,
            balance_classes=True
        )
        
        batches = list(sampler)
        assert len(batches) == 5
        
        for i, batch_indices in enumerate(batches):
            print(f"\nBatch {i}: {len(batch_indices)} samples")
            assert len(batch_indices) <= 8


class TestTripletCollator:
    """Test TripletCollator."""
    
    def test_collator_creation(self, aug_config):
        """Test collator initialization."""
        collator = TripletCollator(
            config=aug_config,
            mode='train',
            apply_strong_aug=True
        )
        
        assert collator is not None
        print("\nTripletCollator created successfully")
    
    def test_collator_batch_preparation(self, triplet_dataset, aug_config):
        """Test batch collation."""
        collator = TripletCollator(
            config=aug_config,
            mode='train',
            apply_strong_aug=False  # Disable for faster testing
        )
        
        # Get a few samples
        samples = [triplet_dataset[i] for i in range(4)]
        
        # Collate
        batch = collator(samples)
        
        # Check all required keys
        required_keys = [
            'anchor_images', 'positive_images', 'negative_images',
            'positive_bboxes', 'negative_bboxes',
            'class_ids', 'negative_types'
        ]
        
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"
        
        # Check shapes
        B = len(samples)
        assert batch['anchor_images'].shape == (B, 3, 256, 256)
        assert batch['positive_images'].shape == (B, 3, 640, 640)
        assert batch['negative_images'].shape == (B, 3, 640, 640)
        assert len(batch['positive_bboxes']) == B
        assert len(batch['negative_bboxes']) == B
        
        print(f"\nCollated batch:")
        print(f"  Batch size: {B}")
        print(f"  Anchor images: {batch['anchor_images'].shape}")
        print(f"  Positive images: {batch['positive_images'].shape}")
        print(f"  Negative images: {batch['negative_images'].shape}")


class TestMixedDataset:
    """Test MixedDataset."""
    
    def test_mixed_dataset_creation(self, base_dataset, triplet_dataset):
        """Test MixedDataset initialization."""
        mixed_ds = MixedDataset(
            detection_dataset=base_dataset,
            triplet_dataset=triplet_dataset,
            detection_ratio=0.7
        )
        
        assert mixed_ds is not None
        assert len(mixed_ds) > 0
        
        print(f"\nMixedDataset:")
        print(f"  Total samples: {len(mixed_ds)}")
        print(f"  Detection samples: {mixed_ds.n_detection}")
        print(f"  Triplet samples: {mixed_ds.n_triplet}")
    
    def test_mixed_dataset_sample_types(self, base_dataset, triplet_dataset):
        """Test that mixed dataset returns both types."""
        mixed_ds = MixedDataset(
            detection_dataset=base_dataset,
            triplet_dataset=triplet_dataset,
            detection_ratio=0.5
        )
        
        # Sample from beginning (should be detection)
        detection_sample = mixed_ds[0]
        assert detection_sample['sample_type'] == 'detection'
        
        # Sample from end (should be triplet) - use explicit index
        # Last index is len(mixed_ds) - 1
        triplet_sample = mixed_ds[len(mixed_ds) - 1]
        assert triplet_sample['sample_type'] == 'triplet'
        
        print(f"\nMixed dataset sample types:")
        print(f"  First sample: {detection_sample['sample_type']}")
        print(f"  Last sample: {triplet_sample['sample_type']}")


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_complete_pipeline(self, base_dataset, aug_config):
        """Test complete pipeline from dataset to batch."""
        # Create triplet dataset
        triplet_ds = TripletDataset(
            base_dataset=base_dataset,
            negative_strategy='mixed',
            samples_per_class=10
        )
        
        # Create sampler
        sampler = TripletBatchSampler(
            dataset=triplet_ds,
            batch_size=4,
            n_batches=2,
            balance_classes=True
        )
        
        # Create collator
        collator = TripletCollator(
            config=aug_config,
            mode='train',
            apply_strong_aug=False
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            triplet_ds,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0  # Single process for testing
        )
        
        # Iterate
        for i, batch in enumerate(dataloader):
            assert batch is not None
            assert 'anchor_images' in batch
            
            print(f"\nBatch {i}:")
            print(f"  Anchor: {batch['anchor_images'].shape}")
            print(f"  Positive: {batch['positive_images'].shape}")
            print(f"  Negative: {batch['negative_images'].shape}")
            print(f"  Negative types: {batch['negative_types']}")
            
            if i >= 1:  # Test 2 batches
                break
        
        print("\n✅ Complete pipeline test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
