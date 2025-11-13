"""
Triplet Dataset Wrapper for Contrastive Learning in Few-Shot Detection

This module provides dataset and sampler for triplet-based contrastive learning,
addressing the limitation of training only on frames with bounding boxes.

Key Features:
- Includes negative samples (background frames)
- Supports multiple negative sampling strategies
- Enables better real-world performance with background/empty frame handling
"""

import numpy as np
import random
from torch.utils.data import Dataset, Sampler
from typing import Dict, List
from .refdet_dataset import RefDetDataset


class TripletDataset(Dataset):
    """
    Wrapper around RefDetDataset for triplet sampling.
    
    Each sample is a triplet:
    - Anchor: Support/reference image
    - Positive: Query frame with same object
    - Negative: Background frame OR frame from different class
    
    This addresses the limitation of training only on positive samples
    by including negative samples (background frames without objects).
    
    Args:
        base_dataset: RefDetDataset instance
        negative_strategy: Negative sampling strategy
            - 'background': Use background frames (no objects) as negatives
            - 'cross_class': Use frames from different classes as hard negatives
            - 'mixed': Mix both strategies (50/50)
        samples_per_class: Number of triplet samples to generate per class per epoch
    """
    
    def __init__(
        self,
        base_dataset: RefDetDataset,
        negative_strategy: str = 'mixed',
        samples_per_class: int = 100,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.negative_strategy = negative_strategy
        self.samples_per_class = samples_per_class
        
        # Validate strategy
        valid_strategies = ['background', 'cross_class', 'mixed']
        if negative_strategy not in valid_strategies:
            raise ValueError(f"negative_strategy must be one of {valid_strategies}")
        
        self.classes = base_dataset.classes
        self.total_samples = len(self.classes) * samples_per_class
        
        print(f"\nTripletDataset initialized:")
        print(f"  Negative strategy: {negative_strategy}")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Total samples per epoch: {self.total_samples}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a triplet sample.
        
        Returns:
            dict with:
                - anchor_image: (H, W, 3) numpy array
                - positive_frame: (H, W, 3) numpy array
                - positive_bboxes: (N, 4) numpy array
                - negative_frame: (H, W, 3) numpy array
                - negative_bboxes: (M, 4) numpy array (empty for background)
                - class_id: int
                - class_name: str
                - negative_type: 'background' or 'cross_class'
        """
        # Map index to class
        class_idx = idx % len(self.classes)
        class_name = self.classes[class_idx]
        
        # Get triplet sample from base dataset
        return self.base_dataset.get_triplet_sample(
            class_name=class_name,
            negative_strategy=self.negative_strategy,
        )


class TripletBatchSampler(Sampler):
    """
    Batch sampler for triplet training.
    
    Creates balanced batches with samples from different classes.
    Each batch contains triplets from multiple classes for better
    contrastive learning.
    
    Args:
        dataset: TripletDataset instance
        batch_size: Number of triplets per batch
        n_batches: Number of batches per epoch
        balance_classes: If True, ensure each batch has samples from different classes
    """
    
    def __init__(
        self,
        dataset: TripletDataset,
        batch_size: int = 16,
        n_batches: int = 100,
        balance_classes: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.balance_classes = balance_classes
        
        self.num_classes = len(dataset.classes)
        self.samples_per_class = dataset.samples_per_class
    
    def __iter__(self):
        """Generate batch indices."""
        for _ in range(self.n_batches):
            if self.balance_classes and self.batch_size >= self.num_classes:
                # Ensure each batch has samples from different classes
                samples_per_class_in_batch = self.batch_size // self.num_classes
                remainder = self.batch_size % self.num_classes
                
                batch_indices = []
                for class_idx in range(self.num_classes):
                    # Calculate index range for this class
                    start_idx = class_idx * self.samples_per_class
                    end_idx = start_idx + self.samples_per_class
                    
                    # Sample from this class
                    n_samples = samples_per_class_in_batch
                    if class_idx < remainder:
                        n_samples += 1
                    
                    class_indices = random.sample(
                        range(start_idx, end_idx),
                        min(n_samples, self.samples_per_class)
                    )
                    batch_indices.extend(class_indices)
            else:
                # Random sampling
                batch_indices = random.sample(
                    range(len(self.dataset)),
                    min(self.batch_size, len(self.dataset))
                )
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.n_batches


class MixedDataset(Dataset):
    """
    Mixed dataset combining standard detection and triplet samples.
    
    This allows training with both:
    - Standard detection samples (support + query with bboxes)
    - Triplet samples (anchor + positive + negative)
    
    Args:
        detection_dataset: RefDetDataset for standard detection
        triplet_dataset: TripletDataset for contrastive learning
        detection_ratio: Ratio of detection samples (0.0 to 1.0)
            - 1.0: Only detection samples
            - 0.5: 50% detection, 50% triplet
            - 0.0: Only triplet samples
    """
    
    def __init__(
        self,
        detection_dataset: RefDetDataset,
        triplet_dataset: TripletDataset,
        detection_ratio: float = 0.7,
    ):
        super().__init__()
        self.detection_dataset = detection_dataset
        self.triplet_dataset = triplet_dataset
        self.detection_ratio = detection_ratio
        
        if not 0.0 <= detection_ratio <= 1.0:
            raise ValueError("detection_ratio must be between 0.0 and 1.0")
        
        # Calculate total length
        total_detection = len(detection_dataset)
        total_triplet = len(triplet_dataset)
        
        self.n_detection = int(total_detection * detection_ratio / (1.0 if detection_ratio == 1.0 else detection_ratio))
        self.n_triplet = int(total_triplet * (1.0 - detection_ratio) / (1.0 if detection_ratio == 0.0 else (1.0 - detection_ratio)))
        
        self.total_length = self.n_detection + self.n_triplet
        
        print(f"\nMixedDataset initialized:")
        print(f"  Detection samples: {self.n_detection} ({detection_ratio*100:.1f}%)")
        print(f"  Triplet samples: {self.n_triplet} ({(1-detection_ratio)*100:.1f}%)")
        print(f"  Total samples: {self.total_length}")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample (either detection or triplet).
        
        Returns:
            dict with 'sample_type' key ('detection' or 'triplet')
            and corresponding sample data
        """
        if idx < self.n_detection:
            # Detection sample
            det_idx = idx % len(self.detection_dataset)
            sample = self.detection_dataset[det_idx]
            sample['sample_type'] = 'detection'
            return sample
        else:
            # Triplet sample
            trip_idx = (idx - self.n_detection) % len(self.triplet_dataset)
            sample = self.triplet_dataset[trip_idx]
            sample['sample_type'] = 'triplet'
            return sample
