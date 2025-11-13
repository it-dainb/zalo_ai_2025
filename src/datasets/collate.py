"""
Collate function for Reference-Based Detection batches.
Handles augmentation and batch preparation for both detection and triplet samples.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.augmentations.query_augmentation import QueryAugmentation
from src.augmentations.support_augmentation import SupportAugmentation
from src.augmentations.augmentation_config import AugmentationConfig


class RefDetCollator:
    """
    Collate function for episodic batches.
    Applies augmentations and prepares tensors for model input.
    """
    
    def __init__(
        self,
        config: AugmentationConfig,
        mode: str = 'train',
        stage: int = 2,
    ):
        """
        Args:
            config: Augmentation configuration
            mode: 'train' or 'val'
            stage: Training stage (1, 2, or 3)
        """
        self.config = config
        self.mode = mode
        self.stage = stage
        
        # Initialize augmentation pipelines
        self.query_aug = QueryAugmentation(
            img_size=config.query_img_size,
            mosaic_prob=config.query_mosaic_prob if mode == 'train' else 0.0,
            mixup_prob=config.query_mixup_prob if mode == 'train' else 0.0,
        )
        
        self.support_aug = SupportAugmentation(
            img_size=config.support_img_size,
            mode=config.support_mode if mode == 'train' else 'weak',
        )
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples from RefDetDataset
            
        Returns:
            dict with keys:
                - query_images: (B, 3, 640, 640)
                - support_images: (N, K, 3, 518, 518) where N is num classes
                - target_bboxes: List of (N_i, 4) tensors
                - target_classes: List of (N_i,) tensors (class indices within episode)
                - class_ids: (B,) tensor of original class IDs
                - video_ids: List of video_id strings
        """
        # Group by class (for episodic training)
        class_groups = {}
        for sample in batch:
            class_id = sample['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(sample)
        
        # Prepare support images (one set per class)
        support_images_list = []
        class_id_mapping = {}  # original class_id -> episodic class_id (0 to N-1)
        
        for episodic_idx, (class_id, samples) in enumerate(class_groups.items()):
            class_id_mapping[class_id] = episodic_idx
            
            # Use support images from first sample of this class
            support_imgs = samples[0]['support_images']
            
            # Augment each support image
            augmented_supports = []
            for img in support_imgs:
                # Convert to tensor and augment
                aug_img = self.support_aug(img)
                augmented_supports.append(aug_img)
            
            # Stack support images (K, 3, 518, 518)
            support_images_list.append(torch.stack(augmented_supports))
        
        # Prepare query images and targets
        query_images = []
        target_bboxes = []
        target_classes = []
        class_ids = []
        video_ids = []
        
        for sample in batch:
            query_frame = sample['query_frame']
            bboxes = sample['bboxes']
            class_id = sample['class_id']
            episodic_class_id = class_id_mapping[class_id]
            
            # Apply query augmentation
            if self.mode == 'train':
                # Augment with bboxes
                aug_result = self.query_aug(
                    query_frame,
                    bboxes=bboxes,
                    labels=np.array([episodic_class_id] * len(bboxes)),
                    apply_mosaic=False  # Disable mosaic for individual samples
                )
                aug_image = aug_result['image']
                aug_bboxes = aug_result['bboxes']
                aug_labels = aug_result['labels']
            else:
                # Validation: minimal augmentation (use same call but apply_mosaic=False)
                aug_result = self.query_aug(
                    query_frame,
                    bboxes=bboxes,
                    labels=np.array([episodic_class_id] * len(bboxes)),
                    apply_mosaic=False
                )
                aug_image = aug_result['image']
                aug_bboxes = aug_result['bboxes']
                aug_labels = aug_result['labels']
            
            # Ensure query image is float32 tensor
            if isinstance(aug_image, torch.Tensor):
                query_images.append(aug_image.float())
            else:
                query_images.append(torch.from_numpy(aug_image).float())
            
            # Handle both tensor and numpy array types
            if isinstance(aug_bboxes, torch.Tensor):
                target_bboxes.append(aug_bboxes.float())
            else:
                target_bboxes.append(torch.from_numpy(aug_bboxes).float())
            
            if isinstance(aug_labels, torch.Tensor):
                target_classes.append(aug_labels.long())
            else:
                target_classes.append(torch.from_numpy(aug_labels).long())
            class_ids.append(class_id)
            video_ids.append(sample['video_id'])
        
        # Stack query images
        query_images = torch.stack(query_images)  # (B, 3, 640, 640)
        
        # Stack support images (N, K, 3, 518, 518)
        support_images = torch.stack(support_images_list)
        
        return {
            'query_images': query_images,
            'support_images': support_images,
            'target_bboxes': target_bboxes,
            'target_classes': target_classes,
            'class_ids': torch.tensor(class_ids),
            'video_ids': video_ids,
            'num_classes': len(class_groups),
        }


def prepare_yolo_targets(
    bboxes_list: List[torch.Tensor],
    classes_list: List[torch.Tensor],
    img_size: int = 640,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare targets in YOLO format.
    
    Args:
        bboxes_list: List of (N_i, 4) bbox tensors (x1, y1, x2, y2) in pixel coords
        classes_list: List of (N_i,) class tensors
        img_size: Image size for normalization
        
    Returns:
        targets: (M, 6) tensor [batch_idx, class_id, x_center, y_center, w, h]
                 where x,y,w,h are normalized to [0,1]
        num_targets: Number of targets per image
    """
    targets = []
    
    for batch_idx, (bboxes, classes) in enumerate(zip(bboxes_list, classes_list)):
        if len(bboxes) == 0:
            continue
        
        # Convert (x1, y1, x2, y2) to (x_center, y_center, w, h)
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        
        # Normalize to [0, 1]
        x_center = x_center / img_size
        y_center = y_center / img_size
        w = w / img_size
        h = h / img_size
        
        # Create target tensor
        batch_indices = torch.full((len(bboxes),), batch_idx, dtype=torch.float32)
        
        target = torch.stack([
            batch_indices,
            classes.float(),
            x_center,
            y_center,
            w,
            h
        ], dim=1)
        
        targets.append(target)
    
    if len(targets) == 0:
        return torch.zeros((0, 6)), torch.zeros(len(bboxes_list), dtype=torch.long)
    
    targets = torch.cat(targets, dim=0)
    
    # Count targets per image
    num_targets = torch.zeros(len(bboxes_list), dtype=torch.long)
    for i, bboxes in enumerate(bboxes_list):
        num_targets[i] = len(bboxes)
    
    return targets, num_targets


def compute_dfl_targets(bboxes: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """
    Convert bbox coordinates to DFL (Distribution Focal Loss) targets.
    
    DFL represents each coordinate as a distribution over reg_max+1 bins,
    allowing the model to learn uncertainty in box predictions.
    
    Args:
        bboxes: (N, 4) tensor of normalized bboxes (x1, y1, x2, y2) in [0, 1]
        reg_max: Maximum regression value (YOLOv8n uses 16)
        
    Returns:
        dfl_targets: (N, 4) tensor of integer bin indices for each corner
    """
    # Scale coordinates to [0, reg_max] range
    # In DFL, coordinates are discretized into bins
    dfl_targets = (bboxes * reg_max).long()
    dfl_targets = torch.clamp(dfl_targets, 0, reg_max)
    
    return dfl_targets


class TripletCollator:
    """
    Collate function for triplet batches (contrastive learning).
    
    Handles augmentation and batch preparation for triplet samples:
    - Anchor: Support/reference image
    - Positive: Query frame with object
    - Negative: Background frame or cross-class frame
    
    Args:
        config: Augmentation configuration
        mode: 'train' or 'val'
        apply_strong_aug: Whether to apply strong augmentation for contrastive learning
    """
    
    def __init__(
        self,
        config: AugmentationConfig,
        mode: str = 'train',
        apply_strong_aug: bool = True,
    ):
        """
        Args:
            config: Augmentation configuration
            mode: 'train' or 'val'
            apply_strong_aug: If True, apply strong augmentations to all triplet components
        """
        self.config = config
        self.mode = mode
        self.apply_strong_aug = apply_strong_aug
        
        # Support augmentation (for anchors)
        support_mode = 'strong' if (mode == 'train' and apply_strong_aug) else 'weak'
        self.support_aug = SupportAugmentation(
            img_size=config.support_img_size,
            mode=support_mode,
        )
        
        # Query augmentation (for positives and negatives)
        self.query_aug = QueryAugmentation(
            img_size=config.query_img_size,
            mosaic_prob=0.0,  # Disable mosaic for triplet samples
            mixup_prob=0.0,   # Disable mixup for triplet samples
        )
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of triplet samples.
        
        Args:
            batch: List of triplet samples from TripletDataset
            
        Returns:
            dict with keys:
                - anchor_images: (B, 3, 518, 518) - Support images
                - positive_images: (B, 3, 640, 640) - Query frames with object
                - positive_bboxes: List of (N_i, 4) tensors
                - negative_images: (B, 3, 640, 640) - Background or cross-class frames
                - negative_bboxes: List of (M_i, 4) tensors (empty for background)
                - class_ids: (B,) tensor
                - negative_types: List of strings ('background' or 'cross_class')
        """
        anchor_images = []
        positive_images = []
        positive_bboxes = []
        negative_images = []
        negative_bboxes = []
        class_ids = []
        negative_types = []
        
        for sample in batch:
            # 1. Augment anchor (support image)
            anchor_img = self.support_aug(sample['anchor_image'])
            anchor_images.append(anchor_img)
            
            # 2. Augment positive (query frame with object)
            pos_bboxes = sample['positive_bboxes']
            pos_aug_result = self.query_aug(
                sample['positive_frame'],
                bboxes=pos_bboxes,
                labels=np.zeros(len(pos_bboxes)),  # Dummy labels
                apply_mosaic=False
            )
            
            if isinstance(pos_aug_result['image'], torch.Tensor):
                positive_images.append(pos_aug_result['image'].float())
            else:
                positive_images.append(torch.from_numpy(pos_aug_result['image']).float())
            
            if isinstance(pos_aug_result['bboxes'], torch.Tensor):
                positive_bboxes.append(pos_aug_result['bboxes'].float())
            else:
                positive_bboxes.append(torch.from_numpy(pos_aug_result['bboxes']).float())
            
            # 3. Augment negative (background or cross-class frame)
            neg_bboxes = sample['negative_bboxes']
            
            if len(neg_bboxes) == 0:
                # Background frame - no bboxes
                neg_aug_result = self.query_aug(
                    sample['negative_frame'],
                    bboxes=None,
                    labels=None,
                    apply_mosaic=False
                )
            else:
                # Cross-class frame - has bboxes
                neg_aug_result = self.query_aug(
                    sample['negative_frame'],
                    bboxes=neg_bboxes,
                    labels=np.zeros(len(neg_bboxes)),  # Dummy labels
                    apply_mosaic=False
                )
            
            if isinstance(neg_aug_result['image'], torch.Tensor):
                negative_images.append(neg_aug_result['image'].float())
            else:
                negative_images.append(torch.from_numpy(neg_aug_result['image']).float())
            
            if neg_aug_result.get('bboxes') is not None:
                if isinstance(neg_aug_result['bboxes'], torch.Tensor):
                    negative_bboxes.append(neg_aug_result['bboxes'].float())
                else:
                    negative_bboxes.append(torch.from_numpy(neg_aug_result['bboxes']).float())
            else:
                negative_bboxes.append(torch.zeros((0, 4)))
            
            class_ids.append(sample['class_id'])
            negative_types.append(sample['negative_type'])
        
        # Stack images
        anchor_images = torch.stack(anchor_images)      # (B, 3, 518, 518)
        positive_images = torch.stack(positive_images)  # (B, 3, 640, 640)
        negative_images = torch.stack(negative_images)  # (B, 3, 640, 640)
        
        return {
            'anchor_images': anchor_images,
            'positive_images': positive_images,
            'positive_bboxes': positive_bboxes,
            'negative_images': negative_images,
            'negative_bboxes': negative_bboxes,
            'class_ids': torch.tensor(class_ids),
            'negative_types': negative_types,
        }


class MixedCollator:
    """
    Collate function for mixed batches (detection + triplet).
    
    Automatically handles both detection and triplet samples in the same batch,
    routing them to the appropriate collator.
    
    Args:
        detection_collator: RefDetCollator for detection samples
        triplet_collator: TripletCollator for triplet samples
    """
    
    def __init__(
        self,
        detection_collator: RefDetCollator,
        triplet_collator: TripletCollator,
    ):
        self.detection_collator = detection_collator
        self.triplet_collator = triplet_collator
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate mixed batch.
        
        Args:
            batch: List of samples (detection or triplet)
            
        Returns:
            dict with 'batch_type' key and corresponding batch data
        """
        # Separate detection and triplet samples
        detection_samples = [s for s in batch if s.get('sample_type') == 'detection']
        triplet_samples = [s for s in batch if s.get('sample_type') == 'triplet']
        
        result = {}
        
        if detection_samples:
            detection_batch = self.detection_collator(detection_samples)
            result['detection'] = detection_batch
        
        if triplet_samples:
            triplet_batch = self.triplet_collator(triplet_samples)
            result['triplet'] = triplet_batch
        
        # Add batch composition info
        result['n_detection'] = len(detection_samples)
        result['n_triplet'] = len(triplet_samples)
        result['batch_type'] = 'mixed' if (detection_samples and triplet_samples) else \
                               ('detection' if detection_samples else 'triplet')
        
        return result
