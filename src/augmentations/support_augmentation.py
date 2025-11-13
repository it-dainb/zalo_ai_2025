"""
Support Path Augmentation (Reference Images)
Conservative augmentation to preserve prototype quality for DINOv2.

UPGRADED: Now uses AlbumentationsX for 10-23x faster augmentation
while preserving semantic consistency for prototype matching.
Uses LetterBox resizing for proper aspect ratio preservation.
"""

import torch
import torch.nn as nn
import numpy as np
import albumentations as A  
from albumentations.pytorch import ToTensorV2
from typing import Dict, List
import cv2


class LetterBoxSupport:
    """
    LetterBox resizing for support images (DINOv2).
    Preserves aspect ratio and uses padding.
    """
    
    def __init__(
        self,
        new_shape: int = 518,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Args:
            new_shape: Target size (518 for DINOv2)
            padding_value: Value for padding
            interpolation: cv2 interpolation method
        """
        self.new_shape = new_shape
        self.padding_value = padding_value
        self.interpolation = interpolation
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and pad image for DINOv2.
        
        Args:
            image: Input image [H, W, 3]
            
        Returns:
            Resized and padded image [new_shape, new_shape, 3]
        """
        shape = image.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(self.new_shape / shape[0], self.new_shape / shape[1])
        
        # Compute new size
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = self.new_shape - new_unpad[0]
        dh = self.new_shape - new_unpad[1]
        
        # Center padding
        dw /= 2
        dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)
            if image.ndim == 2:
                image = image[..., None]
        
        # Add padding
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        
        if image.shape[2] == 3:
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:
            h, w, c = image.shape
            pad_img = np.full(
                (h + top + bottom, w + left + right, c),
                fill_value=self.padding_value,
                dtype=image.dtype
            )
            pad_img[top:top + h, left:left + w] = image
            image = pad_img
        
        return image


class SupportAugmentation:
    """
    Conservative augmentation for support/reference images.
    Preserves semantic consistency for prototype matching.
    """
    
    def __init__(
        self,
        img_size: int = 518,  # DINOv2 input size
        mode: str = "weak",  # "weak" or "strong"
    ):
        """
        Args:
            img_size: Target size for DINOv2 (518x518)
            mode: "weak" for training, "strong" for contrastive learning
        """
        self.img_size = img_size
        self.mode = mode
        
        if mode == "weak":
            self.transform = self._get_weak_transform()
        else:
            self.transform = self._get_strong_transform()
    
    def _get_weak_transform(self):
        """
        Weak augmentation - preserves semantic content.
        Use for: Training Stage 2-3, inference preprocessing
        UPGRADED: Uses AlbumentationsX for faster processing
        Note: LetterBox resizing is applied separately in __call__
        """
        return A.Compose([
            A.HorizontalFlip(p=0.3),  # Only horizontal, not vertical (preserves orientation)
            A.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.85, 1.0),  # Conservative crop
                ratio=(0.9, 1.1),
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats (DINOv2 standard)
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _get_strong_transform(self):
        """
        Strong augmentation - for contrastive learning positive pairs.
        Use for: Supervised Contrastive Loss training only
        UPGRADED: Uses AlbumentationsX for faster processing
        Note: LetterBox resizing is applied separately in __call__
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                p=0.5
            ),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.1),  # Reduced noise for support images
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply augmentation to support image with LetterBox resizing.
        
        Args:
            image: Input image [H, W, 3] in RGB format
            
        Returns:
            Augmented image tensor [3, img_size, img_size]
        """
        # Apply LetterBox resizing first
        letterbox = LetterBoxSupport(
            new_shape=self.img_size,
            padding_value=114
        )
        image = letterbox(image)
        
        # Apply augmentations
        transformed = self.transform(image=image)
        return transformed['image']
    
    def augment_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Augment a batch of support images.
        
        Args:
            images: List of K support images
            
        Returns:
            Batch tensor [K, 3, 224, 224]
        """
        augmented = [self(img) for img in images]
        return torch.stack(augmented, dim=0)


class FeatureSpaceAugmentation(nn.Module):
    """
    Feature-space augmentation for few-shot learning.
    Augments embeddings instead of images for efficiency.
    """
    
    def __init__(
        self,
        feature_dim: int = 384,  # DINOv2-S output dimension
        noise_std: float = 0.1,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of feature vectors
            noise_std: Standard deviation of Gaussian noise
            dropout_rate: Feature dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply feature-space augmentation.
        
        Args:
            features: Input features [B, D] or [B, N, D]
            training: Whether in training mode
            
        Returns:
            Augmented features (same shape)
        """
        if not training:
            return features
        
        # Add Gaussian noise
        noise = torch.randn_like(features) * self.noise_std
        augmented = features + noise
        
        # Apply dropout
        augmented = self.dropout(augmented)
        
        # L2 normalize (important for prototype matching)
        augmented = nn.functional.normalize(augmented, p=2, dim=-1)
        
        return augmented
    
    def augment_prototypes(
        self,
        prototypes: torch.Tensor,
        num_augmentations: int = 5
    ) -> torch.Tensor:
        """
        Generate multiple augmented versions of prototypes.
        
        Args:
            prototypes: Class prototypes [C, D]
            num_augmentations: Number of augmented copies per prototype
            
        Returns:
            Augmented prototypes [C*num_augmentations, D]
        """
        C, D = prototypes.shape
        augmented_list = []
        
        for _ in range(num_augmentations):
            aug = self.forward(prototypes, training=True)
            augmented_list.append(aug)
        
        # Stack: [C, num_aug, D] -> [C*num_aug, D]
        augmented = torch.stack(augmented_list, dim=1).reshape(-1, D)
        
        return augmented


class ContrastiveAugmentation:
    """
    Dual augmentation for contrastive learning.
    Generates two augmented views of the same support image.
    """
    
    def __init__(self, img_size: int = 224):
        """
        Args:
            img_size: Target image size
        """
        self.img_size = img_size
        self.weak_aug = SupportAugmentation(img_size, mode="weak")
        self.strong_aug = SupportAugmentation(img_size, mode="strong")
    
    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Generate two augmented views for contrastive learning.
        
        Args:
            image: Input image [H, W, 3]
            
        Returns:
            Dictionary with 'view1' and 'view2' (both should map to same prototype)
        """
        view1 = self.weak_aug(image)
        view2 = self.strong_aug(image)
        
        return {
            'view1': view1,
            'view2': view2
        }
    
    def augment_batch(
        self,
        images: List[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate contrastive pairs for a batch of images.
        
        Args:
            images: List of K images
            
        Returns:
            Dictionary with 'view1' [K, 3, 224, 224] and 'view2' [K, 3, 224, 224]
        """
        view1_list = []
        view2_list = []
        
        for img in images:
            views = self(img)
            view1_list.append(views['view1'])
            view2_list.append(views['view2'])
        
        return {
            'view1': torch.stack(view1_list, dim=0),
            'view2': torch.stack(view2_list, dim=0)
        }


def get_support_augmentation(mode: str = "weak", img_size: int = 224) -> SupportAugmentation:
    """
    Factory function to get support augmentation.
    
    Args:
        mode: "weak" for standard training, "strong" for contrastive learning
        img_size: Target image size (224 for DINOv2)
        
    Returns:
        SupportAugmentation instance
    """
    return SupportAugmentation(img_size=img_size, mode=mode)


def get_contrastive_augmentation(img_size: int = 224) -> ContrastiveAugmentation:
    """
    Factory function for contrastive learning augmentation.
    
    Args:
        img_size: Target image size
        
    Returns:
        ContrastiveAugmentation instance
    """
    return ContrastiveAugmentation(img_size=img_size)
