"""
Query Path Augmentation (Drone Video Frames)
Aggressive augmentation for maximum diversity in detection training.

UPGRADED: Now uses hybrid approach:
- Standalone Mosaic/MixUp/CopyPaste implementations (Ultralytics-compatible)
- AlbumentationsX (color/blur/geometric) - 10-23x faster augmentation
- LetterBox resizing for proper aspect ratio preservation
"""

import torch
import torch.nn as nn
import numpy as np
import albumentations as A  
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, List, Tuple, Optional, Any
import random
from copy import deepcopy
import warnings

# Suppress albumentations division warnings (filtered by min_area and min_visibility in BboxParams)
warnings.filterwarnings('ignore', message='.*invalid value encountered in divide.*', category=RuntimeWarning)


# ============================================================================
# Helper Functions
# ============================================================================

def bbox_ioa(box1, box2, eps=1e-7):
    """
    Calculate intersection over area (IoA) between boxes.
    
    Args:
        box1: [N, 4] in xyxy format
        box2: [M, 4] in xyxy format
        
    Returns:
        ioa: [N, M] intersection over box2 area
    """
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    
    # Expand dimensions for broadcasting
    box1 = box1[:, None, :]  # [N, 1, 4]
    box2 = box2[None, :, :]  # [1, M, 4]
    
    # Intersection area
    inter_mins = np.maximum(box1[..., :2], box2[..., :2])
    inter_maxs = np.minimum(box1[..., 2:], box2[..., 2:])
    inter_wh = np.maximum(inter_maxs - inter_mins, 0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    
    # Box2 area
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    return inter_area / (box2_area + eps)


def filter_valid_bboxes(bboxes: List, labels: List, min_area: float = 1.0, min_visibility: float = 0.0):
    """
    Filter out invalid bounding boxes (zero area, negative coords, etc.).
    
    This prevents Albumentations warnings about division by zero in transforms
    like Erasing, CoarseDropout, etc.
    
    Args:
        bboxes: List of bboxes in [x1, y1, x2, y2] format
        labels: List of corresponding labels
        min_area: Minimum bbox area in pixels (default 1.0)
        min_visibility: Minimum visibility ratio (0.0-1.0)
        
    Returns:
        filtered_bboxes, filtered_labels
    """
    if len(bboxes) == 0:
        return bboxes, labels
    
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Check if bbox has valid coordinates
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Check if bbox has minimum area
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        
        valid_indices.append(i)
    
    # Return filtered boxes and labels
    filtered_bboxes = [bboxes[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    return filtered_bboxes, filtered_labels


class LetterBox:
    """
    Resize image and padding for detection while preserving aspect ratio.
    Based on Ultralytics implementation for YOLO compatibility.
    """
    
    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """
        Initialize LetterBox for resizing and padding images.
        
        Args:
            new_shape: Target size (height, width)
            auto: Use minimum rectangle
            scale_fill: Stretch image to new_shape without padding
            scaleup: Allow scaling up (if False, only scale down)
            center: Center the image (if False, top-left alignment)
            stride: Stride for rounding padding
            padding_value: Value for padding
            interpolation: cv2 interpolation method
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Resize and pad image, update bboxes.
        
        Args:
            image: Input image [H, W, 3]
            bboxes: Bounding boxes [N, 4] in xyxy format
            labels: Class labels [N]
            
        Returns:
            Dictionary with 'image', 'bboxes', 'labels', 'ratio', 'pad'
        """
        shape = image.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up
            r = min(r, 1.0)
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)
            if image.ndim == 2:
                image = image[..., None]
        
        # Add padding
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
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
        
        # Update bboxes
        if bboxes is not None and len(bboxes) > 0:
            bboxes = bboxes.copy().astype(np.float32)
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ratio[0] + left
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ratio[1] + top
        
        return {
            'image': image,
            'bboxes': bboxes if bboxes is not None else np.zeros((0, 4)),
            'labels': labels if labels is not None else np.zeros((0,)),
            'ratio': ratio,
            'pad': (left, top)
        }


# ============================================================================
# Mosaic/MixUp/CopyPaste Augmentations (Ultralytics-compatible)
# ============================================================================

class MosaicAugmentation:
    """
    Mosaic augmentation following Ultralytics' approach.
    Most impactful augmentation: +5-7% mAP improvement.
    
    Supports 2x2 (4 images) and 3x3 (9 images) mosaics.
    Implementation follows Ultralytics YOLOv8 Mosaic algorithm for compatibility.
    """
    
    def __init__(self, img_size: int = 640, prob: float = 1.0, n: int = 4):
        """
        Args:
            img_size: Target output size (640x640 for YOLOv8)
            prob: Probability of applying mosaic
            n: Number of images in mosaic (4 for 2x2, 9 for 3x3)
        """
        assert n in {4, 9}, "n must be 4 (2x2) or 9 (3x3)"
        self.img_size = img_size
        self.prob = prob
        self.n = n
        self.border = (-img_size // 2, -img_size // 2)  # width, height border
    
    def __call__(
        self, 
        images: List[np.ndarray], 
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation using Ultralytics implementation.
        
        Args:
            images: List of n images [H, W, 3]
            bboxes_list: List of n bbox arrays [N, 4] in format (x1, y1, x2, y2)
            labels_list: List of n label arrays [N]
            
        Returns:
            mosaic_img: Stitched image [img_size, img_size, 3]
            mosaic_bboxes: Combined bboxes [M, 4]
            mosaic_labels: Combined labels [M]
        """
        if random.random() > self.prob:
            # Return first image without mosaic
            return images[0], bboxes_list[0], labels_list[0]
        
        assert len(images) == self.n, f"Mosaic requires exactly {self.n} images"
        
        if self.n == 4:
            return self._mosaic4(images, bboxes_list, labels_list)
        else:
            return self._mosaic9(images, bboxes_list, labels_list)
    
    def _mosaic4(
        self,
        images: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 2x2 mosaic (Ultralytics-compatible).
        """
        s = self.img_size
        yc = int(random.uniform(0.5 * s, 1.5 * s))  # mosaic center y
        xc = int(random.uniform(0.5 * s, 1.5 * s))  # mosaic center x
        
        # Create canvas
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Calculate placement coordinates (Ultralytics algorithm)
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place image segment
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust bboxes
            if len(bboxes) > 0:
                padw = x1a - x1b
                padh = y1a - y1b
                
                adjusted_bboxes = bboxes.copy().astype(np.float32)
                adjusted_bboxes[:, [0, 2]] += padw
                adjusted_bboxes[:, [1, 3]] += padh
                
                # Clip to canvas
                adjusted_bboxes[:, [0, 2]] = np.clip(adjusted_bboxes[:, [0, 2]], 0, s * 2)
                adjusted_bboxes[:, [1, 3]] = np.clip(adjusted_bboxes[:, [1, 3]], 0, s * 2)
                
                # Filter invalid boxes
                widths = adjusted_bboxes[:, 2] - adjusted_bboxes[:, 0]
                heights = adjusted_bboxes[:, 3] - adjusted_bboxes[:, 1]
                valid_mask = (widths > 2) & (heights > 2)
                
                if np.any(valid_mask):
                    mosaic_bboxes.append(adjusted_bboxes[valid_mask])
                    mosaic_labels.append(labels[valid_mask])
        
        # Crop to final size (Ultralytics-style)
        mosaic_img = mosaic_img[:s, :s]
        
        # Combine annotations
        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
            
            # Final clipping
            mosaic_bboxes[:, [0, 2]] = np.clip(mosaic_bboxes[:, [0, 2]], 0, s)
            mosaic_bboxes[:, [1, 3]] = np.clip(mosaic_bboxes[:, [1, 3]], 0, s)
            
            # Remove zero-area boxes
            widths = mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]
            heights = mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1]
            valid_mask = (widths > 1) & (heights > 1)
            mosaic_bboxes = mosaic_bboxes[valid_mask]
            mosaic_labels = mosaic_labels[valid_mask]
        else:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels
    
    def _mosaic9(
        self,
        images: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 3x3 mosaic (Ultralytics-compatible).
        """
        s = self.img_size
        
        # Create canvas (3x image size)
        mosaic_img = np.full((s * 3, s * 3, 3), 114, dtype=np.uint8)
        mosaic_bboxes = []
        mosaic_labels = []
        
        hp, wp = -1, -1  # height, width previous
        
        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Calculate placement (Ultralytics 3x3 algorithm)
            if i == 0:  # center
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top-right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom-right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom-left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top-left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp
            
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            
            # Place image
            mosaic_img[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
            hp, wp = h, w
            
            # Adjust bboxes
            if len(bboxes) > 0:
                adjusted_bboxes = bboxes.copy().astype(np.float32)
                adjusted_bboxes[:, [0, 2]] += padw + self.border[0]
                adjusted_bboxes[:, [1, 3]] += padh + self.border[1]
                
                mosaic_bboxes.append(adjusted_bboxes)
                mosaic_labels.append(labels)
        
        # Crop to final size
        mosaic_img = mosaic_img[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        
        # Combine and clip annotations
        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
            
            mosaic_bboxes[:, [0, 2]] = np.clip(mosaic_bboxes[:, [0, 2]], 0, s)
            mosaic_bboxes[:, [1, 3]] = np.clip(mosaic_bboxes[:, [1, 3]], 0, s)
            
            # Remove invalid boxes
            widths = mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]
            heights = mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1]
            valid_mask = (widths > 1) & (heights > 1)
            mosaic_bboxes = mosaic_bboxes[valid_mask]
            mosaic_labels = mosaic_labels[valid_mask]
        else:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels


class MixUpAugmentation:
    """
    MixUp augmentation following Ultralytics' approach.
    Formula: mixed = r × img1 + (1-r) × img2, where r ~ Beta(32, 32)
    
    Implementation follows Ultralytics YOLOv8 MixUp algorithm for compatibility.
    """
    
    def __init__(self, alpha: float = 32.0, prob: float = 0.15):
        """
        Args:
            alpha: Beta distribution parameter (Ultralytics uses 32.0)
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        img1: np.ndarray,
        bboxes1: np.ndarray,
        labels1: np.ndarray,
        img2: np.ndarray,
        bboxes2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation following Ultralytics' approach.
        
        Returns:
            mixed_img, mixed_bboxes, mixed_labels
        """
        if random.random() > self.prob:
            return img1, bboxes1, labels1
        
        # Sample mixing ratio (Ultralytics uses beta(32.0, 32.0))
        r = np.random.beta(self.alpha, self.alpha)
        
        # Mix images (Ultralytics-style weighted blending)
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # Concatenate bboxes and labels (both images contribute)
        mixed_bboxes = np.concatenate([bboxes1, bboxes2], axis=0) if len(bboxes1) > 0 or len(bboxes2) > 0 else np.zeros((0, 4), dtype=np.float32)
        mixed_labels = np.concatenate([labels1, labels2], axis=0) if len(labels1) > 0 or len(labels2) > 0 else np.zeros((0,), dtype=np.int64)
        
        return mixed_img, mixed_bboxes, mixed_labels


class CopyPasteAugmentation:
    """
    CopyPaste augmentation following Ultralytics' approach.
    Copies objects from one image and pastes them onto another.
    
    Implementation follows Ultralytics YOLOv8 CopyPaste algorithm.
    Requires segmentation masks for proper operation.
    """
    
    def __init__(self, prob: float = 0.5, mode: str = "flip"):
        """
        Args:
            prob: Probability of applying copy-paste per object
            mode: "flip" (horizontal flip) or "mixup" (from another image)
        """
        assert mode in {"flip", "mixup"}, f"mode must be 'flip' or 'mixup', got {mode}"
        self.prob = prob
        self.mode = mode
    
    def __call__(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        segments: Optional[List[np.ndarray]] = None,
        img2: Optional[np.ndarray] = None,
        bboxes2: Optional[np.ndarray] = None,
        labels2: Optional[np.ndarray] = None,
        segments2: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply copy-paste augmentation following Ultralytics' approach.
        
        Args:
            img: Input image [H, W, 3]
            bboxes: Bounding boxes [N, 4] in xyxy format
            labels: Class labels [N]
            segments: Segmentation masks (list of [M, 2] polygons)
            img2: Second image for mixup mode
            bboxes2: Bboxes for second image
            labels2: Labels for second image
            segments2: Segments for second image
            
        Returns:
            augmented_img, augmented_bboxes, augmented_labels
        """
        # Skip if no segments or probability check fails
        if segments is None or len(segments) == 0 or self.prob == 0:
            return img, bboxes, labels
        
        h, w = img.shape[:2]
        img = img.copy()
        
        # Create mask canvas
        im_new = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Prepare source instances
        if self.mode == "flip":
            # Flip mode: create flipped version of same image
            img2 = cv2.flip(img, 1)  # horizontal flip
            bboxes2 = bboxes.copy()
            bboxes2[:, [0, 2]] = w - bboxes2[:, [2, 0]]  # flip bboxes
            labels2 = labels.copy()
            segments2 = []
            for seg in segments:
                seg_flip = seg.copy()
                seg_flip[:, 0] = w - seg_flip[:, 0]  # flip x coordinates
                segments2.append(seg_flip)
        
        if img2 is None or bboxes2 is None or segments2 is None:
            return img, bboxes, labels
        
        # Calculate IoA (intersection over area) to avoid overlap
        ioa = bbox_ioa(bboxes2, bboxes)  # [N2, N1]
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # objects with low overlap
        
        if len(indexes) == 0:
            return img, bboxes, labels
        
        n = len(indexes)
        # Sort by max IoA (paste least overlapping first)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        
        # Select subset to paste (based on prob)
        num_to_paste = int(round(self.prob * n))
        indexes = indexes[:num_to_paste]
        
        new_bboxes = []
        new_labels = []
        
        for j in indexes:
            # Draw segment mask
            segment = segments2[j].astype(np.int32)
            cv2.fillPoly(im_new, [segment], 1)
            
            new_bboxes.append(bboxes2[j])
            new_labels.append(labels2[j])
        
        # Apply mask: paste from img2 where mask is 1
        mask_bool = im_new.astype(bool)
        if mask_bool.any():
            if img2.ndim == 2:
                img2 = img2[..., None]
            img[mask_bool] = img2[mask_bool]
        
        # Combine annotations
        if len(new_bboxes) > 0:
            bboxes = np.concatenate([bboxes, np.array(new_bboxes)], axis=0)
            labels = np.concatenate([labels, np.array(new_labels)], axis=0)
        
        return img, bboxes, labels


class QueryAugmentation:
    """
    Complete augmentation pipeline for query images (drone frames).
    Aggressive augmentation strategy for maximum diversity.
    """
    
    def __init__(
        self,
        img_size: int = 640,
        stage: str = "stage1",
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.15
    ):
        """
        Args:
            img_size: Target image size (640 for YOLOv8n)
            stage: Training stage ("stage1", "stage2", "stage3")
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
        """
        self.img_size = img_size
        self.stage = stage
        
        # Initialize specialized augmentations
        self.mosaic = MosaicAugmentation(img_size, mosaic_prob)
        self.mixup = MixUpAugmentation(prob=mixup_prob)
        
        # Stage-specific augmentation strengths
        if stage == "stage1":
            self.transform = self._get_stage1_transform()
        elif stage == "stage2":
            self.transform = self._get_stage2_transform()
        else:  # stage3
            self.transform = self._get_stage3_transform()
    
    def _get_stage1_transform(self):
        """
        Strong augmentation for stage 1 (base training).
        UPGRADED: Uses AlbumentationsX new features (10-23x faster)
        Note: LetterBox resizing is applied separately in __call__
        """
        return A.Compose([
            # NEW: D4 symmetry - replaces HorizontalFlip + VerticalFlip + RandomRotate90
            # Implements all 8 dihedral group transformations (identity, rotations, flips)
            A.D4(p=0.5),
            
            # Affine transforms
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5
            ),
            
            # NEW: PlanckianJitter - physically-accurate color temperature changes
            # Simulates different lighting conditions (warm/cool, indoor/outdoor)
            A.PlanckianJitter(
                mode='blackbody',
                sampling_method='uniform',
                temperature_limit=(3000, 15000),  # 3000K (warm) to 15000K (cool daylight)
                p=0.3
            ),
            
            # HSV (complements PlanckianJitter for saturation/value)
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.4
            ),
            
            # Brightness/Contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
            
            # NEW: AdvancedBlur - replaces GaussianBlur with motion+defocus+zoom+glass
            # More realistic blur types for drone footage
            A.AdvancedBlur(
                blur_limit=(3, 7),
                sigma_x_limit=(0.2, 1.0),
                sigma_y_limit=(0.2, 1.0),
                rotate_limit=90,
                beta_limit=(0.5, 8.0),
                noise_limit=(0.9, 1.1),
                p=0.2
            ),
            
            # Noise
            A.GaussNoise(std_range=(0.1, 0.3), p=0.2),
            
            # NEW: Erasing - better than CoarseDropout for occlusion simulation
            # More flexible and faster
            A.Erasing(
                scale=(0.01, 0.1),  # min/max area as fraction of image
                ratio=(0.3, 3.3),  # aspect ratio range
                p=0.3
            ),
            
            # Normalize to ImageNet stats (required for YOLOv8 backbone)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=2.0,  # Filter out bboxes with area < 2 pixels (prevents div by zero)
            min_visibility=0.1  # Require at least 10% visibility to avoid zero-area boxes
        ))
    
    def _get_stage2_transform(self):
        """
        Medium augmentation for stage 2 (few-shot training).
        REDUCED strength to avoid confusing prototypes.
        Note: LetterBox resizing is applied separately in __call__
        """
        return A.Compose([
            # Reduced D4 probability
            A.D4(p=0.3),
            
            # Conservative affine
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.4
            ),
            
            # Reduced PlanckianJitter range
            A.PlanckianJitter(temperature_limit=(3000, 9000), p=0.2),
            
            # Reduced HSV
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
            
            # Reduced brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            
            # Light blur
            A.AdvancedBlur(blur_limit=(3, 5), p=0.1),
            
            # Normalize to ImageNet stats (required for YOLOv8 backbone)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=2.0,  # Filter out bboxes with area < 2 pixels (prevents div by zero)
            min_visibility=0.1  # Require at least 10% visibility to avoid zero-area boxes
        ))
    
    def _get_stage3_transform(self):
        """
        Weak augmentation for stage 3 (fine-tuning).
        MINIMAL augmentation for final convergence.
        Note: LetterBox resizing is applied separately in __call__
        """
        return A.Compose([
            # Only horizontal flip
            A.HorizontalFlip(p=0.3),
            
            # Very conservative affine
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.02, 0.02),
                rotate=(-5, 5),
                p=0.3
            ),
            
            # Minimal color adjustment
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.2
            ),
            
            # Normalize to ImageNet stats (required for YOLOv8 backbone)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=2.0,  # Filter out bboxes with area < 2 pixels (prevents div by zero)
            min_visibility=0.1  # Require at least 10% visibility to avoid zero-area boxes
        ))
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        apply_mosaic: bool = True,
        mosaic_images: Optional[List[np.ndarray]] = None,
        mosaic_bboxes: Optional[List[np.ndarray]] = None,
        mosaic_labels: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Apply augmentation pipeline with LetterBox resizing.
        
        Args:
            image: Input image [H, W, 3]
            bboxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
            labels: Class labels [N]
            apply_mosaic: Whether to attempt mosaic augmentation
            mosaic_images: Additional 3 images for mosaic (total 4)
            mosaic_bboxes: Bboxes for mosaic images
            mosaic_labels: Labels for mosaic images
            
        Returns:
            Dictionary with 'image' (tensor), 'bboxes' (tensor), 'labels' (tensor)
        """
        # Apply mosaic if requested and data provided
        if apply_mosaic and mosaic_images is not None:
            all_images = [image] + mosaic_images
            all_bboxes = [bboxes] + mosaic_bboxes
            all_labels = [labels] + mosaic_labels
            
            image, bboxes, labels = self.mosaic(all_images, all_bboxes, all_labels)
        
        # Apply LetterBox resizing first (preserves aspect ratio)
        letterbox = LetterBox(
            new_shape=(self.img_size, self.img_size),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            stride=32,
            padding_value=114
        )
        letterbox_result = letterbox(image, bboxes, labels)
        image = letterbox_result['image']
        bboxes = letterbox_result['bboxes']
        labels = letterbox_result['labels']
        
        # Apply standard augmentations
        if len(bboxes) > 0:
            # Filter out invalid bboxes before passing to Albumentations
            # This prevents division by zero warnings in Erasing/CoarseDropout
            bbox_list = bboxes.tolist()
            label_list = labels.tolist()
            bbox_list, label_list = filter_valid_bboxes(bbox_list, label_list, min_area=1.0)
            
            transformed = self.transform(
                image=image,
                bboxes=bbox_list,
                labels=label_list
            )
            
            aug_image = transformed['image']
            aug_bboxes = np.array(transformed['bboxes'], dtype=np.float32) if len(transformed['bboxes']) > 0 else np.zeros((0, 4), dtype=np.float32)
            aug_labels = np.array(transformed['labels'], dtype=np.int64) if len(transformed['labels']) > 0 else np.zeros((0,), dtype=np.int64)
        else:
            # No bboxes, pass empty arrays to transform (albumentations requires label_fields)
            transformed = self.transform(
                image=image,
                bboxes=[],
                labels=[]
            )
            aug_image = transformed['image']
            aug_bboxes = np.zeros((0, 4), dtype=np.float32)
            aug_labels = np.zeros((0,), dtype=np.int64)
        
        return {
            'image': aug_image,
            'bboxes': torch.from_numpy(aug_bboxes) if isinstance(aug_bboxes, np.ndarray) else aug_bboxes,
            'labels': torch.from_numpy(aug_labels) if isinstance(aug_labels, np.ndarray) else aug_labels
        }


def get_query_augmentation(stage: str = "stage1", img_size: int = 640) -> QueryAugmentation:
    """
    Factory function to get stage-specific query augmentation.
    
    Args:
        stage: One of "stage1", "stage2", "stage3"
        img_size: Target image size
        
    Returns:
        QueryAugmentation instance
    """
    mosaic_probs = {
        "stage1": 1.0,
        "stage2": 0.5,
        "stage3": 0.3
    }
    
    mixup_probs = {
        "stage1": 0.15,
        "stage2": 0.0,  # Disable in few-shot stage
        "stage3": 0.0
    }
    
    return QueryAugmentation(
        img_size=img_size,
        stage=stage,
        mosaic_prob=mosaic_probs.get(stage, 0.5),
        mixup_prob=mixup_probs.get(stage, 0.0)
    )
