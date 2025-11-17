"""
Loss functions for Reference-Based UAV Object Detection
Implements the recommended loss stack from loss-functions-guide.md
"""

from .wiou_loss import WIoULoss
from .ciou_loss import CIoULoss
from .bce_loss import BCEClassificationLoss
from .dfl_loss import DFLoss
from .supervised_contrastive_loss import SupervisedContrastiveLoss
from .cpe_loss import CPELoss
from .triplet_loss import TripletLoss, BatchHardTripletLoss, AdaptiveTripletLoss
from .combined_loss import ReferenceBasedDetectionLoss

__all__ = [
    'WIoULoss',
    'CIoULoss',
    'BCEClassificationLoss',
    'DFLoss',
    'SupervisedContrastiveLoss',
    'CPELoss',
    'TripletLoss',
    'BatchHardTripletLoss',
    'AdaptiveTripletLoss',
    'ReferenceBasedDetectionLoss',
]
