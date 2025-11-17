"""
Combined Reference-Based Detection Loss
Integrates all loss components according to the recommended configuration
from loss-functions-guide.md

Loss Stack (CPE Removed - see docs/CPE_REMOVAL_SUMMARY.md):
- WIoU v3 / CIoU: 7.5 (bbox regression, selectable)
- BCE: 0.5 (classification)
- SupCon: 1.0 → 0.5 (prototype matching)
- Triplet: 0.2 (Stage 3, prevents catastrophic forgetting)
"""

import torch
import torch.nn as nn

from .wiou_loss import WIoULoss
from .ciou_loss import CIoULoss
from .bce_loss import BCEClassificationLoss
from .supervised_contrastive_loss import SupervisedContrastiveLoss, PrototypeContrastiveLoss
from .triplet_loss import TripletLoss, BatchHardTripletLoss


class ReferenceBasedDetectionLoss(nn.Module):
    """
    Combined loss for reference-based UAV detection
    
    Supports different training stages with adaptive loss weighting:
    - Stage 1 (Base pre-training): bbox + cls + dfl only
    - Stage 2 (Few-shot meta): add supcon + triplet
    - Stage 3 (Fine-tuning): reduce contrastive weights
    
    Args:
        stage (int): Training stage (1, 2, or 3)
        bbox_loss_type (str): Type of bbox loss - 'wiou' (default, dynamic focusing) or 'ciou' (Ultralytics standard, more stable)
        bbox_weight (float): Weight for bbox regression loss
        cls_weight (float): Weight for classification loss
        supcon_weight (float): Weight for supervised contrastive loss
        triplet_weight (float): Weight for triplet loss (Stage 2+ only)
        use_batch_hard_triplet (bool): Use BatchHardTripletLoss instead of regular TripletLoss
        debug_mode (bool): Enable debug prints in SupCon losses
        smooth_wiou (bool): Use smooth focusing in WIoU to reduce noise (recommended for few-shot)
    """
    
    def __init__(
        self,
        stage=1,
        bbox_loss_type='wiou',
        bbox_weight=7.5,
        cls_weight=0.5,
        supcon_weight=1.0,
        triplet_weight=0.2,
        use_batch_hard_triplet=False,
        debug_mode=False,
        smooth_wiou=False
    ):
        super().__init__()
        
        self.stage = stage
        self.debug_mode = debug_mode
        self.bbox_loss_type = bbox_loss_type
        
        # Core detection losses (always active)
        # Choose between WIoU (dynamic focusing, more aggressive) or CIoU (Ultralytics standard, more stable)
        if bbox_loss_type == 'ciou':
            self.bbox_loss = CIoULoss(eps=1e-7)
        elif bbox_loss_type == 'wiou':
            self.bbox_loss = WIoULoss(monotonous=True, smooth_focusing=smooth_wiou)
        else:
            raise ValueError(f"Invalid bbox_loss_type: {bbox_loss_type}. Must be 'wiou' or 'ciou'")
        
        self.cls_loss = BCEClassificationLoss()
        
        # Contrastive losses (stage 2+)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07, debug_mode=debug_mode)
        self.prototype_loss = PrototypeContrastiveLoss(temperature=0.07, debug_mode=debug_mode)
        
        # Triplet loss (stage 2+) - prevents catastrophic forgetting
        # Use cosine distance for normalized features (more stable than euclidean)
        if use_batch_hard_triplet:
            self.triplet_loss = BatchHardTripletLoss(margin=0.3, distance='cosine')
        else:
            self.triplet_loss = TripletLoss(margin=0.3, distance='cosine')
        
        # Set weights based on stage
        self.weights = self._get_stage_weights(
            stage, bbox_weight, cls_weight, 
            supcon_weight, triplet_weight
        )
    
    def _get_stage_weights(self, stage, bbox_w, cls_w, supcon_w, triplet_w):
        """Get loss weights for each training stage"""
        if stage == 1:
            # Stage 1: Base pre-training (no contrastive, no triplet)
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'supcon': 0.0,
                'triplet': 0.0
            }
        elif stage == 2:
            # Stage 2: Few-shot meta-learning (full contrastive, optional triplet)
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'supcon': supcon_w,
                'triplet': triplet_w
            }
        else:  # stage == 3
            # Stage 3: Fine-tuning (reduced contrastive, add triplet)
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'supcon': supcon_w * 0.5,
                'triplet': triplet_w
            }
    
    def forward(
        self,
        # Detection outputs
        pred_bboxes,
        pred_cls_logits,
        # Targets
        target_bboxes,
        target_cls,
        # Contrastive learning (optional)
        query_features=None,
        support_prototypes=None,
        feature_labels=None,
        # Triplet loss (optional, Stage 3)
        triplet_anchors=None,
        triplet_positives=None,
        triplet_negatives=None,
        triplet_embeddings=None,
        triplet_labels=None
    ):
        """
        Forward pass
        
        Args:
            # Core detection
            pred_bboxes (torch.Tensor): Predicted boxes [N, 4] (x1,y1,x2,y2)
            pred_cls_logits (torch.Tensor): Class logits [N, num_classes]
            target_bboxes (torch.Tensor): Target boxes [N, 4]
            target_cls (torch.Tensor): Target classes [N, num_classes]
            
            # Contrastive learning (stage 2+)
            query_features (torch.Tensor, optional): Query features [M, D]
            support_prototypes (torch.Tensor, optional): Prototypes [K, D]
            feature_labels (torch.Tensor, optional): Labels for features [M]
            
            # Triplet loss (stage 3)
            triplet_anchors (torch.Tensor, optional): Anchor features [N, D]
            triplet_positives (torch.Tensor, optional): Positive features [N, D]
            triplet_negatives (torch.Tensor, optional): Negative features [N, D]
            triplet_embeddings (torch.Tensor, optional): Embeddings for batch hard [N, D]
            triplet_labels (torch.Tensor, optional): Labels for batch hard [N]
            
        Returns:
            dict: Dictionary of loss components and total loss
        """
        losses = {}
        
        # 1. Bounding box regression loss (WIoU or CIoU)
        if pred_bboxes.numel() > 0 and target_bboxes.numel() > 0:
            losses['bbox_loss'] = self.bbox_loss(pred_bboxes, target_bboxes)
        else:
            losses['bbox_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        
        # 2. Classification loss (BCE)
        if pred_cls_logits.numel() > 0 and target_cls.numel() > 0:
            # CRITICAL: Clamp logits BEFORE BCE loss to prevent gradient explosion
            # Even though BCE loss clamps internally, we need to clamp logits to prevent
            # gradient explosion during backward pass through feature_proj layers
            pred_cls_logits_clamped = torch.clamp(pred_cls_logits, min=-10.0, max=10.0)
            losses['cls_loss'] = self.cls_loss(pred_cls_logits_clamped, target_cls)
        else:
            losses['cls_loss'] = torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        
        # 4. Supervised Contrastive Loss (stage 2+)
        if self.weights['supcon'] > 0:
            if query_features is not None and support_prototypes is not None:
                # Use prototype-based contrastive loss
                losses['supcon_loss'] = self.prototype_loss(
                    query_features, 
                    support_prototypes, 
                    feature_labels
                )
            elif query_features is not None and feature_labels is not None:
                # Use standard supervised contrastive loss
                losses['supcon_loss'] = self.supcon_loss(
                    query_features,
                    feature_labels
                )
            else:
                losses['supcon_loss'] = torch.tensor(
                    0.0, 
                    device=pred_bboxes.device
                )
        else:
            losses['supcon_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        
        # 5. Triplet Loss (stage 2+) - prevents catastrophic forgetting
        if self.weights['triplet'] > 0:
            if isinstance(self.triplet_loss, BatchHardTripletLoss):
                # Batch hard triplet: needs embeddings and labels
                if triplet_embeddings is not None and triplet_labels is not None:
                    losses['triplet_loss'] = self.triplet_loss(
                        triplet_embeddings,
                        triplet_labels
                    )
                else:
                    losses['triplet_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
            else:
                # Regular triplet: needs anchor, positive, negative
                if (triplet_anchors is not None and 
                    triplet_positives is not None and 
                    triplet_negatives is not None):
                    losses['triplet_loss'] = self.triplet_loss(
                        triplet_anchors,
                        triplet_positives,
                        triplet_negatives
                    )
                else:
                    losses['triplet_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        else:
            losses['triplet_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        
        # Calculate total loss
        total_loss = (
            self.weights['bbox'] * losses['bbox_loss'] +
            self.weights['cls'] * losses['cls_loss'] +
            self.weights['supcon'] * losses['supcon_loss'] +
            self.weights['triplet'] * losses['triplet_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def set_stage(self, stage, bbox_weight=7.5, cls_weight=0.5,
                  supcon_weight=1.0, triplet_weight=0.2):
        """Update training stage and corresponding weights"""
        self.stage = stage
        self.weights = self._get_stage_weights(
            stage, bbox_weight, cls_weight, 
            supcon_weight, triplet_weight
        )


def create_loss_fn(stage=1, **kwargs):
    """
    Factory function to create loss function for specific training stage
    
    Args:
        stage (int): Training stage (1, 2, or 3)
        **kwargs: Additional arguments passed to ReferenceBasedDetectionLoss
        
    Returns:
        ReferenceBasedDetectionLoss: Configured loss function
    """
    return ReferenceBasedDetectionLoss(stage=stage, **kwargs)
