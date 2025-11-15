"""
Combined Reference-Based Detection Loss
Integrates all loss components according to the recommended configuration
from loss-functions-guide.md

Loss Stack:
- WIoU v3: 7.5 (bbox regression)
- BCE: 0.5 (classification)
- DFL: 1.5 (distribution focal loss)
- SupCon: 1.0 → 0.5 (prototype matching)
- CPE: 0.5 → 0.3 (contrastive proposal encoding)
"""

import torch
import torch.nn as nn

from .wiou_loss import WIoULoss
from .bce_loss import BCEClassificationLoss
from .dfl_loss import DFLoss
from .supervised_contrastive_loss import SupervisedContrastiveLoss, PrototypeContrastiveLoss
from .cpe_loss import SimplifiedCPELoss
from .triplet_loss import TripletLoss, BatchHardTripletLoss


class ReferenceBasedDetectionLoss(nn.Module):
    """
    Combined loss for reference-based UAV detection
    
    Supports different training stages with adaptive loss weighting:
    - Stage 1 (Base pre-training): bbox + cls + dfl only
    - Stage 2 (Few-shot meta): add supcon + cpe
    - Stage 3 (Fine-tuning): reduce contrastive weights
    
    Args:
        stage (int): Training stage (1, 2, or 3)
        bbox_weight (float): Weight for bbox regression loss
        cls_weight (float): Weight for classification loss
        dfl_weight (float): Weight for DFL loss
        supcon_weight (float): Weight for supervised contrastive loss
        cpe_weight (float): Weight for CPE loss
        triplet_weight (float): Weight for triplet loss (Stage 3 only)
        reg_max (int): Maximum regression value for DFL
        use_batch_hard_triplet (bool): Use BatchHardTripletLoss instead of regular TripletLoss
    """
    
    def __init__(
        self,
        stage=1,
        bbox_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
        supcon_weight=1.0,
        cpe_weight=0.5,
        triplet_weight=0.2,
        reg_max=16,
        use_batch_hard_triplet=False
    ):
        super().__init__()
        
        self.stage = stage
        
        # Core detection losses (always active)
        self.bbox_loss = WIoULoss(monotonous=True)
        self.cls_loss = BCEClassificationLoss()
        self.dfl_loss = DFLoss(reg_max=reg_max)
        
        # Contrastive losses (stage 2+)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07)
        self.prototype_loss = PrototypeContrastiveLoss(temperature=0.07)
        self.cpe_loss = SimplifiedCPELoss(temperature=0.1)
        
        # Triplet loss (stage 2+) - prevents catastrophic forgetting
        # Use cosine distance for normalized features (more stable than euclidean)
        if use_batch_hard_triplet:
            self.triplet_loss = BatchHardTripletLoss(margin=0.3, distance='cosine')
        else:
            self.triplet_loss = TripletLoss(margin=0.3, distance='cosine')
        
        # Set weights based on stage
        self.weights = self._get_stage_weights(
            stage, bbox_weight, cls_weight, dfl_weight, 
            supcon_weight, cpe_weight, triplet_weight
        )
    
    def _get_stage_weights(self, stage, bbox_w, cls_w, dfl_w, supcon_w, cpe_w, triplet_w):
        """Get loss weights for each training stage"""
        if stage == 1:
            # Stage 1: Base pre-training (no contrastive, no triplet)
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'dfl': dfl_w,
                'supcon': 0.0,
                'cpe': 0.0,
                'triplet': 0.0
            }
        elif stage == 2:
            # Stage 2: Few-shot meta-learning (full contrastive, optional triplet)
            # Triplet loss can be enabled to prevent catastrophic forgetting
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'dfl': dfl_w,
                'supcon': supcon_w,
                'cpe': cpe_w,
                'triplet': triplet_w  # Use provided triplet weight (0.0 if not using triplet)
            }
        else:  # stage == 3
            # Stage 3: Fine-tuning (reduced contrastive, add triplet)
            return {
                'bbox': bbox_w,
                'cls': cls_w,
                'dfl': dfl_w,
                'supcon': supcon_w * 0.5,
                'cpe': cpe_w * 0.6,
                'triplet': triplet_w  # Prevent catastrophic forgetting
            }
    
    def forward(
        self,
        # Detection outputs
        pred_bboxes,
        pred_cls_logits,
        pred_dfl_dist,
        # Targets
        target_bboxes,
        target_cls,
        target_dfl,
        # Contrastive learning (optional)
        query_features=None,
        support_prototypes=None,
        feature_labels=None,
        proposal_features=None,
        proposal_labels=None,
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
            pred_dfl_dist (torch.Tensor): DFL distribution [N, 4*(reg_max+1)]
            target_bboxes (torch.Tensor): Target boxes [N, 4]
            target_cls (torch.Tensor): Target classes [N, num_classes]
            target_dfl (torch.Tensor): Target DFL coords [N, 4]
            
            # Contrastive learning (stage 2+)
            query_features (torch.Tensor, optional): Query features [M, D]
            support_prototypes (torch.Tensor, optional): Prototypes [K, D]
            feature_labels (torch.Tensor, optional): Labels for features [M]
            proposal_features (torch.Tensor, optional): Proposal features [P, D]
            proposal_labels (torch.Tensor, optional): Proposal labels [P]
            
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
        
        # 1. Bounding box regression loss (WIoU)
        if pred_bboxes.numel() > 0 and target_bboxes.numel() > 0:
            losses['bbox_loss'] = self.bbox_loss(pred_bboxes, target_bboxes)
        else:
            losses['bbox_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        
        # 2. Classification loss (BCE)
        if pred_cls_logits.numel() > 0 and target_cls.numel() > 0:
            losses['cls_loss'] = self.cls_loss(pred_cls_logits, target_cls)
        else:
            losses['cls_loss'] = torch.tensor(0.0, device=pred_cls_logits.device, requires_grad=True)
        
        # 3. Distribution Focal Loss
        if pred_dfl_dist.numel() > 0 and target_dfl.numel() > 0:
            losses['dfl_loss'] = self.dfl_loss(pred_dfl_dist, target_dfl)
        else:
            losses['dfl_loss'] = torch.tensor(0.0, device=pred_dfl_dist.device, requires_grad=True)
        
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
        
        # 5. Contrastive Proposal Encoding Loss (stage 2+)
        if self.weights['cpe'] > 0:
            if proposal_features is not None and proposal_labels is not None:
                losses['cpe_loss'] = self.cpe_loss(
                    proposal_features,
                    proposal_labels
                )
            else:
                losses['cpe_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        else:
            losses['cpe_loss'] = torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)
        
        # 6. Triplet Loss (stage 3 only) - prevents catastrophic forgetting
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
            self.weights['dfl'] * losses['dfl_loss'] +
            self.weights['supcon'] * losses['supcon_loss'] +
            self.weights['cpe'] * losses['cpe_loss'] +
            self.weights['triplet'] * losses['triplet_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def set_stage(self, stage, bbox_weight=7.5, cls_weight=0.5, dfl_weight=1.5, 
                  supcon_weight=1.0, cpe_weight=0.5, triplet_weight=0.2):
        """Update training stage and corresponding weights"""
        self.stage = stage
        self.weights = self._get_stage_weights(
            stage, bbox_weight, cls_weight, dfl_weight, 
            supcon_weight, cpe_weight, triplet_weight
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
