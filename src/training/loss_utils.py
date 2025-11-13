"""
Training utilities for loss computation.
Handles target matching, feature extraction, and loss input preparation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) tensor of boxes (x1, y1, x2, y2)
        boxes2: (M, 4) tensor of boxes (x1, y1, x2, y2)
        
    Returns:
        iou: (N, M) tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    iou = inter / union.clamp(min=1e-6)
    return iou


def match_predictions_to_targets(
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_bboxes: List[torch.Tensor],
    target_classes: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match predictions to ground truth targets using IoU-based assignment.
    
    This is a simplified version. For production, consider using:
    - Task-Aligned Assigner (YOLOv8 default)
    - Hungarian matching (DETR-style)
    
    Args:
        pred_bboxes: (N_pred, 4) predicted boxes
        pred_scores: (N_pred,) prediction confidence scores
        target_bboxes: List of (N_i, 4) target boxes per image
        target_classes: List of (N_i,) target classes per image
        iou_threshold: Minimum IoU for positive match
        
    Returns:
        matched_pred_indices: Indices of matched predictions
        matched_target_bboxes: Corresponding target bboxes
        matched_target_classes: Corresponding target classes
    """
    matched_pred_indices = []
    matched_target_bboxes = []
    matched_target_classes = []
    
    # Process each image in batch
    pred_offset = 0
    for img_idx, (gt_boxes, gt_classes) in enumerate(zip(target_bboxes, target_classes)):
        if len(gt_boxes) == 0:
            continue
        
        # Get predictions for this image (simplified - assumes all preds are from single image)
        # In practice, need to track which predictions belong to which image
        img_pred_boxes = pred_bboxes[pred_offset:pred_offset + len(pred_bboxes)]
        
        if len(img_pred_boxes) == 0:
            continue
        
        # Compute IoU between predictions and ground truth
        ious = box_iou(img_pred_boxes, gt_boxes)  # (N_pred, N_gt)
        
        # For each ground truth, find best matching prediction
        for gt_idx in range(len(gt_boxes)):
            gt_ious = ious[:, gt_idx]
            if gt_ious.max() < iou_threshold:
                continue
            
            best_pred_idx = gt_ious.argmax()
            
            matched_pred_indices.append(pred_offset + best_pred_idx)
            matched_target_bboxes.append(gt_boxes[gt_idx])
            matched_target_classes.append(gt_classes[gt_idx])
    
    if len(matched_pred_indices) == 0:
        return (torch.zeros(0, dtype=torch.long),
                torch.zeros((0, 4)),
                torch.zeros(0, dtype=torch.long))
    
    return (torch.tensor(matched_pred_indices),
            torch.stack(matched_target_bboxes),
            torch.tensor(matched_target_classes))


def prepare_loss_inputs(
    model_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    stage: int = 2,
    reg_max: int = 16,
) -> Dict:
    """
    Prepare inputs for ReferenceBasedDetectionLoss.
    
    Args:
        model_outputs: Output dict from YOLOv8nRefDet forward pass
        batch: Batch dict from collator
        stage: Training stage (1, 2, or 3)
        reg_max: Maximum regression value for DFL
        
    Returns:
        loss_inputs: Dict ready for loss function
    """
    device = batch['query_images'].device
    
    # Extract predictions
    pred_bboxes = model_outputs.get('pred_bboxes', torch.zeros((0, 4), device=device))
    pred_scores = model_outputs.get('pred_scores', torch.zeros(0, device=device))
    pred_cls_logits = model_outputs.get('pred_cls_logits', torch.zeros((0, batch['num_classes']), device=device))
    pred_dfl_dist = model_outputs.get('pred_dfl_dist', torch.zeros((0, 4 * (reg_max + 1)), device=device))
    
    # Match predictions to targets
    matched_pred_indices, matched_target_bboxes, matched_target_classes = match_predictions_to_targets(
        pred_bboxes=pred_bboxes,
        pred_scores=pred_scores,
        target_bboxes=batch['target_bboxes'],
        target_classes=batch['target_classes'],
    )
    
    # Prepare detection loss inputs
    if len(matched_pred_indices) > 0:
        # Get matched predictions
        matched_pred_bboxes = pred_bboxes[matched_pred_indices]
        matched_pred_cls_logits = pred_cls_logits[matched_pred_indices]
        matched_pred_dfl_dist = pred_dfl_dist[matched_pred_indices]
        
        # Prepare classification targets (one-hot encoding)
        num_classes = batch['num_classes']
        target_cls_onehot = F.one_hot(matched_target_classes, num_classes=num_classes).float()
        
        # Prepare DFL targets
        # Convert bbox coordinates to DFL bin indices
        # Normalize bboxes to [0, 1] first
        img_size = 640  # Query image size
        normalized_bboxes = matched_target_bboxes.clone()
        normalized_bboxes[:, [0, 2]] /= img_size
        normalized_bboxes[:, [1, 3]] /= img_size
        
        # Convert to DFL targets (discretize into bins)
        target_dfl = (normalized_bboxes * reg_max).long()
        target_dfl = torch.clamp(target_dfl, 0, reg_max)
        
    else:
        # No matches - use dummy tensors
        matched_pred_bboxes = torch.zeros((0, 4), device=device)
        matched_pred_cls_logits = torch.zeros((0, batch['num_classes']), device=device)
        matched_pred_dfl_dist = torch.zeros((0, 4 * (reg_max + 1)), device=device)
        target_cls_onehot = torch.zeros((0, batch['num_classes']), device=device)
        target_dfl = torch.zeros((0, 4), dtype=torch.long, device=device)
        matched_target_bboxes = torch.zeros((0, 4), device=device)
    
    # Prepare loss inputs dict
    loss_inputs = {
        # Detection outputs
        'pred_bboxes': matched_pred_bboxes,
        'pred_cls_logits': matched_pred_cls_logits,
        'pred_dfl_dist': matched_pred_dfl_dist,
        # Detection targets
        'target_bboxes': matched_target_bboxes,
        'target_cls': target_cls_onehot,
        'target_dfl': target_dfl,
    }
    
    # Add contrastive learning inputs for Stage 2+
    if stage >= 2:
        # Extract features from model outputs
        fused_features = model_outputs.get('fused_features', {})
        
        # For contrastive loss, we need query features and support prototypes
        # These are typically extracted from intermediate layers
        # Here we use a simplified approach
        
        if 'query_features' in model_outputs:
            loss_inputs['query_features'] = model_outputs['query_features']
        
        if 'support_prototypes' in model_outputs:
            loss_inputs['support_prototypes'] = model_outputs['support_prototypes']
        
        # Feature labels for contrastive learning
        if len(matched_target_classes) > 0:
            loss_inputs['feature_labels'] = matched_target_classes
    
    # Add triplet loss inputs for Stage 3
    if stage >= 3:
        if 'triplet_embeddings' in model_outputs:
            loss_inputs['triplet_embeddings'] = model_outputs['triplet_embeddings']
            loss_inputs['triplet_labels'] = matched_target_classes
    
    return loss_inputs


def extract_roi_features(
    feature_maps: Dict[str, torch.Tensor],
    bboxes: torch.Tensor,
    output_size: int = 7,
) -> torch.Tensor:
    """
    Extract ROI features from feature maps using RoIAlign.
    
    Args:
        feature_maps: Dict of multi-scale feature maps
        bboxes: (N, 4) tensor of boxes (x1, y1, x2, y2) in pixel coords
        output_size: ROI pooling output size
        
    Returns:
        roi_features: (N, C) tensor of ROI features
    """
    from torchvision.ops import roi_align
    
    # Use P4 feature map (middle scale)
    feat_map = feature_maps.get('p4', feature_maps.get('P4'))
    if feat_map is None:
        # Fallback to any available feature map
        feat_map = list(feature_maps.values())[0]
    
    if len(bboxes) == 0:
        return torch.zeros((0, feat_map.shape[1]), device=feat_map.device)
    
    # RoIAlign expects boxes in (x1, y1, x2, y2) format
    # Scale boxes to feature map resolution
    stride = 640 // feat_map.shape[-1]  # Assuming 640 input size
    scaled_bboxes = bboxes / stride
    
    # Add batch index (assuming single image)
    batch_indices = torch.zeros((len(bboxes), 1), device=bboxes.device)
    rois = torch.cat([batch_indices, scaled_bboxes], dim=1)
    
    # ROI pooling
    roi_feats = roi_align(
        feat_map,
        rois,
        output_size=output_size,
        spatial_scale=1.0,
        sampling_ratio=2,
    )
    
    # Global average pooling
    roi_feats = roi_feats.flatten(start_dim=2).mean(dim=2)  # (N, C)
    
    return roi_feats


def compute_prototype_similarity(
    query_features: torch.Tensor,
    support_prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute cosine similarity between query features and support prototypes.
    
    Args:
        query_features: (N, D) query feature vectors
        support_prototypes: (K, D) support prototype vectors
        temperature: Temperature scaling factor
        
    Returns:
        similarity: (N, K) similarity scores
    """
    # Normalize features
    query_norm = F.normalize(query_features, p=2, dim=1)
    proto_norm = F.normalize(support_prototypes, p=2, dim=1)
    
    # Cosine similarity
    similarity = torch.mm(query_norm, proto_norm.t()) / temperature
    
    return similarity


def prepare_detection_loss_inputs(
    model_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    stage: int = 2,
    reg_max: int = 16,
) -> Dict:
    """
    Prepare inputs for detection loss from standard detection batch.
    
    Args:
        model_outputs: Output dict from YOLOv8nRefDet forward pass
        batch: Detection batch dict with query_images, support_images, target_bboxes, etc.
        stage: Training stage (1, 2, or 3)
        reg_max: Maximum regression value for DFL
        
    Returns:
        loss_inputs: Dict ready for detection loss function
    """
    # Use existing prepare_loss_inputs function
    return prepare_loss_inputs(model_outputs, batch, stage, reg_max)


def prepare_triplet_loss_inputs(
    model_outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    project_dim: int = 256,
) -> Dict:
    """
    Prepare inputs for triplet loss from triplet batch.
    
    Args:
        model_outputs: Output dict from YOLOv8nRefDet forward pass with return_features=True
                      Must contain 'support_global_feat' (anchor) and 'query_global_feat' (pos/neg)
        batch: Triplet batch dict with:
               - anchor_images: (B, 3, 256, 256) support images
               - positive_images: (B, 3, 640, 640) query frames with objects
               - negative_images: (B, 3, 640, 640) background or cross-class frames
               - labels: (B,) class labels for anchors
        project_dim: Target dimension for feature projection (default: 256)
        
    Returns:
        loss_inputs: Dict with triplet features ready for TripletLoss
            - anchor_features: (B, D) from support encoder, projected to project_dim
            - positive_features: (B, D) from backbone on positive frames
            - negative_features: (B, D) from backbone on negative frames
    """
    # Extract anchor features (from support encoder - DINOv2)
    anchor_features = model_outputs.get('support_global_feat', None)
    
    # Extract positive/negative features (from backbone - YOLOv8)
    # For triplet batch, query_global_feat contains BOTH positive and negative
    # Split them based on batch structure
    query_global_feat = model_outputs.get('query_global_feat', None)
    
    if anchor_features is None or query_global_feat is None:
        raise ValueError(
            "Triplet loss requires 'support_global_feat' and 'query_global_feat' in model outputs. "
            "Make sure to call model with return_features=True."
        )
    
    # Project anchor features to match query feature dimension if needed
    # Anchor: (B, 384) from DINOv2 -> project to (B, 256)
    # Query: (B, 256) from YOLOv8 backbone
    anchor_dim = anchor_features.shape[-1]
    query_dim = query_global_feat.shape[-1]
    
    if anchor_dim != query_dim:
        # Simple linear projection: project to lower dimension
        if anchor_dim > query_dim:
            # Project down: take first query_dim dimensions (truncate)
            anchor_features = anchor_features[..., :query_dim]
        else:
            # Pad up: zero-pad to match query_dim
            pad_size = query_dim - anchor_dim
            padding = F.pad(anchor_features, (0, pad_size), mode='constant', value=0)
            anchor_features = padding
    
    # Split query features into positive and negative
    # Batch structure: [positive_1, negative_1, positive_2, negative_2, ...]
    # OR: first half positive, second half negative (depends on collator implementation)
    batch_size = anchor_features.shape[0]
    
    # Check if batch contains 'is_positive' flag
    if 'is_positive' in batch:
        # Use explicit flags from batch
        is_positive = batch['is_positive']
        positive_features = query_global_feat[is_positive]
        negative_features = query_global_feat[~is_positive]
    else:
        # Assume interleaved structure: [pos, neg, pos, neg, ...]
        # This is the default from TripletBatchCollator
        positive_features = query_global_feat[0::2]  # Even indices
        negative_features = query_global_feat[1::2]  # Odd indices
    
    loss_inputs = {
        'anchor_features': anchor_features,
        'positive_features': positive_features,
        'negative_features': negative_features,
    }
    
    # Add labels if available (for semi-hard mining)
    if 'labels' in batch:
        loss_inputs['labels'] = batch['labels']
    
    return loss_inputs


def prepare_mixed_loss_inputs(
    detection_outputs: Dict[str, torch.Tensor],
    triplet_outputs: Dict[str, torch.Tensor],
    detection_batch: Dict[str, torch.Tensor],
    triplet_batch: Dict[str, torch.Tensor],
    stage: int = 2,
    reg_max: int = 16,
) -> Tuple[Dict, Dict]:
    """
    Prepare inputs for mixed detection + triplet training.
    
    Args:
        detection_outputs: Model outputs from detection forward pass
        triplet_outputs: Model outputs from triplet forward pass with return_features=True
        detection_batch: Detection batch dict
        triplet_batch: Triplet batch dict
        stage: Training stage
        reg_max: Maximum regression value for DFL
        
    Returns:
        detection_loss_inputs: Dict for detection loss
        triplet_loss_inputs: Dict for triplet loss
    """
    detection_loss_inputs = prepare_detection_loss_inputs(
        detection_outputs, detection_batch, stage, reg_max
    )
    
    triplet_loss_inputs = prepare_triplet_loss_inputs(
        triplet_outputs, triplet_batch
    )
    
    return detection_loss_inputs, triplet_loss_inputs
