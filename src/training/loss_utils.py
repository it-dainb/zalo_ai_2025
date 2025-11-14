"""
Training utilities for loss computation.
Handles target matching, feature extraction, and loss input preparation.

IMPORTANT: Now uses ANCHOR-BASED TARGET ASSIGNMENT (YOLOv8-style) to work
directly with raw detection head outputs without confidence filtering.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from src.losses.dfl_loss import DFLoss


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """
    Transform distance(ltrb) to box(xywh or xyxy).
    
    Args:
        distance: (N, 4) distances [left, top, right, bottom] from anchor points
        anchor_points: (N, 2) anchor center points [x, y]
        xywh: If True, return xywh format; else return xyxy format
        dim: Dimension to split on
    
    Returns:
        boxes: (N, 4) boxes in xywh or xyxy format
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def assign_targets_to_anchors(
    proto_boxes_list: List[torch.Tensor],  # [(B, 4*reg_max, H, W), ...] for each scale
    proto_sim_list: List[torch.Tensor],     # [(B, K, H, W), ...] for each scale
    target_bboxes: List[torch.Tensor],      # List of (N_i, 4) boxes per image
    target_classes: List[torch.Tensor],     # List of (N_i,) classes per image
    img_size: int = 640,
    reg_max: int = 16,
    strides: List[int] = [4, 8, 16, 32],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign ground truth targets to anchor points using center-based assignment.
    
    This is a simplified version of YOLOv8's TAL (Task-Aligned Assigner).
    For each GT box, we assign it to anchors whose center falls within the GT box.
    
    Args:
        proto_boxes_list: Raw box predictions from detection head, list of (B, 4*(reg_max+1), H, W)
        proto_sim_list: Raw similarity scores from detection head, list of (B, K, H, W)
        target_bboxes: List of ground truth boxes per image, each (N_i, 4) in xyxy format
        target_classes: List of ground truth classes per image, each (N_i,)
        img_size: Image size (default 640)
        reg_max: DFL regression max value
        strides: Stride for each scale level
        
    Returns:
        assigned_boxes: (M, 4*(reg_max+1)) box predictions for matched anchors
        assigned_cls_logits: (M, K) class logits for matched anchors  
        assigned_dfl_dist: (M, 4*(reg_max+1)) DFL distributions for matched anchors
        assigned_anchor_points: (M, 2) anchor points (x, y) in pixel coordinates
        assigned_strides: (M,) stride values for each matched anchor
        target_boxes: (M, 4) target boxes for matched anchors
        target_cls_onehot: (M, K) one-hot class targets
        target_dfl: (M, 4) DFL regression targets
    """
    device = proto_boxes_list[0].device
    batch_size = proto_boxes_list[0].shape[0]
    num_classes = proto_sim_list[0].shape[1]
    
    # Collect all assigned anchors across batch and scales
    all_assigned_boxes = []
    all_assigned_cls = []
    all_assigned_dfl = []
    all_assigned_anchor_points = []
    all_assigned_strides = []
    all_target_boxes = []
    all_target_cls = []
    all_target_dfl = []
    
    # Process each image in batch
    for batch_idx in range(batch_size):
        gt_boxes = target_bboxes[batch_idx]  # (N_gt, 4)
        gt_classes = target_classes[batch_idx]  # (N_gt,)
        
        if len(gt_boxes) == 0:
            # No targets for this image, skip
            continue
        
        # Convert GT boxes to center format for assignment
        gt_centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2  # (N_gt,)
        gt_centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2  # (N_gt,)
        
        # Process each scale
        for scale_idx, (boxes, sim, stride) in enumerate(zip(proto_boxes_list, proto_sim_list, strides)):
            _, C_box, H, W = boxes.shape
            _, K, _, _ = sim.shape
            
            # Extract predictions for this image
            img_boxes = boxes[batch_idx]  # (C_box, H, W) where C_box = 4*(reg_max+1)
            img_sim = sim[batch_idx]      # (K, H, W)
            
            # Create anchor grid for this scale
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            # Anchor centers in pixel coordinates
            anchor_x = (grid_x + 0.5) * stride  # (H, W)
            anchor_y = (grid_y + 0.5) * stride  # (H, W)
            
            # For each GT box, find anchors whose centers fall inside it
            for gt_idx in range(len(gt_boxes)):
                gt_box = gt_boxes[gt_idx]  # (4,) [x1, y1, x2, y2]
                gt_cls = gt_classes[gt_idx]  # scalar
                
                # Find anchors inside this GT box
                mask = (
                    (anchor_x >= gt_box[0]) & (anchor_x <= gt_box[2]) &
                    (anchor_y >= gt_box[1]) & (anchor_y <= gt_box[3])
                )  # (H, W)
                
                if not mask.any():
                    # No anchors assigned to this GT, assign to closest anchor
                    gt_cx, gt_cy = gt_centers_x[gt_idx], gt_centers_y[gt_idx]
                    dist = (anchor_x - gt_cx) ** 2 + (anchor_y - gt_cy) ** 2
                    closest_idx = dist.argmin()
                    mask = torch.zeros_like(mask)
                    mask.view(-1)[closest_idx] = True
                
                # Extract assigned anchor predictions
                # boxes: (C_box, H, W) -> select mask positions
                assigned_box_preds = img_boxes[:, mask].t()  # (N_assigned, C_box)
                assigned_cls_preds = img_sim[:, mask].t()     # (N_assigned, K)
                
                # Extract anchor points for assigned anchors (in pixel coordinates)
                assigned_anchor_x = anchor_x[mask]  # (N_assigned,)
                assigned_anchor_y = anchor_y[mask]  # (N_assigned,)
                assigned_anchor_pts = torch.stack([assigned_anchor_x, assigned_anchor_y], dim=1)  # (N_assigned, 2)
                
                # Repeat target for all assigned anchors
                n_assigned = assigned_box_preds.shape[0]
                repeated_gt_box = gt_box.unsqueeze(0).repeat(n_assigned, 1)  # (N_assigned, 4)
                repeated_gt_cls = gt_cls.unsqueeze(0).repeat(n_assigned)      # (N_assigned,)
                repeated_stride = torch.full((n_assigned,), stride, dtype=torch.float32, device=device)  # (N_assigned,)
                
                # Collect this assignment
                all_assigned_boxes.append(assigned_box_preds)
                all_assigned_cls.append(assigned_cls_preds)
                all_assigned_dfl.append(assigned_box_preds)  # DFL dist same as box preds
                all_assigned_anchor_points.append(assigned_anchor_pts)
                all_assigned_strides.append(repeated_stride)
                all_target_boxes.append(repeated_gt_box)
                all_target_cls.append(repeated_gt_cls)
                
                # Compute DFL targets (convert xyxy box to DFL bin format)
                # Normalize to [0, reg_max] range
                normalized_box = repeated_gt_box.clone()
                normalized_box[:, [0, 2]] = (normalized_box[:, [0, 2]] / img_size * reg_max).clamp(0, reg_max)
                normalized_box[:, [1, 3]] = (normalized_box[:, [1, 3]] / img_size * reg_max).clamp(0, reg_max)
                all_target_dfl.append(normalized_box)
    
    # Concatenate all assignments
    if len(all_assigned_boxes) == 0:
        # No assignments at all - return dummy tensors
        return (
            torch.zeros((0, 4 * (reg_max + 1)), device=device),
            torch.zeros((0, num_classes), device=device),
            torch.zeros((0, 4 * (reg_max + 1)), device=device),
            torch.zeros((0, 2), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0, 4), device=device),
            torch.zeros((0, num_classes), device=device),
            torch.zeros((0, 4), device=device, dtype=torch.long),
        )
    
    assigned_boxes = torch.cat(all_assigned_boxes, dim=0)       # (M, 4*(reg_max+1))
    assigned_cls_logits = torch.cat(all_assigned_cls, dim=0)    # (M, K)
    assigned_dfl_dist = torch.cat(all_assigned_dfl, dim=0)      # (M, 4*(reg_max+1))
    assigned_anchor_points = torch.cat(all_assigned_anchor_points, dim=0)  # (M, 2)
    assigned_strides = torch.cat(all_assigned_strides, dim=0)   # (M,)
    target_boxes = torch.cat(all_target_boxes, dim=0)           # (M, 4)
    target_cls_idx = torch.cat(all_target_cls, dim=0)           # (M,)
    target_dfl = torch.cat(all_target_dfl, dim=0).long()        # (M, 4)
    
    # Validate class indices are within bounds before one-hot encoding
    if target_cls_idx.numel() > 0:
        min_cls = target_cls_idx.min().item()
        max_cls = target_cls_idx.max().item()
        if min_cls < 0 or max_cls >= num_classes:
            # Debug info
            unique_classes = torch.unique(target_cls_idx).cpu().numpy()
            raise ValueError(
                f"Class indices out of bounds!\n"
                f"  min_cls={min_cls}, max_cls={max_cls}\n"
                f"  num_classes (from proto_sim K)={num_classes}\n"
                f"  unique_classes in batch={unique_classes}\n"
                f"  target_cls_idx shape={target_cls_idx.shape}\n"
                f"  Class indices must be in [0, {num_classes-1}].\n"
                f"  This usually means the episodic class remapping is not working correctly."
            )
    
    # Convert target classes to one-hot
    target_cls_onehot = F.one_hot(target_cls_idx.long(), num_classes=num_classes).float()  # (M, K)
    
    return (
        assigned_boxes,
        assigned_cls_logits,
        assigned_dfl_dist,
        assigned_anchor_points,
        assigned_strides,
        target_boxes,
        target_cls_onehot,
        target_dfl,
    )


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
    model_outputs: Dict,
    batch: Dict,
    stage: int = 2,
    reg_max: int = 16,
) -> Dict:
    """
    Prepare inputs for ReferenceBasedDetectionLoss using anchor-based assignment.
    
    This function now works directly with RAW detection head outputs and assigns
    targets to anchors, ensuring gradients flow even during early training.
    
    Args:
        model_outputs: Output dict from YOLOv8nRefDet forward pass with:
                       - 'prototype_boxes': List[(B, 4*(reg_max+1), H, W)]
                       - 'prototype_sim': List[(B, K, H, W)]
        batch: Batch dict from collator
        stage: Training stage (1, 2, or 3)
        reg_max: Maximum regression value for DFL
        
    Returns:
        loss_inputs: Dict ready for loss function
    """
    device = batch['query_images'].device
    num_classes = int(batch['num_classes'])  # Ensure it's int, not tensor
    
    # Extract target lists from batch (these are always Lists from collator)
    target_bboxes = batch['target_bboxes']  # type: List[torch.Tensor]
    target_classes = batch['target_classes']  # type: List[torch.Tensor]
    
    # Validate target_classes are within bounds BEFORE anchor assignment
    for img_idx, tc in enumerate(target_classes):
        if tc.numel() > 0:
            min_cls = tc.min().item()
            max_cls = tc.max().item()
            if min_cls < 0 or max_cls >= num_classes:
                raise ValueError(
                    f"Target class indices out of bounds at image {img_idx}!\n"
                    f"  Classes in image: {tc.cpu().numpy()}\n"
                    f"  min={min_cls}, max={max_cls}\n"
                    f"  num_classes={num_classes} (valid range: [0, {num_classes-1}])\n"
                    f"  This means the collator's episodic class remapping failed."
                )
    
    # Check if we have raw detection head outputs
    if 'prototype_boxes' in model_outputs and 'prototype_sim' in model_outputs:
        # NEW: Anchor-based assignment with raw outputs
        proto_boxes = model_outputs['prototype_boxes']  # type: List[torch.Tensor]
        proto_sim = model_outputs['prototype_sim']  # type: List[torch.Tensor]
        
        # Assign targets to anchors
        (matched_pred_bboxes, matched_pred_cls_logits, matched_pred_dfl_dist,
         matched_anchor_points, matched_assigned_strides, matched_target_bboxes, target_cls_onehot, target_dfl) = assign_targets_to_anchors(
            proto_boxes_list=proto_boxes,
            proto_sim_list=proto_sim,
            target_bboxes=target_bboxes,
            target_classes=target_classes,
            img_size=640,
            reg_max=reg_max,
        )
        
        # Decode DFL distributions to bbox coordinates using actual stride per anchor
        if matched_pred_dfl_dist.shape[0] > 0:
            # Initialize DFL decoder
            dfl_decoder = DFLoss(reg_max=reg_max)
            
            # Decode DFL dist to distances in grid cell units [0, reg_max]
            # matched_pred_dfl_dist: (M, 4*(reg_max+1))
            # decoded_dists_grid: (M, 4) in [0, reg_max] range (grid cell units)
            decoded_dists_grid = dfl_decoder.decode(matched_pred_dfl_dist)  # (M, 4) [left, top, right, bottom]
            
            # Convert grid distances to pixel distances using actual stride per anchor
            # matched_assigned_strides: (M,) stride value for each anchor
            # decoded_dists_grid: (M, 4) in [0, reg_max] grid units
            # Multiply each anchor's distances by its corresponding stride
            decoded_dists_pixel = decoded_dists_grid * matched_assigned_strides.unsqueeze(-1)  # (M, 4) in pixels
            
            # Convert distance format (ltrb from anchor) to xyxy bbox coordinates
            # matched_anchor_points: (M, 2) [x, y] in pixel coordinates
            # decoded_dists_pixel: (M, 4) [left, top, right, bottom] in pixels
            matched_pred_bboxes = dist2bbox(decoded_dists_pixel, matched_anchor_points, xywh=False, dim=-1)
        else:
            # No matches, keep as-is (empty tensor)
            matched_pred_bboxes = torch.zeros((0, 4), device=device)
        
        # Extract matched target classes from one-hot encoding for contrastive/triplet loss
        if target_cls_onehot.shape[0] > 0:
            matched_target_classes = target_cls_onehot.argmax(dim=1)
        else:
            matched_target_classes = torch.zeros(0, dtype=torch.long, device=device)
        
    else:
        # OLD: Backward compatibility with decoded predictions (fallback)
        pred_bboxes = model_outputs.get('pred_bboxes', torch.zeros((0, 4), device=device))
        pred_scores = model_outputs.get('pred_scores', torch.zeros(0, device=device))
        pred_cls_logits = model_outputs.get('pred_cls_logits', torch.zeros((0, num_classes), device=device))
        pred_dfl_dist = model_outputs.get('pred_dfl_dist', torch.zeros((0, 4 * (reg_max + 1)), device=device))
        
        # Match predictions to targets
        matched_pred_indices, matched_target_bboxes_list, matched_target_classes = match_predictions_to_targets(
            pred_bboxes=pred_bboxes,
            pred_scores=pred_scores,
            target_bboxes=target_bboxes,
            target_classes=target_classes,
        )
        
        # Prepare detection loss inputs
        if len(matched_pred_indices) > 0:
            # Get matched predictions
            matched_pred_bboxes = pred_bboxes[matched_pred_indices]
            matched_pred_cls_logits = pred_cls_logits[matched_pred_indices]
            matched_pred_dfl_dist = pred_dfl_dist[matched_pred_indices]
            matched_target_bboxes = matched_target_bboxes_list
            
            # Prepare classification targets (one-hot encoding)
            target_cls_onehot = F.one_hot(matched_target_classes, num_classes=num_classes).float()
            
            # Prepare DFL targets
            img_size = 640  # Query image size
            normalized_bboxes = matched_target_bboxes.clone()
            normalized_bboxes[:, [0, 2]] /= img_size
            normalized_bboxes[:, [1, 3]] /= img_size
            
            target_dfl = (normalized_bboxes * reg_max).long()
            target_dfl = torch.clamp(target_dfl, 0, reg_max)
            
        else:
            # No matches - use dummy tensors
            matched_pred_bboxes = torch.zeros((0, 4), device=device)
            matched_pred_cls_logits = torch.zeros((0, num_classes), device=device)
            matched_pred_dfl_dist = torch.zeros((0, 4 * (reg_max + 1)), device=device)
            target_cls_onehot = torch.zeros((0, num_classes), device=device)
            target_dfl = torch.zeros((0, 4), dtype=torch.long, device=device)
            matched_target_bboxes = torch.zeros((0, 4), device=device)
            matched_target_classes = torch.zeros(0, dtype=torch.long, device=device)
            matched_target_classes = torch.zeros(0, dtype=torch.long, device=device)
    
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
        # Proposal features and labels for CPE loss
        # Use query features from matched anchors as proposal features
        'proposal_features': None,  # Will be populated below
        'proposal_labels': None,    # Will be populated below
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
        
        # Feature labels for contrastive learning (per-image labels for episodic learning)
        # Use class_ids from batch if available (episodic learning)
        # Otherwise fall back to matched_target_classes (detection)
        if 'class_ids' in batch and batch['class_ids'] is not None:
            # Episodic learning: per-image class labels (B,)
            loss_inputs['feature_labels'] = batch['class_ids']
        elif len(matched_target_classes) > 0:
            # Fallback: per-anchor matched classes (M,)
            loss_inputs['feature_labels'] = matched_target_classes
        
        # For batch-hard triplet loss in Stage 2+, use query_features as embeddings
        # BatchHardTripletLoss will automatically mine hard triplets from the batch
        if 'query_features' in model_outputs:
            loss_inputs['triplet_embeddings'] = model_outputs['query_features']
            # Use same labels as contrastive loss
            if 'class_ids' in batch and batch['class_ids'] is not None:
                loss_inputs['triplet_labels'] = batch['class_ids']
            elif len(matched_target_classes) > 0:
                loss_inputs['triplet_labels'] = matched_target_classes
        
        # For CPE loss: extract ROI features from fused feature maps for each matched anchor
        # CPE loss requires proposal features and labels for contrastive learning
        if len(matched_pred_bboxes) > 0 and len(matched_target_classes) > 0:
            fused_features = model_outputs.get('fused_features', {})
            
            if fused_features and len(fused_features) > 0:
                # Extract ROI features from fused feature maps
                try:
                    proposal_features = extract_roi_features(
                        feature_maps=fused_features,
                        bboxes=matched_pred_bboxes,  # Use predicted bboxes as proposals
                        output_size=7,
                    )
                    
                    if proposal_features.shape[0] > 0:
                        loss_inputs['proposal_features'] = proposal_features
                        loss_inputs['proposal_labels'] = matched_target_classes
                except Exception as e:
                    # If ROI extraction fails, fall back to per-image query features
                    if 'query_features' in model_outputs:
                        query_features = model_outputs['query_features']
                        # Repeat query features for each matched anchor as fallback
                        batch_size = batch['query_images'].shape[0]
                        if query_features.shape[0] == batch_size and matched_pred_bboxes.shape[0] > 0:
                            # Assign each anchor to its corresponding image
                            # This is simplified - in practice, need proper batch indexing
                            num_anchors = matched_pred_bboxes.shape[0]
                            anchors_per_image = num_anchors // batch_size
                            expanded_features = query_features.repeat_interleave(anchors_per_image, dim=0)
                            if expanded_features.shape[0] == num_anchors:
                                loss_inputs['proposal_features'] = expanded_features
                                loss_inputs['proposal_labels'] = matched_target_classes
            else:
                # No fused_features available, try using query_features directly
                if 'query_features' in model_outputs:
                    query_features = model_outputs['query_features']
                    batch_size = batch['query_images'].shape[0]
                    
                    # Check if query_features matches anchor count
                    if query_features.shape[0] == matched_pred_bboxes.shape[0]:
                        # Per-anchor features: perfect for CPE loss
                        loss_inputs['proposal_features'] = query_features
                        loss_inputs['proposal_labels'] = matched_target_classes
                    elif query_features.shape[0] == batch_size and matched_pred_bboxes.shape[0] > 0:
                        # Per-image features: expand to per-anchor
                        num_anchors = matched_pred_bboxes.shape[0]
                        anchors_per_image = num_anchors // batch_size
                        expanded_features = query_features.repeat_interleave(anchors_per_image, dim=0)
                        if expanded_features.shape[0] == num_anchors:
                            loss_inputs['proposal_features'] = expanded_features
                            loss_inputs['proposal_labels'] = matched_target_classes
    
    # Add triplet loss inputs for Stage 3 (for regular triplet loss with anchor/pos/neg)
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
    num_boxes = bboxes.shape[0] if len(bboxes.shape) > 1 else len(bboxes)  # type: ignore
    batch_indices = torch.zeros((num_boxes, 1), device=bboxes.device)
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
    
    # Add batch-hard triplet loss inputs (embeddings + labels)
    # For BatchHardTripletLoss: concatenate all features as embeddings
    # Labels: 0=anchor, 1=positive, 2=negative (or use class_ids from batch)
    if 'class_ids' in batch:
        # Use actual class IDs if available
        class_ids = batch['class_ids']
        # Create embeddings: concatenate anchor, positive, negative
        all_embeddings = torch.cat([anchor_features, positive_features, negative_features], dim=0)
        # Create labels: repeat class_ids for anchor, positive, negative
        # For proper triplet mining: anchors and positives should have same label
        all_labels = torch.cat([class_ids, class_ids, class_ids], dim=0)
        
        loss_inputs['triplet_embeddings'] = all_embeddings
        loss_inputs['triplet_labels'] = all_labels
    elif 'labels' in batch:
        # Fallback to generic labels
        labels = batch['labels']
        all_embeddings = torch.cat([anchor_features, positive_features, negative_features], dim=0)
        all_labels = torch.cat([labels, labels, labels], dim=0)
        
        loss_inputs['triplet_embeddings'] = all_embeddings
        loss_inputs['triplet_labels'] = all_labels
    
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
