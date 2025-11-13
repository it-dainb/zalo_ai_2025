"""
Standard Detection Metrics (Precision, Recall, AP, mAP).

These metrics complement ST-IoU for comprehensive model evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        iou: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def compute_precision_recall(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        pred_bboxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        pred_classes: (N,) predicted classes
        gt_bboxes: (M, 4) ground truth boxes
        gt_classes: (M,) ground truth classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        metrics: Dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if len(pred_bboxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_bboxes)}
    
    if len(gt_bboxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_bboxes), 'fn': 0}
    
    # Match predictions to ground truth
    matched_gt = set()
    tp = 0
    fp = 0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        pred_box = pred_bboxes[pred_idx]
        pred_class = pred_classes[pred_idx]
        
        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_bboxes)):
            if gt_idx in matched_gt:
                continue
            
            if gt_classes[gt_idx] != pred_class:
                continue
            
            iou = compute_iou(pred_box, gt_bboxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is valid
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # Count unmatched GT as false negatives
    fn = len(gt_bboxes) - len(matched_gt)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def compute_ap(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None,
) -> float:
    """
    Compute Average Precision (AP) for a single class.
    
    Args:
        pred_bboxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        pred_classes: (N,) predicted classes
        gt_bboxes: (M, 4) ground truth boxes
        gt_classes: (M,) ground truth classes
        iou_threshold: IoU threshold for matching
        class_id: If provided, compute AP only for this class
        
    Returns:
        ap: Average Precision value
    """
    # Filter by class if specified
    if class_id is not None:
        pred_mask = pred_classes == class_id
        gt_mask = gt_classes == class_id
        
        pred_bboxes = pred_bboxes[pred_mask]
        pred_scores = pred_scores[pred_mask]
        pred_classes = pred_classes[pred_mask]
        
        gt_bboxes = gt_bboxes[gt_mask]
        gt_classes = gt_classes[gt_mask]
    
    if len(gt_bboxes) == 0:
        return 0.0
    
    if len(pred_bboxes) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_bboxes = pred_bboxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_classes = pred_classes[sorted_indices]
    
    # Compute TP and FP for each prediction
    tp = np.zeros(len(pred_bboxes))
    fp = np.zeros(len(pred_bboxes))
    matched_gt = set()
    
    for pred_idx in range(len(pred_bboxes)):
        pred_box = pred_bboxes[pred_idx]
        pred_class = pred_classes[pred_idx]
        
        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_bboxes)):
            if gt_idx in matched_gt:
                continue
            
            if gt_classes[gt_idx] != pred_class:
                continue
            
            iou = compute_iou(pred_box, gt_bboxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is valid
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall at each threshold
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(gt_bboxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP using 11-point interpolation
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Monotonic precision
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Compute area under PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return float(ap)


def compute_map(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean Average Precision (mAP) across all classes.
    
    Args:
        pred_bboxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        pred_classes: (N,) predicted classes
        gt_bboxes: (M, 4) ground truth boxes
        gt_classes: (M,) ground truth classes
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes (if None, infer from data)
        
    Returns:
        map_score: Mean Average Precision
        ap_per_class: Dict mapping class_id -> AP
    """
    # Get all classes
    all_classes = set(gt_classes.tolist())
    if len(pred_classes) > 0:
        all_classes.update(pred_classes.tolist())
    
    if num_classes is not None:
        all_classes.update(range(num_classes))
    
    # Compute AP for each class
    ap_per_class = {}
    
    for class_id in all_classes:
        ap = compute_ap(
            pred_bboxes,
            pred_scores,
            pred_classes,
            gt_bboxes,
            gt_classes,
            iou_threshold=iou_threshold,
            class_id=class_id,
        )
        ap_per_class[class_id] = ap
    
    # Compute mAP
    if len(ap_per_class) > 0:
        map_score = np.mean(list(ap_per_class.values()))
    else:
        map_score = 0.0
    
    return float(map_score), ap_per_class


def compute_map_at_iou_range(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    num_classes: Optional[int] = None,
) -> Tuple[float, Dict[float, float]]:
    """
    Compute mAP at multiple IoU thresholds (similar to COCO mAP@[.5:.95]).
    
    Args:
        pred_bboxes: (N, 4) predicted boxes
        pred_scores: (N,) confidence scores
        pred_classes: (N,) predicted classes
        gt_bboxes: (M, 4) ground truth boxes
        gt_classes: (M,) ground truth classes
        iou_thresholds: List of IoU thresholds to evaluate
        num_classes: Number of classes
        
    Returns:
        mean_map: Mean mAP across all IoU thresholds
        map_per_iou: Dict mapping iou_threshold -> mAP
    """
    map_per_iou = {}
    
    for iou_threshold in iou_thresholds:
        map_score, _ = compute_map(
            pred_bboxes,
            pred_scores,
            pred_classes,
            gt_bboxes,
            gt_classes,
            iou_threshold=iou_threshold,
            num_classes=num_classes,
        )
        map_per_iou[iou_threshold] = map_score
    
    # Compute mean mAP
    mean_map = np.mean(list(map_per_iou.values())) if len(map_per_iou) > 0 else 0.0
    
    return float(mean_map), map_per_iou
