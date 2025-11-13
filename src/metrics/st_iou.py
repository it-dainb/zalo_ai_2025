"""
Spatio-Temporal IoU (ST-IoU) Metric for Video Object Detection.

The ST-IoU metric jointly measures when and where the target object is correctly
detected in the video, treating temporal and spatial accuracy as a continuous
space-time volume.

Reference:
    Zalo AI Challenge 2025 - Qualification Round
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def compute_spatial_iou(
    box1: Union[torch.Tensor, np.ndarray],
    box2: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute spatial IoU between two bounding boxes.
    
    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]
        
    Returns:
        iou: IoU value between 0 and 1
    """
    # Convert to numpy for easier computation
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return float(iou)


def compute_st_iou(
    gt_detections: Dict[int, np.ndarray],
    pred_detections: Dict[int, np.ndarray],
    video_length: Optional[int] = None,
) -> float:
    """
    Compute Spatio-Temporal IoU (ST-IoU) for a single video.
    
    ST-IoU Formula:
        ST-IoU = Σ(IoU(B_f, B'_f) for f in intersection) / |union|
    
    where:
        - intersection: overlapping frames between predicted and ground-truth
        - IoU(B_f, B'_f): spatial IoU of bounding boxes at frame f
        - union: all frames that belong to either ground-truth or predicted
    
    Args:
        gt_detections: Dict mapping frame_id -> bbox array [x1, y1, x2, y2]
        pred_detections: Dict mapping frame_id -> bbox array [x1, y1, x2, y2]
        video_length: Total video length (optional, for normalization)
        
    Returns:
        st_iou: Spatio-temporal IoU value between 0 and 1
        
    Example:
        >>> gt_dets = {0: [10, 10, 50, 50], 1: [15, 15, 55, 55], 2: [20, 20, 60, 60]}
        >>> pred_dets = {1: [12, 12, 52, 52], 2: [18, 18, 58, 58], 3: [25, 25, 65, 65]}
        >>> st_iou = compute_st_iou(gt_dets, pred_dets)
        >>> # intersection frames: {1, 2}
        >>> # union frames: {0, 1, 2, 3}
        >>> # st_iou = (IoU(gt[1], pred[1]) + IoU(gt[2], pred[2])) / 4
    """
    # Get frame sets
    gt_frames = set(gt_detections.keys())
    pred_frames = set(pred_detections.keys())
    
    # Compute intersection and union of frames
    intersection_frames = gt_frames.intersection(pred_frames)
    union_frames = gt_frames.union(pred_frames)
    
    # Handle empty case
    if len(union_frames) == 0:
        return 0.0
    
    # Compute sum of spatial IoUs for overlapping frames
    iou_sum = 0.0
    for frame_id in intersection_frames:
        gt_box = gt_detections[frame_id]
        pred_box = pred_detections[frame_id]
        spatial_iou = compute_spatial_iou(gt_box, pred_box)
        iou_sum += spatial_iou
    
    # Compute ST-IoU
    st_iou = iou_sum / len(union_frames)
    
    return st_iou


def compute_st_iou_batch(
    gt_detections_batch: List[Dict[int, np.ndarray]],
    pred_detections_batch: List[Dict[int, np.ndarray]],
    video_lengths: Optional[List[int]] = None,
) -> Tuple[float, List[float]]:
    """
    Compute ST-IoU for a batch of videos and return mean + individual scores.
    
    Args:
        gt_detections_batch: List of ground-truth detection dicts per video
        pred_detections_batch: List of predicted detection dicts per video
        video_lengths: Optional list of video lengths
        
    Returns:
        mean_st_iou: Mean ST-IoU across all videos
        st_iou_per_video: List of individual ST-IoU scores
        
    Example:
        >>> gt_batch = [
        ...     {0: [10, 10, 50, 50], 1: [15, 15, 55, 55]},
        ...     {0: [20, 20, 60, 60], 1: [25, 25, 65, 65]},
        ... ]
        >>> pred_batch = [
        ...     {0: [12, 12, 52, 52], 1: [17, 17, 57, 57]},
        ...     {0: [22, 22, 62, 62], 1: [27, 27, 67, 67]},
        ... ]
        >>> mean_st_iou, per_video = compute_st_iou_batch(gt_batch, pred_batch)
    """
    assert len(gt_detections_batch) == len(pred_detections_batch), \
        "Number of GT and pred videos must match"
    
    st_iou_scores = []
    
    for i in range(len(gt_detections_batch)):
        gt_dets = gt_detections_batch[i]
        pred_dets = pred_detections_batch[i]
        video_len = video_lengths[i] if video_lengths is not None else None
        
        st_iou = compute_st_iou(gt_dets, pred_dets, video_len)
        st_iou_scores.append(st_iou)
    
    # Compute mean ST-IoU
    mean_st_iou = np.mean(st_iou_scores) if len(st_iou_scores) > 0 else 0.0
    
    return float(mean_st_iou), st_iou_scores


def match_predictions_to_gt(
    pred_bboxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    gt_bboxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Dict[int, Tuple[int, float]]:
    """
    Match predicted bounding boxes to ground truth for a single frame.
    
    Uses Hungarian matching: each prediction is matched to at most one GT box,
    and vice versa.
    
    Args:
        pred_bboxes: (N_pred, 4) predicted boxes [x1, y1, x2, y2]
        pred_scores: (N_pred,) confidence scores
        pred_classes: (N_pred,) predicted class IDs
        gt_bboxes: (N_gt, 4) ground truth boxes
        gt_classes: (N_gt,) ground truth class IDs
        iou_threshold: Minimum IoU to consider a match
        score_threshold: Minimum confidence to consider prediction
        
    Returns:
        matches: Dict mapping gt_idx -> (pred_idx, iou)
        
    Example:
        >>> pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        >>> pred_scores = np.array([0.9, 0.8])
        >>> pred_classes = np.array([0, 0])
        >>> gt_boxes = np.array([[12, 12, 52, 52]])
        >>> gt_classes = np.array([0])
        >>> matches = match_predictions_to_gt(pred_boxes, pred_scores, pred_classes,
        ...                                   gt_boxes, gt_classes)
        >>> # matches = {0: (0, 0.85)} - gt[0] matched to pred[0] with IoU 0.85
    """
    matches = {}
    
    # Filter predictions by score threshold
    valid_mask = pred_scores >= score_threshold
    pred_bboxes = pred_bboxes[valid_mask]
    pred_scores = pred_scores[valid_mask]
    pred_classes = pred_classes[valid_mask]
    
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return matches
    
    # Compute IoU matrix (N_gt x N_pred)
    iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)))
    
    for gt_idx in range(len(gt_bboxes)):
        for pred_idx in range(len(pred_bboxes)):
            # Only match same class
            if gt_classes[gt_idx] != pred_classes[pred_idx]:
                continue
            
            iou = compute_spatial_iou(gt_bboxes[gt_idx], pred_bboxes[pred_idx])
            iou_matrix[gt_idx, pred_idx] = iou
    
    # Greedy matching: assign highest IoU first
    matched_preds = set()
    matched_gts = set()
    
    # Sort by IoU (descending)
    gt_indices, pred_indices = np.where(iou_matrix >= iou_threshold)
    ious = iou_matrix[gt_indices, pred_indices]
    sorted_indices = np.argsort(-ious)
    
    for idx in sorted_indices:
        gt_idx = gt_indices[idx]
        pred_idx = pred_indices[idx]
        iou = ious[idx]
        
        # Skip if already matched
        if gt_idx in matched_gts or pred_idx in matched_preds:
            continue
        
        matches[gt_idx] = (pred_idx, iou)
        matched_gts.add(gt_idx)
        matched_preds.add(pred_idx)
    
    return matches


def extract_st_detections_from_video_predictions(
    video_predictions: Dict,
    score_threshold: float = 0.25,
    class_filter: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract frame-level detections from video predictions.
    
    Converts model outputs to the format expected by ST-IoU computation.
    
    Args:
        video_predictions: Dict with keys:
            - 'frame_ids': List of frame indices
            - 'bboxes': (N_frames, N_boxes, 4) or list of (N_boxes_i, 4)
            - 'scores': (N_frames, N_boxes) or list of (N_boxes_i,)
            - 'classes': (N_frames, N_boxes) or list of (N_boxes_i,)
        score_threshold: Minimum confidence to include detection
        class_filter: If provided, only include this class
        
    Returns:
        detections: Dict mapping frame_id -> best bbox [x1, y1, x2, y2]
        
    Example:
        >>> video_preds = {
        ...     'frame_ids': [0, 1, 2],
        ...     'bboxes': [np.array([[10, 10, 50, 50]]),
        ...                np.array([[15, 15, 55, 55]]),
        ...                np.array([[20, 20, 60, 60]])],
        ...     'scores': [np.array([0.9]), np.array([0.85]), np.array([0.8])],
        ...     'classes': [np.array([0]), np.array([0]), np.array([0])],
        ... }
        >>> detections = extract_st_detections_from_video_predictions(video_preds)
        >>> # detections = {0: [10, 10, 50, 50], 1: [15, 15, 55, 55], 2: [20, 20, 60, 60]}
    """
    detections = {}
    
    frame_ids = video_predictions['frame_ids']
    bboxes_per_frame = video_predictions['bboxes']
    scores_per_frame = video_predictions['scores']
    classes_per_frame = video_predictions['classes']
    
    for i, frame_id in enumerate(frame_ids):
        frame_bboxes = bboxes_per_frame[i]
        frame_scores = scores_per_frame[i]
        frame_classes = classes_per_frame[i]
        
        # Filter by score
        valid_mask = frame_scores >= score_threshold
        frame_bboxes = frame_bboxes[valid_mask]
        frame_scores = frame_scores[valid_mask]
        frame_classes = frame_classes[valid_mask]
        
        # Filter by class if specified
        if class_filter is not None:
            class_mask = frame_classes == class_filter
            frame_bboxes = frame_bboxes[class_mask]
            frame_scores = frame_scores[class_mask]
        
        # Select highest confidence detection for this frame
        if len(frame_bboxes) > 0:
            best_idx = np.argmax(frame_scores)
            detections[frame_id] = frame_bboxes[best_idx]
    
    return detections
