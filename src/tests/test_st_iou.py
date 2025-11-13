"""
Test ST-IoU metric implementation.
"""

import pytest
import numpy as np
from src.metrics.st_iou import (
    compute_spatial_iou,
    compute_st_iou,
    compute_st_iou_batch,
    extract_st_detections_from_video_predictions,
)


def test_spatial_iou():
    """Test spatial IoU computation."""
    # Perfect overlap
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([10, 10, 50, 50])
    iou = compute_spatial_iou(box1, box2)
    assert iou == 1.0
    
    # No overlap
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([60, 60, 100, 100])
    iou = compute_spatial_iou(box1, box2)
    assert iou == 0.0
    
    # Partial overlap
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([30, 30, 70, 70])
    iou = compute_spatial_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_st_iou_single_video():
    """Test ST-IoU for single video."""
    # Test case 1: Perfect tracking
    gt_dets = {
        0: np.array([10, 10, 50, 50]),
        1: np.array([15, 15, 55, 55]),
        2: np.array([20, 20, 60, 60]),
    }
    pred_dets = {
        0: np.array([10, 10, 50, 50]),
        1: np.array([15, 15, 55, 55]),
        2: np.array([20, 20, 60, 60]),
    }
    st_iou = compute_st_iou(gt_dets, pred_dets)
    assert st_iou == 1.0
    
    # Test case 2: Partial temporal overlap
    gt_dets = {
        0: np.array([10, 10, 50, 50]),
        1: np.array([15, 15, 55, 55]),
        2: np.array([20, 20, 60, 60]),
    }
    pred_dets = {
        1: np.array([12, 12, 52, 52]),  # Frame 1 only
        2: np.array([18, 18, 58, 58]),  # Frame 2 only
        3: np.array([25, 25, 65, 65]),  # Frame 3 (false positive)
    }
    st_iou = compute_st_iou(gt_dets, pred_dets)
    # intersection frames: {1, 2}
    # union frames: {0, 1, 2, 3}
    # st_iou = (IoU(gt[1], pred[1]) + IoU(gt[2], pred[2])) / 4
    assert 0.0 < st_iou < 1.0
    
    # Test case 3: No temporal overlap
    gt_dets = {
        0: np.array([10, 10, 50, 50]),
        1: np.array([15, 15, 55, 55]),
    }
    pred_dets = {
        2: np.array([20, 20, 60, 60]),
        3: np.array([25, 25, 65, 65]),
    }
    st_iou = compute_st_iou(gt_dets, pred_dets)
    # No overlapping frames, so ST-IoU = 0
    assert st_iou == 0.0


def test_st_iou_batch():
    """Test ST-IoU batch computation."""
    gt_batch = [
        {0: np.array([10, 10, 50, 50]), 1: np.array([15, 15, 55, 55])},
        {0: np.array([20, 20, 60, 60]), 1: np.array([25, 25, 65, 65])},
    ]
    pred_batch = [
        {0: np.array([12, 12, 52, 52]), 1: np.array([17, 17, 57, 57])},
        {0: np.array([22, 22, 62, 62]), 1: np.array([27, 27, 67, 67])},
    ]
    
    mean_st_iou, per_video = compute_st_iou_batch(gt_batch, pred_batch)
    
    assert len(per_video) == 2
    assert 0.0 <= mean_st_iou <= 1.0
    for st_iou in per_video:
        assert 0.0 <= st_iou <= 1.0


def test_extract_st_detections():
    """Test extraction of ST detections from video predictions."""
    video_preds = {
        'frame_ids': [0, 1, 2],
        'bboxes': [
            np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            np.array([[15, 15, 55, 55]]),
            np.array([[20, 20, 60, 60]]),
        ],
        'scores': [
            np.array([0.9, 0.3]),  # First box has higher confidence
            np.array([0.85]),
            np.array([0.8]),
        ],
        'classes': [
            np.array([0, 1]),
            np.array([0]),
            np.array([0]),
        ],
    }
    
    detections = extract_st_detections_from_video_predictions(
        video_preds,
        score_threshold=0.25,
        class_filter=0,
    )
    
    # Should extract 3 detections (one per frame) for class 0
    assert len(detections) == 3
    assert 0 in detections
    assert 1 in detections
    assert 2 in detections
    
    # Frame 0 should select first box (class 0, score 0.9)
    np.testing.assert_array_equal(detections[0], [10, 10, 50, 50])


def test_empty_detections():
    """Test handling of empty detections."""
    # Empty GT and predictions
    st_iou = compute_st_iou({}, {})
    assert st_iou == 0.0
    
    # Empty GT
    pred_dets = {0: np.array([10, 10, 50, 50])}
    st_iou = compute_st_iou({}, pred_dets)
    assert st_iou == 0.0
    
    # Empty predictions
    gt_dets = {0: np.array([10, 10, 50, 50])}
    st_iou = compute_st_iou(gt_dets, {})
    assert st_iou == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
