"""
Simple evaluation script for trained YOLOv8n-RefDet model.
Computes few-shot detection metrics on test set.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.augmentations.augmentation_config import AugmentationConfig


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def compute_ap(recalls, precisions):
    """Compute Average Precision."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_episode(model, batch, device, iou_threshold=0.5):
    """
    Evaluate model on one episode.
    
    Returns:
        metrics: Dict with precision, recall, AP per class
    """
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Prepare support images
    support_images = batch['support_images']
    N, K, C, H, W = support_images.shape
    support_flat = support_images.reshape(N * K, C, H, W)
    
    # Cache support features
    model.set_reference_images(support_flat, average_prototypes=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            query_image=batch['query_images'],
            mode='prototype',  # Use prototype matching for novel objects
            use_cache=True,
        )
    
    # Extract predictions
    pred_bboxes = outputs['pred_bboxes'].cpu().numpy()
    pred_scores = outputs['pred_scores'].cpu().numpy()
    pred_classes = outputs['pred_class_ids'].cpu().numpy()
    
    # Ground truth
    gt_bboxes_list = [b.cpu().numpy() for b in batch['target_bboxes']]
    gt_classes_list = [c.cpu().numpy() for c in batch['target_classes']]
    
    # Compute metrics per class
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'scores': []})
    
    for img_idx in range(len(batch['query_images'])):
        gt_bboxes = gt_bboxes_list[img_idx]
        gt_classes = gt_classes_list[img_idx]
        
        # Get predictions for this image (simplified - assumes all preds are from one image)
        img_preds = pred_bboxes
        img_scores = pred_scores
        img_classes = pred_classes
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred_idx in range(len(img_preds)):
            pred_box = img_preds[pred_idx]
            pred_class = img_classes[pred_idx]
            pred_score = img_scores[pred_idx]
            
            # Find best matching ground truth
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
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                class_metrics[pred_class]['tp'] += 1
                class_metrics[pred_class]['scores'].append(pred_score)
                matched_gt.add(best_gt_idx)
            else:
                class_metrics[pred_class]['fp'] += 1
                class_metrics[pred_class]['scores'].append(pred_score)
        
        # Count false negatives (unmatched ground truth)
        for gt_idx in range(len(gt_bboxes)):
            if gt_idx not in matched_gt:
                class_metrics[gt_classes[gt_idx]]['fn'] += 1
    
    # Compute precision, recall, AP per class
    results = {}
    for class_id, metrics in class_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results[class_id] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8n-RefDet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./datasets/test/samples',
                        help='Root directory containing test samples')
    parser.add_argument('--annotations', type=str, default='./datasets/test/annotations/annotations.json',
                        help='Path to test annotations.json')
    parser.add_argument('--n_way', type=int, default=2,
                        help='Number of classes per episode')
    parser.add_argument('--n_query', type=int, default=4,
                        help='Number of query samples per class')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='Number of episodes to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to evaluate on')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("YOLOv8n-RefDet Evaluation")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    model = YOLOv8nRefDet(
        yolo_weights='yolov8n.pt',
        nc_base=0,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print(f"✓ Model loaded from {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Create dataset
    print("\nLoading test dataset...")
    test_dataset = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='val',
        cache_frames=True,
    )
    
    test_sampler = EpisodicBatchSampler(
        dataset=test_dataset,
        n_way=args.n_way,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
    )
    
    aug_config = AugmentationConfig()
    test_collator = RefDetCollator(
        config=aug_config,
        mode='val',
        stage=2,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=test_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"✓ Test dataset loaded")
    print(f"  Episodes: {args.n_episodes}")
    
    # Evaluate
    print("\nEvaluating...")
    all_results = []
    
    for batch in tqdm(test_loader, desc="Evaluation"):
        results = evaluate_episode(model, batch, args.device)
        all_results.append(results)
    
    # Aggregate results
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for results in all_results:
        for class_id, metrics in results.items():
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
    
    # Compute overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Total TP: {total_tp}")
    print(f"  Total FP: {total_fp}")
    print(f"  Total FN: {total_fn}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
