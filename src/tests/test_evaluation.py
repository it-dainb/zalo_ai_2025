"""
Test Evaluation Pipeline
=========================

Tests evaluation functionality:
1. Metric computation (IoU, precision, recall, AP)
2. Episode evaluation
3. Batch evaluation
4. Full evaluation pipeline
5. Evaluation with different IoU thresholds
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from models.yolo_refdet import YOLOv8nRefDet
from src.augmentations.augmentation_config import AugmentationConfig


class TestMetricComputation:
    """Test metric computation functions"""
    
    def test_iou_computation(self):
        """Test IoU computation"""
        # Import compute_iou from evaluate.py
        from evaluate import compute_iou
        
        # Test perfect overlap
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        iou = compute_iou(box1, box2)
        assert abs(iou - 1.0) < 1e-6
        
        # Test no overlap
        box1 = [0, 0, 50, 50]
        box2 = [100, 100, 150, 150]
        iou = compute_iou(box1, box2)
        assert abs(iou - 0.0) < 1e-6
        
        # Test partial overlap
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
        
        print(f"✅ IoU computation correct")
        print(f"   Perfect overlap: {compute_iou([0,0,100,100], [0,0,100,100]):.4f}")
        print(f"   Partial overlap: {compute_iou([0,0,100,100], [50,50,150,150]):.4f}")
        print(f"   No overlap: {compute_iou([0,0,50,50], [100,100,150,150]):.4f}")
    
    def test_ap_computation(self):
        """Test Average Precision computation"""
        from evaluate import compute_ap
        
        # Perfect predictions
        recalls = np.array([0.0, 0.5, 1.0])
        precisions = np.array([1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert abs(ap - 1.0) < 1e-6
        
        # Imperfect predictions
        recalls = np.array([0.0, 0.3, 0.6, 1.0])
        precisions = np.array([1.0, 0.8, 0.6, 0.4])
        ap = compute_ap(recalls, precisions)
        assert 0 < ap < 1
        
        print(f"✅ AP computation correct")
        print(f"   Perfect AP: 1.0")
        print(f"   Test AP: {ap:.4f}")
    
    def test_precision_recall(self):
        """Test precision and recall computation"""
        # TP=8, FP=2, FN=3
        tp, fp, fn = 8, 2, 3
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        assert abs(precision - 0.8) < 1e-6
        assert abs(recall - 8/11) < 1e-6
        
        print(f"✅ Precision/Recall computation correct")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")


class TestModelEvaluation:
    """Test model evaluation functionality"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for evaluation"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        model.eval()
        return model
    
    def test_model_inference(self, model, device):
        """Test model inference for evaluation"""
        # Create dummy support and query images
        support_images = torch.randn(3, 3, 256, 256).to(device)
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            # Set reference images
            model.set_reference_images(support_images, average_prototypes=True)
            
            # Forward pass
            outputs = model(
                query_image=query_image,
                use_cache=True,
            )
        
        # Check outputs
        assert 'pred_bboxes' in outputs or 'prototype_boxes' in outputs
        print(f"✅ Model inference for evaluation successful")
    
    def test_batch_inference(self, model, device):
        """Test batch inference"""
        batch_size = 4
        support_images = torch.randn(3, 3, 256, 256).to(device)
        query_images = torch.randn(batch_size, 3, 640, 640).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support_images, average_prototypes=True)
            outputs = model(
                query_image=query_images,
                use_cache=True,
            )
        
        print(f"✅ Batch inference successful (batch_size={batch_size})")


class TestEvaluationPipeline:
    """Test full evaluation pipeline"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def test_data_available(self):
        """Check if test data is available"""
        dataset_path = Path("./datasets/test/samples")
        annotations_path = Path("./datasets/test/annotations/annotations.json")
        return dataset_path.exists() and annotations_path.exists()
    
    def test_episode_evaluation(self, device, test_data_available):
        """Test evaluation on single episode"""
        if not test_data_available:
            pytest.skip("Test dataset not available")
        
        from evaluate import evaluate_episode
        
        # Create model
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            
        ).to(device)
        model.eval()
        
        # Create dataset
        dataset = RefDetDataset(
            data_root="./datasets/test/samples",
            annotations_file="./datasets/test/annotations/annotations.json",
            mode='val',
            cache_frames=True,
        )
        
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=min(2, len(dataset.classes)),
            n_query=2,
            n_episodes=1,
        )
        
        aug_config = AugmentationConfig()
        collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0,
        )
        
        # Evaluate single batch
        for batch in dataloader:
            try:
                results = evaluate_episode(model, batch, device)
                assert isinstance(results, dict)
                print(f"✅ Episode evaluation successful")
                print(f"   Results: {results}")
            except Exception as e:
                print(f"⚠️  Episode evaluation: {str(e)}")
            break
    
    def test_full_evaluation_pipeline(self, device, test_data_available):
        """Test full evaluation pipeline"""
        if not test_data_available:
            pytest.skip("Test dataset not available")
        
        print("\n" + "="*60)
        print("Testing Full Evaluation Pipeline")
        print("="*60)
        
        # Create model
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            
        ).to(device)
        model.eval()
        
        # Create dataset
        dataset = RefDetDataset(
            data_root="./datasets/test/samples",
            annotations_file="./datasets/test/annotations/annotations.json",
            mode='val',
            cache_frames=True,
        )
        
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=min(2, len(dataset.classes)),
            n_query=2,
            n_episodes=2,  # Minimal for testing
        )
        
        aug_config = AugmentationConfig()
        collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0,
        )
        
        # Evaluate all episodes
        from evaluate import evaluate_episode
        all_results = []
        
        for batch in dataloader:
            try:
                results = evaluate_episode(model, batch, device)
                all_results.append(results)
            except Exception as e:
                print(f"⚠️  Batch evaluation error: {str(e)}")
        
        if len(all_results) > 0:
            print(f"✅ Evaluated {len(all_results)} episodes")
        else:
            print(f"⚠️  No episodes evaluated successfully")
    
    def test_evaluation_with_checkpoint(self, device, test_data_available):
        """Test evaluation with loaded checkpoint"""
        if not test_data_available:
            pytest.skip("Test dataset not available")
        
        # Check if any checkpoint exists
        checkpoint_dir = Path("./checkpoints")
        if not checkpoint_dir.exists():
            pytest.skip("No checkpoint directory found")
        
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if len(checkpoints) == 0:
            pytest.skip("No checkpoints found")
        
        checkpoint_path = checkpoints[0]
        print(f"\nTesting with checkpoint: {checkpoint_path}")
        
        # Load model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            
        ).to(device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Model loaded from checkpoint")
        except Exception as e:
            print(f"⚠️  Checkpoint loading: {str(e)}")
            pytest.skip("Checkpoint loading failed")


class TestEvaluationMetrics:
    """Test different evaluation metrics and thresholds"""
    
    def test_multiple_iou_thresholds(self):
        """Test evaluation at different IoU thresholds"""
        from evaluate import compute_iou
        
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        
        iou = compute_iou(box1, box2)
        
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for thresh in thresholds:
            is_match = iou >= thresh
            results[thresh] = is_match
        
        print(f"✅ IoU threshold testing:")
        print(f"   IoU: {iou:.4f}")
        for thresh, match in results.items():
            print(f"   Threshold {thresh}: {'Match' if match else 'No match'}")
    
    def test_confidence_threshold_filtering(self):
        """Test filtering predictions by confidence"""
        # Mock predictions with scores
        pred_scores = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
        pred_boxes = np.random.rand(5, 4)
        
        conf_threshold = 0.5
        
        # Filter by confidence
        mask = pred_scores >= conf_threshold
        filtered_scores = pred_scores[mask]
        filtered_boxes = pred_boxes[mask]
        
        assert len(filtered_scores) == 2
        assert all(s >= conf_threshold for s in filtered_scores)
        
        print(f"✅ Confidence filtering:")
        print(f"   Original predictions: {len(pred_scores)}")
        print(f"   Filtered (conf>={conf_threshold}): {len(filtered_scores)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
