"""
Quick test script to verify the training pipeline setup.
Tests dataset loading, model forward pass, and loss computation.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.augmentations.augmentation_config import AugmentationConfig


def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    try:
        dataset = RefDetDataset(
            data_root='./datasets/train/samples',
            annotations_file='./datasets/train/annotations/annotations.json',
            mode='train',
            cache_frames=False,
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total classes: {len(dataset.classes)}")
        print(f"  Total frames: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print(f"✓ Single sample loaded")
        print(f"  Query frame shape: {sample['query_frame'].shape}")
        print(f"  Bboxes shape: {sample['bboxes'].shape}")
        print(f"  Support images: {len(sample['support_images'])}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_episodic_sampler():
    """Test episodic sampler."""
    print("\n" + "="*60)
    print("Testing Episodic Sampler")
    print("="*60)
    
    try:
        dataset = RefDetDataset(
            data_root='./datasets/train/samples',
            annotations_file='./datasets/train/annotations/annotations.json',
            mode='train',
            cache_frames=False,
        )
        
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=2,
            n_query=2,
            n_episodes=5,
        )
        
        print(f"✓ Episodic sampler created")
        print(f"  N-way: 2")
        print(f"  Q-query: 2")
        print(f"  Episodes: 5")
        
        # Test iteration
        for episode_idx, indices in enumerate(sampler):
            print(f"✓ Episode {episode_idx}: {len(indices)} samples")
            if episode_idx >= 1:  # Test first 2 episodes
                break
        
        return True
    except Exception as e:
        print(f"✗ Episodic sampler failed: {e}")
        return False


def test_collator():
    """Test collate function."""
    print("\n" + "="*60)
    print("Testing Collate Function")
    print("="*60)
    
    try:
        dataset = RefDetDataset(
            data_root='./datasets/train/samples',
            annotations_file='./datasets/train/annotations/annotations.json',
            mode='train',
            cache_frames=False,
        )
        
        # Get a small batch
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        
        config = AugmentationConfig()
        collator = RefDetCollator(
            config=config,
            mode='train',
            stage=2,
        )
        
        batch = collator(samples)
        
        print(f"✓ Batch collated successfully")
        print(f"  Query images shape: {batch['query_images'].shape}")
        print(f"  Support images shape: {batch['support_images'].shape}")
        print(f"  Num classes: {batch['num_classes']}")
        print(f"  Targets: {len(batch['target_bboxes'])} images")
        
        return True
    except Exception as e:
        print(f"✗ Collate function failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)
    
    try:
        # Check if yolov8n.pt exists
        if not Path('yolov8n.pt').exists():
            print("⚠ yolov8n.pt not found, skipping model test")
            print("  Download with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
            return True
        
        model = YOLOv8nRefDet(
            yolo_weights='yolov8n.pt',
            nc_base=0,
        )
        
        print(f"✓ Model created successfully")
        
        # Test forward pass
        query_img = torch.randn(2, 3, 640, 640)
        support_imgs = torch.randn(6, 3, 256, 256)  # 2 classes × 3 shots
        
        model.eval()
        with torch.no_grad():
            outputs = model(
                query_image=query_img,
                support_images=support_imgs,
                mode='dual',
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Predictions: {outputs['pred_bboxes'].shape[0]} detections")
        
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test loss computation."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    try:
        loss_fn = ReferenceBasedDetectionLoss(stage=2)
        
        print(f"✓ Loss function created")
        print(f"  Stage: 2")
        print(f"  Weights: {loss_fn.weights}")
        
        # Test with dummy inputs
        pred_bboxes = torch.randn(5, 4)
        pred_cls_logits = torch.randn(5, 2)
        target_bboxes = torch.randn(5, 4)
        target_cls = torch.randn(5, 2)
        
        losses = loss_fn(
            pred_bboxes=pred_bboxes,
            pred_cls_logits=pred_cls_logits,
            target_bboxes=target_bboxes,
            target_cls=target_cls,
        )
        
        print(f"✓ Loss computed successfully")
        print(f"  Total loss: {losses['total_loss'].item():.4f}")
        print(f"  Components:")
        for key, value in losses.items():
            if key != 'total_loss':
                print(f"    {key}: {value.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("YOLOv8n-RefDet Training Pipeline Test Suite")
    print("="*60)
    
    results = {
        'Dataset Loading': test_dataset(),
        'Episodic Sampler': test_episodic_sampler(),
        'Collate Function': test_collator(),
        'Model Forward Pass': test_model(),
        'Loss Computation': test_loss(),
    }
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Pipeline is ready for training.")
    else:
        print("✗ Some tests failed. Please fix the issues before training.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
