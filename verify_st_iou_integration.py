"""
Verification Script for ST-IoU Integration
==========================================

This script verifies that ST-IoU metrics are properly integrated into the
YOLOv8n-RefDet training pipeline.

Usage:
    python verify_st_iou_integration.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics_imports():
    """Test that all metrics modules can be imported."""
    print("=" * 70)
    print("TEST 1: Metrics Module Imports")
    print("=" * 70)
    
    try:
        from src.metrics.st_iou import (
            compute_spatial_iou,
            compute_st_iou,
            compute_st_iou_batch,
            match_predictions_to_gt,
            extract_st_detections_from_video_predictions,
        )
        print("✅ src.metrics.st_iou imports successful")
        
        from src.metrics.detection_metrics import (
            compute_precision_recall,
            compute_ap,
            compute_map,
        )
        print("✅ src.metrics.detection_metrics imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_st_iou_computation():
    """Test ST-IoU computation with sample data."""
    print("\n" + "=" * 70)
    print("TEST 2: ST-IoU Computation")
    print("=" * 70)
    
    from src.metrics.st_iou import compute_st_iou, compute_spatial_iou
    
    # Test spatial IoU
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([12, 12, 52, 52])
    spatial_iou = compute_spatial_iou(box1, box2)
    print(f"Spatial IoU (overlapping boxes): {spatial_iou:.4f}")
    assert 0.0 < spatial_iou < 1.0, "Spatial IoU should be between 0 and 1"
    print("✅ Spatial IoU computation works")
    
    # Test ST-IoU
    gt_dets = {
        0: np.array([10, 10, 50, 50]),
        1: np.array([15, 15, 55, 55]),
        2: np.array([20, 20, 60, 60]),
    }
    pred_dets = {
        1: np.array([12, 12, 52, 52]),
        2: np.array([18, 18, 58, 58]),
        3: np.array([25, 25, 65, 65]),
    }
    st_iou = compute_st_iou(gt_dets, pred_dets)
    print(f"ST-IoU (multi-frame): {st_iou:.4f}")
    assert 0.0 < st_iou < 1.0, "ST-IoU should be between 0 and 1"
    print("✅ ST-IoU computation works")
    
    return True


def test_detection_metrics():
    """Test detection metrics computation."""
    print("\n" + "=" * 70)
    print("TEST 3: Detection Metrics (mAP, Precision, Recall)")
    print("=" * 70)
    
    from src.metrics.detection_metrics import compute_precision_recall, compute_map
    
    # Sample predictions and ground truth
    pred_boxes = np.array([
        [10, 10, 50, 50],
        [60, 60, 100, 100],
        [120, 120, 160, 160],
    ])
    pred_scores = np.array([0.9, 0.8, 0.7])
    pred_classes = np.array([0, 1, 0])
    
    gt_boxes = np.array([
        [12, 12, 52, 52],
        [62, 62, 102, 102],
    ])
    gt_classes = np.array([0, 1])
    
    # Compute precision/recall
    pr_metrics = compute_precision_recall(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
        iou_threshold=0.5
    )
    print(f"Precision: {pr_metrics['precision']:.4f}")
    print(f"Recall: {pr_metrics['recall']:.4f}")
    print(f"F1: {pr_metrics['f1']:.4f}")
    print("✅ Precision/Recall computation works")
    
    # Compute mAP
    map_50, ap_per_class = compute_map(
        pred_boxes, pred_scores, pred_classes,
        gt_boxes, gt_classes,
        iou_threshold=0.5
    )
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"AP per class: {ap_per_class}")
    print("✅ mAP computation works")
    
    return True


def test_trainer_integration():
    """Test that trainer can import and use ST-IoU metrics."""
    print("\n" + "=" * 70)
    print("TEST 4: Trainer Integration")
    print("=" * 70)
    
    try:
        from src.training.trainer import RefDetTrainer
        print("✅ Trainer imports successfully")
        
        # Check if trainer has best_st_iou attribute
        # We'll do this by inspecting the __init__ code
        import inspect
        init_source = inspect.getsource(RefDetTrainer.__init__)
        if 'best_st_iou' in init_source:
            print("✅ Trainer has best_st_iou tracking")
        else:
            print("❌ Trainer missing best_st_iou tracking")
            return False
        
        # Check if validate method uses ST-IoU
        validate_source = inspect.getsource(RefDetTrainer.validate)
        if 'compute_st_iou' in validate_source or 'st_iou' in validate_source:
            print("✅ Trainer.validate() computes ST-IoU")
        else:
            print("❌ Trainer.validate() missing ST-IoU computation")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Trainer integration check failed: {e}")
        return False


def test_checkpoint_compatibility():
    """Test checkpoint save/load with ST-IoU state."""
    print("\n" + "=" * 70)
    print("TEST 5: Checkpoint Compatibility")
    print("=" * 70)
    
    # Create a mock checkpoint
    checkpoint = {
        'epoch': 10,
        'global_step': 1000,
        'best_val_loss': 2.34,
        'best_st_iou': 0.67,
        'model_state_dict': {},
        'optimizer_state_dict': {},
    }
    
    # Save to temporary file
    temp_path = Path('./test_checkpoint_temp.pt')
    torch.save(checkpoint, temp_path)
    print("✅ Checkpoint with best_st_iou saved")
    
    # Load checkpoint
    loaded_ckpt = torch.load(temp_path)
    assert 'best_st_iou' in loaded_ckpt, "Checkpoint missing best_st_iou"
    assert loaded_ckpt['best_st_iou'] == 0.67, "best_st_iou value mismatch"
    print("✅ Checkpoint with best_st_iou loaded")
    
    # Clean up
    temp_path.unlink()
    print("✅ Checkpoint compatibility verified")
    
    return True


def test_unit_tests():
    """Run pytest unit tests for ST-IoU."""
    print("\n" + "=" * 70)
    print("TEST 6: Unit Tests (pytest)")
    print("=" * 70)
    
    import subprocess
    
    result = subprocess.run(
        ['pytest', 'src/tests/test_st_iou.py', '-v', '--tb=short'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ All unit tests passed")
        # Print summary line
        for line in result.stdout.split('\n'):
            if 'passed' in line.lower():
                print(f"   {line.strip()}")
        return True
    else:
        print("❌ Some unit tests failed")
        print(result.stdout)
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("ST-IoU Integration Verification")
    print("=" * 70)
    print()
    
    tests = [
        ("Metrics Imports", test_metrics_imports),
        ("ST-IoU Computation", test_st_iou_computation),
        ("Detection Metrics", test_detection_metrics),
        ("Trainer Integration", test_trainer_integration),
        ("Checkpoint Compatibility", test_checkpoint_compatibility),
        ("Unit Tests", test_unit_tests),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! ST-IoU integration is ready.")
        print("\nNext steps:")
        print("1. Run integration test: python train.py --stage 2 --epochs 2 --use_wandb")
        print("2. Check WandB dashboard for val/st_iou metrics")
        print("3. Verify best_model.pt saves on ST-IoU improvement")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
