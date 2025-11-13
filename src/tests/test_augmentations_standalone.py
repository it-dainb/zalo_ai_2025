"""
Quick test script to verify augmentation implementations.
Tests all standalone classes without external dependencies.
"""

import numpy as np
import torch
import sys
sys.path.append('/mnt/data/HACKATHON/zalo_ai_2025')

from src.augmentations.query_augmentation import (
    LetterBox, MosaicAugmentation, MixUpAugmentation, 
    CopyPasteAugmentation, QueryAugmentation, bbox_ioa
)
from src.augmentations.support_augmentation import (
    LetterBoxSupport, SupportAugmentation
)
from src.augmentations.temporal_augmentation import (
    LetterBoxTemporal, TemporalConsistentAugmentation
)

print("=" * 80)
print("Testing Augmentation Implementations")
print("=" * 80)

# Create dummy data
def create_dummy_image(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def create_dummy_bboxes(n=5):
    bboxes = np.random.rand(n, 4) * 400 + 20
    bboxes[:, 2:] = bboxes[:, :2] + np.random.rand(n, 2) * 100 + 50
    return bboxes.astype(np.float32)

def create_dummy_labels(n=5):
    return np.random.randint(0, 10, n, dtype=np.int64)


# Test 1: LetterBox for Query
print("\n1. Testing LetterBox (Query - 640x640)...")
try:
    letterbox = LetterBox(new_shape=(640, 640))
    img = create_dummy_image()
    bboxes = create_dummy_bboxes()
    labels = create_dummy_labels()
    
    result = letterbox(img, bboxes, labels)
    assert result['image'].shape == (640, 640, 3), f"Expected (640, 640, 3), got {result['image'].shape}"
    assert result['bboxes'].shape[1] == 4, "Bboxes should have 4 coordinates"
    print(f"   ✓ Input: {img.shape} → Output: {result['image'].shape}")
    print(f"   ✓ Bboxes: {bboxes.shape} → {result['bboxes'].shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 2: MosaicAugmentation (4 images)
print("\n2. Testing Mosaic Augmentation (2x2)...")
try:
    mosaic = MosaicAugmentation(img_size=640, n=4)
    images = [create_dummy_image() for _ in range(4)]
    bboxes_list = [create_dummy_bboxes(3) for _ in range(4)]
    labels_list = [create_dummy_labels(3) for _ in range(4)]
    
    result_img, result_bboxes, result_labels = mosaic(images, bboxes_list, labels_list)
    assert result_img.shape == (640, 640, 3), f"Expected (640, 640, 3), got {result_img.shape}"
    assert result_bboxes.shape[1] == 4, "Bboxes should have 4 coordinates"
    print(f"   ✓ Mosaic: 4 images → {result_img.shape}")
    print(f"   ✓ Combined bboxes: {result_bboxes.shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 3: MixUpAugmentation
print("\n3. Testing MixUp Augmentation...")
try:
    mixup = MixUpAugmentation(alpha=32.0, prob=1.0)  # Force apply
    img1 = create_dummy_image()
    img2 = create_dummy_image()
    bboxes1 = create_dummy_bboxes(3)
    bboxes2 = create_dummy_bboxes(2)
    labels1 = create_dummy_labels(3)
    labels2 = create_dummy_labels(2)
    
    mixed_img, mixed_bboxes, mixed_labels = mixup(img1, bboxes1, labels1, img2, bboxes2, labels2)
    assert mixed_img.shape == img1.shape, "Mixed image should have same shape"
    assert len(mixed_bboxes) == len(bboxes1) + len(bboxes2), "Should concatenate all bboxes"
    print(f"   ✓ Mixed: {img1.shape} + {img2.shape} → {mixed_img.shape}")
    print(f"   ✓ Combined: {len(bboxes1)} + {len(bboxes2)} = {len(mixed_bboxes)} bboxes")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 4: bbox_ioa helper
print("\n4. Testing bbox_ioa helper...")
try:
    box1 = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
    box2 = np.array([[20, 20, 40, 40], [70, 70, 90, 90]])
    ioa = bbox_ioa(box1, box2)
    assert ioa.shape == (2, 2), f"Expected (2, 2), got {ioa.shape}"
    assert np.all(ioa >= 0) and np.all(ioa <= 1), "IoA should be in [0, 1]"
    print(f"   ✓ IoA shape: {ioa.shape}")
    print(f"   ✓ IoA range: [{ioa.min():.3f}, {ioa.max():.3f}]")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 5: QueryAugmentation (full pipeline)
print("\n5. Testing QueryAugmentation (Stage 1)...")
try:
    aug = QueryAugmentation(img_size=640, stage="stage1")
    img = create_dummy_image()
    bboxes = create_dummy_bboxes()
    labels = create_dummy_labels()
    
    result = aug(img, bboxes, labels, apply_mosaic=False)
    assert isinstance(result['image'], torch.Tensor), "Should return tensor"
    assert result['image'].shape == (3, 640, 640), f"Expected (3, 640, 640), got {result['image'].shape}"
    print(f"   ✓ Input: {img.shape} → Output: {result['image'].shape}")
    print(f"   ✓ Output type: {type(result['image'])}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 6: LetterBoxSupport (DINOv3)
print("\n6. Testing LetterBoxSupport (256x256)...")
try:
    letterbox = LetterBoxSupport(new_shape=256)
    img = create_dummy_image()
    
    result = letterbox(img)
    assert result.shape == (256, 256, 3), f"Expected (256, 256, 3), got {result.shape}"
    print(f"   ✓ Input: {img.shape} → Output: {result.shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 7: SupportAugmentation (weak mode)
print("\n7. Testing SupportAugmentation (Weak)...")
try:
    aug = SupportAugmentation(img_size=256, mode="weak")
    img = create_dummy_image()
    
    result = aug(img)
    assert isinstance(result, torch.Tensor), "Should return tensor"
    assert result.shape == (3, 256, 256), f"Expected (3, 256, 256), got {result.shape}"
    print(f"   ✓ Input: {img.shape} → Output: {result.shape}")
    print(f"   ✓ Normalized: min={result.min():.3f}, max={result.max():.3f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 8: LetterBoxTemporal
print("\n8. Testing LetterBoxTemporal (640x640)...")
try:
    letterbox = LetterBoxTemporal(new_shape=640)
    img = create_dummy_image()
    bboxes = create_dummy_bboxes()
    
    result = letterbox(img, bboxes)
    assert result['image'].shape == (640, 640, 3), f"Expected (640, 640, 3), got {result['image'].shape}"
    print(f"   ✓ Input: {img.shape} → Output: {result['image'].shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 9: TemporalConsistentAugmentation
print("\n9. Testing TemporalConsistentAugmentation...")
try:
    aug = TemporalConsistentAugmentation(img_size=640, consistency_window=4)
    
    # Process 8 frames
    results = []
    for i in range(8):
        img = create_dummy_image()
        bboxes = create_dummy_bboxes(2)
        labels = create_dummy_labels(2)
        result = aug(img, bboxes, labels)
        results.append(result)
    
    assert all(isinstance(r['image'], torch.Tensor) for r in results), "All should be tensors"
    print(f"   ✓ Processed 8 frames with consistency_window=4")
    print(f"   ✓ Output shape: {results[0]['image'].shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


# Test 10: Stage transitions
print("\n10. Testing Stage Transitions...")
try:
    for stage in ["stage1", "stage2", "stage3"]:
        aug = QueryAugmentation(img_size=640, stage=stage)
        img = create_dummy_image()
        bboxes = create_dummy_bboxes()
        labels = create_dummy_labels()
        result = aug(img, bboxes, labels, apply_mosaic=False)
        assert result['image'].shape == (3, 640, 640)
        print(f"   ✓ {stage}: {result['image'].shape}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")


print("\n" + "=" * 80)
print("All Tests Completed!")
print("=" * 80)
