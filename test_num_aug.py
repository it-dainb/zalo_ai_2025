"""
Quick test to verify num_aug parameter works correctly.
"""
import sys
sys.path.insert(0, '/mnt/data/HACKATHON/zalo_ai_2025')

from src.datasets.refdet_dataset import RefDetDataset
from pathlib import Path

def test_num_aug():
    """Test that num_aug multiplies dataset size correctly."""
    
    # Check if dataset exists
    data_root = './datasets/train/samples'
    annotations = './datasets/train/annotations/annotations.json'
    
    if not Path(data_root).exists():
        print(f"⚠️  Dataset not found at {data_root}")
        print("This test requires a valid dataset.")
        return
    
    print("="*60)
    print("Testing num_aug Parameter")
    print("="*60)
    
    # Test with num_aug=1 (baseline)
    print("\n1. Testing with num_aug=1 (baseline)...")
    dataset1 = RefDetDataset(
        data_root=data_root,
        annotations_file=annotations,
        mode='train',
        cache_frames=True,
        num_aug=1,
    )
    base_len = len(dataset1)
    print(f"   ✓ Dataset length with num_aug=1: {base_len}")
    
    # Test with num_aug=4
    print("\n2. Testing with num_aug=4...")
    dataset4 = RefDetDataset(
        data_root=data_root,
        annotations_file=annotations,
        mode='train',
        cache_frames=True,
        num_aug=4,
    )
    aug_len = len(dataset4)
    print(f"   ✓ Dataset length with num_aug=4: {aug_len}")
    
    # Verify multiplication
    expected_len = base_len * 4
    assert aug_len == expected_len, f"Expected {expected_len}, got {aug_len}"
    print(f"   ✓ Length correctly multiplied: {base_len} × 4 = {aug_len}")
    
    # Test sampling
    print("\n3. Testing sample access...")
    sample = dataset4[0]
    print(f"   ✓ Sample keys: {list(sample.keys())}")
    print(f"   ✓ Aug index in sample: {sample.get('aug_idx', 'NOT FOUND')}")
    
    # Test different aug_idx values
    print("\n4. Testing aug_idx values...")
    for i in range(min(4, base_len)):
        idx = i * base_len  # First sample of each aug group
        sample = dataset4[idx]
        aug_idx = sample.get('aug_idx', -1)
        frame_idx = sample.get('frame_idx', -1)
        print(f"   Sample {idx}: aug_idx={aug_idx}, frame_idx={frame_idx}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nUsage in training:")
    print("  python train.py --num_aug 4 [other args...]")
    print("\nSee docs/NUM_AUG_GUIDE.md for more information.")

if __name__ == '__main__':
    try:
        test_num_aug()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
