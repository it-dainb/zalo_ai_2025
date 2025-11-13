"""
Test script for caching system performance
"""

import time
import argparse
from pathlib import Path
from src.datasets.refdet_dataset import RefDetDataset

def test_caching_performance():
    """Test caching system performance improvements."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/train/samples')
    parser.add_argument('--annotations', type=str, default='./datasets/train/annotations/annotations.json')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to test')
    args = parser.parse_args()
    
    print("="*70)
    print("CACHING SYSTEM PERFORMANCE TEST")
    print("="*70)
    
    # Test 1: With caching
    print("\n" + "="*70)
    print("TEST 1: With Caching (default settings)")
    print("="*70)
    
    dataset_cached = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        cache_frames=True,
        frame_cache_size=500,
        support_cache_size_mb=200,
    )
    
    start_time = time.time()
    for i in range(args.iterations):
        idx = i % len(dataset_cached)
        sample = dataset_cached[idx]
        if i % 20 == 0:
            print(f"  Iteration {i}/{args.iterations}...")
    
    cached_time = time.time() - start_time
    print(f"\n✓ Completed {args.iterations} iterations in {cached_time:.2f}s")
    print(f"  Average time per sample: {cached_time/args.iterations*1000:.2f}ms")
    
    dataset_cached.print_cache_stats()
    
    # Test 2: Without caching
    print("\n" + "="*70)
    print("TEST 2: Without Caching")
    print("="*70)
    
    dataset_uncached = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        cache_frames=False,
        frame_cache_size=0,
        support_cache_size_mb=1,
    )
    
    start_time = time.time()
    for i in range(args.iterations):
        idx = i % len(dataset_uncached)
        sample = dataset_uncached[idx]
        if i % 20 == 0:
            print(f"  Iteration {i}/{args.iterations}...")
    
    uncached_time = time.time() - start_time
    print(f"\n✓ Completed {args.iterations} iterations in {uncached_time:.2f}s")
    print(f"  Average time per sample: {uncached_time/args.iterations*1000:.2f}ms")
    
    # Comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\nWith Caching:    {cached_time:.2f}s ({cached_time/args.iterations*1000:.2f}ms per sample)")
    print(f"Without Caching: {uncached_time:.2f}s ({uncached_time/args.iterations*1000:.2f}ms per sample)")
    
    speedup = uncached_time / cached_time
    time_saved = uncached_time - cached_time
    
    print(f"\n🚀 Speedup: {speedup:.2f}x faster")
    print(f"⏱️  Time saved: {time_saved:.2f}s ({time_saved/uncached_time*100:.1f}% faster)")
    
    # Estimate time saved over full training
    print("\n" + "="*70)
    print("ESTIMATED TRAINING TIME SAVINGS")
    print("="*70)
    
    epochs = 100
    samples_per_epoch = len(dataset_cached)
    
    cached_epoch_time = (cached_time / args.iterations) * samples_per_epoch
    uncached_epoch_time = (uncached_time / args.iterations) * samples_per_epoch
    
    cached_total_time = cached_epoch_time * epochs / 3600  # hours
    uncached_total_time = uncached_epoch_time * epochs / 3600  # hours
    time_saved_hours = uncached_total_time - cached_total_time
    
    print(f"\nFor {epochs} epochs with {samples_per_epoch} samples/epoch:")
    print(f"  With Caching:    ~{cached_total_time:.1f} hours")
    print(f"  Without Caching: ~{uncached_total_time:.1f} hours")
    print(f"  Time Saved:      ~{time_saved_hours:.1f} hours ({time_saved_hours*60:.0f} minutes)")
    
    print("\n" + "="*70)
    print("Note: Actual training time includes model forward/backward passes,")
    print("      so data loading speedup impact will be proportionally smaller.")
    print("      However, caching is especially valuable for I/O bound systems.")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_caching_performance()
