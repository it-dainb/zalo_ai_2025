"""
Simple Training Profiler for YOLOv8n-RefDet
==========================================

Profiles actual training components with minimal overhead.
"""

import torch
import time
import argparse
from pathlib import Path

from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.datasets.refdet_dataset import RefDetDataset
from src.datasets.collate import RefDetCollator
from src.augmentations.augmentation_config import AugmentationConfig


def profile_model(model, dataset, device, num_episodes=10):
    """Profile model forward pass on actual training data."""
    
    print("\n" + "="*70)
    print("PROFILING MODEL FORWARD PASS")
    print("="*70)
    
    # Setup collator
    aug_config = AugmentationConfig()
    collator = RefDetCollator(config=aug_config, mode='train', stage=2)
    
    timings = {
        'support_encoding': [],
        'yolo_backbone': [],
        'full_forward_cached': [],
        'full_forward_uncached': [],
    }
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}:")
        
        # Get a sample
        sample = dataset[ep]
        batch = collator([sample])
        
        query_images = batch['query_images'].to(device)
        support_images = batch['support_images'].to(device)
        N, K, C, H, W = support_images.shape
        B = query_images.shape[0]
        
        print(f"  Batch: N={N} classes, K={K} support, B={B} queries")
        
        support_flat = support_images.reshape(N * K, C, H, W)
        
        with torch.no_grad():
            # Time 1: Support encoding
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            _ = model.support_encoder(support_flat)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            t_support = t1 - t0
            timings['support_encoding'].append(t_support)
            print(f"  Support encoding: {t_support*1000:.1f}ms")
            
            # Time 2: YOLOv8 backbone
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            _ = model.backbone(query_images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            t_yolo = t1 - t0
            timings['yolo_backbone'].append(t_yolo)
            print(f"  YOLOv8 backbone: {t_yolo*1000:.1f}ms")
            
            # Time 3: Full forward with cache
            # Set reference images first
            model.set_reference_images(support_flat, average_prototypes=True, n_way=N, n_support=K)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            _ = model(query_images, mode='dual', use_cache=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            t_cached = t1 - t0
            timings['full_forward_cached'].append(t_cached)
            print(f"  Full forward (cached): {t_cached*1000:.1f}ms")
            
            # Time 4: Full forward without cache
            model.clear_cache()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            _ = model(query_images, support_images=support_flat, mode='dual', use_cache=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.time()
            t_uncached = t1 - t0
            timings['full_forward_uncached'].append(t_uncached)
            print(f"  Full forward (uncached): {t_uncached*1000:.1f}ms")
            
            cache_benefit = t_uncached - t_cached
            print(f"  Cache speedup: {cache_benefit*1000:.1f}ms ({cache_benefit/t_uncached*100:.1f}%)")
    
    # Print summary
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)
    
    print(f"\nAverage timings over {num_episodes} episodes:\n")
    
    for key, label in [
        ('support_encoding', 'Support Encoding (DINOv3)'),
        ('yolo_backbone', 'YOLOv8 Backbone'),
        ('full_forward_cached', 'Full Forward (cached)'),
        ('full_forward_uncached', 'Full Forward (uncached)'),
    ]:
        times = timings[key]
        avg = sum(times) / len(times)
        print(f"  {label:30s}: {avg*1000:6.1f}ms")
    
    # Cache benefit
    avg_cached = sum(timings['full_forward_cached']) / len(timings['full_forward_cached'])
    avg_uncached = sum(timings['full_forward_uncached']) / len(timings['full_forward_uncached'])
    cache_speedup = avg_uncached - avg_cached
    speedup_pct = cache_speedup / avg_uncached * 100
    
    print(f"\n  Cache Speedup: {cache_speedup*1000:.1f}ms ({speedup_pct:.1f}% faster)")
    
    # Estimated iteration time
    # Full forward (uncached) is the worst case
    iter_time = avg_uncached
    throughput = 1.0 / iter_time
    
    print(f"\n  Estimated iteration time: {iter_time*1000:.1f}ms")
    print(f"  Throughput: {throughput:.2f} iter/sec")
    
    # Estimate epoch time
    total_samples = len(dataset)
    epoch_time_sec = iter_time * total_samples
    epoch_time_min = epoch_time_sec / 60
    
    print(f"\n  Dataset size: {total_samples} samples")
    print(f"  Estimated epoch time: {epoch_time_min:.1f} minutes")
    
    # Component breakdown
    avg_support = sum(timings['support_encoding']) / len(timings['support_encoding'])
    avg_yolo = sum(timings['yolo_backbone']) / len(timings['yolo_backbone'])
    
    # Fusion + detection = cached forward - yolo backbone
    fusion_detection = avg_cached - avg_yolo
    
    print("\n" + "="*70)
    print("COMPONENT BREAKDOWN")
    print("="*70)
    
    total = avg_uncached
    
    print(f"\n  {'Component':<30s}  {'Time':>8s}  {'% of Total':>10s}")
    print("  " + "-"*52)
    
    components = [
        ('DINOv3 Support Encoding', avg_support),
        ('YOLOv8 Backbone', avg_yolo),
        ('Fusion + Detection', fusion_detection),
    ]
    
    for name, time_val in components:
        pct = (time_val / total * 100) if total > 0 else 0
        print(f"  {name:<30s}  {time_val*1000:6.1f}ms  {pct:9.1f}%")
    
    print("  " + "-"*52)
    print(f"  {'TOTAL':<30s}  {total*1000:6.1f}ms  {100.0:9.1f}%")
    
    # Recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    dinov3_pct = avg_support / total * 100
    yolo_pct = avg_yolo / total * 100
    fusion_pct = fusion_detection / total * 100
    
    print("\n1. QUICK WINS (low risk):")
    print("   - Enable torch.compile() for 20-30% speedup")
    print("   - Increase DataLoader num_workers (currently 1)")
    print("   - Increase frame_cache_size to 1000-2000")
    
    if dinov3_pct > 30:
        print(f"\n2. DINOv3 IS THE BOTTLENECK ({dinov3_pct:.0f}%):")
        print("   Priority actions:")
        print("   - Use smaller DINOv3: vit_tiny (5M params vs 21M)")
        print("   - Use larger patches: patch32 vs patch16 (4x faster)")
        print("   - Apply INT8 quantization (2-4x faster)")
        print(f"   Expected speedup: 2-4x (epoch time: {epoch_time_min/3:.1f}-{epoch_time_min/2:.1f} min)")
    
    if yolo_pct > 20:
        print(f"\n3. YOLOv8 is {yolo_pct:.0f}% of time:")
        print("   - Already optimal for this task")
        print("   - Consider gradient checkpointing if memory-bound")
    
    if fusion_pct > 25:
        print(f"\n4. Fusion + Detection is {fusion_pct:.0f}% of time:")
        print("   - Reduce PSALM num_heads: 4 -> 2")
        print("   - Simplify detection head")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Profile YOLOv8n-RefDet')
    parser.add_argument('--data_root', type=str, default='./datasets/train/samples')
    parser.add_argument('--annotations', type=str, default='./datasets/train/annotations/annotations.json')
    parser.add_argument('--episodes', type=int, default=10)
    
    args = parser.parse_args()
    
    print("="*70)
    print("YOLOv8n-RefDet Training Profiler")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print("\nLoading model...")
    model = YOLOv8nRefDet(
        yolo_weights='yolov8n.pt',
        nc_base=80,
        freeze_yolo=True,
        freeze_dinov3=True,
    ).to(device)
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    print(f"  Total: {total_params:.2f}M params")
    print(f"  Trainable: {trainable_params:.2f}M params")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        frame_cache_size=100,
    )
    
    print(f"  Size: {len(dataset)} samples")
    
    # Profile
    profile_model(model, dataset, device, num_episodes=args.episodes)
    
    print("\nSee SPEED_OPTIMIZATION_GUIDE.md for detailed optimization options.")


if __name__ == '__main__':
    main()
