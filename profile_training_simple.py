"""
Simple Training Profiler for YOLOv8n-RefDet
==========================================

Profiles actual training components with minimal overhead.
Identifies bottlenecks by timing each model component.
"""

import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from models.yolo_refdet import YOLOv8nRefDet
from src.datasets.refdet_dataset import RefDetDataset
from src.datasets.collate import refdet_collate_fn


def time_component(fn, *args, **kwargs):
    """Time a function call and return result + elapsed time."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    return result, elapsed


def profile_forward_pass(model, batch, device):
    """Profile a complete forward pass broken down by component."""
    timings = {}
    
    query_images = batch['query_images'].to(device)
    support_images = batch['support_images'].to(device)
    N, K, C, H, W = support_images.shape
    B = query_images.shape[0]
    
    print(f"\n  Batch: N={N} classes, K={K} support, B={B} queries")
    
    # Flatten support images
    support_flat = support_images.reshape(N * K, C, H, W)
    
    # 1. Support encoding (DINOv3)
    _, t = time_component(model.support_encoder, support_flat)
    timings['dinov3_support'] = t
    print(f"  [1/5] DINOv3 Support Encoding: {t*1000:.1f}ms")
    
    # 2. Cache support features (averaging prototypes)
    _, t = time_component(
        model.set_reference_images,
        support_flat,
        average_prototypes=True,
        n_way=N,
        n_support=K
    )
    timings['prototype_averaging'] = t
    print(f"  [2/5] Prototype Averaging: {t*1000:.1f}ms")
    
    # 3. YOLOv8 backbone on query
    _, t = time_component(model.backbone, query_images)
    timings['yolov8_backbone'] = t
    print(f"  [3/5] YOLOv8 Backbone: {t*1000:.1f}ms")
    
    # 4. Full forward pass with cached support (includes fusion + detection)
    # This will use cached support features from step 2
    class_ids = batch['class_ids'].to(device) if 'class_ids' in batch else None
    
    _, t = time_component(
        model,
        query_images,
        support_images=None,  # Use cache
        mode='dual',
        use_cache=True,
        class_ids=class_ids
    )
    timings['full_forward_cached'] = t
    print(f"  [4/5] Full Forward (cached): {t*1000:.1f}ms")
    
    # 5. Full forward pass without cache (includes all: support encoding + fusion + detection)
    model.clear_cache()
    _, t = time_component(
        model,
        query_images,
        support_images=support_flat,
        mode='dual',
        use_cache=False,
        class_ids=class_ids
    )
    timings['full_forward_uncached'] = t
    print(f"  [5/5] Full Forward (uncached): {t*1000:.1f}ms")
    
    # Calculate derived metrics
    timings['fusion_plus_detection'] = timings['full_forward_cached'] - timings['yolov8_backbone']
    print(f"  [Derived] Fusion + Detection: {timings['fusion_plus_detection']*1000:.1f}ms")
    
    # Cache benefit
    cache_benefit = timings['full_forward_uncached'] - timings['full_forward_cached']
    timings['cache_speedup'] = cache_benefit
    print(f"  [Derived] Cache Speedup: {cache_benefit*1000:.1f}ms ({cache_benefit/timings['full_forward_uncached']*100:.1f}%)")
    
    return timings


def profile_training_iteration(model, batch, device, stage=2):
    """Profile a complete training iteration including loss computation."""
    from src.training.loss_utils import compute_combined_loss
    
    timings = {}
    
    # Forward pass
    print("\n[Forward Pass]")
    forward_timings = profile_forward_pass(model, batch, device)
    timings.update(forward_timings)
    
    # Loss computation
    print("\n[Loss Computation]")
    
    # Get model outputs for loss
    query_images = batch['query_images'].to(device)
    support_images = batch['support_images'].to(device)
    N, K, C, H, W = support_images.shape
    support_flat = support_images.reshape(N * K, C, H, W)
    
    model.set_reference_images(support_flat, average_prototypes=True, n_way=N, n_support=K)
    class_ids = batch['class_ids'].to(device) if 'class_ids' in batch else None
    
    with torch.no_grad():
        outputs = model(query_images, mode='dual', use_cache=True, class_ids=class_ids)
    
    # Time loss computation
    targets = {
        'boxes': [b.to(device) for b in batch['boxes']],
        'labels': [l.to(device) for l in batch['labels']],
    }
    
    _, t = time_component(
        compute_combined_loss,
        outputs,
        targets,
        stage=stage,
        device=device
    )
    timings['loss_computation'] = t
    print(f"  Loss Computation: {t*1000:.1f}ms")
    
    # Total iteration time
    total_time = forward_timings['full_forward_uncached'] + t
    timings['total_iteration'] = total_time
    print(f"\n  Total Iteration Time: {total_time*1000:.1f}ms ({1.0/total_time:.2f} iter/sec)")
    
    return timings


def run_profiling(args):
    """Run profiling on training data."""
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
        yolo_weights=args.yolo_weights,
        nc_base=80,
        freeze_yolo=args.stage >= 2,
        freeze_dinov3=True,
    ).to(device)
    
    model.eval()  # Use eval mode for consistent timing
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    print(f"  Total parameters: {total_params:.2f}M")
    print(f"  Trainable parameters: {trainable_params:.2f}M")
    print(f"  DINOv3 frozen: {model.support_encoder.freeze_backbone}")
    print(f"  YOLOv8 frozen: {model.backbone.freeze_backbone}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        frame_cache_size=100,  # Small cache for profiling
    )
    
    print(f"  Dataset size: {len(dataset)} episodes")
    print(f"  N-way: {args.n_way}")
    print(f"  K-shot: {args.n_support}")
    print(f"  Queries: {args.n_query}")
    
    # Profile multiple episodes
    print(f"\nProfiling {args.profile_episodes} episodes...")
    print("="*70)
    
    all_timings = defaultdict(list)
    
    for ep in range(args.profile_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep+1}/{args.profile_episodes}")
        print('='*70)
        
        # Get batch
        episode = dataset[ep]
        batch = refdet_collate_fn([episode])
        
        # Profile
        with torch.no_grad():
            timings = profile_training_iteration(model, batch, device, stage=args.stage)
        
        for key, val in timings.items():
            all_timings[key].append(val)
    
    # Summary statistics
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)
    
    print(f"\nAverage timings over {args.profile_episodes} episodes:\n")
    
    # Group timings
    component_order = [
        ('dinov3_support', 'DINOv3 Support Encoding'),
        ('prototype_averaging', 'Prototype Averaging'),
        ('yolov8_backbone', 'YOLOv8 Backbone'),
        ('fusion_plus_detection', 'Fusion + Detection Head'),
        ('loss_computation', 'Loss Computation'),
        ('total_iteration', 'Total Iteration'),
    ]
    
    total_avg = sum(all_timings['total_iteration']) / len(all_timings['total_iteration'])
    
    for key, label in component_order:
        if key in all_timings:
            times = all_timings[key]
            avg = sum(times) / len(times)
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"  {label:30s}: {avg*1000:6.1f}ms  ({pct:5.1f}%)")
    
    # Cache effectiveness
    if 'cache_speedup' in all_timings:
        cache_times = all_timings['cache_speedup']
        avg_speedup = sum(cache_times) / len(cache_times)
        uncached_avg = sum(all_timings['full_forward_uncached']) / len(all_timings['full_forward_uncached'])
        speedup_pct = avg_speedup / uncached_avg * 100
        print(f"\n  Cache Speedup: {avg_speedup*1000:.1f}ms ({speedup_pct:.1f}% faster)")
    
    # Estimate epoch time
    iterations_per_epoch = len(dataset)
    estimated_epoch_time = total_avg * iterations_per_epoch
    print(f"\n  Iterations per epoch: {iterations_per_epoch}")
    print(f"  Estimated epoch time: {estimated_epoch_time/60:.1f} minutes")
    print(f"  Throughput: {1.0/total_avg:.2f} iterations/sec")
    
    # Bottleneck analysis
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS")
    print("="*70)
    
    bottlenecks = []
    for key, label in component_order[:-1]:  # Exclude total
        if key in all_timings:
            times = all_timings[key]
            avg = sum(times) / len(times)
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            bottlenecks.append((pct, label, avg))
    
    bottlenecks.sort(reverse=True)
    
    print("\nTop bottlenecks:")
    for i, (pct, label, time_s) in enumerate(bottlenecks[:3], 1):
        print(f"  {i}. {label}: {pct:.1f}% ({time_s*1000:.1f}ms)")
    
    # Recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    dinov3_pct = sum(all_timings['dinov3_support']) / len(all_timings['dinov3_support']) / total_avg * 100
    yolo_pct = sum(all_timings['yolov8_backbone']) / len(all_timings['yolov8_backbone']) / total_avg * 100
    fusion_pct = sum(all_timings['fusion_plus_detection']) / len(all_timings['fusion_plus_detection']) / total_avg * 100
    loss_pct = sum(all_timings['loss_computation']) / len(all_timings['loss_computation']) / total_avg * 100
    
    print("\n1. Quick Wins (low risk, immediate gains):")
    print("   - Enable torch.compile() for 20-30% speedup")
    print("   - Increase num_workers if data loading is slow")
    print("   - Increase frame cache size to 1000-2000 frames")
    
    if dinov3_pct > 30:
        print(f"\n2. DINOv3 is {dinov3_pct:.0f}% of iteration time:")
        print("   - Consider smaller DINOv3 model (vit_tiny)")
        print("   - Try larger patches (patch32 vs patch16)")
        print("   - Apply INT8 quantization to frozen DINOv3")
    
    if yolo_pct > 20:
        print(f"\n3. YOLOv8 is {yolo_pct:.0f}% of iteration time:")
        print("   - Already optimal for this architecture")
        print("   - Consider enabling gradient checkpointing")
    
    if fusion_pct > 25:
        print(f"\n4. Fusion + Detection is {fusion_pct:.0f}% of iteration time:")
        print("   - Reduce num_heads in PSALM fusion (4 -> 2)")
        print("   - Simplify detection head architecture")
    
    if loss_pct > 20:
        print(f"\n5. Loss computation is {loss_pct:.0f}% of iteration time:")
        print("   - Reduce number of loss components")
        print("   - Optimize anchor assignment")
        print("   - Reduce DFL bins (16 -> 8)")
    
    print("\n" + "="*70)
    print("Profiling complete! See SPEED_OPTIMIZATION_GUIDE.md for details.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Profile YOLOv8n-RefDet training')
    parser.add_argument('--data_root', type=str, default='./datasets/train/samples',
                        help='Path to training data')
    parser.add_argument('--annotations', type=str, default='./datasets/train/annotations/annotations.json',
                        help='Path to annotations')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 weights')
    parser.add_argument('--stage', type=int, default=2,
                        help='Training stage (1/2/3)')
    parser.add_argument('--n_way', type=int, default=2,
                        help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=2,
                        help='Number of support images per class')
    parser.add_argument('--n_query', type=int, default=4,
                        help='Number of query images')
    parser.add_argument('--profile_episodes', type=int, default=10,
                        help='Number of episodes to profile')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.data_root).exists():
        print(f"ERROR: Data root not found: {args.data_root}")
        return
    
    if not Path(args.annotations).exists():
        print(f"ERROR: Annotations not found: {args.annotations}")
        return
    
    run_profiling(args)


if __name__ == '__main__':
    main()
