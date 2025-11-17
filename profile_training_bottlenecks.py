"""
Comprehensive Training Profiler for YOLOv8n-RefDet
===================================================

Profiles all components to identify performance bottlenecks:
1. DINOv3 encoder forward pass (support + query encoding)
2. YOLOv8 backbone forward pass
3. PSALM fusion module
4. Detection heads (standard + prototype)
5. Loss computation (all components)
6. Data loading time
7. Support feature caching effectiveness

Usage:
    python profile_training_bottlenecks.py --data_root ./datasets/train/samples \
        --annotations ./datasets/train/annotations/annotations.json \
        --profile_epochs 2 --batch_size 4

Outputs:
    - Detailed timing breakdown per component
    - Cache hit rates
    - Memory usage statistics
    - Recommendations for optimization
"""

import argparse
import torch
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from models.yolo_refdet import YOLORefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.loss_utils import prepare_loss_inputs
from src.augmentations import get_stage_config


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings
        self.start = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.start is not None:
            elapsed = time.perf_counter() - self.start
            self.timings[self.name].append(elapsed)


def profile_model_components(model, batch, device):
    """Profile individual model components separately."""
    timings = defaultdict(list)
    
    query_images = batch['query_images'].to(device)
    support_images = batch['support_images'].to(device)
    B, C, H, W = query_images.shape
    N = batch['n_way']
    K = batch['n_support']
    
    # 1. Profile DINOv3 support encoder (K times for K support images)
    # Flatten N*K support images
    support_flat = support_images.view(N * K, C, 256, 256)
    
    with TimingContext('dinov3_support_encoding', timings):
        support_features = model.support_encoder(support_flat)
    
    # 2. Pre-compute support prototypes (average K samples per class)
    # This simulates what happens in set_reference_images()
    with TimingContext('support_prototype_averaging', timings):
        model.set_reference_images(support_flat, average_prototypes=True, n_way=N, n_support=K)
    
    # 3. Profile YOLOv8 backbone on query images
    with TimingContext('yolov8_backbone', timings):
        query_features_yolo = model.backbone(query_images)
    
    # 4. Profile PSALM fusion (scs_fusion is the actual attribute name)
    # Note: Support features are already cached from step 2
    with TimingContext('psalm_fusion', timings):
        fused_features = model.scs_fusion(
            query_features_yolo,
            model._cached_support_features,
        )
    
    # 5. Profile detection head (dual head handles both standard and prototype)
    # Prepare prototypes for detection head
    prototypes = {
        'p2': model._cached_support_features['p2'],
        'p3': model._cached_support_features['p3'],
        'p4': model._cached_support_features['p4'],
        'p5': model._cached_support_features['p5'],
    }
    
    # Clamp fused features (as done in actual model)
    fused_features_clamped = {
        scale: torch.clamp(feat, min=-10.0, max=10.0)
        for scale, feat in fused_features.items()
    }
    
    with TimingContext('detection_head_dual', timings):
        detections = model.detection_head(fused_features_clamped, prototypes, mode='dual')
    
    return timings


def profile_loss_components(loss_fn, model_outputs, batch):
    """Profile individual loss components."""
    timings = defaultdict(list)
    
    # Prepare loss inputs
    with TimingContext('prepare_loss_inputs', timings):
        loss_inputs = prepare_loss_inputs(
            model_outputs=model_outputs,
            batch=batch,
            stage=loss_fn.stage,
        )
        # Remove diagnostic_data before loss computation
        diagnostic_data = loss_inputs.pop('diagnostic_data', None)
    
    # Profile each loss component
    with TimingContext('bbox_loss', timings):
        if 'pred_bboxes' in loss_inputs and 'gt_bboxes' in loss_inputs:
            _ = loss_fn.bbox_loss(
                pred_bboxes=loss_inputs['pred_bboxes'],
                gt_bboxes=loss_inputs['gt_bboxes'],
                valid_mask=loss_inputs.get('valid_mask')
            )
    
    with TimingContext('cls_loss', timings):
        if 'pred_scores' in loss_inputs and 'gt_class_ids' in loss_inputs:
            _ = loss_fn.cls_loss(
                pred_scores=loss_inputs['pred_scores'],
                gt_class_ids=loss_inputs['gt_class_ids'],
                valid_mask=loss_inputs.get('valid_mask')
            )
    
    if loss_fn.weights['supcon'] > 0:
        with TimingContext('supcon_loss', timings):
            if 'query_embeddings' in loss_inputs and 'support_prototypes' in loss_inputs:
                _ = loss_fn.prototype_loss(
                    query_embeddings=loss_inputs['query_embeddings'],
                    support_prototypes=loss_inputs['support_prototypes'],
                    gt_class_ids=loss_inputs['gt_class_ids'],
                )
    
    if loss_fn.weights['cpe'] > 0:
        with TimingContext('cpe_loss', timings):
            if 'query_embeddings' in loss_inputs and 'support_prototypes' in loss_inputs:
                _ = loss_fn.cpe_loss(
                    query_embeddings=loss_inputs['query_embeddings'],
                    support_prototypes=loss_inputs['support_prototypes'],
                    gt_class_ids=loss_inputs['gt_class_ids'],
                )
    
    # Total loss computation
    with TimingContext('total_loss', timings):
        losses = loss_fn(**loss_inputs)
    
    return timings


def run_profiling(args):
    """Run comprehensive profiling."""
    
    print("=" * 80)
    print("YOLOv8n-RefDet Training Profiler")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Dataset setup
    print("\n" + "-" * 80)
    print("Setting up dataset...")
    print("-" * 80)
    
    dataset = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        cache_frames=True,
        frame_cache_size=args.frame_cache_size,
        support_cache_size_mb=args.support_cache_size_mb,
        num_aug=args.num_aug,
    )
    
    sampler = EpisodicBatchSampler(
        dataset=dataset,
        n_way=args.n_way,
        n_query=args.n_query,
        n_episodes=args.profile_episodes,
    )
    
    aug_config = get_stage_config(stage=args.stage)
    collator = RefDetCollator(config=aug_config, mode='train', stage=args.stage)
    
    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ Episodes: {args.profile_episodes}")
    print(f"✓ N-way: {args.n_way}, N-query: {args.n_query}")
    
    # Model setup
    print("\n" + "-" * 80)
    print("Setting up model...")
    print("-" * 80)
    
    model = YOLORefDet().to(device)
    
    model.eval()  # Eval mode for consistent timing
    
    print(f"✓ Model loaded")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  DINOv3 frozen: {model.support_encoder.freeze_backbone}")
    
    # Loss setup
    loss_fn = ReferenceBasedDetectionLoss(
        stage=args.stage,
        bbox_weight=7.5,
        cls_weight=0.5,
        supcon_weight=1.0 if args.stage >= 2 else 0.0,
        cpe_weight=0.5 if args.stage >= 2 else 0.0,
    ).to(device)
    
    # Profiling
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    
    all_timings = defaultdict(list)
    cache_stats_list = []
    data_load_times = []
    
    with torch.no_grad():
        sampler_iter = iter(sampler)
        for episode_idx in range(args.profile_episodes):
            # Profile data loading
            data_start = time.time()
            indices = next(sampler_iter)
            samples = [dataset[i] for i in indices]
            batch = collator(samples)
            
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], list) and len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch[key]]
            
            data_load_times.append(time.time() - data_start)
            
            # Profile model components
            model_timings = profile_model_components(model, batch, device)
            for key, times in model_timings.items():
                all_timings[key].extend(times)
            
            # Full forward pass for loss profiling
            support_images = batch['support_images']
            N, K, C, H, W = support_images.shape
            support_flat = support_images.reshape(N * K, C, H, W)
            
            model.set_reference_images(support_flat, average_prototypes=True, n_way=N, n_support=K)
            model_outputs = model(
                query_image=batch['query_images'],
                mode='dual',
                use_cache=True,
                class_ids=batch.get('class_ids', None),
            )
            
            # Profile loss components
            loss_timings = profile_loss_components(loss_fn, model_outputs, batch)
            for key, times in loss_timings.items():
                all_timings[key].extend(times)
            
            # Collect cache stats
            cache_stats = dataset.get_cache_stats()
            cache_stats_list.append(cache_stats)
            
            print(f"\rEpisode {episode_idx + 1}/{args.profile_episodes}...", end='', flush=True)
    
    print("\n")
    
    # Print results
    print_profiling_results(all_timings, data_load_times, cache_stats_list)
    
    # Save results
    save_profiling_results(all_timings, data_load_times, cache_stats_list, args)


def print_profiling_results(timings, data_load_times, cache_stats_list):
    """Print formatted profiling results."""
    
    print("\n" + "-" * 80)
    print("TIMING BREAKDOWN (per episode)")
    print("-" * 80)
    
    # Sort by mean time
    sorted_timings = sorted(
        [(k, np.mean(v), np.std(v)) for k, v in timings.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    total_time = sum(np.mean(v) for v in timings.values())
    
    print(f"{'Component':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% Total':<10}")
    print("-" * 80)
    
    for name, mean_time, std_time in sorted_timings:
        pct = (mean_time / total_time * 100) if total_time > 0 else 0
        print(f"{name:<30} {mean_time*1000:>10.2f}   {std_time*1000:>10.2f}   {pct:>8.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL MODEL TIME':<30} {total_time*1000:>10.2f} ms")
    print(f"{'DATA LOADING TIME':<30} {np.mean(data_load_times)*1000:>10.2f} ms")
    print(f"{'TOTAL ITERATION TIME':<30} {(total_time + np.mean(data_load_times))*1000:>10.2f} ms")
    
    # Cache statistics
    print("\n" + "-" * 80)
    print("CACHE STATISTICS")
    print("-" * 80)
    
    if cache_stats_list:
        final_stats = cache_stats_list[-1]
        
        print(f"\nSupport Image Cache:")
        print(f"  Hit rate: {final_stats['support_cache']['hit_rate']*100:.1f}%")
        print(f"  Hits: {final_stats['support_cache']['hits']}")
        print(f"  Misses: {final_stats['support_cache']['misses']}")
        print(f"  Size: {final_stats['support_cache']['size']} / {final_stats['support_cache']['capacity']}")
        
        print(f"\nFrame Cache:")
        print(f"  Hit rate: {final_stats['frame_cache']['hit_rate']*100:.1f}%")
        print(f"  Hits: {final_stats['frame_cache']['hits']}")
        print(f"  Misses: {final_stats['frame_cache']['misses']}")
        print(f"  Size: {final_stats['frame_cache']['size']} / {final_stats['frame_cache']['capacity']}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # DINOv3 bottleneck
    dino_total = sum(np.mean(timings.get(k, [0])) for k in ['dino_support_encoding', 'dino_query_encoding'])
    if dino_total / total_time > 0.3:
        recommendations.append(
            f"⚠ DINOv3 encoding takes {dino_total/total_time*100:.1f}% of model time.\n"
            "  Consider: Verify support feature caching is working (support encoding should run once per episode)"
        )
    
    # Fusion bottleneck
    fusion_time = np.mean(timings.get('psalm_fusion', [0]))
    if fusion_time / total_time > 0.2:
        recommendations.append(
            f"⚠ PSALM fusion takes {fusion_time/total_time*100:.1f}% of model time.\n"
            "  Consider: Use CHEAF fusion or reduce number of attention heads"
        )
    
    # Loss computation bottleneck
    loss_total = sum(np.mean(timings.get(k, [0])) for k in ['bbox_loss', 'cls_loss', 'supcon_loss', 'cpe_loss', 'total_loss'])
    if loss_total / total_time > 0.25:
        recommendations.append(
            f"⚠ Loss computation takes {loss_total/total_time*100:.1f}% of model time.\n"
            "  Consider: Reduce contrastive loss weight or simplify loss components"
        )
    
    # Data loading bottleneck
    data_pct = np.mean(data_load_times) / (total_time + np.mean(data_load_times))
    if data_pct > 0.2:
        recommendations.append(
            f"⚠ Data loading takes {data_pct*100:.1f}% of iteration time.\n"
            "  Consider: Increase num_workers or increase cache sizes"
        )
    
    # Cache effectiveness
    if cache_stats_list:
        final_stats = cache_stats_list[-1]
        if final_stats['support_cache']['hit_rate'] < 0.5:
            recommendations.append(
                f"⚠ Support cache hit rate is low ({final_stats['support_cache']['hit_rate']*100:.1f}%).\n"
                "  Consider: Increase support_cache_size_mb"
            )
        if final_stats['frame_cache']['hit_rate'] < 0.3:
            recommendations.append(
                f"⚠ Frame cache hit rate is low ({final_stats['frame_cache']['hit_rate']*100:.1f}%).\n"
                "  Consider: Increase frame_cache_size"
            )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✓ No major bottlenecks detected!")
    
    print()


def save_profiling_results(timings, data_load_times, cache_stats_list, args):
    """Save profiling results to JSON file."""
    results = {
        'config': {
            'stage': args.stage,
            'n_way': args.n_way,
            'n_query': args.n_query,
            'profile_episodes': args.profile_episodes,
            'frame_cache_size': args.frame_cache_size,
            'support_cache_size_mb': args.support_cache_size_mb,
        },
        'timings': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in timings.items()},
        'data_load_time': {'mean': float(np.mean(data_load_times)), 'std': float(np.std(data_load_times))},
        'cache_stats': cache_stats_list[-1] if cache_stats_list else {},
    }
    
    output_file = Path('./profiling_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Profile YOLOv8n-RefDet training')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./datasets/train/samples',
                        help='Root directory containing training samples')
    parser.add_argument('--annotations', type=str, default='./datasets/train/annotations/annotations.json',
                        help='Path to annotations.json')
    
    # Profiling arguments
    parser.add_argument('--profile_episodes', type=int, default=10,
                        help='Number of episodes to profile')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2, 3],
                        help='Training stage')
    parser.add_argument('--n_way', type=int, default=2,
                        help='Number of classes per episode')
    parser.add_argument('--n_query', type=int, default=4,
                        help='Number of query samples per class')
    
    # Cache arguments
    parser.add_argument('--frame_cache_size', type=int, default=500,
                        help='Number of video frames to cache')
    parser.add_argument('--support_cache_size_mb', type=int, default=200,
                        help='Support image cache size in MB')
    parser.add_argument('--num_aug', type=int, default=1,
                        help='Number of augmented versions per image')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_profiling(args)
