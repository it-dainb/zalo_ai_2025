"""
Example script demonstrating triplet dataset usage for training.

This script shows how to integrate triplet training into the existing pipeline.
"""

import torch
from torch.utils.data import DataLoader

from src.datasets.refdet_dataset import RefDetDataset
from src.datasets.triplet_dataset import TripletDataset, TripletBatchSampler, MixedDataset
from src.datasets.collate import TripletCollator, RefDetCollator, MixedCollator
from src.augmentations.augmentation_config import AugmentationConfig


def example_triplet_only_training():
    """Example: Pure triplet training (for contrastive learning)."""
    
    print("="*70)
    print("Example 1: Pure Triplet Training")
    print("="*70)
    
    # 1. Create base dataset
    base_dataset = RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=True
    )
    
    # 2. Create triplet dataset
    triplet_dataset = TripletDataset(
        base_dataset=base_dataset,
        negative_strategy='mixed',  # Use both background and cross-class
        samples_per_class=100,      # 100 triplets per class per epoch
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total classes: {len(base_dataset.classes)}")
    print(f"  Triplet samples per epoch: {len(triplet_dataset)}")
    
    # 3. Create batch sampler
    sampler = TripletBatchSampler(
        dataset=triplet_dataset,
        batch_size=16,
        n_batches=100,
        balance_classes=True
    )
    
    # 4. Create collator
    config = AugmentationConfig()
    collator = TripletCollator(
        config=config,
        mode='train',
        apply_strong_aug=True  # Strong augmentation for contrastive learning
    )
    
    # 5. Create DataLoader
    dataloader = DataLoader(
        triplet_dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"\n⚙️  DataLoader Configuration:")
    print(f"  Batch size: 16")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Total samples per epoch: ~{len(dataloader) * 16}")
    
    # 6. Iterate (example)
    print(f"\n🔄 Iterating through batches...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Anchor images: {batch['anchor_images'].shape}")
        print(f"  Positive images: {batch['positive_images'].shape}")
        print(f"  Negative images: {batch['negative_images'].shape}")
        print(f"  Negative types: {batch['negative_types']}")
        
        # Count negative types
        bg_count = batch['negative_types'].count('background')
        cc_count = batch['negative_types'].count('cross_class')
        print(f"  └─ Background: {bg_count}, Cross-class: {cc_count}")
        
        if i >= 2:  # Show 3 batches
            break
    
    print(f"\n✅ Pure triplet training setup complete!")


def example_mixed_training():
    """Example: Mixed training (detection + triplet)."""
    
    print("\n" + "="*70)
    print("Example 2: Mixed Training (Detection + Triplet)")
    print("="*70)
    
    # 1. Create datasets
    base_dataset = RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=True
    )
    
    triplet_dataset = TripletDataset(
        base_dataset=base_dataset,
        negative_strategy='mixed',
        samples_per_class=100,
    )
    
    # 2. Create mixed dataset (70% detection, 30% triplet)
    mixed_dataset = MixedDataset(
        detection_dataset=base_dataset,
        triplet_dataset=triplet_dataset,
        detection_ratio=0.7
    )
    
    print(f"\n📊 Mixed Dataset Statistics:")
    print(f"  Total samples: {len(mixed_dataset)}")
    print(f"  Detection samples: {mixed_dataset.n_detection} (70%)")
    print(f"  Triplet samples: {mixed_dataset.n_triplet} (30%)")
    
    # 3. Create collators
    config = AugmentationConfig()
    
    detection_collator = RefDetCollator(
        config=config,
        mode='train',
        stage=2
    )
    
    triplet_collator = TripletCollator(
        config=config,
        mode='train',
        apply_strong_aug=True
    )
    
    mixed_collator = MixedCollator(
        detection_collator=detection_collator,
        triplet_collator=triplet_collator
    )
    
    # 4. Create DataLoader
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=mixed_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"\n⚙️  DataLoader Configuration:")
    print(f"  Batch size: 16")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # 5. Iterate (example)
    print(f"\n🔄 Iterating through mixed batches...")
    for i, batch in enumerate(dataloader):
        batch_type = batch['batch_type']
        n_det = batch['n_detection']
        n_trip = batch['n_triplet']
        
        print(f"\nBatch {i+1} - Type: {batch_type}")
        print(f"  Detection samples: {n_det}")
        print(f"  Triplet samples: {n_trip}")
        
        if 'detection' in batch:
            det_batch = batch['detection']
            print(f"  Detection query: {det_batch['query_images'].shape}")
        
        if 'triplet' in batch:
            trip_batch = batch['triplet']
            print(f"  Triplet anchors: {trip_batch['anchor_images'].shape}")
        
        if i >= 2:  # Show 3 batches
            break
    
    print(f"\n✅ Mixed training setup complete!")


def example_negative_sampling_strategies():
    """Example: Different negative sampling strategies."""
    
    print("\n" + "="*70)
    print("Example 3: Comparing Negative Sampling Strategies")
    print("="*70)
    
    base_dataset = RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=True
    )
    
    strategies = ['background', 'cross_class', 'mixed']
    
    for strategy in strategies:
        print(f"\n📌 Strategy: {strategy}")
        
        triplet_dataset = TripletDataset(
            base_dataset=base_dataset,
            negative_strategy=strategy,
            samples_per_class=10,  # Small for demo
        )
        
        # Sample a few triplets
        negative_types = []
        for i in range(30):
            sample = triplet_dataset[i]
            negative_types.append(sample['negative_type'])
        
        bg_count = negative_types.count('background')
        cc_count = negative_types.count('cross_class')
        
        print(f"  Sampled 30 triplets:")
        print(f"    Background negatives: {bg_count} ({bg_count/30*100:.1f}%)")
        print(f"    Cross-class negatives: {cc_count} ({cc_count/30*100:.1f}%)")


def example_background_statistics():
    """Example: Analyze background frame statistics."""
    
    print("\n" + "="*70)
    print("Example 4: Background Frame Statistics")
    print("="*70)
    
    base_dataset = RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=False
    )
    
    print(f"\n📊 Per-Class Statistics:\n")
    print(f"{'Class':<20} {'Total':>8} {'Annotated':>10} {'Background':>11} {'BG %':>8}")
    print("-" * 70)
    
    total_frames = 0
    total_annotated = 0
    total_background = 0
    
    for class_name in base_dataset.classes:
        data = base_dataset.class_data[class_name]
        
        n_total = data['total_frames']
        n_annotated = len(data['frame_indices'])
        n_background = len(data['background_frames'])
        bg_pct = n_background / n_total * 100
        
        print(f"{class_name:<20} {n_total:>8} {n_annotated:>10} {n_background:>11} {bg_pct:>7.1f}%")
        
        total_frames += n_total
        total_annotated += n_annotated
        total_background += n_background
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_frames:>8} {total_annotated:>10} {total_background:>11} {total_background/total_frames*100:>7.1f}%")
    
    print(f"\n💡 Key Insights:")
    print(f"  • {total_background:,} background frames available")
    print(f"  • {total_background/total_frames*100:.1f}% of all frames are background")
    print(f"  • This is {total_background/total_annotated:.1f}x more data than annotated frames!")
    print(f"  • Using these frames improves model robustness significantly")


def example_visualization():
    """Example: Visualize triplet samples."""
    
    print("\n" + "="*70)
    print("Example 5: Visualize Triplet Samples")
    print("="*70)
    
    import cv2
    import numpy as np
    from pathlib import Path
    
    # Create dataset
    base_dataset = RefDetDataset(
        data_root='./datasets/train/samples',
        annotations_file='./datasets/train/annotations/annotations.json',
        mode='train',
        cache_frames=True
    )
    
    triplet_dataset = TripletDataset(
        base_dataset=base_dataset,
        negative_strategy='mixed',
        samples_per_class=10,
    )
    
    # Create output directory
    output_dir = Path('./visualization_outputs/triplet_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📷 Generating visualizations...")
    
    # Visualize 5 triplets
    for i in range(5):
        sample = triplet_dataset[i]
        
        # Get images
        anchor = sample['anchor_image']
        positive = sample['positive_frame']
        negative = sample['negative_frame']
        
        # Resize for visualization (all to same height)
        h = 400
        anchor_resized = cv2.resize(anchor, (int(anchor.shape[1] * h / anchor.shape[0]), h))
        positive_resized = cv2.resize(positive, (int(positive.shape[1] * h / positive.shape[0]), h))
        negative_resized = cv2.resize(negative, (int(negative.shape[1] * h / negative.shape[0]), h))
        
        # Draw bboxes on positive
        for bbox in sample['positive_bboxes']:
            x1, y1, x2, y2 = bbox.astype(int)
            # Scale bbox coordinates
            scale_x = positive_resized.shape[1] / positive.shape[1]
            scale_y = positive_resized.shape[0] / positive.shape[0]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(positive_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw bboxes on negative (if cross_class)
        for bbox in sample['negative_bboxes']:
            x1, y1, x2, y2 = bbox.astype(int)
            scale_x = negative_resized.shape[1] / negative.shape[1]
            scale_y = negative_resized.shape[0] / negative.shape[0]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(negative_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add labels
        cv2.putText(anchor_resized, "Anchor (Support)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(positive_resized, "Positive (Same Class)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        neg_label = f"Negative ({sample['negative_type']})"
        cv2.putText(negative_resized, neg_label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Concatenate horizontally
        triplet_vis = np.hstack([anchor_resized, positive_resized, negative_resized])
        
        # Save
        output_path = output_dir / f"triplet_{i+1}_{sample['class_name']}_{sample['negative_type']}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(triplet_vis, cv2.COLOR_RGB2BGR))
        
        print(f"  Saved: {output_path.name}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Triplet Dataset Usage Examples")
    print("="*70)
    
    # Run examples
    example_triplet_only_training()
    example_mixed_training()
    example_negative_sampling_strategies()
    example_background_statistics()
    
    # Optional: Uncomment to generate visualizations
    # example_visualization()
    
    print("\n" + "="*70)
    print("All examples completed successfully! ✅")
    print("="*70 + "\n")
