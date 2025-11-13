"""
Main Training Script for YOLOv8n-RefDet (P2-P5 Architecture)
=============================================================

YOLOv8n-RefDet with 4-Scale Detection (P2, P3, P4, P5):
- P2 (160×160, stride 4): Optimized for tiny UAV objects
- P3 (80×80, stride 8): Small object detection
- P4 (40×40, stride 16): Medium object detection
- P5 (20×20, stride 32): Large object detection

3-Stage Training Pipeline:
- Stage 1: Base pre-training (optional, skip if using pretrained YOLOv8)
- Stage 2: Few-shot meta-learning (main training)
- Stage 3: Fine-tuning with triplet loss

Model Architecture:
- DINOv3 Support Encoder: 21.77M params (frozen)
- YOLOv8n Backbone (P2-P5): 3.16M params
- CHEAF Fusion Module: 2.12M params
- Dual Detection Head: 4.47M params
- Total: 31.52M params (63% of 50M budget)

Usage:
    # Stage 2: Meta-learning with episodic training
    python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4
    
    # Stage 2 with triplet loss for better feature learning
    python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_triplet --triplet_ratio 0.3
    
    # Resume from checkpoint
    python train.py --stage 2 --epochs 100 --resume ./checkpoints/checkpoint_epoch_50.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator, TripletCollator
from src.datasets.triplet_dataset import TripletDataset
from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.augmentations import get_stage_config, get_yolov8_augmentation_params, print_stage_config
from src.training.trainer import RefDetTrainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8n-RefDet')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./datasets/train/samples',
                        help='Root directory containing training samples')
    parser.add_argument('--annotations', type=str, default='./datasets/train/annotations/annotations.json',
                        help='Path to annotations.json')
    parser.add_argument('--test_data_root', type=str, default='./datasets/test/samples',
                        help='Root directory containing test samples (optional)')
    parser.add_argument('--test_annotations', type=str, default='./datasets/test/annotations/annotations.json',
                        help='Path to test annotations.json (optional)')
    parser.add_argument('--val_st_iou_cache', type=str, default=None,
                        help='Path to validation ST-IoU cache directory (optional, for faster validation)')
    parser.add_argument('--num_aug', type=int, default=1,
                        help='Number of augmented versions per image (1-5 recommended, 1=no augmentation multiplier)')
    
    # Caching arguments
    parser.add_argument('--frame_cache_size', type=int, default=500,
                        help='Number of video frames to cache (default 500 frames ~= 300MB)')
    parser.add_argument('--support_cache_size_mb', type=int, default=200,
                        help='Support image cache size in MB (default 200MB)')
    parser.add_argument('--disable_cache', action='store_true',
                        help='Disable all caching (useful for debugging memory issues)')
    parser.add_argument('--print_cache_stats', action='store_true',
                        help='Print cache statistics after each epoch')
    
    # Training arguments
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2, 3],
                        help='Training stage (1=base, 2=meta, 3=finetune)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of query images per episode)')
    parser.add_argument('--n_way', type=int, default=2,
                        help='Number of classes per episode (N-way)')
    parser.add_argument('--n_query', type=int, default=4,
                        help='Number of query samples per class (Q-query)')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes per epoch')
    
    # Triplet training arguments (Stage 2 & 3)
    parser.add_argument('--use_triplet', action='store_true', default=False,
                        help='Enable triplet loss training')
    parser.add_argument('--triplet_ratio', type=float, default=0.3,
                        help='Ratio of triplet batches to total batches (0.0-1.0)')
    parser.add_argument('--negative_strategy', type=str, default='mixed',
                        choices=['background', 'cross_class', 'mixed'],
                        help='Strategy for selecting negative samples')
    parser.add_argument('--triplet_batch_size', type=int, default=8,
                        help='Batch size for triplet learning')
    
    # Model arguments
    parser.add_argument('--yolo_weights', type=str, default='./models/yolov8-n.pt',
                        help='Path to pretrained YOLOv8n weights')
    parser.add_argument('--dinov3_model', type=str, default='vit_small_patch16_dinov3.lvd1689m',
                        help='DINOv3 model name from timm (e.g., vit_small_patch16_dinov3.lvd1689m)')
    parser.add_argument('--freeze_yolo', action='store_true',
                        help='Freeze YOLOv8 backbone')
    parser.add_argument('--freeze_dinov3', action='store_true', default=True,
                        help='Freeze DINOv3 encoder')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                        help='Gradient clipping max norm (0 = no clipping, recommended: 0.5-2.0)')
    
    # Loss arguments
    parser.add_argument('--bbox_weight', type=float, default=7.5,
                        help='Weight for bbox regression loss')
    parser.add_argument('--cls_weight', type=float, default=0.5,
                        help='Weight for classification loss')
    parser.add_argument('--dfl_weight', type=float, default=1.5,
                        help='Weight for DFL loss')
    parser.add_argument('--supcon_weight', type=float, default=1.0,
                        help='Weight for supervised contrastive loss')
    parser.add_argument('--cpe_weight', type=float, default=0.5,
                        help='Weight for CPE loss')
    parser.add_argument('--triplet_weight', type=float, default=0.2,
                        help='Weight for triplet loss (stage 3)')
    
    # Training settings
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    
    # Weights & Biases arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='yolov8n-refdet',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                        help='W&B tags for the run')
    parser.add_argument('--wandb_notes', type=str, default=None,
                        help='W&B notes for the run')
    
    return parser.parse_args()


def create_dataloaders(args, aug_config):
    """
    Create training and validation data loaders with episodic sampling.
    
    Episodic Training:
        - N-way: Number of classes per episode
        - Q-query: Number of query images per class
        - Support set: 3 reference images per class (fixed)
    """
    print(f"\n{'='*60}")
    print("Creating data loaders...")
    print(f"{'='*60}\n")
    
    # Training dataset
    train_dataset = RefDetDataset(
        data_root=args.data_root,
        annotations_file=args.annotations,
        mode='train',
        cache_frames=not args.disable_cache,
        num_aug=args.num_aug,
        frame_cache_size=args.frame_cache_size if not args.disable_cache else 0,
        support_cache_size_mb=args.support_cache_size_mb if not args.disable_cache else 1,
    )
    
    # Episodic batch sampler
    train_sampler = EpisodicBatchSampler(
        dataset=train_dataset,
        n_way=args.n_way,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
    )
    
    # Collate function
    train_collator = RefDetCollator(
        config=aug_config,
        mode='train',
        stage=args.stage,
    )
    
    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create triplet data loader if enabled
    triplet_loader = None
    if args.use_triplet and args.stage >= 2:
        print(f"\n✓ Creating triplet data loader...")
        print(f"  Negative strategy: {args.negative_strategy}")
        print(f"  Triplet ratio: {args.triplet_ratio}")
        print(f"  Triplet batch size: {args.triplet_batch_size}")
        
        # Create triplet dataset
        triplet_dataset = TripletDataset(
            base_dataset=train_dataset,
            negative_strategy=args.negative_strategy,
        )
        
        # Create triplet collator
        triplet_collator = TripletCollator(
            config=aug_config,
            mode='train',
            apply_strong_aug=True,
        )
        
        # Create triplet loader
        triplet_loader = DataLoader(
            triplet_dataset,
            batch_size=args.triplet_batch_size,
            shuffle=True,
            collate_fn=triplet_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        print(f"  Triplet dataset size: {len(triplet_dataset)}")
        print(f"  Triplet batches per epoch: {len(triplet_loader)}")
    
    # Validation data loader (optional)
    val_loader = None
    val_dataset = None
    if Path(args.test_data_root).exists() and Path(args.test_annotations).exists():
        val_dataset = RefDetDataset(
            data_root=args.test_data_root,
            annotations_file=args.test_annotations,
            mode='val',
            cache_frames=not args.disable_cache,
            num_aug=1,  # No augmentation multiplier for validation
            frame_cache_size=args.frame_cache_size if not args.disable_cache else 0,
            support_cache_size_mb=args.support_cache_size_mb if not args.disable_cache else 1,
        )
        
        val_sampler = EpisodicBatchSampler(
            dataset=val_dataset,
            n_way=min(args.n_way, len(val_dataset.classes)),
            n_query=args.n_query,
            n_episodes=20,  # Fewer episodes for validation
        )
        
        val_collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=args.stage,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        print(f"✓ Validation data loader created")
    
    print(f"✓ Training data loader created")
    print(f"  Episodes per epoch: {args.n_episodes}")
    print(f"  N-way: {args.n_way}")
    print(f"  Q-query: {args.n_query}")
    
    return train_loader, val_loader, triplet_loader, train_dataset, val_dataset


def create_model(args):
    """
    Create YOLOv8n-RefDet model with P2-P5 architecture.
    
    Architecture:
        - 4 detection scales: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
        - Channel progression: [32, 64, 128, 256] → [128, 256, 512, 512]
        - P2 provides 4× more spatial detail for tiny object detection
    """
    print(f"\n{'='*60}")
    print("Creating YOLOv8n-RefDet (P2-P5 Architecture)...")
    print(f"{'='*60}\n")
    
    model = YOLOv8nRefDet(
        yolo_weights=args.yolo_weights,
        nc_base=0,  # No base classes for novel object detection
        dinov3_model=args.dinov3_model,
        freeze_yolo=args.freeze_yolo,
        freeze_dinov3=args.freeze_dinov3,
        freeze_dinov3_layers=6,  # Freeze first 6 transformer blocks
    )
    
    return model


def create_loss_fn(args):
    """
    Create multi-component loss function for reference-based detection.
    
    Loss Components:
        - WIoU: Wise-IoU for bbox regression (dynamic gradient allocation)
        - BCE: Binary cross-entropy for classification
        - DFL: Distribution focal loss for box refinement
        - SupCon: Supervised contrastive loss for feature learning
        - CPE: Cross-prototype enhancement for few-shot learning
        - Triplet: Metric learning loss (Stage 3 only)
    """
    print(f"\n{'='*60}")
    print(f"Creating loss function (Stage {args.stage})...")
    print(f"{'='*60}\n")
    
    loss_fn = ReferenceBasedDetectionLoss(
        stage=args.stage,
        bbox_weight=args.bbox_weight,
        cls_weight=args.cls_weight,
        dfl_weight=args.dfl_weight,
        supcon_weight=args.supcon_weight,
        cpe_weight=args.cpe_weight,
        triplet_weight=args.triplet_weight,
    )
    
    print(f"Loss weights:")
    for key, value in loss_fn.weights.items():
        print(f"  {key}: {value}")
    
    return loss_fn


def create_optimizer(args, model):
    """
    Create AdamW optimizer with layerwise learning rates.
    
    Learning Rate Strategy:
        - DINOv3 encoder: 0.1× base LR (frozen, minimal updates)
        - YOLOv8 backbone: 1.0× base LR (fine-tuning pretrained)
        - CHEAF fusion: 2.0× base LR (training from scratch)
        - Detection head: 2.0× base LR (training from scratch)
    """
    print(f"\n{'='*60}")
    print("Creating optimizer with layerwise LR...")
    print(f"{'='*60}\n")
    
    # Layerwise learning rates
    param_groups = [
        {
            'params': model.support_encoder.parameters(),
            'lr': args.lr * 0.1,  # Lower LR for frozen/pretrained DINOv3
            'name': 'dinov3_encoder'
        },
        {
            'params': model.backbone.parameters(),
            'lr': args.lr,  # Base LR for YOLOv8 backbone
            'name': 'yolo_backbone'
        },
        {
            'params': model.scs_fusion.parameters(),
            'lr': args.lr * 2.0,  # Higher LR for CHEAF fusion (training from scratch)
            'name': 'cheaf_fusion'
        },
        {
            'params': model.detection_head.parameters(),
            'lr': args.lr * 2.0,  # Higher LR for dual detection head
            'name': 'detection_head'
        },
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    print(f"Optimizer: AdamW")
    print(f"  Base LR: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Layerwise learning rates:")
    for pg in param_groups:
        print(f"    {pg['name']}: {pg['lr']:.6f}")
    
    return optimizer


def create_scheduler(args, optimizer, train_loader):
    """
    Create cosine annealing learning rate scheduler.
    
    Schedule:
        - Cosine decay from base LR to 0.01× base LR
        - Smooth transitions for stable training
        - No warmup (using pretrained backbones)
    """
    total_steps = args.epochs * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )
    
    print(f"\nScheduler: CosineAnnealingLR")
    print(f"  Total steps: {total_steps} ({args.epochs} epochs × {len(train_loader)} batches)")
    print(f"  Initial LR: {args.lr:.6f}")
    print(f"  Min LR: {args.lr * 0.01:.6f}")
    
    return scheduler


def main():
    """
    Main training function for YOLOv8n-RefDet with P2-P5 architecture.
    
    Architecture Highlights:
        - 4 detection scales (P2-P5) for multi-scale UAV object detection
        - P2 (160×160) provides 4× more spatial detail than P3
        - 31.52M total parameters (63% of 50M budget)
        - DINOv3 encoder + YOLOv8n backbone + CHEAF fusion + Dual head
    """
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"YOLOv8n-RefDet Training (P2-P5 Architecture)")
    print(f"{'='*70}")
    print(f"Stage: {args.stage}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Mixed Precision: {args.mixed_precision}")
    print(f"Episodic Training: {args.n_way}-way {args.n_query}-query")
    print(f"Triplet Loss: {'Enabled' if args.use_triplet else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Initialize Weights & Biases
    if args.use_wandb and WANDB_AVAILABLE:
        # Generate run name if not provided
        run_name = args.wandb_name
        if run_name is None:
            run_name = f"stage{args.stage}_{args.n_way}way_{args.n_query}query"
            if args.use_triplet:
                run_name += "_triplet"
        
        # Prepare config dict for wandb
        wandb_config = {
            # Training config
            'stage': args.stage,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'n_way': args.n_way,
            'n_query': args.n_query,
            'n_episodes': args.n_episodes,
            
            # Model config
            'yolo_weights': args.yolo_weights,
            'dinov3_model': args.dinov3_model,
            'freeze_yolo': args.freeze_yolo,
            'freeze_dinov3': args.freeze_dinov3,
            
            # Optimizer config
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'gradient_accumulation': args.gradient_accumulation,
            
            # Loss weights
            'bbox_weight': args.bbox_weight,
            'cls_weight': args.cls_weight,
            'dfl_weight': args.dfl_weight,
            'supcon_weight': args.supcon_weight,
            'cpe_weight': args.cpe_weight,
            'triplet_weight': args.triplet_weight,
            
            # Triplet config
            'use_triplet': args.use_triplet,
            'triplet_ratio': args.triplet_ratio,
            'negative_strategy': args.negative_strategy,
            'triplet_batch_size': args.triplet_batch_size,
            
            # System config
            'device': args.device,
            'mixed_precision': args.mixed_precision,
            'num_workers': args.num_workers,
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            resume='allow',  # Allow resuming runs
        )
        
        print(f"✓ Weights & Biases initialized")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run name: {run_name}")
        print(f"  Run URL: {wandb.run.get_url()}\n")
        
        # Log code to wandb
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    elif args.use_wandb and not WANDB_AVAILABLE:
        print(f"⚠ Warning: wandb requested but not installed. Install with: pip install wandb")
        args.use_wandb = False
    
    # Create stage-specific augmentation config
    stage_name = f"stage{args.stage}"
    aug_config = get_stage_config(stage_name)
    
    # Print detailed augmentation configuration
    print_stage_config(stage_name)
    
    # Create data loaders
    train_loader, val_loader, triplet_loader, train_dataset, val_dataset = create_dataloaders(args, aug_config)
    
    # Create model
    model = create_model(args)
    
    # Create loss function
    loss_fn = create_loss_fn(args)
    
    # Create optimizer
    optimizer = create_optimizer(args, model)
    
    # Create scheduler
    scheduler = create_scheduler(args, optimizer, train_loader)
    
    # Calculate wandb logging interval (log every 1% of steps per epoch)
    wandb_log_interval = None
    if args.use_wandb:
        # Log every 1% of steps, minimum 1 step
        wandb_log_interval = max(1, len(train_loader) // 100)
        print(f"\nWandb logging: Every {wandb_log_interval} steps (~1% of {len(train_loader)} total steps per epoch)")
    
    # Create trainer
    trainer = RefDetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_dir=args.checkpoint_dir,
        wandb_log_interval=wandb_log_interval,  # Per-step wandb logging interval
        aug_config=aug_config,  # Pass augmentation config
        stage=args.stage,  # Pass training stage
        use_wandb=args.use_wandb,  # Enable wandb logging
        val_st_iou_cache_dir=args.val_st_iou_cache,  # Pass ST-IoU cache directory
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Watch model with wandb (optional, logs gradients and parameters)
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log='all', log_freq=100, log_graph=True)
    
    # Train with optional triplet loader
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        triplet_loader=triplet_loader,
        triplet_ratio=args.triplet_ratio if triplet_loader is not None else 0.0,
    )
    
    print(f"\n{'='*70}")
    print("🎉 Training completed successfully!")
    print(f"{'='*70}")
    print(f"Final checkpoint saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")
    print(f"{'='*70}\n")
    
    # Print cache statistics if requested
    if args.print_cache_stats:
        print("\n" + "="*70)
        print("FINAL CACHE STATISTICS")
        print("="*70)
        train_dataset.print_cache_stats()
        if val_dataset is not None:
            print("\nValidation Dataset Cache:")
            val_dataset.print_cache_stats()
    
    # Print summary statistics
    print("Training Summary:")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Training stage: {args.stage}")
    print(f"  Architecture: P2-P5 (4 scales)")
    print(f"  Model parameters: 31.52M")
    print(f"  Detection strides: [4, 8, 16, 32]")
    print()
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("✓ Weights & Biases run completed")


if __name__ == '__main__':
    main()
