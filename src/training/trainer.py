"""
Training Pipeline for Reference-Based Detection.
Main trainer class with training loop, validation, and checkpointing.
"""

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm
import numpy as np
import json

from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.loss_utils import (
    prepare_loss_inputs,
    prepare_detection_loss_inputs,
    prepare_triplet_loss_inputs,
    prepare_mixed_loss_inputs,
)
from src.augmentations.augmentation_config import AugmentationConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class RefDetTrainer:
    """
    Training pipeline for YOLOv8n-RefDet.
    
    Supports:
    - 3-stage training (base, meta-learning, fine-tuning)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    """
    
    def __init__(
        self,
        model: YOLOv8nRefDet,
        loss_fn: ReferenceBasedDetectionLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        aug_config: Optional[AugmentationConfig] = None,
        stage: int = 2,
        use_wandb: bool = False,
        val_st_iou_cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model: YOLOv8nRefDet model
            loss_fn: Combined loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Use automatic mixed precision
            gradient_accumulation_steps: Accumulate gradients over N steps
            gradient_clip_norm: Gradient clipping max norm (0 = no clipping)
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N iterations
            aug_config: Augmentation configuration
            stage: Training stage (1, 2, or 3)
            use_wandb: Enable Weights & Biases logging
            val_st_iou_cache_dir: Optional path to validation ST-IoU cache directory
                                  (contains *_st_iou_gt.npz and *_st_iou_metadata.json)
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.aug_config = aug_config
        self.stage = stage
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # ST-IoU cache for faster validation
        self.val_st_iou_cache_dir = Path(val_st_iou_cache_dir) if val_st_iou_cache_dir else None
        self._cached_st_iou_gt = None
        self._cached_st_iou_metadata = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_st_iou = 0.0  # Track best ST-IoU
        
        # NaN/Inf tracking for adaptive handling
        self.nan_count = 0
        self.nan_threshold = 10  # Raise error if too many NaN batches in an epoch
        self.loss_ema = None  # Exponential moving average for anomaly detection
        self.loss_ema_alpha = 0.1  # EMA smoothing factor
        self.grad_norm_history = []  # Track recent gradient norms for adaptive clipping
        self.grad_norm_window = 100  # Window size for gradient norm tracking
        
        print(f"\n{'='*60}")
        print(f"RefDetTrainer initialized")
        print(f"{'='*60}")
        print(f"Stage: {stage}")
        print(f"Device: {device}")
        print(f"Mixed Precision: {mixed_precision}")
        print(f"Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"Gradient Clipping: {gradient_clip_norm if gradient_clip_norm > 0 else 'Disabled'}")
        print(f"WandB Logging: {self.use_wandb}")
        if self.val_st_iou_cache_dir:
            print(f"ST-IoU Cache: {self.val_st_iou_cache_dir}")
        print(f"{'='*60}\n")
    
    def _load_st_iou_cache(self, split_name: str = 'test') -> bool:
        """
        Load precomputed ST-IoU ground truth cache for faster validation.
        
        Args:
            split_name: Name of the split (e.g., 'test', 'val')
            
        Returns:
            True if cache loaded successfully, False otherwise
        """
        if not self.val_st_iou_cache_dir or self._cached_st_iou_gt is not None:
            return False
        
        try:
            # Load ground truth NPZ
            gt_file = self.val_st_iou_cache_dir / f'{split_name}_st_iou_gt.npz'
            metadata_file = self.val_st_iou_cache_dir / f'{split_name}_st_iou_metadata.json'
            
            if not gt_file.exists() or not metadata_file.exists():
                print(f"⚠️  ST-IoU cache not found at {self.val_st_iou_cache_dir}")
                print(f"   Falling back to per-batch GT extraction")
                return False
            
            # Load cached data
            import numpy as np
            self._cached_st_iou_gt = np.load(gt_file)
            
            with open(metadata_file, 'r') as f:
                self._cached_st_iou_metadata = json.load(f)
            
            num_videos = len(self._cached_st_iou_metadata)
            print(f"✅ Loaded ST-IoU cache: {num_videos} videos from {gt_file.name}")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load ST-IoU cache: {e}")
            print(f"   Falling back to per-batch GT extraction")
            self._cached_st_iou_gt = None
            self._cached_st_iou_metadata = None
            return False
    
    def _get_cached_gt_for_video(self, video_id: str, frame_idx: int):
        """
        Get cached ground truth bbox for a specific video and frame.
        
        Args:
            video_id: Video identifier
            frame_idx: Frame index
            
        Returns:
            Ground truth bbox as numpy array (4,) or None if not found
        """
        if self._cached_st_iou_gt is None or self._cached_st_iou_metadata is None:
            return None
        
        try:
            # Get frame IDs and bboxes for this video
            frame_key = f'{video_id}_frame_ids'
            bbox_key = f'{video_id}_bboxes'
            
            if frame_key not in self._cached_st_iou_gt.files:
                return None
            
            frame_ids = self._cached_st_iou_gt[frame_key]
            bboxes = self._cached_st_iou_gt[bbox_key]
            
            # Find matching frame
            mask = frame_ids == frame_idx
            if not mask.any():
                return None
            
            # Return first matching bbox (assuming one bbox per frame)
            bbox_idx = np.where(mask)[0][0]
            return bboxes[bbox_idx]
            
        except Exception:
            return None

    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        triplet_loader: Optional[DataLoader] = None,
        triplet_ratio: float = 0.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader (detection)
            epoch: Current epoch number
            triplet_loader: Optional triplet data loader
            triplet_ratio: Ratio of triplet batches (0.0-1.0)
            
        Returns:
            metrics: Dict of average metrics for the epoch
        """
        self.model.train()
        self.epoch = epoch
        
        # Reset NaN counter at start of epoch
        self.nan_count = 0
        
        # Metrics accumulation
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        # Create iterator for triplet loader if provided
        triplet_iter = None
        if triplet_loader is not None and triplet_ratio > 0.0:
            triplet_iter = iter(triplet_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Decide whether to use detection or triplet batch
            use_triplet = (triplet_iter is not None and 
                          torch.rand(1).item() < triplet_ratio)
            
            if use_triplet and triplet_iter is not None:
                try:
                    # Get triplet batch
                    triplet_batch = next(triplet_iter)
                    triplet_batch = self._move_batch_to_device(triplet_batch)
                    # Add batch_type marker
                    triplet_batch['batch_type'] = 'triplet'
                    batch_to_use = triplet_batch
                except StopIteration:
                    # Restart triplet iterator if exhausted
                    if triplet_loader is not None:
                        triplet_iter = iter(triplet_loader)
                        triplet_batch = next(triplet_iter)
                        triplet_batch = self._move_batch_to_device(triplet_batch)
                        triplet_batch['batch_type'] = 'triplet'
                        batch_to_use = triplet_batch
                    else:
                        # Fallback to detection batch
                        batch = self._move_batch_to_device(batch)
                        batch['batch_type'] = 'detection'
                        batch_to_use = batch
            else:
                # Use detection batch
                batch = self._move_batch_to_device(batch)
                batch['batch_type'] = 'detection'
                batch_to_use = batch
            
            # Forward pass
            try:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.mixed_precision):
                    # DEBUG: Check model outputs before loss computation
                    if batch_idx == 0:
                        # Get model outputs directly to debug
                        if batch_to_use['batch_type'] == 'detection':
                            support_images = batch_to_use['support_images']
                            N, K, C, H, W = support_images.shape
                            support_flat = support_images.reshape(N * K, C, H, W)
                            self.model.set_reference_images(support_flat, average_prototypes=True)
                            model_outputs = self.model(
                                query_image=batch_to_use['query_images'],
                                mode='dual',
                                use_cache=True,
                            )
                            print(f"🔍 DEBUG: Model outputs keys: {list(model_outputs.keys())}")
                            if 'pred_bboxes' in model_outputs:
                                print(f"   pred_bboxes shape: {model_outputs['pred_bboxes'].shape}")
                                print(f"   pred_bboxes numel: {model_outputs['pred_bboxes'].numel()}")
                            if 'pred_scores' in model_outputs:
                                print(f"   pred_scores shape: {model_outputs['pred_scores'].shape}")
                                if model_outputs['pred_scores'].numel() > 0:
                                    print(f"   pred_scores stats: min={model_outputs['pred_scores'].min():.4f}, max={model_outputs['pred_scores'].max():.4f}")
                            print(f"   target_bboxes: {len(batch_to_use['target_bboxes'])} samples")
                            for i, tb in enumerate(batch_to_use['target_bboxes'][:2]):  # Show first 2
                                print(f"     Sample {i}: {len(tb)} boxes")
                    
                    loss, losses_dict = self._forward_step(batch_to_use)
                    
                    # Check for NaN/Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n❌ ERROR: NaN or Inf detected in loss at batch {batch_idx}")
                        print(f"Batch type: {batch_to_use.get('batch_type', 'unknown')}")
                        print(f"Loss value: {loss.item()}")
                        print(f"Loss components:")
                        for key, val in losses_dict.items():
                            print(f"  {key}: {val}")
                        
                        # Check batch for NaN values
                        for key, val in batch_to_use.items():
                            if isinstance(val, torch.Tensor):
                                has_nan = torch.isnan(val).any().item()
                                has_inf = torch.isinf(val).any().item()
                                if has_nan or has_inf:
                                    print(f"  ⚠️ {key}: shape={val.shape}, has_nan={has_nan}, has_inf={has_inf}")
                        
                        raise ValueError(f"NaN or Inf detected in loss. Check augmentation pipeline or data.")
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # DEBUG: Check if ANY gradients were created
                if batch_idx == 0:
                    grads_present = sum(1 for n, p in self.model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
                    grads_total = sum(1 for n, p in self.model.named_parameters() if p.requires_grad)
                    print(f"🔍 DEBUG batch {batch_idx}: {grads_present}/{grads_total} parameters have non-zero gradients")
                    if grads_present == 0:
                        print(f"❌ ERROR: NO GRADIENTS! Loss requires_grad={loss.requires_grad}")
                        print(f"   Loss components: {losses_dict}")
                        for k, v in losses_dict.items():
                            print(f"     {k}: {v}")
                
                # Check for NaN/Inf in gradients
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"⚠️ NaN/Inf gradient in {name}")
                            has_nan_grad = True
                            if batch_idx == 0:
                                print(f"   Grad stats: min={param.grad.min()}, max={param.grad.max()}, mean={param.grad.mean()}")
                
                if has_nan_grad:
                    print(f"❌ NaN/Inf gradients detected at batch {batch_idx}. Skipping optimizer step.")
                    self.optimizer.zero_grad()
                    
                    # Reset scaler state if using mixed precision
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler = GradScaler(enabled=True)
                    
                    continue
                
                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.mixed_precision:
                        # Unscale gradients for gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping (if enabled)
                        if self.gradient_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.gradient_clip_norm
                            )
                        
                        # Step optimizer (gradients already unscaled)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping (if enabled)
                        if self.gradient_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.gradient_clip_norm
                            )
                        
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Learning rate scheduling
                    if self.scheduler is not None:
                        self.scheduler.step()
            
            except Exception as e:
                # Handle errors during forward/backward pass
                import traceback
                print(f"\n" + "="*70)
                print(f"❌ DETAILED ERROR at batch {batch_idx}/{len(train_loader)}")
                print(f"="*70)
                print(f"Batch type: {batch_to_use.get('batch_type', 'unknown')}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"\nBatch contents:")
                for key, val in batch_to_use.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, device={val.device}")
                        # Only compute stats for floating point tensors
                        if val.dtype in [torch.float16, torch.float32, torch.float64]:
                            print(f"    min={val.min().item():.4f}, max={val.max().item():.4f}, mean={val.mean().item():.4f}")
                            print(f"    has_nan={torch.isnan(val).any().item()}, has_inf={torch.isinf(val).any().item()}")
                        else:
                            print(f"    (non-float tensor, skipping stats)")
                    elif isinstance(val, dict):
                        print(f"  {key}: dict with keys {list(val.keys())}")
                    else:
                        print(f"  {key}: {type(val)}")
                
                print(f"\nFull traceback:")
                print(traceback.format_exc())
                print("="*70)
                print(f"⚠️  SKIPPING batch {batch_idx} due to error (graceful recovery mode)")
                print("="*70 + "\n")
                
                # Clear gradients and skip this batch
                self.optimizer.zero_grad()
                
                # Reset scaler state if using mixed precision (prevents "No inf checks" error)
                if self.mixed_precision and self.scaler is not None:
                    # Create a new scaler instance to reset state completely
                    # This avoids the "No inf checks were recorded" error
                    self.scaler = GradScaler(enabled=True)
                
                # Continue to next batch instead of crashing
                # (uncomment the 'raise' below to debug specific errors)
                # raise
                continue
            
            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            for key, value in losses_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.6f}",
                })
        
        # Average metrics
        avg_metrics = {
            'total_loss': total_loss / num_batches,
        }
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
        
        # Log epoch summary to wandb
        if self.use_wandb:
            wandb_log = {f'train_epoch/{k}': v for k, v in avg_metrics.items()}
            wandb_log['train_epoch/epoch'] = epoch
            wandb.log(wandb_log, step=self.global_step)
        
        return avg_metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        compute_detection_metrics: bool = True,
        use_st_iou_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Validate on validation set with comprehensive metrics.
        
        Args:
            val_loader: Validation data loader
            compute_detection_metrics: If True, compute detection metrics (ST-IoU, mAP, etc.)
            use_st_iou_cache: If True, use precomputed ST-IoU ground truth cache (faster)
            
        Returns:
            metrics: Dict of validation metrics including:
                - total_loss: Average validation loss
                - st_iou: Spatio-temporal IoU (if computed)
                - map_50: mAP@0.5 (if computed)
                - map_75: mAP@0.75 (if computed)
                - precision: Average precision (if computed)
                - recall: Average recall (if computed)
                - f1: Average F1 score (if computed)
        """
        self.model.eval()
        
        from src.metrics.st_iou import compute_st_iou, compute_spatial_iou
        from src.metrics.detection_metrics import compute_precision_recall, compute_map
        import numpy as np
        
        # Try to load ST-IoU cache if enabled
        cache_loaded = False
        if use_st_iou_cache and compute_detection_metrics:
            cache_loaded = self._load_st_iou_cache(split_name='test')
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        # Accumulate predictions and ground truth for metrics
        all_st_ious = []
        all_pred_bboxes = []
        all_pred_scores = []
        all_pred_classes = []
        all_gt_bboxes = []
        all_gt_classes = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass for loss
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.mixed_precision):
                    loss, losses_dict = self._forward_step(batch)
                
                # Accumulate loss metrics
                total_loss += loss.item()
                for key, value in losses_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                num_batches += 1
                
                # Compute detection metrics if requested
                if compute_detection_metrics:
                    # Forward pass for predictions
                    support_images = batch['support_images']
                    N, K, C, H, W = support_images.shape
                    support_flat = support_images.reshape(N * K, C, H, W)
                    self.model.set_reference_images(support_flat, average_prototypes=True)
                    
                    model_outputs = self.model(
                        query_image=batch['query_images'],
                        mode='prototype',
                        use_cache=True,
                    )
                    
                    # Extract predictions (batch_size, num_boxes, 4/1/1)
                    pred_bboxes = model_outputs['pred_bboxes'].cpu().numpy()  # (B, N, 4)
                    pred_scores = model_outputs['pred_scores'].cpu().numpy()  # (B, N)
                    pred_classes = model_outputs['pred_class_ids'].cpu().numpy()  # (B, N)
                    
                    # Extract ground truth
                    gt_bboxes_list = [b.cpu().numpy() for b in batch['target_bboxes']]
                    gt_classes_list = [c.cpu().numpy() for c in batch['target_classes']]
                    
                    # Compute ST-IoU per video/episode
                    for i in range(len(batch['query_images'])):
                        # Get predictions for this sample
                        sample_pred_bboxes = pred_bboxes[i]  # (N,)
                        sample_pred_scores = pred_scores[i]  # (N,)
                        sample_pred_classes = pred_classes[i]  # (N,)
                        
                        # Try to get cached GT first (faster)
                        cached_gt = None
                        if cache_loaded and 'video_id' in batch and 'frame_idx' in batch:
                            video_id = batch['video_id'][i] if isinstance(batch['video_id'], list) else batch['video_id']
                            frame_idx_val = batch['frame_idx'][i].item() if torch.is_tensor(batch['frame_idx']) else batch['frame_idx']
                            frame_idx_int: int = int(frame_idx_val)  # Ensure it's an integer
                            cached_gt = self._get_cached_gt_for_video(video_id, frame_idx_int)
                        
                        # Fallback to batch GT if cache miss
                        if cached_gt is not None:
                            sample_gt_bboxes = np.array([cached_gt])
                            sample_gt_classes = np.array([0])  # Assume single class for cached GT
                        else:
                            sample_gt_bboxes = gt_bboxes_list[i]  # (M, 4)
                            sample_gt_classes = gt_classes_list[i]  # (M,)
                        
                        # For ST-IoU: treat each sample as a single-frame "video"
                        # In real evaluation, you'd have multi-frame videos
                        if len(sample_gt_bboxes) > 0:
                            # Find best matching prediction for first GT box
                            gt_box = sample_gt_bboxes[0]
                            gt_class = sample_gt_classes[0]
                            
                            # Find highest confidence prediction of same class
                            class_mask = sample_pred_classes == gt_class
                            if class_mask.any():
                                class_scores = sample_pred_scores[class_mask]
                                class_bboxes = sample_pred_bboxes[class_mask]
                                best_idx = class_scores.argmax()
                                pred_box = class_bboxes[best_idx]
                                
                                # Compute spatial IoU (ST-IoU with single frame)
                                spatial_iou = compute_spatial_iou(gt_box, pred_box)
                                all_st_ious.append(spatial_iou)
                        
                        # Accumulate for mAP computation (always use batch GT for mAP)
                        all_pred_bboxes.append(sample_pred_bboxes)
                        all_pred_scores.append(sample_pred_scores)
                        all_pred_classes.append(sample_pred_classes)
                        # Use batch GT for mAP (not cached, to ensure consistency)
                        all_gt_bboxes.append(gt_bboxes_list[i])
                        all_gt_classes.append(gt_classes_list[i])
        
        # Average loss metrics
        avg_metrics = {
            'total_loss': total_loss / num_batches,
        }
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
        
        # Compute detection metrics
        if compute_detection_metrics and len(all_st_ious) > 0:
            import numpy as np
            
            # ST-IoU (mean spatial IoU for single-frame case)
            avg_st_iou = np.mean(all_st_ious)
            avg_metrics['st_iou'] = float(avg_st_iou)
            
            # Concatenate all predictions and GT
            all_pred_bboxes_flat = np.concatenate([p for p in all_pred_bboxes if len(p) > 0])
            all_pred_scores_flat = np.concatenate([s for s in all_pred_scores if len(s) > 0])
            all_pred_classes_flat = np.concatenate([c for c in all_pred_classes if len(c) > 0])
            all_gt_bboxes_flat = np.concatenate([g for g in all_gt_bboxes if len(g) > 0])
            all_gt_classes_flat = np.concatenate([c for c in all_gt_classes if len(c) > 0])
            
            # Compute precision/recall
            if len(all_pred_bboxes_flat) > 0 and len(all_gt_bboxes_flat) > 0:
                pr_metrics = compute_precision_recall(
                    all_pred_bboxes_flat,
                    all_pred_scores_flat,
                    all_pred_classes_flat,
                    all_gt_bboxes_flat,
                    all_gt_classes_flat,
                    iou_threshold=0.5,
                )
                avg_metrics['precision'] = pr_metrics['precision']
                avg_metrics['recall'] = pr_metrics['recall']
                avg_metrics['f1'] = pr_metrics['f1']
                
                # Compute mAP@0.5
                map_50, _ = compute_map(
                    all_pred_bboxes_flat,
                    all_pred_scores_flat,
                    all_pred_classes_flat,
                    all_gt_bboxes_flat,
                    all_gt_classes_flat,
                    iou_threshold=0.5,
                )
                avg_metrics['map_50'] = float(map_50)
                
                # Compute mAP@0.75
                map_75, _ = compute_map(
                    all_pred_bboxes_flat,
                    all_pred_scores_flat,
                    all_pred_classes_flat,
                    all_gt_bboxes_flat,
                    all_gt_classes_flat,
                    iou_threshold=0.75,
                )
                avg_metrics['map_75'] = float(map_75)
        
        # Log validation summary to wandb
        if self.use_wandb:
            wandb_log = {f'val/{k}': v for k, v in avg_metrics.items()}
            wandb.log(wandb_log, step=self.global_step)
        
        return avg_metrics
    
    def _forward_step(self, batch: Dict) -> tuple:
        """
        Single forward step through model and loss.
        
        Handles three types of batches:
        1. Detection batch: Standard episode with support + query images
        2. Triplet batch: (anchor, positive, negative) triplets for contrastive learning
        3. Mixed batch: Combined detection + triplet (for weighted training)
        
        Args:
            batch: Batch dict from collator
            
        Returns:
            loss: Total loss value
            losses_dict: Dict of individual loss components
        """
        # Detect batch type
        batch_type = batch.get('batch_type', 'detection')
        
        if batch_type == 'triplet':
            return self._forward_triplet_step(batch)
        elif batch_type == 'mixed':
            return self._forward_mixed_step(batch)
        else:
            return self._forward_detection_step(batch)
    
    def _forward_detection_step(self, batch: Dict) -> tuple:
        """Forward step for standard detection batch."""
        # Prepare support images
        # support_images shape: (N_classes, K_shots, 3, H, W)
        support_images = batch['support_images']
        N, K, C, H, W = support_images.shape
        
        # For episodic learning with N classes and K support images per class
        # Reshape to (N*K, 3, H, W) for batch processing
        support_flat = support_images.reshape(N * K, C, H, W)
        
        # Cache support features: compute N class prototypes (average K images per class)
        self.model.set_reference_images(
            support_flat,
            average_prototypes=True,
            n_way=N,
            n_support=K
        )
        
        # Forward pass with query images
        model_outputs = self.model(
            query_image=batch['query_images'],
            mode='dual',  # Use dual head (base + novel)
            use_cache=True,  # Use cached support features
            class_ids=batch.get('class_ids', None),  # Pass class IDs for episodic learning
        )
        
        # DEBUG: Check raw model outputs
        if self.epoch == 1 and not hasattr(self, '_debug_printed'):
            print(f"\n🔍 DEBUG: Raw model outputs")
            for k, v in model_outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"   {k}: shape={v.shape}, numel={v.numel()}")
                    if v.numel() > 0 and v.dtype in [torch.float16, torch.float32, torch.float64]:
                        print(f"      min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
            self._debug_printed = True
        
        # Prepare loss inputs
        loss_inputs = prepare_loss_inputs(
            model_outputs=model_outputs,
            batch=batch,
            stage=self.loss_fn.stage,
        )
        
        # DEBUG: Check loss inputs after matching
        if self.epoch == 1 and not hasattr(self, '_debug_loss_inputs_printed'):
            print(f"\n🔍 DEBUG: Loss inputs after matching")
            print(f"   pred_bboxes: shape={loss_inputs['pred_bboxes'].shape}")
            print(f"   target_bboxes: shape={loss_inputs['target_bboxes'].shape}")
            print(f"   Number of targets in batch: {[len(t) for t in batch['target_bboxes'][:3]]}")
            self._debug_loss_inputs_printed = True
        
        # Compute loss
        losses = self.loss_fn(**loss_inputs)
        
        # Extract total loss and components
        total_loss = losses['total_loss']
        losses_dict = {k: v.item() for k, v in losses.items() if k != 'total_loss'}
        
        return total_loss, losses_dict
    
    def _forward_triplet_step(self, batch: Dict) -> tuple:
        """
        Forward step for triplet batch.
        
        Triplet batch structure:
        - anchor_images: (B, 3, 256, 256) support images
        - positive_images: (B, 3, 640, 640) query frames with objects
        - negative_images: (B, 3, 640, 640) background/cross-class frames
        """
        # Extract anchor features directly from support encoder (no fusion needed)
        anchor_features = self.model.support_encoder(
            batch['anchor_images'],
            return_global_feat=True,
        )
        support_global_feat = anchor_features['global_feat']  # (B, 384)
        
        # Extract positive features directly from backbone (no fusion needed)
        positive_features = self.model.backbone(
            batch['positive_images'],
            return_global_feat=True,
        )
        positive_global_feat = positive_features['global_feat']  # (B, 256)
        
        # Extract negative features directly from backbone (no fusion needed)
        negative_features = self.model.backbone(
            batch['negative_images'],
            return_global_feat=True,
        )
        negative_global_feat = negative_features['global_feat']  # (B, 256)
        
        # Combine outputs for triplet loss preparation
        combined_outputs = {
            'support_global_feat': support_global_feat,
            'query_global_feat': torch.cat([positive_global_feat, negative_global_feat], dim=0)
        }
        
        # Prepare triplet loss inputs
        loss_inputs = prepare_triplet_loss_inputs(
            model_outputs=combined_outputs,
            batch=batch,
        )
        
        # Compute triplet loss
        # Note: Assumes loss_fn has a triplet_loss component
        triplet_loss = self.loss_fn.triplet_loss(
            anchor=loss_inputs['anchor_features'],
            positive=loss_inputs['positive_features'],
            negative=loss_inputs['negative_features'],
        )
        
        total_loss = triplet_loss
        losses_dict = {'triplet_loss': triplet_loss.item()}
        
        return total_loss, losses_dict
    
    def _forward_mixed_step(self, batch: Dict) -> tuple:
        """
        Forward step for mixed detection + triplet batch.
        
        Applies both detection loss and triplet loss with weighting.
        """
        # Split mixed batch
        detection_batch = batch['detection_batch']
        triplet_batch = batch['triplet_batch']
        
        # Detection forward pass
        detection_loss, detection_losses_dict = self._forward_detection_step(detection_batch)
        
        # Triplet forward pass
        triplet_loss, triplet_losses_dict = self._forward_triplet_step(triplet_batch)
        
        # Weighted combination
        triplet_weight = batch.get('triplet_weight', 0.3)  # Default 30% triplet loss
        total_loss = (1.0 - triplet_weight) * detection_loss + triplet_weight * triplet_loss
        
        # Combine loss components
        losses_dict = {
            **{f'det_{k}': v for k, v in detection_losses_dict.items()},
            **{f'tri_{k}': v for k, v in triplet_losses_dict.items()},
            'total_detection': detection_loss.item(),
            'total_triplet': triplet_loss.item(),
        }
        
        return total_loss, losses_dict
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    device_batch[key] = [v.to(self.device) for v in value]
                else:
                    device_batch[key] = value
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(
        self,
        metrics: Dict,
        is_best: bool = False,
        is_best_st_iou: bool = False,
    ) -> Path:
        """
        Save training checkpoint.
        
        Args:
            metrics: Current metrics dict (can contain nested dicts)
            is_best: Whether this is the best model so far (by loss)
            is_best_st_iou: Whether this is the best model by ST-IoU
            
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_st_iou': self.best_st_iou,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model (by ST-IoU - primary metric)
        if is_best or is_best_st_iou:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            metric_str = f"ST-IoU: {self.best_st_iou:.4f}" if is_best_st_iou else f"Loss: {self.best_val_loss:.4f}"
            print(f"Saved best model: {best_path} ({metric_str})")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_st_iou = checkpoint.get('best_st_iou', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best ST-IoU: {self.best_st_iou:.4f}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_interval: int = 10,
        triplet_loader: Optional[DataLoader] = None,
        triplet_ratio: float = 0.0,
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader (detection)
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
            triplet_loader: Optional triplet data loader
            triplet_ratio: Ratio of triplet batches (0.0-1.0)
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        if triplet_loader is not None and triplet_ratio > 0.0:
            print(f"Mixed training enabled: {(1-triplet_ratio)*100:.0f}% detection + {triplet_ratio*100:.0f}% triplet")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epoch + 1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(
                train_loader, 
                epoch,
                triplet_loader=triplet_loader,
                triplet_ratio=triplet_ratio,
            )
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, compute_detection_metrics=True)
                val_loss = val_metrics['total_loss']
                val_st_iou = val_metrics.get('st_iou', 0.0)
                
                # Check if best model (prioritize ST-IoU over loss)
                is_best = val_st_iou > self.best_st_iou
                if is_best:
                    self.best_st_iou = val_st_iou
                    self.best_val_loss = val_loss
                
                # Also update best loss if ST-IoU is same
                if val_st_iou == self.best_st_iou and val_loss < self.best_val_loss:
                    is_best = True
                    self.best_val_loss = val_loss
            else:
                val_metrics = {}
                is_best = False
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary ({epoch_time:.2f}s):")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
                if 'st_iou' in val_metrics:
                    print(f"  Val ST-IoU: {val_metrics['st_iou']:.4f}")
                if 'map_50' in val_metrics:
                    print(f"  Val mAP@0.5: {val_metrics['map_50']:.4f}")
                if 'map_75' in val_metrics:
                    print(f"  Val mAP@0.75: {val_metrics['map_75']:.4f}")
                if 'precision' in val_metrics:
                    print(f"  Val Precision: {val_metrics['precision']:.4f}")
                if 'recall' in val_metrics:
                    print(f"  Val Recall: {val_metrics['recall']:.4f}")
                if is_best:
                    print(f"  ✓ New best model! (ST-IoU: {self.best_st_iou:.4f})")
            
            # Log epoch summary to wandb
            if self.use_wandb:
                wandb_summary = {
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    'train/total_loss': train_metrics['total_loss'],
                }
                if val_metrics:
                    wandb_summary['val/total_loss'] = val_metrics['total_loss']
                    wandb_summary['val/best_loss'] = self.best_val_loss
                    
                    # Log detection metrics
                    if 'st_iou' in val_metrics:
                        wandb_summary['val/st_iou'] = val_metrics['st_iou']
                        wandb_summary['val/best_st_iou'] = self.best_st_iou
                    if 'map_50' in val_metrics:
                        wandb_summary['val/map_50'] = val_metrics['map_50']
                    if 'map_75' in val_metrics:
                        wandb_summary['val/map_75'] = val_metrics['map_75']
                    if 'precision' in val_metrics:
                        wandb_summary['val/precision'] = val_metrics['precision']
                    if 'recall' in val_metrics:
                        wandb_summary['val/recall'] = val_metrics['recall']
                    if 'f1' in val_metrics:
                        wandb_summary['val/f1'] = val_metrics['f1']
                
                wandb.log(wandb_summary, step=self.global_step)
            
            # Save checkpoint
            if epoch % save_interval == 0 or is_best:
                checkpoint_path = self.save_checkpoint(
                    metrics={'train': train_metrics, 'val': val_metrics},
                    is_best=is_best,
                )
                
                # Log checkpoint as wandb artifact
                if self.use_wandb and is_best:
                    artifact = wandb.Artifact(
                        name=f'model-{wandb.run.id}',
                        type='model',
                        description=f'Best model at epoch {epoch}',
                    )
                    artifact.add_file(str(self.checkpoint_dir / 'best_model.pt'))
                    wandb.log_artifact(artifact)
            
            print(f"{'='*60}\n")
        
        print(f"\nTraining completed!")
        print(f"  Best ST-IoU: {self.best_st_iou:.4f}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Total steps: {self.global_step}")
