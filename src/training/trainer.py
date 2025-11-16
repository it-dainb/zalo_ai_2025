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
import logging

from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.loss_utils import (
    prepare_loss_inputs,
    prepare_detection_loss_inputs,
    prepare_triplet_loss_inputs,
    prepare_mixed_loss_inputs,
)
from src.augmentations.augmentation_config import AugmentationConfig
from src.training.diagnostics import TrainingDiagnostics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def postprocess_model_outputs(
    model_outputs: Dict,
    mode: str = 'prototype',
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> Dict[str, torch.Tensor]:
    """
    Post-process raw model outputs to extract final predictions.
    Follows ultralytics YOLOv8 inference pipeline.
    
    Args:
        model_outputs: Raw model outputs from YOLOv8nRefDet
        mode: Detection mode ('standard', 'prototype', 'dual')
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections
    
    Returns:
        Dictionary with:
            - 'pred_bboxes': (B, N, 4) predicted boxes in xyxy format
            - 'pred_scores': (B, N) prediction confidence scores
            - 'pred_class_ids': (B, N) predicted class IDs
    """
    # Select which head outputs to use based on mode
    if mode == 'prototype' and 'prototype_boxes' in model_outputs:
        boxes_list = model_outputs['prototype_boxes']  # List of (B, 4, H, W) per scale
        scores_list = model_outputs['prototype_sim']   # List of (B, K, H, W) per scale
    elif mode == 'standard' and 'standard_boxes' in model_outputs:
        boxes_list = model_outputs['standard_boxes']   # List of (B, 4, H, W) per scale
        scores_list = model_outputs['standard_cls']    # List of (B, nc, H, W) per scale
    elif mode == 'dual':
        # Use prototype head if available, otherwise standard
        if 'prototype_boxes' in model_outputs:
            boxes_list = model_outputs['prototype_boxes']
            scores_list = model_outputs['prototype_sim']
        elif 'standard_boxes' in model_outputs:
            boxes_list = model_outputs['standard_boxes']
            scores_list = model_outputs['standard_cls']
        else:
            raise ValueError("No detection outputs found in model_outputs")
    else:
        raise ValueError(f"Invalid mode '{mode}' or missing outputs in model_outputs")
    
    device = boxes_list[0].device
    batch_size = boxes_list[0].shape[0]
    
    # Strides for each scale [P2, P3, P4, P5]
    strides = torch.tensor([4, 8, 16, 32], device=device, dtype=torch.float32)
    
    # Concatenate all scales following direct bbox prediction format
    # boxes: (B, 4, H, W) -> (B, 4, H*W) for each scale (direct bbox predictions in xyxy format)
    # scores: (B, K, H, W) -> (B, K, H*W) for each scale
    box_list_flat = []
    scores_list_flat = []
    
    for boxes, scores in zip(boxes_list, scores_list):
        B, C, H, W = boxes.shape
        box_list_flat.append(boxes.view(B, C, -1))  # (B, 4, H*W)
        scores_list_flat.append(scores.view(B, scores.shape[1], -1))  # (B, K, H*W)
    
    # Concatenate across scales: (B, C, total_anchors)
    box_cat = torch.cat(box_list_flat, dim=2)  # (B, 4, total_anchors)
    scores_cat = torch.cat(scores_list_flat, dim=2)  # (B, K, total_anchors)
    
    # Generate anchor points
    anchor_points_list = []
    stride_tensor_list = []
    for i, (boxes_scale, stride) in enumerate(zip(boxes_list, strides)):
        _, _, h, w = boxes_scale.shape
        # Create grid (with 0.5 offset)
        sy = torch.arange(h, device=device, dtype=torch.float32) + 0.5
        sx = torch.arange(w, device=device, dtype=torch.float32) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points_list.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor_list.append(torch.full((h * w, 1), stride.item(), device=device, dtype=torch.float32))
    
    anchor_points = torch.cat(anchor_points_list)  # (total_anchors, 2)
    stride_tensor = torch.cat(stride_tensor_list)  # (total_anchors, 1)
    
    # Decode bbox predictions from ltrb offsets to xyxy coordinates
    # Detection head outputs format [l, t, r, b] - DISTANCES from anchor [0, 10]
    # Head applies ReLU to guarantee positive (distances can't be negative)
    # Decoding: x1 = anchor_x - l*stride, y1 = anchor_y - t*stride,
    #           x2 = anchor_x + r*stride, y2 = anchor_y + b*stride
    # Since l,t,r,b >= 0, we guarantee x1 < x2 and y1 < y2
    # Permute to (B, total_anchors, 4)
    box_preds = box_cat.permute(0, 2, 1)  # (B, total_anchors, 4) [l, t, r, b]
    
    # Decode to xyxy format using anchor points
    anchor_x = anchor_points[:, 0:1] * stride_tensor  # (total_anchors, 1) - anchor x in pixels
    anchor_y = anchor_points[:, 1:2] * stride_tensor  # (total_anchors, 1) - anchor y in pixels
    
    # Add small epsilon to r,b to ensure x2 > x1 and y2 > y1 even when predictions are 0
    # This handles the case where ReLU outputs zeros in early training
    # CRITICAL: epsilon must be added AFTER stride multiplication to ensure numerical stability
    # Using 1e-4 instead of 1e-6 to account for float32 precision limits (spacing at 320.0 is ~3e-5)
    eps = 1e-4
    decoded_boxes = torch.stack([
        anchor_x.squeeze(1) - box_preds[:, :, 0] * stride_tensor.squeeze(1),  # x1 (left)
        anchor_y.squeeze(1) - box_preds[:, :, 1] * stride_tensor.squeeze(1),  # y1 (top)
        anchor_x.squeeze(1) + box_preds[:, :, 2] * stride_tensor.squeeze(1) + eps,  # x2 (right) + eps AFTER stride
        anchor_y.squeeze(1) + box_preds[:, :, 3] * stride_tensor.squeeze(1) + eps,  # y2 (bottom) + eps AFTER stride
    ], dim=2)  # (B, total_anchors, 4)
    
    # Apply sigmoid to scores and get class predictions
    scores_cat = scores_cat.sigmoid().permute(0, 2, 1)  # (B, total_anchors, K)
    max_scores, class_ids = scores_cat.max(dim=-1)  # (B, total_anchors)
    
    # Apply confidence threshold and NMS per batch
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    for b in range(batch_size):
        # Filter by confidence
        conf_mask = max_scores[b] >= conf_thres
        boxes_b = decoded_boxes[b][conf_mask]
        scores_b = max_scores[b][conf_mask]
        class_ids_b = class_ids[b][conf_mask]
        
        if len(boxes_b) == 0:
            # No detections above threshold
            final_boxes.append(torch.zeros((0, 4), device=device))
            final_scores.append(torch.zeros(0, device=device))
            final_class_ids.append(torch.zeros(0, dtype=torch.long, device=device))
            continue
        
        # Apply NMS
        from torchvision.ops import nms
        keep_indices = nms(boxes_b, scores_b, iou_thres)
        keep_indices = keep_indices[:max_det]  # Limit to max_det
        
        final_boxes.append(boxes_b[keep_indices])
        final_scores.append(scores_b[keep_indices])
        final_class_ids.append(class_ids_b[keep_indices])
    
    # Pad to same length for batching
    max_dets = max(len(b) for b in final_boxes) if final_boxes else 0
    max_dets = min(max_dets, max_det)
    
    if max_dets == 0:
        max_dets = 1  # At least 1 to avoid empty tensors
    
    padded_boxes = torch.zeros((batch_size, max_dets, 4), device=device)
    padded_scores = torch.zeros((batch_size, max_dets), device=device)
    padded_class_ids = torch.zeros((batch_size, max_dets), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        n = len(final_boxes[b])
        if n > 0:
            n = min(n, max_dets)
            padded_boxes[b, :n] = final_boxes[b][:n]
            padded_scores[b, :n] = final_scores[b][:n]
            padded_class_ids[b, :n] = final_class_ids[b][:n]
    
    return {
        'pred_bboxes': padded_boxes,
        'pred_scores': padded_scores,
        'pred_class_ids': padded_class_ids,
    }


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
        wandb_log_interval: Optional[int] = None,
        aug_config: Optional[AugmentationConfig] = None,
        stage: int = 2,
        use_wandb: bool = False,
        val_st_iou_cache_dir: Optional[str] = None,
        debug_mode: bool = False,
        logger: Optional[logging.Logger] = None,
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
            log_interval: Log every N iterations (for progress bar)
            wandb_log_interval: Log to wandb every N steps (None = only log per epoch)
            aug_config: Augmentation configuration
            stage: Training stage (1, 2, or 3)
            use_wandb: Enable Weights & Biases logging
            val_st_iou_cache_dir: Optional path to validation ST-IoU cache directory
                                  (contains *_st_iou_gt.npz and *_st_iou_metadata.json)
            debug_mode: Enable detailed debug logging
            logger: Optional pre-configured logger instance. If None, creates default logger.
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
        self.wandb_log_interval = wandb_log_interval
        self.aug_config = aug_config
        self.stage = stage
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.debug_mode = debug_mode
        
        # ST-IoU cache for faster validation
        self.val_st_iou_cache_dir = Path(val_st_iou_cache_dir) if val_st_iou_cache_dir else None
        self._cached_st_iou_gt = None
        self._cached_st_iou_metadata = None
        
        # Create checkpoint directory first (before logger setup)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided logger or create default one
        # The new logging system is configured in train.py and passed here
        if logger is not None:
            self.logger = logger
        else:
            # Fallback to old behavior for backward compatibility
            self.logger = logging.getLogger('RefDetTrainer')
            self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            if not self.logger.handlers:
                # Console handler
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
                
                # File handler for debug mode
                if debug_mode:
                    log_file = self.checkpoint_dir / 'training_debug.log'
                    file_handler = logging.FileHandler(log_file, mode='a')
                    file_formatter = logging.Formatter(
                        '[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    file_handler.setFormatter(file_formatter)
                    self.logger.addHandler(file_handler)
                    print(f"Debug logging enabled. Logs will be saved to: {log_file}")
        
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
        
        # Per-loss gradient check tracking
        self._detection_gradient_check_done = False
        
        # Initialize diagnostics
        self.diagnostics = TrainingDiagnostics(
            logger=self.logger,
            enable=debug_mode,  # Enable diagnostics when debug mode is on
            log_frequency=10,
            detailed_frequency=50
        )
        
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
        
        # Reset per-loss gradient check flag for this epoch
        self._detection_gradient_check_done = False
        
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
                # Debug logging for first batch and periodically
                if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                    self.logger.debug(f"\n{'='*70}")
                    self.logger.debug(f"BATCH {batch_idx} - Type: {batch_to_use.get('batch_type', 'unknown')}")
                    self.logger.debug(f"{'='*70}")
                    self._log_batch_info(batch_to_use, batch_idx)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.mixed_precision):
                    loss, losses_dict = self._forward_step(batch_to_use)
                    
                    # Debug log loss values
                    if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                        self.logger.debug(f"\nLoss Components:")
                        self.logger.debug(f"  Total Loss: {loss.item():.6f}")
                        for key, val in losses_dict.items():
                            self.logger.debug(f"  {key}: {val:.6f}")
                    
                    # Log diagnostics if enabled
                    if self.diagnostics.enable and hasattr(self, '_current_diagnostic_data'):
                        diag_data = self._current_diagnostic_data
                        loss_inputs = diag_data['loss_inputs']
                        diagnostic_data = diag_data.get('diagnostic_data')
                        
                        if diagnostic_data is not None:
                            self.diagnostics.log_batch_diagnostics(
                                step=self.global_step,
                                pred_bboxes=loss_inputs['pred_bboxes'],
                                target_bboxes=loss_inputs['target_bboxes'],
                                pred_cls_logits=loss_inputs['pred_cls_logits'],
                                target_cls=loss_inputs['target_cls'],
                                anchor_points=diagnostic_data['anchor_points'],
                                strides=diagnostic_data['strides'],
                                losses_dict=diag_data['losses_dict'],
                                proto_boxes_list=diagnostic_data['proto_boxes_list'],
                                proto_sim_list=diagnostic_data['proto_sim_list'],
                                gt_bboxes_list=diag_data['batch']['target_bboxes'],
                                gt_classes_list=diag_data['batch']['target_classes'],
                            )
                        # Clear diagnostic data after logging
                        delattr(self, '_current_diagnostic_data')
                    
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
                
                # Per-loss gradient check on first DETECTION batch to isolate NaN source
                is_detection = batch_to_use.get('batch_type') == 'detection'
                if is_detection and not self._detection_gradient_check_done and self.epoch == 1 and self.debug_mode:
                    print(f"\n🔍 Performing per-loss gradient check on first DETECTION batch (batch_idx={batch_idx})...")
                    
                    # Use stored tensor losses (not the .item() converted ones in losses_dict)
                    loss_tensors = getattr(self, '_loss_tensors_for_check', {})
                    print(f"Available loss components: {list(loss_tensors.keys())}")
                    self._detection_gradient_check_done = True
                    self.optimizer.zero_grad()
                    
                    # Test each loss component individually
                    loss_test_results = {}
                    for loss_name, loss_value in loss_tensors.items():
                        if not isinstance(loss_value, torch.Tensor):
                            print(f"  Skipping {loss_name} (not a tensor)")
                            continue
                        if loss_value.item() == 0.0:
                            loss_test_results[loss_name] = "SKIPPED (zero)"
                            continue
                        
                        print(f"  Testing {loss_name} (value={loss_value.item():.6f})...")
                        
                        # Clear gradients
                        self.optimizer.zero_grad()
                        
                        try:
                            # Test backward on this loss component only
                            test_loss = loss_value / self.gradient_accumulation_steps
                            if self.mixed_precision:
                                self.scaler.scale(test_loss).backward(retain_graph=True)
                            else:
                                test_loss.backward(retain_graph=True)
                            
                            # Check for NaN in gradients
                            has_nan = False
                            for param in self.model.parameters():
                                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                    has_nan = True
                                    break
                            
                            loss_test_results[loss_name] = "❌ NaN GRADIENTS" if has_nan else "✅ OK"
                        except Exception as e:
                            loss_test_results[loss_name] = f"❌ ERROR: {str(e)}"
                    
                    # Print results
                    print(f"\nPer-Loss Gradient Test Results:")
                    for loss_name, result in loss_test_results.items():
                        print(f"  {loss_name}: {result}")
                    print()
                    
                    # Clear gradients and stored tensors before actual training
                    self.optimizer.zero_grad()
                    if hasattr(self, '_loss_tensors_for_check'):
                        delattr(self, '_loss_tensors_for_check')
                
                # Backward pass
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale gradients for NaN/Inf checking and gradient clipping
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Check for NaN/Inf in gradients and log gradient stats
                    has_nan_grad = False
                    grad_norms = {}
                    
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_norms[name] = grad_norm
                            
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                self.logger.warning(f"⚠️ NaN/Inf gradient in {name}")
                                has_nan_grad = True
                                if self.debug_mode or batch_idx == 0:
                                    self.logger.debug(f"   Grad stats: min={param.grad.min():.4e}, max={param.grad.max():.4e}, mean={param.grad.mean():.4e}")
                    
                    # Debug log gradient norms
                    if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                        self.logger.debug(f"\nGradient Norms:")
                        total_grad_norm = sum(grad_norms.values())
                        self.logger.debug(f"  Total Grad Norm: {total_grad_norm:.4e}")
                        # Log top 5 largest gradients
                        top_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
                        for name, norm in top_grads:
                            self.logger.debug(f"  {name}: {norm:.4e}")
                    
                    if has_nan_grad:
                        self._log_batch_info(batch_to_use, batch_idx)
                        print(f"Loss components:")
                        for key, val in losses_dict.items():
                            print(f"  {key}: {val}")
                        self.logger.error(f"❌ NaN/Inf gradients detected at batch {batch_idx}. Skipping optimizer step.")
                        self.optimizer.zero_grad()
                        
                        # Reset scaler state if using mixed precision
                        if self.mixed_precision and self.scaler is not None:
                            self.scaler = GradScaler(enabled=True)
                        
                        continue
                    
                    # Compute total gradient norm before clipping
                    total_norm_before = 0.0
                    if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                        for param in self.model.parameters():
                            if param.grad is not None:
                                total_norm_before += param.grad.norm().item() ** 2
                        total_norm_before = total_norm_before ** 0.5
                    
                    if self.mixed_precision:
                        # Gradient clipping (if enabled) - gradients already unscaled
                        clipped_norm = 0.0
                        if self.gradient_clip_norm > 0:
                            clipped_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.gradient_clip_norm
                            )
                        
                        # Track gradient norm history (with bounded size)
                        if len(self.grad_norm_history) >= self.grad_norm_window:
                            self.grad_norm_history.pop(0)  # Remove oldest
                        self.grad_norm_history.append(float(clipped_norm))
                        
                        if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                            self.logger.debug(f"\nGradient Clipping:")
                            self.logger.debug(f"  Norm before clip: {total_norm_before:.4e}")
                            self.logger.debug(f"  Norm after clip: {clipped_norm:.4e}")
                            self.logger.debug(f"  Clip threshold: {self.gradient_clip_norm}")
                        
                        # Step optimizer (gradients already unscaled)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping (if enabled)
                        clipped_norm = 0.0
                        if self.gradient_clip_norm > 0:
                            clipped_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                max_norm=self.gradient_clip_norm
                            )
                        
                        # Track gradient norm history (with bounded size)
                        if len(self.grad_norm_history) >= self.grad_norm_window:
                            self.grad_norm_history.pop(0)  # Remove oldest
                        self.grad_norm_history.append(float(clipped_norm))
                        
                        if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                            self.logger.debug(f"\nGradient Clipping:")
                            self.logger.debug(f"  Norm before clip: {total_norm_before:.4e}")
                            self.logger.debug(f"  Norm after clip: {clipped_norm:.4e}")
                            self.logger.debug(f"  Clip threshold: {self.gradient_clip_norm}")
                        
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Debug log parameter updates
                    if self.debug_mode and (batch_idx == 0 or batch_idx % 50 == 0):
                        self.logger.debug(f"\nParameter Updates (Step {self.global_step}):")
                        self.logger.debug(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6e}")
                    
                    # Per-step wandb logging (if enabled)
                    if self.use_wandb and self.wandb_log_interval is not None:
                        if self.global_step % self.wandb_log_interval == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            step_log = {
                                'train_step/loss': loss.item() * self.gradient_accumulation_steps,
                                'train_step/learning_rate': current_lr,
                            }
                            # Log individual loss components
                            for key, value in losses_dict.items():
                                step_log[f'train_step/{key}'] = value
                            wandb.log(step_log, step=self.global_step)
                    
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
                    'loss': f"{total_loss / max(num_batches, 1):.4f}",
                    'lr': f"{current_lr:.6f}",
                })
            
            # Aggressive memory cleanup every batch to prevent worker OOM
            del batch, batch_to_use, loss, losses_dict
            
            # Clear CUDA cache periodically (every 50 batches)
            if batch_idx % 50 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
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
        
        # Clear model's internal caches every epoch
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
        # Force garbage collection after each epoch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                
                # Prepare support images
                support_images = batch['support_images']
                N, K, C, H, W = support_images.shape
                support_flat = support_images.reshape(N * K, C, H, W)
                
                # Set reference images for both loss and metrics
                self.model.set_reference_images(
                    support_flat,
                    average_prototypes=True,
                    n_way=N,
                    n_support=K
                )
                
                # Single forward pass for both loss and metrics
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.mixed_precision):
                    # Get raw model outputs
                    raw_outputs = self.model(
                        query_image=batch['query_images'],
                        mode='dual',  # Use dual head for comprehensive evaluation
                        use_cache=True,
                        class_ids=batch.get('class_ids', None),
                    )
                    
                    # Compute loss using raw outputs
                    from src.training.loss_utils import prepare_loss_inputs
                    loss_inputs = prepare_loss_inputs(
                        model_outputs=raw_outputs,
                        batch=batch,
                        stage=self.loss_fn.stage,
                    )
                    losses = self.loss_fn(**loss_inputs)
                    loss = losses['total_loss']
                    losses_dict = {k: v.item() for k, v in losses.items() if k != 'total_loss'}
                
                # Accumulate loss metrics
                total_loss += loss.item()
                for key, value in losses_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value
                num_batches += 1
                
                # Compute detection metrics if requested
                if compute_detection_metrics:
                    # Post-process raw outputs to get final predictions
                    # Use prototype head outputs for detection metrics
                    model_outputs = postprocess_model_outputs(
                        raw_outputs,
                        mode='prototype',
                        conf_thres=0.25,
                        iou_thres=0.45,
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
                                # Convert to Python float to avoid keeping computation graph
                                all_st_ious.append(float(spatial_iou) if isinstance(spatial_iou, torch.Tensor) else spatial_iou)
                        
                        # Accumulate for mAP computation (always use batch GT for mAP)
                        # Already numpy arrays from line 835-837
                        all_pred_bboxes.append(sample_pred_bboxes)
                        all_pred_scores.append(sample_pred_scores)
                        all_pred_classes.append(sample_pred_classes)
                        # Use batch GT for mAP (not cached, to ensure consistency)
                        all_gt_bboxes.append(gt_bboxes_list[i])
                        all_gt_classes.append(gt_classes_list[i])
                    
                    # Memory cleanup after detection metrics computation
                    del model_outputs, pred_bboxes, pred_scores, pred_classes
                    del gt_bboxes_list, gt_classes_list
                
                # Memory cleanup after each validation batch
                del batch, loss, losses_dict, raw_outputs, support_images, support_flat
        
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
        
        # Memory cleanup to prevent memory leak
        import gc
        
        # CRITICAL FIX: Delete large accumulated lists to prevent memory leak
        # These lists grow with every validation sample and cause OOM
        del all_st_ious, all_pred_bboxes, all_pred_scores, all_pred_classes
        del all_gt_bboxes, all_gt_classes
        
        # Also delete flattened arrays if they were created
        try:
            del all_pred_bboxes_flat, all_pred_scores_flat, all_pred_classes_flat
            del all_gt_bboxes_flat, all_gt_classes_flat
        except NameError:
            pass  # Variables don't exist if no detection metrics computed
        
        # Clear CUDA cache and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
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
        
        if self.debug_mode:
            self.logger.debug(f"\nDetection Forward Pass:")
            self.logger.debug(f"  Support images: N={N}, K={K}, C={C}, H={H}, W={W}")
            self.logger.debug(f"  Query images: {batch['query_images'].shape}")
        
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
        
        if self.debug_mode:
            self.logger.debug(f"\nModel Outputs:")
            for key, val in model_outputs.items():
                if isinstance(val, torch.Tensor):
                    self.logger.debug(f"  {key}: {val.shape}, range=[{val.min().item():.4f}, {val.max().item():.4f}]")
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    self.logger.debug(f"  {key}: List of {len(val)} tensors, first shape={val[0].shape}")
        
        # Prepare loss inputs
        loss_inputs = prepare_loss_inputs(
            model_outputs=model_outputs,
            batch=batch,
            stage=self.loss_fn.stage,
        )
        
        if self.debug_mode:
            self.logger.debug(f"\nLoss Inputs:")
            for key, val in loss_inputs.items():
                if isinstance(val, torch.Tensor):
                    self.logger.debug(f"  {key}: {val.shape}")
                elif isinstance(val, list):
                    self.logger.debug(f"  {key}: List of length {len(val)}")
        
        # Extract diagnostic_data before passing to loss function
        diagnostic_data = loss_inputs.pop('diagnostic_data', None)
        
        # Compute loss
        losses = self.loss_fn(**loss_inputs)
        
        # Extract total loss and components
        total_loss = losses['total_loss']
        losses_dict = {k: v.item() for k, v in losses.items() if k != 'total_loss'}
        
        # Store original tensor losses for gradient checking (only in epoch 1)
        if self.epoch == 1 and not hasattr(self, '_loss_tensors_for_check'):
            self._loss_tensors_for_check = {k: v for k, v in losses.items() if k != 'total_loss'}
        
        # Store diagnostic data for later use
        if self.diagnostics.enable and diagnostic_data is not None:
            self._current_diagnostic_data = {
                'loss_inputs': loss_inputs,
                'diagnostic_data': diagnostic_data,
                'losses_dict': losses_dict,
                'batch': batch,
            }
        
        return total_loss, losses_dict
    
    def _forward_triplet_step(self, batch: Dict) -> tuple:
        """
        Forward step for triplet batch.
        
        Triplet batch structure:
        - anchor_images: (B, 3, 256, 256) support images
        - positive_images: (B, 3, 640, 640) query frames with objects
        - negative_images: (B, 3, 640, 640) background/cross-class frames
        """
        if self.debug_mode:
            self.logger.debug(f"\nTriplet Forward Pass:")
            self.logger.debug(f"  Anchor images: {batch['anchor_images'].shape}")
            self.logger.debug(f"  Positive images: {batch['positive_images'].shape}")
            self.logger.debug(f"  Negative images: {batch['negative_images'].shape}")
        
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
        
        if self.debug_mode:
            self.logger.debug(f"\nTriplet Features:")
            self.logger.debug(f"  Anchor: {support_global_feat.shape}, norm={support_global_feat.norm(dim=1).mean().item():.4f}")
            self.logger.debug(f"  Positive: {positive_global_feat.shape}, norm={positive_global_feat.norm(dim=1).mean().item():.4f}")
            self.logger.debug(f"  Negative: {negative_global_feat.shape}, norm={negative_global_feat.norm(dim=1).mean().item():.4f}")
        
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
        # Check if triplet_loss is BatchHardTripletLoss (needs embeddings+labels)
        # or regular TripletLoss (needs anchor/positive/negative)
        triplet_loss_fn = self.loss_fn.triplet_loss
        
        if hasattr(triplet_loss_fn, '__class__') and 'BatchHard' in triplet_loss_fn.__class__.__name__:
            # BatchHardTripletLoss: expects (embeddings, labels)
            triplet_loss = triplet_loss_fn(
                embeddings=loss_inputs['triplet_embeddings'],
                labels=loss_inputs['triplet_labels'],
            )
        else:
            # Regular TripletLoss or AdaptiveTripletLoss: expects (anchor, positive, negative)
            triplet_loss = triplet_loss_fn(
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
    
    def _log_batch_info(self, batch: Dict, batch_idx: int):
        """
        Log detailed information about a batch for debugging.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        self.logger.debug(f"\nBatch Contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                self.logger.debug(f"  {key}:")
                self.logger.debug(f"    Shape: {value.shape}")
                self.logger.debug(f"    Dtype: {value.dtype}")
                self.logger.debug(f"    Device: {value.device}")
                
                # Only compute stats for floating point tensors
                if value.dtype in [torch.float16, torch.float32, torch.float64]:
                    if value.numel() > 0:
                        self.logger.debug(f"    Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                        self.logger.debug(f"    Mean: {value.mean().item():.4f}")
                        self.logger.debug(f"    Std: {value.std().item():.4f}")
                        self.logger.debug(f"    Has NaN: {torch.isnan(value).any().item()}")
                        self.logger.debug(f"    Has Inf: {torch.isinf(value).any().item()}")
            elif isinstance(value, list):
                if len(value) > 0:
                    if isinstance(value[0], torch.Tensor):
                        self.logger.debug(f"  {key}: List of {len(value)} tensors")
                        self.logger.debug(f"    First tensor shape: {value[0].shape}")
                    else:
                        self.logger.debug(f"  {key}: List of {len(value)} {type(value[0]).__name__}")
                else:
                    self.logger.debug(f"  {key}: Empty list")
            elif isinstance(value, dict):
                self.logger.debug(f"  {key}: Dict with keys {list(value.keys())}")
            else:
                self.logger.debug(f"  {key}: {type(value).__name__}")
    
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
        
        # Print model summary if debug mode
        if self.debug_mode:
            self.print_model_summary()
            self.logger.debug(f"\nInitial learning rate: {self.optimizer.param_groups[0]['lr']:.6e}")
            self.logger.debug(f"Optimizer: {type(self.optimizer).__name__}")
            if self.scheduler is not None:
                self.logger.debug(f"Scheduler: {type(self.scheduler).__name__}")
        
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
    
    def get_model_summary(self) -> Dict:
        """
        Get detailed model summary for debugging.
        
        Returns:
            summary: Dict containing model architecture and parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Get parameter counts by module
        module_params = {}
        for name, module in self.model.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module_params[name] = {
                'total': module_total,
                'trainable': module_trainable,
                'frozen': module_total - module_trainable
            }
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'module_parameters': module_params,
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_clip_norm': self.gradient_clip_norm,
        }
        
        return summary
    
    def print_model_summary(self):
        """Print detailed model summary."""
        summary = self.get_model_summary()
        
        print(f"\n{'='*70}")
        print(f"MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        print(f"Trainable Parameters: {summary['trainable_parameters']:,} ({summary['trainable_parameters']/summary['total_parameters']*100:.2f}%)")
        print(f"Frozen Parameters: {summary['frozen_parameters']:,} ({summary['frozen_parameters']/summary['total_parameters']*100:.2f}%)")
        print(f"\nModule Breakdown:")
        for module_name, params in summary['module_parameters'].items():
            print(f"  {module_name}:")
            print(f"    Total: {params['total']:,}")
            print(f"    Trainable: {params['trainable']:,}")
            print(f"    Frozen: {params['frozen']:,}")
        print(f"\nTraining Configuration:")
        print(f"  Device: {summary['device']}")
        print(f"  Mixed Precision: {summary['mixed_precision']}")
        print(f"  Gradient Accumulation: {summary['gradient_accumulation_steps']}")
        print(f"  Gradient Clipping: {summary['gradient_clip_norm']}")
        print(f"{'='*70}\n")
    
    def log_parameter_statistics(self):
        """Log detailed parameter statistics for debugging."""
        if not self.debug_mode:
            return
        
        self.logger.debug(f"\n{'='*70}")
        self.logger.debug(f"PARAMETER STATISTICS")
        self.logger.debug(f"{'='*70}")
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.debug(f"\n{name}:")
                self.logger.debug(f"  Shape: {param.shape}")
                self.logger.debug(f"  Range: [{param.min().item():.4e}, {param.max().item():.4e}]")
                self.logger.debug(f"  Mean: {param.mean().item():.4e}")
                self.logger.debug(f"  Std: {param.std().item():.4e}")
                self.logger.debug(f"  Norm: {param.norm().item():.4e}")
                
                if param.grad is not None:
                    self.logger.debug(f"  Grad Range: [{param.grad.min().item():.4e}, {param.grad.max().item():.4e}]")
                    self.logger.debug(f"  Grad Mean: {param.grad.mean().item():.4e}")
                    self.logger.debug(f"  Grad Norm: {param.grad.norm().item():.4e}")
