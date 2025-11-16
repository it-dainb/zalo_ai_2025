"""
Training Diagnostics Module
============================

Comprehensive diagnostic logging for:
1. Anchor Assignment Quality
2. Coordinate Format Verification
3. BBox Statistics (IoU distribution, sizes)
4. Loss Component Analysis

Usage:
    from src.training.diagnostics import TrainingDiagnostics
    
    diagnostics = TrainingDiagnostics(logger=logger, enable=True)
    
    # In training loop after loss computation:
    diagnostics.log_batch_diagnostics(
        step=global_step,
        pred_bboxes=matched_pred_bboxes,
        target_bboxes=matched_target_bboxes,
        pred_cls_logits=matched_pred_cls_logits,
        target_cls=target_cls_onehot,
        anchor_points=matched_anchor_points,
        strides=matched_strides,
        losses_dict=losses_dict,
    )
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging


class TrainingDiagnostics:
    """
    Comprehensive training diagnostics for YOLOv8n-RefDet.
    
    Features:
    - Anchor assignment quality monitoring
    - Coordinate format verification
    - BBox statistics (IoU, sizes, aspect ratios)
    - Loss component breakdown
    - Gradient flow analysis
    
    Args:
        logger: Python logger instance
        enable: Enable diagnostics (default True)
        log_frequency: Log every N batches (default 10)
        detailed_frequency: Detailed logging every N batches (default 50)
    """
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        enable: bool = True,
        log_frequency: int = 10,
        detailed_frequency: int = 50,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.enable = enable
        self.log_frequency = log_frequency
        self.detailed_frequency = detailed_frequency
        
        # Statistics tracking
        self.stats_history = {
            'anchors_per_batch': [],
            'mean_iou': [],
            'mean_bbox_size': [],
            'mean_aspect_ratio': [],
        }
    
    def log_batch_diagnostics(
        self,
        step: int,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_cls_logits: Optional[torch.Tensor] = None,
        target_cls: Optional[torch.Tensor] = None,
        anchor_points: Optional[torch.Tensor] = None,
        strides: Optional[torch.Tensor] = None,
        losses_dict: Optional[Dict[str, torch.Tensor]] = None,
        proto_boxes_list: Optional[List[torch.Tensor]] = None,
        proto_sim_list: Optional[List[torch.Tensor]] = None,
        gt_bboxes_list: Optional[List[torch.Tensor]] = None,
        gt_classes_list: Optional[List[torch.Tensor]] = None,
    ):
        """
        Log comprehensive batch diagnostics.
        
        Args:
            step: Global training step
            pred_bboxes: (M, 4) predicted boxes in xyxy format
            target_bboxes: (M, 4) target boxes in xyxy format
            pred_cls_logits: (M, K) classification logits
            target_cls: (M, K) one-hot class targets
            anchor_points: (M, 2) anchor center points in pixels
            strides: (M,) stride values for each anchor
            losses_dict: Dictionary of loss components
            proto_boxes_list: Raw proto boxes before assignment (for anchor stats)
            proto_sim_list: Raw proto similarities before assignment
            gt_bboxes_list: List of GT boxes per image
            gt_classes_list: List of GT classes per image
        """
        if not self.enable:
            return
        
        # Skip logging if not at frequency interval
        if step % self.log_frequency != 0:
            return
        
        detailed = (step % self.detailed_frequency == 0)
        
        # === 1. Anchor Assignment Quality ===
        self._log_anchor_assignment(
            step=step,
            num_anchors=pred_bboxes.shape[0],
            proto_boxes_list=proto_boxes_list,
            proto_sim_list=proto_sim_list,
            gt_bboxes_list=gt_bboxes_list,
            gt_classes_list=gt_classes_list,
            detailed=detailed,
        )
        
        # === 2. Coordinate Format Verification ===
        self._verify_coordinate_formats(
            step=step,
            pred_bboxes=pred_bboxes,
            target_bboxes=target_bboxes,
            detailed=detailed,
        )
        
        # === 3. BBox Statistics ===
        self._log_bbox_statistics(
            step=step,
            pred_bboxes=pred_bboxes,
            target_bboxes=target_bboxes,
            anchor_points=anchor_points,
            strides=strides,
            detailed=detailed,
        )
        
        # === 4. Classification Statistics ===
        if pred_cls_logits is not None and target_cls is not None:
            self._log_classification_stats(
                step=step,
                pred_cls_logits=pred_cls_logits,
                target_cls=target_cls,
                detailed=detailed,
            )
        
        # === 5. Loss Component Breakdown ===
        if losses_dict is not None:
            self._log_loss_breakdown(
                step=step,
                losses_dict=losses_dict,
                detailed=detailed,
            )
    
    def _log_anchor_assignment(
        self,
        step: int,
        num_anchors: int,
        proto_boxes_list: Optional[List[torch.Tensor]],
        proto_sim_list: Optional[List[torch.Tensor]],
        gt_bboxes_list: Optional[List[torch.Tensor]],
        gt_classes_list: Optional[List[torch.Tensor]],
        detailed: bool = False,
    ):
        """Log anchor assignment quality metrics."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[Step {step}] ANCHOR ASSIGNMENT DIAGNOSTICS")
        self.logger.info(f"{'='*60}")
        
        # Number of assigned anchors
        self.logger.info(f"📍 Assigned Anchors: {num_anchors}")
        self.stats_history['anchors_per_batch'].append(num_anchors)
        
        # Compute total available anchors if raw predictions available
        if proto_boxes_list is not None:
            total_anchors = 0
            for boxes in proto_boxes_list:
                B, C, H, W = boxes.shape
                total_anchors += B * H * W
            
            assignment_rate = (num_anchors / total_anchors * 100) if total_anchors > 0 else 0.0
            self.logger.info(f"   Total Available Anchors: {total_anchors}")
            self.logger.info(f"   Assignment Rate: {assignment_rate:.2f}%")
            
            # Per-scale breakdown
            if detailed and proto_sim_list is not None:
                self.logger.info(f"\n   Per-Scale Breakdown:")
                for scale_idx, (boxes, sim) in enumerate(zip(proto_boxes_list, proto_sim_list)):
                    B, C, H, W = boxes.shape
                    scale_anchors = B * H * W
                    self.logger.info(f"     Scale {scale_idx} (stride={[4,8,16,32][scale_idx]}): {H}x{W} = {scale_anchors} anchors")
        
        # GT object statistics
        if gt_bboxes_list is not None:
            total_gt = sum(len(boxes) for boxes in gt_bboxes_list)
            avg_gt_per_image = total_gt / len(gt_bboxes_list) if len(gt_bboxes_list) > 0 else 0
            
            self.logger.info(f"\n🎯 Ground Truth Objects:")
            self.logger.info(f"   Total GT Boxes: {total_gt}")
            self.logger.info(f"   Avg GT per Image: {avg_gt_per_image:.2f}")
            self.logger.info(f"   Anchors per GT: {num_anchors / max(total_gt, 1):.2f}")
            
            # Expected: 10-50 anchors per GT box
            if num_anchors / max(total_gt, 1) < 5:
                self.logger.warning(f"   ⚠️  LOW: < 5 anchors/GT (expected 10-50)")
            elif num_anchors / max(total_gt, 1) > 100:
                self.logger.warning(f"   ⚠️  HIGH: > 100 anchors/GT (may cause class imbalance)")
        
        # Recommendation
        if num_anchors == 0:
            self.logger.error(f"   ❌ CRITICAL: No anchors assigned! This will cause NaN losses.")
            self.logger.error(f"      → Check if GT boxes are in valid range [0, 640]")
            self.logger.error(f"      → Check if anchor assignment logic is correct")
        elif num_anchors < 10:
            self.logger.warning(f"   ⚠️  Very few anchors assigned. Consider:")
            self.logger.warning(f"      → Verify GT box coordinates are in pixel space")
            self.logger.warning(f"      → Check if objects are too small for current scales")
    
    def _verify_coordinate_formats(
        self,
        step: int,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        detailed: bool = False,
    ):
        """Verify bbox coordinate formats are correct (xyxy format)."""
        if pred_bboxes.numel() == 0 or target_bboxes.numel() == 0:
            self.logger.warning(f"\n⚠️  COORDINATE VERIFICATION SKIPPED: No boxes to verify")
            return
        
        self.logger.info(f"\n📐 COORDINATE FORMAT VERIFICATION")
        self.logger.info(f"   Format: xyxy (x1, y1, x2, y2)")
        
        # Check predictions
        pred_x1 = pred_bboxes[:, 0]
        pred_y1 = pred_bboxes[:, 1]
        pred_x2 = pred_bboxes[:, 2]
        pred_y2 = pred_bboxes[:, 3]
        
        # Verify x2 > x1 and y2 > y1
        pred_valid_x = (pred_x2 > pred_x1).float().mean().item()
        pred_valid_y = (pred_y2 > pred_y1).float().mean().item()
        
        self.logger.info(f"   Predictions:")
        self.logger.info(f"     Valid x2 > x1: {pred_valid_x*100:.1f}%")
        self.logger.info(f"     Valid y2 > y1: {pred_valid_y*100:.1f}%")
        self.logger.info(f"     Range: x=[{pred_x1.min():.1f}, {pred_x2.max():.1f}], y=[{pred_y1.min():.1f}, {pred_y2.max():.1f}]")
        
        # Check targets
        tgt_x1 = target_bboxes[:, 0]
        tgt_y1 = target_bboxes[:, 1]
        tgt_x2 = target_bboxes[:, 2]
        tgt_y2 = target_bboxes[:, 3]
        
        tgt_valid_x = (tgt_x2 > tgt_x1).float().mean().item()
        tgt_valid_y = (tgt_y2 > tgt_y1).float().mean().item()
        
        self.logger.info(f"   Targets:")
        self.logger.info(f"     Valid x2 > x1: {tgt_valid_x*100:.1f}%")
        self.logger.info(f"     Valid y2 > y1: {tgt_valid_y*100:.1f}%")
        self.logger.info(f"     Range: x=[{tgt_x1.min():.1f}, {tgt_x2.max():.1f}], y=[{tgt_y1.min():.1f}, {tgt_y2.max():.1f}]")
        
        # Warnings
        if pred_valid_x < 0.95 or pred_valid_y < 0.95:
            self.logger.error(f"   ❌ PREDICTION FORMAT ERROR: Some boxes have x2<=x1 or y2<=y1")
            self.logger.error(f"      → Check bbox decoding logic in detection head")
        
        if tgt_valid_x < 1.0 or tgt_valid_y < 1.0:
            self.logger.error(f"   ❌ TARGET FORMAT ERROR: Some GT boxes have x2<=x1 or y2<=y1")
            self.logger.error(f"      → Check dataset/collator GT box loading")
        
        # Check if coordinates are in pixel space (0-640)
        pred_in_range = (
            (pred_x1 >= -50).float().mean().item() * 
            (pred_x2 <= 690).float().mean().item()
        )
        tgt_in_range = (
            (tgt_x1 >= 0).float().mean().item() * 
            (tgt_x2 <= 640).float().mean().item()
        )
        
        if pred_in_range < 0.8:
            self.logger.warning(f"   ⚠️  Many predictions outside [0, 640] range")
            self.logger.warning(f"      → This is OK early in training, but should improve")
        
        if tgt_in_range < 1.0:
            self.logger.error(f"   ❌ Some GT boxes outside [0, 640] range!")
            self.logger.error(f"      → Check if GT boxes are normalized (should be pixels)")
    
    def _log_bbox_statistics(
        self,
        step: int,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        anchor_points: Optional[torch.Tensor],
        strides: Optional[torch.Tensor],
        detailed: bool = False,
    ):
        """Log bbox statistics including IoU, sizes, aspect ratios."""
        if pred_bboxes.numel() == 0 or target_bboxes.numel() == 0:
            return
        
        self.logger.info(f"\n📊 BBOX STATISTICS")
        
        # === IoU Distribution ===
        ious = self._compute_iou(pred_bboxes, target_bboxes)
        mean_iou = ious.mean().item()
        self.stats_history['mean_iou'].append(mean_iou)
        
        self.logger.info(f"   IoU Distribution:")
        self.logger.info(f"     Mean: {mean_iou:.3f}")
        self.logger.info(f"     Median: {ious.median().item():.3f}")
        self.logger.info(f"     Min: {ious.min().item():.3f}, Max: {ious.max().item():.3f}")
        
        # IoU histogram
        iou_bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.logger.info(f"     Histogram:")
        for i in range(len(iou_bins) - 1):
            low, high = iou_bins[i], iou_bins[i+1]
            count = ((ious >= low) & (ious < high)).sum().item()
            pct = count / len(ious) * 100
            self.logger.info(f"       [{low:.1f}, {high:.1f}): {count:4d} ({pct:5.1f}%)")
        
        # === BBox Size Distribution ===
        pred_widths = pred_bboxes[:, 2] - pred_bboxes[:, 0]
        pred_heights = pred_bboxes[:, 3] - pred_bboxes[:, 1]
        pred_areas = pred_widths * pred_heights
        
        tgt_widths = target_bboxes[:, 2] - target_bboxes[:, 0]
        tgt_heights = target_bboxes[:, 3] - target_bboxes[:, 1]
        tgt_areas = tgt_widths * tgt_heights
        
        mean_tgt_size = tgt_areas.sqrt().mean().item()
        self.stats_history['mean_bbox_size'].append(mean_tgt_size)
        
        self.logger.info(f"\n   Target BBox Sizes:")
        self.logger.info(f"     Mean size (√area): {mean_tgt_size:.1f} pixels")
        self.logger.info(f"     Mean width: {tgt_widths.mean().item():.1f}, height: {tgt_heights.mean().item():.1f}")
        self.logger.info(f"     Min: {tgt_areas.sqrt().min().item():.1f}, Max: {tgt_areas.sqrt().max().item():.1f}")
        
        # Size categories (COCO-style)
        small = (tgt_areas < 32**2).sum().item()
        medium = ((tgt_areas >= 32**2) & (tgt_areas < 96**2)).sum().item()
        large = (tgt_areas >= 96**2).sum().item()
        total = len(tgt_areas)
        
        self.logger.info(f"     Size Distribution:")
        self.logger.info(f"       Small (<32²): {small} ({small/total*100:.1f}%)")
        self.logger.info(f"       Medium (32²-96²): {medium} ({medium/total*100:.1f}%)")
        self.logger.info(f"       Large (>96²): {large} ({large/total*100:.1f}%)")
        
        # === Aspect Ratio ===
        pred_aspect = pred_widths / (pred_heights + 1e-6)
        tgt_aspect = tgt_widths / (tgt_heights + 1e-6)
        
        mean_tgt_aspect = tgt_aspect.mean().item()
        self.stats_history['mean_aspect_ratio'].append(mean_tgt_aspect)
        
        self.logger.info(f"\n   Aspect Ratios (width/height):")
        self.logger.info(f"     Target mean: {mean_tgt_aspect:.2f}")
        self.logger.info(f"     Pred mean: {pred_aspect.mean().item():.2f}")
        
        # === Anchor-specific stats ===
        if anchor_points is not None and strides is not None and detailed:
            self.logger.info(f"\n   Anchor Statistics:")
            
            # Per-stride breakdown
            unique_strides = torch.unique(strides)
            for stride in unique_strides:
                mask = (strides == stride)
                count = mask.sum().item()
                mean_iou_stride = ious[mask].mean().item() if count > 0 else 0.0
                
                self.logger.info(f"     Stride {int(stride):2d}: {count:4d} anchors, mean IoU: {mean_iou_stride:.3f}")
    
    def _log_classification_stats(
        self,
        step: int,
        pred_cls_logits: torch.Tensor,
        target_cls: torch.Tensor,
        detailed: bool = False,
    ):
        """Log classification statistics."""
        if pred_cls_logits.numel() == 0 or target_cls.numel() == 0:
            return
        
        self.logger.info(f"\n🎯 CLASSIFICATION STATISTICS")
        
        # Get predictions
        pred_probs = torch.sigmoid(pred_cls_logits)
        pred_classes = pred_cls_logits.argmax(dim=1)
        target_classes = target_cls.argmax(dim=1)
        
        # Accuracy
        accuracy = (pred_classes == target_classes).float().mean().item()
        self.logger.info(f"   Accuracy: {accuracy*100:.2f}%")
        
        # Confidence statistics
        pred_conf = pred_probs.max(dim=1)[0]
        self.logger.info(f"   Confidence (max prob):")
        self.logger.info(f"     Mean: {pred_conf.mean().item():.3f}")
        self.logger.info(f"     Min: {pred_conf.min().item():.3f}, Max: {pred_conf.max().item():.3f}")
        
        # Class distribution
        if detailed:
            num_classes = target_cls.shape[1]
            self.logger.info(f"\n   Class Distribution:")
            for cls_idx in range(min(num_classes, 5)):  # Show first 5 classes
                count = (target_classes == cls_idx).sum().item()
                if count > 0:
                    self.logger.info(f"     Class {cls_idx}: {count} samples")
    
    def _log_loss_breakdown(
        self,
        step: int,
        losses_dict: Dict[str, torch.Tensor],
        detailed: bool = False,
    ):
        """Log loss component breakdown."""
        self.logger.info(f"\n💰 LOSS BREAKDOWN")
        
        # Handle both tensor and float values
        total_loss = losses_dict.get('total_loss', 0.0)
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        self.logger.info(f"   Total Loss: {total_loss:.4f}")
        
        # Individual components
        components = ['bbox_loss', 'cls_loss', 'supcon_loss', 'cpe_loss', 'triplet_loss']
        
        for comp in components:
            if comp in losses_dict:
                loss_val = losses_dict[comp]
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()
                # Calculate percentage of total
                pct = (loss_val / total_loss * 100) if total_loss > 0 else 0.0
                self.logger.info(f"     {comp:15s}: {loss_val:8.4f} ({pct:5.1f}%)")
        
        # Check for anomalies
        if 'bbox_loss' in losses_dict:
            bbox_loss = losses_dict['bbox_loss']
            if isinstance(bbox_loss, torch.Tensor):
                bbox_loss = bbox_loss.item()
            
            if torch.isnan(torch.tensor(bbox_loss)) or torch.isinf(torch.tensor(bbox_loss)):
                self.logger.error(f"   ❌ BBOX LOSS IS NaN/Inf!")
            elif bbox_loss > 5.0:
                self.logger.warning(f"   ⚠️  BBOX LOSS HIGH (>5.0)")
            elif bbox_loss < 0.1 and step > 100:
                self.logger.warning(f"   ⚠️  BBOX LOSS VERY LOW (<0.1) - may be underfitting")
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU between two sets of boxes."""
        # boxes: (N, 4) in xyxy format
        
        # Intersection
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area + 1e-6
        
        iou = inter_area / union_area
        return iou
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics over training history."""
        if not self.enable:
            return {}
        
        summary = {}
        
        for key, values in self.stats_history.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = sum(values) / len(values)
                summary[f'{key}_last'] = values[-1]
        
        return summary
    
    def reset_stats(self):
        """Reset statistics history."""
        self.stats_history = {
            'anchors_per_batch': [],
            'mean_iou': [],
            'mean_bbox_size': [],
            'mean_aspect_ratio': [],
        }


def create_diagnostics(
    logger: Optional[logging.Logger] = None,
    enable: bool = True,
    log_frequency: int = 10,
    detailed_frequency: int = 50,
) -> TrainingDiagnostics:
    """
    Factory function to create diagnostics instance.
    
    Args:
        logger: Python logger
        enable: Enable diagnostics
        log_frequency: Log every N batches
        detailed_frequency: Detailed logging every N batches
        
    Returns:
        TrainingDiagnostics instance
    """
    return TrainingDiagnostics(
        logger=logger,
        enable=enable,
        log_frequency=log_frequency,
        detailed_frequency=detailed_frequency,
    )
